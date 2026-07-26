#!/usr/bin/env python3
"""Produce the minimal PISA 2018 extract needed to verify the rebuilt pipeline.

WHY THIS EXISTS
---------------
The full student questionnaire (CY07_MSU_STU_QQQ.sav) is ~1.3 GB and expands
to ~4 GB as CSV. Nothing about verifying the pipeline needs 640,000 students
and 1,120 columns. This script pulls the ~130 columns and ~42,000 rows
(Spain + Portugal) the analysis actually touches, so the pipeline can be
smoke-tested on real data before committing Colab hours to the full run.

WHAT IT WRITES
--------------
1. ``pisa2018_verification_extract.parquet``  (~15-25 MB)
       Raw values, NOT cleaned, NOT imputed, NOT filtered on missingness.
       The exclusion flow is part of what needs verifying, so it must be
       reproduced from raw input rather than received pre-applied.

2. ``pisa2018_meta.json``  (~1-2 MB)
       Variable labels and value labels from the SPSS metadata. This is the
       authoritative *as-released* coding and becomes layer 3 of the codebook
       (src/vlpso_xai/data/codebook.py). It is what settles, for example,
       whether ST019AQ01T is coded 1..6 as printed in the instrument or
       recoded to 1 = country of test / 2 = other country as released.

USAGE (Colab)
-------------
    !pip install -q pyreadstat pyarrow
    !curl -L -o SPSS_STU_QQQ.zip https://webfs.oecd.org/pisa2018/SPSS_STU_QQQ.zip
    !unzip -q SPSS_STU_QQQ.zip
    !python make_verification_extract.py --sav STU/CY07_MSU_STU_QQQ.sav --out .

USAGE (local, if you already have the .sav)
-------------------------------------------
    python scripts/make_verification_extract.py \
        --sav /path/to/CY07_MSU_STU_QQQ.sav --out .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Exactly the columns the analysis touches.
# ---------------------------------------------------------------------------

# Outcome: the ten mathematics plausible values. Required -- the label is
# derived from these, once per PV, per Rubin's rules.
PV_MATH = [f"PV{i}MATH" for i in range(1, 11)]

# Reading and science PVs. Not predictors (forbidden), but needed to reproduce
# the leakage demonstration in notebook 01: ranking the OLD feature matrix by
# mutual information must show these at the top.
PV_OTHER = [f"PV{i}{d}" for d in ("READ", "SCIE") for i in range(1, 11)]

# Survey design. W_FSTUWT for weighted estimates; the 80 Fay BRR replicates
# for standard errors; CNTSCHID is the PSU and the CV grouping variable.
DESIGN = (
    ["W_FSTUWT"]
    + [f"W_FSTURWT{r}" for r in range(1, 81)]
    + ["SENWT", "CNTSCHID", "CNTSTUID", "STRATUM", "SUBNATIO", "CNT", "CYC", "OECD"]
)

# The 31 allowlisted candidate predictors (config/predictor_allowlist.yaml).
PREDICTORS = (
    # ST011 home educational resources (binary) -- ADDED in revision
    [f"ST011Q{n:02d}TA" for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)]
    + ["ST011Q16NA"]
    # ST012 household possession counts (ordinal) -- used in the original
    + ["ST012Q01TA", "ST012Q02TA", "ST012Q03TA",
       "ST012Q05NA", "ST012Q06NA", "ST012Q07NA", "ST012Q08NA", "ST012Q09NA"]
    # ST013 books in the home
    + ["ST013Q01TA"]
    # ST166 phishing-email judgement (ordinal 1-6)
    + [f"ST166Q0{n}HA" for n in range(1, 6)]
    # demographics
    + ["ST004D01T", "ST019AQ01T", "ST019BQ01T", "ST019CQ01T"]
)

WANTED = PV_MATH + PV_OTHER + DESIGN + PREDICTORS
COUNTRIES = ("ESP", "PRT")   # Spain primary, Portugal external validation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sav", required=True, type=Path,
                    help="Path to CY07_MSU_STU_QQQ.sav")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory")
    ap.add_argument("--countries", nargs="+", default=list(COUNTRIES))
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = ap.parse_args()

    import pandas as pd
    import pyreadstat

    args.out.mkdir(parents=True, exist_ok=True)

    print("Reading SPSS metadata (no rows) to resolve column names ...")
    _, meta = pyreadstat.read_sav(str(args.sav), metadataonly=True)
    available = set(meta.column_names)

    present = [c for c in WANTED if c in available]
    absent = [c for c in WANTED if c not in available]
    print(f"  requested {len(WANTED)} columns; {len(present)} present, {len(absent)} absent")
    if absent:
        print("  ABSENT (reported so the allowlist can be corrected rather than"
              " silently shrinking the feature set):")
        for c in absent:
            print(f"    - {c}")

    print(f"Reading {len(present)} columns ...")
    df, _ = pyreadstat.read_sav(str(args.sav), usecols=present)

    if "CNT" in df.columns:
        before = len(df)
        df = df[df["CNT"].isin(args.countries)].copy()
        print(f"  country filter {args.countries}: {before} -> {len(df)} rows")
        for c in args.countries:
            print(f"    {c}: {(df['CNT'] == c).sum()} students, "
                  f"{df.loc[df['CNT'] == c, 'CNTSCHID'].nunique()} schools")

    stem = args.out / "pisa2018_verification_extract"
    if args.format == "parquet":
        path = stem.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    else:
        path = stem.with_suffix(".csv")
        df.to_csv(path, index=False)
    print(f"Wrote {path}  ({path.stat().st_size / 1e6:.1f} MB, "
          f"{len(df)} rows x {df.shape[1]} cols)")

    # --- metadata: the authoritative as-released coding --------------------
    v2l = dict(getattr(meta, "variable_to_label", {}) or {})
    labelsets = getattr(meta, "value_labels", {}) or {}
    value_labels = {
        col: {str(k): str(v) for k, v in labelsets[setname].items()}
        for col, setname in v2l.items()
        if setname in labelsets and col in present
    }
    payload = {
        "source_file": args.sav.name,
        "n_columns_in_source": len(meta.column_names),
        "columns_extracted": present,
        "columns_absent": absent,
        "variable_labels": {
            c: l for c, l in zip(meta.column_names, meta.column_labels) if c in present
        },
        "value_labels": value_labels,
        "missing_ranges": {
            k: str(v) for k, v in (getattr(meta, "missing_ranges", {}) or {}).items()
            if k in present
        },
    }
    meta_path = args.out / "pisa2018_meta.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {meta_path}  ({meta_path.stat().st_size / 1e6:.1f} MB, "
          f"{len(value_labels)} variables with value labels)")

    print("\nDone. Copy BOTH files into the project folder.")


if __name__ == "__main__":
    main()
