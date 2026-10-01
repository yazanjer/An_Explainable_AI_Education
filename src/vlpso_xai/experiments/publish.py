"""Copy ONLY non-student-level outputs into a directory that may be made public.

The OECD licence forbids redistributing the microdata, and anything at the
level of an individual student is derived microdata. Round-2 cells write some
student-level tables (outer-test predictions for the BRR and external
validation; feature values of the six students explained locally). Those stay
on the machine that computed them. This module copies the aggregate tables and
the fold-level results, and REFUSES -- raising, not skipping -- any table that
carries a student-level column, so a future change to a cell cannot leak one
through the publishing step.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd

FORBIDDEN_COLUMNS = {"row", "y_true", "y_score", "feature_value", "group", "CNTSCHID",
                     "CNTSTUID", "sample_weight", "W_FSTUWT"}
PUBLISHABLE = {
    # kind -> tables that are fold- or aggregate-level
    "sel": ["result"], "vlstab": ["result"], "perm": ["folds"],
    "shap": ["global", "spec", "lime_features", "lime_instances"],
    "ext": [], "brr": ["summary", "replicates"],
    "eda": ["items", "pv_distribution", "task_sizes", "categories"],
}


class PublishError(RuntimeError):
    pass


def check_frame(df: pd.DataFrame, where: str) -> None:
    bad = sorted(set(df.columns) & FORBIDDEN_COLUMNS)
    if bad:
        raise PublishError(f"{where}: student-level column(s) {bad}; refusing to publish")


def publish(results: Path, dest: Path, kinds: Iterable[str] = tuple(PUBLISHABLE)) -> int:
    results, dest = Path(results), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    tabs = results / "r2_tables"
    if tabs.exists():
        (dest / "r2_tables").mkdir(exist_ok=True)
        for f in sorted(tabs.glob("*.csv")):
            if f.stat().st_size == 0:
                continue
            df = pd.read_csv(f)
            check_frame(df, f.name)
            shutil.copy2(f, dest / "r2_tables" / f.name)
            n += 1
    for kind in kinds:
        frames, metas = {t: [] for t in PUBLISHABLE[kind]}, []
        for sub in sorted((results / "cells" / kind).glob("*")):
            if not (sub / "meta.json").exists():
                continue
            m = json.loads((sub / "meta.json").read_text())
            metas.append(m)
            for t in PUBLISHABLE[kind]:
                f = sub / f"{t}.parquet"
                if f.exists():
                    frames[t].append(pd.read_parquet(f).assign(cell_id=m["cell_id"]))
        if not metas:
            continue
        out = dest / "cells" / kind
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metas).to_json(out / "meta.jsonl", orient="records", lines=True, default_handler=str)
        n += 1
        for t, fs in frames.items():
            if fs:
                df = pd.concat(fs, ignore_index=True)
                check_frame(df, f"{kind}/{t}")
                df.to_parquet(out / f"{t}.parquet", index=False)
                n += 1
    for f in ("cells_manifest.csv", "cells_reconciliation.csv", "environment.json"):
        if (results / f).exists():
            shutil.copy2(results / f, dest / f)
    return n
