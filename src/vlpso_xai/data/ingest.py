"""SPSS -> parquet ingest, Drive-aware and cached.

PISA data is not redistributed. This module downloads from the OECD on first
run, verifies size, converts once, and caches. Every later stage reads the
cached parquet and fails with an actionable message if it is absent.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def ensure_raw(cfg) -> Path:
    """Download and extract the student questionnaire if absent."""
    raw = cfg.paths.data_raw
    raw.mkdir(parents=True, exist_ok=True)
    sav = raw / cfg.section("data", "source", "student_sav")
    if sav.exists():
        return sav
    for candidate in raw.rglob(cfg.section("data", "source", "student_sav")):
        return candidate

    zp = raw / "SPSS_STU_QQQ.zip"
    if not zp.exists():
        url = cfg.section("data", "source", "student_zip_url")
        logger.info("downloading %s (~500 MB)", url)
        subprocess.run(["curl", "-L", "-o", str(zp), url], check=True)
    logger.info("zip sha256: %s", sha256(zp))
    subprocess.run(["unzip", "-o", "-q", str(zp), "-d", str(raw)], check=True)

    for candidate in raw.rglob(cfg.section("data", "source", "student_sav")):
        return candidate
    raise FileNotFoundError(
        f"{sav} not found after extraction. Download the PISA 2018 student "
        f"questionnaire from {cfg.section('data','source','landing_page')} and "
        f"place the .sav under {raw}."
    )


def build_analytic_frame(cfg, force: bool = False) -> pd.DataFrame:
    """Country-filtered frame with full-width missingness, cached as parquet.

    Missingness is counted across ALL 1,119 source columns because the
    exclusion rule is defined that way; counting it over a subset would give a
    different analytic sample.
    """
    out = cfg.paths.data_processed / "analytic.parquet"
    if out.exists() and not force:
        logger.info("using cached %s", out)
        return pd.read_parquet(out)

    import pyreadstat

    sav = ensure_raw(cfg)
    countries = [cfg.section("data", "primary_country")] + list(
        cfg.section("data", "external_countries"))

    _, meta = pyreadstat.read_sav(str(sav), metadataonly=True)
    cnt, _ = pyreadstat.read_sav(str(sav), usecols=["CNT"])
    keep = cnt["CNT"].isin(countries).to_numpy()
    logger.info("%d of %d rows in %s", keep.sum(), len(cnt), countries)

    miss = np.zeros(len(cnt), dtype=np.int32)
    for i in range(0, len(meta.column_names), 250):
        d, _ = pyreadstat.read_sav(str(sav), usecols=meta.column_names[i:i + 250])
        miss += d.isna().sum(axis=1).to_numpy(dtype=np.int32)
        del d

    df, _ = pyreadstat.read_sav(str(sav))
    df["n_missing_allcols"] = miss
    df = df[keep].reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("cached %s (%d rows x %d cols)", out, len(df), df.shape[1])
    return df
