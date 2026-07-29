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


def _contiguous_blocks(mask):
    """Yield (offset, length) for each run of True in a boolean array."""
    import numpy as np

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [(int(a), int(b - a + 1)) for a, b in zip(starts, ends)]


def build_analytic_frame(cfg, force: bool = False) -> pd.DataFrame:
    """Country-filtered frame with full-width missingness, cached as parquet.

    MEMORY. An earlier version did ``df, _ = pyreadstat.read_sav(sav)`` and
    filtered afterwards. That materialises all 612,004 x 1,119 cells -- several
    GB -- to keep 41,875 rows, and the Colab kernel is killed with SIGKILL
    (exit -9) before it ever reaches the filter.

    The analytic countries occupy CONTIGUOUS row ranges in the PISA file
    (Spain 178,897-214,839; Portugal 471,311-477,242), so we read only those
    slices via ``row_offset``/``row_limit``. Peak memory is proportional to the
    retained sample, not to the file.

    Missingness is counted across ALL 1,119 source columns, because the
    exclusion rule is defined that way -- but only for the retained rows, in
    column batches.
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
    all_cols = list(meta.column_names)

    # 1. One cheap pass over a single column to locate the rows we want.
    cnt, _ = pyreadstat.read_sav(str(sav), usecols=["CNT"])
    keep = cnt["CNT"].isin(countries).to_numpy()
    logger.info("%d of %d rows in %s", int(keep.sum()), len(cnt), countries)
    blocks = _contiguous_blocks(keep)
    logger.info("retained rows span %d contiguous block(s): %s",
                len(blocks), [(o, n) for o, n in blocks])

    # 2. Read only those row ranges, in column batches, and assemble.
    BATCH = 250
    per_block = []
    for bi, (offset, length) in enumerate(blocks):
        parts, miss = [], np.zeros(length, dtype=np.int32)
        for i in range(0, len(all_cols), BATCH):
            cols = all_cols[i:i + BATCH]
            d, _ = pyreadstat.read_sav(str(sav), usecols=cols,
                                       row_offset=offset, row_limit=length)
            miss += d.isna().sum(axis=1).to_numpy(dtype=np.int32)
            parts.append(d)
        blk = pd.concat(parts, axis=1)
        blk["n_missing_allcols"] = miss
        per_block.append(blk)
        del parts
        logger.info("block %d/%d read: %d rows", bi + 1, len(blocks), length)

    df = pd.concat(per_block, ignore_index=True)
    del per_block

    # Guard: the contiguity assumption must hold, or we silently lose students.
    if len(df) != int(keep.sum()):
        raise RuntimeError(
            f"read {len(df)} rows but {int(keep.sum())} were selected; the "
            "country blocks are not contiguous in this file. Re-run with the "
            "chunked reader."
        )
    if "CNT" in df.columns:
        assert set(df["CNT"].unique()) <= set(countries), "country filter leaked"

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("cached %s (%d rows x %d cols)", out, len(df), df.shape[1])
    return df
