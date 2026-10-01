"""Analytic frames for the round-2 cells, built exactly as the headline run built them.

The headline run (scripts/run_all.py: stage_sample -> stage_nested) used

* Spain only, rows with at most ``missing_row_threshold`` missing values over
  the 1,119 student-questionnaire columns;
* one proficiency category per plausible value (official cut points);
* ``X_all[m].reset_index(drop=True)`` for a task, with outer splits from
  ``school_grouped_splitter(5, shuffle=True, random_state=seed + repeat)``.

Every cell reconstructs its task view through :func:`task_view`, so a cell's
outer fold is the headline fold with the same index, and paired comparisons
across methods are matched by construction. Portugal is built the same way
for the external validation and is asserted absent from the Spanish frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TASKS = {"low_vs_medium": ("Low", "Medium"),
         "medium_vs_high": ("Medium", "High"),
         "low_vs_high": ("Low", "High")}


@dataclass
class Frames:
    prim: pd.DataFrame           # Spain, analytic sample, with cat_pv1..10
    X: pd.DataFrame              # allowlisted design matrix aligned with prim
    ext: pd.DataFrame            # Portugal, same exclusion rule, with cat_pv1..10
    X_ext: pd.DataFrame
    seed: int


def _with_categories(df: pd.DataFrame) -> pd.DataFrame:
    from ..data.outcome import build_pv_categories

    df = df.drop(columns=[c for c in df.columns if c.startswith("cat_pv")])
    out = pd.concat([df, build_pv_categories(df)], axis=1)
    assert not out.columns.duplicated().any(), "duplicate columns in analytic frame"
    return out


def build_frames(cfg, *, expected_n: Optional[int] = 29786,
                 expected_schools: Optional[int] = 1087) -> Frames:
    from ..data.features import Allowlist, build_design_matrix
    from ..data.ingest import build_analytic_frame

    df = build_analytic_frame(cfg)
    thr = cfg.section("data", "missing_row_threshold")
    primary = cfg.section("data", "primary_country")
    external = list(cfg.section("data", "external_countries"))
    prim = df[(df["CNT"] == primary) & (df["n_missing_allcols"] <= thr)].reset_index(drop=True)
    ext = df[df["CNT"].isin(external) & (df["n_missing_allcols"] <= thr)].reset_index(drop=True)
    logger.info("sample flow: read %d | %s analytic %d students in %d schools | %s %d in %d",
                len(df), primary, len(prim), prim.CNTSCHID.nunique(),
                external, len(ext), ext.CNTSCHID.nunique())
    if expected_n is not None and (len(prim), prim.CNTSCHID.nunique()) != (expected_n, expected_schools):
        raise RuntimeError(
            f"analytic sample is {len(prim)} students in {prim.CNTSCHID.nunique()} "
            f"schools; the headline run used {expected_n} in {expected_schools}. "
            "Refusing to produce cells that are not matched to the headline folds.")
    assert set(prim["CNT"]) == {primary}
    assert not set(prim["CNTSCHID"]) & set(ext["CNTSCHID"]), "external schools inside the training country"
    prim, ext = _with_categories(prim), _with_categories(ext)
    root = Path(cfg.paths.root)
    alw = Allowlist.load(root / "config" / "predictor_allowlist.yaml")
    X = build_design_matrix(prim, alw, where="round2:prim")
    X_ext = build_design_matrix(ext, alw, where="round2:ext")
    assert list(X.columns) == list(X_ext.columns)
    return Frames(prim=prim, X=X, ext=ext, X_ext=X_ext, seed=int(cfg.seed))


def task_view(fr: Frames, task: str, pv: int, *, external: bool = False
              ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """``(X, y, groups, rows)`` for one task at one PV; ``rows`` index the frame."""
    neg, pos = TASKS[task]
    d = fr.ext if external else fr.prim
    Xf = fr.X_ext if external else fr.X
    cat = d[f"cat_pv{pv}"]
    m = cat.isin([neg, pos]).to_numpy()
    rows = np.flatnonzero(m)
    X = Xf[m].reset_index(drop=True)
    y = (cat[m] == pos).astype(int).to_numpy()
    g = d["CNTSCHID"].to_numpy()[m]
    return X, y, g, rows


def outer_split(fr: Frames, X, y, g, repeat: int, fold: int, n_splits: int = 5):
    """The headline outer fold ``(repeat, fold)``."""
    from ..data.design import assert_group_disjoint, school_grouped_splitter

    sp = school_grouped_splitter(n_splits, shuffle=True, random_state=fr.seed + repeat)
    tr, te = list(sp.split(X, y, groups=g))[fold]
    assert_group_disjoint(tr, te, g)
    return tr, te


def synthetic_frames(n_schools: int = 120, per: int = 25, seed: int = 0) -> Frames:
    """Small synthetic stand-in with the real column layout, for tests and benchmarks."""
    rng = np.random.default_rng(seed)
    items = ([f"ST011Q{i:02d}TA" for i in range(1, 13)] + ["ST011Q16NA"]
             + ["ST012Q01TA", "ST012Q02TA", "ST012Q03TA", "ST012Q05NA", "ST012Q06NA",
                "ST012Q07NA", "ST012Q08NA", "ST012Q09NA", "ST013Q01TA",
                "ST019AQ01T", "ST019BQ01T", "ST019CQ01T"]
             + [f"ST166Q0{i}HA" for i in range(1, 6)] + ["ST004D01T"])
    def make(nsch, off):
        n = nsch * per
        sch = np.repeat(np.arange(off, off + nsch), per)
        school_eff = rng.normal(0, 30, nsch)[sch - off]
        ability = rng.normal(0, 1, n)
        X = pd.DataFrame({c: rng.integers(1, 5, n).astype(float) for c in items})
        X["ST013Q01TA"] = np.clip(np.round(3 + ability + rng.normal(0, 1, n)), 1, 6)
        X["ST166Q03HA"] = np.clip(np.round(3.5 - ability + rng.normal(0, 1, n)), 1, 6)
        for c in items:
            X.loc[rng.random(n) < 0.03, c] = np.nan
        base = 480 + 60 * ability + school_eff
        d = pd.DataFrame({"CNTSCHID": sch, "W_FSTUWT": rng.uniform(1, 50, n)})
        for r in range(1, 81):
            d[f"W_FSTURWT{r}"] = d["W_FSTUWT"] * rng.choice([0.5, 1.5], n)
        for i in range(1, 11):
            d[f"PV{i}MATH"] = base + rng.normal(0, 25, n)
        d = _with_categories(pd.concat([d, X], axis=1))
        return d, X
    prim, X = make(n_schools, 0)
    ext, X_ext = make(max(10, n_schools // 4), 10_000)
    return Frames(prim=prim, X=X, ext=ext, X_ext=X_ext, seed=42)
