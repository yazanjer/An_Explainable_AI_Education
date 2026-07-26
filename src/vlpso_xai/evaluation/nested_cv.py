"""Nested cross-validation: the structural core of the revision.

Answers editor comment 1 in full.

    OUTER: StratifiedGroupKFold(groups=CNTSCHID), repeated
    |  for each outer fold:
    |  +-- OUTER TRAIN  (the only data anything below may see)
    |  |     INNER: StratifiedGroupKFold on outer-train
    |  |       impute / scale        fitted on inner train
    |  |       feature selection     fitted on inner train
    |  |       hyperparameters       tuned on inner folds
    |  |       MODEL SELECTION       decided on inner folds
    |  |       decision threshold    chosen on inner folds
    |  +-- OUTER TEST   (touched exactly once, for the final metric)

Rules enforced structurally, not by discipline:

* every preprocessing and selection step is a step inside a ``Pipeline``;
* the outer test fold is scored exactly once per fold;
* the threshold comes from inner-fold predictions and is applied unchanged;
* **model selection happens inside the inner loop.** The submitted code chose
  the best model by test-set performance and reported that number
  (``evaluation.py:60-81``, ``validate_model_selection``), which is itself
  leakage;
* results are checkpointed per (task, pv, repeat, fold) so a Colab disconnect
  costs at most one fold.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.design import assert_group_disjoint, school_grouped_splitter
from ..data.features import assert_no_leakage
from ..models.pipeline import make_pipeline, score_matrix, select_threshold
from .metrics import classification_metrics

logger = logging.getLogger(__name__)


@dataclass
class NestedCVConfig:
    outer_splits: int = 5
    outer_repeats: int = 5
    inner_splits: int = 5
    scoring: str = "roc_auc"
    threshold_metric: str = "balanced_accuracy"
    random_state: int = 42
    n_jobs: int = 1
    checkpoint_dir: Optional[Path] = None
    verbose: bool = True


def _config_fingerprint(
    cfg: NestedCVConfig,
    models: Sequence[str],
    *,
    selector: Optional[Any] = None,
    feature_names: Sequence[str] = (),
    weighted: bool = False,
) -> str:
    """Short hash of everything that changes what a fold MEANS.

    Without this, a checkpoint written by a 3-fold run is happily reloaded by a
    5-fold run: the keys (task, pv, method, rep, fold) match, but the folds are
    different partitions of different sizes.

    An audit then found the first version too narrow: it hashed only the CV
    shape and model names, omitting **the selector and its hyperparameters, the
    feature matrix, and whether weights were used**. With `resume=True` and a
    shared checkpoint directory, changing `Chi2Filter(k=15)` to `k=10`, or
    editing the allowlist, silently reloaded the old fold and reported it as
    new. Everything that can change a result is now hashed.
    """
    import hashlib

    sel_desc = None
    if selector is not None:
        try:
            params = selector.get_params(deep=True)
        except Exception:
            params = {}
        sel_desc = {
            "class": type(selector).__name__,
            "params": {k: str(v) for k, v in sorted(params.items())},
        }

    blob = json.dumps(
        {
            "outer_splits": cfg.outer_splits,
            "outer_repeats": cfg.outer_repeats,
            "inner_splits": cfg.inner_splits,
            "scoring": cfg.scoring,
            "threshold_metric": cfg.threshold_metric,
            "random_state": cfg.random_state,
            "models": sorted(models),
            "selector": sel_desc,
            "n_features": len(feature_names),
            "features": sorted(str(f) for f in feature_names),
            "weighted": bool(weighted),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:10]


def _checkpoint_path(
    cfg: NestedCVConfig, key: Dict[str, Any], fingerprint: str
) -> Optional[Path]:
    if cfg.checkpoint_dir is None:
        return None
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    stem = "_".join(f"{k}-{v}" for k, v in key.items())
    return Path(cfg.checkpoint_dir) / f"fold_{stem}_cfg-{fingerprint}.parquet"


def run_nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    models: Dict[str, Dict[str, Any]],
    selector_factory: Callable[[], Any] = lambda: None,
    cfg: NestedCVConfig = NestedCVConfig(),
    sample_weight: Optional[np.ndarray] = None,
    task: str = "task",
    pv: int = 1,
    method: str = "none",
    resume: bool = True,
) -> pd.DataFrame:
    """Run the full nested CV for one (task, PV, selector) cell.

    Returns one row per outer fold, with the winning model, its inner score,
    the outer-test metrics (weighted and unweighted), the selected feature
    names, and timings.
    """
    from sklearn.model_selection import GridSearchCV

    # Guard at THIS entry point too, not just in data prep.
    assert_no_leakage(X, where=f"run_nested_cv[{task}/{method}]")

    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    rows: List[Dict[str, Any]] = []
    fingerprint = _config_fingerprint(
        cfg, list(models), selector=selector_factory(),
        feature_names=list(X.columns), weighted=sample_weight is not None,
    )

    for rep in range(cfg.outer_repeats):
        outer = school_grouped_splitter(
            cfg.outer_splits, shuffle=True, random_state=cfg.random_state + rep
        )
        for fold, (tr, te) in enumerate(outer.split(X, y, groups=groups)):
            key = {"task": task, "pv": pv, "method": method, "rep": rep, "fold": fold}
            ckpt = _checkpoint_path(cfg, key, fingerprint)
            if resume and ckpt is not None and ckpt.exists():
                rows.extend(pd.read_parquet(ckpt).to_dict("records"))
                logger.info("resumed %s", ckpt.name)
                continue

            t0 = time.perf_counter()
            # Hard structural check: no school straddles the boundary.
            assert_group_disjoint(tr, te, groups)

            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y[tr], y[te]
            g_tr = groups[tr]

            inner = school_grouped_splitter(
                cfg.inner_splits, shuffle=True, random_state=cfg.random_state + rep
            )
            inner_splits = list(inner.split(X_tr, y_tr, groups=g_tr))

            # ---- MODEL SELECTION INSIDE THE INNER LOOP --------------------
            best_name, best_est, best_score, best_params = None, None, -np.inf, None
            per_model = {}
            for name, spec in models.items():
                pipe = make_pipeline(spec["model"], selector_factory())
                gs = GridSearchCV(
                    pipe, spec["params"], scoring=cfg.scoring,
                    cv=inner_splits, n_jobs=cfg.n_jobs, refit=True,
                    error_score=np.nan,
                )
                gs.fit(X_tr, y_tr)
                per_model[name] = float(gs.best_score_)
                if np.isfinite(gs.best_score_) and gs.best_score_ > best_score:
                    best_name, best_score = name, float(gs.best_score_)
                    best_est, best_params = gs.best_estimator_, gs.best_params_

            if best_est is None:
                logger.error("every model failed on %s; skipping fold", key)
                continue

            # ---- THRESHOLD FROM INNER FOLDS ONLY --------------------------
            oof_true, oof_score = [], []
            for itr, iva in inner_splits:
                p = make_pipeline(models[best_name]["model"], selector_factory())
                p.set_params(**best_params)
                p.fit(X_tr.iloc[itr], y_tr[itr])
                oof_true.append(y_tr[iva])
                oof_score.append(score_matrix(p, X_tr.iloc[iva]))
            thr = select_threshold(
                np.concatenate(oof_true), np.concatenate(oof_score),
                metric=cfg.threshold_metric,
            )

            # ---- OUTER TEST: touched exactly once -------------------------
            s_te = score_matrix(best_est, X_te)
            m_unw = classification_metrics(y_te, s_te, thr)
            m_w = (
                classification_metrics(y_te, s_te, thr, sample_weight=np.asarray(sample_weight)[te])
                if sample_weight is not None else {}
            )

            selected = list(X.columns)
            if selector_factory() is not None and "select" in best_est.named_steps:
                sel = best_est.named_steps["select"]
                if hasattr(sel, "selected_feature_names_"):
                    selected = sel.selected_feature_names_

            row = dict(key)
            row["cfg_fingerprint"] = fingerprint
            row.update({
                "best_model": best_name,
                "inner_score": best_score,
                "best_params": json.dumps({k: str(v) for k, v in (best_params or {}).items()}),
                "threshold": thr,
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "n_schools_train": int(len(np.unique(groups[tr]))),
                "n_schools_test": int(len(np.unique(groups[te]))),
                "n_selected": len(selected),
                "selected": selected,
                "inner_scores_all_models": json.dumps(per_model),
                "fit_seconds": round(time.perf_counter() - t0, 2),
            })
            row.update({k: v for k, v in m_unw.items()})
            row.update({f"w_{k}": v for k, v in m_w.items()})
            rows.append(row)

            if ckpt is not None:
                pd.DataFrame([row]).to_parquet(ckpt, index=False)
                # Persist outer-test predictions. Required downstream for the
                # school-clustered BCa bootstrap, the paired method contrasts
                # and the permutation null -- none of which can be recomputed
                # from summary metrics alone.
                pd.DataFrame({
                    "y_true": y_te,
                    "y_score": s_te,
                    "group": groups[te],
                    "sample_weight": (np.asarray(sample_weight)[te]
                                      if sample_weight is not None else np.nan),
                    "threshold": thr,
                    **{k: v for k, v in key.items()},
                }).to_parquet(
                    ckpt.with_name(ckpt.stem + "_preds.parquet"), index=False
                )
            if cfg.verbose:
                logger.info(
                    "%s rep%d fold%d -> %s auc=%.4f (inner %.4f) %.1fs",
                    task, rep, fold, best_name, row["auc"], best_score, row["fit_seconds"],
                )

    return pd.DataFrame(rows)
