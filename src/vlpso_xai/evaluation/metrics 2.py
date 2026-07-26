"""Metrics, weighted and unweighted, with clustered bootstrap CIs.

Answers editor comments 4 and 6.

Also fixes ``codes/evaluation.py:97``, where the false-negative rate was
computed as ``fn / (fn + tn)``. The denominator must be ``fn + tp``.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    *,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """The six reported metrics plus rates, optionally survey-weighted."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        brier_score_loss, confusion_matrix, f1_score, precision_score,
        recall_score, roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], sample_weight=w)
    tn, fp, fn, tp = cm.ravel()

    out = {
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=w),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred, sample_weight=w),
        "recall": recall_score(y_true, y_pred, zero_division=0, sample_weight=w),
        "precision": precision_score(y_true, y_pred, zero_division=0, sample_weight=w),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0, sample_weight=w),
        "auc": roc_auc_score(y_true, y_score, sample_weight=w),
        "average_precision": average_precision_score(y_true, y_score, sample_weight=w),
        # CORRECTED: was fn / (fn + tn) at codes/evaluation.py:97
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "prevalence": float(np.average(y_true, weights=w)),
        "weighted": w is not None,
    }
    try:
        out["brier"] = brier_score_loss(y_true, np.clip(y_score, 0, 1), sample_weight=w)
    except Exception:
        out["brier"] = float("nan")
    return out


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    *,
    metric: str = "auc",
    threshold: float = 0.5,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    method: str = "BCa",
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI that resamples SCHOOLS, not students.

    Students within a school are not independent. Resampling students treats
    ~29,786 correlated observations as 29,786 independent ones and produces
    intervals that are far too narrow. Resampling the 1,087 schools respects
    the design.

    ``method='BCa'`` applies bias-correction and acceleration via jackknife
    over schools; ``'percentile'`` is the cheaper fallback.
    """
    from scipy import stats

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(random_state)

    def _stat(idx: np.ndarray) -> float:
        if len(np.unique(y_true[idx])) < 2:
            return np.nan
        w = None if sample_weight is None else np.asarray(sample_weight)[idx]
        return classification_metrics(
            y_true[idx], y_score[idx], threshold, sample_weight=w
        )[metric]

    observed = _stat(np.arange(len(y_true)))
    uniq = np.unique(groups)
    by_group = {g: np.where(groups == g)[0] for g in uniq}

    boots = np.empty(n_resamples)
    for b in range(n_resamples):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([by_group[g] for g in picked])
        boots[b] = _stat(idx)
    boots = boots[np.isfinite(boots)]
    if boots.size < 10:
        return {"estimate": observed, "ci_low": np.nan, "ci_high": np.nan,
                "method": "failed", "n_resamples": int(boots.size)}

    if method.upper() == "BCA":
        z0 = stats.norm.ppf(np.clip((boots < observed).mean(), 1e-6, 1 - 1e-6))
        jack = np.array([
            _stat(np.concatenate([by_group[g] for g in uniq if g != leave]))
            for leave in uniq
        ])
        jack = jack[np.isfinite(jack)]
        jm = jack.mean()
        num = np.sum((jm - jack) ** 3)
        den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
        acc = num / den if den != 0 else 0.0
        zl, zu = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
        a1 = stats.norm.cdf(z0 + (z0 + zl) / (1 - acc * (z0 + zl)))
        a2 = stats.norm.cdf(z0 + (z0 + zu) / (1 - acc * (z0 + zu)))
        lo, hi = np.quantile(boots, [np.clip(a1, 0, 1), np.clip(a2, 0, 1)])
        used = "BCa (school-clustered)"
    else:
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        used = "percentile (school-clustered)"

    return {
        "estimate": float(observed), "ci_low": float(lo), "ci_high": float(hi),
        "boot_mean": float(boots.mean()), "boot_sd": float(boots.std(ddof=1)),
        "method": used, "n_resamples": int(boots.size),
        "n_clusters": int(uniq.size), "metric": metric,
    }
