"""PISA complex survey design: weights, BRR replicates, and clustered CV.

Answers editor comment 3 ("...survey weights, plausible-value handling, and
treatment of PISA's stratified and clustered sampling design. If survey
weights and design variables were not used, justify this and discuss the
implications").

THE DEFECT THIS REPLACES
------------------------
The submitted analysis ignored the design entirely, and worse, inverted it:

* ``W_FSTUWT`` and the 80 replicate weights were never applied, but *were*
  left in the feature matrix as predictors (``data_preparation.py:253``).
* ``STRATUM`` was one-hot encoded and used as a predictor
  (``data_preparation.py:90, 134-140``), then its SHAP value was interpreted
  as a substantive regional effect (``main.tex:746``).
* Cross-validation split students at random. PISA samples *schools* first and
  then students within schools, so random splitting puts classmates on both
  sides of the train/test boundary. Because achievement is strongly clustered
  within schools, this inflates every performance estimate.

WHAT THIS MODULE PROVIDES
-------------------------
* ``W_FSTUWT`` for weighted descriptives and weighted performance metrics;
* the 80 Fay BRR replicate weights (Fay factor 0.5) for standard errors that
  respect stratification and clustering;
* ``StratifiedGroupKFold`` splitters grouped on ``CNTSCHID``, so no school
  straddles a fold boundary.

Expect weighted, school-grouped estimates to be LOWER than the naive ones.
That is the design working, not a regression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FINAL_WEIGHT = "W_FSTUWT"
REPLICATE_TEMPLATE = "W_FSTURWT{r}"
N_REPLICATES = 80
FAY_K = 0.5          # PISA uses Fay's variant of BRR with k = 0.5
PSU = "CNTSCHID"
STRATUM = "STRATUM"


class DesignError(RuntimeError):
    """Raised when the survey design cannot be honoured and silence would mislead."""


@dataclass
class SurveyDesign:
    """The design information attached to an analytic sample."""

    weights: np.ndarray                 # (n,) final student weights
    replicates: Optional[np.ndarray]    # (n, R) replicate weights, or None
    psu: np.ndarray                     # (n,) school ids
    stratum: Optional[np.ndarray]       # (n,) explicit strata
    fay_k: float = FAY_K

    @property
    def n(self) -> int:
        return self.weights.shape[0]

    @property
    def n_replicates(self) -> int:
        return 0 if self.replicates is None else self.replicates.shape[1]

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        weight_col: str = FINAL_WEIGHT,
        replicate_template: str = REPLICATE_TEMPLATE,
        n_replicates: int = N_REPLICATES,
        psu_col: str = PSU,
        stratum_col: str = STRATUM,
        require_replicates: bool = True,
    ) -> "SurveyDesign":
        if weight_col not in df.columns:
            raise DesignError(
                f"Final weight {weight_col!r} absent. Weighted estimates cannot "
                "be produced. Either supply the weights or state explicitly in "
                "the manuscript that estimates are unweighted and why "
                "(editor comment 3)."
            )
        rep_cols = [replicate_template.format(r=r) for r in range(1, n_replicates + 1)]
        have = [c for c in rep_cols if c in df.columns]
        if len(have) < n_replicates:
            msg = (
                f"Only {len(have)}/{n_replicates} BRR replicate weights present. "
                "Standard errors from the replicate procedure are unavailable."
            )
            if require_replicates:
                raise DesignError(
                    msg + " Pass require_replicates=False to proceed with naive "
                    "SEs, which MUST then be labelled as such in every table."
                )
            logger.warning(msg + " Falling back to naive standard errors.")
            replicates = None
        else:
            replicates = df.loc[:, rep_cols].to_numpy(dtype=float)

        if psu_col not in df.columns:
            raise DesignError(
                f"PSU column {psu_col!r} absent. School-grouped cross-validation "
                "is impossible and any CV estimate would be optimistically biased."
            )

        return cls(
            weights=df[weight_col].to_numpy(dtype=float),
            replicates=replicates,
            psu=df[psu_col].to_numpy(),
            stratum=df[stratum_col].to_numpy() if stratum_col in df.columns else None,
        )

    def subset(self, idx: np.ndarray) -> "SurveyDesign":
        return SurveyDesign(
            weights=self.weights[idx],
            replicates=None if self.replicates is None else self.replicates[idx],
            psu=self.psu[idx],
            stratum=None if self.stratum is None else self.stratum[idx],
            fay_k=self.fay_k,
        )

    def normalised_weights(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """Weights scaled to sum to the sample size.

        Estimators that accept ``sample_weight`` but use it for regularisation
        or early stopping behave badly with raw PISA weights, which are in the
        thousands. Normalising preserves relative weighting.
        """
        w = self.weights if idx is None else self.weights[idx]
        s = np.sum(w)
        if s <= 0:
            raise DesignError("Weights sum to zero; cannot normalise.")
        return w * (len(w) / s)


# ---------------------------------------------------------------------------
# Weighted descriptives
# ---------------------------------------------------------------------------
def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w)
    if not ok.any() or np.sum(w[ok]) == 0:
        return float("nan")
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))


def weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    m = weighted_mean(x, w)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(w)
    if not ok.any():
        return float("nan")
    return float(np.sum(w[ok] * (x[ok] - m) ** 2) / np.sum(w[ok]))


def weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(weighted_var(x, w)))


def weighted_proportion(mask: np.ndarray, w: np.ndarray) -> float:
    return weighted_mean(np.asarray(mask, dtype=float), w)


# ---------------------------------------------------------------------------
# BRR variance estimation
# ---------------------------------------------------------------------------
def brr_standard_error(
    statistic: Callable[[np.ndarray], float],
    design: SurveyDesign,
    *,
    fay_k: Optional[float] = None,
) -> Dict[str, float]:
    r"""Fay-BRR standard error for any weighted statistic.

    .. math::

        \mathrm{Var}(\hat\theta) =
        \frac{1}{R(1-k)^2}\sum_{r=1}^{R}(\hat\theta_r - \hat\theta)^2

    with :math:`R = 80` replicates and Fay factor :math:`k = 0.5`, giving the
    familiar PISA divisor :math:`R(1-k)^2 = 20`.

    Parameters
    ----------
    statistic:
        Callable taking a weight vector and returning a scalar. It must use
        *only* the weights passed to it -- closing over the full-sample
        weights defeats the whole procedure.
    """
    if design.replicates is None:
        raise DesignError(
            "BRR standard errors requested but replicate weights are absent. "
            "Report naive SEs explicitly labelled as naive, or supply the "
            "replicate weights."
        )
    k = design.fay_k if fay_k is None else fay_k
    theta = float(statistic(design.weights))
    reps = np.array(
        [float(statistic(design.replicates[:, r])) for r in range(design.n_replicates)]
    )
    ok = np.isfinite(reps)
    if ok.sum() < 2:
        raise DesignError("Fewer than two usable replicate estimates.")
    var = np.sum((reps[ok] - theta) ** 2) / (ok.sum() * (1.0 - k) ** 2)
    se = float(np.sqrt(var))
    return {
        "estimate": theta,
        "brr_variance": float(var),
        "brr_se": se,
        "ci_low": theta - 1.96 * se,
        "ci_high": theta + 1.96 * se,
        "n_replicates_used": int(ok.sum()),
        "fay_k": float(k),
    }


def brr_weighted_mean_se(x: np.ndarray, design: SurveyDesign) -> Dict[str, float]:
    """Convenience wrapper: weighted mean with a BRR standard error."""
    x = np.asarray(x, dtype=float)
    return brr_standard_error(lambda w: weighted_mean(x, w), design)


def brr_metric_se(
    metric: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    design: SurveyDesign,
) -> Dict[str, float]:
    """BRR standard error for a weighted performance metric.

    ``metric(y_true, y_score, sample_weight) -> float``.
    """
    return brr_standard_error(lambda w: metric(y_true, y_score, w), design)


# ---------------------------------------------------------------------------
# Clustered cross-validation
# ---------------------------------------------------------------------------
def school_grouped_splitter(
    n_splits: int = 5,
    *,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    stratified: bool = True,
):
    """Return a splitter that never lets a school straddle a fold boundary.

    ``StratifiedGroupKFold`` preserves the class balance of each fold *and*
    keeps every group (school) whole. Note that scikit-learn's implementation
    ignores ``shuffle``/``random_state`` prior to 1.6 in some versions; we pass
    them where supported and record the effective behaviour.
    """
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    if not stratified:
        return GroupKFold(n_splits=n_splits)
    try:
        return StratifiedGroupKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
    except TypeError:  # pragma: no cover - very old scikit-learn
        return StratifiedGroupKFold(n_splits=n_splits)


def repeated_school_grouped_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield ``(repeat, fold, train_idx, test_idx)`` for repeated grouped CV."""
    for rep in range(n_repeats):
        splitter = school_grouped_splitter(
            n_splits=n_splits, shuffle=True, random_state=random_state + rep
        )
        for fold, (tr, te) in enumerate(splitter.split(X, y, groups=groups)):
            yield rep, fold, tr, te


def assert_group_disjoint(
    train_idx: np.ndarray, test_idx: np.ndarray, groups: np.ndarray
) -> None:
    """Hard check that no school appears in both sides of a split."""
    overlap = set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[test_idx])
    if overlap:
        raise DesignError(
            f"{len(overlap)} school(s) appear in BOTH the training and test "
            f"folds, e.g. {sorted(list(overlap))[:5]}. Students from one school "
            "are not independent; this split would inflate performance."
        )


def design_diagnostics(design: SurveyDesign) -> Dict[str, float]:
    """Summary of the realised design, reported in the sample-description table."""
    w = design.weights
    _, counts = np.unique(design.psu, return_counts=True)
    # Kish's effective sample size under unequal weighting.
    n_eff = float(np.sum(w) ** 2 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else float("nan")
    return {
        "n_students": int(design.n),
        "n_schools": int(len(counts)),
        "students_per_school_mean": float(counts.mean()),
        "students_per_school_min": int(counts.min()),
        "students_per_school_max": int(counts.max()),
        "sum_weights": float(np.sum(w)),
        "weight_min": float(np.min(w)),
        "weight_max": float(np.max(w)),
        "weight_cv": float(np.std(w) / np.mean(w)) if np.mean(w) else float("nan"),
        "kish_effective_n": n_eff,
        "design_effect_kish": float(design.n / n_eff) if n_eff else float("nan"),
        "n_strata": int(len(np.unique(design.stratum))) if design.stratum is not None else 0,
        "n_replicate_weights": design.n_replicates,
    }
