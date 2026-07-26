"""Outcome construction: plausible values, proficiency cut points, Rubin's rules.

Answers editor comments 2 and 3.

THE DEFECT THIS REPLACES
------------------------
``codes/data_preparation.py:157-180``::

    math_score_cols = [f'PV{i}MATH' for i in range(1, 11)]
    df['math_score'] = df[math_score_cols].mean(axis=1)
    df['math_category'] = df['math_score'].apply(categorize_score)

Two problems, one of which is fatal.

1. **Averaging the plausible values is invalid.** PVs are ten draws from each
   student's posterior proficiency distribution. Their mean is a
   *shrunken* estimate whose variance is systematically deflated: it discards
   precisely the measurement uncertainty the PV methodology exists to carry.
   Thresholding that mean produces a proficiency category with no defensible
   standard error. The OECD-sanctioned procedure is to run the complete
   analysis once per PV and combine with Rubin's rules.
2. ``math_score`` was then left in the feature matrix, so the label was a
   deterministic function of an available column. That is handled in
   ``features.py``; this module ensures the outcome is never materialised as a
   single leakable column in the first place.

WHAT THIS MODULE PROVIDES
-------------------------
* per-PV proficiency categories at the official PISA 2018 boundaries;
* the binary task labels for each PV;
* Rubin's-rules combination with the within/between variance decomposition
  reported separately, so reviewers can see how much uncertainty the PVs
  contribute;
* the proportion of students whose category assignment is unstable across
  PVs -- a substantive finding in its own right.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Official PISA 2018 mathematics proficiency level lower bounds (score points).
#: Source: OECD (2019), *PISA 2018 Results (Volume I)*, Annex A1; and the
#: *PISA 2018 Technical Report*, Ch. 15. Mirrored in config/default.yaml so the
#: analysis has a single source of truth; this dict is the documented default.
PISA2018_MATH_LEVEL_BOUNDS: Dict[str, float] = {
    "level_1a": 357.77,
    "level_2": 420.07,
    "level_3": 482.38,
    "level_4": 544.68,
    "level_5": 606.99,
    "level_6": 669.30,
}

#: Low = below Level 3; Medium = Levels 3-4; High = Levels 5-6.
DEFAULT_CATEGORY_CUTS: Tuple[float, float] = (
    PISA2018_MATH_LEVEL_BOUNDS["level_3"],
    PISA2018_MATH_LEVEL_BOUNDS["level_5"],
)
DEFAULT_CATEGORY_LABELS = ("Low", "Medium", "High")


def pv_columns(df: pd.DataFrame, domain: str = "MATH", m: int = 10) -> List[str]:
    cols = [f"PV{i}{domain}" for i in range(1, m + 1)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Plausible-value columns absent from the frame: {missing}. "
            "The analytic file must retain the PVs so the outcome can be "
            "derived; features.py keeps them out of X."
        )
    return cols


def categorize(
    scores: np.ndarray | pd.Series,
    cuts: Sequence[float] = DEFAULT_CATEGORY_CUTS,
    labels: Sequence[str] = DEFAULT_CATEGORY_LABELS,
) -> pd.Categorical:
    """Map PISA scale scores to ordered proficiency categories.

    Note the boundary convention. The original code used ``s <= 482`` and
    ``s <= 607``; the official Level 3 boundary is 482.38 and Level 5 is
    606.99, and the intervals are half-open from below. Students scoring in
    [482, 482.38) were misclassified as Medium and students in
    [606.99, 607] as Medium rather than High.
    """
    s = np.asarray(scores, dtype=float)
    idx = np.digitize(s, np.asarray(cuts, dtype=float), right=False)
    idx = np.where(np.isnan(s), -1, idx)
    out = np.array([labels[i] if i >= 0 else None for i in idx], dtype=object)
    return pd.Categorical(out, categories=list(labels), ordered=True)


def build_pv_categories(
    df: pd.DataFrame,
    *,
    domain: str = "MATH",
    m: int = 10,
    cuts: Sequence[float] = DEFAULT_CATEGORY_CUTS,
    labels: Sequence[str] = DEFAULT_CATEGORY_LABELS,
) -> pd.DataFrame:
    """One proficiency category per plausible value: shape ``(n, m)``."""
    cols = pv_columns(df, domain, m)
    return pd.DataFrame(
        {f"cat_pv{i+1}": categorize(df[c], cuts, labels) for i, c in enumerate(cols)},
        index=df.index,
    )


def task_labels(
    categories: pd.Series | pd.Categorical,
    negative: str,
    positive: str,
) -> pd.Series:
    """Binary label for one task; rows in neither class become NaN."""
    s = pd.Series(np.asarray(categories, dtype=object))
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s == negative] = 0.0
    out[s == positive] = 1.0
    return out


# ---------------------------------------------------------------------------
# Category instability across plausible values
# ---------------------------------------------------------------------------
def category_instability(
    pv_cats: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """How often does a student's category depend on which PV you picked?

    Reported as its own table in the revision. A student whose ten PVs
    straddle a proficiency boundary has no stable category, and any analysis
    that thresholds a single averaged score silently assigns them one.
    """
    arr = pv_cats.to_numpy(dtype=object)
    n, m = arr.shape
    unstable = np.array([len(set(row[pd.notna(row)])) > 1 for row in arr])
    modal_share = np.array(
        [
            (pd.Series(row).value_counts(normalize=True).iloc[0] if pd.notna(row).any() else np.nan)
            for row in arr
        ]
    )

    def _mean(v):
        v = np.asarray(v, dtype=float)
        ok = np.isfinite(v)
        if weights is None:
            return float(np.nanmean(v[ok])) if ok.any() else float("nan")
        w = np.asarray(weights, dtype=float)[ok]
        return float(np.sum(v[ok] * w) / np.sum(w)) if ok.any() else float("nan")

    return {
        "n_students": int(n),
        "m_plausible_values": int(m),
        "prop_unstable": _mean(unstable.astype(float)),
        "mean_modal_share": _mean(modal_share),
        "prop_fully_stable": _mean((~unstable).astype(float)),
        "weighted": weights is not None,
    }


# ---------------------------------------------------------------------------
# Rubin's rules
# ---------------------------------------------------------------------------
@dataclass
class RubinResult:
    """Combined estimate across M plausible values, variance decomposed."""

    estimate: float
    within_variance: float
    between_variance: float
    total_variance: float
    standard_error: float
    df: float
    ci_low: float
    ci_high: float
    m: int
    fraction_missing_information: float
    relative_increase_variance: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "estimate": self.estimate,
            "within_variance": self.within_variance,
            "between_variance": self.between_variance,
            "total_variance": self.total_variance,
            "standard_error": self.standard_error,
            "df": self.df,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "m": self.m,
            "fmi": self.fraction_missing_information,
            "riv": self.relative_increase_variance,
        }


def rubin_combine(
    estimates: Sequence[float],
    variances: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
) -> RubinResult:
    r"""Combine M per-PV estimates with Rubin's rules.

    .. math::

        \bar{Q} = \frac{1}{M}\sum_m Q_m \qquad
        U = \frac{1}{M}\sum_m U_m \qquad
        B = \frac{1}{M-1}\sum_m (Q_m - \bar{Q})^2

        T = U + \left(1 + \frac{1}{M}\right) B

    ``U`` (within-imputation) is the average sampling variance of the
    individual analyses; ``B`` (between-imputation) is the variance across
    plausible values, i.e. the measurement uncertainty in proficiency. Both
    components are returned so the manuscript can report the PV contribution
    separately, as editor comment 3 requires.

    Parameters
    ----------
    estimates:
        One estimate per plausible value.
    variances:
        Sampling variance of each estimate. If omitted, ``U = 0`` and the
        result reflects PV variance ONLY -- valid as a decomposition, but it
        understates total uncertainty. The caller must say which it used.
    """
    from scipy import stats

    q = np.asarray(estimates, dtype=float)
    q = q[np.isfinite(q)]
    m = q.size
    if m == 0:
        raise ValueError("No finite estimates to combine.")
    if m == 1:
        logger.warning(
            "rubin_combine received a single plausible value. Between-"
            "imputation variance is undefined and reported as 0. This is the "
            "quick-mode path and must not be used for published numbers."
        )

    qbar = float(np.mean(q))
    u = float(np.mean(np.asarray(variances, dtype=float))) if variances is not None else 0.0
    b = float(np.var(q, ddof=1)) if m > 1 else 0.0
    t = u + (1.0 + 1.0 / m) * b
    se = float(np.sqrt(t)) if t > 0 else 0.0

    if t > 0 and b > 0 and m > 1:
        riv = (1.0 + 1.0 / m) * b / u if u > 0 else np.inf
        df = (m - 1) * (1.0 + u / ((1.0 + 1.0 / m) * b)) ** 2 if b > 0 else np.inf
        fmi = ((1.0 + 1.0 / m) * b) / t
    else:
        riv, df, fmi = 0.0, np.inf, 0.0

    crit = stats.t.ppf(1 - alpha / 2, df) if np.isfinite(df) else stats.norm.ppf(1 - alpha / 2)
    return RubinResult(
        estimate=qbar,
        within_variance=u,
        between_variance=b,
        total_variance=t,
        standard_error=se,
        df=float(df),
        ci_low=qbar - crit * se,
        ci_high=qbar + crit * se,
        m=int(m),
        fraction_missing_information=float(fmi),
        relative_increase_variance=float(riv),
    )


def rubin_frame(
    per_pv: pd.DataFrame,
    metric_col: str = "value",
    variance_col: Optional[str] = None,
    group_cols: Sequence[str] = (),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply :func:`rubin_combine` within each group of a long results frame."""
    rows = []
    grouped = per_pv.groupby(list(group_cols), dropna=False) if group_cols else [((), per_pv)]
    for key, sub in grouped:
        res = rubin_combine(
            sub[metric_col].to_numpy(),
            sub[variance_col].to_numpy() if variance_col else None,
            alpha=alpha,
        )
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update(res.as_dict())
        rows.append(row)
    return pd.DataFrame(rows)
