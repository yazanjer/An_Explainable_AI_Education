"""Effect sizes, rewritten from scratch.

Answers editor comment 6: "Correct the effect-size section. Explain exactly
how Cohen's d was calculated and what groups or distributions were compared.
The values in Tables 3 and 4 are identical and should be checked. Values
around 0.13-0.18 should not be described as large effects. Consider whether
paired comparisons across repeated outer folds, with uncertainty intervals and
correction for multiple comparisons, would be more appropriate."

WHAT WAS DELETED AND WHY
------------------------
``codes/evaluation.py:39-57`` did this::

    y_pred = model.predict(X)
    pos_class = X[y_pred == 1]; neg_class = X[y_pred == 0]
    mean1, mean2 = pos_class.mean(), neg_class.mean()   # per-column Series
    ...
    cohens_d = (mean1.mean() - mean2.mean()) / s_pooled.mean()

That is the difference in the **grand mean of all standardised feature
columns** between rows the model *predicted* positive and rows it predicted
negative, divided by the average of per-column pooled SDs. It is:

* not an effect size for classifier performance;
* not an effect size for any group contrast in the data;
* computed against *predicted* rather than true labels;
* necessarily near zero, because standardised features average to ~0 by
  construction -- which is the whole explanation for the 0.05-0.18 range the
  manuscript then described as "large effects" (main.tex:651).

The function is gone. Two clearly distinct quantities replace it, and they are
never reported in the same table.

(a) METHOD COMPARISON -- :func:`paired_method_contrast`
    Is selector A better than selector B? Paired across matched outer folds,
    with the Nadeau-Bengio correction for the dependence induced by
    overlapping CV training sets, CIs, and multiplicity correction.

(b) GROUP DIFFERENCES -- :func:`group_standardised_difference`
    How far apart are the true proficiency groups on a given predictor?
    Survey-weighted, with BRR standard errors, reported PER VARIABLE and never
    averaged across variables.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Cohen's conventions. Reported honestly: 0.13-0.18 is NEGLIGIBLE.
COHEN_THRESHOLDS = {"negligible": 0.2, "small": 0.5, "medium": 0.8}


def interpret_d(d: float, thresholds: Dict[str, float] = COHEN_THRESHOLDS) -> str:
    """Map |d| to Cohen's conventional bands. No inflation."""
    a = abs(float(d))
    if not np.isfinite(a):
        return "undefined"
    if a < thresholds["negligible"]:
        return "negligible"
    if a < thresholds["small"]:
        return "small"
    if a < thresholds["medium"]:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Textbook Cohen's d -- unit-testable against analytic values
# ---------------------------------------------------------------------------
def cohens_d_independent(a: Sequence[float], b: Sequence[float]) -> float:
    r"""Cohen's *d* for two independent samples.

    .. math::

        d = \frac{\bar{a} - \bar{b}}{s_p}, \qquad
        s_p = \sqrt{\frac{(n_a-1)s_a^2 + (n_b-1)s_b^2}{n_a + n_b - 2}}
    """
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's *d* with the small-sample bias correction J."""
    d = cohens_d_independent(a, b)
    na = np.isfinite(np.asarray(a, float)).sum()
    nb = np.isfinite(np.asarray(b, float)).sum()
    df = na + nb - 2
    if df <= 0 or not np.isfinite(d):
        return float("nan")
    return float(d * (1 - 3 / (4 * df - 1)))


def cohens_d_paired(diff: Sequence[float]) -> float:
    r"""Cohen's *d* for paired samples: :math:`\bar{D} / s_D`."""
    d = np.asarray(diff, dtype=float); d = d[np.isfinite(d)]
    if d.size < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1))


# ---------------------------------------------------------------------------
# (a) Method comparison across matched outer folds
# ---------------------------------------------------------------------------
@dataclass
class PairedContrast:
    method_a: str
    method_b: str
    metric: str
    n_folds: int
    mean_a: float
    mean_b: float
    mean_difference: float
    cohens_d_paired: float
    magnitude: str
    t_statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    test: str
    train_test_ratio: Optional[float] = None

    def as_dict(self) -> Dict:
        return asdict(self)


def nadeau_bengio_t(
    differences: Sequence[float],
    n_train: int,
    n_test: int,
    alpha: float = 0.05,
) -> Dict[str, float]:
    r"""Corrected resampled *t*-test (Nadeau & Bengio, 2003).

    In repeated k-fold CV the per-fold differences are NOT independent: the
    training sets overlap heavily, so the naive paired *t*-test has a variance
    estimate that is far too small and a Type-I error rate that can exceed the
    nominal level by an order of magnitude. The correction inflates the
    variance by the train/test size ratio:

    .. math::

        t = \frac{\bar{D}}{\sqrt{\left(\frac{1}{J} +
            \frac{n_{\text{test}}}{n_{\text{train}}}\right) s_D^2}}

    with ``J`` the number of folds. This is why the editor's suggestion of
    "paired comparisons across repeated outer folds" is implemented with this
    statistic rather than a plain paired *t*.
    """
    from scipy import stats

    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    j = d.size
    if j < 2:
        return {"t": float("nan"), "p": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "ratio": float("nan")}
    var = d.var(ddof=1)
    ratio = n_test / max(n_train, 1)
    corrected_var = (1.0 / j + ratio) * var
    se = np.sqrt(corrected_var)
    t = d.mean() / se if se > 0 else np.nan
    df = j - 1
    p = float(2 * (1 - stats.t.cdf(abs(t), df))) if np.isfinite(t) else float("nan")
    crit = stats.t.ppf(1 - alpha / 2, df)
    return {
        "t": float(t),
        "p": p,
        "ci_low": float(d.mean() - crit * se),
        "ci_high": float(d.mean() + crit * se),
        "ratio": float(ratio),
    }


def paired_method_contrast(
    fold_results: pd.DataFrame,
    method_a: str,
    method_b: str,
    *,
    metric: str = "auc",
    method_col: str = "method",
    fold_cols: Sequence[str] = ("task", "pv", "repeat", "outer_fold"),
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    alpha: float = 0.05,
    test: str = "nadeau_bengio",
) -> PairedContrast:
    """Compare two selectors on MATCHED outer folds.

    Matching is essential: the two methods must be compared on the same folds
    of the same plausible value and the same repeat, otherwise fold-to-fold
    variation swamps the method difference.
    """
    from scipy import stats

    keys = list(fold_cols)
    a = fold_results[fold_results[method_col] == method_a].set_index(keys)[metric]
    b = fold_results[fold_results[method_col] == method_b].set_index(keys)[metric]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if joined.empty:
        raise ValueError(
            f"No matched folds for {method_a!r} vs {method_b!r} on {metric!r}. "
            f"Matching keys: {keys}."
        )
    diff = (joined["a"] - joined["b"]).to_numpy()

    if test == "nadeau_bengio":
        if n_train is None or n_test is None:
            raise ValueError(
                "The Nadeau-Bengio correction needs n_train and n_test. Pass "
                "them, or set test='paired_t' and label the result as "
                "uncorrected in the manuscript."
            )
        stat = nadeau_bengio_t(diff, n_train, n_test, alpha)
        t, p, lo, hi, ratio = stat["t"], stat["p"], stat["ci_low"], stat["ci_high"], stat["ratio"]
        label = "Nadeau-Bengio corrected resampled t"
    elif test == "paired_t":
        t, p = stats.ttest_rel(joined["a"], joined["b"])
        se = diff.std(ddof=1) / np.sqrt(diff.size)
        crit = stats.t.ppf(1 - alpha / 2, diff.size - 1)
        lo, hi, ratio = diff.mean() - crit * se, diff.mean() + crit * se, None
        label = "paired t (UNCORRECTED for CV fold dependence)"
    elif test == "wilcoxon":
        t, p = stats.wilcoxon(joined["a"], joined["b"])
        lo = hi = float("nan"); ratio = None
        label = "Wilcoxon signed-rank"
    else:
        raise ValueError(f"Unknown test {test!r}")

    d = cohens_d_paired(diff)
    return PairedContrast(
        method_a=method_a, method_b=method_b, metric=metric,
        n_folds=int(diff.size),
        mean_a=float(joined["a"].mean()), mean_b=float(joined["b"].mean()),
        mean_difference=float(diff.mean()),
        cohens_d_paired=d, magnitude=interpret_d(d),
        t_statistic=float(t), p_value=float(p),
        ci_low=float(lo), ci_high=float(hi),
        test=label, train_test_ratio=ratio,
    )


def contrast_table(
    fold_results: pd.DataFrame,
    reference: str = "vlpso",
    *,
    metric: str = "auc",
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    correction: str = "holm",
    method_col: str = "method",
    by: Sequence[str] = ("task",),
    **kwargs,
) -> pd.DataFrame:
    """Every baseline against the reference, with multiplicity correction.

    ``correction`` is ``'holm'`` (Holm-Bonferroni, controls FWER) or
    ``'benjamini_hochberg'`` (controls FDR). The family is defined by ``by``:
    by default, corrections are applied within task, because the three tasks
    answer different questions.
    """
    rows: List[Dict] = []
    groups = fold_results.groupby(list(by)) if by else [((), fold_results)]
    for key, sub in groups:
        others = [m for m in sub[method_col].unique() if m != reference]
        for other in others:
            try:
                c = paired_method_contrast(
                    sub, reference, other, metric=metric,
                    n_train=n_train, n_test=n_test, method_col=method_col, **kwargs
                )
            except ValueError:
                continue
            row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
            row.update(c.as_dict())
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_adjusted"] = np.nan
    out["correction"] = correction
    for key, idx in out.groupby(list(by)).groups.items() if by else [((), out.index)]:
        p = out.loc[idx, "p_value"].to_numpy()
        out.loc[idx, "p_adjusted"] = _adjust(p, correction)
    out["significant_adjusted"] = out["p_adjusted"] < 0.05
    return out


def _adjust(p: np.ndarray, method: str) -> np.ndarray:
    """Holm-Bonferroni or Benjamini-Hochberg adjustment."""
    p = np.asarray(p, dtype=float)
    n = p.size
    out = np.full(n, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * pv[order])
    elif method in ("benjamini_hochberg", "bh", "fdr_bh"):
        adj = np.minimum.accumulate(((m / (np.arange(m) + 1)) * pv[order])[::-1])[::-1]
    else:
        raise ValueError(f"Unknown correction {method!r}")
    adj = np.clip(adj, 0, 1)
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


# ---------------------------------------------------------------------------
# (b) Group differences on individual predictors
# ---------------------------------------------------------------------------
def group_standardised_difference(
    x: np.ndarray,
    group: np.ndarray,
    positive: str | int,
    negative: str | int,
    *,
    weights: Optional[np.ndarray] = None,
    design=None,
    variable: str = "",
) -> Dict[str, float]:
    """Standardised mean difference between TRUE proficiency groups.

    Survey-weighted when weights are supplied, with a BRR standard error when
    a :class:`~vlpso_xai.data.design.SurveyDesign` is supplied. Reported per
    variable -- averaging a standardised difference across variables, as the
    deleted function effectively did, is meaningless.
    """
    from ..data.design import brr_standard_error, weighted_mean, weighted_var

    x = np.asarray(x, dtype=float)
    group = np.asarray(group)
    mpos, mneg = group == positive, group == negative
    if mpos.sum() < 2 or mneg.sum() < 2:
        return {"variable": variable, "d": float("nan"), "n_pos": int(mpos.sum()),
                "n_neg": int(mneg.sum())}

    if weights is None:
        d = cohens_d_independent(x[mpos], x[mneg])
        g = hedges_g(x[mpos], x[mneg])
        res = {
            "variable": variable, "positive": str(positive), "negative": str(negative),
            "mean_pos": float(np.nanmean(x[mpos])), "mean_neg": float(np.nanmean(x[mneg])),
            "d": d, "hedges_g": g, "magnitude": interpret_d(d),
            "n_pos": int(mpos.sum()), "n_neg": int(mneg.sum()), "weighted": False,
        }
        return res

    def _d(w: np.ndarray) -> float:
        mp, mn = weighted_mean(x[mpos], w[mpos]), weighted_mean(x[mneg], w[mneg])
        vp, vn = weighted_var(x[mpos], w[mpos]), weighted_var(x[mneg], w[mneg])
        np_, nn_ = w[mpos].sum(), w[mneg].sum()
        sp = np.sqrt(((np_ - 1) * vp + (nn_ - 1) * vn) / max(np_ + nn_ - 2, 1e-9))
        return float((mp - mn) / sp) if sp > 0 else float("nan")

    w = np.asarray(weights, dtype=float)
    d = _d(w)
    res = {
        "variable": variable, "positive": str(positive), "negative": str(negative),
        "mean_pos": weighted_mean(x[mpos], w[mpos]),
        "mean_neg": weighted_mean(x[mneg], w[mneg]),
        "d": d, "magnitude": interpret_d(d),
        "n_pos": int(mpos.sum()), "n_neg": int(mneg.sum()), "weighted": True,
    }
    if design is not None and design.replicates is not None:
        res.update({f"brr_{k}": v for k, v in brr_standard_error(_d, design).items()})
    return res


def group_difference_table(
    X: pd.DataFrame,
    group: np.ndarray,
    positive: str,
    negative: str,
    *,
    weights: Optional[np.ndarray] = None,
    design=None,
    codebook=None,
) -> pd.DataFrame:
    """One row per predictor. Never collapsed to a single number."""
    rows = [
        group_standardised_difference(
            X[c].to_numpy(dtype=float), group, positive, negative,
            weights=weights, design=design, variable=c,
        )
        for c in X.columns
    ]
    out = pd.DataFrame(rows)
    if codebook is not None and not out.empty:
        out["codebook_label"] = [codebook.label(v) for v in out["variable"]]
    return out.sort_values("d", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
