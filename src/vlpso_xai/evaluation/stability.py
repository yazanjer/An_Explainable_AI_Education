"""Feature-selection stability across outer folds.

Answers editor comment 4 ("...feature-selection stability...").

The submitted analysis ran VLPSO once, on the full dataset, and reported a
single feature set -- which, because of the index-misalignment bug
(AUDIT_REPORT.md section C.2), was not even the set the algorithm chose. There
was nothing to be stable or unstable.

Under nested CV the selector is refitted inside every inner loop, so the
selected subset varies by fold. That variation is DATA, not noise to be
averaged away: a selector that picks a different subset every fold is telling
you its choices are not identifiable from this sample.

An honest result here may well be that VLPSO is LESS stable than embedded
methods. Report it either way.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """|A ∩ B| / |A ∪ B|. Undefined (NaN) when both sets are empty."""
    sa, sb = set(a), set(b)
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else float("nan")


def kuncheva(a: Sequence[str], b: Sequence[str], n_total: int) -> float:
    r"""Kuncheva's consistency index.

    .. math::

        I_C(A,B) = \frac{rn - k^2}{k(n - k)}

    where :math:`r = |A \cap B|`, :math:`k = |A| = |B|` and :math:`n` is the
    total number of candidate features. Unlike Jaccard it corrects for the
    overlap expected by chance, which matters because two random subsets of
    size 20 drawn from 31 candidates already share about 13 features. Defined
    only for equal-sized subsets; returns NaN otherwise.
    """
    sa, sb = set(a), set(b)
    k = len(sa)
    if k != len(sb) or k == 0 or k == n_total:
        return float("nan")
    r = len(sa & sb)
    return float((r * n_total - k * k) / (k * (n_total - k)))


def nogueira(subsets: Sequence[Sequence[str]], all_features: Sequence[str]) -> Dict[str, float]:
    r"""Nogueira, Sechidis & Brown (2018) stability estimator, with its variance.

    .. math::

        \hat\Phi = 1 - \frac{\tfrac{1}{d}\sum_f s_f^2}
                         {\tfrac{\bar k}{d}\left(1-\tfrac{\bar k}{d}\right)},
        \qquad s_f^2 = \tfrac{M}{M-1}\hat p_f(1-\hat p_f)

    over ``M`` selected subsets of a ``d``-feature universe. Unlike Kuncheva's
    index it is defined when subset sizes differ -- which is exactly the case
    for the variable-length search, whose subsets ranged from 2 to 17 features
    and for which Kuncheva's index was therefore undefined in round 1. It is
    corrected for chance (0 = random selection of the same average size,
    1 = identical subsets). The asymptotic variance follows Theorem 6 of the
    paper and gives a 95% interval.
    """
    feats = list(all_features)
    idx = {f: i for i, f in enumerate(feats)}
    subsets = [list(s) for s in subsets if s is not None]
    M, d = len(subsets), len(feats)
    if M < 2 or d < 2:
        return {"nogueira": float("nan"), "nogueira_ci_low": float("nan"),
                "nogueira_ci_high": float("nan"), "n_subsets": M}
    Z = np.zeros((M, d))
    for i, s in enumerate(subsets):
        for f in s:
            Z[i, idx[f]] = 1.0
    p_hat = Z.mean(axis=0)
    k_bar = Z.sum(axis=1).mean()
    denom = (k_bar / d) * (1 - k_bar / d)
    if denom <= 0:
        return {"nogueira": float("nan"), "nogueira_ci_low": float("nan"),
                "nogueira_ci_high": float("nan"), "n_subsets": M}
    s2 = M / (M - 1) * p_hat * (1 - p_hat)
    phi = 1 - s2.mean() / denom
    # Variance (Nogueira et al. 2018, Thm 6).
    ki = Z.sum(axis=1)
    phi_i = np.empty(M)
    for i in range(M):
        phi_i[i] = (1.0 / denom) * (np.mean(Z[i] * p_hat) - ki[i] * k_bar / d ** 2
                                     + (phi / 2) * (2 * k_bar * ki[i] / d ** 2 - ki[i] / d - k_bar / d + 1))
    phi_bar = phi_i.mean()
    var = 4.0 / M ** 2 * np.sum((phi_i - phi_bar) ** 2)
    half = 1.959963984540054 * np.sqrt(var)
    return {"nogueira": float(phi), "nogueira_ci_low": float(phi - half),
            "nogueira_ci_high": float(phi + half), "n_subsets": M}


def pairwise_stability(
    subsets: Sequence[Sequence[str]],
    n_total: int,
    indices: Sequence[str] = ("jaccard", "kuncheva"),
) -> Dict[str, float]:
    """Mean pairwise stability over all fold pairs."""
    subsets = [list(s) for s in subsets if s is not None]
    if len(subsets) < 2:
        return {f"{i}_mean": float("nan") for i in indices} | {"n_subsets": len(subsets)}

    out: Dict[str, float] = {"n_subsets": len(subsets)}
    for name in indices:
        fn = {"jaccard": jaccard, "kuncheva": kuncheva}[name]
        vals = [
            fn(a, b) if name == "jaccard" else fn(a, b, n_total)
            for a, b in combinations(subsets, 2)
        ]
        vals = np.asarray(vals, dtype=float)
        finite = vals[np.isfinite(vals)]
        out[f"{name}_mean"] = float(finite.mean()) if finite.size else float("nan")
        out[f"{name}_sd"] = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
        out[f"{name}_n_pairs"] = int(finite.size)
    sizes = [len(s) for s in subsets]
    out.update({
        "size_mean": float(np.mean(sizes)), "size_sd": float(np.std(sizes, ddof=1)) if len(sizes) > 1 else 0.0,
        "size_min": int(np.min(sizes)), "size_max": int(np.max(sizes)),
    })
    return out


def selection_frequency(
    subsets: Sequence[Sequence[str]],
    all_features: Sequence[str],
    *,
    consensus_threshold: float = 0.80,
    alpha: float = 0.05,
    codebook=None,
) -> pd.DataFrame:
    """Per-feature selection frequency with binomial (Wilson) CIs.

    The consensus set -- features selected in at least ``consensus_threshold``
    of folds -- is what the manuscript may describe as "the features the method
    chose". A feature selected in 40% of folds is not a finding.
    """
    subsets = [set(s) for s in subsets]
    n = len(subsets)
    rows = []
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z(alpha)
    for f in all_features:
        k = sum(f in s for s in subsets)
        p = k / n if n else float("nan")
        if n:
            denom = 1 + z ** 2 / n
            centre = (p + z ** 2 / (2 * n)) / denom
            half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
            lo, hi = max(0.0, centre - half), min(1.0, centre + half)
        else:
            lo = hi = float("nan")
        row = {
            "feature": f, "n_folds": n, "n_selected": k, "frequency": p,
            "ci_low": lo, "ci_high": hi,
            "consensus": bool(p >= consensus_threshold),
        }
        if codebook is not None:
            row["codebook_label"] = codebook.label(f)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("frequency", ascending=False).reset_index(drop=True)


def _z(alpha: float) -> float:
    from scipy import stats
    return float(stats.norm.ppf(1 - alpha / 2))


def stability_table(
    fold_selections: pd.DataFrame,
    all_features: Sequence[str],
    *,
    group_cols: Sequence[str] = ("task", "method"),
    selection_col: str = "selected",
    consensus_threshold: float = 0.80,
) -> pd.DataFrame:
    """Stability indices for every task x selector combination."""
    rows = []
    for key, sub in fold_selections.groupby(list(group_cols)):
        subsets = [list(s) for s in sub[selection_col]]
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update(pairwise_stability(subsets, len(all_features)))
        freq = selection_frequency(subsets, all_features, consensus_threshold=consensus_threshold)
        consensus = freq.loc[freq["consensus"], "feature"].tolist()
        row["n_consensus"] = len(consensus)
        row["consensus_features"] = ", ".join(consensus)
        rows.append(row)
    return pd.DataFrame(rows)
