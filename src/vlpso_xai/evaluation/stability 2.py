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
