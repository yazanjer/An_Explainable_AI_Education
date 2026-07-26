"""Filter selectors, including a real symmetric-uncertainty implementation.

Answers editor comment 5 ("...established filter, wrapper, and embedded
feature-selection methods. Include ablations that separately evaluate ...
the symmetric-uncertainty component ...").

THE DEFECT THIS REPLACES
------------------------
The manuscript describes a symmetric-uncertainty component of the VLPSO
search. The submitted code never implemented it. ``feature_selection.ipynb``
cell 8 ranks features with ``mutual_info_classif`` and says so in a comment::

    # Rank features using mutual information (instead of SU for now)

The author has elected to implement SU properly and ablate it against MI
rather than delete the claim, so this module provides the real thing.

Symmetric uncertainty
---------------------
.. math::

    SU(X, Y) = \\frac{2\\,I(X;Y)}{H(X) + H(Y)} \\in [0, 1]

SU is mutual information normalised by the summed marginal entropies. Unlike
raw MI it is bounded, and it does not inflate for high-cardinality variables
-- which matters here because the candidate set mixes binary items
(ST011, 2 levels) with 6-point ordinal items (ST013, ST166).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .base import BaseSelector


# ---------------------------------------------------------------------------
# Information-theoretic primitives
# ---------------------------------------------------------------------------
def _discretise(x: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Equal-frequency binning for continuous columns; discrete passed through.

    PISA questionnaire items are already discrete with few levels, so they are
    left alone. Only genuinely continuous columns are binned, and the binning
    is equal-frequency so that entropy is not dominated by a sparse tail.
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    out = np.full(x.shape, -1, dtype=np.int64)
    if not finite.any():
        return out
    vals = x[finite]
    uniq = np.unique(vals)
    if uniq.size <= n_bins:
        lookup = {v: i for i, v in enumerate(uniq)}
        out[finite] = np.array([lookup[v] for v in vals], dtype=np.int64)
        return out
    quantiles = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1])
    out[finite] = np.searchsorted(np.unique(quantiles), vals, side="right")
    return out


def entropy(labels: np.ndarray) -> float:
    """Shannon entropy in nats of a discrete vector, ignoring missing (-1)."""
    labels = np.asarray(labels)
    labels = labels[labels >= 0] if labels.dtype.kind in "iu" else labels
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def conditional_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """H(X | Y)."""
    x, y = np.asarray(x), np.asarray(y)
    ok = (x >= 0) & (y >= 0) if x.dtype.kind in "iu" else np.ones(len(x), bool)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return 0.0
    total = 0.0
    for v in np.unique(y):
        m = y == v
        total += (m.sum() / x.size) * entropy(x[m])
    return float(total)


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """I(X;Y) = H(X) - H(X|Y), on discretised inputs."""
    return float(max(entropy(x) - conditional_entropy(x, y), 0.0))


def symmetric_uncertainty(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    r"""SU(X, Y) = 2 I(X;Y) / (H(X) + H(Y)), in [0, 1]."""
    xd = _discretise(x, n_bins)
    yd = _discretise(y, n_bins)
    hx, hy = entropy(xd), entropy(yd)
    denom = hx + hy
    if denom <= 0:
        return 0.0
    return float(np.clip(2.0 * mutual_information(xd, yd) / denom, 0.0, 1.0))


def su_matrix(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Pairwise SU between features; used for the redundancy/interpretability term."""
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    disc = [_discretise(X[:, j], n_bins) for j in range(p)]
    ent = [entropy(d) for d in disc]
    M = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            denom = ent[i] + ent[j]
            v = 0.0 if denom <= 0 else 2.0 * mutual_information(disc[i], disc[j]) / denom
            M[i, j] = M[j, i] = float(np.clip(v, 0.0, 1.0))
    return M


def su_scores(X, y, n_bins: int = 10) -> np.ndarray:
    """SU of every feature against the label. The VLPSO ranking criterion."""
    values = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)
    y = np.asarray(y)
    return np.array(
        [symmetric_uncertainty(values[:, j], y, n_bins) for j in range(values.shape[1])]
    )


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------
class _TopKFilter(BaseSelector):
    """Shared machinery for score-then-take-top-k filters."""

    def __init__(self, k: int = 10, n_bins: int = 10):
        self.k = k
        self.n_bins = n_bins

    def _score(self, values: np.ndarray, y: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def fit(self, X, y, **kwargs):
        values = self._record_input(X)
        y = np.asarray(y)
        self.scores_ = np.nan_to_num(self._score(values, y), nan=-np.inf)
        k = int(min(self.k, values.shape[1]))
        mask = np.zeros(values.shape[1], dtype=bool)
        mask[np.argsort(-self.scores_)[:k]] = True
        self._finalise(mask)
        return self


class SymmetricUncertaintyFilter(_TopKFilter):
    """Rank by SU(feature, label) and keep the top ``k``."""

    def _score(self, values, y):
        return np.array(
            [symmetric_uncertainty(values[:, j], y, self.n_bins) for j in range(values.shape[1])]
        )


class MutualInfoFilter(_TopKFilter):
    """Rank by ``mutual_info_classif``. The MI arm of the SU ablation."""

    def __init__(self, k: int = 10, n_bins: int = 10, random_state: int = 42):
        super().__init__(k=k, n_bins=n_bins)
        self.random_state = random_state

    def _score(self, values, y):
        from sklearn.feature_selection import mutual_info_classif

        return mutual_info_classif(
            np.nan_to_num(values), y, random_state=self.random_state
        )


class Chi2Filter(_TopKFilter):
    """Chi-squared on min-max shifted features (chi2 requires non-negative input)."""

    def _score(self, values, y):
        from sklearn.feature_selection import chi2

        v = np.nan_to_num(values)
        v = v - v.min(axis=0, keepdims=True)
        stat, _ = chi2(v, y)
        return stat


class ReliefFFilter(_TopKFilter):
    """ReliefF: weights features by how well they separate near hits from near misses.

    Uses ``skrebate`` when available and otherwise falls back to a compact
    reference implementation, so the comparison never silently drops a
    baseline. Which path was taken is recorded in ``implementation_``.
    """

    def __init__(self, k: int = 10, n_neighbors: int = 10, n_samples: int = 500,
                 random_state: int = 42, n_bins: int = 10):
        super().__init__(k=k, n_bins=n_bins)
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.random_state = random_state

    def _score(self, values, y):
        try:
            from skrebate import ReliefF as _SkReliefF

            est = _SkReliefF(n_neighbors=self.n_neighbors, n_features_to_select=self.k)
            est.fit(np.nan_to_num(values), y)
            self.implementation_ = "skrebate"
            return np.asarray(est.feature_importances_, dtype=float)
        except Exception:
            self.implementation_ = "internal"
            return self._relieff(np.nan_to_num(values), np.asarray(y))

    def _relieff(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape
        rng_span = X.max(axis=0) - X.min(axis=0)
        rng_span[rng_span == 0] = 1.0
        Xs = X / rng_span
        idx = rng.choice(n, size=int(min(self.n_samples, n)), replace=False)
        classes, counts = np.unique(y, return_counts=True)
        priors = dict(zip(classes, counts / n))
        w = np.zeros(p)
        for i in idx:
            d = np.abs(Xs - Xs[i])
            dist = d.sum(axis=1)
            dist[i] = np.inf
            for c in classes:
                same = c == y[i]
                cand = np.where(y == c)[0]
                cand = cand[cand != i]
                if cand.size == 0:
                    continue
                nn = cand[np.argsort(dist[cand])[: self.n_neighbors]]
                contrib = d[nn].mean(axis=0)
                if same:
                    w -= contrib / idx.size
                else:
                    w += (priors[c] / (1 - priors[y[i]] + 1e-12)) * contrib / idx.size
        return w
