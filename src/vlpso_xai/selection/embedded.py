"""Embedded selectors: L1-penalised logistic regression, tree importance, Boruta.

Round-2 editor comment ("Five planned comparison methods were not evaluated")
and audit item 2: these three were declared in ``config/default.yaml`` but had
no implementation anywhere in ``src/``. They are implemented here so the
comparison covers the filter, wrapper and embedded families the original
manuscript named.

Rankers vs. selectors
---------------------
``L1PathRanker`` and ``TreeImportanceRanker`` produce a score per feature
(``scores_``, higher = more important) and keep the top ``k``. In the
comparison ``k`` is NOT fixed: they are wrapped in
:class:`~vlpso_xai.selection.tuned.TunedTopKSelector`, which chooses ``k`` from
the configured grid by school-grouped inner cross-validation, exactly as the
filters are. ``BorutaSelector`` is an all-relevant method whose subset size is
decided by its own statistical test, so it is used directly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .base import BaseSelector


class _Ranker(BaseSelector):
    """Score features, keep the top ``k`` (``k=None`` keeps all, ranked)."""

    k: Optional[int] = None

    def _score(self, values: np.ndarray, y: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def fit(self, X, y, groups=None, **kwargs):
        from ..data.features import assert_no_leakage

        if isinstance(X, pd.DataFrame):
            assert_no_leakage(X, where=f"{type(self).__name__}.fit")
        values = np.nan_to_num(self._record_input(X))
        y = np.asarray(y).astype(int)
        self.scores_ = np.nan_to_num(self._score(values, y), nan=-np.inf)
        k = values.shape[1] if self.k is None else int(min(self.k, values.shape[1]))
        mask = np.zeros(values.shape[1], dtype=bool)
        mask[np.argsort(-self.scores_, kind="stable")[:k]] = True
        self._finalise(mask)
        return self


class L1PathRanker(_Ranker):
    """Rank by the order in which features ENTER the L1 regularisation path.

    A feature that acquires a non-zero coefficient at a stronger penalty (smaller
    ``C``) is ranked higher. Features that never enter are ordered by their
    absolute coefficient at the weakest penalty on the path. This is the
    standard embedded use of the lasso for ranking, and it does not depend on a
    single arbitrary choice of ``C``.
    """

    def __init__(self, k: Optional[int] = None, n_cs: int = 25,
                 c_min: float = 1e-4, c_max: float = 1.0, random_state: int = 42):
        self.k = k
        self.n_cs = n_cs
        self.c_min = c_min
        self.c_max = c_max
        self.random_state = random_state

    def _score(self, values, y):
        from sklearn.linear_model import LogisticRegression

        cs = np.logspace(np.log10(self.c_min), np.log10(self.c_max), self.n_cs)
        entry = np.full(values.shape[1], np.inf)
        last = np.zeros(values.shape[1])
        for c in cs:
            m = LogisticRegression(penalty="l1", solver="liblinear", C=float(c),
                                   class_weight="balanced", max_iter=2000,
                                   random_state=self.random_state)
            m.fit(values, y)
            coef = np.abs(m.coef_.ravel())
            newly = (coef > 0) & ~np.isfinite(entry)
            entry[newly] = c
            last = coef
        self.entry_C_ = entry
        # Higher score = earlier entry; never-entered features rank below all
        # entered ones, ordered by |coef| at C_max.
        score = np.where(np.isfinite(entry), -np.log10(entry) + 100.0, last)
        return score


class TreeImportanceRanker(_Ranker):
    """Rank by random-forest mean decrease in impurity, fitted on the training fold."""

    def __init__(self, k: Optional[int] = None, n_estimators: int = 200,
                 min_samples_leaf: int = 10, max_features: str = "sqrt",
                 random_state: int = 42):
        self.k = k
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _score(self, values, y):
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, class_weight="balanced",
            random_state=self.random_state, n_jobs=1,
        ).fit(values, y)
        return rf.feature_importances_


class BorutaSelector(BaseSelector):
    """All-relevant selection by comparison with permuted shadow features.

    Kursa & Rudnicki (2010). In each iteration every feature is paired with a
    column-wise permuted copy ("shadow"); a random forest is fitted on the
    doubled matrix and a feature scores a *hit* when its importance exceeds the
    largest shadow importance. After ``max_iter`` iterations a two-sided
    binomial test against p = 0.5, Bonferroni-corrected over features, marks
    each feature confirmed, rejected or tentative. Tentative features are
    resolved by the usual rough fix: kept if their median importance exceeds
    the median of the per-iteration maximum shadow importance.

    Implemented here rather than imported because the ``boruta`` package pins
    NumPy aliases removed in NumPy 2, which the project requires.
    """

    def __init__(self, max_iter: int = 30, n_estimators: int = 100,
                 max_depth: Optional[int] = 7, alpha: float = 0.05,
                 min_features: int = 2, random_state: int = 42):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.alpha = alpha
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y, groups=None, **kwargs):
        from scipy.stats import binomtest
        from sklearn.ensemble import RandomForestClassifier

        from ..data.features import assert_no_leakage

        if isinstance(X, pd.DataFrame):
            assert_no_leakage(X, where="BorutaSelector.fit")
        values = np.nan_to_num(self._record_input(X))
        y = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        p = values.shape[1]
        hits = np.zeros(p, dtype=int)
        imps, shadow_max = [], []
        for it in range(self.max_iter):
            shadow = np.column_stack([rng.permutation(values[:, j]) for j in range(p)])
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                class_weight="balanced", n_jobs=1,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).fit(np.hstack([values, shadow]), y)
            imp = rf.feature_importances_
            real, sh = imp[:p], imp[p:]
            hits += real > sh.max()
            imps.append(real)
            shadow_max.append(sh.max())
        imps = np.asarray(imps)
        pvals_hi = np.array([binomtest(int(h), self.max_iter, 0.5, alternative="greater").pvalue for h in hits])
        pvals_lo = np.array([binomtest(int(h), self.max_iter, 0.5, alternative="less").pvalue for h in hits])
        thr = self.alpha / p                      # Bonferroni
        confirmed = pvals_hi < thr
        rejected = pvals_lo < thr
        tentative = ~confirmed & ~rejected
        rough = np.median(imps, axis=0) > np.median(shadow_max)
        mask = confirmed | (tentative & rough)
        self.hits_ = hits
        self.status_ = np.where(confirmed, "confirmed",
                                np.where(rejected, "rejected", "tentative"))
        self.scores_ = imps.mean(axis=0)
        self._finalise(mask, min_features=self.min_features)
        return self
