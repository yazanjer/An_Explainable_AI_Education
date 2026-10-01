"""Wrapper selectors: recursive feature elimination and sequential forward selection.

Implements the two wrapper baselines declared in ``config/default.yaml`` that
had no implementation in ``src/`` (round-2 editor comment; audit item 2).

* :class:`RFERanker` ranks by recursive elimination with an L2 logistic
  regression and keeps the top ``k``. In the comparison ``k`` is chosen by
  :class:`~vlpso_xai.selection.tuned.TunedTopKSelector` on school-grouped inner
  folds, like every filter.
* :class:`SFSSelector` grows a subset greedily, scoring each candidate by
  school-grouped internal cross-validated AUC, and returns the prefix of its
  own forward path with the best internal score. Its subset size is therefore
  chosen by the same kind of internal resampling the swarm methods use.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .base import BaseSelector, fitness_splits
from .embedded import _Ranker


class RFERanker(_Ranker):
    """Recursive feature elimination with an L2 logistic regression."""

    def __init__(self, k: Optional[int] = None, C: float = 1.0, step: int = 1,
                 random_state: int = 42):
        self.k = k
        self.C = C
        self.step = step
        self.random_state = random_state

    def _score(self, values, y):
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression

        rfe = RFE(LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000,
                                     random_state=self.random_state),
                  n_features_to_select=1, step=self.step).fit(values, y)
        return -rfe.ranking_.astype(float)          # rank 1 = kept longest = best


class SFSSelector(BaseSelector):
    """Greedy sequential forward selection with school-grouped internal CV.

    At each step every remaining feature is added in turn and the candidate
    subset is scored by the mean AUC of an L2 logistic regression over
    ``cv_splits`` internal folds (school-grouped when ``groups`` is passed).
    The best addition is kept. The search stops at ``k_max`` features, or
    earlier when the internal score has not improved for ``patience`` steps;
    the returned subset is the best-scoring prefix of the path (ties resolved
    towards the smaller subset).
    """

    def __init__(self, k_max: int = 30, cv_splits: int = 3, patience: int = 5,
                 C: float = 1.0, min_features: int = 2, require_groups: bool = False,
                 random_state: int = 42):
        self.k_max = k_max
        self.cv_splits = cv_splits
        self.patience = patience
        self.C = C
        self.min_features = min_features
        self.require_groups = require_groups
        self.random_state = random_state

    def fit(self, X, y, groups=None, **kwargs):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        from ..data.features import assert_no_leakage

        if isinstance(X, pd.DataFrame):
            assert_no_leakage(X, where="SFSSelector.fit")
        values = np.nan_to_num(self._record_input(X))
        y = np.asarray(y).astype(int)
        splits, self.fitness_splitter_ = fitness_splits(
            y, groups, self.cv_splits, self.random_state,
            require_groups=self.require_groups, where="SFSSelector")
        p = values.shape[1]

        def score(cols: List[int]) -> float:
            s = []
            for tr, va in splits:
                m = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000)
                m.fit(values[np.ix_(tr, cols)], y[tr])
                s.append(roc_auc_score(y[va], m.decision_function(values[np.ix_(va, cols)])))
            return float(np.mean(s))

        chosen: List[int] = []
        path_scores: List[float] = []
        best_so_far, since = -np.inf, 0
        while len(chosen) < min(self.k_max, p):
            cand = [j for j in range(p) if j not in chosen]
            sc = [score(chosen + [j]) for j in cand]
            j_best = int(np.argmax(sc))
            chosen.append(cand[j_best])
            path_scores.append(sc[j_best])
            if sc[j_best] > best_so_far + 1e-6:
                best_so_far, since = sc[j_best], 0
            else:
                since += 1
                if since >= self.patience:
                    break
        k_star = int(np.argmax(path_scores)) + 1     # first maximum = smaller subset
        mask = np.zeros(p, dtype=bool)
        mask[chosen[:k_star]] = True
        self.path_ = [str(self.feature_names_in_[j]) for j in chosen]
        self.path_scores_ = path_scores
        self.k_ = k_star
        self.scores_ = np.zeros(p)
        self.scores_[chosen] = np.linspace(1.0, 0.5, len(chosen))
        self._finalise(mask, min_features=self.min_features)
        return self
