"""Choose ``k`` for a ranking selector by school-grouped inner cross-validation.

Round-2 editor comment: "the filter methods used a fixed value of k rather than
an inner-loop selection procedure. The comparison should be redesigned so that
competing methods receive comparable tuning opportunities".

``config/default.yaml`` always declared ``k_filter: [5, 10, 15, 20, 30]  #
tuned on inner folds``; the reduced-budget comparison ran every filter at a
fixed k = 15. :class:`TunedTopKSelector` implements the declared procedure:

1. split the training data it is given into school-grouped inner folds;
2. in each inner fold, refit the ranker on the inner-training part ONLY (so
   the ranking that decides k never sees the inner-validation rows);
3. for every k in the grid, fit the scoring learner on the top-k columns and
   score the inner-validation rows by AUC;
4. choose the k with the highest mean inner AUC (ties -> smaller k);
5. refit the ranker on all the training data and keep its top k.

Nothing here sees the outer test fold: the selector is a step fitted on the
outer-training partition only.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from .base import BaseSelector, fitness_splits


class TunedTopKSelector(BaseSelector):
    def __init__(self, ranker=None, k_grid: Sequence[int] = (5, 10, 15, 20, 30),
                 inner_splits: int = 5, C: float = 1.0, require_groups: bool = False,
                 random_state: int = 42):
        self.ranker = ranker
        self.k_grid = k_grid
        self.inner_splits = inner_splits
        self.C = C
        self.require_groups = require_groups
        self.random_state = random_state

    def fit(self, X, y, groups=None, **kwargs):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        if self.ranker is None:
            raise ValueError("TunedTopKSelector needs a ranker")
        values = self._record_input(X)
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(values, columns=self.feature_names_in_)
        y = np.asarray(y).astype(int)
        p = values.shape[1]
        grid = sorted({int(min(k, p)) for k in self.k_grid})
        splits, self.inner_splitter_ = fitness_splits(
            y, groups, self.inner_splits, self.random_state,
            require_groups=self.require_groups, where="TunedTopKSelector")
        vals = np.nan_to_num(values)
        cv = np.zeros((len(splits), len(grid)))
        for i, (tr, va) in enumerate(splits):
            g_tr = None if groups is None else np.asarray(groups)[tr]
            r = clone(self.ranker).fit(Xdf.iloc[tr], y[tr], groups=g_tr)
            order = np.argsort(-np.asarray(r.scores_), kind="stable")
            for j, k in enumerate(grid):
                cols = order[:k]
                m = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000)
                m.fit(vals[np.ix_(tr, cols)], y[tr])
                cv[i, j] = roc_auc_score(y[va], m.decision_function(vals[np.ix_(va, cols)]))
        mean = cv.mean(axis=0)
        self.k_grid_ = grid
        self.cv_auc_by_k_ = mean.tolist()
        self.k_ = int(grid[int(np.argmax(mean))])         # first max = smaller k
        self.ranker_ = clone(self.ranker).fit(Xdf, y, groups=groups)
        self.scores_ = np.asarray(self.ranker_.scores_, dtype=float)
        mask = np.zeros(p, dtype=bool)
        mask[np.argsort(-self.scores_, kind="stable")[: self.k_]] = True
        self._finalise(mask)
        return self
