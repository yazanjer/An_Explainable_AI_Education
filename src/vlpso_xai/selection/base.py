"""Common interface for every feature selector in the comparison.

Answers editor comment 5. All eleven selector configurations must run inside
the identical nested-CV harness so that any difference is attributable to the
selector and not to the surrounding machinery.

Two invariants every selector here must satisfy:

1. **It is a scikit-learn transformer.** Selection happens as a *step inside a
   Pipeline*, so it is fitted on the inner-training fold and nothing else. In
   the submitted code the VLPSO search ran on the full dataset
   (``feature_selection.ipynb`` cells 13-15) and the chosen features were then
   reused for training and testing.
2. **It carries feature NAMES, not positional indices.** The submitted code
   applied positional indices from a twice-reduced matrix back onto the
   original dataframe (``feature_selection.ipynb`` cell 20), so the columns
   saved as "selected" were not the columns the algorithm chose
   (AUDIT_REPORT.md section C.2). ``BaseSelector`` records
   ``feature_names_in_`` at fit time and exposes ``selected_feature_names_``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class BaseSelector(SelectorMixin, BaseEstimator):
    """Name-carrying selector base class."""

    def _record_input(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            values = X.to_numpy(dtype=float)
        else:
            X = np.asarray(X, dtype=float)
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(X.shape[1])], dtype=object
            )
            values = X
        self.n_features_in_ = values.shape[1]
        return values

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.support_

    @property
    def selected_feature_names_(self) -> List[str]:
        """The selected columns, BY NAME. The only sanctioned way to report them."""
        check_is_fitted(self, "support_")
        return [str(n) for n, keep in zip(self.feature_names_in_, self.support_) if keep]

    @property
    def n_selected_(self) -> int:
        check_is_fitted(self, "support_")
        return int(self.support_.sum())

    @property
    def selected_items_(self) -> List[str]:
        """Selected columns EXCLUDING imputation artefacts.

        `SimpleImputer(add_indicator=True)` appends one `missingindicator_*`
        column per feature with missing values, so a 31-item allowlist becomes
        a 61-column matrix. An audit found those pseudo-features were being
        counted and reported as "features selected", inflating every count and
        corrupting the stability indices (which compared subsets drawn from a
        61-column space against a 31-name universe).

        Report `n_items_` and `selected_items_` in the manuscript;
        `n_selected_` remains available for the raw matrix width.
        """
        return [n for n in self.selected_feature_names_
                if not n.startswith("missingindicator_")]

    @property
    def n_items_(self) -> int:
        """Number of substantive items selected, excluding indicators."""
        return len(self.selected_items_)

    def selection_report(self) -> dict:
        """Everything a fold needs to contribute to the stability analysis."""
        check_is_fitted(self, "support_")
        return {
            "selector": type(self).__name__,
            "n_features_in": int(self.n_features_in_),
            "n_selected": self.n_selected_,
            "n_items": self.n_items_,
            "selected": list(self.selected_feature_names_),
            "selected_items": list(self.selected_items_),
        }

    def _finalise(self, mask: np.ndarray, *, min_features: int = 1) -> np.ndarray:
        """Guarantee a non-degenerate support mask."""
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() < min_features:
            logger.warning(
                "%s selected %d features; forcing %d to keep the pipeline "
                "runnable. This fold's result should be treated as a failure "
                "of the selector, not as a legitimate small subset.",
                type(self).__name__, mask.sum(), min_features,
            )
            order = np.argsort(-np.asarray(getattr(self, "scores_", np.zeros(len(mask)))))
            mask = np.zeros(len(mask), dtype=bool)
            mask[order[:min_features]] = True
        self.support_ = mask
        return mask


def fitness_splits(
    y: np.ndarray,
    groups: Optional[np.ndarray],
    n_splits: int,
    random_state: int,
    *,
    require_groups: bool = False,
    where: str = "selector",
):
    """Folds for a wrapper selector's INTERNAL fitness evaluation.

    Audit M7 (and editor/R2 comment, round 2): the swarm selectors scored
    candidate subsets with an UNGROUPED ``StratifiedKFold`` inside the outer
    training fold, so a student's classmates could sit on the other side of an
    internal split. The headline estimates were unaffected (that arm uses no
    selector), but the selector comparison was not school-respecting.

    With ``groups`` supplied the folds come from ``StratifiedGroupKFold`` and no
    school straddles an internal boundary; this is asserted, not assumed. With
    ``require_groups=True`` a missing ``groups`` argument is an error rather
    than a silent fall-back to the ungrouped splitter -- the comparison runs
    set it, so the defect cannot return unnoticed.

    Returns ``(splits, splitter_name)``; the name is recorded on the selector so
    every result row states which protocol produced it.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    y = np.asarray(y).astype(int)
    if groups is None:
        if require_groups:
            raise ValueError(
                f"{where}: groups are required for school-grouped wrapper "
                "fitness (audit M7) but were not passed to fit(). Pass "
                "select__groups=<school ids> through the Pipeline."
            )
        k = int(max(2, min(n_splits, np.bincount(y).min())))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        return list(skf.split(np.zeros(len(y)), y)), "StratifiedKFold(ungrouped)"

    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError(f"{where}: groups has length {len(groups)}, y has {len(y)}")
    k = int(max(2, min(n_splits, len(np.unique(groups)))))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state)
    splits = list(sgkf.split(np.zeros(len(y)), y, groups=groups))
    for tr, va in splits:
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError(f"{where}: a school straddles a fitness fold")
    return splits, "StratifiedGroupKFold(CNTSCHID)"


class IdentitySelector(BaseSelector):
    """The 'no selection' baseline: keeps every feature.

    Present so that "full feature set" runs through exactly the same harness
    as every other condition rather than bypassing it.
    """

    def fit(self, X, y=None, **kwargs) -> "IdentitySelector":
        values = self._record_input(X)
        self.scores_ = np.ones(values.shape[1])
        self._finalise(np.ones(values.shape[1], dtype=bool))
        return self
