"""Pipeline factory. Every fitted step lives inside a Pipeline, by construction.

Answers editor comment 1 ("Please specify whether imputation, scaling,
encoding, feature ranking, VLPSO feature selection, hyperparameter tuning,
model selection, and threshold selection were performed independently within
each training fold").

The submitted code fitted `StandardScaler` and `OneHotEncoder` on the entire
dataset before any split (``data_preparation.py:98-100``), imputed with global
column means (``data_preparation.py:253``), ran feature ranking and the VLPSO
search on all ~26k rows (``feature_selection.ipynb`` cells 13-15), and never
constructed a ``Pipeline`` object anywhere.

Here the isolation is STRUCTURAL rather than a matter of discipline: because
imputation, scaling and selection are Pipeline *steps*, scikit-learn's own
``fit``/``transform`` contract guarantees they see only the training fold.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def make_pipeline(
    estimator,
    selector=None,
    *,
    impute_strategy: str = "median",
    scale: bool = True,
    add_missing_indicator: bool = True,
):
    """impute -> scale -> select -> classify, as a single fitted unit.

    Parameters
    ----------
    add_missing_indicator:
        PISA non-response is informative: a student who skipped the household
        possessions block differs systematically from one who answered. The
        submitted code used ``.fillna(0)``, which places non-response at 0 --
        BELOW the legitimate minimum of 1 on every ordinal item, silently
        creating an out-of-range category confounded with non-response. Median
        imputation plus an explicit indicator keeps the information without
        inventing a scale point.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list = [
        (
            "impute",
            SimpleImputer(
                strategy=impute_strategy,
                add_indicator=add_missing_indicator,
                keep_empty_features=True,
            ),
        )
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("clf", estimator))
    pipe = Pipeline(steps)

    # ------------------------------------------------------------------
    # CRITICAL: force pandas output through every transformer.
    #
    # By default SimpleImputer and StandardScaler emit bare numpy arrays, so a
    # selector placed after them receives a nameless matrix and can only report
    # positional labels ("x3", "x17"). That is precisely the defect the audit
    # recorded at AUDIT_REPORT.md section C.2 -- positional indices that no
    # longer correspond to the columns anyone thinks they do -- reintroduced
    # through a different mechanism.
    #
    # It is worse here than in the original code, because add_indicator=True
    # APPENDS missing-indicator columns, so position i after imputation is not
    # column i of the input.
    #
    # set_output(transform="pandas") keeps a DataFrame flowing through the whole
    # pipeline, so selected_feature_names_ returns real PISA item codes.
    # tests/test_nested_cv_isolation.py::test_names_survive_the_pipeline locks
    # this in.
    # ------------------------------------------------------------------
    pipe.set_output(transform="pandas")
    return pipe


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str = "balanced_accuracy",
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Choose a decision threshold. Called on INNER-fold predictions only.

    The chosen value is then applied unchanged to the outer test fold. The
    submitted code never selected a threshold at all -- it used the default
    0.5 while reporting balanced accuracy on data with a 6.9% positive class.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    fn = {"balanced_accuracy": balanced_accuracy_score, "f1": f1_score}[metric]
    grid = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
    best_t, best_v = 0.5, -np.inf
    for t in grid:
        v = fn(y_true, (y_score >= t).astype(int), sample_weight=sample_weight)
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t


def score_matrix(estimator, X) -> np.ndarray:
    """Continuous scores, whatever the estimator exposes."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    raise AttributeError(
        f"{type(estimator).__name__} exposes neither predict_proba nor "
        "decision_function; AUC cannot be computed."
    )
