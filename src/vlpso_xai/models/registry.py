"""Model and hyperparameter-grid definitions.

Fixes two defects from the submitted code:

* ``config.py`` defined ``LogisticRegression_L1`` and ``LogisticRegression_L2``
  while ``results_interpretation_and_validation.ipynb`` cell 10 iterated over a
  list containing ``'LogisticRegression'``. That loop cannot have run.
  Names are now defined in exactly one place and validated.
* Class imbalance was handled with global ``RandomUnderSampler`` applied to the
  WHOLE dataset before ``GridSearchCV`` (``training.py:85-87``), which discards
  data and leaks the resampling decision across folds. Imbalance is now handled
  with ``class_weight='balanced'`` inside the pipeline, so it is refitted per
  fold and no data is thrown away. Medium vs. High is 6.9% positive, so this
  matters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def define_models(random_state: int = 42, fast: bool = False) -> Dict[str, Dict[str, Any]]:
    """Return ``{name: {"model": estimator, "params": grid}}``.

    ``fast=True`` trims the grids for quick-mode smoke tests.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    g = (lambda full, quick: quick if fast else full)

    models: Dict[str, Dict[str, Any]] = {
        "LogisticRegression_L2": {
            "model": LogisticRegression(
                penalty="l2", max_iter=2000, random_state=random_state,
                class_weight="balanced",
            ),
            "params": {"clf__C": g([0.01, 0.1, 1, 10], [0.1, 1])},
        },
        "LogisticRegression_L1": {
            "model": LogisticRegression(
                penalty="l1", solver="saga", max_iter=3000,
                random_state=random_state, class_weight="balanced",
            ),
            "params": {"clf__C": g([0.01, 0.1, 1, 10], [0.1, 1])},
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(
                random_state=random_state, class_weight="balanced"
            ),
            "params": {
                "clf__criterion": g(["gini", "entropy"], ["gini"]),
                "clf__max_depth": g([3, 5, 10, None], [5, 10]),
                "clf__min_samples_leaf": g([1, 20, 50], [20]),
            },
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                random_state=random_state, class_weight="balanced", n_jobs=1
            ),
            "params": {
                "clf__n_estimators": g([100, 300], [100]),
                "clf__max_depth": g([5, 10, None], [10]),
                "clf__min_samples_leaf": g([1, 10], [10]),
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=random_state),
            "params": {
                "clf__n_estimators": g([100, 200], [50]),
                "clf__learning_rate": g([0.01, 0.1], [0.1]),
                "clf__max_depth": g([2, 3], [3]),
            },
        },
        "LinearSVC": {
            "model": LinearSVC(
                max_iter=5000, dual="auto", random_state=random_state,
                class_weight="balanced",
            ),
            "params": {"clf__C": g([0.01, 0.1, 1], [0.1])},
        },
        "SVM": {
            "model": SVC(
                kernel="rbf", probability=True, random_state=random_state,
                class_weight="balanced",
            ),
            "params": {"clf__C": g([0.1, 1, 10], [1]),
                       "clf__gamma": g(["scale", "auto"], ["scale"])},
        },
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = {
            "model": XGBClassifier(
                random_state=random_state, eval_metric="logloss",
                tree_method="hist", n_jobs=1,
            ),
            "params": {
                "clf__n_estimators": g([100, 300], [100]),
                "clf__learning_rate": g([0.01, 0.1], [0.1]),
                "clf__max_depth": g([3, 6], [3]),
            },
        }
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = {
            "model": LGBMClassifier(
                random_state=random_state, class_weight="balanced",
                n_jobs=1, verbose=-1,
            ),
            "params": {
                "clf__n_estimators": g([100, 300], [100]),
                "clf__learning_rate": g([0.01, 0.1], [0.1]),
                "clf__num_leaves": g([15, 31], [31]),
            },
        }
    except ImportError:
        pass

    try:
        from sklearn.neural_network import MLPClassifier

        models["MLP"] = {
            "model": MLPClassifier(
                random_state=random_state, max_iter=500, early_stopping=True
            ),
            "params": {
                "clf__hidden_layer_sizes": g([(100,), (50, 50)], [(50,)]),
                "clf__alpha": g([1e-4, 1e-2], [1e-3]),
            },
        }
    except ImportError:
        pass

    return models


def get_models(names: Optional[List[str]] = None, **kwargs) -> Dict[str, Dict[str, Any]]:
    """Select a subset by name, failing loudly on an unknown name."""
    all_models = define_models(**kwargs)
    if names is None:
        return all_models
    unknown = [n for n in names if n not in all_models]
    if unknown:
        raise KeyError(
            f"Unknown model name(s): {unknown}. Available: {sorted(all_models)}. "
            "(The submitted code asked for 'LogisticRegression' while the "
            "registry defined 'LogisticRegression_L1'/'_L2'; that loop could "
            "never have run.)"
        )
    return {n: all_models[n] for n in names}


def supports_predict_proba(name: str) -> bool:
    return name not in {"LinearSVC"}
