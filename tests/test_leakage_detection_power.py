"""Meta-tests: does the test suite actually DETECT leakage?

WHY THIS FILE EXISTS
--------------------
An independent audit of this repository mutated `nested_cv.py` to introduce
three catastrophic leaks -- threshold tuned on the outer test fold, GridSearchCV
fitted on the full dataset, and model selection by outer-test AUC -- and **all
104 tests passed**. The isolation tests were proxies that could not fail:

* `test_sentinel_column_is_never_selected` hand-fitted a selector on train rows
  and asserted it did not see test rows. A tautology; it never called
  `run_nested_cv`.
* `test_threshold_comes_from_inner_folds` asserted only that thresholds vary
  across folds. Thresholds tuned ON the outer test fold also vary.
* `test_nested_cv_reports_disjoint_schools` asserted a property true by
  construction of any `GroupKFold`.

That is precisely the failure that shipped the original paper: a guarantee
asserted in a docstring and "tested" by something that cannot detect its
violation. These tests exercise the harness END TO END and are calibrated
against a known-leaky reference implementation, so they have demonstrated
power to detect the thing they claim to detect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV

from vlpso_xai.data.design import school_grouped_splitter
from vlpso_xai.evaluation.nested_cv import NestedCVConfig, run_nested_cv
from vlpso_xai.models.pipeline import make_pipeline, score_matrix, select_threshold
from vlpso_xai.models.registry import get_models
from vlpso_xai.selection.filters import SymmetricUncertaintyFilter


@pytest.fixture
def noise_data():
    """Labels are PURE NOISE. Any honest pipeline must score ~0.50."""
    rng = np.random.default_rng(11)
    n_schools, per = 60, 20
    n = n_schools * per
    X = pd.DataFrame(rng.normal(size=(n, 12)),
                     columns=[f"ST0{i:02d}Q01TA" for i in range(12)])
    y = rng.integers(0, 2, n)
    groups = np.repeat(np.arange(n_schools), per)
    return X, y, groups


# ---------------------------------------------------------------------------
# 1. End-to-end null: the harness on noise must return chance
# ---------------------------------------------------------------------------
def test_harness_returns_chance_on_pure_noise(noise_data):
    """THE test. Runs the real harness; no proxy."""
    X, y, g = noise_data
    res = run_nested_cv(
        X, y, g,
        models=get_models(["LogisticRegression_L2"], fast=True),
        selector_factory=lambda: SymmetricUncertaintyFilter(k=4),
        cfg=NestedCVConfig(outer_splits=4, outer_repeats=1, inner_splits=3,
                           verbose=False),
        task="null", pv=1, method="su",
    )
    assert len(res) == 4
    assert abs(res["auc"].mean() - 0.5) < 0.08, (
        f"pipeline scored {res['auc'].mean():.4f} on random labels; "
        "information is reaching the model"
    )


# ---------------------------------------------------------------------------
# 2. Calibration: a KNOWN-LEAKY implementation must be caught
# ---------------------------------------------------------------------------
def _leaky_threshold_auc(X, y, groups, seed=0):
    """Reference implementation with threshold tuned on the OUTER TEST fold."""
    spec = get_models(["LogisticRegression_L2"], fast=True)["LogisticRegression_L2"]
    outer = school_grouped_splitter(4, shuffle=True, random_state=seed)
    accs = []
    for tr, te in outer.split(X, y, groups=groups):
        inner = list(school_grouped_splitter(3, shuffle=True, random_state=seed)
                     .split(X.iloc[tr], y[tr], groups=groups[tr]))
        gs = GridSearchCV(make_pipeline(spec["model"]), spec["params"],
                          scoring="roc_auc", cv=inner).fit(X.iloc[tr], y[tr])
        s_te = score_matrix(gs.best_estimator_, X.iloc[te])
        thr = select_threshold(y[te], s_te)            # <-- THE LEAK
        from vlpso_xai.evaluation.metrics import classification_metrics
        accs.append(classification_metrics(y[te], s_te, thr)["balanced_accuracy"])
    return float(np.mean(accs))


def test_outer_test_threshold_tuning_is_detectable(noise_data):
    """Calibrates our detector: a leaky threshold beats chance on NOISE."""
    X, y, g = noise_data
    leaky = _leaky_threshold_auc(X, y, g)
    assert leaky > 0.53, (
        "the leaky reference did not beat chance, so this test has no power "
        "to distinguish leaky from clean and must be redesigned"
    )


def test_our_threshold_does_not_beat_chance_on_noise(noise_data):
    """...and the real harness does NOT show that inflation."""
    X, y, g = noise_data
    res = run_nested_cv(
        X, y, g,
        models=get_models(["LogisticRegression_L2"], fast=True),
        cfg=NestedCVConfig(outer_splits=4, outer_repeats=1, inner_splits=3,
                           verbose=False),
        task="null", pv=1, method="none",
    )
    clean = res["balanced_accuracy"].mean()
    leaky = _leaky_threshold_auc(X, y, g)
    assert clean < leaky, (
        f"clean balanced accuracy {clean:.4f} is not below the known-leaky "
        f"reference {leaky:.4f}; the harness may be tuning on the test fold"
    )
    assert clean < 0.56, f"clean balanced accuracy {clean:.4f} exceeds chance on noise"


# ---------------------------------------------------------------------------
# 3. Sentinel THROUGH the harness, not beside it
# ---------------------------------------------------------------------------
def test_sentinel_through_the_full_harness():
    """A column predictive only on ONE fold's test rows must not be exploited.

    Construction matters. An earlier draft made the sentinel predictive on
    EVERY fold's test rows -- which, since each row is a test row exactly once,
    makes it predictive everywhere, including in training. It scored AUC 1.000
    legitimately. The sentinel must be confined to a SINGLE held-out fold, so
    that when that fold is the test set the training data contains only noise.
    """
    rng = np.random.default_rng(5)
    n_schools, per = 60, 20
    n = n_schools * per
    X = pd.DataFrame(rng.normal(size=(n, 10)),
                     columns=[f"ST0{i:02d}Q01TA" for i in range(10)])
    y = rng.integers(0, 2, n)
    groups = np.repeat(np.arange(n_schools), per)

    # The split seed MUST match the one run_nested_cv uses internally
    # (NestedCVConfig.random_state + repeat). An earlier draft used seed 0 here
    # while the harness used 42, so the "test-only" rows landed in TRAINING
    # folds and the sentinel carried genuine signal -- the test failed for a
    # reason unrelated to leakage.
    SEED = 42
    outer = school_grouped_splitter(4, shuffle=True, random_state=SEED)
    tr0, te0 = next(iter(outer.split(X, y, groups=groups)))

    sentinel = rng.normal(size=n)
    sentinel[te0] = y[te0] * 8.0 + rng.normal(0, 0.01, len(te0))
    X["ST099Q01TA"] = sentinel          # signal ONLY on fold 0's test rows

    res = run_nested_cv(
        X, y, groups,
        models=get_models(["LogisticRegression_L2"], fast=True),
        selector_factory=lambda: SymmetricUncertaintyFilter(k=3),
        cfg=NestedCVConfig(outer_splits=4, outer_repeats=1, inner_splits=3,
                           random_state=SEED, verbose=False),
        task="sentinel", pv=1, method="su",
    )
    fold0 = res.iloc[0]
    assert "ST099Q01TA" not in fold0["selected"], (
        "the sentinel was selected on the fold where it is signal only in the "
        "TEST rows; the selector saw outer-test data"
    )
    assert abs(fold0["auc"] - 0.5) < 0.12, (
        f"fold-0 AUC {fold0['auc']:.4f} on noise labels; the pipeline is "
        "exploiting information visible only in the outer test fold"
    )


def test_sentinel_IS_exploited_without_isolation():
    """Calibration: the same sentinel DOES inflate a leaky fit. Proves power."""
    rng = np.random.default_rng(5)
    n_schools, per = 60, 20
    n = n_schools * per
    X = pd.DataFrame(rng.normal(size=(n, 10)),
                     columns=[f"ST0{i:02d}Q01TA" for i in range(10)])
    y = rng.integers(0, 2, n)
    groups = np.repeat(np.arange(n_schools), per)
    outer = school_grouped_splitter(4, shuffle=True, random_state=42)
    tr0, te0 = next(iter(outer.split(X, y, groups=groups)))
    sentinel = rng.normal(size=n)
    sentinel[te0] = y[te0] * 8.0 + rng.normal(0, 0.01, len(te0))
    X["ST099Q01TA"] = sentinel

    # Leaky: selector fitted on the FULL data, as the submitted pipeline did.
    leaky = SymmetricUncertaintyFilter(k=3).fit(X, y)
    assert "ST099Q01TA" in leaky.selected_feature_names_, (
        "the sentinel is not even detectable by a leaky fit, so this test "
        "has no power and must be redesigned"
    )


# ---------------------------------------------------------------------------
# 4. Model selection must not use outer-test performance
# ---------------------------------------------------------------------------
def test_selected_model_tracks_inner_not_outer_score(noise_data):
    """If selection used outer-test AUC, the winner would be the outer argmax."""
    X, y, g = noise_data
    models = get_models(["LogisticRegression_L2", "DecisionTree"], fast=True)
    res = run_nested_cv(
        X, y, g, models=models,
        cfg=NestedCVConfig(outer_splits=4, outer_repeats=1, inner_splits=3,
                           verbose=False),
        task="null", pv=1, method="none",
    )
    import json
    for _, row in res.iterrows():
        inner = json.loads(row["inner_scores_all_models"])
        best_by_inner = max(inner, key=inner.get)
        assert row["best_model"] == best_by_inner, (
            "the reported model is not the inner-loop argmax; selection is "
            "using information other than the inner folds"
        )
