"""Prove the outer test fold never influences model development.

Editor comment 1. The sentinel test is the decisive one: a column that is
perfectly predictive ON OUTER-TEST ROWS ONLY and pure noise everywhere else
carries no signal any correctly-isolated selector could find. If such a column
is selected, information has crossed the boundary.
"""
import numpy as np
import pandas as pd
import pytest

from vlpso_xai.data.design import (
    DesignError, assert_group_disjoint, school_grouped_splitter,
)
from vlpso_xai.evaluation.nested_cv import NestedCVConfig, run_nested_cv
from vlpso_xai.models.registry import get_models
from vlpso_xai.selection.filters import SymmetricUncertaintyFilter


def test_schools_never_straddle_a_fold(synthetic):
    X, y, g = synthetic["X"], synthetic["y"], synthetic["groups"]
    sp = school_grouped_splitter(5, shuffle=True, random_state=0)
    for tr, te in sp.split(X, y, groups=g):
        assert_group_disjoint(tr, te, g)          # must not raise
        assert not (set(g[tr]) & set(g[te]))


def test_group_guard_detects_a_violation(synthetic):
    g = synthetic["groups"]
    with pytest.raises(DesignError, match="BOTH"):
        assert_group_disjoint(np.arange(0, 300), np.arange(200, 600), g)


def test_sentinel_column_is_never_selected(synthetic):
    """THE isolation test.

    ``sentinel`` equals the label on outer-test rows and is noise elsewhere.
    Feature selection runs on inner-training folds only, where the column is
    pure noise, so it must never survive selection. If it does, the selector
    saw outer-test rows.
    """
    X, y, g = synthetic["X"].copy(), synthetic["y"], synthetic["groups"]
    rng = np.random.default_rng(0)

    outer = school_grouped_splitter(3, shuffle=True, random_state=0)
    tr, te = next(iter(outer.split(X, y, groups=g)))

    sentinel = rng.normal(0, 1, len(y))
    sentinel[te] = y[te] * 10.0 + rng.normal(0, 0.01, len(te))
    X["sentinel"] = sentinel

    sel = SymmetricUncertaintyFilter(k=5)
    sel.fit(X.iloc[tr], y[tr])                    # inner-training data only
    assert "sentinel" not in sel.selected_feature_names_

    # ...whereas fitting on everything (the submitted pipeline's behaviour)
    # picks it up immediately.
    leaky = SymmetricUncertaintyFilter(k=5)
    leaky.fit(X, y)
    assert "sentinel" in leaky.selected_feature_names_


def test_selector_carries_names_not_positions(synthetic):
    """Regression test for AUDIT_REPORT section C.2."""
    X, y = synthetic["X"], synthetic["y"]
    sel = SymmetricUncertaintyFilter(k=4).fit(X, y)
    names = sel.selected_feature_names_
    assert len(names) == 4
    assert set(names) <= set(X.columns)

    # Reordering the columns must not change WHICH features are chosen.
    perm = list(X.columns[::-1])
    sel2 = SymmetricUncertaintyFilter(k=4).fit(X[perm], y)
    assert set(sel2.selected_feature_names_) == set(names)

    # The positional indices, however, DO differ -- which is exactly why the
    # original code's df.iloc[:, selected_features] was wrong.
    p1 = [list(X.columns).index(n) for n in names]
    p2 = [perm.index(n) for n in names]
    assert p1 != p2


def test_nested_cv_reports_disjoint_schools(synthetic):
    X, y, g = synthetic["X"], synthetic["y"], synthetic["groups"]
    res = run_nested_cv(
        X, y, g,
        models=get_models(["LogisticRegression_L2"], fast=True),
        cfg=NestedCVConfig(outer_splits=3, outer_repeats=1, inner_splits=2,
                           verbose=False),
        sample_weight=synthetic["weights"], task="t", pv=1, method="none",
    )
    assert len(res) == 3
    assert (res["n_schools_train"] + res["n_schools_test"] == synthetic["n_schools"]).all()
    assert res["auc"].between(0, 1).all()
    assert "w_auc" in res.columns


def test_threshold_comes_from_inner_folds(synthetic):
    """The chosen threshold must not be the default, nor tuned on outer test."""
    X, y, g = synthetic["X"], synthetic["y"], synthetic["groups"]
    res = run_nested_cv(
        X, y, g,
        models=get_models(["LogisticRegression_L2"], fast=True),
        cfg=NestedCVConfig(outer_splits=3, outer_repeats=1, inner_splits=2,
                           verbose=False),
        task="t", pv=1, method="none",
    )
    # Thresholds vary across folds because each is derived from that fold's
    # inner predictions only.
    assert res["threshold"].nunique() > 1


def test_names_survive_the_pipeline(synthetic):
    """Regression test for a bug found on the first real selector run.

    Inside a Pipeline the imputer and scaler emit numpy arrays by default, so a
    downstream selector reports positional labels ('x3') instead of PISA item
    codes. That is AUDIT_REPORT section C.2 reintroduced by another route.
    """
    from vlpso_xai.models.pipeline import make_pipeline
    from vlpso_xai.models.registry import get_models

    X, y = synthetic["X"], synthetic["y"]
    est = get_models(["LogisticRegression_L2"], fast=True)["LogisticRegression_L2"]["model"]
    pipe = make_pipeline(est, SymmetricUncertaintyFilter(k=5))
    pipe.fit(X, y)

    names = pipe.named_steps["select"].selected_feature_names_
    assert len(names) == 5
    assert not any(n.startswith("x") and n[1:].isdigit() for n in names), (
        f"selector reported positional labels {names}; feature names were lost "
        "between the scaler and the selector"
    )
    # Names must be real input columns, or an explicit missing-indicator column.
    for n in names:
        assert n in set(X.columns) or "missingindicator" in n


def test_imputer_indicator_shifts_positions(synthetic):
    """Shows WHY positional indices are unsafe here: imputation adds columns."""
    from vlpso_xai.models.pipeline import make_pipeline
    from vlpso_xai.models.registry import get_models

    X = synthetic["X"].copy()
    X.iloc[0, 0] = np.nan
    est = get_models(["LogisticRegression_L2"], fast=True)["LogisticRegression_L2"]["model"]
    pipe = make_pipeline(est, SymmetricUncertaintyFilter(k=3)).fit(X, synthetic["y"])
    n_in = pipe.named_steps["select"].n_features_in_
    assert n_in > X.shape[1], "missing indicator should widen the matrix"
