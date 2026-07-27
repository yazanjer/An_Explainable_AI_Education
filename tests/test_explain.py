"""Explainability tests. Editor comments 7 and 8.

SHAP and LIME are optional dependencies; tests that need them skip cleanly.
The tests that matter most -- feature-space alignment, instance selection, the
negative-control benchmark and the absence of causal language -- need neither.
"""
import numpy as np
import pandas as pd
import pytest

from vlpso_xai.explain.consistency import CAUSAL_CAVEAT, rank_agreement
from vlpso_xai.explain.shap_global import (noise_control_benchmark,
                                           stratified_sample,
                                           transformed_frames)
from vlpso_xai.explain.shap_local import select_local_instances
from vlpso_xai.models.pipeline import make_pipeline
from vlpso_xai.models.registry import get_models

shap = pytest.importorskip  # alias for readability below


@pytest.fixture
def fitted(synthetic):
    X, y = synthetic["X"].iloc[:400], synthetic["y"][:400]
    X = X.copy()
    X.iloc[0, 0] = np.nan            # force a missing indicator
    est = get_models(["RandomForest"], fast=True)["RandomForest"]["model"]
    return make_pipeline(est).fit(X, y), X, y


# ---------------------------------------------------------------- alignment
def test_transformed_frames_matches_the_estimator_feature_space(fitted):
    """Audit finding C3: the bare classifier was being handed RAW frames."""
    pipe, X, y = fitted
    Xtr_t, Xte_t = transformed_frames(pipe, X, X)
    clf = pipe.named_steps["clf"]
    assert Xtr_t.shape[1] == clf.n_features_in_
    assert Xtr_t.shape[1] > X.shape[1]          # indicator column was added
    assert list(Xtr_t.columns) == list(Xte_t.columns)
    assert not Xtr_t.isna().any().any()          # imputation applied


def test_global_shap_refuses_the_wrong_feature_space(fitted):
    from vlpso_xai.explain.shap_global import global_shap
    pipe, X, y = fitted
    with pytest.raises(ValueError, match="fitted on"):
        global_shap(pipe.named_steps["clf"], X, X, y)


def test_global_shap_refuses_a_whole_pipeline(fitted):
    from vlpso_xai.explain.shap_global import global_shap
    pipe, X, y = fitted
    Xtr_t, Xte_t = transformed_frames(pipe, X, X)
    with pytest.raises(TypeError, match="bare fitted estimator"):
        global_shap(pipe, Xtr_t, Xte_t, y)


def test_local_shap_applies_the_same_guards(fitted):
    from vlpso_xai.explain.shap_local import local_shap
    pipe, X, y = fitted
    with pytest.raises(TypeError):
        local_shap(pipe, X, X)
    with pytest.raises(ValueError, match="fitted on"):
        local_shap(pipe.named_steps["clf"], X, X)


# ------------------------------------------------------- instance selection
def test_instance_selection_is_documented_and_reproducible():
    rng = np.random.default_rng(0)
    s, y = rng.random(500), (rng.random(500) < 0.3).astype(int)
    a = select_local_instances(s, y, n=10, seed=7)
    b = select_local_instances(s, y, n=10, seed=7)
    assert np.array_equal(a["index"], b["index"])       # reproducible
    assert a["rule"] and a["strategy"] == "stratified_quantile"
    assert len(a["index"]) > 2                          # not just two extremes


def test_extremes_strategy_is_labelled_as_unrepresentative():
    """The submitted analysis used exactly this and generalised from it."""
    rng = np.random.default_rng(0)
    s, y = rng.random(200), (rng.random(200) < 0.3).astype(int)
    r = select_local_instances(s, y, strategy="extremes")
    assert len(r["index"]) == 2
    assert "NOT representative" in r["rule"]


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        select_local_instances(np.zeros(10), np.zeros(10), strategy="vibes")


def test_stratified_sample_covers_rare_classes():
    y = np.array([0] * 950 + [1] * 50)
    X = pd.DataFrame({"a": np.arange(1000)})
    idx = stratified_sample(X, y, 100, seed=0)
    assert y[idx].sum() >= 3, "rare class dropped from the explained sample"


# ------------------------------------------------------- negative control
def test_noise_control_benchmark_reports_features_below_noise():
    imp = pd.DataFrame({
        "feature": ["ST013Q01TA", "ST166Q03HA", "noise_control", "ST012Q09NA"],
        "mean_abs_shap": [0.30, 0.20, 0.05, 0.01],
        "rank": [1, 2, 3, 4],
    })
    out = noise_control_benchmark(imp)
    assert out["available"] and out["noise_rank"] == 3
    assert out["features_below_noise"] == ["ST012Q09NA"]
    assert "not distinguishable" in out["interpretation"]


def test_noise_control_absent_is_reported_not_silently_ignored():
    imp = pd.DataFrame({"feature": ["a"], "mean_abs_shap": [1.0], "rank": [1]})
    out = noise_control_benchmark(imp)
    assert out["available"] is False and "diagnostic" in out["note"]


# ------------------------------------------------------------- consistency
def test_rank_agreement_detects_agreement_and_disagreement():
    feats = [f"f{i}" for i in range(10)]
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.linspace(1, 0.1, 10)})
    agree = pd.DataFrame({"feature": feats, "weight_mean": np.linspace(1, 0.1, 10)})
    disagree = pd.DataFrame({"feature": feats, "weight_mean": np.linspace(0.1, 1, 10)})
    assert rank_agreement(imp, agree)["spearman_rho"] > 0.9
    assert rank_agreement(imp, disagree)["spearman_rho"] < -0.9


# --------------------------------------------------------- causal language
def test_causal_caveat_is_present_and_explicit():
    """Editor comment 8."""
    t = CAUSAL_CAVEAT.lower()
    assert "do not show" in t and "cause" in t
    assert "no time axis" in t
    assert "no causal or interventional claim is supported" in t


@pytest.mark.parametrize(
    "banned",
    ["through time", "path dependence", "cumulative advantage",
     "accounts for", "pedagogically meaningful"],
)
def test_banned_causal_phrases_absent_from_explain_modules(banned):
    """The phrases the editor flagged must not reappear as our own prose."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "vlpso_xai" / "explain"
    for path in root.glob("*.py"):
        text = path.read_text().lower()
        # allowed only where we quote the manuscript to say it was removed
        for line in text.splitlines():
            if banned in line:
                assert any(m in line for m in
                           ("main.tex", "removed", "deleted", "must not",
                            "do not", "does not", "no time axis")), (
                    f"{path.name}: unqualified causal phrasing {banned!r}: {line.strip()[:90]}"
                )
