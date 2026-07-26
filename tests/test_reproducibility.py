"""Reproducibility is a deliverable, not an aspiration.

Also covers the outcome/design modules and the manifest.
"""
import numpy as np
import pandas as pd
import pytest

from vlpso_xai.data.design import (
    SurveyDesign, brr_standard_error, brr_weighted_mean_se, design_diagnostics,
    weighted_mean, weighted_var,
)
from vlpso_xai.data.outcome import (
    DEFAULT_CATEGORY_CUTS, PISA2018_MATH_LEVEL_BOUNDS, build_pv_categories,
    categorize, category_instability, rubin_combine, task_labels,
)
from vlpso_xai.evaluation.metrics import classification_metrics
from vlpso_xai.evaluation.permutation import (
    PermutationSanityError, assert_permutation_null_is_chance,
    decompose_performance, permute_within_groups,
)
from vlpso_xai.evaluation.stability import jaccard, kuncheva, pairwise_stability


# ---------------------------------------------------------------- outcome
def test_official_cut_points():
    assert PISA2018_MATH_LEVEL_BOUNDS["level_3"] == 482.38
    assert PISA2018_MATH_LEVEL_BOUNDS["level_5"] == 606.99
    assert DEFAULT_CATEGORY_CUTS == (482.38, 606.99)


def test_boundary_behaviour_differs_from_the_submitted_rule():
    """Original: s <= 482 -> Low, s <= 607 -> Medium."""
    got = list(categorize([481.9, 482.0, 482.38, 482.4, 606.98, 606.99, 607.0]))
    assert got == ["Low", "Low", "Medium", "Medium", "Medium", "High", "High"]
    # Direction of the correction: the old rule was `s <= 482 -> Low`, so a
    # student scoring 482.2 was classified MEDIUM. The official Level 3 boundary
    # is 482.38, so that student is in fact below Level 3, i.e. LOW.
    assert categorize([482.2])[0] == "Low"
    old_rule = lambda s: "Low" if s <= 482 else ("Medium" if s <= 607 else "High")
    assert old_rule(482.2) == "Medium"
    # ...and symmetrically at the upper boundary.
    assert categorize([606.995])[0] == "High"
    assert old_rule(606.995) == "Medium"


def test_task_labels_exclude_the_third_class():
    cats = pd.Series(["Low", "Medium", "High", "Low"])
    lab = task_labels(cats, "Low", "High")
    assert lab.iloc[0] == 0.0          # Low  -> negative
    assert np.isnan(lab.iloc[1])       # Medium is in neither class
    assert lab.iloc[2] == 1.0          # High -> positive
    assert lab.iloc[3] == 0.0
    assert lab.notna().sum() == 3


def test_rubin_variance_decomposition():
    est = [0.80, 0.82, 0.84, 0.81, 0.83]
    var = [0.0004] * 5
    r = rubin_combine(est, var)
    assert r.estimate == pytest.approx(np.mean(est))
    assert r.within_variance == pytest.approx(0.0004)
    assert r.between_variance == pytest.approx(np.var(est, ddof=1))
    # T = U + (1 + 1/M) B
    assert r.total_variance == pytest.approx(
        0.0004 + (1 + 1 / 5) * np.var(est, ddof=1)
    )
    assert r.total_variance > r.within_variance      # PVs ADD uncertainty
    assert 0 < r.fraction_missing_information < 1
    assert r.ci_low < r.estimate < r.ci_high


def test_rubin_with_identical_pvs_has_no_between_variance():
    r = rubin_combine([0.8] * 10, [0.0004] * 10)
    assert r.between_variance == pytest.approx(0.0)
    assert r.total_variance == pytest.approx(0.0004)


def test_averaging_pvs_understates_uncertainty():
    """The core argument against data_preparation.py:164."""
    rng = np.random.default_rng(0)
    est = rng.normal(0.82, 0.02, 10)
    proper = rubin_combine(est, [0.0004] * 10)
    naive = rubin_combine([est.mean()], [0.0004])
    assert proper.total_variance > naive.total_variance


def test_pv_instability_detected():
    rng = np.random.default_rng(0)
    true = rng.normal(480, 90, 2000)
    df = pd.DataFrame({f"PV{i}MATH": true + rng.normal(0, 30, 2000) for i in range(1, 11)})
    inst = category_instability(build_pv_categories(df))
    assert 0.3 < inst["prop_unstable"] < 0.95
    assert inst["m_plausible_values"] == 10


# ----------------------------------------------------------------- design
def test_brr_matches_the_pisa_divisor():
    """Var = sum (theta_r - theta)^2 / (R (1-k)^2); with R=80, k=0.5 -> /20."""
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"W_FSTUWT": np.ones(n), "CNTSCHID": rng.integers(0, 40, n)})
    for r in range(1, 81):
        df[f"W_FSTURWT{r}"] = 1.0
    d = SurveyDesign.from_frame(df)
    x = rng.normal(0, 1, n)
    out = brr_weighted_mean_se(x, d)
    assert out["brr_se"] == pytest.approx(0.0, abs=1e-9)   # identical replicates
    assert out["fay_k"] == 0.5
    assert out["n_replicates_used"] == 80


def test_weighted_mean_is_exact():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 2.0])
    assert weighted_mean(x, w) == pytest.approx(9 / 4)


def test_design_requires_replicates_by_default():
    from vlpso_xai.data.design import DesignError
    df = pd.DataFrame({"W_FSTUWT": [1.0, 2.0], "CNTSCHID": [1, 2]})
    with pytest.raises(DesignError, match="replicate"):
        SurveyDesign.from_frame(df)
    d = SurveyDesign.from_frame(df, require_replicates=False)
    assert d.n_replicates == 0


def test_kish_effective_n_below_actual_under_unequal_weights():
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({"W_FSTUWT": rng.gamma(2, 100, n), "CNTSCHID": rng.integers(0, 50, n)})
    for r in range(1, 81):
        df[f"W_FSTURWT{r}"] = df["W_FSTUWT"]
    diag = design_diagnostics(SurveyDesign.from_frame(df))
    assert diag["kish_effective_n"] < diag["n_students"]
    assert diag["design_effect_kish"] > 1.0


# ------------------------------------------------------------ permutation
def test_within_group_permutation_preserves_group_composition():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 50)
    y = (rng.random(1000) < 0.3).astype(int)
    yp = permute_within_groups(y, g, rng)
    for grp in np.unique(g):
        assert y[g == grp].sum() == yp[g == grp].sum()


def test_unrestricted_permutation_does_not_preserve_composition():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 50)
    y = np.concatenate([np.ones(50 * 10), np.zeros(50 * 10)]).astype(int)
    yp = permute_within_groups(y, None, rng)
    assert not all(y[g == grp].sum() == yp[g == grp].sum() for grp in np.unique(g))


def test_chance_gate_passes_and_fails_correctly():
    assert_permutation_null_is_chance(0.502)            # must not raise
    with pytest.raises(PermutationSanityError, match="HALT"):
        assert_permutation_null_is_chance(0.62)


def test_chance_gate_refuses_within_group_nulls():
    """The two nulls have different expectations; conflating them is the bug
    that surfaced on the first real run."""
    with pytest.raises(ValueError, match="UNRESTRICTED"):
        assert_permutation_null_is_chance(0.62, within_groups=True)


def test_performance_decomposition_sums_correctly():
    d = decompose_performance(0.8629, 0.6188, 0.5034)
    assert d["between_school_component"] + d["within_school_component"] == pytest.approx(
        d["total_above_chance"]
    )
    assert d["between_school_share"] + d["within_school_share"] == pytest.approx(1.0)


# --------------------------------------------------------------- metrics
def test_false_negative_rate_is_fixed():
    """codes/evaluation.py:97 divided by (fn + tn). Recall + FNR must be 1."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    m = classification_metrics(y, s, 0.5)
    assert m["recall"] + m["false_negative_rate"] == pytest.approx(1.0)
    assert m["specificity"] + m["false_positive_rate"] == pytest.approx(1.0)


# -------------------------------------------------------------- stability
def test_kuncheva_is_zero_at_chance_overlap():
    """k=10 of n=20: expected overlap k^2/n = 5 -> index 0."""
    a = list("abcdefghij")
    b = list("abcde") + list("pqrst")
    assert kuncheva(a, b, 20) == pytest.approx(0.0)
    assert kuncheva(a, a, 20) == pytest.approx(1.0)
    assert jaccard(a, a) == 1.0


def test_kuncheva_corrects_for_chance_where_jaccard_does_not():
    a, b = list("abcdefghij"), list("abcde") + list("pqrst")
    assert jaccard(a, b) > 0.3          # Jaccard looks like agreement
    assert kuncheva(a, b, 20) == pytest.approx(0.0)   # but it is chance


def test_pairwise_stability_reports_sizes():
    out = pairwise_stability([["a", "b"], ["a", "c"], ["a", "b"]], 10)
    assert out["n_subsets"] == 3
    assert out["size_mean"] == 2.0
    assert 0 <= out["jaccard_mean"] <= 1


# --------------------------------------------------------------- config
def test_config_resolves_without_hardcoded_drive_path(tmp_path, monkeypatch):
    """codes/config.py raised unless the path contained a magic substring."""
    from vlpso_xai.config import load_config, resolve_project_root

    monkeypatch.setenv("VLPSO_PROJECT_ROOT", str(tmp_path))
    assert resolve_project_root() == tmp_path.resolve()
    monkeypatch.delenv("VLPSO_PROJECT_ROOT")
    cfg = load_config("default")
    assert cfg.seed == 42
    assert len(cfg.seeds) == cfg.section("reproducibility", "n_seeds")
    assert len(cfg.hash()) == 64


def test_quick_config_is_genuinely_reduced():
    from vlpso_xai.config import load_config

    d, q = load_config("default"), load_config("quick")
    assert q.section("outcome", "n_plausible_values") < d.section("outcome", "n_plausible_values")
    assert q.section("cv", "outer_repeats") < d.section("cv", "outer_repeats")
    assert q.hash() != d.hash()


def test_codebook_refuses_undocumented_variables():
    from pathlib import Path

    from vlpso_xai.data.codebook import Codebook, UndocumentedVariableError

    cb = Codebook.load(Path(__file__).resolve().parents[1] / "config")
    assert "Televisions" in cb.label("ST012Q01TA")
    assert "books" in cb.label("ST013Q01TA").lower()
    cb.require_documented(["ST012Q01TA", "ST013Q01TA"])
    with pytest.raises(UndocumentedVariableError):
        cb.require_documented(["MADE_UP_ITEM"])
