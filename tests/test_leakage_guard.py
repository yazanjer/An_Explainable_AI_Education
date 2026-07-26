"""The guard must RAISE, not warn. Editor comments 1 and 2."""
import numpy as np
import pandas as pd
import pytest

from vlpso_xai.data.features import (
    Allowlist, LeakageError, assert_no_leakage, build_design_matrix,
    classify_columns, forbidden_columns, single_variable_auc_screen,
)


@pytest.mark.parametrize(
    "col",
    [
        "math_score",        # the outcome itself
        "math_category",     # the 3-level label
        "label",
        "PV1MATH", "PV3MATH", "PV10MATH",
        "PV1READ", "PV7SCIE",
        "W_FSTUWT", "W_FSTURWT1", "W_FSTURWT80",
        "SENWT",
        "STRATUM", "STRATUM_ESP9033",
        "CNTSCHID", "CNTSTUID", "CNT", "CYC", "NatCen", "SUBNATIO", "OECD",
        "noise_control",
    ],
)
def test_forbidden_column_raises(col):
    """Explicitly covers the audit's named columns."""
    df = pd.DataFrame({"ST013Q01TA": [1.0, 2.0], col: [0.0, 1.0]})
    with pytest.raises(LeakageError):
        assert_no_leakage(df, where="test")


def test_clean_matrix_passes():
    df = pd.DataFrame({"ST013Q01TA": [1.0, 2.0], "ST012Q01TA": [3.0, 4.0]})
    assert_no_leakage(df, where="test")  # must not raise


def test_error_message_names_the_column_and_reason():
    df = pd.DataFrame({"ST013Q01TA": [1.0], "math_score": [500.0]})
    with pytest.raises(LeakageError) as e:
        assert_no_leakage(df, where="stage-x")
    msg = str(e.value)
    assert "math_score" in msg and "stage-x" in msg
    assert "data_preparation.py:253" in msg


def test_numpy_input_without_names_is_refused():
    """Positional matrices hide exactly the bug in AUDIT_REPORT section C.2."""
    with pytest.raises(ValueError, match="feature_names"):
        assert_no_leakage(np.zeros((3, 2)), where="test")


def test_select_dtypes_object_would_not_have_caught_it():
    """Reproduces the original defect: math_score is float64 and survives."""
    df = pd.DataFrame({
        "CNT": ["ESP"], "math_category": ["Low"],
        "math_score": [451.2], "PV1MATH": [449.8], "ST013Q01TA": [3.0],
    })
    survivors = df.drop(columns=list(df.select_dtypes(include=["object"]).columns))
    assert "math_score" in survivors.columns   # the original bug
    assert "PV1MATH" in survivors.columns
    with pytest.raises(LeakageError):
        assert_no_leakage(survivors, where="original-pipeline")


def test_allowlist_has_justification_for_every_entry(allowlist):
    assert len(allowlist.names) > 0
    for name, entry in allowlist.entries.items():
        assert entry.justification.strip(), f"{name} has no justification"
        assert entry.label.strip(), f"{name} has no codebook label"


def test_allowlist_contains_nothing_forbidden(allowlist):
    assert forbidden_columns(allowlist.names) == []


def test_build_design_matrix_drops_leaks(allowlist, synthetic):
    df = synthetic["X"].copy()
    df["math_score"] = 500.0
    df["PV1MATH"] = 501.0
    df["W_FSTUWT"] = 12.0
    X = build_design_matrix(df, allowlist, where="test")
    assert "math_score" not in X.columns
    assert set(X.columns) <= set(allowlist.names)


def test_build_design_matrix_fails_loudly_on_missing_predictor(allowlist, synthetic):
    df = synthetic["X"].drop(columns=[allowlist.names[0]])
    with pytest.raises(KeyError):
        build_design_matrix(df, allowlist, strict=True, where="test")


def test_auc_screen_flags_the_outcome(allowlist, synthetic):
    """A leaking column must trip the empirical screen."""
    X = synthetic["X"].copy()
    y = synthetic["y"]
    X["sneaky_outcome_copy"] = y.astype(float) + np.random.default_rng(0).normal(0, 0.01, len(y))
    with pytest.raises(LeakageError, match="0.95"):
        single_variable_auc_screen(X, y, allowlist=allowlist, raise_on_flag=True)


def test_auc_screen_passes_on_clean_features(allowlist, synthetic):
    out = single_variable_auc_screen(
        synthetic["X"], synthetic["y"], allowlist=allowlist, raise_on_flag=True
    )
    assert not out["flagged"].any()
    assert out["auc"].max() < 0.95


def test_auc_screen_is_direction_free():
    """A perfectly INVERTED predictor leaks just as much as an aligned one."""
    y = np.array([0] * 200 + [1] * 200)
    X = pd.DataFrame({"inverted": 1.0 - y})
    out = single_variable_auc_screen(X, y, raise_on_flag=False)
    assert out.loc[0, "auc"] > 0.99
    assert out.loc[0, "auc_signed"] < 0.01


def test_classify_columns_covers_every_category():
    cols = ["math_score", "PV1MATH", "PV2READ", "W_FSTURWT12", "STRATUM",
            "CNTSCHID", "noise_control", "ST013Q01TA"]
    out = classify_columns(cols)
    assert len(out) == len(cols)
    assert set(out["classification"]) == {
        "outcome", "outcome-derived", "design/weight", "identifier",
        "negative-control", "candidate predictor",
    }
    assert out["justification"].str.len().gt(0).all()


# ---------------------------------------------------------------------------
# Adversarial names. Every one of these slipped through the first version of
# the denylist, which was case-sensitive, whitespace-intolerant, and matched
# only canonical spellings. Found by an independent audit of this repository.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "col",
    [
        # case variants
        "pv1math", "PV1math", "Pv1Math", "MATH_SCORE", "math_Score",
        # whitespace
        " math_score", "math_score ", "math_score\n", "PV1MATH ",
        # derived / transformed forms -- a standardised outcome is the outcome
        "math_score_z", "math_score_mean", "math_score_rank", "log_math_score",
        "PV1MATH_imputed",
        # imputation artefacts
        "missingindicator_math_score", "missingindicator_PV1MATH",
        # one-hot / binned encodings
        "PV1MATH_1.0", "math_category_Low", "math_category_High",
        # OECD composites built from overlapping material; ESCS additionally
        # enters the conditioning model that generates the plausible values
        "ESCS", "HISEI", "PARED", "MISCED", "FISCED", "HOMEPOS", "CULTPOSS",
        # identifier case variant
        "cntschid",
    ],
)
def test_adversarial_column_names_are_caught(col):
    df = pd.DataFrame({"ST013Q01TA": [1.0, 2.0], col: [0.0, 1.0]})
    with pytest.raises(LeakageError):
        assert_no_leakage(df, where="adversarial")


@pytest.mark.parametrize(
    "col",
    [
        "ST013Q01TA", "ST166Q03HA", "ST011Q04TA", "ST004D01T", "ST012Q01TA",
        "ST019AQ01T", "missingindicator_ST013Q01TA", "ST004D01T_1.0",
    ],
)
def test_legitimate_columns_are_not_false_positives(col):
    """A denylist that rejects real predictors is as useless as one that lets
    the outcome through."""
    df = pd.DataFrame({col: [1.0, 2.0]})
    assert_no_leakage(df, where="false-positive check")


def test_derived_form_message_explains_itself():
    df = pd.DataFrame({"ST013Q01TA": [1.0], "math_score_z": [0.5]})
    with pytest.raises(LeakageError) as e:
        assert_no_leakage(df, where="x")
    assert "derived form" in str(e.value)
