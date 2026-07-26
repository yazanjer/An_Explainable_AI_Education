"""Effect sizes must reproduce analytically known values. Editor comment 6."""
import numpy as np
import pandas as pd
import pytest

from vlpso_xai.evaluation.effect_size import (
    _adjust, cohens_d_independent, cohens_d_paired, contrast_table, hedges_g,
    interpret_d, nadeau_bengio_t, paired_method_contrast,
)


def test_cohens_d_exact_small_sample():
    """Hand-computable: means 3 and 5, both s^2 = 2.5, so d = -2/sqrt(2.5)."""
    d = cohens_d_independent([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
    assert d == pytest.approx(-2 / np.sqrt(2.5), rel=1e-12)
    assert d == pytest.approx(-1.2649110640673518, rel=1e-12)


@pytest.mark.parametrize("mu,sd", [(1.0, 1.0), (2.0, 2.0), (0.5, 1.0), (0.0, 1.0)])
def test_cohens_d_recovers_population_value(mu, sd):
    rng = np.random.default_rng(7)
    a = rng.normal(mu, sd, 300_000)
    b = rng.normal(0.0, sd, 300_000)
    assert cohens_d_independent(a, b) == pytest.approx(mu / sd, abs=0.01)


def test_zero_difference_gives_zero():
    x = [1, 2, 3, 4, 5]
    assert cohens_d_independent(x, x) == pytest.approx(0.0)


def test_hedges_g_is_smaller_in_magnitude_than_d():
    a, b = [1, 2, 3, 4, 5], [3, 4, 5, 6, 7]
    assert abs(hedges_g(a, b)) < abs(cohens_d_independent(a, b))


def test_paired_d():
    diff = np.array([0.1, 0.2, 0.15, 0.05, 0.1])
    assert cohens_d_paired(diff) == pytest.approx(diff.mean() / diff.std(ddof=1))


def test_the_submitted_values_are_negligible_not_large():
    """main.tex:651 calls 0.13-0.18 'large effects'. They are not."""
    for v in (0.05, 0.13, 0.1345, 0.18, 0.1800):
        assert interpret_d(v) == "negligible"
    assert interpret_d(0.3) == "small"
    assert interpret_d(0.6) == "medium"
    assert interpret_d(0.9) == "large"
    assert interpret_d(-0.9) == "large"   # sign-independent


def test_nadeau_bengio_is_more_conservative_than_naive():
    """The correction must widen the interval and shrink |t|."""
    from scipy import stats

    rng = np.random.default_rng(3)
    d = rng.normal(0.01, 0.01, 25)
    nb = nadeau_bengio_t(d, n_train=8000, n_test=2000)
    t_naive, p_naive = stats.ttest_1samp(d, 0.0)
    assert abs(nb["t"]) < abs(t_naive)
    assert nb["p"] > p_naive
    assert nb["ratio"] == pytest.approx(0.25)


def test_holm_and_bh_adjustments():
    p = np.array([0.01, 0.02, 0.03, 0.04])
    holm = _adjust(p, "holm")
    assert np.all(holm >= p)
    assert np.all(np.diff(holm) >= -1e-12)     # monotone
    bh = _adjust(p, "benjamini_hochberg")
    assert np.all(bh >= p)
    assert np.all(bh <= holm + 1e-12)          # BH never exceeds Holm


def _fold_frame(delta: float, n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n):
        base = rng.normal(0.80, 0.02)
        rows.append(dict(task="t", pv=1, repeat=0, outer_fold=f,
                         method="vlpso", auc=base + delta))
        rows.append(dict(task="t", pv=1, repeat=0, outer_fold=f,
                         method="bpso", auc=base))
    return pd.DataFrame(rows)


def test_paired_contrast_detects_a_real_difference():
    c = paired_method_contrast(_fold_frame(0.05), "vlpso", "bpso",
                               n_train=8000, n_test=2000)
    assert c.mean_difference == pytest.approx(0.05, abs=1e-9)
    assert c.n_folds == 20
    assert "Nadeau-Bengio" in c.test


def test_paired_contrast_calls_a_null_difference_negligible():
    c = paired_method_contrast(_fold_frame(0.0), "vlpso", "bpso",
                               n_train=8000, n_test=2000)
    assert c.magnitude in {"negligible", "undefined"}
    assert c.p_value > 0.05 or not np.isfinite(c.p_value)


def test_nadeau_bengio_requires_sizes():
    with pytest.raises(ValueError, match="n_train"):
        paired_method_contrast(_fold_frame(0.01), "vlpso", "bpso")


def _multi_method_frame(n: int = 20, seed: int = 1) -> pd.DataFrame:
    """One row per (fold, method). Each fold shares a common base level so the
    methods are genuinely paired, as they are across real outer folds."""
    rng = np.random.default_rng(seed)
    deltas = {"vlpso": 0.030, "bpso": 0.000, "rfe": 0.012, "chi2": -0.004}
    rows = []
    for f in range(n):
        base = rng.normal(0.80, 0.02)
        for method, d in deltas.items():
            rows.append(dict(task="t", pv=1, repeat=0, outer_fold=f,
                             method=method, auc=base + d + rng.normal(0, 0.002)))
    return pd.DataFrame(rows)


def test_contrast_table_applies_multiplicity_correction():
    out = contrast_table(_multi_method_frame(), reference="vlpso",
                         n_train=8000, n_test=2000)
    assert not out.empty
    assert set(out["method_b"]) == {"bpso", "rfe", "chi2"}
    assert (out["p_adjusted"] >= out["p_value"] - 1e-12).all()
    assert out["p_adjusted"].notna().all()
    assert set(out["correction"]) == {"holm"}
    # vlpso beats bpso by ~0.03 and chi2 by ~0.034; both should survive Holm.
    assert out.loc[out.method_b == "bpso", "mean_difference"].iloc[0] > 0


def test_contrast_table_propagates_undefined_p_as_nan():
    """A degenerate comparison must yield NaN, not a fabricated p-value."""
    df = _fold_frame(0.0)
    df.loc[df.method == "vlpso", "auc"] = df.loc[df.method == "bpso", "auc"].to_numpy()
    out = contrast_table(df, reference="vlpso", n_train=8000, n_test=2000)
    assert out["p_adjusted"].isna().all()
