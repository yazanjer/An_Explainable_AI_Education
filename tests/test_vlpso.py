"""VLPSO must actually be variable-length. Editor comment 9.

The submitted implementation used fixed-length binary vectors whose cardinality
cap only ever shrank. These tests would all have failed against it.
"""
import numpy as np
import pandas as pd
import pytest

from vlpso_xai.data.features import LeakageError
from vlpso_xai.selection.bpso import BPSOSelector
from vlpso_xai.selection.filters import (
    SymmetricUncertaintyFilter, entropy, su_matrix, su_scores,
    symmetric_uncertainty,
)
from vlpso_xai.selection.vlpso import VLPSOSelector


# --------------------------------------------------------------------------
# Symmetric uncertainty
# --------------------------------------------------------------------------
def test_su_of_a_variable_with_itself_is_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    assert symmetric_uncertainty(y.astype(float), y) == pytest.approx(1.0, abs=1e-9)


def test_su_of_noise_is_near_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    assert symmetric_uncertainty(rng.normal(size=5000), y) < 0.02


def test_su_is_bounded_and_does_not_inflate_with_cardinality():
    """The candidate set mixes 2-level and 6-level items; raw MI would favour
    the high-cardinality ones purely for being high-cardinality."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 6000)
    binary_noise = rng.integers(0, 2, 6000).astype(float)
    many_noise = rng.integers(0, 50, 6000).astype(float)
    su_b = symmetric_uncertainty(binary_noise, y)
    su_m = symmetric_uncertainty(many_noise, y)
    assert 0.0 <= su_b <= 1.0 and 0.0 <= su_m <= 1.0
    assert abs(su_m - su_b) < 0.02


def test_entropy_of_fair_coin_is_ln2():
    y = np.array([0, 1] * 5000)
    assert entropy(y) == pytest.approx(np.log(2), abs=1e-6)


def test_su_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 4, (400, 6)).astype(float)
    M = su_matrix(X)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 1.0)
    assert ((M >= 0) & (M <= 1)).all()


# --------------------------------------------------------------------------
# VLPSO: the variable-length claim
# --------------------------------------------------------------------------
@pytest.fixture
def small(synthetic):
    X = synthetic["X"].iloc[:600, :12]
    return X, synthetic["y"][:600]


def test_particle_lengths_actually_vary(small):
    """The core claim. Fixed-length particles would give a single value."""
    X, y = small
    sel = VLPSOSelector(population_size=12, max_iter=10, divisions=4,
                        beta_stagnation=2, random_state=0,
                        interpretability_weight=0.0).fit(X, y)
    assert len(set(sel.final_lengths_)) > 1, "all particles ended the same length"
    assert sel.convergence_.mean_length[0] != sel.convergence_.mean_length[-1]


def test_length_can_grow_not_only_shrink(small):
    """The submitted code's lengths were monotonically non-increasing."""
    X, y = small
    sel = VLPSOSelector(population_size=10, max_iter=14, divisions=3,
                        beta_stagnation=1, growth_prob=1.0,
                        length_direction="grow_only", min_length=2,
                        random_state=0, interpretability_weight=0.0).fit(X, y)
    lengths = np.array(sel.length_history_)
    assert lengths[-1].mean() > lengths[0].mean(), "grow_only did not grow"


def test_shrink_only_reproduces_the_old_behaviour(small):
    X, y = small
    sel = VLPSOSelector(population_size=10, max_iter=14, divisions=3,
                        beta_stagnation=1, length_direction="shrink_only",
                        random_state=0, interpretability_weight=0.0).fit(X, y)
    lengths = np.array(sel.length_history_)
    assert lengths[-1].mean() <= lengths[0].mean()


def test_cross_length_fallbacks_all_run(small):
    X, y = small
    for rule in ("gbest_longest", "pbest", "inertia"):
        sel = VLPSOSelector(beta_stagnation=2, population_size=8, max_iter=6, divisions=3,
                            exemplar_fallback=rule, random_state=0,
                            interpretability_weight=0.0).fit(X, y)
        assert sel.n_selected_ >= 1


def test_fitness_is_not_resubstitution(small):
    """An internal CV score on real data must be below a memorised one."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    X, y = small
    sel = VLPSOSelector(beta_stagnation=2, population_size=6, max_iter=3, divisions=2,
                        random_state=0, interpretability_weight=0.0).fit(X, y)
    cols = np.where(sel.support_)[0]
    Xv = np.nan_to_num(X.to_numpy(float))[:, cols]
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xv, y)
    resub = balanced_accuracy_score(y, knn.predict(Xv))
    internal = sel._fitness(np.nan_to_num(X.to_numpy(float)), np.asarray(y), cols)
    assert internal < resub * sel.gamma_accuracy + 1e-9


def test_cardinality_penalty_reduces_subset_size(small):
    X, y = small
    kw = dict(population_size=12, max_iter=10, divisions=3, random_state=0,
              interpretability_weight=0.0)
    free = VLPSOSelector(beta_stagnation=2, length_penalty=0.0, **kw).fit(X, y)
    penal = VLPSOSelector(beta_stagnation=2, length_penalty=3.0, **kw).fit(X, y)
    assert penal.n_selected_ <= free.n_selected_


def test_reports_direction_and_parameters(small):
    X, y = small
    sel = VLPSOSelector(beta_stagnation=2, population_size=6, max_iter=4, divisions=2,
                        random_state=0, interpretability_weight=0.0).fit(X, y)
    rep = sel.algorithm_report()
    assert rep["direction"] == "maximize"          # never stated in the original
    assert rep["n_evaluations"] > 0
    assert rep["stopping"] and rep["cross_length_rule"]
    assert "alpha" not in rep                      # unused parameter removed
    assert sel.complexity().startswith("O(")


def test_selection_is_by_name(small):
    X, y = small
    sel = VLPSOSelector(beta_stagnation=2, population_size=6, max_iter=4, divisions=2,
                        random_state=0, interpretability_weight=0.0).fit(X, y)
    assert set(sel.selected_feature_names_) <= set(X.columns)
    assert sel.transform(X).shape[1] == sel.n_selected_


def test_deterministic_under_a_fixed_seed(small):
    X, y = small
    kw = dict(population_size=8, max_iter=6, divisions=2, random_state=123,
              interpretability_weight=0.0)
    a = VLPSOSelector(beta_stagnation=2, **kw).fit(X, y)
    b = VLPSOSelector(beta_stagnation=2, **kw).fit(X, y)
    assert a.selected_feature_names_ == b.selected_feature_names_
    assert a.gbest_fitness_ == pytest.approx(b.gbest_fitness_)


def test_stops_early_on_convergence(small):
    X, y = small
    sel = VLPSOSelector(beta_stagnation=2, population_size=6, max_iter=100, divisions=2,
                        convergence_patience=2, random_state=0,
                        interpretability_weight=0.0).fit(X, y)
    assert sel.stopped_early_ is True
    assert sel.stop_iteration_ < 99


def test_selector_refuses_a_leaking_frame(small):
    X, y = small
    X = X.copy()
    X["math_score"] = np.asarray(y, dtype=float) * 100
    with pytest.raises(LeakageError):
        VLPSOSelector(beta_stagnation=2, population_size=4, max_iter=2, divisions=2).fit(X, y)


def test_ranking_can_be_switched_for_the_su_ablation(small):
    X, y = small
    su = VLPSOSelector(beta_stagnation=2, ranking="symmetric_uncertainty", population_size=6,
                       max_iter=3, divisions=2, random_state=0,
                       interpretability_weight=0.0).fit(X, y)
    mi = VLPSOSelector(beta_stagnation=2, ranking="mutual_info", population_size=6, max_iter=3,
                       divisions=2, random_state=0,
                       interpretability_weight=0.0).fit(X, y)
    assert su.algorithm_report()["ranking"] == "symmetric_uncertainty"
    assert mi.algorithm_report()["ranking"] == "mutual_info"


def test_bad_parameters_raise():
    X = pd.DataFrame(np.random.default_rng(0).integers(0, 4, (200, 6)).astype(float))
    y = np.random.default_rng(1).integers(0, 2, 200)
    with pytest.raises(ValueError, match="length_direction"):
        VLPSOSelector(length_direction="sideways", population_size=4,
                      max_iter=2, divisions=2, beta_stagnation=1).fit(X, y)
    with pytest.raises(ValueError, match="ranking"):
        VLPSOSelector(beta_stagnation=2, ranking="vibes", population_size=4, max_iter=2,
                      divisions=2).fit(X, y)
    with pytest.raises(ValueError, match="init"):
        VLPSOSelector(beta_stagnation=1, init="lucky", population_size=4,
                      max_iter=2, divisions=2).fit(X, y)
    with pytest.raises(ValueError, match="resubstitution"):
        VLPSOSelector(beta_stagnation=1, fitness_cv_splits=1, population_size=4,
                      max_iter=2, divisions=2).fit(X, y)


# --------------------------------------------------------------------------
# BPSO: matched baseline
# --------------------------------------------------------------------------
def test_bpso_runs_and_reports(small):
    X, y = small
    sel = BPSOSelector(population_size=10, max_iter=6, random_state=0,
                       interpretability_weight=0.0).fit(X, y)
    assert sel.n_selected_ >= 2
    rep = sel.algorithm_report()
    assert rep["direction"] == "maximize"
    assert "fixed binary vector" in rep["representation"]


def test_bpso_respects_its_cardinality_cap(small):
    X, y = small
    sel = BPSOSelector(population_size=10, max_iter=5, max_cardinality=4,
                       random_state=0, interpretability_weight=0.0).fit(X, y)
    assert sel.n_selected_ <= 4


def test_bpso_and_vlpso_share_the_objective(small):
    """Matching discipline: identical objective, only the length mechanism differs."""
    X, y = small
    v = VLPSOSelector(beta_stagnation=2, population_size=6, max_iter=3, divisions=2, random_state=0,
                      interpretability_weight=0.0)
    b = BPSOSelector(population_size=6, max_iter=3, random_state=0,
                     interpretability_weight=0.0)
    v.fit(X, y); b.fit(X, y)
    for k in ("gamma_accuracy", "length_penalty", "interpretability_weight",
              "k_neighbors", "objective", "direction"):
        assert v.algorithm_report()[k] == b.algorithm_report()[k]


def test_inert_length_adaptation_is_refused(small):
    """beta_stagnation >= max_iter makes the ablation vacuous. Raise, do not run."""
    X, y = small
    with pytest.raises(ValueError, match="never"):
        VLPSOSelector(population_size=6, max_iter=8, beta_stagnation=9,
                      divisions=2).fit(X, y)


def test_length_direction_ablation_is_not_vacuous(small):
    """With adaptation able to fire, the ablation arms must actually differ."""
    X, y = small
    kw = dict(population_size=12, max_iter=14, divisions=3, beta_stagnation=2,
              random_state=0, interpretability_weight=0.0)
    both = VLPSOSelector(length_direction="both", **kw).fit(X, y)
    shrink = VLPSOSelector(length_direction="shrink_only", **kw).fit(X, y)
    assert (both.final_lengths_ != shrink.final_lengths_
            or both.selected_feature_names_ != shrink.selected_feature_names_)
