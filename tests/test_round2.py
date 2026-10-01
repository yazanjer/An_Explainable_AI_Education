"""Round-2 code: grouped wrapper fitness (M7), matched swarms (M11), the five
selectors that did not exist, inner-loop k, Nogueira stability, LIME statistics
that are not zero-fill artefacts, and the cell runner's idempotence."""

import json

import numpy as np
import pandas as pd
import pytest

from vlpso_xai.data.features import LeakageError
from vlpso_xai.experiments import kinds as K
from vlpso_xai.experiments.cells import (cell_seed, is_done, load_r2, manifest, out_dir,
                                         parse_cell_id, reconcile, run_cell)
from vlpso_xai.experiments.data import outer_split, synthetic_frames, task_view
from vlpso_xai.evaluation.stability import nogueira
from vlpso_xai.selection.base import fitness_splits
from vlpso_xai.selection.embedded import BorutaSelector, L1PathRanker, TreeImportanceRanker
from vlpso_xai.selection.filters import SymmetricUncertaintyFilter
from vlpso_xai.selection.tuned import TunedTopKSelector
from vlpso_xai.selection.wrappers import RFERanker, SFSSelector


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(0)
    n_sch, per = 40, 15
    n = n_sch * per
    X = pd.DataFrame(rng.normal(size=(n, 12)), columns=[f"ST0{i:02d}Q01TA" for i in range(12)])
    y = (X["ST000Q01TA"] + 0.7 * X["ST001Q01TA"] + rng.normal(0, 1, n) > 0).astype(int).to_numpy()
    g = np.repeat(np.arange(n_sch), per)
    return X, y, g


@pytest.fixture(scope="module")
def r2():
    r = load_r2()
    r = json.loads(json.dumps(r))
    r["sel"]["swarm"].update(population_size=8, max_iter=12, convergence_patience=4)
    r["sel"]["vlpso_only"]["beta_stagnation"] = 3
    r["sel"]["boruta"]["max_iter"] = 6
    r["sel"]["sfs"]["k_max"] = 5
    return r


# --- M7 ---------------------------------------------------------------------
def test_fitness_splits_are_school_grouped(small):
    _, y, g = small
    splits, name = fitness_splits(y, g, 3, 0)
    assert name.startswith("StratifiedGroupKFold")
    for tr, va in splits:
        assert not set(g[tr]) & set(g[va])


def test_require_groups_raises_without_groups(small):
    _, y, _ = small
    with pytest.raises(ValueError, match="groups are required"):
        fitness_splits(y, None, 3, 0, require_groups=True)


@pytest.mark.parametrize("maker", ["make_vlpso", "make_bpso"])
def test_swarm_fit_uses_grouped_folds(small, r2, maker):
    X, y, g = small
    sel = getattr(K, maker)(r2, 1)
    with pytest.raises(ValueError, match="groups are required"):
        sel.fit(X, y)
    sel.fit(X, y, groups=g)
    assert sel.fitness_splitter_.startswith("StratifiedGroupKFold")
    for tr, va in sel._splits:
        gs = g if sel._fit_idx is None else g[sel._fit_idx]
        assert not set(gs[tr]) & set(gs[va])


# --- M11 --------------------------------------------------------------------
def test_bpso_and_vlpso_share_every_common_setting():
    r = load_r2()
    v, b = K.make_vlpso(r, 7), K.make_bpso(r, 7)
    common, _ = K.swarm_kwargs(r, 7)
    for k in common:
        assert getattr(v, k) == getattr(b, k), k
    assert v.min_length == b.min_features
    assert v.max_length == b.max_cardinality      # both uncapped


# --- the five selectors that did not exist ------------------------------------
@pytest.mark.parametrize("sel", [
    lambda: TunedTopKSelector(L1PathRanker(), k_grid=(2, 4), inner_splits=3),
    lambda: TunedTopKSelector(TreeImportanceRanker(n_estimators=30), k_grid=(2, 4), inner_splits=3),
    lambda: TunedTopKSelector(RFERanker(), k_grid=(2, 4), inner_splits=3),
    lambda: SFSSelector(k_max=4, cv_splits=3),
    lambda: BorutaSelector(max_iter=8, n_estimators=30),
])
def test_new_selectors_find_the_signal(small, sel):
    X, y, g = small
    s = sel().fit(X, y, groups=g)
    assert "ST000Q01TA" in s.selected_feature_names_
    assert all(n in X.columns for n in s.selected_feature_names_)


def test_tuned_k_is_chosen_from_the_grid_on_inner_folds(small):
    X, y, g = small
    s = TunedTopKSelector(SymmetricUncertaintyFilter(k=100), k_grid=(1, 2, 6), inner_splits=3,
                          require_groups=True).fit(X, y, groups=g)
    assert s.k_ in (1, 2, 6) and s.n_selected_ == s.k_
    assert len(s.cv_auc_by_k_) == 3
    assert s.inner_splitter_.startswith("StratifiedGroupKFold")


@pytest.mark.parametrize("sel", [L1PathRanker(), TreeImportanceRanker(n_estimators=10),
                                 RFERanker(), SFSSelector(k_max=2), BorutaSelector(max_iter=2)])
def test_new_selectors_refuse_a_leaking_matrix(small, sel):
    X, y, g = small
    with pytest.raises(LeakageError):
        sel.fit(X.assign(PV1MATH=y * 100.0), y, groups=g)


# --- stability ---------------------------------------------------------------
def test_nogueira_reference_values():
    f = [f"f{i}" for i in range(10)]
    assert nogueira([["f0", "f1"]] * 4, f)["nogueira"] == pytest.approx(1.0)
    assert nogueira([["f0", "f1"], ["f2", "f3"], ["f4", "f5"]], f)["nogueira"] < 0
    rng = np.random.default_rng(1)
    rand = [list(rng.choice(f, 3, replace=False)) for _ in range(400)]
    assert abs(nogueira(rand, f)["nogueira"]) < 0.02
    # defined for unequal sizes, where Kuncheva is not
    assert np.isfinite(nogueira([["f0"], ["f0", "f1", "f2"], ["f0", "f1"]], f)["nogueira"])


# --- LIME statistics ---------------------------------------------------------
def test_lime_conditional_sd_is_not_a_zero_fill_artefact():
    pytest.importorskip("lime")
    from sklearn.linear_model import LogisticRegression

    from vlpso_xai.explain.lime_local import lime_seed_stability

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 8)), columns=[f"c{i}" for i in range(8)])
    y = (X.c0 + rng.normal(0, 0.5, 300) > 0).astype(int)
    m = LogisticRegression().fit(X, y)
    out = lime_seed_stability(m.predict_proba, X, X.iloc[:2], n_repeats=6, n_samples=300, n_features=3)
    f = out["features"]
    assert {"selection_frequency", "sd_weight_when_named", "cv_when_named"} <= set(f.columns)
    top = f[(f.feature == "c0")]
    assert (top.n_seeds_naming == 6).all()
    assert (top.cv_when_named < 0.5).all()              # stable leading term
    assert out["settings"]["unnamed_weight_convention"].startswith("excluded")


# --- cells -------------------------------------------------------------------
def test_manifest_ids_are_unique_and_seeds_stable():
    r = load_r2()
    m = manifest(r)
    assert m.cell_id.is_unique
    assert len(m[m.kind == "sel"]) == 3 * 10 * 5 * 5 * 16
    row = m.iloc[123]
    assert cell_seed(row.cell_id) == row.seed
    assert parse_cell_id(row.cell_id)["kind"] == row.kind


def test_cell_outer_fold_is_the_headline_fold():
    from vlpso_xai.data.design import school_grouped_splitter

    fr = synthetic_frames(n_schools=30, per=20)
    X, y, g, _ = task_view(fr, "low_vs_medium", 1)
    tr, te = outer_split(fr, X, y, g, 2, 3)
    ref = list(school_grouped_splitter(5, shuffle=True, random_state=42 + 2).split(X, y, groups=g))[3]
    assert np.array_equal(te, ref[1])


def test_run_cell_is_atomic_and_idempotent(tmp_path, r2):
    fr = synthetic_frames(n_schools=40, per=20)
    m = manifest(r2, ["sel"])
    row = m[(m.method == "chi2") & (m.task == "low_vs_medium") & (m.pv == 1) & (m.rep == 0)
            & (m.fold == 0)].iloc[0].to_dict()
    assert run_cell(row, fr, r2, tmp_path) == "done"
    d = out_dir(tmp_path, row)
    assert (d / "meta.json").exists() and (d / "result.parquet").exists()
    assert not list(d.parent.glob(".tmp_*"))
    res = pd.read_parquet(d / "result.parquet").iloc[0]
    assert res.fitness_splitter.startswith("StratifiedGroupKFold")
    assert run_cell(row, fr, r2, tmp_path) == "skipped"
    assert is_done(tmp_path, row)
    rec = reconcile(tmp_path, m)
    assert int(rec.loc[rec.kind == "sel", "completed"].iloc[0]) == 1


def test_a_failing_cell_leaves_no_result(tmp_path, r2):
    fr = synthetic_frames(n_schools=30, per=20)
    row = manifest(r2, ["sel"]).iloc[0].to_dict()
    row["method"] = "does_not_exist"
    assert run_cell(row, fr, r2, tmp_path) == "failed"
    assert not out_dir(tmp_path, row).exists()
    assert list((tmp_path / "cells" / "_errors").glob("*.txt"))


def test_selector_aggregation_on_a_complete_fabricated_design(tmp_path):
    from vlpso_xai.experiments.aggregate import selector_tables
    from vlpso_xai.experiments.cells import write_atomic

    r = load_r2()
    r["sel"].update(pvs=[1, 2], repeats=[0], folds=[0, 1, 2, 3], methods=["none", "vlpso", "chi2"],
                    tasks=["low_vs_high"])
    rng = np.random.default_rng(0)
    m = manifest(r, ["sel"])
    for row in m.to_dict("records"):
        base = {"none": 0.88, "vlpso": 0.85, "chi2": 0.87}[row["method"]]
        items = ["a", "b", "c"] if row["method"] != "vlpso" else list(rng.choice(list("abcdef"), 2, replace=False))
        res = {k: row[k] for k in ("task", "pv", "rep", "fold", "method")} | {
            "auc": base + rng.normal(0, 0.005), "n_selected": len(items), "n_items": len(items),
            "selected": json.dumps(items), "selected_items": json.dumps(items), "k_chosen": None,
            "selector_seconds": 1.0, "n_evaluations": None, "fitness_splitter": "StratifiedGroupKFold(CNTSCHID)",
            "convergence": None, "n_train": 12000, "n_test": 3000}
        write_atomic(out_dir(tmp_path, row), {"result": pd.DataFrame([res])}, {"cell_id": row["cell_id"]})
    t = selector_tables(tmp_path, r, all_features=list("abcdef"))
    s = t["sel_summary"].set_index("method")
    assert s.loc["none", "auc"] == pytest.approx(0.88, abs=0.01)
    assert s.loc["chi2", "nogueira_mean"] == pytest.approx(1.0)
    assert s.loc["vlpso", "nogueira_mean"] < 0.5
    c = t["sel_contrasts"]
    vn = c[(c.reference == "none") & (c.method == "vlpso")].iloc[0]
    assert vn.delta_auc == pytest.approx(-0.03, abs=0.01) and vn.p_holm < 0.05
    assert t["sel_budget"].complete.all()


def test_publish_refuses_student_level_tables(tmp_path):
    from vlpso_xai.experiments.cells import write_atomic
    from vlpso_xai.experiments.publish import PublishError, publish

    r = load_r2()
    row = manifest(r, ["brr"]).iloc[0].to_dict()
    write_atomic(out_dir(tmp_path, row),
                 {"summary": pd.DataFrame({"task": ["t"], "y_true": [1], "y_score": [0.9]})},
                 {"cell_id": row["cell_id"]})
    with pytest.raises(PublishError, match="student-level"):
        publish(tmp_path, tmp_path / "pub")
