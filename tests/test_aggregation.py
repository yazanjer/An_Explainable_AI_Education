"""Tests for the M2 aggregation hierarchy and the M4 effect-size guard.

All synthetic. The prediction checkpoints are fabricated to the schema written
by :func:`vlpso_xai.evaluation.nested_cv.run_nested_cv`, so no PISA microdata
and no 34-hour run is needed to exercise the aggregation contract.

The tests are written to have *demonstrated* power, in the sense of
``tests/test_leakage_detection_power.py``: each one constructs a checkpoint set
that exhibits the specific defect and asserts the code refuses it. A test that
only checks the happy path would have passed against the buggy notebook too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vlpso_xai.evaluation.aggregate import (
    aggregate_task_method,
    aggregation_table,
    inventory,
    load_fold_predictions,
    oof_predictions,
    parse_checkpoint_name,
)
from vlpso_xai.evaluation.effect_size import (
    MIN_FOLDS_FOR_MAGNITUDE,
    cohens_d_paired_ci,
    contrast_table,
    interpret_d,
    interpret_d_guarded,
    paired_method_contrast,
)

N_SCHOOLS = 60
PER_SCHOOL = 8


def _write_repeat(
    d,
    *,
    task="low_vs_high",
    method="none",
    pv=1,
    rep=0,
    n_folds=5,
    cfg="aaaaaaaaaa",
    signal=1.0,
    seed=0,
    embed_fingerprint=True,
):
    """Write one repeat as ``n_folds`` school-disjoint prediction checkpoints."""
    rng = np.random.default_rng(seed)
    schools = np.arange(N_SCHOOLS)
    rng.shuffle(schools)
    chunks = np.array_split(schools, n_folds)

    for f, chunk in enumerate(chunks):
        groups = np.repeat(chunk, PER_SCHOOL)
        n = groups.size
        y = rng.integers(0, 2, n)
        score = np.clip(0.5 + signal * (y - 0.5) * 0.4 + rng.normal(0, 0.2, n), 0, 1)
        frame = pd.DataFrame({
            "y_true": y,
            "y_score": score,
            "group": groups,
            "sample_weight": np.nan,
            "threshold": 0.5,
            "task": task, "pv": pv, "method": method, "rep": rep, "fold": f,
        })
        if embed_fingerprint:
            frame["cfg_fingerprint"] = cfg
        name = (f"fold_task-{task}_pv-{pv}_method-{method}_rep-{rep}"
                f"_fold-{f}_cfg-{cfg}_preds.parquet")
        frame.to_parquet(d / name, index=False)


@pytest.fixture
def one_config(tmp_path):
    """One fingerprint, one method, 2 PVs x 2 repeats x 5 folds."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    seed = 0
    for pv in (1, 2):
        for rep in (0, 1):
            _write_repeat(d, pv=pv, rep=rep, seed=seed, signal=1.0 + 0.05 * pv)
            seed += 1
    return d


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
def test_parse_checkpoint_name_roundtrip():
    r = parse_checkpoint_name(
        "fold_task-low_vs_high_pv-3_method-vlpso_rep-2_fold-4_cfg-350625c08f_preds.parquet"
    )
    assert r["task"] == "low_vs_high"      # underscores in the task survive
    assert (r["pv"], r["method"], r["rep"], r["fold"]) == (3, "vlpso", 2, 4)
    assert r["cfg_fingerprint"] == "350625c08f"


def test_parse_checkpoint_name_handles_legacy_unfingerprinted():
    r = parse_checkpoint_name(
        "fold_task-low_vs_med_pv-1_method-none_rep-0_fold-0_preds.parquet"
    )
    assert r["cfg_fingerprint"] is None


def test_parse_checkpoint_name_ignores_non_prediction_files():
    # The fold-result checkpoints share the stem but not the _preds suffix.
    assert parse_checkpoint_name(
        "fold_task-low_vs_high_pv-1_method-none_rep-0_fold-0_cfg-2a228fe0c5.parquet"
    ) is None
    assert parse_checkpoint_name("manifest.json") is None


# ---------------------------------------------------------------------------
# M2 (1): the glob must not mix configuration fingerprints
# ---------------------------------------------------------------------------
def test_two_configs_for_the_SAME_cell_raise_and_name_the_candidates(tmp_path):
    """Same (task, method, pv), run twice under different configurations."""
    d = tmp_path / "ckpt"; d.mkdir()
    _write_repeat(d, cfg="aaaaaaaaaa", seed=1)
    _write_repeat(d, cfg="bbbbbbbbbb", seed=2)

    with pytest.raises(ValueError) as exc:
        load_fold_predictions(d)
    msg = str(exc.value)
    assert "more than one configuration" in msg
    assert "aaaaaaaaaa" in msg and "bbbbbbbbbb" in msg   # both listed, none chosen


def test_different_configs_across_DIFFERENT_cells_are_accepted(tmp_path):
    """The real-world layout, and the bug in the first version of this module.

    stage_nested calls run_nested_cv once per (task, pv) on that combination's
    row subset, and the fingerprint hashes the row signature -- so every task
    and every PV necessarily has its own fingerprint. Demanding one fingerprint
    per directory could never be satisfied by a real run: it rejected all 30
    cells of the 750-fold run.
    """
    # Fingerprints must be hex: run_nested_cv writes a sha256 prefix and the
    # filename regex only accepts [0-9a-f].
    d = tmp_path / "ckpt"; d.mkdir()
    seed = 0
    for task in ("low_vs_high", "low_vs_med", "med_vs_high"):
        for pv in (1, 2):
            _write_repeat(d, task=task, pv=pv, cfg=f"{seed:010x}", seed=seed)
            seed += 1
    cset = load_fold_predictions(d)          # must NOT raise
    assert len(cset.fingerprints) == 6       # one per cell
    assert len(set(cset.fingerprints.values())) == 6
    assert cset.fingerprint is None          # no single directory-wide value
    assert sorted(cset.tasks) == ["low_vs_high", "low_vs_med", "med_vs_high"]


def test_legacy_files_count_as_a_distinct_configuration(tmp_path):
    """A pre-fingerprinting file cannot be matched to a config, so it is not
    quietly absorbed into the fingerprinted set."""
    d = tmp_path / "ckpt"; d.mkdir()
    _write_repeat(d, cfg="aaaaaaaaaa", seed=1)
    for f in range(5):
        pd.DataFrame({
            "y_true": [0, 1], "y_score": [0.2, 0.8], "group": [900, 901],
            "sample_weight": [np.nan] * 2, "threshold": [0.5] * 2,
            "task": "low_vs_high", "pv": 1, "method": "none", "rep": 0, "fold": f,
        }).to_parquet(
            d / f"fold_task-low_vs_high_pv-1_method-none_rep-0_fold-{f}_preds.parquet",
            index=False,
        )
    with pytest.raises(ValueError, match="legacy"):
        load_fold_predictions(d)


def test_explicit_fingerprint_selects_one_configuration(tmp_path):
    d = tmp_path / "ckpt"; d.mkdir()
    _write_repeat(d, cfg="aaaaaaaaaa", seed=1)
    _write_repeat(d, cfg="bbbbbbbbbb", seed=2)
    cset = load_fold_predictions(d, cfg_fingerprint="bbbbbbbbbb")
    assert cset.fingerprint == "bbbbbbbbbb"
    assert len(cset.index) == 5


def test_declared_budget_separates_a_quick_run_from_the_full_one(tmp_path):
    """The exact situation in the live checkpoint directory: 5x5 cells and
    leftover 3-fold quick-mode cells for the same (task, pv)."""
    d = tmp_path / "ckpt"; d.mkdir()
    for rep in range(5):                                  # full: 5 repeats x 5 folds
        _write_repeat(d, rep=rep, n_folds=5, cfg="fadedfaded", seed=20 + rep)
    _write_repeat(d, rep=0, n_folds=3, cfg="0badbadbad", seed=99)   # leftover

    with pytest.raises(ValueError, match="more than one configuration"):
        load_fold_predictions(d)

    cset = load_fold_predictions(d, outer_splits=5, outer_repeats=5)
    assert cset.fingerprint == "fadedfaded"
    assert len(cset.index) == 25
    b = cset.budget()
    assert b["n_repeats"].iloc[0] == 5 and b["folds_per_repeat"].iloc[0] == 5


def test_declared_budget_that_matches_nothing_raises_with_an_inventory(tmp_path):
    d = tmp_path / "ckpt"; d.mkdir()
    _write_repeat(d, n_folds=5, cfg="aaaaaaaaaa", seed=1)
    with pytest.raises(ValueError, match="No cell matches the declared budget"):
        load_fold_predictions(d, outer_splits=5, outer_repeats=5)


def test_inventory_lists_everything_and_never_raises(tmp_path):
    d = tmp_path / "ckpt"; d.mkdir()
    for rep in range(2):
        _write_repeat(d, rep=rep, n_folds=5, cfg="fadedfaded", seed=30 + rep)
    _write_repeat(d, rep=0, n_folds=3, cfg="0badbadbad", seed=98)
    inv = inventory(d)                                    # must not raise
    assert set(inv["cfg_fingerprint"]) == {"fadedfaded", "0badbadbad"}
    assert inv.loc[inv.cfg_fingerprint == "fadedfaded", "total_folds"].iloc[0] == 10
    assert inv.loc[inv.cfg_fingerprint == "0badbadbad", "folds_per_repeat"].iloc[0] == 3


def test_renamed_file_is_caught_by_the_embedded_fingerprint(tmp_path):
    """Filename says one config, payload says another."""
    d = tmp_path / "ckpt"; d.mkdir()
    _write_repeat(d, cfg="aaaaaaaaaa", seed=1)
    victim = next(d.glob("*fold-0*"))
    frame = pd.read_parquet(victim)
    frame["cfg_fingerprint"] = "ffffffffff"
    frame.to_parquet(victim, index=False)

    cset = load_fold_predictions(d, cfg_fingerprint="aaaaaaaaaa")
    with pytest.raises(ValueError, match="renamed or copied"):
        oof_predictions(cset, "low_vs_high", "none", 1, 0)


# ---------------------------------------------------------------------------
# M2 (2): structural integrity of a level-1 out-of-fold vector
# ---------------------------------------------------------------------------
def test_oof_is_disjoint_and_covers_every_school(one_config):
    cset = load_fold_predictions(one_config)
    d = oof_predictions(cset, "low_vs_high", "none", 1, 0, expected_folds=5)
    assert len(d) == N_SCHOOLS * PER_SCHOOL
    assert d["group"].nunique() == N_SCHOOLS
    # each school in exactly one fold
    assert (d.groupby("group")["fold"].nunique() == 1).all()


def test_missing_fold_raises_rather_than_scoring_a_partial_sample(tmp_path):
    d = tmp_path / "ckpt"; d.mkdir()
    _write_repeat(d, seed=3)
    next(d.glob("*fold-4*")).unlink()
    cset = load_fold_predictions(d)
    with pytest.raises(ValueError, match="expected 5"):
        oof_predictions(cset, "low_vs_high", "none", 1, 0, expected_folds=5)


def test_overlapping_folds_raise(tmp_path):
    """The exact duplication M2 describes: a school in two outer folds."""
    d = tmp_path / "ckpt"; d.mkdir()
    _write_repeat(d, seed=4)
    f0 = next(d.glob("*fold-0*"))
    f1 = next(d.glob("*fold-1*"))
    dup = pd.read_parquet(f0)
    pd.concat([pd.read_parquet(f1), dup], ignore_index=True).to_parquet(f1, index=False)

    cset = load_fold_predictions(d)
    with pytest.raises(ValueError, match="more than one outer fold"):
        oof_predictions(cset, "low_vs_high", "none", 1, 0)


# ---------------------------------------------------------------------------
# M2 (3): the hierarchy produces a WIDER interval than pooling
# ---------------------------------------------------------------------------
def test_rubin_interval_is_wider_than_the_naive_pooled_one(one_config):
    """The headline regression test.

    Pooling repeats and PVs enters each student four times here (2 PV x 2 rep)
    and discards between-PV variance. The correct hierarchy must therefore give
    a strictly wider interval. If this ever inverts, the fix has been undone.
    """
    from vlpso_xai.evaluation.metrics import cluster_bootstrap_ci
    import glob

    cset = load_fold_predictions(one_config)
    res = aggregate_task_method(
        cset, "low_vs_high", "none", n_resamples=300, expected_folds=5,
    )
    s = res["summary"]
    assert s["cfg_fingerprints"]        # provenance recorded, one entry per PV
    assert s["n_pv"] == 2 and s["n_repeats"] == 2
    assert not s["single_pv"]
    assert s["between_variance"] > 0            # PV uncertainty is carried
    assert 0.0 < s["fmi"] <= 1.0

    pooled = pd.concat(
        [pd.read_parquet(p) for p in glob.glob(str(one_config / "*_preds.parquet"))],
        ignore_index=True,
    )
    assert len(pooled) == 4 * N_SCHOOLS * PER_SCHOOL     # each student x4
    naive = cluster_bootstrap_ci(
        pooled.y_true.to_numpy(), pooled.y_score.to_numpy(),
        pooled.group.to_numpy(), metric="auc", n_resamples=300,
    )
    assert (s["ci_high"] - s["ci_low"]) > (naive["ci_high"] - naive["ci_low"])


def test_single_pv_is_flagged_not_silently_reported(tmp_path):
    d = tmp_path / "ckpt"; d.mkdir()
    _write_repeat(d, pv=1, rep=0, seed=5)
    cset = load_fold_predictions(d)
    res = aggregate_task_method(cset, "low_vs_high", "none", n_resamples=200,
                                expected_folds=5)
    assert res["summary"]["single_pv"] is True
    assert res["summary"]["between_variance"] == 0.0


def test_methods_are_never_combined(tmp_path):
    d = tmp_path / "ckpt"; d.mkdir()
    _write_repeat(d, method="vlpso", seed=6, signal=1.2)
    _write_repeat(d, method="chi2", seed=7, signal=0.6)
    cset = load_fold_predictions(d)
    out = aggregation_table(cset, n_resamples=200, expected_folds=5)
    assert set(out["summary"]["method"]) == {"vlpso", "chi2"}
    assert len(out["summary"]) == 2                     # two rows, not one pooled row


def test_aggregation_table_writes_all_three_levels(one_config):
    cset = load_fold_predictions(one_config)
    out = aggregation_table(cset, n_resamples=200, expected_folds=5)
    assert set(out) == {"summary", "per_pv", "per_repeat"}
    assert len(out["per_repeat"]) == 4      # 2 pv x 2 rep
    assert len(out["per_pv"]) == 2
    assert len(out["summary"]) == 1


def test_empty_checkpoint_dir_raises(tmp_path):
    d = tmp_path / "empty"; d.mkdir()
    with pytest.raises(FileNotFoundError, match="No '\\*_preds.parquet'"):
        load_fold_predictions(d)


# ---------------------------------------------------------------------------
# M4: the effect-size guard
# ---------------------------------------------------------------------------
def _paired_frame(delta, n_folds, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_folds):
        base = rng.normal(0.80, 0.02)
        rows.append(dict(task="t", pv=1, repeat=0, outer_fold=f,
                         method="vlpso", auc=base + delta + rng.normal(0, sd)))
        rows.append(dict(task="t", pv=1, repeat=0, outer_fold=f,
                         method="bpso", auc=base))
    return pd.DataFrame(rows)


def test_three_folds_never_get_a_magnitude_label():
    """The M4 defect verbatim: a paired d over 3 folds called 'large'."""
    c = paired_method_contrast(_paired_frame(0.05, 3, sd=0.002), "vlpso", "bpso",
                               n_train=8000, n_test=2000)
    assert c.n_folds == 3
    assert abs(c.cohens_d_paired) > 0.8            # unguarded, this IS "large"
    assert c.magnitude_unguarded == "large"        # the old behaviour
    assert c.underpowered is True
    assert c.magnitude == "indeterminate (J=3 < 10)"


def test_a_wide_ci_blocks_the_label_even_with_enough_folds():
    c = paired_method_contrast(_paired_frame(0.01, 20, sd=0.03, seed=11),
                               "vlpso", "bpso", n_train=8000, n_test=2000)
    assert c.n_folds == 20
    assert c.magnitude.startswith("indeterminate (CI spans")


def test_a_well_resolved_effect_still_gets_a_label():
    c = paired_method_contrast(_paired_frame(0.05, 40, sd=0.001, seed=12),
                               "vlpso", "bpso", n_train=8000, n_test=2000)
    assert c.underpowered is False
    assert c.magnitude in {"small", "medium", "large"}
    assert c.magnitude == c.magnitude_unguarded


def test_d_ci_is_reported_and_brackets_the_point_estimate():
    c = paired_method_contrast(_paired_frame(0.03, 25, sd=0.01, seed=13),
                               "vlpso", "bpso", n_train=8000, n_test=2000)
    assert np.isfinite(c.d_ci_low) and np.isfinite(c.d_ci_high)
    assert c.d_ci_low <= c.cohens_d_paired <= c.d_ci_high


def test_d_ci_widens_as_folds_shrink():
    wide = cohens_d_paired_ci(np.random.default_rng(1).normal(0.5, 1.0, 4))
    tight = cohens_d_paired_ci(np.random.default_rng(1).normal(0.5, 1.0, 200))
    assert (wide["ci_high"] - wide["ci_low"]) > (tight["ci_high"] - tight["ci_low"])


def test_interpret_d_guarded_matches_interpret_d_when_well_resolved():
    for d in (0.05, 0.3, 0.6, 0.95):
        assert interpret_d_guarded(d, 50, ci_low=d - 0.01, ci_high=d + 0.01) == interpret_d(d)


def test_interpret_d_guarded_treats_a_zero_crossing_as_negligible():
    out = interpret_d_guarded(0.9, 50, ci_low=-0.2, ci_high=1.4)
    assert out.startswith("indeterminate (CI spans negligible")


def test_contrast_table_exposes_the_underpowered_flag():
    out = contrast_table(_paired_frame(0.05, 3, sd=0.002), reference="vlpso",
                         n_train=8000, n_test=2000)
    assert out["underpowered"].all()
    assert out["magnitude"].str.startswith("indeterminate").all()
    assert (out["n_folds"] == 3).all()
    assert MIN_FOLDS_FOR_MAGNITUDE == 10


def test_contrast_table_logs_skipped_pairs_instead_of_dropping_them(caplog):
    """A dropped contrast shrinks the multiplicity family silently."""
    df = _paired_frame(0.02, 20, seed=14)
    orphan = df[df.method == "bpso"].copy()
    orphan["method"] = "mrmr"
    orphan["outer_fold"] = orphan["outer_fold"] + 1000     # matches nothing
    with caplog.at_level("WARNING"):
        contrast_table(pd.concat([df, orphan], ignore_index=True),
                       reference="vlpso", n_train=8000, n_test=2000)
    assert "skipped" in caplog.text and "mrmr" in caplog.text
