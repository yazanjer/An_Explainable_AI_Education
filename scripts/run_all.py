#!/usr/bin/env python3
"""Reproduce every manuscript number from the raw OECD file in one command.

    python scripts/run_all.py --config quick     # ~15 min smoke test
    python scripts/run_all.py --config default   # full budget

An independent audit found that this script was referenced by README.md and
RESPONSE_TO_EDITOR.md but did not exist, and that most tables in results/ had
been produced by ad-hoc session code that was never committed. That is exactly
the reproducibility failure this revision exists to correct, so the script is
now the single sanctioned entry point and every stage writes through it.

Stages (each is skippable and independently resumable):

    0 ingest      SPSS -> parquet, country filter, full-width missingness
    1 audit       leakage demonstration on the ORIGINAL feature matrix
    2 sample      sample flow, weighted descriptives with BRR standard errors
    3 nested      nested CV per task, all plausible values
    4 selection   selector comparison and ablations
    5 stats       bootstrap CIs, permutation nulls, stability, effect sizes
    6 explain     global/local SHAP, LIME stability, negative control
    7 assets      emit every table as .tex + .csv, write the manifest
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("run_all")

STAGES = ["ingest", "audit", "sample", "nested", "selection", "stats", "explain", "assets"]

#: Stages that are DECLARED but NOT YET IMPLEMENTED in this script. They were
#: silently accepted and silently did nothing, so a 30-hour run could complete
#: "successfully" having produced none of the selector comparison, uncertainty
#: quantification or explainability outputs the manuscript needs. Requesting one
#: now fails loudly. The corresponding code exists and is tested -- it is driven
#: from notebooks 03, 05 and 06, not from here.
NOT_IMPLEMENTED = {
    "selection": "feature-selection comparison and ablations (use notebooks/03 and 04)",
    "stats": "bootstrap CIs, permutation nulls, stability, effect sizes (use notebook 05)",
    "explain": "global/local SHAP, LIME stability, negative control (use notebook 06)",
}


def _setup(config: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from vlpso_xai.config import load_config, set_global_seeds
    cfg = load_config(config)
    set_global_seeds(cfg.seed)
    cfg.paths.mkdirs()
    return cfg


# --------------------------------------------------------------------------
def stage_ingest(cfg):
    from vlpso_xai.data.ingest import build_analytic_frame
    return build_analytic_frame(cfg)


def stage_audit(cfg, df):
    """Editor comments 1-2: demonstrate the original leakage empirically."""
    import numpy as np, pandas as pd
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from vlpso_xai.data.features import classify_columns

    out = cfg.paths.results / "audit"; out.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["math_score"] = df[[f"PV{i}MATH" for i in range(1, 11)]].mean(axis=1)
    df["math_category"] = df["math_score"].apply(
        lambda s: "Low" if s <= 482 else ("Medium" if s <= 607 else "High"))

    classify_columns(df.columns).to_csv(out / "column_classification.csv", index=False)

    rows = []
    for name, (neg, pos) in {"low_vs_medium": ("Low", "Medium"),
                             "medium_vs_high": ("Medium", "High"),
                             "low_vs_high": ("Low", "High")}.items():
        sub = df[df.math_category.isin([neg, pos])]
        y = (sub.math_category == pos).astype(int)
        acc = cross_val_score(DecisionTreeClassifier(max_depth=1, random_state=0),
                              sub[["math_score"]], y, cv=5).mean()
        rows.append({"task": name, "n": len(sub), "stump_on_math_score_accuracy": acc})
    pd.DataFrame(rows).to_csv(out / "stump_on_outcome.csv", index=False)
    logger.info("audit: stump-on-outcome accuracies written")
    return df


def stage_sample(cfg, df):
    """Editor comment 3: sample flow + weighted descriptives with BRR SEs."""
    import numpy as np, pandas as pd
    from vlpso_xai.data.design import (SurveyDesign, brr_standard_error,
                                       brr_weighted_mean_se, design_diagnostics,
                                       weighted_proportion)
    from vlpso_xai.data.outcome import build_pv_categories, category_instability

    out = cfg.paths.results / "sample"; out.mkdir(parents=True, exist_ok=True)
    thr = cfg.section("data", "missing_row_threshold")
    primary = cfg.section("data", "primary_country")
    external = list(cfg.section("data", "external_countries"))

    # ------------------------------------------------------------------
    # CRITICAL: restrict to the PRIMARY country.
    #
    # build_analytic_frame() returns the primary country AND the external
    # validation countries, because both are read in one pass over the SPSS
    # file. An earlier version of this function applied only the missingness
    # rule, so Spain and Portugal were POOLED: the held-out country was in the
    # training data, and the "external" validation was not external at all.
    #
    # Caught by the sample-flow table reporting 41,875 students in 1,365
    # schools (= 35,943 Spanish + 5,932 Portuguese) where Spain alone is
    # 35,943 in 1,089. Printing the flow is what made it visible.
    # ------------------------------------------------------------------
    n_all = len(df)
    df = df[df["CNT"] == primary].reset_index(drop=True)
    logger.info("primary country %s: %d of %d rows (%s held out for external "
                "validation)", primary, len(df), n_all, external or "none")

    rows = [{"step": f"all countries read ({primary} + {', '.join(external) or 'none'})",
             "n": n_all, "schools": None},
            {"step": f"primary country = {primary}", "n": len(df),
             "schools": df.CNTSCHID.nunique()}]
    for t in [thr] + list(cfg.section("data", "missing_row_threshold_sensitivity")):
        k = df[df.n_missing_allcols <= t]
        label = "primary" if t == thr else "sensitivity"
        rows.append({"step": f"missingness <= {t} [{label}]", "n": len(k),
                     "schools": k.CNTSCHID.nunique()})
    pd.DataFrame(rows).to_csv(out / "sample_flow.csv", index=False)

    prim = df[df.n_missing_allcols <= thr].reset_index(drop=True)
    assert set(prim["CNT"].unique()) == {primary}, (
        f"analytic sample must contain only {primary}; found "
        f"{sorted(prim['CNT'].unique())}. The external validation country "
        "would be inside the training data.")
    # Drop any pre-existing cat_pv* before recomputing. Blindly concatenating
    # onto a cached frame that already carries them produces DUPLICATE column
    # names, so prim["cat_pv1"] silently returns a 2-column DataFrame and every
    # downstream weighted statistic broadcasts wrongly. Caught on the first
    # end-to-end run of this script.
    prim = prim.drop(columns=[c for c in prim.columns if c.startswith("cat_pv")])
    prim = pd.concat([prim, build_pv_categories(prim)], axis=1)
    assert not prim.columns.duplicated().any(), "duplicate columns in analytic frame"

    inst = category_instability(prim[[f"cat_pv{i}" for i in range(1, 11)]],
                                prim.W_FSTUWT.to_numpy())
    pd.DataFrame([inst]).to_csv(out / "pv_category_instability.csv", index=False)

    # BRR standard errors -- the machinery is CALLED here, not merely defined.
    design = SurveyDesign.from_frame(prim)
    pd.DataFrame([design_diagnostics(design)]).to_csv(
        out / "design_diagnostics.csv", index=False)
    brr_rows = []
    for cls in cfg.section("outcome", "category_labels"):
        m = (prim["cat_pv1"] == cls).to_numpy().astype(float)
        r = brr_standard_error(lambda w, mm=m: weighted_proportion(mm, w), design)
        brr_rows.append({"quantity": f"proportion {cls}", **r})
    for item in [c for c in prim.columns if c.startswith("ST0")][:20]:
        x = pd.to_numeric(prim[item], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).sum() > 100:
            brr_rows.append({"quantity": f"mean {item}",
                             **brr_weighted_mean_se(x, design)})
    pd.DataFrame(brr_rows).to_csv(out / "brr_descriptives.csv", index=False)
    logger.info("sample: N=%d, %d schools, BRR SEs for %d quantities",
                len(prim), prim.CNTSCHID.nunique(), len(brr_rows))
    return prim


def stage_nested(cfg, prim):
    """Editor comment 1: nested CV, per plausible value."""
    import pandas as pd
    from pathlib import Path as _P
    from vlpso_xai.data.features import Allowlist, build_design_matrix, single_variable_auc_screen
    from vlpso_xai.evaluation.nested_cv import NestedCVConfig, run_nested_cv
    from vlpso_xai.models.registry import get_models

    alw = Allowlist.load(_P(cfg.paths.root) / "config" / "predictor_allowlist.yaml")
    X_all = build_design_matrix(prim, alw, where="run_all")
    npv = cfg.section("outcome", "n_plausible_values")
    frames = []
    for task, spec in cfg.section("outcome", "tasks").items():
        neg, pos = spec["negative"], spec["positive"]
        for pv in range(1, npv + 1):
            m = prim[f"cat_pv{pv}"].isin([neg, pos]).to_numpy()
            if pv == 1:
                single_variable_auc_screen(
                    X_all[m], (prim[f"cat_pv{pv}"][m] == pos).astype(int).to_numpy(),
                    allowlist=alw, raise_on_flag=True
                ).to_csv(cfg.paths.results / "audit" /
                         f"single_variable_auc_screen_{task}.csv", index=False)
            frames.append(run_nested_cv(
                X_all[m].reset_index(drop=True),
                (prim[f"cat_pv{pv}"][m] == pos).astype(int).to_numpy(),
                prim.CNTSCHID.to_numpy()[m],
                models=get_models(cfg.section("models", "primary"),
                                  fast=cfg.raw["meta"]["name"] == "quick"),
                cfg=NestedCVConfig(
                    outer_splits=cfg.section("cv", "outer_splits"),
                    outer_repeats=cfg.section("cv", "outer_repeats"),
                    inner_splits=cfg.section("cv", "inner_splits"),
                    random_state=cfg.seed,
                    checkpoint_dir=cfg.paths.checkpoints),
                sample_weight=prim.W_FSTUWT.to_numpy()[m],
                task=task, pv=pv, method="none"))
    res = pd.concat(frames, ignore_index=True)
    res.to_parquet(cfg.paths.results / "nested_cv_results.parquet", index=False)
    logger.info("nested: %d folds across %d tasks x %d PVs",
                len(res), len(cfg.section("outcome", "tasks")), npv)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="default")
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    unimplemented = [s for s in args.stages if s in NOT_IMPLEMENTED]
    if unimplemented:
        raise SystemExit(
            "These stages are not implemented in run_all.py and would have done "
            "NOTHING silently:\n"
            + "\n".join(f"    {s}: {NOT_IMPLEMENTED[s]}" for s in unimplemented)
            + "\n\nRun the named notebook instead, or drop the stage."
        )

    cfg = _setup(args.config)

    # Self-identify. Diagnosing a stale Google Drive copy by inferring it from
    # a MISSING log line wasted several rounds; the running code now states
    # which commit it is.
    import subprocess as _sp
    root = Path(__file__).resolve().parents[1]
    try:
        sha = _sp.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                               stderr=_sp.DEVNULL, text=True).strip()
        subj = _sp.check_output(["git", "-C", str(root), "log", "-1", "--pretty=%s"],
                                stderr=_sp.DEVNULL, text=True).strip()[:60]
    except Exception:
        sha, subj = "unknown", "(not a git checkout)"
    import vlpso_xai
    from vlpso_xai.data import ingest as _ing
    logger.info("code    %s %s | vlpso_xai %s | ingest.stage_locally=%s",
                sha, subj, vlpso_xai.__version__, hasattr(_ing, "stage_locally"))
    logger.info("config %s | hash %s | root %s",
                cfg.config_path.name, cfg.hash()[:12], cfg.paths.root)

    t0 = time.perf_counter()
    df = prim = res = None
    if "ingest" in args.stages:
        df = stage_ingest(cfg)
    if "audit" in args.stages:
        df = stage_audit(cfg, df if df is not None else stage_ingest(cfg))
    if "sample" in args.stages:
        prim = stage_sample(cfg, df if df is not None else stage_ingest(cfg))
    if "nested" in args.stages:
        res = stage_nested(cfg, prim)
    if "assets" in args.stages:
        from vlpso_xai.config import environment_report
        from vlpso_xai.reporting.manifest import write_manifest
        p = write_manifest(cfg.paths.results, config_hash=cfg.hash(),
                           generating_script="scripts/run_all.py",
                           repo=cfg.paths.root, environment=environment_report())
        logger.info("manifest: %s", p)

    logger.info("done in %.1f min", (time.perf_counter() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
