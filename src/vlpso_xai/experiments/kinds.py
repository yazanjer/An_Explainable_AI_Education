"""The six round-2 cell kinds. Each ``run_<kind>(row, frames, r2)`` returns
``(tables, meta)``; :func:`vlpso_xai.experiments.cells.run_cell` writes them.

sel     selector comparison, one (task, pv, repeat, fold, method)
vlstab  VLPSO/BPSO seed stability and one-at-a-time sensitivity
perm    one permutation draw (or the observed run) of the headline fold spec
shap    global/local SHAP (+ LIME at the configured PVs) for one headline fold
ext     Spain -> Portugal external validation at one PV
brr     Fay-BRR standard error of the weighted fold-weighted AUC at one PV
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import Frames, outer_split, task_view

logger = logging.getLogger(__name__)

Tables = Dict[str, pd.DataFrame]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def preprocessor():
    """impute (+ missingness indicators) -> scale, fitted on the training part only."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    p = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True,
                                           keep_empty_features=True)),
                  ("scale", StandardScaler())])
    p.set_output(transform="pandas")
    return p


def inner_splits(X, y, g, seed: int, n: int = 5):
    from ..data.design import school_grouped_splitter

    return list(school_grouped_splitter(n, shuffle=True, random_state=seed).split(X, y, groups=g))


def tune_lr(Xtr: pd.DataFrame, ytr, splits, C_grid) -> Tuple[float, List[float]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    vals = Xtr.to_numpy()
    scores = []
    for C in C_grid:
        s = []
        for tr, va in splits:
            m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
            m.fit(vals[tr], ytr[tr])
            s.append(roc_auc_score(ytr[va], m.decision_function(vals[va])))
        scores.append(float(np.mean(s)))
    return float(C_grid[int(np.argmax(scores))]), scores


def fold_weighted(aucs, ns) -> float:
    a, n = np.asarray(aucs, float), np.asarray(ns, float)
    return float(np.sum(a * n) / np.sum(n))


def swarm_kwargs(r2: Dict[str, Any], seed: int) -> Tuple[dict, dict]:
    """(common, vlpso_only) settings; common is applied IDENTICALLY to BPSO and VLPSO."""
    s = dict(r2["sel"]["swarm"])
    common = {k: s[k] for k in ["population_size", "max_iter", "inertia_start", "inertia_end",
                                "c1", "c2", "gamma_accuracy", "length_penalty",
                                "interpretability_weight", "k_neighbors", "fitness_cv_splits",
                                "fitness_subsample", "convergence_patience"]}
    common.update(random_state=int(seed), require_groups=True)
    return common, dict(r2["sel"]["vlpso_only"]) | {"min_length": s["min_features"],
                                                     "max_length": s["max_features"]}


def make_vlpso(r2, seed, **over):
    from ..selection.vlpso import VLPSOSelector

    common, vonly = swarm_kwargs(r2, seed)
    kw = common | vonly
    kw.update(over)
    return VLPSOSelector(**kw)


def make_bpso(r2, seed, **over):
    from ..selection.bpso import BPSOSelector

    common, _ = swarm_kwargs(r2, seed)
    s = r2["sel"]["swarm"]
    kw = common | {"min_features": s["min_features"], "max_cardinality": s["max_features"]}
    kw.update(over)
    return BPSOSelector(**kw)


def build_selector(method: str, r2: Dict[str, Any], seed: int):
    from ..selection.base import IdentitySelector
    from ..selection.embedded import BorutaSelector, L1PathRanker, TreeImportanceRanker
    from ..selection.filters import (Chi2Filter, MutualInfoFilter, ReliefFFilter,
                                     SymmetricUncertaintyFilter)
    from ..selection.tuned import TunedTopKSelector
    from ..selection.wrappers import RFERanker, SFSSelector

    s = r2["sel"]
    tuned = lambda ranker: TunedTopKSelector(ranker=ranker, k_grid=s["k_grid"],
                                             inner_splits=s["downstream"]["inner_splits"],
                                             require_groups=True, random_state=seed)
    big = 10_000                                  # rankers expose all scores; k is tuned
    if method == "none":
        return IdentitySelector()
    if method == "symmetric_uncertainty":
        return tuned(SymmetricUncertaintyFilter(k=big))
    if method == "mutual_info":
        return tuned(MutualInfoFilter(k=big, random_state=seed))
    if method == "chi2":
        return tuned(Chi2Filter(k=big))
    if method == "relieff":
        return tuned(ReliefFFilter(k=big, random_state=seed, use_skrebate=False))
    if method == "rfe":
        return tuned(RFERanker(random_state=seed))
    if method == "l1_logistic":
        return tuned(L1PathRanker(random_state=seed))
    if method == "tree_importance":
        return tuned(TreeImportanceRanker(random_state=seed))
    if method == "sfs":
        return SFSSelector(**s["sfs"], require_groups=True, random_state=seed)
    if method == "boruta":
        return BorutaSelector(**s["boruta"], random_state=seed)
    if method == "bpso":
        return make_bpso(r2, seed)
    if method == "vlpso":
        return make_vlpso(r2, seed)
    if method == "vlpso_nopenalty":
        return make_vlpso(r2, seed, length_penalty=0.0)
    if method == "vlpso_shrink_only":
        return make_vlpso(r2, seed, length_direction="shrink_only")
    if method == "vlpso_random_init":
        return make_vlpso(r2, seed, init="random")
    if method == "vlpso_mi":
        return make_vlpso(r2, seed, ranking="mutual_info")
    raise KeyError(f"unknown selection method {method!r}")


def _convergence(sel) -> Optional[str]:
    c = getattr(sel, "convergence_", None)
    if c is None:
        return None
    d = c.__dict__ if hasattr(c, "__dict__") and not isinstance(c, dict) else c
    return json.dumps({k: list(map(float, v)) for k, v in d.items()})


def _select_and_score(sel, X, y, g, tr, te, seed, r2) -> Dict[str, Any]:
    """Fit preprocessing and the selector on the outer-training part, tune C, score once."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    pre = preprocessor().fit(X.iloc[tr])
    Xtr, Xte = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te])
    t0 = time.perf_counter()
    sel.fit(Xtr, y[tr], groups=g[tr])
    sel_seconds = time.perf_counter() - t0
    cols = list(sel.selected_feature_names_)
    d = r2["sel"]["downstream"]
    sp = inner_splits(Xtr, y[tr], g[tr], seed, d["inner_splits"])
    C, cv = tune_lr(Xtr[cols], y[tr], sp, d["C_grid"])
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Xtr[cols].to_numpy(), y[tr])
    auc = float(roc_auc_score(y[te], m.decision_function(Xte[cols].to_numpy())))
    return {
        "auc": auc, "C": C, "inner_auc_by_C": json.dumps(cv),
        "n_selected": int(len(cols)), "n_items": int(sum(not c.startswith("missingindicator_") for c in cols)),
        "selected": json.dumps(cols),
        "selected_items": json.dumps([c for c in cols if not c.startswith("missingindicator_")]),
        "n_columns": int(Xtr.shape[1]), "k_chosen": getattr(sel, "k_", None),
        "selector_seconds": round(sel_seconds, 2),
        "n_evaluations": getattr(sel, "n_evaluations_", None),
        "fitness_splitter": getattr(sel, "fitness_splitter_", getattr(sel, "inner_splitter_", None)),
        "gbest_fitness": getattr(sel, "gbest_fitness_", None),
        "stop_iteration": getattr(sel, "stop_iteration_", None),
        "convergence": _convergence(sel),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_schools_train": int(len(np.unique(g[tr]))), "n_schools_test": int(len(np.unique(g[te]))),
    }


# ---------------------------------------------------------------------------
# sel
# ---------------------------------------------------------------------------
def run_sel(row, fr: Frames, r2) -> Tuple[Tables, Dict]:
    X, y, g, _ = task_view(fr, row["task"], int(row["pv"]))
    tr, te = outer_split(fr, X, y, g, int(row["rep"]), int(row["fold"]), r2["sel"]["outer_splits"])
    sel = build_selector(row["method"], r2, int(row["seed"]))
    res = _select_and_score(sel, X, y, g, tr, te, fr.seed + int(row["rep"]), r2)
    out = {k: row[k] for k in ("task", "pv", "rep", "fold", "method")} | res
    return {"result": pd.DataFrame([out])}, {"auc": res["auc"], "n_selected": res["n_selected"]}


# ---------------------------------------------------------------------------
# vlstab
# ---------------------------------------------------------------------------
def run_vlstab(row, fr: Frames, r2) -> Tuple[Tables, Dict]:
    v = r2["vlstab"]
    X, y, g, _ = task_view(fr, v["task"], int(v["pv"]))
    tr, te = outer_split(fr, X, y, g, int(v["repeat"]), int(row["fold"]), r2["sel"]["outer_splits"])
    variant, sseed = row["variant"], int(row["swarm_seed"])
    if variant == "seed_vlpso":
        sel = make_vlpso(r2, sseed)
    elif variant == "seed_bpso":
        sel = make_bpso(r2, sseed)
    else:
        sel = make_vlpso(r2, sseed, **v["variants"][variant])
    res = _select_and_score(sel, X, y, g, tr, te, fr.seed + int(v["repeat"]), r2)
    out = {"variant": variant, "fold": int(row["fold"]), "swarm_seed": sseed} | res
    return {"result": pd.DataFrame([out])}, {"auc": res["auc"], "n_selected": res["n_selected"]}


# ---------------------------------------------------------------------------
# perm
# ---------------------------------------------------------------------------
def run_perm(row, fr: Frames, r2) -> Tuple[Tables, Dict]:
    from ..evaluation.nested_cv import NestedCVConfig, run_nested_cv
    from ..evaluation.permutation import permute_within_groups
    from ..models.registry import get_models

    p = r2["perm"]
    X, y, g, _ = task_view(fr, p["task"], int(p["pv"]))
    null, draw = row["null"], int(row["draw"])
    if null == "observed":
        yp, seed = y, None
    elif null == "unrestricted":
        seed = p["seed_offset_unrestricted"] + draw
        yp = np.random.default_rng(seed).permutation(y)
    elif null == "within_school":
        seed = p["seed_offset_within"] + draw
        yp = permute_within_groups(y, g, np.random.default_rng(seed))
    else:
        raise KeyError(null)
    res = run_nested_cv(
        X, yp, g, models=get_models(["GradientBoosting", "RandomForest", "LogisticRegression_L2"]),
        cfg=NestedCVConfig(outer_splits=p["outer_splits"], outer_repeats=1,
                           inner_splits=p["inner_splits"], random_state=fr.seed,
                           checkpoint_dir=None, verbose=False),
        task=p["task"], pv=int(p["pv"]), method=f"perm_{null}", resume=False)
    folds = res[["fold", "auc", "n_test", "best_model"]].copy()
    auc = fold_weighted(folds["auc"], folds["n_test"])
    return ({"folds": folds.assign(null=null, draw=draw, perm_seed=seed)},
            {"null": null, "draw": draw, "perm_seed": seed, "auc_fold_weighted": auc})


# ---------------------------------------------------------------------------
# shap (+ LIME)
# ---------------------------------------------------------------------------
def inner_model_selection(Xtr, ytr, gtr, seed: int, n_inner: int = 5):
    """The headline inner loop: grid search over GB, RF, LR-L2 on school-grouped folds."""
    from sklearn.model_selection import GridSearchCV

    from ..models.pipeline import make_pipeline
    from ..models.registry import get_models

    sp = inner_splits(Xtr, ytr, gtr, seed, n_inner)
    best = (None, -np.inf, None, None)
    per_model = {}
    for name, spec in get_models(["GradientBoosting", "RandomForest", "LogisticRegression_L2"]).items():
        gs = GridSearchCV(make_pipeline(spec["model"], None), spec["params"], scoring="roc_auc",
                          cv=sp, n_jobs=1, refit=True)
        gs.fit(Xtr, ytr)
        per_model[name] = float(gs.best_score_)
        if gs.best_score_ > best[1]:
            best = (name, float(gs.best_score_), gs.best_estimator_, gs.best_params_)
    return best, per_model


def run_shap(row, fr: Frames, r2) -> Tuple[Tables, Dict]:
    from sklearn.metrics import roc_auc_score

    from ..explain.lime_local import lime_seed_stability
    from ..explain.shap_global import global_shap, transformed_frames

    s = r2["shap"]
    rep, fold, pv, task = int(s["repeat"]), int(row["fold"]), int(row["pv"]), row["task"]
    X, y, g, rows = task_view(fr, task, pv)
    tr, te = outer_split(fr, X, y, g, rep, fold, 5)
    (name, inner, est, params), per_model = inner_model_selection(
        X.iloc[tr], y[tr], g[tr], fr.seed + rep)
    score = est.predict_proba(X.iloc[te])[:, 1]
    auc = float(roc_auc_score(y[te], score))
    Xtr_t, Xte_t = transformed_frames(est, X.iloc[tr], X.iloc[te])
    clf = est.named_steps["clf"]
    gs = global_shap(clf, Xtr_t, Xte_t, y[te], model_name=name, n_instances=s["n_instances"],
                     background_size=s["background_size"], background_method=s["background_method"],
                     n_bootstrap=s["n_bootstrap"], seed=int(row["seed"]))
    imp = gs["importance"].assign(task=task, pv=pv, fold=fold)
    vals, eidx = gs["shap_values"], gs["explained_index"]
    p_e = clf.predict_proba(Xte_t.iloc[eidx])[:, 1]
    loc = []
    chosen = []
    for q in s["local_quantiles"]:
        i = int(np.argmin(np.abs(p_e - np.quantile(p_e, q))))
        chosen.append(i)
        for j, f in enumerate(Xte_t.columns):
            loc.append({"task": task, "pv": pv, "fold": fold, "quantile": q,
                        "predicted_probability": float(p_e[i]), "feature": f,
                        "feature_value": float(Xte_t.iloc[eidx[i], j]), "shap": float(vals[i, j])})
    tables: Tables = {
        "global": imp, "local": pd.DataFrame(loc),
        "preds": pd.DataFrame({"row": rows[te], "y_true": y[te], "y_score": score, "group": g[te]}),
        "spec": pd.DataFrame([gs["spec"].as_row()]),
    }
    if pv in s["lime_pvs"]:
        lime = lime_seed_stability(clf.predict_proba, Xtr_t, Xte_t.iloc[[eidx[i] for i in chosen]],
                                   n_repeats=s["lime_repeats"], n_samples=s["lime_samples"],
                                   n_features=s["lime_features"], seed=int(row["seed"]))
        tables["lime_features"] = lime["features"].assign(task=task, pv=pv, fold=fold)
        tables["lime_instances"] = lime["instances"].assign(
            task=task, pv=pv, fold=fold, quantile=s["local_quantiles"][: len(chosen)])
    meta = {"task": task, "pv": pv, "fold": fold, "repeat": rep, "auc": auc,
            "best_model": name, "best_params": {k: str(v) for k, v in params.items()},
            "inner_scores": per_model, "n_train": int(len(tr)), "n_test": int(len(te)),
            "shap_additivity_max_abs_error": gs["additivity_max_abs_error"],
            "shap_explainer": gs["spec"].explainer_class}
    return tables, meta


# ---------------------------------------------------------------------------
# ext
# ---------------------------------------------------------------------------
def cluster_bootstrap_auc(y, s, groups, n_boot: int, seed: int, w=None) -> np.ndarray:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    pos = {gg: np.flatnonzero(groups == gg) for gg in ug}
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = np.concatenate([pos[gg] for gg in rng.choice(ug, len(ug), replace=True)])
        if len(np.unique(y[idx])) == 2:
            out[b] = roc_auc_score(y[idx], s[idx], sample_weight=None if w is None else w[idx])
    return out


def run_ext(row, fr: Frames, r2) -> Tuple[Tables, Dict]:
    from sklearn.metrics import roc_auc_score

    e = r2["ext"]
    task, pv = row["task"], int(row["pv"])
    X, y, g, _ = task_view(fr, task, pv)
    Xp, yp, gp, rows_p = task_view(fr, task, pv, external=True)
    assert not set(g) & set(gp)
    (name, inner, est, params), per_model = inner_model_selection(X, y, g, fr.seed, e["inner_splits"])
    s = est.predict_proba(Xp)[:, 1]
    w = fr.ext["W_FSTUWT"].to_numpy()[rows_p]
    auc = float(roc_auc_score(yp, s))
    auc_w = float(roc_auc_score(yp, s, sample_weight=w))
    boot = cluster_bootstrap_auc(yp, s, gp, e["n_bootstrap"], int(row["seed"]))
    bw = cluster_bootstrap_auc(yp, s, gp, e["n_bootstrap"], int(row["seed"]) + 1, w)
    meta = {"task": task, "pv": pv, "auc": auc, "auc_weighted": auc_w,
            "boot_var": float(np.nanvar(boot, ddof=1)), "boot_var_weighted": float(np.nanvar(bw, ddof=1)),
            "best_model": name, "inner_auc": inner, "inner_scores": per_model,
            "best_params": {k: str(v) for k, v in params.items()},
            "n_train": int(len(y)), "n_schools_train": int(len(np.unique(g))),
            "n_external": int(len(yp)), "n_schools_external": int(len(np.unique(gp))),
            "prevalence_train": float(y.mean()), "prevalence_external": float(yp.mean())}
    return {"preds": pd.DataFrame({"row": rows_p, "y_true": yp, "y_score": s, "group": gp})}, meta


# ---------------------------------------------------------------------------
# brr
# ---------------------------------------------------------------------------
def _weighted_auc(y, s, w):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s, sample_weight=w))


def _r3_folds(r3_dir: Path, task: str, pv: int) -> Dict[Tuple[int, int], pd.DataFrame]:
    files = sorted(Path(r3_dir).glob(f"fold_task-{task}_pv-{pv}_method-none_rep-*_fold-*_cfg-*_preds.parquet"))
    fps = {f.name.split("_cfg-")[1].split("_")[0] for f in files}
    if len(fps) != 1:
        raise RuntimeError(f"{task} pv{pv}: expected one configuration fingerprint, found {sorted(fps)}")
    out = {}
    for f in files:
        rep = int(f.name.split("_rep-")[1].split("_")[0])
        fold = int(f.name.split("_fold-")[1].split("_")[0])
        out[(rep, fold)] = pd.read_parquet(f)
    return out


def run_brr(row, fr: Frames, r2, r3_dir: Optional[str] = None, cells_dir: Optional[str] = None
            ) -> Tuple[Tables, Dict]:
    """BRR on stored predictions. Models are NOT refitted per replicate: the
    replicate weights re-weight the evaluation sample, which is the standard
    design-based treatment of a fixed prediction rule (state this in the paper).
    """
    from sklearn.metrics import roc_auc_score

    b = r2["brr"]
    task, pv = row["task"], int(row["pv"])
    X, y, g, rows = task_view(fr, task, pv)
    w = fr.prim["W_FSTUWT"].to_numpy()[rows]
    R = fr.prim[[f"W_FSTURWT{r}" for r in range(1, b["n_replicates"] + 1)]].to_numpy()[rows]

    folds: Dict[Tuple[int, int], np.ndarray] = {}
    source = None
    if r3_dir and Path(r3_dir).exists():
        stored = _r3_folds(Path(r3_dir), task, pv)
        for (rep, fold), d in stored.items():
            _, te = outer_split(fr, X, y, g, rep, fold, 5)
            if not (np.array_equal(d["y_true"].to_numpy(), y[te])
                    and np.array_equal(d["group"].to_numpy(), g[te])):
                raise RuntimeError(f"stored predictions for {task} pv{pv} rep{rep} fold{fold} do not "
                                   "align with the reconstructed outer fold; refusing to attach weights")
            if "sample_weight" in d and np.isfinite(d["sample_weight"]).all():
                assert np.allclose(d["sample_weight"].to_numpy(), w[te])
            folds[(rep, fold)] = (te, d["y_score"].to_numpy())
        source = "results-3 (all repeats)"
    elif cells_dir:
        for f in sorted(Path(cells_dir).glob(f"shap__fold-*__pv-{pv}__task-{task}__*/preds.parquet")):
            d = pd.read_parquet(f)
            fold = int(f.parent.name.split("fold-")[1].split("__")[0])
            _, te = outer_split(fr, X, y, g, 0, fold, 5)
            assert np.array_equal(d["y_true"].to_numpy(), y[te])
            folds[(0, fold)] = (te, d["y_score"].to_numpy())
        source = "round-2 shap cells (repeat 0)"
    if not folds:
        raise RuntimeError("no prediction source for BRR")

    reps = sorted({r for r, _ in folds})
    est_u, est_w, est_r = [], [], []
    for rep in reps:
        items = [folds[(rep, f)] for f in sorted(f for r_, f in folds if r_ == rep)]
        ns = [len(te) for te, _ in items]
        est_u.append(fold_weighted([roc_auc_score(y[te], s) for te, s in items], ns))
        est_w.append(fold_weighted([_weighted_auc(y[te], s, w[te]) for te, s in items], ns))
        est_r.append([fold_weighted([_weighted_auc(y[te], s, R[te, r]) for te, s in items], ns)
                      for r in range(R.shape[1])])
    theta_w = float(np.mean(est_w))
    theta_r = np.mean(np.asarray(est_r), axis=0)
    var = float(np.sum((theta_r - theta_w) ** 2) / (len(theta_r) * (1 - b["fay_k"]) ** 2))
    out = {"task": task, "pv": pv, "source": source, "n_repeats": len(reps),
           "n_folds": len(folds), "auc_unweighted": float(np.mean(est_u)),
           "auc_weighted": theta_w, "brr_var": var, "brr_se": float(np.sqrt(var))}
    return ({"summary": pd.DataFrame([out]),
             "replicates": pd.DataFrame({"replicate": np.arange(1, len(theta_r) + 1),
                                         "auc_weighted": theta_r})}, out)


# ---------------------------------------------------------------------------
# eda (R1.1: describe the dataset, the predictors and the target)
# ---------------------------------------------------------------------------
def run_eda(row, fr: Frames, r2) -> Tuple[Tables, Dict]:
    """Aggregate descriptives only; no student-level row leaves this function."""
    from ..data.design import SurveyDesign, brr_standard_error, weighted_mean

    d, X = fr.prim, fr.X
    w = d["W_FSTUWT"].to_numpy(float)
    items = []
    for c in X.columns:
        x = pd.to_numeric(X[c], errors="coerce").to_numpy(float)
        ok = np.isfinite(x)
        vc = pd.Series(x[ok]).value_counts(normalize=True).sort_index()
        items.append({"item": c, "n_valid": int(ok.sum()), "pct_missing": float(100 * (~ok).mean()),
                      "n_levels": int(vc.size), "min": float(np.nanmin(x)), "max": float(np.nanmax(x)),
                      "mean": float(np.nanmean(x)), "sd": float(np.nanstd(x, ddof=1)),
                      "weighted_mean": float(weighted_mean(x[ok], w[ok])),
                      "distribution": json.dumps({str(k): round(float(v), 4) for k, v in vc.items()})})
    pvs = []
    for i in range(1, 11):
        v = d[f"PV{i}MATH"].to_numpy(float)
        pvs.append({"pv": i, "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "weighted_mean": float(weighted_mean(v, w)),
                    "q25": float(np.quantile(v, 0.25)), "median": float(np.median(v)),
                    "q75": float(np.quantile(v, 0.75))})
    tasks = []
    for t in ("low_vs_high", "low_vs_medium", "medium_vs_high"):
        for pv in range(1, 11):
            _, y, g, rows = task_view(fr, t, pv)
            tasks.append({"task": t, "pv": pv, "n": int(len(y)), "n_schools": int(len(np.unique(g))),
                          "n_positive": int(y.sum()), "prevalence": float(y.mean()),
                          "weighted_prevalence": float(weighted_mean(y.astype(float), w[rows]))})
    cats = []
    for pv in range(1, 11):
        cat = d[f"cat_pv{pv}"].astype(str)
        for c in ("Low", "Medium", "High"):
            m = (cat == c).to_numpy()
            cats.append({"pv": pv, "category": c, "n": int(m.sum()), "share": float(m.mean()),
                         "weighted_share": float(w[m].sum() / w.sum())})
    sch = d.groupby("CNTSCHID").size()
    meta = {"n_students": int(len(d)), "n_schools": int(d.CNTSCHID.nunique()),
            "students_per_school_mean": float(sch.mean()), "students_per_school_min": int(sch.min()),
            "students_per_school_max": int(sch.max()), "n_columns_design": int(X.shape[1]),
            "n_external": int(len(fr.ext)), "n_schools_external": int(fr.ext.CNTSCHID.nunique())}
    return ({"items": pd.DataFrame(items), "pv_distribution": pd.DataFrame(pvs),
             "task_sizes": pd.DataFrame(tasks), "categories": pd.DataFrame(cats)}, meta)
