"""Tables from completed round-2 cells. Nothing here refits a model.

Every function reads the cells that EXIST and reports how many it used, so a
partially completed run produces correctly labelled tables rather than
silently treating a subset as the full design.

Combination across plausible values always uses Rubin's rules; across matched
outer folds within a plausible value it uses the Nadeau-Bengio corrected
variance, because repeated k-fold training sets overlap.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _cells(results: Path, kind: str, table: str) -> pd.DataFrame:
    d = Path(results) / "cells" / kind
    frames = []
    for sub in sorted(d.glob("*")) if d.exists() else []:
        f = sub / f"{table}.parquet"
        if (sub / "meta.json").exists() and f.exists():
            frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _metas(results: Path, kind: str) -> pd.DataFrame:
    d = Path(results) / "cells" / kind
    rows = []
    for sub in sorted(d.glob("*")) if d.exists() else []:
        m = sub / "meta.json"
        if m.exists():
            rows.append(json.loads(m.read_text()))
    return pd.DataFrame(rows)


def _rubin(est, var):
    from ..data.outcome import rubin_combine

    r = rubin_combine(list(est), list(var))
    return {"estimate": r.estimate, "se": r.standard_error, "ci_low": r.ci_low,
            "ci_high": r.ci_high, "within_var": r.within_variance,
            "between_var": r.between_variance, "fmi": r.fraction_missing_information,
            "df": r.df, "m": r.m}


def _nb_var(values: np.ndarray, n_train: float, n_test: float) -> float:
    v = np.asarray(values, float)
    return float((1.0 / v.size + n_test / n_train) * v.var(ddof=1)) if v.size > 1 else float("nan")


def _holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return adj


# ---------------------------------------------------------------------------
# selector comparison
# ---------------------------------------------------------------------------
def selector_tables(results: Path, r2: Dict, all_features: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    from ..evaluation.stability import nogueira, pairwise_stability

    df = _cells(results, "sel", "result")
    if df.empty:
        return {}
    df["selected_items_l"] = df["selected_items"].map(json.loads)
    items = all_features or sorted({c for s in df["selected_items_l"] for c in s}
                                   | {c for s in df["selected"].map(json.loads) for c in s
                                      if not c.startswith("missingindicator_")})
    n_folds_design = len(r2["sel"]["repeats"]) * len(r2["sel"]["folds"])

    # Completeness: a (task, pv) enters only if every method has every fold, so
    # all contrasts are on identical matched folds.
    methods = r2["sel"]["methods"]
    cnt = df.groupby(["task", "pv"]).apply(
        lambda s: s.groupby("method").size().reindex(methods).fillna(0).min())
    complete = cnt[cnt == n_folds_design].index
    use = df.set_index(["task", "pv"]).loc[list(complete)].reset_index() if len(complete) else df.iloc[0:0]

    summ, stab_rows = [], []
    for (task, method), s in use.groupby(["task", "method"]):
        per_pv = []
        for pv, sp in s.groupby("pv"):
            per_pv.append({"pv": pv, "mean": sp.auc.mean(),
                           "var": _nb_var(sp.auc.to_numpy(), sp.n_train.mean(), sp.n_test.mean()),
                           "phi": nogueira(list(sp.selected_items_l), items)["nogueira"],
                           "jac": pairwise_stability(list(sp.selected_items_l), len(items))["jaccard_mean"]})
        pp = pd.DataFrame(per_pv)
        rb = _rubin(pp["mean"], pp["var"])
        summ.append({"task": task, "method": method, "n_pv": len(pp), "n_folds": len(s),
                     "auc": rb["estimate"], "ci_low": rb["ci_low"], "ci_high": rb["ci_high"],
                     "fmi": rb["fmi"], "auc_fold_sd": s.auc.std(ddof=1),
                     "n_items_mean": s.n_items.mean(), "n_items_sd": s.n_items.std(ddof=1),
                     "n_items_min": int(s.n_items.min()), "n_items_max": int(s.n_items.max()),
                     "n_columns_mean": s.n_selected.mean(),
                     "nogueira_mean": pp.phi.mean(), "nogueira_min": pp.phi.min(), "nogueira_max": pp.phi.max(),
                     "jaccard_mean": pp.jac.mean(),
                     "k_chosen_median": s.k_chosen.median() if s.k_chosen.notna().any() else np.nan,
                     "selector_seconds_mean": s.selector_seconds.mean(),
                     "n_evaluations_mean": s.n_evaluations.mean() if s.n_evaluations.notna().any() else np.nan,
                     "fitness_splitters": ";".join(sorted(set(map(str, s.fitness_splitter.dropna()))))})
    summary = pd.DataFrame(summ)

    # Paired contrasts against two references, Rubin over PVs, Holm per task x reference.
    con = []
    for ref in ("vlpso", "none"):
        for task, s in use.groupby("task"):
            a = s[s.method == ref].set_index(["pv", "rep", "fold"])
            rows = []
            for method in methods:
                if method == ref:
                    continue
                b = s[s.method == method].set_index(["pv", "rep", "fold"])
                j = a[["auc", "n_train", "n_test"]].join(b[["auc"]], rsuffix="_m", how="inner")
                d = j["auc_m"] - j["auc"]                 # method minus reference
                per = []
                for pv, dp in d.groupby(level="pv"):
                    jj = j.loc[pv]
                    per.append({"mean": dp.mean(), "var": _nb_var(dp.to_numpy(), jj.n_train.mean(), jj.n_test.mean()),
                                "sd": dp.std(ddof=1)})
                per = pd.DataFrame(per)
                rb = _rubin(per["mean"], per["var"])
                from scipy import stats
                t = rb["estimate"] / rb["se"] if rb["se"] > 0 else np.nan
                p = float(2 * stats.t.sf(abs(t), rb["df"])) if np.isfinite(t) else np.nan
                rows.append({"task": task, "reference": ref, "method": method, "n_pv": len(per),
                             "n_pairs": int(d.size), "delta_auc": rb["estimate"],
                             "ci_low": rb["ci_low"], "ci_high": rb["ci_high"], "t": t, "df": rb["df"],
                             "p": p, "d": rb["estimate"] / per["sd"].mean() if per["sd"].mean() > 0 else np.nan})
            r = pd.DataFrame(rows)
            if not r.empty:
                r["p_holm"] = _holm(r["p"].fillna(1.0).to_numpy())
                con.append(r)
    contrasts = pd.concat(con, ignore_index=True) if con else pd.DataFrame()
    if not contrasts.empty:
        thr = r2.get("effect_size_thresholds", {"negligible": 0.2, "small": 0.5, "medium": 0.8})
        contrasts["magnitude"] = contrasts["d"].abs().map(
            lambda x: "negligible" if x < thr["negligible"] else "small" if x < thr["small"]
            else "medium" if x < thr["medium"] else "large")

    # Selection frequency per item (over all folds and PVs of a task).
    freq = []
    for (task, method), s in use.groupby(["task", "method"]):
        n = len(s)
        cnts = pd.Series([c for lst in s.selected_items_l for c in lst]).value_counts()
        for it in items:
            freq.append({"task": task, "method": method, "item": it,
                         "frequency": float(cnts.get(it, 0) / n), "n_folds": n})
    frequency = pd.DataFrame(freq)

    # Convergence of the swarms (median trace per method).
    conv = []
    for (task, method), s in use[use.convergence.notna()].groupby(["task", "method"]):
        traces = [json.loads(c) for c in s.convergence]
        L = max(len(t["best_fitness"]) for t in traces)
        bf = np.full((len(traces), L), np.nan)
        for i, t in enumerate(traces):
            v = np.asarray(t["best_fitness"])
            bf[i, : len(v)] = v
            bf[i, len(v):] = v[-1]                      # stopped: fitness stays at its final value
        stop = np.array([len(t["best_fitness"]) for t in traces])
        for it in range(L):
            conv.append({"task": task, "method": method, "iteration": it,
                         "best_fitness_median": float(np.nanmedian(bf[:, it])),
                         "best_fitness_q25": float(np.nanquantile(bf[:, it], 0.25)),
                         "best_fitness_q75": float(np.nanquantile(bf[:, it], 0.75)),
                         "share_still_running": float(np.mean(stop > it))})
    convergence = pd.DataFrame(conv)

    budget = df.groupby(["task", "pv"]).size().rename("cells").reset_index()
    budget["complete"] = budget.set_index(["task", "pv"]).index.isin(complete)
    return {"sel_summary": summary, "sel_contrasts": contrasts, "sel_frequency": frequency,
            "sel_convergence": convergence, "sel_budget": budget}


def vlstab_tables(results: Path, r2: Dict, items: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    from ..evaluation.stability import nogueira

    df = _cells(results, "vlstab", "result")
    if df.empty:
        return {}
    df["items_l"] = df["selected_items"].map(json.loads)
    items = items or sorted({c for s in df.items_l for c in s})
    seed_rows = []
    for (variant, fold), s in df[df.variant.str.startswith("seed_")].groupby(["variant", "fold"]):
        nz = nogueira(list(s.items_l), items)
        seed_rows.append({"method": variant.replace("seed_", ""), "fold": fold, "n_seeds": len(s),
                          "nogueira_across_seeds": nz["nogueira"],
                          "auc_mean": s.auc.mean(), "auc_sd_across_seeds": s.auc.std(ddof=1),
                          "n_items_mean": s.n_items.mean(), "n_items_min": int(s.n_items.min()),
                          "n_items_max": int(s.n_items.max())})
    seeds = pd.DataFrame(seed_rows)
    sens = []
    base = df[(df.variant == "seed_vlpso") & (df.swarm_seed == 0)].assign(variant="base")
    for variant, s in pd.concat([base, df[~df.variant.str.startswith("seed_")]]).groupby("variant"):
        nz = nogueira(list(s.items_l), items)
        sens.append({"variant": variant, "n_folds": len(s), "auc_mean": s.auc.mean(),
                     "auc_sd": s.auc.std(ddof=1), "n_items_mean": s.n_items.mean(),
                     "n_items_min": int(s.n_items.min()), "n_items_max": int(s.n_items.max()),
                     "nogueira_across_folds": nz["nogueira"],
                     "evaluations_mean": s.n_evaluations.mean(),
                     "stop_iteration_mean": s.stop_iteration.mean(),
                     "selector_seconds_mean": s.selector_seconds.mean()})
    return {"vlstab_seeds": seeds, "vlstab_sensitivity": pd.DataFrame(sens)}


def perm_tables(results: Path, r2: Dict) -> Dict[str, pd.DataFrame]:
    from ..evaluation.permutation import decompose_performance

    m = _metas(results, "perm")
    if m.empty:
        return {}
    obs = m.loc[m.null == "observed", "auc_fold_weighted"]
    U = m.loc[m.null == "unrestricted", "auc_fold_weighted"].to_numpy()
    W = m.loc[m.null == "within_school", "auc_fold_weighted"].to_numpy()
    o = float(obs.iloc[0]) if len(obs) else np.nan
    tol = r2["perm"]["gate_tolerance"]
    rows = []
    for name, v in (("unrestricted", U), ("within_school", W)):
        if v.size:
            rows.append({"null": name, "n_draws": int(v.size), "mean": v.mean(),
                         "sd": v.std(ddof=1) if v.size > 1 else np.nan,
                         "q025": np.quantile(v, 0.025), "q975": np.quantile(v, 0.975),
                         "max": v.max(), "observed": o,
                         "p_value": float((np.sum(v >= o) + 1) / (v.size + 1)) if np.isfinite(o) else np.nan,
                         "gate_abs_dev_from_0.5": abs(v.mean() - 0.5) if name == "unrestricted" else np.nan,
                         "gate_passed": bool(abs(v.mean() - 0.5) <= tol) if name == "unrestricted" else None})
    out = {"perm_summary": pd.DataFrame(rows)}
    if U.size and W.size and np.isfinite(o):
        out["perm_decomposition"] = pd.DataFrame([decompose_performance(o, W.mean(), U.mean())])
    return out


def _kendall_w(rank_matrix: np.ndarray) -> float:
    m, n = rank_matrix.shape
    R = rank_matrix.sum(axis=0)
    S = float(np.sum((R - R.mean()) ** 2))
    return 12 * S / (m ** 2 * (n ** 3 - n))


def shap_tables(results: Path, r2: Dict, r3_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    g = _cells(results, "shap", "global")
    if g.empty:
        return {}
    g["unit"] = g.pv.astype(str) + "_" + g.fold.astype(str)
    out_rows, w_rows = [], []
    for task, s in g.groupby("task"):
        piv = s.pivot_table(index="unit", columns="feature", values="mean_abs_shap")
        ranks = piv.rank(axis=1, ascending=False)
        pv_means = s.groupby(["pv", "feature"]).mean_abs_shap.mean().unstack()
        w_rows.append({"task": task, "n_units": len(piv), "n_pv": s.pv.nunique(),
                       "kendall_w": _kendall_w(ranks.to_numpy()),
                       "kendall_w_items_only": _kendall_w(
                           ranks[[c for c in ranks.columns if not c.startswith("missingindicator_")]]
                           .rank(axis=1).to_numpy())})
        for f in piv.columns:
            out_rows.append({"task": task, "feature": f,
                             "mean_abs_shap": piv[f].mean(),
                             "sd_between_pv": pv_means[f].std(ddof=1),
                             "sd_between_units": piv[f].std(ddof=1),
                             "rank_median": ranks[f].median(),
                             "rank_q25": ranks[f].quantile(0.25), "rank_q75": ranks[f].quantile(0.75),
                             "share_top5": float((ranks[f] <= 5).mean()),
                             "mean_signed_shap": s[s.feature == f].mean_signed_shap.mean()})
    imp = pd.DataFrame(out_rows).sort_values(["task", "mean_abs_shap"], ascending=[True, False])
    imp["rank"] = imp.groupby("task").mean_abs_shap.rank(ascending=False, method="first").astype(int)
    meta = _metas(results, "shap")
    out = {"shap_importance": imp, "shap_agreement": pd.DataFrame(w_rows),
           "shap_fold_models": meta[["task", "pv", "fold", "auc", "best_model", "n_train", "n_test"]]}
    if r3_dir and (Path(r3_dir).parent / "nested_cv_results.parquet").exists():
        r3 = pd.read_parquet(Path(r3_dir).parent / "nested_cv_results.parquet")
        r3 = r3[(r3.method == "none") & (r3.rep == 0)][["task", "pv", "fold", "auc", "best_model"]]
        chk = meta.merge(r3, on=["task", "pv", "fold"], suffixes=("_r2", "_r3"))
        chk["auc_diff"] = chk.auc_r2 - chk.auc_r3
        out["shap_reproduction_check"] = chk[["task", "pv", "fold", "auc_r2", "auc_r3", "auc_diff",
                                               "best_model_r2", "best_model_r3"]]
    lf = _cells(results, "shap", "lime_features")
    li = _cells(results, "shap", "lime_instances")
    if not lf.empty:
        out["lime_instances"] = li
        out["lime_summary"] = (lf.assign(named_by_all=lf.n_seeds_naming == lf.n_seeds)
                               .groupby(["task", "named_by_all"])
                               .agg(n=("feature", "size"), cv_median=("cv_when_named", "median"),
                                    sign_consistency_mean=("sign_consistency", "mean"))
                               .reset_index())
        out["lime_features"] = lf
    return out


def ext_tables(results: Path, r2: Dict) -> Dict[str, pd.DataFrame]:
    m = _metas(results, "ext")
    if m.empty:
        return {}
    rows = []
    for task, s in m.groupby("task"):
        u, w = _rubin(s.auc, s.boot_var), _rubin(s.auc_weighted, s.boot_var_weighted)
        rows.append({"task": task, "n_pv": len(s), "auc": u["estimate"], "ci_low": u["ci_low"],
                     "ci_high": u["ci_high"], "fmi": u["fmi"], "auc_weighted": w["estimate"],
                     "ci_low_weighted": w["ci_low"], "ci_high_weighted": w["ci_high"],
                     "n_external": int(s.n_external.median()),
                     "n_schools_external": int(s.n_schools_external.median()),
                     "models": ";".join(f"{k}:{v}" for k, v in s.best_model.value_counts().items())})
    return {"ext_summary": pd.DataFrame(rows), "ext_per_pv": m.drop(columns=["inner_scores", "best_params"], errors="ignore")}


def brr_tables(results: Path, r2: Dict) -> Dict[str, pd.DataFrame]:
    s = _cells(results, "brr", "summary")
    if s.empty:
        return {}
    rows = []
    for task, t in s.groupby("task"):
        rb = _rubin(t.auc_weighted, t.brr_var)
        rows.append({"task": task, "n_pv": len(t), "source": ";".join(sorted(set(t.source))),
                     "auc_weighted": rb["estimate"], "se": rb["se"], "ci_low": rb["ci_low"],
                     "ci_high": rb["ci_high"], "fmi": rb["fmi"],
                     "within_var_brr": rb["within_var"], "between_var_pv": rb["between_var"],
                     "auc_unweighted_same_predictions": t.auc_unweighted.mean()})
    return {"brr_summary": pd.DataFrame(rows), "brr_per_pv": s}


def aggregate_all(results: Path, r2: Dict, r3_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    out_dir = Path(results) / "r2_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    allt: Dict[str, pd.DataFrame] = {}
    for fn in (lambda: selector_tables(results, r2), lambda: vlstab_tables(results, r2),
               lambda: perm_tables(results, r2), lambda: shap_tables(results, r2, r3_dir),
               lambda: ext_tables(results, r2), lambda: brr_tables(results, r2)):
        allt.update(fn())
    for name, df in allt.items():
        if df is None or df.empty:
            logger.info("skipped %s (no completed cells yet)", name)
            continue
        df.to_csv(out_dir / f"{name}.csv", index=False)
        logger.info("wrote %s (%d rows)", name, len(df))
    return allt
