"""LIME with seeds and stability. Answers editor comment 7.

WHAT WAS WRONG
--------------
``lime_functions.py:29-30`` explained exactly two instances per task -- the
argmax and argmin of predicted probability -- with no random seed. LIME is
stochastic: it fits a local surrogate to a random perturbation sample, so a
single unseeded run is not reproducible and its feature weights are not stable.
The manuscript quotes those weights to two decimal places.

WHAT THIS MODULE DOES
---------------------
* explains a DOCUMENTED, stratified selection of instances rather than two
  extremes;
* repeats each explanation >= 30 times with different seeds;
* reports the SD of every feature weight and the rank correlation between
  repeats, so the reader can see how reproducible the explanation is.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def select_instances(
    y_score: np.ndarray, y_true: np.ndarray, n: int = 12, seed: int = 42
) -> Dict[str, np.ndarray]:
    """Stratified-by-predicted-quantile instance selection.

    Explicitly NOT argmax/argmin. Two extreme cases describe two students; a
    quantile-stratified sample describes the range of model behaviour, and the
    selection rule is stated so a reader can reproduce it.
    """
    rng = np.random.default_rng(seed)
    q = np.quantile(y_score, np.linspace(0, 1, n + 1))
    idx = []
    for i in range(n):
        pool = np.where((y_score >= q[i]) & (y_score <= q[i + 1]))[0]
        if pool.size:
            idx.append(int(rng.choice(pool)))
    idx = np.array(sorted(set(idx)))
    return {"index": idx, "rule": f"one random instance from each of {n} "
                                  "equal-width quantiles of predicted score",
            "predicted_score": y_score[idx], "true_label": y_true[idx]}


def lime_with_stability(
    model,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    *,
    n_repeats: int = 30,
    n_samples: int = 5000,
    n_features: int = 15,
    class_names: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Repeat each LIME explanation and quantify how much it moves."""
    from lime.lime_tabular import LimeTabularExplainer
    from scipy.stats import spearmanr

    fn = (
        model.predict_proba if hasattr(model, "predict_proba")
        else (lambda z: np.column_stack([-model.decision_function(z),
                                         model.decision_function(z)]))
    )
    rows, stability = [], []
    for pos in range(len(X_explain)):
        x = X_explain.iloc[pos].to_numpy()
        runs = np.zeros((n_repeats, X_train.shape[1]))
        for r in range(n_repeats):
            expl = LimeTabularExplainer(
                X_train.to_numpy(), feature_names=list(X_train.columns),
                class_names=class_names or ["neg", "pos"],
                discretize_continuous=True, random_state=seed + r,
            )
            e = expl.explain_instance(x, fn, num_features=n_features,
                                      num_samples=n_samples)
            for fidx, w in e.as_map()[1]:
                runs[r, fidx] = w
        mean, sd = runs.mean(axis=0), runs.std(axis=0, ddof=1)
        for j, feat in enumerate(X_train.columns):
            rows.append({"instance": pos, "feature": feat,
                         "weight_mean": mean[j], "weight_sd": sd[j],
                         "cv": abs(sd[j] / mean[j]) if mean[j] else np.nan})
        rhos = [
            spearmanr(runs[a], runs[b]).statistic
            for a in range(min(n_repeats, 10)) for b in range(a + 1, min(n_repeats, 10))
        ]
        stability.append({"instance": pos, "n_repeats": n_repeats,
                          "mean_rank_correlation": float(np.nanmean(rhos)),
                          "min_rank_correlation": float(np.nanmin(rhos)),
                          "mean_weight_sd": float(np.nanmean(sd))})
    return {"weights": pd.DataFrame(rows),
            "stability": pd.DataFrame(stability),
            "settings": {"n_repeats": n_repeats, "n_samples": n_samples,
                         "n_features": n_features, "seed": seed,
                         "discretize_continuous": True}}


def lime_seed_stability(
    predict_proba,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    *,
    n_repeats: int = 30,
    n_samples: int = 5000,
    n_features: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Seed stability of LIME, with statistics that are not artefacts of zero-filling.

    The round-1 analysis (notebook 08) recorded a weight of 0 for every seed
    that did not name a feature and then reported that "every feature named by
    fewer than half the seeds has a standard deviation exceeding its own mean".
    That statement is true by construction: for a feature named by m of R seeds
    with similar weights w, the zero-filled mean is (m/R)w and the zero-filled
    SD is about w*sqrt((m/R)(1-m/R)), which exceeds the mean whenever m < R/2.
    It carries no information about LIME.

    Here the two questions are separated:

    * **selection frequency** -- in how many of the R runs the feature appears
      among the top ``n_features`` (the selection instability);
    * **conditional weight dispersion** -- mean, SD and coefficient of
      variation of the weight computed ONLY over the runs that named it (the
      weight instability given selection). Undefined for m < 2.

    Rank agreement between runs is reported twice: Spearman over the union of
    named features with unnamed weights set to 0 (``rho_union``, the round-1
    definition, stated so it can be compared) and Spearman over the features
    named by BOTH runs of a pair (``rho_shared``).
    """
    from lime.lime_tabular import LimeTabularExplainer
    from scipy.stats import spearmanr

    feats = list(X_train.columns)
    rows, inst_rows = [], []
    for pos in range(len(X_explain)):
        x = X_explain.iloc[pos].to_numpy()
        W = np.full((n_repeats, len(feats)), np.nan)
        for r in range(n_repeats):
            expl = LimeTabularExplainer(
                X_train.to_numpy(), feature_names=feats, class_names=["neg", "pos"],
                discretize_continuous=True, mode="classification", random_state=seed + r)
            e = expl.explain_instance(x, predict_proba, num_features=n_features,
                                      num_samples=n_samples)
            for fid, w in e.as_map()[1]:
                W[r, fid] = w
        named = np.isfinite(W)
        m = named.sum(axis=0)
        for j, f in enumerate(feats):
            if m[j] == 0:
                continue
            w = W[named[:, j], j]
            mean_c = float(w.mean())
            sd_c = float(w.std(ddof=1)) if m[j] > 1 else float("nan")
            rows.append({"instance": pos, "feature": f, "n_seeds": n_repeats,
                         "n_seeds_naming": int(m[j]),
                         "selection_frequency": float(m[j] / n_repeats),
                         "mean_weight_when_named": mean_c,
                         "sd_weight_when_named": sd_c,
                         "cv_when_named": (abs(sd_c / mean_c) if mean_c and np.isfinite(sd_c) else float("nan")),
                         "sign_consistency": float(max((w > 0).mean(), (w < 0).mean()))})
        Z = np.nan_to_num(W, nan=0.0)
        used = named.any(axis=0)
        rho_u, rho_s = [], []
        for a in range(n_repeats):
            for b in range(a + 1, n_repeats):
                ru = spearmanr(Z[a, used], Z[b, used]).statistic
                if np.isfinite(ru):
                    rho_u.append(ru)
                both = named[a] & named[b]
                if both.sum() >= 3:
                    rs = spearmanr(W[a, both], W[b, both]).statistic
                    if np.isfinite(rs):
                        rho_s.append(rs)
        inst_rows.append({"instance": pos, "n_distinct_features": int(used.sum()),
                          "n_features_named_by_all": int((m == n_repeats).sum()),
                          "rho_union_mean": float(np.mean(rho_u)) if rho_u else float("nan"),
                          "rho_shared_mean": float(np.mean(rho_s)) if rho_s else float("nan")})
    return {"features": pd.DataFrame(rows), "instances": pd.DataFrame(inst_rows),
            "settings": {"n_repeats": n_repeats, "n_samples": n_samples,
                         "n_features": n_features, "seed": seed,
                         "discretize_continuous": True,
                         "unnamed_weight_convention": "excluded (conditional statistics)"}}
