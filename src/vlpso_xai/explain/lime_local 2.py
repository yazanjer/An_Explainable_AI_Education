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
