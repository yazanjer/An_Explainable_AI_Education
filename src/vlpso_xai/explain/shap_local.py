"""Local SHAP: individual instances, with a documented selection rule.

Answers editor comment 7. Kept in a separate module from ``shap_global`` so the
two can never be conflated in a table or a paragraph -- which is precisely what
the submitted manuscript did.

WHAT WAS WRONG
--------------
``lime_functions.py:29-30`` explained exactly two instances per task, the
``argmax`` and ``argmin`` of predicted probability, and the manuscript then
discussed them as though they described the population (``main.tex:746``
onward). Two extreme cases describe two students.

WHAT THIS MODULE DOES
---------------------
* explains a documented, quantile-stratified set of instances;
* records, for every explained instance, its predicted score, true label and
  the reason it was chosen, so a reader can reproduce the selection;
* returns per-instance attributions in long form, never aggregated -- if you
  want a population statement, use ``shap_global``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def select_local_instances(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    n: int = 12,
    strategy: str = "stratified_quantile",
    seed: int = 42,
) -> Dict[str, Any]:
    """Choose instances to explain, and say why.

    ``stratified_quantile`` (default)
        one random instance from each of ``n`` equal-width quantiles of the
        predicted score, so the whole range of model behaviour is represented.
    ``extremes``
        the argmax and argmin. Provided ONLY so the submitted analysis can be
        reproduced for comparison; it is not a defensible sample.
    ``correct_and_errors``
        a balanced mix of confident hits and confident misses, useful for an
        error analysis.
    """
    rng = np.random.default_rng(seed)
    y_score = np.asarray(y_score, dtype=float)
    y_true = np.asarray(y_true).astype(int)

    if strategy == "extremes":
        idx = np.array([int(np.argmin(y_score)), int(np.argmax(y_score))])
        rule = "argmin and argmax of predicted probability (submitted analysis; NOT representative)"
    elif strategy == "correct_and_errors":
        pred = (y_score >= 0.5).astype(int)
        hit = np.where(pred == y_true)[0]
        miss = np.where(pred != y_true)[0]
        k = max(1, n // 2)
        idx = np.concatenate([
            rng.choice(hit, min(k, hit.size), replace=False) if hit.size else np.array([], int),
            rng.choice(miss, min(k, miss.size), replace=False) if miss.size else np.array([], int),
        ]).astype(int)
        rule = f"{k} confident correct and {k} confident incorrect predictions"
    elif strategy == "stratified_quantile":
        q = np.quantile(y_score, np.linspace(0, 1, n + 1))
        picks: List[int] = []
        for i in range(n):
            pool = np.where((y_score >= q[i]) & (y_score <= q[i + 1]))[0]
            if pool.size:
                picks.append(int(rng.choice(pool)))
        idx = np.array(sorted(set(picks)))
        rule = f"one random instance from each of {n} equal-width quantiles of the predicted score"
    else:
        raise ValueError(
            f"strategy must be 'stratified_quantile', 'extremes' or "
            f"'correct_and_errors', got {strategy!r}"
        )

    return {
        "index": idx,
        "rule": rule,
        "strategy": strategy,
        "seed": seed,
        "predicted_score": y_score[idx],
        "true_label": y_true[idx],
    }


def local_shap(
    model,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    *,
    background_size: int = 200,
    background_method: str = "kmeans",
    output_scale: str = "log_odds",
    seed: int = 42,
) -> Dict[str, Any]:
    """Per-instance SHAP attributions, in long form.

    ``X_train`` and ``X_explain`` must already be in the estimator's feature
    space -- use :func:`vlpso_xai.explain.shap_global.transformed_frames`.
    """
    from .shap_global import _pick_explainer

    if hasattr(model, "named_steps") or hasattr(model, "steps"):
        raise TypeError(
            "local_shap expects a bare fitted estimator and frames already in "
            "its feature space. Use transformed_frames(pipeline, ...) and pass "
            "pipeline.named_steps['clf']."
        )
    n_expected = getattr(model, "n_features_in_", None)
    if n_expected is not None and X_explain.shape[1] != n_expected:
        raise ValueError(
            f"{type(model).__name__} was fitted on {n_expected} features but "
            f"was given {X_explain.shape[1]}."
        )

    import shap

    rng = np.random.default_rng(seed)
    if background_method == "kmeans":
        bg = shap.kmeans(X_train, min(background_size, len(X_train)))
    else:
        pick = rng.choice(len(X_train), min(background_size, len(X_train)), replace=False)
        bg = X_train.iloc[pick]

    explainer, cls_name, perturb = _pick_explainer(model, bg, output_scale, seed)
    vals = explainer.shap_values(X_explain)
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    rows = []
    for i in range(vals.shape[0]):
        for j, feat in enumerate(X_explain.columns):
            rows.append({
                "instance": int(i),
                "feature": str(feat),
                "shap_value": float(vals[i, j]),
                "feature_value": float(X_explain.iloc[i, j]),
            })
    long = pd.DataFrame(rows)
    long["abs_shap"] = long["shap_value"].abs()

    return {
        "attributions": long,
        "explainer_class": cls_name,
        "feature_perturbation": perturb,
        "output_scale": output_scale,
        "background_size": int(min(background_size, len(X_train))),
        "background_method": background_method,
        "background_source": "train",
        "seed": seed,
        "n_instances": int(vals.shape[0]),
    }


def top_contributors(attributions: pd.DataFrame, instance: int, k: int = 8,
                     codebook=None) -> pd.DataFrame:
    """The k largest absolute attributions for one instance."""
    sub = (attributions[attributions.instance == instance]
           .nlargest(k, "abs_shap")
           .drop(columns="abs_shap")
           .reset_index(drop=True))
    if codebook is not None:
        sub["codebook_label"] = [codebook.label(f) for f in sub.feature]
    return sub
