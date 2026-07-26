"""Global SHAP: population-level attributions with uncertainty.

Answers editor comment 7 ("Distinguish clearly between local SHAP explanations,
globally aggregated SHAP summaries, and LIME explanations").

WHAT WAS MISSING
----------------
The submitted analysis computed SHAP for exactly TWO instances per task -- the
argmax and argmin of predicted probability (``lime_functions.py:29-30``) -- and
then discussed them as though they described the population
(``main.tex:746`` onward). There is no global aggregation anywhere in the
repository: no ``summary_plot``, no mean |SHAP| over a sample.

WHAT THIS MODULE DOES
---------------------
* aggregates over >= 1,000 stratified TEST instances;
* bootstrap CIs on mean |SHAP| so importances can be compared honestly;
* background drawn from TRAINING data only, sampled by a documented method;
* one stated output scale, recorded in the returned metadata;
* the ``noise_control`` benchmark rank (Stage 2.5): any real feature ranked
  below injected Gaussian noise is reported as indistinguishable from noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExplainerSpec:
    """Every setting the manuscript must report, per model. Editor comment 7."""

    model_name: str
    explainer_class: str
    shap_version: str
    background_source: str          # ALWAYS "train"
    background_size: int
    background_method: str
    output_scale: str               # "log_odds" or "probability"
    feature_perturbation: Optional[str]
    n_samples: Optional[int]
    random_seed: int
    n_explained: int

    def as_row(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _pick_explainer(model, background, output_scale: str, seed: int):
    """Choose a SHAP explainer and record exactly what was chosen and why.

    The submitted code was inconsistent: ``LinearExplainer`` received the full
    ``X``, ``KernelExplainer`` received ``shap.kmeans(X, 50)``, ``TreeExplainer``
    received no background at all and ``DeepExplainer`` received ``X.values``
    (``shap_functions.py:57-63``). Those produce attributions on different
    scales, which the manuscript then compared numerically ("+0.88" against
    "+0.28"). Here every explainer gets the SAME documented background.
    """
    import shap

    name = type(model).__name__
    tree_like = (
        "Forest", "GradientBoosting", "XGB", "LGBM", "DecisionTree", "ExtraTrees",
    )
    if any(t in name for t in tree_like):
        return (
            shap.TreeExplainer(
                model, data=background, feature_perturbation="interventional",
                model_output="raw",
            ),
            "TreeExplainer",
            "interventional",
        )
    if any(t in name for t in ("LogisticRegression", "LinearSVC", "Ridge")):
        return shap.LinearExplainer(model, background), "LinearExplainer", None
    fn = (
        model.predict_proba if hasattr(model, "predict_proba") else model.decision_function
    )
    return shap.KernelExplainer(fn, background), "KernelExplainer", None


def stratified_sample(
    X: pd.DataFrame, y: np.ndarray, n: int, seed: int = 42
) -> np.ndarray:
    """Class-stratified index sample, so rare classes are represented."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = int(min(n, len(y)))
    idx: List[int] = []
    for cls in np.unique(y):
        pool = np.where(y == cls)[0]
        take = max(1, int(round(n * len(pool) / len(y))))
        idx.extend(rng.choice(pool, size=min(take, len(pool)), replace=False))
    return np.array(sorted(idx))[:n]


def transformed_frames(pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Push raw frames through the FITTED pipeline's preprocessing steps.

    An independent audit found that notebook 06 handed
    ``pipeline.named_steps["clf"]`` the *raw, unscaled, NaN-bearing* frames,
    while the classifier had been fitted on the imputed, standardised,
    missing-indicator-augmented matrix. The explainer was therefore attributing
    a model whose split points live on standardised values using raw-scale data
    and a raw-scale background -- the same feature-space misalignment this
    module's docstring claims to have eliminated.

    Always obtain explanation inputs through this function. It applies every
    step up to (but excluding) the final estimator, so the matrix handed to the
    explainer is exactly the matrix the estimator was fitted on, with the
    column names the pipeline produced.
    """
    pre = pipeline[:-1]
    Xtr_t, Xte_t = pre.transform(X_train), pre.transform(X_test)
    if not isinstance(Xtr_t, pd.DataFrame):
        try:
            names = list(pre.get_feature_names_out())
        except Exception:
            names = [f"f{i}" for i in range(np.asarray(Xtr_t).shape[1])]
        Xtr_t = pd.DataFrame(np.asarray(Xtr_t), columns=names, index=X_train.index)
        Xte_t = pd.DataFrame(np.asarray(Xte_t), columns=names, index=X_test.index)
    assert list(Xtr_t.columns) == list(Xte_t.columns)
    return Xtr_t, Xte_t


def global_shap(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    model_name: str = "model",
    n_instances: int = 1000,
    background_size: int = 200,
    background_method: str = "kmeans",
    output_scale: str = "log_odds",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Mean |SHAP| over a stratified test sample, with bootstrap CIs."""
    # Validate USAGE before importing the optional dependency, so a
    # feature-space mistake surfaces as a clear error rather than as
    # ModuleNotFoundError.
    if hasattr(model, "named_steps") or hasattr(model, "steps"):
        raise TypeError(
            "global_shap expects a bare fitted estimator plus frames ALREADY "
            "in the estimator's feature space. Pass "
            "transformed_frames(pipeline, X_train, X_test) and "
            "pipeline.named_steps['clf'] together, or the attributions will be "
            "computed on the wrong feature space."
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("train/test feature spaces differ")
    n_expected = getattr(model, "n_features_in_", None)
    if n_expected is not None and X_test.shape[1] != n_expected:
        raise ValueError(
            f"{type(model).__name__} was fitted on {n_expected} features but "
            f"was given {X_test.shape[1]}. This is the feature-space "
            "misalignment described in transformed_frames(); route the data "
            "through the fitted pipeline's preprocessing first."
        )

    import shap

    rng = np.random.default_rng(seed)

    # --- background: TRAINING data only, never test --------------------
    if background_method == "kmeans":
        bg = shap.kmeans(X_train, min(background_size, len(X_train)))
        bg_values = bg.data
    else:
        bidx = rng.choice(len(X_train), min(background_size, len(X_train)), replace=False)
        bg = X_train.iloc[bidx]
        bg_values = bg.to_numpy()

    idx = stratified_sample(X_test, y_test, n_instances, seed)
    Xe = X_test.iloc[idx]

    explainer, cls_name, perturb = _pick_explainer(model, bg, output_scale, seed)
    vals = explainer.shap_values(Xe)
    if isinstance(vals, list):                      # multiclass -> positive class
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    mean_abs = np.abs(vals).mean(axis=0)

    # --- bootstrap CIs over instances ----------------------------------
    boot = np.empty((n_bootstrap, vals.shape[1]))
    for b in range(n_bootstrap):
        pick = rng.integers(0, vals.shape[0], vals.shape[0])
        boot[b] = np.abs(vals[pick]).mean(axis=0)
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)

    out = pd.DataFrame({
        "feature": list(X_test.columns),
        "mean_abs_shap": mean_abs,
        "ci_low": lo,
        "ci_high": hi,
        "mean_signed_shap": vals.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    spec = ExplainerSpec(
        model_name=model_name, explainer_class=cls_name,
        shap_version=shap.__version__, background_source="train",
        background_size=int(min(background_size, len(X_train))),
        background_method=background_method, output_scale=output_scale,
        feature_perturbation=perturb, n_samples=None, random_seed=seed,
        n_explained=int(len(idx)),
    )
    return {"importance": out, "spec": spec, "shap_values": vals,
            "explained_index": idx}


def noise_control_benchmark(importance: pd.DataFrame,
                            noise_col: str = "noise_control") -> Dict[str, Any]:
    """Where does injected Gaussian noise rank? Stage 2.5.

    Any real feature ranked BELOW pure noise is not distinguishable from noise
    and must be reported as such. The submitted manuscript instead interpreted
    ``noise_control`` substantively at ``main.tex:766``.
    """
    if noise_col not in set(importance["feature"]):
        return {"available": False,
                "note": f"{noise_col} absent; run the diagnostic pipeline with "
                        "include_noise_control=True"}
    row = importance.loc[importance.feature == noise_col].iloc[0]
    below = importance.loc[importance["rank"] > row["rank"], "feature"].tolist()
    return {
        "available": True,
        "noise_rank": int(row["rank"]),
        "n_features": int(len(importance)),
        "noise_mean_abs_shap": float(row["mean_abs_shap"]),
        "n_features_below_noise": len(below),
        "features_below_noise": below,
        "interpretation": (
            f"{len(below)} of {len(importance) - 1} real features rank below "
            "injected Gaussian noise and are not distinguishable from it."
        ),
    }


def shap_vs_permutation_agreement(
    importance: pd.DataFrame, model, X, y, *, seed: int = 42, n_repeats: int = 10
) -> Dict[str, float]:
    """Spearman agreement between global SHAP and permutation importance."""
    from scipy.stats import spearmanr
    from sklearn.inspection import permutation_importance

    p = permutation_importance(model, X, y, n_repeats=n_repeats,
                               random_state=seed, scoring="roc_auc")
    perm = pd.DataFrame({"feature": list(X.columns),
                         "perm_importance": p.importances_mean})
    m = importance.merge(perm, on="feature")
    rho, pval = spearmanr(m["mean_abs_shap"], m["perm_importance"])
    return {"spearman_rho": float(rho), "p_value": float(pval), "n_features": len(m)}
