"""Do SHAP and LIME agree? Answers editor comment 7.

The manuscript presents SHAP and LIME results side by side without ever asking
whether they rank features the same way. If two explanation methods disagree,
neither can be quoted as "the" explanation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def rank_agreement(shap_importance: pd.DataFrame,
                   lime_weights: pd.DataFrame,
                   top_k: int = 10) -> Dict[str, float]:
    """Spearman rho and top-k overlap between global SHAP and mean |LIME|."""
    from scipy.stats import spearmanr

    lime_agg = (
        lime_weights.assign(abs_w=lambda d: d["weight_mean"].abs())
        .groupby("feature", as_index=False)["abs_w"].mean()
    )
    m = shap_importance.merge(lime_agg, on="feature", how="inner")
    if len(m) < 3:
        return {"spearman_rho": float("nan"), "n_features": len(m)}
    rho, p = spearmanr(m["mean_abs_shap"], m["abs_w"])
    top_shap = set(shap_importance.nlargest(top_k, "mean_abs_shap")["feature"])
    top_lime = set(lime_agg.nlargest(top_k, "abs_w")["feature"])
    return {
        "spearman_rho": float(rho), "p_value": float(p),
        "n_features": int(len(m)), "top_k": top_k,
        "top_k_overlap": len(top_shap & top_lime),
        "top_k_jaccard": len(top_shap & top_lime) / len(top_shap | top_lime),
    }


CAUSAL_CAVEAT = """
These are model-explanation results computed on observational survey data.
SHAP and LIME describe how a fitted model uses a variable; they do not show
that the variable causes achievement, that changing it would change outcomes,
or that features appearing 'early' in a decision plot exert path-dependent
effects. A SHAP decision plot orders features by attribution magnitude and has
no time axis. No causal or interventional claim is supported.
""".strip()
