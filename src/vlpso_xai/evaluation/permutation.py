"""Label-permutation null distribution.

Answers editor comment 4 ("...permutation-based checks...") and is the single
most decisive test that the leakage is gone.

PROTOCOL
--------
For each task x method x model, permute the labels **within outer-training
folds** and re-run the COMPLETE selection-and-training pipeline. Permuting
only at the very end, or permuting after feature selection, tests nothing: the
whole point is that every fitted component sees scrambled labels.

TWO DIFFERENT NULLS -- DO NOT CONFLATE THEM
-------------------------------------------
An earlier version of this module defaulted to within-school permutation AND
asserted that the result should equal 0.50. Those two things are inconsistent,
and the first real run exposed it: within-school permutation returned
AUC = 0.619, which looked like residual leakage and was not.

**(1) Unrestricted permutation** (``within_groups=False``). Labels are shuffled
across the whole sample, destroying every association. This is the LEAKAGE
TEST, and it is the only null for which chance-level performance is the correct
expectation. :func:`assert_permutation_null_is_chance` applies here and here
only.

**(2) Within-school permutation** (``within_groups=True``). Labels are shuffled
only among students in the same school, so each school's class composition is
preserved exactly. Any feature that predicts *which kind of school* a student
attends still predicts the permuted label. The expected value is therefore
ABOVE chance and must be ESTIMATED, never assumed.

Null (2) is not a diagnostic -- it is a substantive decomposition. The gap
between it and chance is the share of performance attributable to between-
school composition; the gap between the observed score and it is the share
attributable to within-school individual differences. For an equity-oriented
paper that decomposition is arguably more interesting than the headline metric,
and it is reported as its own result.

Labels are permuted WITHIN OUTER-TRAINING FOLDS and the COMPLETE pipeline is
re-run per permutation. Permuting after feature selection tests nothing.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PermutationSanityError(RuntimeError):
    """Permuted-label performance is above chance: leakage remains."""


def permute_within_groups(
    y: np.ndarray, groups: Optional[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Shuffle labels within each group (school), preserving cluster structure."""
    y = np.asarray(y)
    if groups is None:
        return rng.permutation(y)
    out = y.copy()
    groups = np.asarray(groups)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        out[idx] = rng.permutation(y[idx])
    return out


def permutation_null(
    fit_score: Callable[[np.ndarray, int], float],
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    n_permutations: int = 100,
    observed: Optional[float] = None,
    random_state: int = 42,
    within_groups: bool = False,
) -> Dict[str, object]:
    """Build the empirical null and the permutation p-value.

    Parameters
    ----------
    fit_score:
        ``fit_score(y_permuted, seed) -> metric``. Must re-run the ENTIRE
        pipeline -- imputation, scaling, feature selection, tuning, fitting --
        on the permuted labels.
    """
    rng = np.random.default_rng(random_state)
    null: List[float] = []
    for i in range(n_permutations):
        yp = permute_within_groups(y, groups if within_groups else None, rng)
        null.append(float(fit_score(yp, random_state + i)))

    arr = np.asarray(null, dtype=float)
    finite = arr[np.isfinite(arr)]
    res: Dict[str, object] = {
        "n_permutations": int(n_permutations),
        "n_usable": int(finite.size),
        "null_mean": float(finite.mean()) if finite.size else float("nan"),
        "null_sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        "null_q025": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
        "null_q975": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
        "null_max": float(finite.max()) if finite.size else float("nan"),
        "within_groups": bool(within_groups and groups is not None),
        "null_distribution": finite.tolist(),
    }
    if observed is not None:
        # (#{null >= observed} + 1) / (n + 1): the standard conservative estimator.
        res["observed"] = float(observed)
        res["p_value"] = float((np.sum(finite >= observed) + 1) / (finite.size + 1))
    return res


def assert_permutation_null_is_chance(
    null_mean: float,
    *,
    expected: float = 0.50,
    tolerance: float = 0.03,
    context: str = "",
    within_groups: bool = False,
) -> None:
    """Hard gate. Stage 10 check 3. UNRESTRICTED permutation only.

    Raises if the permuted-label mean AUC deviates from chance by more than
    ``tolerance``. Deviation means information about the labels is still
    reaching the model.
    """
    if within_groups:
        raise ValueError(
            "assert_permutation_null_is_chance is only valid for UNRESTRICTED "
            "permutation. Within-school permutation preserves each school's "
            "class composition, so its expected value is above chance and must "
            "be estimated rather than asserted. See the module docstring."
        )
    if not np.isfinite(null_mean):
        raise PermutationSanityError(f"Permutation null is not finite {context}.")
    if abs(null_mean - expected) > tolerance:
        raise PermutationSanityError(
            f"Permuted-label mean AUC = {null_mean:.4f}, expected "
            f"{expected:.2f} +/- {tolerance:.2f} {context}. Label information is "
            "still reaching the model. HALT and diagnose -- do not report these "
            "results. Check: (1) is any forbidden column in X? (2) is any "
            "preprocessing step fitted outside the Pipeline? (3) were the "
            "labels permuted before or after feature selection?"
        )
    logger.info("Permutation sanity passed %s: null mean AUC = %.4f", context, null_mean)


def permutation_table(results: Sequence[Dict], group_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Assemble permutation results into the manuscript table."""
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "null_distribution"}
        rows.append(row)
    return pd.DataFrame(rows)


def decompose_performance(
    observed: float,
    null_within_groups: float,
    null_unrestricted: float,
) -> Dict[str, float]:
    """Split performance into between-school and within-school components.

    ``observed``            score with true labels
    ``null_within_groups``  labels shuffled within school (school composition kept)
    ``null_unrestricted``   labels shuffled everywhere (nothing kept)

    The between-school share is what a model could achieve knowing only what
    kind of school a student attends. The within-school share is the additional
    discrimination between students in the SAME school -- which is the part an
    individual-level educational claim actually rests on.
    """
    total = observed - null_unrestricted
    between = null_within_groups - null_unrestricted
    within = observed - null_within_groups
    return {
        "observed": float(observed),
        "null_within_school": float(null_within_groups),
        "null_unrestricted": float(null_unrestricted),
        "total_above_chance": float(total),
        "between_school_component": float(between),
        "within_school_component": float(within),
        "between_school_share": float(between / total) if total else float("nan"),
        "within_school_share": float(within / total) if total else float("nan"),
    }
