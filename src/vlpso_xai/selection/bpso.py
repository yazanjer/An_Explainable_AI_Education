"""Fixed-length binary PSO -- the key baseline for the VLPSO claim.

Answers editor comment 5 ("Compare VLPSO with ordinary binary PSO ...").

MATCHING DISCIPLINE
-------------------
This class is matched to :class:`~vlpso_xai.selection.vlpso.VLPSOSelector` on
population size, iteration count, inertia schedule, learning coefficients,
wrapper learner, internal-CV fitness protocol and objective weights. The ONLY
difference is the length mechanism: every particle here is a fixed-length
binary vector over all ``p`` features, with an optional hard cardinality cap.
Anything else would confound the comparison the editor asked for.

Run it at several fixed cardinalities spanning the range VLPSO selects
(``config/default.yaml: selection.bpso.fixed_cardinalities``) so the
comparison is not decided by one arbitrary choice of subset size.

Note that this class is also what the SUBMITTED code actually implemented,
modulo the resubstitution fitness. Setting ``max_cardinality`` and running the
``shrink_only`` VLPSO ablation reproduces the original behaviour, which is why
the ablation table can quantify what the claimed novelty was worth.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseSelector
from .filters import su_matrix, su_scores


def _sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(v, -10, 10)))


class BPSOSelector(BaseSelector):
    """Standard binary PSO (Kennedy & Eberhart, 1997) for feature selection."""

    def __init__(
        self,
        population_size: int = 60,
        max_iter: int = 100,
        inertia_start: float = 0.9,
        inertia_end: float = 0.4,
        c1: float = 1.49445,
        c2: float = 1.49445,
        gamma_accuracy: float = 0.8,
        length_penalty: float = 0.15,
        interpretability_weight: float = 0.05,
        k_neighbors: int = 5,
        max_cardinality: Optional[int] = None,
        min_features: int = 2,
        fitness_cv_splits: int = 3,
        fitness_subsample: Optional[int] = 3000,
        convergence_tol: float = 1e-6,
        convergence_patience: int = 15,
        random_state: int = 42,
        n_bins: int = 10,
    ):
        self.population_size = population_size
        self.max_iter = max_iter
        self.inertia_start = inertia_start
        self.inertia_end = inertia_end
        self.c1 = c1
        self.c2 = c2
        self.gamma_accuracy = gamma_accuracy
        self.length_penalty = length_penalty
        self.interpretability_weight = interpretability_weight
        self.k_neighbors = k_neighbors
        self.max_cardinality = max_cardinality
        self.min_features = min_features
        self.fitness_cv_splits = fitness_cv_splits
        self.fitness_subsample = fitness_subsample
        self.convergence_tol = convergence_tol
        self.convergence_patience = convergence_patience
        self.random_state = random_state
        self.n_bins = n_bins

    def _fitness(self, values: np.ndarray, y: np.ndarray, cols: np.ndarray) -> float:
        """Identical objective to VLPSO, so only the length mechanism differs."""
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.neighbors import KNeighborsClassifier

        self._n_evaluations += 1
        if cols.size < 1:
            return -np.inf
        if self._fit_idx is not None:
            values, y = values[self._fit_idx], y[self._fit_idx]
        Xs = values[:, cols]
        n_splits = int(max(2, min(self.fitness_cv_splits, np.bincount(y).min())))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        for tr, va in skf.split(Xs, y):
            knn = KNeighborsClassifier(n_neighbors=int(min(self.k_neighbors, max(1, len(tr) - 1))))
            knn.fit(Xs[tr], y[tr])
            scores.append(balanced_accuracy_score(y[va], knn.predict(Xs[va])))
        acc = float(np.mean(scores))
        card = cols.size / values.shape[1]
        red = 0.0
        if self.interpretability_weight and cols.size > 1:
            sub = self._su_pairwise_[np.ix_(cols, cols)]
            iu = np.triu_indices(cols.size, k=1)
            red = float(np.mean(sub[iu]))
        return (
            self.gamma_accuracy * acc
            - self.length_penalty * card
            - self.interpretability_weight * red
        )

    def _apply_cap(self, bits: np.ndarray) -> np.ndarray:
        """Enforce the cardinality cap by keeping the best-ranked selected bits."""
        if self.max_cardinality is None:
            return bits
        on = np.where(bits)[0]
        if on.size <= self.max_cardinality:
            return bits
        keep = on[np.argsort(-self.ranking_scores_[on])[: self.max_cardinality]]
        out = np.zeros_like(bits)
        out[keep] = True
        return out

    def fit(self, X, y, **kwargs) -> "BPSOSelector":
        from ..data.features import assert_no_leakage

        values = self._record_input(X)
        if isinstance(X, pd.DataFrame):
            assert_no_leakage(X, where="BPSOSelector.fit")
        y = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        values = np.nan_to_num(values)
        self._fit_idx = None
        if self.fitness_subsample and values.shape[0] > self.fitness_subsample:
            from sklearn.model_selection import train_test_split

            self._fit_idx, _ = train_test_split(
                np.arange(values.shape[0]),
                train_size=int(self.fitness_subsample),
                stratify=y, random_state=self.random_state,
            )
        p = values.shape[1]
        self._n_evaluations = 0
        t0 = time.perf_counter()

        self.ranking_scores_ = su_scores(values, y, self.n_bins)
        self._su_pairwise_ = (
            su_matrix(values, self.n_bins) if self.interpretability_weight
            else np.zeros((p, p))
        )

        pos = rng.random((self.population_size, p)) < 0.5
        pos = np.array([self._apply_cap(b) for b in pos])
        vel = rng.normal(0, 1, (self.population_size, p))
        pbest = pos.copy()
        pbest_fit = np.array([self._fitness(values, y, np.where(b)[0]) for b in pos])
        gi = int(np.argmax(pbest_fit))
        gbest, gbest_fit = pbest[gi].copy(), float(pbest_fit[gi])

        self.convergence_ = {"iteration": [], "best_fitness": [], "mean_fitness": [],
                             "mean_selected": [], "n_evaluations": []}
        no_improve = 0
        for it in range(self.max_iter):
            w = self.inertia_start - (self.inertia_start - self.inertia_end) * (
                it / max(1, self.max_iter - 1)
            )
            r1, r2 = rng.random((self.population_size, p)), rng.random((self.population_size, p))
            vel = (
                w * vel
                + self.c1 * r1 * (pbest.astype(float) - pos.astype(float))
                + self.c2 * r2 * (gbest.astype(float) - pos.astype(float))
            )
            vel = np.clip(vel, -6, 6)
            pos = rng.random((self.population_size, p)) < _sigmoid(vel)
            pos = np.array([self._apply_cap(b) for b in pos])

            fits = np.array([self._fitness(values, y, np.where(b)[0]) for b in pos])
            improved = fits > pbest_fit + self.convergence_tol
            pbest[improved], pbest_fit[improved] = pos[improved], fits[improved]

            gi = int(np.argmax(pbest_fit))
            if pbest_fit[gi] > gbest_fit + self.convergence_tol:
                gbest, gbest_fit = pbest[gi].copy(), float(pbest_fit[gi])
                no_improve = 0
            else:
                no_improve += 1

            self.convergence_["iteration"].append(it)
            self.convergence_["best_fitness"].append(gbest_fit)
            self.convergence_["mean_fitness"].append(float(np.mean(pbest_fit)))
            self.convergence_["mean_selected"].append(float(pos.sum(axis=1).mean()))
            self.convergence_["n_evaluations"].append(self._n_evaluations)

            if no_improve >= self.convergence_patience:
                self.stopped_early_, self.stop_iteration_ = True, it
                break
        else:
            self.stopped_early_, self.stop_iteration_ = False, self.max_iter - 1

        self.scores_ = self.ranking_scores_
        self._finalise(gbest.astype(bool), min_features=self.min_features)
        self.gbest_fitness_ = gbest_fit
        self.n_evaluations_ = int(self._n_evaluations)
        self.fit_seconds_ = float(time.perf_counter() - t0)
        return self

    def algorithm_report(self) -> Dict[str, object]:
        return {
            "algorithm": "BPSO (fixed-length binary, sigmoid transfer)",
            "direction": "maximize",
            "representation": f"fixed binary vector of length p; cardinality cap = {self.max_cardinality}",
            "population_size": self.population_size,
            "max_iter": self.max_iter,
            "inertia": f"{self.inertia_start} -> {self.inertia_end} (linear)",
            "c1": self.c1, "c2": self.c2,
            "objective": "gamma*BAcc_cv - lambda*(|S|/p) - mu*Redundancy(S)",
            "gamma_accuracy": self.gamma_accuracy,
            "length_penalty": self.length_penalty,
            "interpretability_weight": self.interpretability_weight,
            "k_neighbors": self.k_neighbors,
            "fitness_evaluation": f"internal StratifiedKFold(n_splits={self.fitness_cv_splits})",
            "fitness_subsample": self.fitness_subsample,
            "stopped_early": getattr(self, "stopped_early_", None),
            "stop_iteration": getattr(self, "stop_iteration_", None),
            "n_evaluations": getattr(self, "n_evaluations_", None),
            "fit_seconds": getattr(self, "fit_seconds_", None),
            "gbest_fitness": getattr(self, "gbest_fitness_", None),
            "n_selected": self.n_selected_ if hasattr(self, "support_") else None,
        }
