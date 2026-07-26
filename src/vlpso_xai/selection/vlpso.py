"""Variable-Length Particle Swarm Optimisation for feature selection.

Answers editor comment 9: "Define precisely how variable particle length is
represented and updated, how dimensions are added and removed, how differently
sized particles interact, and whether the optimisation criterion is minimised
or maximised."

AUTHOR DECISION: Path A -- implement genuine variable-length search.

WHAT WAS WRONG WITH THE SUBMITTED IMPLEMENTATION
------------------------------------------------
``feature_selection.ipynb`` cell 8:

* Every particle was ``np.zeros(self.n_features)`` -- a FIXED-length binary
  vector. "Length" was only a cardinality cap enforced by truncating to the
  top-MI bits. The search dimensionality never varied, so the paper's central
  claimed advance over binary PSO was not implemented.
* The length-adaptation block only ever cleared bits
  (``population[i][drop] = 0``). Lengths were monotonically non-increasing;
  a particle could never recover a discarded feature.
* Different-length interaction was trivial because all vectors were the same
  length. The manuscript describes a mechanism the code did not have.
* Fitness was resubstitution accuracy: ``knn.fit(X_sel, y)`` then
  ``knn.predict(X_sel)``. With a leaking ``math_score`` column this saturates
  immediately and the search has no gradient.
* ``alpha=7`` was declared and never used.
* ``best_div = np.argmax(pbest_fitness)`` is a particle index despite its name.
* No stopping criterion beyond a fixed iteration count.
* The fitness had two terms (accuracy + inter-class distance) and NO
  feature-count penalty, contradicting the manuscript's three-term objective
  and the abstract's claim of "minimizing the number of selected features".

THE REPRESENTATION USED HERE
----------------------------
**Ranked-prefix encoding with an explicit per-particle length.**

Features are ranked once per ``fit`` by symmetric uncertainty against the
label, giving an order ``r``. Particle *i* carries

    L_i  in [L_min, L_max]      an explicit integer dimensionality
    x_i  in [0,1]^{L_i}         position; x_i[d] is the selection propensity
    v_i  in R^{L_i}             velocity

Dimension *d* of every particle refers to the SAME feature, ``r[d]``. The
selected subset is ``S_i = { r[d] : x_i[d] > theta }``.

*Why this encoding.* Two properties follow directly, and both are what the
submitted code lacked:

1. ``L_i`` is genuinely the dimensionality of the particle's search space, not
   a cardinality cap. A particle of length ``L_i`` can only ever consider the
   top ``L_i`` features by SU. Growing ``L_i`` genuinely enlarges the space;
   shrinking it genuinely contracts it.
2. Because dimension index means the same thing across particles, cross-length
   learning is well defined on the overlap and the only genuinely hard case is
   the non-overlap, which is specified explicitly below rather than avoided.

CROSS-LENGTH LEARNING (the part the original code sidestepped)
--------------------------------------------------------------
Particle *i* of length ``L_i`` learns from exemplar *e* of length ``L_e`` over
``O = min(L_i, L_e)`` shared dimensions with a standard CLPSO update. For
dimensions ``d`` in ``[O, L_i)`` the exemplar has no coordinate, and the rule
is (configurable via ``exemplar_fallback``):

``"gbest_longest"`` (default)
    use the coordinate of the best-fitness particle whose length exceeds ``d``;
    if no such particle exists, fall back to the particle's own ``pbest``.
``"pbest"``
    use the particle's own personal best coordinate.
``"inertia"``
    apply inertia only -- no social term for that dimension.

LENGTH ADAPTATION
-----------------
Triggered when a particle has not improved its personal best for
``beta_stagnation`` iterations. With probability ``growth_prob`` the length
grows, otherwise it shrinks, by ``length_step``:

*Growth.* New dimensions are appended at the tail (the next-best features by
SU). Position is seeded from the swarm's best particle that already spans that
dimension; if none does, from ``Uniform(0,1)``. Velocity is initialised to
**zero**: the particle has no momentum in a direction it has never explored.

*Shrink.* The tail is truncated. Because the ranking is by descending SU, the
tail holds the least individually informative candidates.

Both directions are available, unlike the shrink-only original. The
``length_direction`` ablation (``both`` / ``shrink_only`` / ``grow_only``)
isolates this.

OBJECTIVE
---------
**Maximised.** Stated explicitly, as the editor asked.

.. math::

    f(S) = \\gamma \\cdot \\mathrm{BAcc}_{\\mathrm{CV}}(S)
         - \\lambda \\cdot \\frac{|S|}{p}
         - \\mu \\cdot \\mathrm{Red}(S)

* ``BAcc_CV`` is balanced accuracy from an **internal stratified k-fold CV**
  on the data given to ``fit`` -- never resubstitution.
* ``|S|/p`` is the feature-count penalty the manuscript claims and the
  submitted code omitted.
* ``Red(S)`` is mean pairwise symmetric uncertainty within ``S``: a compact,
  non-redundant subset is easier to interpret. This operationalises the
  manuscript's third "interpretability" objective term.

Setting ``length_penalty=0`` and ``interpretability_weight=0`` recovers a
single-term objective for the per-term ablation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import BaseSelector
from .filters import su_matrix, su_scores

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    """One variable-length particle."""

    length: int
    position: np.ndarray           # (length,)
    velocity: np.ndarray           # (length,)
    pbest_position: np.ndarray
    pbest_length: int
    pbest_fitness: float = -np.inf
    stagnation: int = 0

    def selected(self, ranking: np.ndarray, theta: float) -> np.ndarray:
        """Indices into the ORIGINAL feature matrix that this particle selects."""
        return ranking[: self.length][self.position > theta]


@dataclass
class ConvergenceRecord:
    """Per-iteration trace, plotted as the convergence curves editor comment 9 asks for."""

    iteration: List[int] = field(default_factory=list)
    best_fitness: List[float] = field(default_factory=list)
    mean_fitness: List[float] = field(default_factory=list)
    mean_length: List[float] = field(default_factory=list)
    mean_selected: List[float] = field(default_factory=list)
    n_evaluations: List[int] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.__dict__)


class VLPSOSelector(BaseSelector):
    """Genuine variable-length PSO feature selection.

    Parameters
    ----------
    population_size, divisions, max_iter, inertia_start, inertia_end, c1, c2:
        Standard PSO controls. ``divisions`` groups the initial swarm into
        length strata so the search starts with diverse dimensionalities.
    gamma_accuracy, length_penalty, interpretability_weight:
        The three objective weights. All are reported in the manuscript.
    beta_stagnation:
        Iterations without personal-best improvement before length adapts.
    growth_prob, length_step, length_direction:
        Length-adaptation controls. ``length_direction`` is the ablation knob.
    ranking:
        ``"symmetric_uncertainty"`` (default) or ``"mutual_info"``. This is the
        SU ablation knob.
    init:
        ``"ranked"`` seeds positions towards the top-ranked features;
        ``"random"`` is the initialisation ablation.
    fitness_cv_splits:
        Folds of the INTERNAL CV used for the wrapper score. Must be >= 2;
        there is deliberately no resubstitution option.
    fitness_subsample:
        Stratified subsample size used for the k-NN wrapper evaluation ONLY.
        The wrapper is O(n^2) in the training-fold size, so on the full Spanish
        sample (~15,000 students in Low vs. High) one fitness evaluation costs
        ~1.5e8 distance computations and a full search is infeasible.
        Subsampling the FITNESS EVALUATION is standard for wrapper selection at
        this scale. It does not touch the final model, which the surrounding
        Pipeline always refits on the complete training fold. The value used is
        reported in the computational-cost table. ``None`` disables it.
    convergence_tol, convergence_patience:
        Stopping criterion. The submitted code had none.
    """

    def __init__(
        self,
        population_size: int = 60,
        divisions: int = 12,
        max_iter: int = 100,
        inertia_start: float = 0.9,
        inertia_end: float = 0.4,
        c1: float = 1.49445,
        c2: float = 1.49445,
        gamma_accuracy: float = 0.8,
        length_penalty: float = 0.15,
        interpretability_weight: float = 0.05,
        beta_stagnation: int = 9,
        k_neighbors: int = 5,
        min_length: int = 2,
        max_length: Optional[int] = None,
        growth_prob: float = 0.5,
        length_step: int = 1,
        length_direction: str = "both",
        selection_threshold: float = 0.5,
        ranking: str = "symmetric_uncertainty",
        init: str = "ranked",
        exemplar_fallback: str = "gbest_longest",
        fitness_cv_splits: int = 3,
        fitness_subsample: Optional[int] = 3000,
        convergence_tol: float = 1e-6,
        convergence_patience: int = 15,
        random_state: int = 42,
        n_bins: int = 10,
        verbose: bool = False,
    ):
        self.population_size = population_size
        self.divisions = divisions
        self.max_iter = max_iter
        self.inertia_start = inertia_start
        self.inertia_end = inertia_end
        self.c1 = c1
        self.c2 = c2
        self.gamma_accuracy = gamma_accuracy
        self.length_penalty = length_penalty
        self.interpretability_weight = interpretability_weight
        self.beta_stagnation = beta_stagnation
        self.k_neighbors = k_neighbors
        self.min_length = min_length
        self.max_length = max_length
        self.growth_prob = growth_prob
        self.length_step = length_step
        self.length_direction = length_direction
        self.selection_threshold = selection_threshold
        self.ranking = ranking
        self.init = init
        self.exemplar_fallback = exemplar_fallback
        self.fitness_cv_splits = fitness_cv_splits
        self.fitness_subsample = fitness_subsample
        self.convergence_tol = convergence_tol
        self.convergence_patience = convergence_patience
        self.random_state = random_state
        self.n_bins = n_bins
        self.verbose = verbose

    # -- objective -------------------------------------------------------
    def _rank_features(self, values: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.ranking == "symmetric_uncertainty":
            self.ranking_scores_ = su_scores(values, y, self.n_bins)
        elif self.ranking == "mutual_info":
            from sklearn.feature_selection import mutual_info_classif

            self.ranking_scores_ = mutual_info_classif(
                np.nan_to_num(values), y, random_state=self.random_state
            )
        else:
            raise ValueError(
                f"ranking must be 'symmetric_uncertainty' or 'mutual_info', "
                f"got {self.ranking!r}"
            )
        return np.argsort(-self.ranking_scores_)

    def _fitness(self, values: np.ndarray, y: np.ndarray, cols: np.ndarray) -> float:
        """The three-term objective. MAXIMISED."""
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.neighbors import KNeighborsClassifier

        self._n_evaluations += 1
        if cols.size < 1:
            return -np.inf
        if self._fit_idx is not None:
            values, y = values[self._fit_idx], y[self._fit_idx]

        Xs = values[:, cols]
        p = values.shape[1]

        # --- performance term: INTERNAL CV, never resubstitution ---------
        n_splits = int(max(2, min(self.fitness_cv_splits, np.bincount(y).min())))
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        scores = []
        for tr, va in skf.split(Xs, y):
            knn = KNeighborsClassifier(
                n_neighbors=int(min(self.k_neighbors, max(1, len(tr) - 1)))
            )
            knn.fit(Xs[tr], y[tr])
            scores.append(balanced_accuracy_score(y[va], knn.predict(Xs[va])))
        acc = float(np.mean(scores))

        # --- cardinality penalty (absent from the submitted code) -------
        card = cols.size / p

        # --- interpretability: mean pairwise redundancy within S --------
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

    # -- swarm -----------------------------------------------------------
    def _init_swarm(self, p: int, rng: np.random.Generator) -> List[Particle]:
        lo, hi = self.min_length, self._max_length
        step = max(1, (hi - lo) // max(1, self.divisions))
        swarm: List[Particle] = []
        for i in range(self.population_size):
            div = i % self.divisions
            length = int(np.clip(lo + div * step + rng.integers(0, max(1, step)), lo, hi))
            if self.init == "ranked":
                # Bias towards selecting the better-ranked (lower-index) dims.
                base = np.linspace(0.85, 0.35, length)
                pos = np.clip(base + rng.normal(0, 0.15, length), 0.0, 1.0)
            elif self.init == "random":
                pos = rng.random(length)
            else:
                raise ValueError(f"init must be 'ranked' or 'random', got {self.init!r}")
            swarm.append(
                Particle(
                    length=length,
                    position=pos,
                    velocity=rng.normal(0, 0.1, length),
                    pbest_position=pos.copy(),
                    pbest_length=length,
                )
            )
        return swarm

    def _exemplar_coord(
        self, swarm: List[Particle], particle: Particle, d: int
    ) -> float:
        """Coordinate to learn from at a dimension the exemplar does not span."""
        if self.exemplar_fallback == "inertia":
            return float("nan")
        if self.exemplar_fallback == "pbest":
            return (
                float(particle.pbest_position[d])
                if d < particle.pbest_length
                else float("nan")
            )
        # "gbest_longest": best-fitness particle that spans dimension d
        best, best_fit = None, -np.inf
        for q in swarm:
            if q.pbest_length > d and q.pbest_fitness > best_fit:
                best, best_fit = q, q.pbest_fitness
        if best is not None:
            return float(best.pbest_position[d])
        return (
            float(particle.pbest_position[d])
            if d < particle.pbest_length
            else float("nan")
        )

    def _grow(self, particle: Particle, swarm: List[Particle], rng) -> None:
        new_len = int(min(particle.length + self.length_step, self._max_length))
        if new_len == particle.length:
            return
        add = new_len - particle.length
        seeds = []
        for d in range(particle.length, new_len):
            donors = [q for q in swarm if q.length > d]
            if donors:
                donor = max(donors, key=lambda q: q.pbest_fitness)
                seeds.append(float(donor.position[d]))
            else:
                seeds.append(float(rng.random()))
        particle.position = np.concatenate([particle.position, np.asarray(seeds)])
        # Zero velocity: no momentum in a direction never explored.
        particle.velocity = np.concatenate([particle.velocity, np.zeros(add)])
        particle.pbest_position = np.concatenate(
            [particle.pbest_position, np.asarray(seeds)]
        )
        particle.length = new_len
        particle.pbest_length = max(particle.pbest_length, new_len)

    def _shrink(self, particle: Particle) -> None:
        new_len = int(max(particle.length - self.length_step, self.min_length))
        if new_len == particle.length:
            return
        particle.position = particle.position[:new_len]
        particle.velocity = particle.velocity[:new_len]
        particle.length = new_len

    def _adapt_length(self, particle: Particle, swarm: List[Particle], rng) -> str:
        if self.length_direction == "shrink_only":
            self._shrink(particle)
            return "shrink"
        if self.length_direction == "grow_only":
            self._grow(particle, swarm, rng)
            return "grow"
        if self.length_direction != "both":
            raise ValueError(
                "length_direction must be 'both', 'shrink_only' or 'grow_only', "
                f"got {self.length_direction!r}"
            )
        if rng.random() < self.growth_prob:
            self._grow(particle, swarm, rng)
            return "grow"
        self._shrink(particle)
        return "shrink"

    # -- fit -------------------------------------------------------------
    def fit(self, X, y, **kwargs) -> "VLPSOSelector":
        from ..data.features import assert_no_leakage

        values = self._record_input(X)
        if isinstance(X, pd.DataFrame):
            # Guard at EVERY entry point, not once at the top.
            assert_no_leakage(X, where="VLPSOSelector.fit")

        y = np.asarray(y).astype(int)

        # ------------------------------------------------------------------
        # Guard against a silently inert length-adaptation mechanism.
        #
        # Length adaptation only fires once a particle has stagnated for
        # beta_stagnation iterations. If beta_stagnation >= max_iter it can
        # NEVER fire: lengths stay at their initial values, and the
        # length_direction ablation ('both' vs 'shrink_only' vs 'grow_only')
        # returns byte-identical results for every setting.
        #
        # Found while running the ablation table: with max_iter=8 and the
        # default beta_stagnation=9, 'both' and 'shrink_only' agreed to every
        # decimal place. That would have been written up as "variable-length
        # search makes no difference" when in fact it had never run.
        # ------------------------------------------------------------------
        # Validate every categorical parameter EAGERLY. Validating lazily (at
        # the point of use) means a typo in length_direction only surfaces if
        # adaptation happens to fire, so a short run silently accepts nonsense.
        if self.length_direction not in ("both", "shrink_only", "grow_only"):
            raise ValueError(
                "length_direction must be 'both', 'shrink_only' or "
                f"'grow_only', got {self.length_direction!r}"
            )
        if self.ranking not in ("symmetric_uncertainty", "mutual_info"):
            raise ValueError(
                "ranking must be 'symmetric_uncertainty' or 'mutual_info', "
                f"got {self.ranking!r}"
            )
        if self.init not in ("ranked", "random"):
            raise ValueError(f"init must be 'ranked' or 'random', got {self.init!r}")
        if self.exemplar_fallback not in ("gbest_longest", "pbest", "inertia"):
            raise ValueError(
                "exemplar_fallback must be 'gbest_longest', 'pbest' or "
                f"'inertia', got {self.exemplar_fallback!r}"
            )
        if self.fitness_cv_splits < 2:
            raise ValueError(
                "fitness_cv_splits must be >= 2. There is deliberately no "
                "resubstitution option: the submitted implementation scored "
                "particles with knn.fit(X_sel, y) then knn.predict(X_sel)."
            )
        if self.beta_stagnation >= self.max_iter:
            raise ValueError(
                f"beta_stagnation ({self.beta_stagnation}) >= max_iter "
                f"({self.max_iter}): the length-adaptation mechanism can never "
                "trigger, so this is not a variable-length search and any "
                "length_direction ablation would be vacuous. Set "
                "beta_stagnation < max_iter."
            )
        rng = np.random.default_rng(self.random_state)
        p = values.shape[1]
        self._max_length = int(self.max_length or p)
        self._max_length = int(np.clip(self._max_length, self.min_length + 1, p))
        self._n_evaluations = 0
        t0 = time.perf_counter()

        values = np.nan_to_num(values)
        self._fit_idx = None
        if self.fitness_subsample and values.shape[0] > self.fitness_subsample:
            from sklearn.model_selection import train_test_split

            self._fit_idx, _ = train_test_split(
                np.arange(values.shape[0]),
                train_size=int(self.fitness_subsample),
                stratify=y, random_state=self.random_state,
            )
        self.ranking_ = self._rank_features(values, y)
        self._su_pairwise_ = (
            su_matrix(values, self.n_bins)
            if self.interpretability_weight
            else np.zeros((p, p))
        )

        swarm = self._init_swarm(p, rng)
        for prt in swarm:
            cols = prt.selected(self.ranking_, self.selection_threshold)
            prt.pbest_fitness = self._fitness(values, y, cols)

        gi = int(np.argmax([q.pbest_fitness for q in swarm]))
        gbest_position = swarm[gi].pbest_position.copy()
        gbest_length = swarm[gi].pbest_length
        gbest_fitness = swarm[gi].pbest_fitness

        self.convergence_ = ConvergenceRecord()
        self.length_history_: List[List[int]] = []
        no_improve = 0

        for it in range(self.max_iter):
            w = self.inertia_start - (self.inertia_start - self.inertia_end) * (
                it / max(1, self.max_iter - 1)
            )
            for prt in swarm:
                overlap = min(prt.length, gbest_length)
                r1 = rng.random(prt.length)
                r2 = rng.random(prt.length)

                social = np.empty(prt.length)
                social[:overlap] = gbest_position[:overlap]
                for d in range(overlap, prt.length):
                    social[d] = self._exemplar_coord(swarm, prt, d)

                cognitive = np.where(
                    np.arange(prt.length) < prt.pbest_length,
                    prt.pbest_position[: prt.length]
                    if prt.pbest_length >= prt.length
                    else np.pad(
                        prt.pbest_position,
                        (0, max(0, prt.length - prt.pbest_length)),
                        constant_values=np.nan,
                    )[: prt.length],
                    np.nan,
                )

                cog_term = np.where(
                    np.isnan(cognitive), 0.0, self.c1 * r1 * (cognitive - prt.position)
                )
                soc_term = np.where(
                    np.isnan(social), 0.0, self.c2 * r2 * (social - prt.position)
                )
                prt.velocity = w * prt.velocity + cog_term + soc_term
                prt.velocity = np.clip(prt.velocity, -0.5, 0.5)
                prt.position = np.clip(prt.position + prt.velocity, 0.0, 1.0)

                cols = prt.selected(self.ranking_, self.selection_threshold)
                fit_val = self._fitness(values, y, cols)

                if fit_val > prt.pbest_fitness + self.convergence_tol:
                    prt.pbest_fitness = fit_val
                    prt.pbest_position = prt.position.copy()
                    prt.pbest_length = prt.length
                    prt.stagnation = 0
                else:
                    prt.stagnation += 1

                if prt.stagnation >= self.beta_stagnation:
                    self._adapt_length(prt, swarm, rng)
                    prt.stagnation = 0

            best_i = int(np.argmax([q.pbest_fitness for q in swarm]))
            if swarm[best_i].pbest_fitness > gbest_fitness + self.convergence_tol:
                gbest_fitness = swarm[best_i].pbest_fitness
                gbest_position = swarm[best_i].pbest_position.copy()
                gbest_length = swarm[best_i].pbest_length
                no_improve = 0
            else:
                no_improve += 1

            fits = [q.pbest_fitness for q in swarm]
            lens = [q.length for q in swarm]
            self.length_history_.append(list(lens))
            self.convergence_.iteration.append(it)
            self.convergence_.best_fitness.append(float(gbest_fitness))
            self.convergence_.mean_fitness.append(float(np.mean(fits)))
            self.convergence_.mean_length.append(float(np.mean(lens)))
            self.convergence_.mean_selected.append(
                float(np.mean([q.selected(self.ranking_, self.selection_threshold).size
                               for q in swarm]))
            )
            self.convergence_.n_evaluations.append(self._n_evaluations)

            if self.verbose and it % 10 == 0:
                logger.info(
                    "iter %3d  gbest %.4f  mean_len %.1f  evals %d",
                    it, gbest_fitness, np.mean(lens), self._n_evaluations,
                )

            # Stopping criterion -- the submitted code had none.
            if no_improve >= self.convergence_patience:
                self.stopped_early_ = True
                self.stop_iteration_ = it
                break
        else:
            self.stopped_early_ = False
            self.stop_iteration_ = self.max_iter - 1

        chosen = self.ranking_[:gbest_length][
            gbest_position[:gbest_length] > self.selection_threshold
        ]
        mask = np.zeros(p, dtype=bool)
        mask[chosen] = True
        self.scores_ = self.ranking_scores_
        self._finalise(mask, min_features=self.min_length)

        self.gbest_fitness_ = float(gbest_fitness)
        self.gbest_length_ = int(gbest_length)
        self.n_evaluations_ = int(self._n_evaluations)
        self.fit_seconds_ = float(time.perf_counter() - t0)
        self.final_lengths_ = [q.length for q in swarm]
        return self

    # -- reporting -------------------------------------------------------
    def algorithm_report(self) -> Dict[str, object]:
        """Everything editor comment 9 asks to be reported, per run."""
        return {
            "algorithm": "VLPSO (variable-length, ranked-prefix encoding)",
            "direction": "maximize",
            "representation": "explicit per-particle length L_i; position in [0,1]^{L_i}; dimension d -> ranked feature r[d]",
            "cross_length_rule": self.exemplar_fallback,
            "length_direction": self.length_direction,
            "ranking": self.ranking,
            "init": self.init,
            "population_size": self.population_size,
            "divisions": self.divisions,
            "max_iter": self.max_iter,
            "inertia": f"{self.inertia_start} -> {self.inertia_end} (linear)",
            "c1": self.c1,
            "c2": self.c2,
            "beta_stagnation": self.beta_stagnation,
            "growth_prob": self.growth_prob,
            "length_step": self.length_step,
            "objective": "gamma*BAcc_cv - lambda*(|S|/p) - mu*Redundancy(S)",
            "gamma_accuracy": self.gamma_accuracy,
            "length_penalty": self.length_penalty,
            "interpretability_weight": self.interpretability_weight,
            "k_neighbors": self.k_neighbors,
            "fitness_evaluation": f"internal StratifiedKFold(n_splits={self.fitness_cv_splits})",
            "fitness_subsample": self.fitness_subsample,
            "stopping": f"max_iter={self.max_iter} or no gbest improvement for {self.convergence_patience} iters",
            "stopped_early": getattr(self, "stopped_early_", None),
            "stop_iteration": getattr(self, "stop_iteration_", None),
            "n_evaluations": getattr(self, "n_evaluations_", None),
            "fit_seconds": getattr(self, "fit_seconds_", None),
            "gbest_fitness": getattr(self, "gbest_fitness_", None),
            "gbest_length": getattr(self, "gbest_length_", None),
            "n_selected": self.n_selected_ if hasattr(self, "support_") else None,
            "final_length_mean": float(np.mean(self.final_lengths_)) if hasattr(self, "final_lengths_") else None,
            "final_length_min": int(np.min(self.final_lengths_)) if hasattr(self, "final_lengths_") else None,
            "final_length_max": int(np.max(self.final_lengths_)) if hasattr(self, "final_lengths_") else None,
        }

    def complexity(self) -> str:
        """Big-O cost, reported in the manuscript."""
        return (
            "O(T * P * (F_cv + L)) where T = iterations, P = population size, "
            "L = mean particle length, and F_cv is the cost of one internal "
            "k-fold wrapper evaluation, i.e. O(k * n^2 * |S|) for exact k-NN on "
            "n training rows with |S| selected features. Ranking adds a one-off "
            "O(p * n) for SU against the label plus O(p^2 * n) for the pairwise "
            "SU matrix when the interpretability term is enabled."
        )
