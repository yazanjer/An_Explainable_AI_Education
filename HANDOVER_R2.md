# Handover — second-round revision (editor first decision + four reviewers)

Supersedes `HANDOVER_RUNPOD_AGENT.md` (round-1 brief). Read §2 and §6 before
touching numbers.

## 1. Where things are

| What | Where |
|---|---|
| Code | `github.com/yazanjer/An_Explainable_AI_Education`, branch `fix/audit-m2-m4-aggregation` (pushed; the round-1 brief's "local only" was stale — the branch was already on GitHub) |
| Round-2 configuration | `config/revision_r2.yaml` |
| Cell runner | `scripts/cells.py` (`manifest`, `run`, `reconcile`, `aggregate`, `publish`, `merge`) |
| Round-2 library code | `src/vlpso_xai/experiments/`, `selection/{embedded,wrappers,tuned}.py` |
| Round-2 outputs (non-student-level only) | branches `r2-results-a` … `r2-results-d`, folder `round2/` |
| Changelog | `CHANGELOG_REVISION.md` §R2 |
| Reviewer report | `reviewers report .docx` (project folder) |
| Round-1 authoritative headline run | `results-3/` (project folder; never leaves the author's machine) |
| Notebook behind manuscript §4.6 of the 18 Aug PDF | `notebooks/08_explainability_corrected.ipynb` (committed in round 2; outputs on Drive `results/explainability/`) |

## 2. What the editor and reviewers asked, and what answers it

| Demand | Answered by | Status |
|---|---|---|
| Ed: moderate claims; VLPSO did not beat the full set, BPSO or any filter, and was least stable | manuscript reframing (protocol = primary contribution; VLPSO = refinement trading features for AUC); claims follow the round-2 `sel` results whatever they show | manuscript |
| Ed / R2.3: comparison too small (3 folds, 1 repeat, 1 PV, 1 task, 1 model; 5 methods missing) | `sel` cells: 3 tasks × 10 PVs × 5 repeats × 5 folds × 16 arms; RFE, SFS, L1, tree importance, Boruta implemented | running |
| Ed / R2.2: grouped resampling in the swarm fitness | `selection.base.fitness_splits` (StratifiedGroupKFold on CNTSCHID, asserted); `require_groups=True` | done |
| Ed: fair comparison (BPSO cap vs uncapped VLPSO; fixed k for filters) | shared swarm settings block, test asserts equality; `TunedTopKSelector` (k on inner folds) | done |
| Ed: instability — repetitions, sensitivity, convergence | `vlstab` cells (10 seeds × 5 folds for VLPSO and BPSO; 13 one-at-a-time variants); convergence traces in every swarm cell | running |
| Ed / R4.6: reproducibility, permanent link | repo + Zenodo release (to mint after results land); allowlist, denylist, seeds, grids, item mappings already in `config/` | pending DOI |
| Ed / R1 / R3.3 / R2.5: language, structure, notation, equations, cross-references, reference order | manuscript rewrite | manuscript |
| R1: abstract length (480 words) | ≤ 200 words | manuscript |
| R1.1: no EDA; target unexplained | `eda` cell (item descriptives, PV distribution, task sizes, categories) + new section | running |
| R1.2: undefined terms (allowlist, Rubin's rules, Fay BRR, 1 + CV², design matrix, §2.3 sentence) | definitions at first use + short glossary | manuscript |
| R2.1: why VLPSO underperforms | `sel` + `vlstab` (budget, penalty, convergence, seed variance) | running |
| R2.4: explanations from one fold / one PV | `shap` cells: 10 PVs × 5 folds × 3 tasks, Kendall's W | running |
| R3.2 / R3.5: third "innovation" is not one; distinguish protocol from VLPSO; retitle | manuscript | manuscript |
| R3.4: Table 5 "Null" header | relabel | manuscript |
| R4.1: EDM undefined in abstract | manuscript | manuscript |
| R4.2: BRR vs bootstrap; estimand | `brr` cells (Fay-BRR on weighted fold-weighted AUC, Rubin); text states both estimands | running |
| R4.3: arithmetic (0.8760 − 0.5029) | full-precision recomputation from cells; arithmetic check script | manuscript |
| R4.4: Figure 1 puts preprocessing before the split | redraw | manuscript |
| R4.5: refs 30 and 9 irrelevant; incomplete refs | remove 9, 16, 26, 30 (all off-topic; 16/26/30 self-citations); complete 31–34, 50 | manuscript |

Additional defects found in round 2 (not raised by reviewers, must be fixed):
the round-1 LIME "SD exceeds mean" sentence is a zero-fill artefact; notebook
08 imputed before the split; §3.4 claims one-hot encoding that never happens;
§3.4/§4.1 list eight model families while three were searched; Equation (9)
is an unweighted fold mean while the estimand is size-weighted; Equations
(1)–(2) (minimised, capped) contradict Equation (8) (maximised, uncapped);
Equation (4) uses gbest while Algorithm 3 uses per-dimension exemplars; §3.8
says the school file was merged — ingest reads the student file only; the
"0.013–0.023 cost of grouping" sentence has no generating run (delete).
Found while rewriting §4.5 against `selection/vlpso.py`: the 18 Aug text
described the third objective term as +λ3·MeanSU(S, Y) (relevance), but the
code subtracts μ·Red(S), the mean pairwise SU *within* S (redundancy); the
text gave a sigmoid position map (Eq. 5) but the code clips x + v to [0, 1]
with |v| ≤ 0.5 and thresholds at τ = 0.5; Algorithm 3's per-dimension
tournament exemplars do not exist (the social target is gbest on the overlap,
the best spanning pbest beyond it); and length adaptation is a ±1 step on
per-particle pbest stagnation (β = 9, p_grow = 0.5), not a redraw over
[L_min, F] on gbest stagnation. The revised §4.5 and Algorithm 1 describe the
code; the 1,122-column count is the student file plus columns derived at
ingestion, not a student–school merge.

## 3. The compute run (October 2026)

Four RunPod CPU pods (cpu3c, 8 vCPU, 16 GB, $0.24/h each), image
`python:3.12-bookworm`, started 2026-10-01 ~17:00 UTC. Large CPU flavours
were out of stock.

| Tag | Pod id | Role (`KINDS`, `SHARD`) | Branch |
|---|---|---|---|
| a | `bnerfsli5edbhb` | eda, shap, ext, perm; then brr | `r2-results-a` |
| b | `xu6ugsmet03yms` | vlstab, sel, shard 0/3 | `r2-results-b` |
| c | `8kllzs2662575l` | vlstab, sel, shard 1/3 | `r2-results-c` |
| d | `x2dzn7fjevw2ga` | vlstab, sel, shard 2/3 | `r2-results-d` |

* The OECD file is downloaded inside each pod and never leaves it.
* Each pod generated its own SSH key; the public halves are registered as
  write deploy keys on the code repo (ids 165080251, 165080384, 165080386,
  165080385). **Delete all four when the run ends.**
* `MAX_HOURS` 20 (a) / 30 (b–d) bounds spend at about $26.
* Each pod publishes every 25 min and at the end; `ALL_DONE` in the log marks
  completion. Stop/terminate pods as soon as they are done.

Merging and aggregating (anywhere with the repo):

```bash
for t in a b c d; do git clone -q --depth 1 -b r2-results-$t <repo> pub_$t; done
python scripts/cells.py merge --results results_r2 --sources pub_*/round2
python scripts/cells.py reconcile --results results_r2
python scripts/cells.py aggregate --results results_r2     # -> results_r2/r2_tables/*.csv
```

## 4. Determinism and reproduction check

Seeds derive from cell ids (`experiments.cells.cell_seed`). Outer folds are
the headline folds (`school_grouped_splitter(5, seed 42 + repeat)`). Each
`shap` cell refits the headline inner loop on a repeat-0 fold, so its AUC must
match `results-3/nested_cv_results.parquet` for that (task, PV, fold);
`shap_fold_models.csv` vs `results-3` is the end-to-end reproduction check.
Pods run scikit-learn 1.6.1 (as Colab did) but numpy 2.5.3 (pulled in by
shap 0.52); a mismatch would show up in that check.

## 5. Hard constraints (unchanged)

Correctness over favourable results; never report a number not produced by
a script and written to results; leakage is a hard error; the two
permutation nulls are not interchangeable; no variable interpreted without a
codebook citation; STRATUM is a design variable; every response-letter claim
must be true of a completed run.

## 6. Scope table — round 2 (fill from `cells_reconciliation.csv`)

| Analysis | Configured | Completed | Evidence |
|---|---|---|---|
| Selector comparison | 12,000 cells | _pending_ | `r2_tables/sel_*.csv` |
| VLPSO stability/sensitivity | 165 cells | _pending_ | `r2_tables/vlstab_*.csv` |
| Permutation, unrestricted / within-school | 100 / 30 (+ observed) | _pending_ | `r2_tables/perm_summary.csv` |
| SHAP across PVs × folds; LIME at PV1 | 150 cells | _pending_ | `r2_tables/shap_*.csv`, `lime_*.csv` |
| External validation ESP→PRT | 30 cells | _pending_ | `r2_tables/ext_*.csv` |
| BRR standard errors | 30 cells | _pending_ | `r2_tables/brr_*.csv` |
| EDA | 1 cell | _pending_ | `cells/eda/*` |
| Independent seeds for the headline (10) | 10 | **not run** (repeat-to-repeat spread 0.0006–0.0012 already reported; not requested by the reviewers) | — |
