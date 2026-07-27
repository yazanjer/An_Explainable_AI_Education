# Verification Report

Stage 10 checks for the rebuilt VLPSO-XAI / PISA 2018 analysis. Every result below was produced by running code, not by inspection. Where a check could not be completed, it says so rather than being marked passed.

**Test suite: 309 tests passing** (286 + 23 VLPSO, run separately for time).

---

## 1. Leakage regression tests — **PASS**

Injecting a forbidden column into the design matrix must raise `LeakageError`.

| Family | Examples tested | Result |
|---|---|---|
| Outcome | `math_score`, `math_category`, `label` | raises |
| Plausible values | `PV1MATH`, `PV3MATH`, `PV10MATH`, `PV1READ`, `PV7SCIE` | raises |
| Weights | `W_FSTUWT`, `W_FSTURWT1`, `W_FSTURWT80`, `SENWT` | raises |
| Design | `STRATUM`, `STRATUM_ESP9033` | raises |
| Identifiers | `CNTSCHID`, `CNTSTUID`, `CNT`, `CYC`, `NatCen`, `SUBNATIO`, `OECD` | raises |
| Negative control | `noise_control` | raises |
| **Adversarial** | `pv1math`, `MATH_SCORE`, `' math_score'`, `math_score\n`, `math_score_z`, `log_math_score`, `PV1MATH_imputed`, `PV1MATH_1.0`, `math_category_Low`, `missingindicator_math_score`, `cntschid` | raises |
| **OECD composites** | `ESCS`, `HISEI`, `PARED`, `MISCED`, `FISCED`, `HOMEPOS`, `CULTPOSS` | raises |
| Legitimate predictors | `ST013Q01TA`, `ST166Q03HA`, `ST004D01T_1.0`, `missingindicator_ST013Q01TA` | pass through |

70 assertions in `tests/test_leakage_guard.py`. Zero false positives on the 31 allowlisted predictors.

Also verified: reproducing the original `select_dtypes(include=['object'])` idiom leaves `math_score` and `PV1MATH` in the frame, and the guard then raises on it.

---

## 2. Isolation test — **PASS, after the first version FAILED**

This check initially gave a false pass and is the most important entry in this report.

**The original isolation tests were vacuous.** An independent audit mutated `nested_cv.py` three ways and all 104 tests passed. We reproduced this ourselves.

| Mutation | Old suite | New suite |
|---|---|---|
| A: threshold selected on the **outer test fold** | passed | **caught** |
| B: `GridSearchCV` fitted on the **full dataset** | passed | **caught** |
| C: model chosen by **outer-test AUC** | passed | **caught by 3 tests** |

The old tests were proxies that could not fail: one hand-fitted a selector on training rows and asserted it had not seen test rows (a tautology that never called `run_nested_cv`); another asserted only that thresholds *varied* across folds, which they do when tuned on test.

`tests/test_leakage_detection_power.py` now runs the harness end to end and is **calibrated against a known-leaky reference implementation**, so a test that loses its power fails loudly rather than passing silently:

* pure-noise labels through `run_nested_cv` → AUC within 0.08 of 0.50;
* a sentinel predictive only on **one fold's** test rows is never selected, and fold-0 AUC stays at chance;
* the paired calibration test asserts the leaky reference **does** beat chance;
* the reported model equals the inner-loop argmax, from the recorded per-model inner scores.

Source restored and hash-verified identical to the pre-mutation file (`md5 b0992664dd39974a774681534fe24df8`).

---

## 3. Permutation sanity — **PASS**

| Null | n | mean AUC | SD | SE of mean | Expected | Result |
|---|---|---|---|---|---|---|
| **Unrestricted** (leakage test) | 20 | **0.5029** | 0.0072 | 0.0016 | 0.50 ± 0.03 | **PASS** |
| Within-school (composition preserved) | 14 | 0.6192 | 0.0048 | 0.0013 | above chance | n/a |

Labels are permuted within outer-training folds and the complete pipeline — imputation, scaling, feature selection, tuning, fitting — is re-run per permutation.

**The two nulls are not interchangeable.** Our first implementation permuted within school *and* asserted the result should be 0.50; those are inconsistent, and the mismatch produced a false leakage alarm at 0.619. Within-school permutation preserves each school's class composition, so any feature predicting *what kind of school* a student attends still predicts the permuted label. `assert_permutation_null_is_chance` now refuses to accept a within-group null.

The corrected pair yields a substantive decomposition: of the 0.360 AUC above chance, **32.3% is between-school composition and 67.7% is within-school discrimination**.

Caveat: the permutation p-value is 0.048, which is the floor at n = 20 (1/(n+1)). The configured budget of 100 is required to report anything smaller.

---

## 4. Effect-size unit tests — **PASS**

Reproduced against analytically known values:

| Case | Expected | Obtained |
|---|---|---|
| `d([1,2,3,4,5], [3,4,5,6,7])` | −1.264911064067 | exact to 1e-12 |
| μ-difference 1.0 at σ = 1 (n = 300k) | 1.000 | 0.9986 |
| μ-difference 2.0 at σ = 2 | 1.000 | 1.0052 |
| μ-difference 0.5 at σ = 1 | 0.500 | 0.5011 |
| identical samples | 0.000 | 0.000 |
| Hedges' *g* | \|g\| < \|d\| | holds |

Nadeau–Bengio verified more conservative than the naive paired *t* (smaller \|t\|, larger p, ratio = 0.25 for 8000/2000). Holm and Benjamini–Hochberg verified monotone with BH ≤ Holm. `interpret_d` classifies **0.13–0.18 as negligible** — the values `main.tex:651` calls "large effects".

---

## 5. Determinism — **PASS**

Two full quick-mode runs with the same seed:

```
identical  : True
compared   : 140 artefacts
differing  : 0
```

Every artefact's SHA256 is byte-identical. `results/manifest.json` records SHA256, generating script, git commit and config hash per artefact.

---

## 6. Cold-start Colab run — **NOT RUN**

Cannot be verified from this environment: the OECD download is blocked by network policy here, and `shap` / `lime` could not be installed on the available connection. `RUN_ALL.ipynb` is written, syntax-checked and committed with outputs stripped, but **has not been executed end to end on a fresh Colab runtime**. This must be done before submission.

Known issues already found and fixed during partial Colab runs:

* `git clone` fails (exit 128) into a pre-existing Drive folder → now detected, falls back to fetch-in-place;
* exact `numpy==1.26.4` pin broke the `pyreadstat` NumPy-2 ABI → floor raised to `>=2.0`, with automatic repair and a restart prompt;
* `skrebate>=0.62` was **unsatisfiable** (PEP 440 reads 0.62 > 0.8.2) and aborted the whole install, silently leaving `pyreadstat` missing → corrected to `>=0.8`, and requirements split into core (fatal on failure) and optional (non-fatal, with documented fallbacks).

---

## 7. Coverage matrix — editor comment → code → result → manuscript

| # | Comment | Code | Result artefact | Status |
|---|---|---|---|---|
| 1 | Audit pipeline; nested CV required | `evaluation/nested_cv.py`, `models/pipeline.py` | `results/tables/table_headline_progression.csv` | done |
| 2 | Rule out outcome leakage; predictor list | `data/features.py`, `config/predictor_allowlist.yaml` | `results/audit/leakage_delta.csv`, `single_variable_auc_screen.csv` | done |
| 3 | Describe the PISA sample | `data/outcome.py`, `data/design.py`, `data/ingest.py` | `results/sample/sample_flow.csv`, `brr_descriptives.csv`, `pv_category_instability.csv` | done |
| 4 | Strengthen validation | `evaluation/metrics.py`, `permutation.py`, `stability.py` | `table_bootstrap_ci.csv`, `table_permutation.csv`, `table_selection_stability.csv`, `table_external_validation.csv` | done |
| 5 | Comparisons and ablations | `selection/{vlpso,bpso,filters}.py` | `table_selector_comparison.csv`, `table_paired_contrasts.csv` | **provisional** — 3 folds, reduced PSO budget |
| 6 | Correct the effect sizes | `evaluation/effect_size.py` | `table_paired_contrasts.csv` | done |
| 7 | Revise explainability | `explain/{shap_global,shap_local,lime_local,consistency}.py`, `data/codebook.py` | `table_permutation_importance.csv` | **code done, SHAP/LIME NOT EXECUTED** |
| 8 | Remove causal/temporal claims | `explain/consistency.py::CAUSAL_CAVEAT` | — | code done; **`main.tex` not yet edited** |
| 9 | Clarify algorithm and contribution | `selection/vlpso.py::algorithm_report()` | convergence curves per run | done |
| 10 | Consistency and presentation | — | — | **`main.tex` not yet edited** |

**Three cells are not green.** Comment 5 is provisional, comment 7's SHAP/LIME numbers do not exist yet, and comments 8 and 10 require manuscript edits that have not begun.

---

## 8. Independent review — **COMPLETED, 15 findings, 8 resolved**

An adversarial audit was run against the rebuilt pipeline with no access to our reasoning. It confirmed by execution that `run_nested_cv` does not leak (noise canary AUC 0.4877), that no `fit` occurs outside a `Pipeline`, and that the BRR divisor, Rubin algebra, Holm/BH, Nadeau–Bengio, Kuncheva and the corrected FNR are all right.

It also found real defects in our rebuild.

| ID | Severity | Status |
|---|---|---|
| C1 | CRITICAL | **FIXED** — isolation tests were vacuous (see §2) |
| C2 | CRITICAL | **FIXED** — Rubin `U` now design-based. *The auditor's predicted magnitude was wrong*: the unweighted CI barely moved, but the weighted variance was understated 2.4× and that CI is now visibly wider |
| C3 | CRITICAL | **FIXED** — SHAP/LIME rebuilt on the fitted pipeline's feature space, school-grouped split |
| C4 | CRITICAL | **FIXED** — BRR now actually computed; SE is **2.28×** the naive i.i.d. value |
| M1 | MAJOR | **FIXED** — denylist hardened against 20 adversarial names |
| M3 | MAJOR | **FIXED** — checkpoint fingerprint covers selector, params, features, weights |
| M6 | MAJOR | **FIXED** — imputation indicators no longer counted as selected features |
| M8 | MAJOR | **FIXED** — `scripts/run_all.py` written and verified end to end |
| M9 | MAJOR | **FIXED** — permutation budget raised to 20; SE of the null mean now 0.0016 |
| M2 | MAJOR | open — notebook 05 pools predictions across methods/repeats/PVs |
| M4 | MAJOR | open — paired *d* over 3 folds labelled "large" |
| M5 | MAJOR | open — silent degradations (`error_score=nan`, bare excepts, Brier on SVM margins) |
| M7 | MAJOR | open — VLPSO/BPSO wrapper fitness uses ungrouped inner CV |
| M10 | MAJOR | open — budget claims exceed what was run |
| M11 | MAJOR | open — BPSO capped at 15 features while VLPSO is uncapped |

**M11 matters for the substantive conclusion.** The current finding that VLPSO underperforms BPSO is partly confounded by BPSO being hard-capped while VLPSO was not. That comparison must be re-run matched before the paper commits to repositioning its contribution.

---

## Bugs found in the rebuild itself

Disclosed because they are the same class of error as the original defect.

1. **Feature names lost inside the `Pipeline`.** The imputer and scaler emit unnamed arrays, so selectors reported `x3`, `x17` instead of PISA item codes — the audit's C.2 name/position confusion arriving by a different route, and worse because `add_indicator` shifts positions. Fixed with `set_output(transform="pandas")`; regression test added.
2. **Checkpoints reloaded across incompatible configurations.** A 3-fold run's checkpoints were reused by a 5-fold run. Fixed with a configuration fingerprint in the filename.
3. **The length-adaptation ablation was silently inert.** With `max_iter=8` and the default `beta_stagnation=9`, adaptation could never fire and every arm returned byte-identical results — it would have been written up as "variable-length search makes no difference" when it had never run. Now raises.
4. **`stage_sample` produced duplicate columns.** Concatenating PV categories onto a cached frame that already had them made `prim["cat_pv1"]` return a 2-column DataFrame, so every weighted statistic broadcast wrongly. Caught on the first end-to-end run; now asserted.

---

## Conclusion

Checks 1–5, 7 (partial) and 8 pass. **Check 6 has not been run**, and three cells of the coverage matrix are incomplete: the selector comparison is provisional, the SHAP/LIME results do not exist yet, and the manuscript itself has not been edited.

**`RESPONSE_TO_EDITOR.md` must not be sent in its current form.** It cites budgets — 5×5 folds, ≥10 seeds, ≥1,000 SHAP instances, 100 permutations — that only `scripts/run_all.py --config default` delivers.
