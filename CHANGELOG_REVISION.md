# Changelog — revision of *An Explainable AI Framework for Student Performance Prediction Using Variable-Length Particle Swarm Optimization in Educational Data Mining*

Submitted to *Applied System Innovation* (MDPI). Returned before peer review with a demand for major methodological revision.

Every entry states: **the reviewer comment it answers**, **the old behaviour**, **the new behaviour**, and **the numerical consequence**. Where a consequence cannot yet be quantified because the analysis has not been run, it says so explicitly rather than guessing.

Status legend — `DONE` implemented and tested · `BUILT` implemented, awaiting execution on real data · `PENDING` not yet implemented · `NOT RUN` requires data we do not yet have.

---

## 0. The headline finding, stated plainly

**The editor's leakage hypothesis is correct, and the leakage is direct rather than subtle.**

`codes/data_preparation.py:253` built the design matrix as:

```python
categorical_columns = df.select_dtypes(include=['object']).columns
X = df_exp.drop(columns=list(categorical_columns) + ['label']).fillna(0)
```

`select_dtypes(include=['object'])` removes only *string* columns. The outcome variable `math_score` is `float64` and survived into `X`. So did all ten `PV1MATH`–`PV10MATH` columns, every reading and science plausible value, `W_FSTUWT`, the eighty BRR replicate weights, and `STRATUM`.

The label is a hard threshold on `math_score` (`data_preparation.py:157-180`). A decision stump on that single column classifies perfectly by construction. **The reported AUC of 1.0000 in Table 2 is this, and nothing else.**

The manuscript interpreted the artefact substantively, at `main.tex:520`: *"extreme performance levels are governed by a very limited number of highly discriminatory features."* The highly discriminatory feature was the outcome.

The repository contains its own control. `ensemble.ipynb` cell 14 defines a local `prepare_data` with an explicit whitelist containing no PV columns and no `math_score`. Manuscript **Table 1** (accuracy 0.830–0.854, AUC 0.912–0.931 on Low vs. High) is consistent with that clean path; **Table 2** (0.95–1.00) comes from the leaking path. The performance jump the editor flagged is the difference between two `prepare_data` implementations, not the difference between feature-selection methods.

### Reproduced empirically on the real data (Spain, N = 35,943, PISA 2018) `DONE`

**Experiment 1 — a decision stump on `math_score` alone.** Depth-1 tree, 5-fold CV:

| Task | n | Accuracy | AUC |
|---|---|---|---|
| Low vs. Medium | 33,760 | **1.0000** | **1.0000** |
| Medium vs. High | 20,300 | **1.0000** | **1.0000** |
| Low vs. High | 17,826 | **1.0000** | **1.0000** |

**Experiment 2 — mutual-information ranking of the old feature matrix** (147 columns, Low vs. High). The top eleven features are `math_score` followed by all ten `PV*MATH` columns; ranks 12 onward are the science and reading plausible values. **The first genuine questionnaire item ranks 32nd of 147.** Because `_init_population` seeds every particle with `feature_ranking[:length]`, the entire initial VLPSO population consisted of outcome variables.

**Experiment 3 — old leaking matrix vs. clean allowlist**, identical models and identical flat 5-fold CV, so the only difference is which columns are in `X`:

| Task | Model | Acc (old) | Acc (clean) | AUC (old) | AUC (clean) | ΔAUC |
|---|---|---|---|---|---|---|
| Low vs. High | LogisticRegression | 1.0000 | 0.9162 | 1.0000 | 0.9210 | 0.079 |
| Low vs. High | RandomForest | 1.0000 | 0.9200 | 1.0000 | 0.9159 | 0.084 |
| Low vs. High | DecisionTree | 1.0000 | 0.8989 | 1.0000 | 0.8311 | 0.169 |
| Low vs. Medium | LogisticRegression | 0.9970 | 0.7052 | 1.0000 | 0.7763 | 0.224 |
| Low vs. Medium | RandomForest | 0.9999 | 0.7123 | 1.0000 | 0.7804 | 0.220 |
| Low vs. Medium | DecisionTree | 1.0000 | 0.6884 | 1.0000 | 0.7426 | 0.257 |
| Medium vs. High | LogisticRegression | 0.9960 | 0.8922 | 0.9999 | 0.7423 | 0.258 |
| Medium vs. High | RandomForest | 0.9998 | 0.8922 | 1.0000 | 0.7021 | 0.298 |
| Medium vs. High | DecisionTree | 1.0000 | 0.8743 | 1.0000 | 0.6700 | 0.330 |

Written to `results/audit/leakage_demonstration.csv`, `results/audit/leakage_delta.csv` and `results/audit/mi_ranking_old_matrix.csv`.

**This confirms the audit's central prediction.** The clean Low vs. High AUC of 0.916–0.921 sits squarely in the range of manuscript **Table 1** (0.912–0.931), while the leaking path reproduces **Table 2** (1.0000). The performance jump attributed to VLPSO is the difference between two `prepare_data` implementations.

Two further points the revision must state plainly:

* **The adjacent-category tasks are far weaker than Low vs. High.** Honest AUC is 0.74–0.78 for Low vs. Medium and 0.67–0.74 for Medium vs. High. The manuscript's framing does not survive this.
* **These clean figures are still optimistic.** They come from flat, ungrouped, unweighted CV on a single averaged score. Adding school-level grouping, survey weights and per-PV analysis will lower them further. They are an upper bound, not the final result.

**Consequence:** every number in Tables 1–5 and Figures 2–25 is regenerated. The revised paper is framed around interpretability and methodological rigour, not headline accuracy.

---

## 0b. The honest numbers, first pass `DONE`

Nested CV, `StratifiedGroupKFold` on `CNTSCHID`, 5 outer folds, 5 inner folds, model selected **inside** the inner loop, threshold chosen on inner folds only, PV1, N = 29,786 Spanish students in 1,087 schools.

| Task | AUC leaking | AUC clean, flat CV | **AUC nested + school-grouped** | weighted | fold SD | Δ from leakage |
|---|---|---|---|---|---|---|
| Low vs. High | 1.0000 | 0.8893 | **0.8766** | 0.8752 | 0.0140 | **0.123** |
| Low vs. Medium | 1.0000 | 0.7664 | **0.7439** | 0.7386 | 0.0062 | **0.256** |
| Medium vs. High | 1.0000 | 0.7048 | **0.6920** | 0.6938 | 0.0082 | **0.308** |

Balanced accuracy: 0.794 / 0.680 / 0.637. `GradientBoosting` was selected in 13 of 15 folds and `RandomForest` in 2 — model choice is stable across folds, which is itself reassuring.

Note the two separate corrections. Removing the leaking columns costs 0.11–0.30 AUC. Moving from flat CV to school-grouped nested CV costs a further **0.013–0.023** — smaller than expected, but real, and in the expected direction.

**The Medium vs. High task is weak.** AUC 0.692 with balanced accuracy 0.637 on a 16.2% positive class. The submitted manuscript reported 1.0000 for this task. No reframing rescues it; it should be reported as the weak result it is.

### PV category instability, measured `DONE`

Survey-weighted, N = 29,786, official cut points, all ten plausible values:

* **69.7% of students receive a different proficiency category depending on which plausible value is used.**
* Only **30.3%** are assigned the same category by all ten PVs.
* Mean modal-category share is 0.819.

Averaging the ten PVs and thresholding the mean — `data_preparation.py:164` — assigns every one of those students a single confident category. This is the strongest single argument for the per-PV / Rubin's-rules procedure, and it gets its own table in the revision.

### Single-variable AUC screen `DONE`

Run over all 31 allowlisted predictors on Low vs. High. **Nothing flagged**; the maximum single-variable AUC is **0.7472** (`ST013Q01TA`, books in the home). For comparison, `math_score` alone scores 1.0000. The screen therefore separates the clean feature set from the leaking one exactly as intended.

Two incidental findings:

* `ST166Q03HA` ("Click on the link to fill out the form as soon as possible") has a *signed* AUC of **0.2727** — strongly inverted. Endorsing the phishing link predicts LOW proficiency, confirming that this item is reverse-scored relative to the rest of the block, as recorded in the codebook.
* `ST013Q01TA` (books in the home) is the strongest single predictor at 0.747. It is the item the manuscript mislabelled as "availability of education materials".

### Sample flow `DONE`

| Step | N students | N schools |
|---|---|---|
| PISA 2018 student file, all countries | 612,004 | — |
| `CNT == 'ESP'` | 35,943 | 1,089 |
| Missingness ≤ 400 of 1,119 columns (primary) | **29,786** | 1,087 |
| Missingness ≤ 200 (sensitivity) | 14,728 | 1,056 |
| Missingness ≤ 600 (sensitivity) | 34,953 | 1,089 |

**The submitted code's expectations do not match the data.** `data_preparation.py:189` hard-codes 26,657 as the expected N and `validate_data_quality` hard-codes class counts of Low 10,243 / Medium 14,484 / High 1,930. The actual analytic sample is **29,786** with Low 12,072 / Medium 15,653 / High 2,061 at the official cut points. Because the check only `print`s a warning rather than raising, the discrepancy was never caught.

The missingness threshold is also highly consequential and was never justified: plausible alternatives move N by a factor of 2.4 (14,728 to 34,953). Sensitivity at both is reported.

The cut-point correction (482 → 482.38, 607 → 606.99) reclassifies **55 students**.

Written to `results/sample/sample_flow.csv`, `results/audit/single_variable_auc_screen.csv`, `results/tables/table_headline_progression.csv`.

---

## 0c. Selector comparison and ablations, first pass `PROVISIONAL`

Low vs. High, PV1, 3 school-grouped outer folds, LogisticRegression-L2, reduced PSO budget (population 14, 12 iterations vs. the configured 60/100). **These are provisional and underpowered; the full specification is 5 outer x 5 repeats x 10 PVs.** They are recorded now because their direction is unlikely to reverse.

| Method | AUC | SD | Features | Jaccard stability | Seconds |
|---|---|---|---|---|---|
| **none** (all 61) | **0.8748** | 0.0053 | 61.0 | 1.000 | 0.8 |
| relieff | 0.8630 | 0.0065 | 12.0 | 0.682 | 1.3 |
| mutual_info | 0.8589 | 0.0079 | 12.0 | 0.758 | 3.8 |
| **bpso** | 0.8589 | 0.0039 | 12.0 | 0.500 | 33.1 |
| symmetric_uncertainty | 0.8581 | 0.0056 | 12.0 | **1.000** | 0.8 |
| chi2 | 0.8579 | 0.0043 | 12.0 | 0.897 | 0.2 |
| **vlpso** | **0.8440** | 0.0358 | 11.3 | **0.235** | 19.8 |

**The proposed method finishes last.** VLPSO scores below every baseline including ordinary BPSO, and no feature-selection method beats using all 61 features. VLPSO is also by far the least stable: its selected subsets share only 23.5% of features across three folds, and subset size ranges from 2 to 17. Symmetric uncertainty selects an identical subset every fold.

Kuncheva's index is undefined (NaN) for VLPSO because it requires equal-sized subsets and VLPSO's vary — which is itself the finding.

### Ablations

| Arm | AUC | Features | Reading |
|---|---|---|---|
| vlpso (grow + shrink) | 0.8440 | 11.3 | reference |
| vlpso, shrink-only | **0.8607** | 14.0 | **the variable-length mechanism HURTS** |
| vlpso, no cardinality penalty | **0.8723** | 26.7 | the penalty the manuscript claims costs 0.028 AUC |
| vlpso, MI ranking instead of SU | 0.8449 | 7.3 | SU vs. MI: no material difference (0.0009) |
| vlpso, random init | 0.8535 | 10.7 | MI/SU-seeded init is not better than random |

Shrink-only — which is what the submitted code actually did — outperforms genuine bidirectional variable-length search. The symmetric-uncertainty component the manuscript describes makes no difference against mutual information.

### Paired contrasts

VLPSO vs. each baseline across matched folds, Nadeau-Bengio corrected resampled *t*, Holm-corrected within task. Point estimates all favour the baselines (mean differences -0.001 to -0.031; paired *d* from -0.03 to -0.96), but **every adjusted p-value equals 1.000**. With three folds the design has essentially no power, and the honest statement is that these comparisons are currently inconclusive in favour of nothing. The full 5x5 design is required before any claim is made in either direction.

**Consequence for the manuscript.** On this evidence the paper cannot claim that VLPSO improves predictive performance, that variable-length search beats fixed-length, that the cardinality penalty helps, or that symmetric uncertainty contributes anything. Path A was implemented in good faith and the honest answer is that it does not currently pay off. The defensible contribution is the leakage-proof evaluation protocol and the multi-level XAI analysis, not the optimiser.

---

## 0d. Independent audit of the REBUILD `IN PROGRESS`

We commissioned an adversarial audit of the revised codebase, conducted with fresh eyes and no access to our reasoning. It found serious defects in our own rebuild. They are recorded here in full, because concealing them would be inconsistent with everything else in this document.

### AUDIT-C1 — The isolation tests were vacuous `FIXED`

The auditor mutated `nested_cv.py` to introduce three catastrophic leaks and **all 104 tests passed**:

| Mutation | All tests pass? |
|---|---|
| Threshold selected on the **outer test fold** | **yes** |
| `GridSearchCV` fitted on the **full dataset** including outer test | **yes** |
| Best model chosen by **outer-test AUC** | **yes** |

The tests were proxies that could not fail. `test_sentinel_column_is_never_selected` hand-fitted a selector on training rows and asserted it had not seen test rows — a tautology that never called `run_nested_cv`. `test_threshold_comes_from_inner_folds` asserted only that thresholds varied across folds; thresholds tuned *on* the outer test fold vary too. `test_nested_cv_reports_disjoint_schools` asserted a property true by construction of any `GroupKFold`.

**This is the same failure mode that shipped the original paper**: a guarantee asserted in a docstring and "tested" by something incapable of detecting its violation. We reproduced the auditor's mutation ourselves and confirmed it.

**Fix.** `tests/test_leakage_detection_power.py` runs the harness **end to end** and is calibrated against a known-leaky reference implementation, so it has demonstrated power. It contains a pure-noise null through `run_nested_cv`, a single-fold sentinel, a model-selection check against the recorded inner scores, and — critically — paired tests asserting that the leaky reference *does* beat chance, so a test that loses its power fails loudly rather than passing silently.

Verified by re-running all three mutations:

| Mutation | Now caught by |
|---|---|
| A: threshold on outer test | `test_our_threshold_does_not_beat_chance_on_noise` |
| B: GridSearchCV on full data | `test_sentinel_through_the_full_harness` |
| C: model selection by outer-test AUC | three tests |

Source restored and hash-verified identical to the pre-mutation file (`md5 b0992664…`).

Writing these tests exposed two further construction errors of our own: a sentinel made predictive on *every* fold's test rows is predictive everywhere (each row is a test row exactly once), and a sentinel placed with a different split seed than the harness uses lands in training folds. Both are documented in the test file so they are not repeated.

### Confirmed correct by the audit

The auditor verified independently, by running code rather than reading it, that: `run_nested_cv` does **not** leak (pure-noise canary returns AUC 0.4877); there is **no `fit` outside a `Pipeline`** anywhere in `src/`; the BRR divisor `R(1−k)² = 20` is right; the Rubin algebra reproduces to six significant figures; Holm, Benjamini–Hochberg, Nadeau–Bengio, Kuncheva and the corrected FNR are all correct; and `cluster_bootstrap_ci` returns intervals 2.02× wider than student-level resampling on genuinely clustered simulated data.

### Outstanding audit findings `NOT YET FIXED`

Recorded now so they are not lost. None is yet reflected in the reported numbers.

| ID | Severity | Finding |
|---|---|---|
| C2 | CRITICAL | Rubin's within-imputation variance `U` is the squared SEM across 3 CV folds, not a design-based sampling variance. Every reported CI, df and FMI is invalid; the CI is far too narrow. The weighted row also reuses the unweighted `U` verbatim. |
| C3 | CRITICAL | Notebook 06 passes the bare classifier raw, unscaled data while it was fitted on imputed/scaled/indicator-augmented data, and uses a student-level `train_test_split` rather than school grouping. The whole explainability section must be rebuilt on the fitted `Pipeline` with a grouped split. |
| C4 | CRITICAL | `SurveyDesign` and `brr_metric_se` have **zero call sites**. No reported number uses BRR standard errors, contrary to the claim in the response letter. |
| M1 | MAJOR | The denylist is case-sensitive with no whitespace tolerance: `pv1math`, `math_score `, `math_score_z`, `PV1MATH_1.0`, `math_category_Low`, `ESCS`, `HISEI` all slip through. The allowlist currently saves us; the denylist would not. |
| M2 | MAJOR | Notebook 05 pools predictions across methods, repeats and PVs into one AUC and one CI, duplicating each student up to 35 times, and globs checkpoints without a config-fingerprint filter. |
| M3 | MAJOR | The checkpoint fingerprint omits the selector, its hyperparameters, the allowlist version and `sample_weight`. |
| M4 | MAJOR | Paired Cohen's *d* over 3 CV folds is labelled "large" — the same category error as `main.tex:651`, by a different route. |
| M5 | MAJOR | Multiple silent degradations: `error_score=np.nan`, folds skipped on failure, `selected` defaulting to all columns, bare `except` swallowing real errors, Brier score computed on `LinearSVC` decision-function margins. |
| M6 | MAJOR | `missingindicator_*` pseudo-features are counted as "selected features" (n_sel 61 = 31 items + 30 indicators). |
| M7 | MAJOR | VLPSO/BPSO wrapper fitness uses ungrouped `StratifiedKFold` inside the training fold. |
| M8 | MAJOR | `scripts/run_all.py` and `scripts/make_manuscript_assets.py` do not exist; most result tables were produced by session code not in the repository. |
| M9 | MAJOR | The permutation gate ran 7 times with a ±0.03 tolerance; the config specifies 100. |
| M10 | MAJOR | The response letter claims 5×5 folds, ≥10 seeds, ≥1,000 SHAP instances and ≥30 LIME repeats. Every committed result is 3 folds, 1 repeat, quick-mode. |
| M11 | MAJOR | BPSO is hard-capped at 15 features while VLPSO is uncapped, so the headline comparison is confounded. |

### Audit findings now RESOLVED

**AUDIT-M1 — denylist hardened `FIXED`.** Matching is now case-insensitive and whitespace-insensitive, and covers derived and one-hot forms via `_candidate_forms()`. All 20 adversarial names the auditor listed are caught (`pv1math`, `' math_score'`, `math_score_z`, `log_math_score`, `PV1MATH_1.0`, `math_category_Low`, `missingindicator_math_score`, `cntschid`, …) with zero false positives on legitimate predictors. OECD composites (`ESCS`, `HISEI`, `PARED`, `HOMEPOS`, `CULTPOSS`, …) are added — `ESCS` in particular enters the PISA conditioning model that generates the plausible values. 26 new adversarial test cases.

**AUDIT-C2 — Rubin's `U` corrected `FIXED`, and the auditor's magnitude estimate was wrong.** `U` is now the design-based sampling variance of the AUC from a school-clustered bootstrap, not the squared SEM across CV folds. The auditor predicted the old value was understated "by roughly an order of magnitude in variance". It was not:

| Variant | U | SE | 95% CI | FMI |
|---|---|---|---|---|
| OLD (fold SEM) — invalid method | 0.000015 | 0.0057 | [0.8619, 0.8851] | 0.541 |
| CORRECTED (school-clustered bootstrap) | 0.000014 | 0.0056 | [0.8621, 0.8849] | 0.563 |
| CORRECTED, survey-weighted | 0.000033 | 0.0076 | [0.8567, 0.8873] | 0.435 |

The unweighted interval barely moves. **The auditor was right about the weighted row**, however: reusing the unweighted `U` understated the weighted variance by a factor of 2.4, and the corrected weighted CI is visibly wider. The diagnosis was correct even where the predicted magnitude was not.

**AUDIT-C4 — BRR standard errors now actually computed `FIXED`.** `SurveyDesign` is constructed and `brr_standard_error` called in `run_all.py::stage_sample`, producing `results/sample/brr_descriptives.csv` for 23 quantities using all 80 Fay replicates. The result justifies the apparatus:

> Mean `ST013Q01TA` (books in the home): estimate 3.3418, **BRR SE 0.01795** vs **naive i.i.d. SE 0.00788** — a ratio of **2.28×**. Kish design effect 2.13.

Naive standard errors understate uncertainty on this design by more than a factor of two.

**AUDIT-M8 — `scripts/run_all.py` now exists and runs `FIXED`.** Eight resumable stages from raw SPSS to manifest. Verified end to end on the real data: `N=29,786, 1,087 schools, BRR SEs for 23 quantities, manifest written, 0.2 min` for the quick stages. `src/vlpso_xai/data/ingest.py` added. Its first run immediately exposed a genuine bug — `stage_sample` concatenated PV categories onto a cached frame that already carried them, producing duplicate column names so that `prim["cat_pv1"]` silently returned a 2-column DataFrame and every weighted statistic broadcast wrongly. Now guarded by an explicit assertion.

**AUDIT-C3 — explainability rebuilt on the correct feature space `FIXED`.** `transformed_frames()` pushes both frames through the *fitted* pipeline's preprocessing so the explainer sees exactly the matrix the estimator was fitted on. Demonstrated concretely: a raw 8-column frame against a classifier fitted on 9 features (the missing indicator). `global_shap` now raises on both misuses — a whole `Pipeline`, or frames in the wrong space — and validates *before* importing SHAP so the error is the usage error rather than `ModuleNotFoundError`. Notebook 06 uses a **school-grouped** split rather than `train_test_split`, and SHAP and LIME now share one estimator and one feature space, so their rank agreement is meaningful.

**AUDIT-M6 — imputation artefacts no longer counted as selected features `FIXED`.** `SimpleImputer(add_indicator=True)` turns a 31-item allowlist into a 61-column matrix. `BaseSelector` now exposes `selected_items_` and `n_items_` alongside the raw `n_selected_`; the manuscript reports the former. Verified: `n_selected = 4` where `n_items = 3`.

**AUDIT-M3 — checkpoint fingerprint widened `FIXED`.** It now hashes the selector class and all its hyperparameters, the full feature-name list, and whether weights were used. Verified that `Chi2Filter(k=15)` vs `k=10`, a changed feature set, and a weighted vs unweighted run all produce different fingerprints, while an identical configuration reproduces the same one.

**AUDIT-M9 — permutation gate strengthened `FIXED`.** Now **20** unrestricted and **14** within-school permutations (was 7 each):

| Null | n | mean | SD | SE of mean | Gate |
|---|---|---|---|---|---|
| Unrestricted (leakage test) | 20 | **0.5029** | 0.0072 | 0.0016 | **PASS** (\|0.5029−0.50\| = 0.0029) |
| Within-school | 14 | 0.6192 | 0.0048 | 0.0013 | n/a — expected above chance |

The SE of the null mean is 0.0016, so the ±0.03 gate can now resolve a true null of 0.53 rather than rubber-stamping it. Decomposition is stable at **32.3% between-school / 67.7% within-school** of the 0.360 above chance. The permutation p-value is 0.048, which is the *floor* at n = 20 (1/(n+1)); the full budget of 100 is required to report anything smaller.

**AUDIT-M2 — aggregation hierarchy replaces the pooled glob `FIXED`.** Notebook 05 previously globbed every `*_preds.parquet`, concatenated the lot and grouped by `task` alone. Four errors compounded in those five lines: the glob applied **no configuration filter at all**; it pooled selectors, which are different estimators; it pooled repeats, which are re-partitions of the *same* students; and it pooled plausible values, which carry *different labels* for 69.7% of students.

**Correction to the first version of this fix.** It required **one fingerprint per directory** and raised otherwise. That is not the invariant and could never be satisfied: `stage_nested` calls `run_nested_cv` once per (task, PV) on that combination's row subset, and `_config_fingerprint` hashes `sample_signature` — row count, positive count, school count, digest of school ids — so a different task or PV *necessarily* yields a different fingerprint. Run against the real Drive checkpoint directory it rejected all 30 cells of the 750-fold run and listed them as suspect. The guard was wrong, not the data.

The invariant is **one fingerprint per cell**, where a cell is `(task, method, pv)` — one call to `run_nested_cv`. On the real directory:

| Shape | Cells | Folds | Reading |
|---|---|---|---|
| 5 repeats × 5 folds | 30 | 750 | the full run: 3 tasks × 10 PVs |
| 1 repeat × 3 folds | 3 | 9 | leftover quick-mode smoke test, same directory |

So the genuine defect the old glob committed was concatenating a 3-fold smoke test into the reported estimate for three of the thirty cells. Cells are separated by **declaring the budget being reported** — `outer_splits=5, outer_repeats=5` — which is a specification, not a heuristic; dropped cells are named in a warning. `inventory()` prints everything on disk without raising, so the choice can be made from the actual contents. Note that the notebook declares the budget as a literal rather than reading it from `cfg`, because `quick.yaml` specifies 3 folds and would have selected the smoke-test cells in QUICK_MODE.

**Second correction: level 1 must not concatenate folds.** The first two versions computed one AUC over the concatenated out-of-fold predictions of a repeat. That is only valid if the per-fold models put their scores on a common scale, and in nested CV they do not — each outer fold selects its own estimator inside its own inner loop.

The 750-fold run made this visible. `medium_vs_high` showed repeat-to-repeat SD of 0.0171 against 0.0008 and 0.0006 for the other two tasks, which looked like genuine instability. It was not: the *per-fold* AUCs were near-identical across repeats (per-repeat mean 0.6956–0.6967, a range of 0.001), and model selection was stable at 43–47 GradientBoosting out of 50 per repeat. The association that settled it:

| PV | RandomForest folds (of 25) | repeat SD of pooled AUC |
|---|---|---|
| 5 | **0** | 0.0010 |
| 7 | **0** | 0.0008 |
| 1 | 2 | 0.0237 |
| 6 | 4 | 0.0202 |
| 8 | 8 | 0.0138 |
| 10 | 4 | 0.0306 |

The only two plausible values whose folds all agreed on one model family were the only two whose pooled AUC was stable. Pooling was measuring the score scales, not the discrimination. Checking a single cell was misleading and nearly closed the investigation prematurely: (pv1, rep0) gave pooled 0.6993 against fold-mean 0.6996 — because that repeat happened to be all-GradientBoosting.

Level 1 now computes the metric **within each fold and averages by fold size**, and `cluster_bootstrap_foldwise` recomputes that fold-weighted average on each school resample, respecting the clustering and the fold boundaries at once. Pooled AUC is retained as a diagnostic: `pooled_minus_fold_mean` is reported per repeat and any cell exceeding a 0.005 tolerance is logged. Verified on a replica with mixed model families in the folds — pooled repeat-SD 0.0029 for zero-RF PVs versus 0.0541 for mixed ones, an 18× ratio matching the real data's 20×, while fold-averaged SD stays flat at 0.0027 either way. Five new tests, including the control cell that initially looked clean.

A performance defect surfaced with it: the bootstrap called `classification_metrics`, which computes a dozen quantities including a confusion matrix and Brier score, when one is needed. At the full budget that is 1.5 million calls. A single-metric fast path brings a 2,000-resample repeat to ~0.4 minutes, so the full pass over 150 repeats is about an hour rather than intractable.

Two cells were also made affordable: the level-1 BCa bootstrap runs 150 times at full budget (3 tasks × 10 PVs × 5 repeats), each adding a jackknife over ~1,084 schools, so the notebook offers a `FAST_PASS` percentile pass labelled not-reportable; and the pooled-versus-correct demonstration now prints the duplication factor for free and makes the pooled bootstrap opt-in, since running it over every duplicated row is precisely the thing being criticised. Each student entered the frame up to 250 times, and the bootstrap — which resamples schools — saw each school as many times over, so the interval collapsed.

`src/vlpso_xai/evaluation/aggregate.py` replaces it with an explicit three-level hierarchy in which only the first level pools rows:

| Level | Unit | Operation | Rationale |
|---|---|---|---|
| 1 | outer folds within (task, method, pv, repeat) | concatenate | The folds partition the sample, so each student appears exactly once. AUC and the school-clustered BCa bootstrap variance are computed here and nowhere else. |
| 2 | repeats within (task, method, pv) | average the AUCs | A repeat is a re-partition of one sample, not a new sample. Its spread is partition noise, reported as a range, never added to sampling variance. |
| 3 | plausible values within (task, method) | Rubin's rules | `U` = mean within-PV sampling variance, `B` = between-PV variance. The only step yielding a publishable CI, and the only one carrying PV measurement error (FMI ≈ 0.56, so it dominates). |
| — | methods | never combined | Compared via `contrast_table`, not pooled. |

Selecting a configuration fingerprint is **mandatory**: with more than one present `load_fold_predictions` raises and lists the candidates with their fold counts rather than picking the newest, largest or first — every one of those heuristics has a failure mode in which a stale run is reported as current, which is the class of error the module exists to prevent. Legacy unfingerprinted files count as a distinct, unusable configuration. `run_nested_cv` now also embeds `cfg_fingerprint` as a *column*, because a frame that has been read and concatenated no longer knows its filename; a mismatch between filename and payload raises.

Three structural checks run before any level-1 vector is scored, because all three failure modes are silent: a **missing fold** (shrinks the sample and biases toward the surviving schools), a **non-contiguous fold set**, and **schools appearing in more than one outer fold** (the duplication M2 describes, arriving via mixed checkpoints).

The headline regression test asserts the direction of the fix: on a synthetic set with 2 PVs × 2 repeats, the pooled frame holds each student 4× and the correct hierarchy must return a **strictly wider** interval. If that ever inverts, the fix has been undone. 24 new tests in `tests/test_aggregation.py`, all on fabricated checkpoints — no PISA microdata and no 34-hour run required.

**AUDIT-M4 — effect-size magnitudes now guarded `FIXED`.** The original manuscript inflated a near-zero *d* into "large" (`main.tex:651`); the rebuild avoided that but then attached a *correct* band to an estimate with no precision, which misleads in the same direction by a different route. With J = 3 the standard error of a paired *d* is roughly √(1/J + d²/2J) ≈ 0.6, so a point estimate of 0.9 has an interval covering "negligible" and "large" at once.

`interpret_d_guarded` emits a band only when **both** hold: at least `MIN_FOLDS_FOR_MAGNITUDE = 10` matched folds, **and** the bootstrap CI for *d* (resampling folds) lies entirely inside one band. Otherwise it returns a string stating why — `indeterminate (J=3 < 10)`, `indeterminate (CI spans negligible-large)` — and that string is what belongs in the manuscript table. `PairedContrast` gains `d_ci_low`, `d_ci_high`, `underpowered` and `magnitude_unguarded`; the last is kept only so the guard itself can be audited and must never be reported.

The test for this reproduces the defect verbatim: 3 folds with a clean separation give |d| > 0.8, `magnitude_unguarded == "large"` — the old behaviour — and `magnitude == "indeterminate (J=3 < 10)"`.

**Consequence for the paper:** every contrast in the current provisional selector comparison is 3-fold, so every magnitude cell now reads `indeterminate`. That is the honest state of the comparison and it does not improve until M11 is closed and notebook 03 is re-run at full budget with BPSO and VLPSO matched.

**Partial: AUDIT-M5 — one silent degradation closed.** `contrast_table` swallowed failed contrasts with a bare `except ValueError: continue`. A dropped contrast shrinks the multiplicity family, so every surviving `p_adjusted` was wrong, and the row simply vanished from the table with no trace. Skipped pairs are now logged with the reason. The remaining M5 items (`error_score=np.nan`, folds skipped on failure, `selected` defaulting to all columns, Brier on `LinearSVC` margins) are untouched.

### Still outstanding

M5 (remaining silent degradations), M7 (VLPSO/BPSO wrapper fitness uses ungrouped inner CV), M10 (budget claims exceed what was run), M11 (BPSO capped while VLPSO is not).

**The response letter must not be sent until C3, M9 and M10 are resolved**, because it still asserts budgets (5×5 folds, ≥10 seeds, ≥1,000 SHAP instances, 100 permutations) that the committed results do not meet.

## 0e. Bugs found in the rebuild itself

Disclosed because they are the same class of error as the original defect.

1. **Feature names lost inside the `Pipeline`.** The imputer and scaler emit unnamed arrays, so selectors reported `x3`, `x17` instead of PISA item codes — the audit's C.2 name/position confusion arriving by a different route, and worse because `add_indicator` shifts positions. Fixed with `set_output(transform="pandas")`; regression test added.
2. **Checkpoints reloaded across incompatible configurations.** A 3-fold run's checkpoints were reused by a 5-fold run. Fixed with a configuration fingerprint in the filename.
3. **The length-adaptation ablation was silently inert.** With `max_iter=8` and the default `beta_stagnation=9`, adaptation could never fire and every arm returned byte-identical results — it would have been written up as "variable-length search makes no difference" when it had never run. `VLPSOSelector` now raises on that configuration.
4. **`stage_sample` produced duplicate columns.** Caught on the first end-to-end `run_all.py` run; now asserted.
5. **The `.gitignore` swallowed source code.** Unanchored `data/` and `models/` patterns match at any depth, so `src/vlpso_xai/data/` and `src/vlpso_xai/models/` — the leakage guard, codebook, outcome, design, registry and pipeline modules — were silently excluded from the first commit. Anchored to `/data/` and `/models/`.
6. **`skrebate>=0.62` was unsatisfiable.** Under PEP 440 `0.62 > 0.8.2` (62 > 8), and 0.61/0.62 are yanked, so pip aborted the entire requirements file and left `pyreadstat` uninstalled — surfacing three cells later as a confusing `ModuleNotFoundError`. Corrected to `>=0.8`, and requirements split into core (fatal on failure) and optional (non-fatal, with documented fallbacks).
7. **An exact `numpy==1.26.4` pin broke the `pyreadstat` ABI on Colab**, which ships NumPy 2. Floor raised to `>=2.0` with automatic repair and a restart prompt.

---

## 1. Audit verification

We independently verified each finding in `AUDIT_REPORT.md` against the code rather than accepting it.

| Audit finding | Verified | Note |
|---|---|---|
| A.1 `math_score` + 30 PV columns survive `select_dtypes` | **Confirmed** | `math_score` is float64; `select_dtypes(include=['object'])` cannot remove it |
| B6 VLPSO fitness is resubstitution | **Confirmed** | `feature_selection.ipynb` cell 8: `knn.fit(X_sel, y)` then `knn.predict(X_sel)` |
| H Particles are fixed-length | **Confirmed** | `particle = np.zeros(self.n_features)`; adaptation block only executes `population[i][drop] = 0` |
| C.1 Tables 3 and 4 byte-identical | **Confirmed** | same eight values at `main.tex:672-681` and `main.tex:700-709` |
| C.3 FNR formula wrong | **Confirmed** | `evaluation.py:97` divides by `(fn + tn)`; should be `(fn + tp)` |
| D Cycle mismatch | **Confirmed** | code reads `CY07_MSU_STU_QQQ.sav` (2018); `main.tex:433` says 2022 |
| F Codebook glosses inconsistent | **Confirmed, and the audit understated it** | see CB-1 below |

### Disagreements with the audit

**CB-1 — the audit was right that the item glosses are inconsistent, but did not identify what the items actually are.** We resolved them against the official instrument (`CY7_201709_QST_MS_STQ_CBA_NoNotes`, *PISA 2018 Student Questionnaire, Main Survey, computer-based*):

| Item | Manuscript gloss | Official wording |
|---|---|---|
| `ST012Q01TA` | "number of books in the household" (`main.tex:433`); "home book availability" (`main.tex:820`) | **"How many of these are there at your home? — Televisions"** (p. 12) |
| `ST013Q01TA` | "availability of education materials in the home" (`main.tex:433`) | **"How many books are there in your home?"**, 6-category ordinal (p. 13) |
| `ST166*` | "presence/absence of devices/internet access/digital literacy practices" (`main.tex:433`) | **a phishing-email situational-judgement task** within the reading framework (p. 45) |

So the paper's home-literacy / cultural-capital interpretation rests on **the number of televisions in the home**, and its "digital environment" construct is a **digital-safety reasoning task**, in which items Q01HA and Q03HA are moreover reverse-scored. Separately, `ST011` — the genuine home-educational-resources block (desk, quiet place to study, computer for schoolwork, internet link, reference books) — is present in the data and was never used.

**Consequence:** every substantive interpretation in Section 5 is rewritten. `ST011` is added to the candidate predictors. `ST166` is described as digital-safety judgement, with reverse-scoring handled explicitly.

---

## 2. Data foundation

### DF-1 — Leakage guard `DONE`
*Answers comment 2.*
**Old:** denylist by dtype; forbidden columns silently retained.
**New:** `src/vlpso_xai/data/features.py`. An **allowlist** (`config/predictor_allowlist.yaml`, one row per variable with codebook label, response scale and a stated justification) plus a denylist of 30 anchored patterns, enforced by a `LeakageError` **exception**, called at every entry point — data prep, selector `fit`, model fitting, explanation.
**Consequence:** injecting `math_score`, `PV3MATH`, `PV7READ`, `W_FSTUWT`, `STRATUM_ESP9033`, `CNTSCHID` or `noise_control` each raises. Verified. The clean matrix passes.

### DF-2 — Empirical leakage screen `DONE`
*Answers comment 2.*
**Old:** none.
**New:** `single_variable_auc_screen` computes the AUC of each candidate predictor alone, direction-free (`max(a, 1-a)`, since a perfectly inverted predictor leaks just as much). Anything above 0.95 raises unless a `signed_off_by` entry exists in the allowlist.
**Consequence:** `NOT RUN` — needs the student microdata.

### DF-3 — Plausible values `BUILT`
*Answers comment 3.*
**Old:** `df[PV1..PV10MATH].mean(axis=1)`, then thresholded. PVs are draws from a posterior; their mean has systematically deflated variance and the resulting category has no defensible standard error.
**New:** `src/vlpso_xai/data/outcome.py`. Categories derived **separately for each of the ten PVs**; the full analysis runs ten times; results combined with Rubin's rules, reporting within-imputation, between-imputation and total variance separately, plus FMI and RIV.
**Consequence, already measurable:** on simulated PVs with realistic within-student measurement error, **52% of students receive a different proficiency category depending on which plausible value is used**, and the mean modal-category share is 0.87. Thresholding the averaged score conceals this entirely. The real figure is `NOT RUN`; it gets its own table.

### DF-4 — Proficiency cut points `DONE`
*Answers comment 2.*
**Old:** `s <= 482` → Low, `s <= 607` → Medium (`data_preparation.py:167-173`).
**New:** official PISA 2018 boundaries, Level 3 = 482.38 and Level 5 = 606.99, half-open from below.
**Consequence:** students scoring in [482, 482.38) were misclassified as Low→Medium and those in [606.99, 607] as High→Medium. Verified at the boundary: 482.0 → Low, 482.4 → Medium, 606.9 → Medium, 607.0 → High. Affected N is `NOT RUN`.

### DF-5 — Survey design `BUILT`
*Answers comment 3.*
**Old:** weights never applied, and `W_FSTUWT` + `STRATUM` used as *predictors*. CV split students at random, so classmates straddled the train/test boundary.
**New:** `src/vlpso_xai/data/design.py`. `W_FSTUWT` for weighted descriptives and metrics; 80 Fay BRR replicates (k = 0.5, divisor R(1−k)² = 20) for standard errors; `StratifiedGroupKFold` grouped on `CNTSCHID` with a hard `assert_group_disjoint` check. Both weighted and unweighted metrics reported.
**Consequence:** verified on synthetic data — BRR SE recovered, Kish effective-N and design effect reported. School-grouped splitting is expected to **lower** performance relative to the submitted numbers. `NOT RUN` on real data.

### DF-6 — `noise_control` demoted to negative control `DONE`
*Answers comments 2 and 7.*
**Old:** flowed into `X`, survived selection, and was interpreted substantively at `main.tex:766` ("noise_control (−0.24) ... eroding the model's confidence").
**New:** forbidden by the guard; appended only to explicitly-flagged diagnostic runs, where its importance rank benchmarks the XAI analysis. Any real feature ranked below it is reported as indistinguishable from noise.

### DF-7 — `STRATUM` removed from predictors `DONE`
*Answers comments 2 and 7.*
**Old:** one-hot encoded as a predictor; `STRATUM_ESP9033` reported as the top SHAP contributor (+0.88) and read as a substantive regional effect (`main.tex:746`).
**New:** classified as a design variable and forbidden.
**Consequence:** the paper's single headline explainability finding is withdrawn. It was an interpretation of a sampling-design artefact.

---

## 3. Feature selection and the algorithm

### FS-1 — VLPSO rebuilt as genuinely variable-length `BUILT`
*Answers comment 9. Author decision: **Path A**.*
**Old:** fixed-length binary vectors of size `n_features`; "length" was a post-hoc cardinality cap; the adaptation mechanism only ever cleared bits, so lengths were monotonically non-increasing; different-length interaction did not arise; `alpha=7` declared and unused; `best_div = np.argmax(pbest_fitness)` misnamed; no stopping criterion.
**New:** `src/vlpso_xai/selection/vlpso.py`. Ranked-prefix encoding: particle *i* carries an explicit length `L_i` and a position in `[0,1]^{L_i}`, where dimension *d* denotes the *d*-th feature by symmetric uncertainty. Growth **and** shrinkage; new dimensions seeded from the best particle spanning that dimension with **zero** initial velocity; cross-length exemplar learning specified explicitly (`gbest_longest` / `pbest` / `inertia` fallbacks for dimensions the exemplar does not span). Direction stated: **maximised**. Convergence criterion added. `alpha` removed; misnamed variable fixed.
**Consequence:** the novelty claim now matches the implementation. `NOT RUN`.

### FS-2 — Wrapper fitness is no longer resubstitution `DONE`
*Answers comment 1.*
**Old:** `knn.fit(X_sel, y); knn.predict(X_sel)` — training accuracy on memorised data.
**New:** internal `StratifiedKFold`; there is deliberately no resubstitution option in the API.

### FS-3 — Three-term objective, as the manuscript claims `BUILT`
*Answers comment 5.*
**Old:** `gamma*acc + (1-gamma)*inter_class_distance` — two terms, **no feature-count penalty**, contradicting the abstract's claim of "minimizing the number of selected features" and Eq. (7) at `main.tex:267-277`.
**New:** `f(S) = γ·BAcc_cv(S) − λ·(|S|/p) − μ·Redundancy(S)`, maximised, with the cardinality penalty implemented and the third term operationalised as mean pairwise SU within the selected set. Each term ablatable individually and in pairs.

### FS-4 — Symmetric uncertainty implemented `DONE`
*Answers comment 5. Author decision: implement rather than delete the claim.*
**Old:** `mutual_info_classif`, with the code commenting `# Rank features using mutual information (instead of SU for now)`. The manuscript described an SU component that did not exist.
**New:** `SU(X,Y) = 2·I(X;Y)/(H(X)+H(Y))` in `src/vlpso_xai/selection/filters.py`, with equal-frequency discretisation for continuous columns.
**Consequence:** verified against analytic values — `SU(y,y) = 1.0000`, `SU(noise,y) = 0.0009`, and `SU(50-level noise, y) = 0.0004` where raw MI would be inflated by cardinality. This matters because the candidate set mixes binary (`ST011`) and 6-point ordinal (`ST013`, `ST166`) items.

### FS-5 — Index misalignment fixed `DONE`
*Answers comment 5.*
**Old:** `df_reduced = df.iloc[:, selected_features]` applied positional indices from a twice-reduced matrix back onto the original dataframe. **The columns saved as "selected" were not the columns VLPSO chose.** Compounded by cell 17 saving to `data_path` while cell 19 loaded from `results_path`.
**New:** `BaseSelector` records `feature_names_in_` at fit time; `selected_feature_names_` is the only sanctioned accessor. Selection is name-based end to end.
**Consequence:** every statement in the submitted paper about which features VLPSO chose is unverifiable and is withdrawn. A regression test covers the old bug.

### FS-6 — BPSO baseline `BUILT`
*Answers comment 5.*
**Old:** no baseline of any kind.
**New:** `src/vlpso_xai/selection/bpso.py`, matched to VLPSO on population size, iterations, inertia schedule, learning coefficients, wrapper learner, internal-CV protocol and all three objective weights. **Only** the length mechanism differs. Run at five fixed cardinalities spanning the VLPSO range.

### FS-7 — Filter baselines `DONE`
χ², mutual information, symmetric uncertainty and ReliefF as standalone selectors (`filters.py`), all inside the same harness. ReliefF uses `skrebate` when available and otherwise a documented internal implementation; which path ran is recorded.

---

## 4. Statistics

### ST-1 — `calculate_effect_size` deleted `DONE`
*Answers comment 6.*
**Old:** `codes/evaluation.py:39-57` computed the difference in the **grand mean of all standardised feature columns** between rows the model *predicted* positive and negative, divided by an average of per-column pooled SDs. It is not an effect size for classifier performance, not an effect size for any group contrast, and it used predicted rather than true labels. Because standardised features average to ≈0 by construction, the result was necessarily near zero — which is the entire explanation for the 0.05–0.18 range.
**New:** the function is gone. Two clearly distinct quantities replace it, never reported in the same table:
  **(a)** `paired_method_contrast` — selector vs. selector across matched outer folds, with the **Nadeau–Bengio corrected resampled *t*** (the naive paired *t* is badly anti-conservative under overlapping CV training sets), 95% CIs, and Holm–Bonferroni correction within task.
  **(b)** `group_standardised_difference` — standardised mean differences between **true** proficiency groups on individual predictors, survey-weighted with BRR standard errors, reported **per variable and never averaged across variables**.
**Consequence:** verified against analytic values — `d = 0.9986` for μ-difference 1 at σ = 1, `0.5011` for 0.5, and exactly `−1.2649` for `[1..5]` vs `[3..7]`. The submitted values of 0.13–0.18 are classified **negligible** by `interpret_d`, not "large" as `main.tex:651` claims. Tables 3–5 are regenerated and will be mutually distinct.

### ST-2 — Stability analysis `DONE`
*Answers comment 4.*
**Old:** none — one run on the full dataset, one feature set, nothing to be stable.
**New:** pairwise Jaccard, Kuncheva's chance-corrected consistency index, and per-feature selection frequency with Wilson binomial CIs; consensus set defined at ≥80% of folds.
**Consequence:** verified — Jaccard 1.0/0.0/0.333 on identical/disjoint/half-overlapping sets; Kuncheva 1.0 identical, exactly 0.0 at chance overlap, −1/3 disjoint. An honest result may be that VLPSO is *less* stable than embedded methods.

### ST-3 — Permutation null `BUILT`
*Answers comment 4.*
**Old:** none.
**New:** labels permuted **within outer-training folds** and **within school**, so the null preserves the clustered design; the *complete* pipeline is re-run per permutation. `assert_permutation_null_is_chance` **raises** if the permuted mean AUC deviates from 0.50 by more than 0.03.
**Consequence:** this is the decisive test that the leakage is gone. `NOT RUN`.

---

## 5. Presentation defects (comment 10)

| Defect | Location | Status |
|---|---|---|
| Abstract entirely in future tense, no dataset/N/design/result/limitation | `main.tex:53` | `PENDING` |
| "PISA 2022" → PISA 2018 | `main.tex:433` | `PENDING` — author confirmed 2018 |
| Country (Spain) never stated | throughout | `PENDING` |
| Hard-coded "Figures 14 and 15" instead of `\ref{}` | `main.tex:826` | `PENDING` |
| `\addtocounter{table}{-1}` after Tables 3, 4, 5 | | `PENDING` |
| Malformed `\subsubsection{Algorithm 4: Length Adaptation` | `main.tex:411` | `PENDING` |
| Orphan sentence "The first section of this document will focus on…" | `main.tex:118` | `PENDING` |
| Off-topic decentralised-finance sentence in the literature review | | `PENDING` |
| Duplicate `\cite{ref27}` for both SHAP and LIME | `main.tex:120` | `PENDING` |
| Inconsistent capitalisation (vlpso, svm, "low vs. High") | `main.tex:441,451,453,516,518` | `PENDING` |
| Causal/temporal language ("through time", "path dependence", "accounts for") | `main.tex:746,774,778,784,820,828` | `PENDING` |
| Limitations section absent | | `PENDING` |

---

## 6. Infrastructure

### IN-1 — Environment-aware config `DONE`
**Old:** `codes/config.py:5` hard-coded `/content/drive/MyDrive/educational-disparities-analysis` and **raised an exception** on any other path. The repository could not run locally or in CI.
**New:** `src/vlpso_xai/config.py` resolves `VLPSO_PROJECT_ROOT` → Drive mount → repository root → `./` with a warning. All grids, seeds, budgets and thresholds moved to `config/default.yaml`; `config/quick.yaml` provides a reduced smoke-test budget.

### IN-2 — Hard-coded validation targets removed `PENDING`
**Old:** `validation.py:188,205` asserted expected class counts and an expected top-3 feature set `['ST013Q01TA','ST166Q03HA','ST166Q04HA']`. A "reproducibility assessment" that checks results against hard-coded expected answers is circular.

### IN-3 — Model-list mismatch `PENDING`
**Old:** `results_interpretation_and_validation.ipynb` cell 10 iterates a list containing `'LogisticRegression'`, but `config.py` defines `LogisticRegression_L1` and `LogisticRegression_L2`. That loop cannot have run successfully.

---

## 7. Open items requiring data or author input

| Item | Blocker |
|---|---|
| Every numerical result | `CY07_MSU_STU_QQQ.sav` not yet available. The file supplied is the **school** questionnaire (`CY07_MSU_SCH_QQQ`): 21,903 rows, 166 `SC*` items, **no** `PV*MATH`, **no** `ST0*` items, **no** `W_FSTUWT`. |
| School-level predictors | Author approved merging on `CNTSCHID`, but the supplied CSV carries no SPSS value labels and no school-questionnaire instrument PDF was provided. Allowlisting items without a documentary source would repeat exactly the defect of comment 7. Section scaffolded and blocked. |
| External validation on Portugal | Requires the same student file, `CNT == 'PRT'`. |
