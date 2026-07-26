# Response to the Editor

**Manuscript:** *An Explainable AI Framework for Student Performance Prediction Using Variable-Length Particle Swarm Optimization in Educational Data Mining*
**Journal:** *Applied System Innovation* (MDPI)

---

## Summary

**The editor's central concern was correct. We confirmed data leakage in our own pipeline, identified the exact mechanism, and have rebuilt the analysis around a leakage-proof nested design. The corrected results are substantially lower than those we submitted, and in several places they do not support the claims we made.**

The defect is a single line. `codes/data_preparation.py:253` constructed the design matrix as:

```python
categorical_columns = df.select_dtypes(include=['object']).columns
X = df_exp.drop(columns=list(categorical_columns) + ['label']).fillna(0)
```

`select_dtypes(include=['object'])` removes only *string* columns. Our outcome variable `math_score` is `float64` and therefore survived into the feature matrix, together with all ten `PV1MATH`–`PV10MATH` plausible values, every reading and science plausible value, the final student weight, the eighty BRR replicate weights, and `STRATUM`. Because our label is a hard threshold on `math_score`, the outcome was literally among the predictors.

We have verified this empirically rather than merely inferring it. On the Spanish PISA 2018 sample (N = 35,943), a **depth-1 decision tree using `math_score` alone achieves accuracy = AUC = 1.0000 on all three tasks**. Ranking the old feature matrix by mutual information places `math_score` and all ten mathematics plausible values in the top eleven positions; the first genuine questionnaire item ranks 32nd of 147. Since our VLPSO implementation seeded every particle with the top-ranked features, the entire initial population consisted of outcome variables.

We regret that this reached submission. We are grateful the editor caught it before review.

Everything below is reproducible from the public repository accompanying this response. Every number is emitted by a script and written to `results/`; none is typed by hand.

---

## Comment 1 — Audit the evaluation pipeline; nested cross-validation is required

> *"Please specify whether imputation, scaling, encoding, feature ranking, VLPSO feature selection, hyperparameter tuning, model selection, and threshold selection were performed independently within each training fold. A nested cross-validation design is required."*

**What we found.** None of them were. Specifically:

| Step | Submitted behaviour | Location |
|---|---|---|
| Scaling / encoding | fitted on the entire dataset before any split | `data_preparation.py:98-100` |
| Imputation | global `.fillna(0)` and global column means | `data_preparation.py:253` |
| Feature ranking + VLPSO | run on all ~26k rows, no split; selection frozen and reused for train and test | `feature_selection.ipynb` cells 13-15 |
| Hyperparameter tuning | `GridSearchCV` fitted on all of `X, y`, then "validated" by `StratifiedKFold` on the same `X, y` | `training.py:85-94` |
| Model selection | best model chosen by test-set performance, then that number reported | `evaluation.py:60-81` |
| Threshold | never selected; default 0.5 used while reporting balanced accuracy on a 6.9% positive class | — |
| Wrapper fitness | `knn.fit(X_sel, y)` then `knn.predict(X_sel)` — resubstitution | `feature_selection.ipynb` cell 8 |

No `Pipeline` object appeared anywhere in the repository, so no preprocessing step was ever confined to a training fold.

**What we changed.** `src/vlpso_xai/evaluation/nested_cv.py` implements a strict two-level design: 5 outer folds × 5 repeats of `StratifiedGroupKFold` grouped on `CNTSCHID`, with 5 inner folds. Every preprocessing and selection step is a *step inside an `sklearn.pipeline.Pipeline`*, so isolation is structural rather than a matter of discipline. Model selection and threshold selection both happen inside the inner loop. The outer test fold is scored exactly once.

**How we verified it.** `tests/test_nested_cv_isolation.py` injects a sentinel column that is perfectly predictive on outer-test rows only and pure noise elsewhere, and asserts it is never selected — while confirming that fitting on the full data (the submitted behaviour) selects it immediately. A hard `assert_group_disjoint` check raises if any school appears on both sides of a split.

**Numerical consequence.** See Comment 2.

---

## Comment 2 — Rule out outcome leakage; provide the complete predictor list

> *"Provide the complete list of candidate predictors and explain how variables directly or indirectly related to the mathematics score or proficiency category were excluded."*

**What we changed.** `src/vlpso_xai/data/features.py` replaces the dtype-based denylist with an **allowlist enforced by an exception**. A column enters `X` only if it appears in `config/predictor_allowlist.yaml`, which carries one row per variable with its PISA item code, official codebook label, response scale and a stated justification. That file is submitted as **Supplementary Table S1** and is the complete candidate-predictor list the editor requested: 31 student-questionnaire items.

Excluded, with reasons: `math_score` and `PV*MATH` (the outcome and its sources); `PV*READ` and `PV*SCIE` (drawn from the same PISA conditioning model, hence contaminated); `W_FSTUWT` and `W_FSTURWT1`–`80` (used for weighting and variance estimation, never as features); `STRATUM` (a sampling-design variable); `noise_control` (see Comment 7).

A denylist of anchored patterns runs *as well*, so an allowlist mistake is still caught, and the guard is called at every entry point — data preparation, selector `fit`, model fitting and explanation. `tests/test_leakage_guard.py` asserts that injecting `math_score`, `math_category`, `PV1/3/10MATH`, `PV1READ`, `PV7SCIE`, `W_FSTUWT`, `W_FSTURWT1/80`, `SENWT`, `STRATUM`, `STRATUM_ESP9033`, `CNTSCHID`, `CNTSTUID` or `noise_control` each raises `LeakageError`.

We added an **empirical screen**: the AUC of every candidate predictor on its own, direction-free. Anything above 0.95 cannot enter without recorded sign-off. On the corrected feature set nothing is flagged; the maximum single-variable AUC is **0.7472** (`ST013Q01TA`, books in the home), against 1.0000 for `math_score`.

**Numerical consequence — this is the central table of the revision.**

| Task | AUC as submitted | AUC, clean features, flat CV | **AUC, nested + school-grouped** |
|---|---|---|---|
| Low vs. High | 1.0000 | 0.8893 | **0.8766** |
| Low vs. Medium | 1.0000 | 0.7664 | **0.7439** |
| Medium vs. High | 1.0000 | 0.7048 | **0.6920** |

Our submitted Table 1 (accuracy 0.830–0.854, AUC 0.912–0.931 for Low vs. High) came from a *different*, whitelisted `prepare_data` defined locally in `ensemble.ipynb` cell 14, which contained no PV columns. Table 2 came from the leaking path. **The performance jump we attributed to VLPSO was the difference between two `prepare_data` implementations.**

---

## Comment 3 — Describe the PISA sample completely

> *"Report the country or countries, initial and final sample sizes, exclusions, missing-data treatment, class distribution, survey weights, plausible-value handling, and treatment of PISA's stratified and clustered sampling design."*

**Cycle.** The manuscript said PISA 2022 at `main.tex:433`; the code read `CY07_MSU_STU_QQQ.sav`, which is **PISA 2018**. 2018 is correct and the text is fixed.

**Country.** **Spain only** (`CNT == 'ESP'`). This was never stated in the submitted manuscript. It now appears in the title area, abstract, methods and limitations.

**Sample flow** (`results/sample/sample_flow.csv`):

| Step | Students | Schools |
|---|---|---|
| PISA 2018 student file, all countries | 612,004 | — |
| `CNT == 'ESP'` | 35,943 | 1,089 |
| ≤ 400 missing of 1,119 columns (primary) | **29,786** | 1,087 |
| ≤ 200 missing (sensitivity) | 14,728 | 1,056 |
| ≤ 600 missing (sensitivity) | 34,953 | 1,089 |

We must report an additional discrepancy. `data_preparation.py:189` hard-coded 26,657 as the expected N, and `validate_data_quality` hard-coded class counts of Low 10,243 / Medium 14,484 / High 1,930. The actual analytic sample is **29,786** with Low 12,072 / Medium 15,653 / High 2,061. Because that check only printed a warning rather than raising, the discrepancy was never caught. The missingness threshold was also never justified and is consequential: plausible alternatives move N by a factor of 2.4. Both sensitivities are now reported.

**Proficiency cut points.** We used `s <= 482` and `s <= 607`. The official PISA 2018 boundaries are **Level 3 = 482.38** and **Level 5 = 606.99**, half-open from below. The correction reclassifies 55 students.

**Plausible values.** We averaged the ten PVs and thresholded the mean (`data_preparation.py:164`). This is invalid: PVs are draws from a posterior, their mean has systematically deflated variance, and the resulting category has no defensible standard error. We now derive the proficiency category **separately for each of the ten PVs**, run the analysis ten times, and combine with **Rubin's rules**, reporting the variance components separately.

> Low vs. High, M = 10: combined AUC **0.8735**, within-imputation variance 0.000014 (school-clustered bootstrap), between-imputation variance 0.000016, total 0.000031, SE **0.0056**, 95% CI **[0.8621, 0.8849]**, df 28.4, **fraction of missing information 0.563**. Survey-weighted: AUC **0.8720**, SE **0.0076**, 95% CI **[0.8567, 0.8873]**.
>
> **More than half the uncertainty in the performance estimate comes from the plausible values.** Averaging them first understates the standard error by a factor of **1.47**.

We also report a finding that we think is substantive in its own right: **69.7% of students (survey-weighted) receive a different proficiency category depending on which plausible value is used.** Only 30.3% are assigned the same category by all ten. Thresholding an averaged score assigns every one of those students a single confident label.

**Survey design.** Weights were never applied — and `W_FSTUWT` and `STRATUM` were used as *predictors*. We now apply `W_FSTUWT` to descriptives and report weighted alongside unweighted metrics, and compute standard errors from the **80 Fay BRR replicate weights** (k = 0.5, divisor R(1−k)² = 20). This is not a formality: for the books-in-the-home item the BRR standard error is **2.28×** the naive i.i.d. value (0.01795 vs 0.00788), and the Kish design effect is 2.13. Weighted descriptives with BRR standard errors are in `results/sample/brr_descriptives.csv`.

**Clustered design.** Cross-validation split students at random, so classmates straddled the train/test boundary. All splits are now grouped on `CNTSCHID`. This costs 0.013–0.023 AUC — smaller than we expected, but real and in the expected direction.

---

## Comment 4 — Strengthen validation

> *"...confidence intervals, repeated runs with different random seeds, feature-selection stability, permutation-based checks, and, where possible, validation using a separate country, cohort, or held-out sample. Claims that there is no overfitting cannot be made without appropriate evidence."*

**Confidence intervals.** BCa bootstrap on outer-fold predictions, **resampling schools rather than students**, since students within a school are not independent. Low vs. High: AUC 0.8735, 95% CI [0.8656, 0.8806]. We note honestly that respecting the clustering widens the interval by only 1.05× for AUC — a rank statistic is less sensitive to clustering than a mean.

**Permutation tests.** This is the decisive check, and it required us to correct our own first implementation. We report **two distinct nulls**:

| Null | Mean AUC | Interpretation |
|---|---|---|
| **Unrestricted** (labels shuffled across the sample) | **0.5034** | The leakage test. Chance, as required. |
| Within-school (each school's composition preserved) | 0.6188 | Expected to exceed chance; must be estimated, not assumed |

Our initial implementation permuted within school *and* asserted the result should be 0.50. Those are inconsistent: within-school permutation preserves each school's class composition, so any feature predicting *what kind of school* a student attends still predicts the permuted label. Corrected, the two nulls give a decomposition we consider a genuine finding:

> Of the model's 0.360 AUC above chance, **0.115 (32%) is attributable to between-school composition** and **0.244 (68%) to discrimination between students within the same school**.

**Feature-selection stability.** Pairwise Jaccard and Kuncheva's chance-corrected index across outer folds, with per-feature selection frequencies and Wilson binomial CIs. Results in Comment 5.

**Repeated seeds.** ≥ 10 distinct seeds entering model initialisation, CV shuffling and PSO stochasticity, with between-seed SD reported for every headline metric.

**External validation.** We hold out **Portugal** (PISA 2018, N = 5,932 in 276 schools) as a fully external test set: everything is fitted on Spain and evaluated once on Portugal, with the transfer gap reported. **We have removed the claim that there is no overfitting** from the manuscript; it was unsupported.

---

## Comment 5 — Comparisons and ablations

> *"Compare VLPSO with ordinary binary PSO and established filter, wrapper, and embedded feature-selection methods. Include ablations that separately evaluate variable-length search, the symmetric-uncertainty component, and each term in the objective function. Report the number and identities of selected features for every task."*

**What we found.** The repository contained no baseline of any kind. Two further problems:

* The manuscript describes a **symmetric-uncertainty** component. The code used `mutual_info_classif` and said so in a comment: `# Rank features using mutual information (instead of SU for now)`.
* The manuscript's objective has three terms including a feature-count penalty. The implemented fitness had **two** (accuracy + inter-class distance) and **no cardinality penalty**, contradicting the abstract's claim of "minimizing the number of selected features".

We have implemented symmetric uncertainty properly and added the cardinality penalty, so both claims are now true of the code.

**We must also report that the identities of the selected features in the submitted manuscript are unrecoverable.** `feature_selection.ipynb` cell 20 applied positional indices from a twice-reduced matrix back onto the original dataframe, so the columns saved as "selected" were not the columns the algorithm chose. Selection is now name-based end to end, with a regression test.

**Results (provisional: Low vs. High, PV1, 3 outer folds, reduced PSO budget).**

| Method | AUC | Features | Jaccard stability |
|---|---|---|---|
| **none** (all 61) | **0.8748** | 61.0 | 1.000 |
| relieff | 0.8630 | 12.0 | 0.682 |
| mutual_info | 0.8589 | 12.0 | 0.758 |
| **bpso** | 0.8589 | 12.0 | 0.500 |
| symmetric_uncertainty | 0.8581 | 12.0 | **1.000** |
| chi2 | 0.8579 | 12.0 | 0.897 |
| **vlpso (proposed)** | **0.8440** | 11.3 | **0.235** |

**Our proposed method finishes last.** VLPSO scores below every baseline including ordinary binary PSO, and no feature-selection method outperforms using all 61 features. VLPSO is also markedly the least stable, sharing only 23.5% of selected features across three folds with subset sizes ranging from 2 to 17. Kuncheva's index is undefined for VLPSO precisely because its subset sizes vary.

**Ablations.**

| Arm | AUC | Features | Result |
|---|---|---|---|
| vlpso (grow + shrink) | 0.8440 | 11.3 | reference |
| shrink-only | **0.8607** | 14.0 | the variable-length mechanism **hurts** |
| no cardinality penalty | **0.8723** | 26.7 | the penalty costs 0.028 AUC |
| MI ranking instead of SU | 0.8449 | 7.3 | SU vs MI: no material difference |
| random initialisation | 0.8535 | 10.7 | ranked seeding is not better than random |

Paired contrasts across matched folds (Nadeau–Bengio corrected resampled *t*, Holm-corrected) give point estimates uniformly favouring the baselines, but **every adjusted p-value equals 1.000**: with three folds the design has no power. We state plainly that these comparisons are currently *inconclusive*, and the full 5 × 5 × 10-PV specification will decide them. We will not claim an advantage that the data do not support.

---

## Comment 6 — Correct the effect-size section

> *"Explain exactly how Cohen's d was calculated and what groups or distributions were compared. The values in Tables 3 and 4 are identical and should be checked. Values around 0.13–0.18 should not be described as large effects."*

**The editor is right on both counts, and the underlying quantity was worse than a labelling error.** `codes/evaluation.py:39-57` computed the difference in the **grand mean of all standardised feature columns** between rows the model *predicted* positive and negative, divided by an average of per-column pooled SDs. That is not an effect size for classifier performance, not an effect size for any group contrast, and it used predicted rather than true labels. Because standardised features average to ≈0 by construction, the result was necessarily near zero — which fully explains the 0.05–0.18 range.

**Tables 3 and 4 are indeed identical**, at `main.tex:672-681` and `main.tex:700-709`. This was a copy-paste error, and unfixable in place because the quantity itself was meaningless.

**We deleted the function.** Two clearly distinct quantities replace it, never reported in the same table:

**(a) Method comparison** — paired across matched outer folds, using the **Nadeau–Bengio corrected resampled *t*** (the naive paired *t* is badly anti-conservative when CV training sets overlap), with 95% CIs and Holm–Bonferroni correction within task.

**(b) Group differences** — standardised mean differences between **true** proficiency groups on **individual** predictors, survey-weighted with BRR standard errors, reported **per variable and never averaged across variables**.

Interpretation now follows convention honestly: |d| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large. **The submitted values of 0.13–0.18 are negligible**, and `main.tex:651` is corrected. Our implementation is unit-tested against analytically known values (e.g. exactly −1.2649 for `[1,2,3,4,5]` vs `[3,4,5,6,7]`). Tables 3–5 are regenerated and mutually distinct.

---

## Comment 7 — Revise the explainability analysis

> *"Distinguish clearly between local SHAP explanations, globally aggregated SHAP summaries, and LIME explanations. Report the model output scale, background/reference data, explainer settings, and explanation stability. Provide official PISA item descriptions and coding for every interpreted variable."*

**What we found.** We computed SHAP and LIME for exactly **two instances per task** — the argmax and argmin of predicted probability — and discussed them as though they characterised the population. There was no global aggregation anywhere. Background data was inconsistent by model (`LinearExplainer` got the full `X`, `KernelExplainer` got `kmeans(X, 50)`, `TreeExplainer` got none, `DeepExplainer` got `X.values`), producing attributions on different scales that the text then compared numerically. No seed was set for LIME, which is stochastic.

**What we changed.** Local SHAP, global SHAP and LIME are computed, labelled and reported separately. Global SHAP aggregates over ≥ 1,000 stratified test instances with bootstrap CIs. Backgrounds are drawn from **training data only**, by a documented method, identically for every model. Output scale is stated once and used consistently. LIME is repeated ≥ 30 times per instance with different seeds, and we report the SD of every feature weight and the rank correlation between runs. A full explainer-configuration table is included. Instance selection is quantile-stratified and the rule is stated, rather than argmax/argmin.

**The `noise_control` variable.** We injected a Gaussian noise variable as a sanity check and then **interpreted it substantively** at `main.tex:766` ("noise_control (−0.24) ... eroding the model's confidence"). It is now excluded from the predictor allowlist and used only as a **negative control** in explicitly flagged diagnostic runs. In that role it is informative: it ranks **29th of 32**, and **3 of 31 real features rank below pure noise** — `ST012Q09NA` (musical instruments), `ST011Q05TA` (educational software) and `ST011Q08TA` (books of poetry). Those three are reported as indistinguishable from noise.

**PISA codebook grounding.** The editor's instruction not to assign educational meanings without documentary support was well directed. Checking our interpreted variables against the official instrument (*PISA 2018 Student Questionnaire, Main Survey, computer-based*, `CY7_201709_QST_MS_STQ_CBA_NoNotes`) and against the SPSS value-label metadata, we found:

| Item | Manuscript gloss | Official wording |
|---|---|---|
| `ST012Q01TA` | "number of books in the household" (`:433`); "home book availability" (`:820`) | **"How many in your home: Televisions"** |
| `ST013Q01TA` | "availability of education materials in the home" | **"How many books are there in your home?"** |
| `ST166*` | "presence/absence of devices/internet access/digital literacy practices" | **a phishing-email situational-judgement task** |

Our home-literacy and cultural-capital interpretation therefore rested on **the number of televisions in the home**, and our "digital environment" construct is a digital-safety reasoning task in which two items are reverse-scored. We had also overlooked `ST011`, the genuine home-educational-resources block (desk, quiet place to study, computer for schoolwork, internet link, reference books), which is now included among the candidate predictors.

The corrected analysis is more coherent, not less. Permutation importance ranks `ST166Q03HA` ("click on the link to fill out the form as soon as possible") first and `ST013Q01TA` (books in the home) second. **`ST012Q01TA` — televisions, the item our discussion was built on — ranks 13th of 32.** Its single-variable AUC direction also confirms the reverse-scoring: endorsing the phishing link predicts *low* proficiency (signed AUC 0.273).

`src/vlpso_xai/data/codebook.py` now resolves every interpreted variable through three layers — the instrument PDF, hand-verified overrides, and the SPSS value labels — and `require_documented()` raises if any variable would be interpreted without a source.

---

## Comment 8 — Remove causal and temporal claims

> *"SHAP and LIME describe how a fitted model uses variables; they do not show that those variables cause achievement... Rewrite the discussion in associational and predictive terms."*

Accepted without reservation. A SHAP decision plot orders features by attribution magnitude and has no time axis, so our "trajectory through time" language had no basis. The following passages are rewritten:

| Line | Removed | Replaced with |
|---|---|---|
| `main.tex:774` | "model output continuously increasing … **through time**" | ordering by attribution magnitude; no temporal reading |
| `main.tex:778` | "**early** negative contributions dominate the pathway" | features with the largest negative attributions |
| `main.tex:828` | "clear **path dependence effect** … long-lasting influences" | deleted; unsupported |
| `main.tex:784` | "consistent with **cumulative advantage models**" | deleted |
| `main.tex:746` | "regional context **accounts for** nearly all of the model confidence" | "the model's output is most sensitive to …" |
| `main.tex:746` | "capturing **pedagogically meaningful structure** as opposed to dataset specific artifacts" | deleted — our own audit shows the opposite |
| `main.tex:820` | "regional context can **limit achievement through constraining opportunities**" | associational phrasing only |

The claim at `main.tex:746` deserves particular note: our headline explainability finding was that `STRATUM_ESP9033` was the top contributor with SHAP +0.88, which we read as a substantive regional effect. `STRATUM` is PISA's sampling stratum — a design variable. **That finding is withdrawn.** A standing caveat now appears in the discussion.

---

## Comment 9 — Clarify the algorithm and contribution

> *"Define precisely how variable particle length is represented and updated, how dimensions are added and removed, how differently sized particles interact, and whether the optimisation criterion is minimised or maximised."*

**What we found.** Our implementation was **not variable-length.** Particles were fixed-length binary vectors of size `n_features`; "length" was a cardinality cap enforced by truncating to the top-ranked bits. The adaptation block only ever *cleared* bits, so lengths were monotonically non-increasing and a discarded feature could never return. Different-length interaction never arose because all vectors were the same length — the mechanism the manuscript describes did not exist. `alpha=7` was declared and never used; `best_div = np.argmax(pbest_fitness)` is a particle index despite its name; there was no stopping criterion beyond a fixed iteration count.

**What we changed.** We implemented genuine variable-length search (`src/vlpso_xai/selection/vlpso.py`) with a **ranked-prefix encoding**: particle *i* carries an explicit dimensionality `L_i` and a position in `[0,1]^{L_i}`, where dimension *d* denotes the *d*-th feature by symmetric uncertainty. Lengths both **grow and shrink**. New dimensions are seeded from the best particle spanning that dimension, with **zero initial velocity** (no momentum in a direction never explored). Cross-length exemplar learning — the part the original code sidestepped — is specified explicitly: particles learn over the shared prefix, and for dimensions the exemplar does not span we use the best-fitness particle that does span them, falling back to the particle's own personal best. The objective is **maximised**; this is now stated. A convergence criterion was added, `alpha` removed and the misnamed variable fixed. The wrapper fitness uses internal stratified CV; there is deliberately no resubstitution option in the API.

**On the contribution.** We must be direct. Having implemented the mechanism properly, **it does not pay off**: the shrink-only ablation (0.8607) outperforms genuine bidirectional variable-length search (0.8440), and VLPSO underperforms ordinary BPSO. On current evidence we cannot claim that variable-length search improves feature selection for this problem.

We therefore **reposition the contribution**. The paper's defensible contributions are the leakage-proof evaluation protocol for PISA-scale educational data mining — including plausible-value handling, survey weighting and school-level grouping, which are routinely neglected in this literature — and the multi-level explainability analysis grounded in the official codebook. The optimiser is presented as one method among several in a fair comparison, and the comparison does not favour it. We would rather publish that than an unsupported novelty claim.

Full parameter values, stopping criteria, convergence curves, wall-clock cost, number of fitness evaluations and a big-O complexity analysis are reported. We disclose that the k-NN wrapper fitness is evaluated on a stratified subsample (2,500–3,000) of each training fold, since the wrapper is O(n²) and a full search on ~15,000 students is otherwise infeasible; the final model is always refitted on the complete training fold.

---

## Comment 10 — Internal consistency and presentation

> *"Correct unresolved figure references, duplicated text, inconsistent task naming, inconsistent capitalisation, incomplete equations, and the mismatch between future-tense statements in the abstract and the completed empirical study."*

All accepted:

* **Abstract** rewritten in past tense, now reporting cycle and country (PISA 2018, Spain), analytic N (29,786 in 1,087 schools), class distribution, validation design (nested CV with school-level grouping, ten plausible values combined by Rubin's rules, survey weights), central results with confidence intervals, and the main limitations.
* **PISA 2022 → PISA 2018** at `main.tex:433`.
* **Spain** stated throughout.
* Hard-coded "Figures 14 and 15" at `main.tex:826` replaced with `\ref{}`.
* `\addtocounter{table}{-1}` removed after Tables 3, 4 and 5, which now number distinctly.
* Malformed `\subsubsection{Algorithm 4: Length Adaptation` at `main.tex:411` fixed.
* Orphan sentence at `main.tex:118` deleted.
* Off-topic decentralised-finance sentence removed from the literature review.
* Duplicate `\cite{ref27}` at `main.tex:120` corrected.
* Capitalisation normalised throughout (VLPSO, PSO, SHAP, LIME, Gradient Boosting, SVM, MLP, Random Forest, XGBoost, LightGBM; "Low vs. Medium", "Medium vs. High", "Low vs. High").
* Truncated equations completed and each verified against the implementation; the objective function in the text now matches the code.
* **A Limitations section has been added**, covering the single country, cross-sectional design, absence of causal identification, plausible-value and weighting assumptions, feature-selection instability, and the computational limits on our permutation and PV budgets.

---

## Reproducibility

The repository accompanying this response contains the complete pipeline. `scripts/run_all.py` regenerates every manuscript number from the raw OECD file in one command. The test suite (104 tests) runs on synthetic data and requires no PISA microdata, so it executes in CI. Dependencies are pinned; seeds are fixed; `results/manifest.json` records a SHA256, generating script, git commit and config hash for every artefact.

We note two defects we found and fixed *in the revised code itself* during this work, since they are the same class of error as the original and we would rather disclose them:

1. Inside a `Pipeline`, the imputer and scaler emit unnamed arrays, so a downstream selector reported positional labels (`x3`, `x17`) instead of PISA item codes — the same name/position confusion as the original bug, arriving by a different route. Fixed by forcing pandas output through the pipeline; locked in by a regression test.
2. Our first ablation configuration set the stagnation threshold above the iteration count, so the length-adaptation mechanism could never fire and the ablation arms returned byte-identical results. That would have been written up as "variable-length search makes no difference" when in fact it had never run. The class now raises on that configuration.

We are grateful for the editor's scrutiny. The revised paper reports considerably weaker performance than the version submitted, and we believe it is a substantially better paper for it.
