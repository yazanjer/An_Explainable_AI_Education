# Handover — VLPSO-XAI / PISA 2018 revision

You commissioned a staged rebuild of a manuscript returned by *Applied System Innovation* for suspected data leakage. This is the handover. **Read the "Do not trust yet" section before using any number.**

Repository: <https://github.com/yazanjer/An_Explainable_AI_Education>
Results: shared Drive folder (`results/`), which you have access to.

---

## 1. What was established

**The editor was right, and the mechanism is a single line.** `codes/data_preparation.py:253` dropped only object-dtype columns, so `math_score` (float64) and all thirty-plus plausible-value columns survived into the feature matrix. The label is a hard threshold on `math_score`.

Verified empirically on the real data, not inferred:

* a depth-1 decision tree on `math_score` alone gives **accuracy = AUC = 1.0000** on all three tasks;
* mutual-information ranking of the old 147-column matrix puts `math_score` and all ten `PV*MATH` in the top eleven; **the first genuine questionnaire item ranks 32nd**;
* since `_init_population` seeded every particle from that ranking, the entire initial VLPSO population consisted of outcome variables.

**Honest headline numbers** (nested CV, `StratifiedGroupKFold` on `CNTSCHID`, 5 outer × 5 repeats × 5 inner, model and threshold selected inside the inner loop, Spain only, N = 29,786 in 1,087 schools, 10 plausible values — **750 folds, ~34 hours of compute**):

| Task | As submitted | Rebuilt |
|---|---|---|
| Low vs. High | 1.0000 | **~0.876** |
| Low vs. Medium | 1.0000 | **~0.752** |
| Medium vs. High | 1.0000 | **~0.695** |

Medium vs. High sat at 0.66–0.72 across all 250 of its folds. It is genuinely weak and no framing rescues it.

**Other findings that stand on their own:**

* **69.7% of students** (survey-weighted) receive a different proficiency category depending on which plausible value is used. Only 30.3% are stable across all ten. Thresholding an averaged score — what the submitted code did — hides this entirely.
* **Unrestricted label permutation returns AUC 0.5029** (n = 20). Within-school permutation returns 0.6192, which is *not* leakage: it preserves each school's class composition. The two nulls decompose performance into **32% between-school composition / 68% within-school discrimination**.
* **BRR standard errors are 2.28× the naive i.i.d. ones** (books-in-the-home item: 0.01795 vs 0.00788; Kish design effect 2.13). Naive SEs understate uncertainty by more than half.
* **Rubin's rules, M = 10:** AUC 0.8735, 95% CI [0.8621, 0.8849], **fraction of missing information 0.563** — most of the uncertainty is measurement error in proficiency, not sampling.
* **The manuscript's codebook glosses are wrong.** Verified against the official instrument and the SPSS value labels: `ST012Q01TA` is **"How many in your home: Televisions"**, not "number of books"; `ST013Q01TA` is the books item; the `ST166` block is a **phishing-email judgement task**, not device ownership. The paper's cultural-capital story rests on television counts. Permutation importance ranks the phishing item 1st, books 2nd, **televisions 13th of 32**.
* **`STRATUM_ESP9033`, the paper's headline SHAP finding (+0.88), is a sampling-design variable.** That result is withdrawn.

---

## 2. Do not trust yet

**These are the load-bearing caveats. Treat anything here as unverified.**

| Item | Status |
|---|---|
| **Selector comparison and ablations** | **Provisional.** 3 folds, one PV, reduced PSO budget (pop 14, 12 iters vs configured 60/100). Every Holm-adjusted p-value = 1.000 — the design has no power. |
| **VLPSO vs BPSO** | **Confounded.** BPSO was hard-capped at 15 features while VLPSO was uncapped (audit M11). The current "VLPSO finishes last" result must be re-run matched before any repositioning claim. |
| **SHAP / LIME** | **Never executed.** Code is written and guard-tested; `shap` would not install in the build environment. No global SHAP, no LIME stability, no SHAP-vs-LIME agreement exists. |
| **Bootstrap CIs, permutation, stability, effect sizes at full budget** | **Not in the long run.** See §3. |
| **External validation on Portugal** | Implemented and run once at reduced budget (transfer gaps −0.011 to −0.019, i.e. no degradation). Not re-run at full budget. |
| **`main.tex`** | **Untouched.** Editor comments 8 and 10 are entirely unaddressed. |
| **`RESPONSE_TO_EDITOR.md`** | **Do not send.** It cites budgets (≥10 seeds, ≥1,000 SHAP instances, 100 permutations) that no completed run delivers. |

---

## 3. What the long run actually produced

`scripts/run_all.py` implements only `ingest`, `audit`, `sample`, `nested`, `assets`. The stages `selection`, `stats` and `explain` were **declared but never implemented**, and until just now were accepted silently — so the 34-hour run completed "successfully" having produced **no selector comparison, no uncertainty quantification and no explainability output**. Requesting them now raises. Those analyses live in notebooks 03–06 and must be run separately.

So `results/` contains: the leakage demonstration, column classification, sample flow with missingness sensitivities, PV instability, BRR descriptives, design diagnostics, **750 nested-CV folds with per-fold predictions**, and `manifest.json` (SHA256 + git commit + config hash per artefact).

---

## 4. Suggested next steps, in order

1. **Re-run the selector comparison with BPSO and VLPSO matched** (audit M11) at full budget. Until then the paper cannot say anything about the optimiser in either direction.
2. **Run notebook 05** for bootstrap CIs, permutation tests, stability and effect sizes on the 750 stored folds. The predictions are already on disk; this is cheap.
3. **Run notebook 06** for SHAP/LIME. Note `global_shap` requires frames from `transformed_frames(pipeline, ...)` — passing raw frames raises, deliberately.
4. **Edit `main.tex`** — comments 8 and 10 have had no work at all.
5. **Reconcile `RESPONSE_TO_EDITOR.md`** against whatever budget was actually achieved. Every claim must be true of a completed run.
6. **Close the remaining audit findings**: M2 (notebook 05 pools predictions across methods/repeats/PVs into one AUC), M4 (paired *d* over 3 folds labelled "large"), M5 (silent degradations), M7 (VLPSO/BPSO wrapper fitness uses ungrouped inner CV).

---

## 5. Constraints that must not be relaxed

* **Do not tune to recover the old numbers.** The honest figures are far lower and that is the correct outcome.
* **Never report a number not produced by a script and written to `results/`.**
* **Leakage is a hard error.** The guard raises; it does not warn.
* **Two permutation nulls are not interchangeable.** Only *unrestricted* permutation should give 0.50. Within-school permutation is expected above chance; asserting otherwise produces a false alarm — this happened and is documented.
* **Interpret no variable without a codebook citation.** `codebook.require_documented()` enforces it.

---

## 6. Verification state

309 tests pass on synthetic data (no PISA microdata needed, so CI runs). `VERIFICATION_REPORT.md` records all eight Stage 10 checks, including the ones that failed first.

An independent adversarial audit found **15 defects in the rebuild**; **10 are fixed** (M2 and M4 closed most recently, M5 partially), 4 remain open and are listed with severities in `CHANGELOG_REVISION.md`.

**M2/M4 update.** `src/vlpso_xai/evaluation/aggregate.py` replaces the pooled glob in notebook 05 with an explicit three-level hierarchy — concatenate folds within a repeat, average AUCs across repeats, Rubin's rules across PVs, never combine methods — and refuses to run on an ambiguous checkpoint set. Effect-size magnitudes are now guarded: a band is emitted only at ≥ 10 matched folds *and* a CI for *d* inside one band, so every cell in the current 3-fold selector comparison reads `indeterminate (J=3 < 10)`. 24 new tests in `tests/test_aggregation.py`; 187 tests pass on synthetic data.

**The most important entry:** the original isolation tests were *vacuous*. Three catastrophic leakage mutations — threshold tuned on the outer test fold, `GridSearchCV` fitted on the full dataset, model selected by outer-test AUC — **all passed 104 tests**. `tests/test_leakage_detection_power.py` now runs the harness end to end and is calibrated against a known-leaky reference, so it has demonstrated power rather than assumed it. All three mutations are caught.

Seven bugs were found in the rebuild itself and are documented in `CHANGELOG_REVISION.md` §0e — including feature names lost inside the `Pipeline`, checkpoints reloaded across incompatible configurations, an ablation that could never fire, and **Portugal pooled into the training data**, which the sample-flow table caught only because it is printed.

That last point is the general lesson: **five of the seven bugs were found by running the pipeline end to end and reading the printed intermediate quantities, not by unit tests.** Print the sample flow and check it every time.
