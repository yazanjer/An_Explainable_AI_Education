# VLPSO-XAI on PISA 2018 — leakage-proof rebuild

Reproducible analysis code for *An Explainable AI Framework for Student Performance Prediction Using Variable-Length Particle Swarm Optimization in Educational Data Mining* (revision submitted to *Applied System Innovation*). The submitted version contained outcome leakage: the mathematics score and all thirty-plus plausible-value columns remained inside the feature matrix, so the reported AUC of 1.0000 was an artefact. This repository rebuilds the analysis around a nested, school-grouped, leakage-guarded design and reports the honest — considerably lower — results. See [`CHANGELOG_REVISION.md`](CHANGELOG_REVISION.md) for every change and its numerical consequence, and [`RESPONSE_TO_EDITOR.md`](RESPONSE_TO_EDITOR.md) for the point-by-point reply.

---

## Notebooks

**Start here:** [`RUN_ALL.ipynb`](notebooks/RUN_ALL.ipynb) does everything in
order — mount Drive, clone, install, test, download PISA, smoke test, full run,
collect results. Seven cells, top to bottom, nothing to edit but two switches.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yazanjer/An_Explainable_AI_Education/blob/main/notebooks/RUN_ALL.ipynb)

The numbered notebooks below are the same analysis split by **which editor
comment each answers** — useful for a reviewer checking a specific claim, and
unnecessary if you just want the results.

| # | Notebook | Answers | Quick | Full | Hardware |
|---|---|---|---|---|---|
| 00 | [Setup and data](notebooks/00_setup_and_data.ipynb) | comment 3 | 10 min | 40 min | CPU |
| 01 | [Leakage audit](notebooks/01_leakage_audit.ipynb) | comments 1, 2 | 5 min | 5 min | CPU |
| 02 | [Nested CV baseline](notebooks/02_nested_cv_baseline.ipynb) | comment 1 | 30 min | 6 h | CPU / T4 |
| 03 | [Feature-selection comparison](notebooks/03_feature_selection_comparison.ipynb) | comment 5 | 1 h | 12 h | CPU |
| 04 | [Ablations](notebooks/04_ablations.ipynb) | comment 5 | 40 min | 4 h | CPU |
| 05 | [Statistical validation](notebooks/05_statistical_validation.ipynb) | comments 4, 6 | 2 h | 20 h | CPU |
| 06 | [Explainability](notebooks/06_explainability.ipynb) | comments 7, 8 | 1 h | 3 h | CPU |
| 07 | [Manuscript assets](notebooks/07_generate_manuscript_tables.ipynb) | comment 10 | 5 min | 5 min | CPU |

Colab badges are embedded at the top of each notebook. Replace `yazanjer` in the badge URLs after publishing.

---

## Google Drive setup

Follow this once. Total time ≈ 20 minutes, most of it the OECD download.

### Step 1 — Do NOT create the folder

`git clone` creates it. Creating it by hand first makes the clone fail with
`exit status 128`, because git refuses to clone into a non-empty directory.
(`RUN_ALL.ipynb` detects this and fetches in place instead, but the numbered
notebooks do not.)

The folder will be:

```
MyDrive/An_Explainable_AI_Education/
```

The name matters: `src/vlpso_xai/config.py` looks for it first when running under Colab. For a different name or location, see Step 6.

This structure is built automatically on first run:

```
MyDrive/An_Explainable_AI_Education/
├── data/
│   ├── raw/          <- the PISA .sav goes here (Step 3)
│   └── processed/    <- generated parquet cache
├── results/
│   ├── tables/       <- .csv and .tex for the manuscript
│   ├── figures/
│   └── checkpoints/  <- per-fold resume points
└── models/
```

### Step 2 — Clone the repository into Drive

In a Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive
!git clone https://github.com/yazanjer/An_Explainable_AI_Education.git
%cd An_Explainable_AI_Education
!pip install -q -r requirements.txt
```

Cloning **into Drive** rather than into the Colab runtime is deliberate: the runtime is wiped on disconnect, Drive is not.

### Step 3 — Get the PISA 2018 data

The OECD licence permits download from their site but not republication, so the data is **not** in this repository.

Easiest — let notebook 00 fetch it:

```python
!python scripts/run_all.py --config quick --stages ingest
```

Or manually:

```python
%cd /content/drive/MyDrive/An_Explainable_AI_Education/data/raw
!curl -L -O https://webfs.oecd.org/pisa2018/SPSS_STU_QQQ.zip
!unzip -q SPSS_STU_QQQ.zip
```

| Item | Size |
|---|---|
| `SPSS_STU_QQQ.zip` | ≈ 500 MB |
| `CY07_MSU_STU_QQQ.sav` (extracted) | ≈ 1.8 GB |
| Contents | 612,004 students × 1,119 columns |
| Generated parquet cache | ≈ 300 MB |

**Budget ~3 GB of Drive space.** Download once; every later run reads the cache.

Landing page (if the direct URL changes): <https://www.oecd.org/en/data/datasets/pisa-2018-database.html>

### Step 4 — Verify before committing to a long run

```python
import os, sys
os.environ['VLPSO_PROJECT_ROOT'] = '/content/drive/MyDrive/An_Explainable_AI_Education'
sys.path.insert(0, os.environ['VLPSO_PROJECT_ROOT'] + '/src')

from pathlib import Path
from vlpso_xai.config import load_config, environment_report

cfg = load_config('quick')
print('root :', cfg.paths.root)
print('hash :', cfg.hash()[:12])

sav = list(Path(cfg.paths.data_raw).rglob('CY07_MSU_STU_QQQ.sav'))
assert sav, 'PISA .sav not found under data/raw - repeat Step 3'
print('data :', sav[0], f'{sav[0].stat().st_size/1e9:.1f} GB')

for pkg, ver in environment_report()['packages'].items():
    if ver is None:
        print('MISSING:', pkg)
```

Then run the test suite — it needs no PISA data and takes about a minute:

```python
!python -m pytest tests/ -q
```

**146 tests should pass.** If they do, the environment is sound.

### Step 5 — Run

```python
# Smoke test first (~15 min). Never skip this before a full run.
!python scripts/run_all.py --config quick

# Full budget (~20-40 T4-hours; resumes after disconnects)
!python scripts/run_all.py --config default
```

Or work through `notebooks/00` to `07` in order. Each has a Colab badge at the top.

**Use a GPU runtime** (Runtime → Change runtime type → T4) for notebooks 02–05. **Use a high-RAM runtime** for notebook 00, which reads the 1.8 GB `.sav`.

### Step 6 — A different Drive location

Set the environment variable **before** importing anything from `vlpso_xai`:

```python
import os
os.environ['VLPSO_PROJECT_ROOT'] = '/content/drive/MyDrive/wherever/you/like'
```

Resolution order is `VLPSO_PROJECT_ROOT` → Drive folder → repository root → `./` with a warning. Nothing is hard-coded, unlike the original `codes/config.py`, which raised an exception unless the path contained a specific magic substring.

### Step 7 — Resume after a disconnect

Just re-run the same command. Every nested-CV fold is checkpointed to `results/checkpoints/` as parquet, keyed by task, plausible value, method, repeat, fold **and a hash of the full configuration** — selector class, all its hyperparameters, the feature list, and whether weights were used. Completed folds are skipped; a disconnect costs at most one fold.

The configuration hash matters. Without it, checkpoints from a 3-fold run would be silently reloaded by a 5-fold run, and changing `Chi2Filter(k=15)` to `k=10` would reuse the old result. Both bugs occurred during development and are now impossible.

To force a clean re-run:

```python
!rm -rf results/checkpoints/*
```

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MessageError: credential propagation was unsuccessful` | Drive mount timed out | Re-run with `drive.mount('/content/drive', force_remount=True)` |
| `OSError: [Errno 5] Input/output error` | Drive rate limit or quota | Wait a few minutes and re-run; checkpoints resume |
| Kernel dies reading the `.sav` | 1.8 GB exceeds standard-runtime RAM | Switch to a high-RAM runtime, or run only `--stages ingest`, which reads in column batches |
| `FileNotFoundError: ... .sav is missing` | Data not downloaded, or in the wrong folder | Repeat Step 3; the file must sit under `data/raw/` (any subfolder depth) |
| `ModuleNotFoundError` after a Colab update | Colab bumped a base package | Re-run `pip install -r requirements.txt`, then Runtime → Restart |
| `LeakageError` | A forbidden column reached `X` | Read the message: it names the column and why. **This is the guard working, not a bug.** |
| `ValueError: beta_stagnation >= max_iter` | Length adaptation could never fire | Lower `beta_stagnation` or raise `max_iter` in `config/default.yaml` |
| `TypeError: global_shap expects a bare fitted estimator` | Passed a whole `Pipeline` | Use `transformed_frames(pipe, Xtr, Xte)` and pass `pipe.named_steps['clf']` |
| Permuted AUC is not ≈ 0.50 | Wrong null | Only **unrestricted** permutation should give 0.50; within-school permutation is expected to be higher (~0.62) |
| Disk full on Drive | zip + sav + parquet ≈ 3 GB | Delete `SPSS_STU_QQQ.zip` after extraction |

## Local installation

```bash
git clone https://github.com/yazanjer/An_Explainable_AI_Education.git
cd vlpso-xai-pisa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export VLPSO_PROJECT_ROOT="$PWD"
pytest -q          # 104 tests, no PISA data required
```

---

## Repository structure

```
config/          default.yaml, quick.yaml, predictor_allowlist.yaml, codebook_*.yaml
src/vlpso_xai/
  data/          ingest, codebook, outcome (plausible values), design (BRR), features (leakage guard)
  selection/     vlpso, bpso, filters (incl. symmetric uncertainty), wrappers, embedded
  models/        registry, pipeline factory
  evaluation/    nested_cv, metrics, effect_size, permutation, stability
  explain/       shap_global, shap_local, lime_local, consistency
  reporting/     tables, figures, manifest
notebooks/       00-07, Colab-ready, outputs stripped
scripts/         run_all.py, make_manuscript_assets.py, build_codebook.py, make_verification_extract.py
tests/           104 tests, synthetic data only
```

---

## Reproducing each manuscript table

| Table | Produced by |
|---|---|
| Sample characteristics and flow | `notebooks/00` → `results/sample/sample_flow.csv` |
| Candidate predictor list (supplementary) | `config/predictor_allowlist.yaml` |
| Leakage demonstration | `notebooks/01` → `results/audit/leakage_delta.csv` |
| Nested-CV performance by task | `notebooks/02` → `results/tables/table_nested_cv.csv` |
| Selector comparison | `notebooks/03` → `results/tables/table_selector_comparison.csv` |
| Ablations | `notebooks/04` → `results/ablations.parquet` |
| Bootstrap CIs, permutation, stability, effect sizes | `notebooks/05` → `results/tables/` |
| Global SHAP, explainer settings, LIME stability | `notebooks/06` → `results/tables/` |
| Everything as `.tex` + manifest | `notebooks/07` |

Or in one command:

```bash
python scripts/run_all.py --config default
```

---

## Data availability

Analyses use the OECD PISA 2018 database, which is publicly available at <https://www.oecd.org/en/data/datasets/pisa-2018-database.html> under the OECD terms of use. The data are **not** redistributed here; `data/` and `results/` are gitignored. All code is MIT-licensed (see [LICENSE](LICENSE)); the licence covers the code only.

---

## Citation

See [`CITATION.cff`](CITATION.cff).

---

## Verification

[`VERIFICATION_REPORT.md`](VERIFICATION_REPORT.md) records the outcome of every
check, including the ones that failed. Headlines:

* **309 tests pass**, on synthetic data, no PISA microdata required.
* Leakage guard raises on outcome, plausible-value, weight, design, identifier
  and 20 adversarial derived/encoded column names, with no false positives.
* Unrestricted permutation returns **AUC 0.5029** (n = 20). The pipeline is clean.
* Two quick-mode runs with the same seed produce **byte-identical** hashes for
  all 140 artefacts.
* An independent adversarial audit found 15 defects in this rebuild; **8 are
  fixed**, 7 remain open and are listed with severities.

Not yet done: a cold-start Colab run end to end, the SHAP/LIME numbers, and the
manuscript edits for editor comments 8 and 10.

## Known limitations

Stated plainly, because the previous version of this work overclaimed.

1. **Single country, single cycle.** Spain, PISA 2018. Portugal is held out as an external test set, but two Western-European systems do not establish general transfer.
2. **Cross-sectional, observational data. No causal identification.** SHAP and LIME describe how a fitted model uses a variable. They do not show that the variable causes achievement or that changing it would change outcomes. A SHAP decision plot has no time axis.
3. **Feature selection is unstable.** VLPSO shares only ~24% of its selected features across outer folds. Any statement about "the" selected feature set is a statement about a modal tendency, not a stable result.
4. **The proposed optimiser does not outperform its baselines** on current evidence. It scores below ordinary binary PSO and below using all features, and the variable-length mechanism performs worse than shrink-only. The contribution is the evaluation protocol and the codebook-grounded explainability analysis, not the optimiser.
5. **Plausible values dominate the uncertainty.** The fraction of missing information is 0.54, so more than half the variance in the performance estimate comes from measurement uncertainty in proficiency rather than from sampling.
6. **Proficiency categories are unstable.** 69.7% of students change category depending on which plausible value is used. The three-class framing is a coarsening of a continuous, uncertain quantity.
7. **Computational budget.** Permutation tests and the per-PV repetition are expensive; where a reduced budget was used it is stated in the results tables rather than silently applied.
8. **The wrapper fitness is evaluated on a stratified subsample** of each training fold, because k-NN is O(n²). The final model is always refitted on the complete training fold.
9. **School-level questionnaire data is not merged.** It requires codebook documentation we do not currently hold, and adding items without a documentary source is the defect this revision exists to correct.
