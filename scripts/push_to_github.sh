#!/usr/bin/env bash
# ===========================================================================
# Publish this repository to GitHub.
#
#   bash scripts/push_to_github.sh
#
# Safe to re-run. Verifies BEFORE pushing that no PISA microdata is staged --
# the OECD licence permits download from their site, not republication.
# ===========================================================================
set -euo pipefail

REPO_URL="https://github.com/yazanjer/An_Explainable_AI_Education.git"
BRANCH="main"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "==> Repository root: $HERE"

# --- 1. Identity -----------------------------------------------------------
git config user.name  >/dev/null 2>&1 || git config user.name  "Yazan Aljeroudi"
git config user.email >/dev/null 2>&1 || git config user.email "ceo@rachis.co"

# --- 2. Init (idempotent) --------------------------------------------------
if [ ! -d .git ]; then
  git init -q
  echo "==> Initialised a new repository"
else
  echo "==> Reusing the existing repository"
fi
git symbolic-ref HEAD "refs/heads/$BRANCH" 2>/dev/null || git checkout -q -B "$BRANCH"

# --- 3. Strip notebook outputs --------------------------------------------
if command -v nbstripout >/dev/null 2>&1; then
  nbstripout notebooks/*.ipynb && echo "==> Notebook outputs stripped"
else
  echo "==> nbstripout not installed (pip install nbstripout) - skipping"
fi

# --- 4. Stage --------------------------------------------------------------
git add -A

# --- 5. SAFETY GATE: refuse to publish PISA microdata ----------------------
echo "==> Checking for data files in the staging area ..."
OFFENDERS="$(git diff --cached --name-only \
  | grep -Ei '\.(sav|parquet|zip|feather)$|(^|/)data(_local)?/|(^|/)STU/|(^|/)results/' \
  || true)"
if [ -n "$OFFENDERS" ]; then
  echo "!! ABORTING - these would be published:"
  echo "$OFFENDERS" | sed 's/^/     /'
  echo "   PISA microdata is not redistributable. Fix .gitignore, then:"
  echo "     git rm -r --cached <path>"
  exit 1
fi
echo "   clean - no microdata staged"

STAGED=$(git diff --cached --name-only | wc -l | tr -d ' ')
echo "==> $STAGED file(s) staged"
[ "$STAGED" -eq 0 ] && { echo "==> Nothing to commit."; exit 0; }

# --- 6. Commit -------------------------------------------------------------
git commit -q -F - <<'MSG'
Leakage-proof rebuild of the VLPSO-XAI PISA 2018 analysis

Rebuilds the analysis after confirming outcome leakage in the submitted
version: data_preparation.py:253 dropped only object-dtype columns, so
math_score and all 30+ plausible-value columns survived into the feature
matrix. A depth-1 tree on math_score alone scores AUC 1.0000.

- allowlist + denylist leakage guard, enforced by exception at every entry point
- nested CV, StratifiedGroupKFold on CNTSCHID, model and threshold selected
  inside the inner loop
- ten plausible values combined by Rubin's rules, variance decomposed
- survey weights and Fay BRR replicate standard errors
- genuinely variable-length PSO, plus matched BPSO and filter baselines
- SHAP/LIME rebuilt on the fitted pipeline's feature space
- 146 tests, including leakage-detection-power tests calibrated against a
  known-leaky reference

Honest headline: AUC falls from 1.0000 to 0.8766 (Low vs. High) once the
leakage is removed and school-level grouping applied.
MSG
echo "==> Committed"

# --- 7. Remote -------------------------------------------------------------
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi
echo "==> origin -> $REPO_URL"

# --- 8. Push ---------------------------------------------------------------
echo "==> Pushing to $BRANCH (use a Personal Access Token as the password)"
git push -u origin "$BRANCH"
echo "==> Done: https://github.com/yazanjer/An_Explainable_AI_Education"
