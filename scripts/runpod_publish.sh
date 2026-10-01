#!/usr/bin/env bash
# Aggregate, then push ONLY non-student-level outputs to the r2-results branch.
set -euo pipefail
WORK=/workspace; CODE=$WORK/code; PUB=$WORK/pub
export VLPSO_PROJECT_ROOT=$WORK/proj
BRANCH="r2-results-${POD_TAG:-main}"
cd "$CODE"
python scripts/cells.py reconcile >/dev/null
python scripts/cells.py aggregate >/dev/null 2>&1 || echo "aggregate failed (partial run?)"
rm -rf "$PUB"
REMOTE=git@github.com:yazanjer/An_Explainable_AI_Education.git
if git clone -q --depth 1 --branch "$BRANCH" "$REMOTE" "$PUB" 2>/dev/null; then :; else
  git init -q "$PUB" && git -C "$PUB" remote add origin "$REMOTE" && git -C "$PUB" checkout -q --orphan "$BRANCH"
fi
rm -rf "$PUB/round2"  # each pod owns its branch; merged by experiments.merge
python scripts/cells.py publish --dest "$PUB/round2"
tail -n 400 "$WORK/run.log" | grep -v DEPLOY_PUBKEY > "$PUB/round2/run_log_tail.txt" || true
cd "$PUB"
git add -A
git -c user.name="Yazan Aljeroudi" -c user.email="ceo@rachis.co" commit -qm "Round-2 results snapshot $(date -u +%FT%TZ) from code $(git -C "$CODE" rev-parse --short HEAD)" || exit 0
git push -q origin "$BRANCH"
echo "published $(date -u +%FT%TZ)"
