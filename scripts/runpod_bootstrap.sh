#!/usr/bin/env bash
# RunPod entry point for the round-2 analyses. Idempotent: a restarted pod
# resumes from the completed cells on the /workspace volume.
#
# Data: the PISA 2018 student file is downloaded from the OECD into this pod
# only. It is never committed, pushed, or copied off the pod (licence: download
# permitted, redistribution not). Only non-student-level outputs leave the pod,
# through scripts/runpod_publish.sh and vlpso_xai.experiments.publish, which
# refuses any table with a student-level column.
set -uo pipefail
WORK=/workspace
CODE=$WORK/code
export VLPSO_PROJECT_ROOT=$WORK/proj
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$WORK"
exec > >(tee -a "$WORK/run.log") 2>&1
echo "== bootstrap $(date -u +%FT%TZ) pod=${RUNPOD_POD_ID:-?} commit=$(git -C "$CODE" rev-parse --short HEAD)"

command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip openssh-client >/dev/null)

# Deploy key generated ON the pod; the private half never leaves it. The public
# half is printed for the operator to register as a deploy key on the code repo.
mkdir -p ~/.ssh && chmod 700 ~/.ssh
[ -f "$WORK/.ssh_id" ] || ssh-keygen -q -t ed25519 -N "" -f "$WORK/.ssh_id" -C "runpod-r2-${RUNPOD_POD_ID:-pod}"
cp "$WORK/.ssh_id" ~/.ssh/id_ed25519 && cp "$WORK/.ssh_id.pub" ~/.ssh/id_ed25519.pub && chmod 600 ~/.ssh/id_ed25519
ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
echo "DEPLOY_PUBKEY_BEGIN $(cat ~/.ssh/id_ed25519.pub) DEPLOY_PUBKEY_END"

export PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
python -m pip install -q -U pip setuptools wheel
python -m pip install -q numpy==2.0.2 pandas==2.2.2 scikit-learn==1.6.1 scipy pyyaml pyarrow \
    pyreadstat matplotlib joblib tqdm || { echo "FATAL: core install failed"; sleep infinity; }
python -m pip install -q shap==0.52.0 lime==0.2.0.1 || { echo "FATAL: shap/lime install failed"; sleep infinity; }
python -c "import numpy, pandas, sklearn, scipy, shap, lime, pyreadstat; print('== versions numpy', numpy.__version__, 'pandas', pandas.__version__, 'sklearn', sklearn.__version__, 'shap', shap.__version__)" \
    || { echo "FATAL: import check failed"; sleep infinity; }

mkdir -p "$VLPSO_PROJECT_ROOT/data/raw" "$VLPSO_PROJECT_ROOT/config" "$VLPSO_PROJECT_ROOT/results"
cp -r "$CODE"/config/* "$VLPSO_PROJECT_ROOT/config/"
if ! find "$VLPSO_PROJECT_ROOT/data/raw" -name CY07_MSU_STU_QQQ.sav | grep -q .; then
  echo "== downloading PISA 2018 student questionnaire from the OECD"
  curl -fsSL -o "$VLPSO_PROJECT_ROOT/data/raw/SPSS_STU_QQQ.zip" https://webfs.oecd.org/pisa2018/SPSS_STU_QQQ.zip
  sha256sum "$VLPSO_PROJECT_ROOT/data/raw/SPSS_STU_QQQ.zip"
  unzip -o -q "$VLPSO_PROJECT_ROOT/data/raw/SPSS_STU_QQQ.zip" -d "$VLPSO_PROJECT_ROOT/data/raw" \
    && rm -f "$VLPSO_PROJECT_ROOT/data/raw/SPSS_STU_QQQ.zip"
fi

cd "$CODE"
python -c "import json,sys; sys.path.insert(0,'src'); from vlpso_xai.config import environment_report as e; json.dump(e(), open('$VLPSO_PROJECT_ROOT/results/environment.json','w'), indent=1, default=str)"
python scripts/cells.py manifest

( while true; do sleep 1500; bash scripts/runpod_publish.sh >> "$WORK/publish.log" 2>&1 || true; done ) &

KINDS="${KINDS:-eda,shap,ext,perm,vlstab,sel}"
SHARD_ARG=""; [ -n "${SHARD:-}" ] && SHARD_ARG="--shard $SHARD"
echo "== pod role: kinds=$KINDS shard=${SHARD:-all} workers=${WORKERS:-32} max_hours=${MAX_HOURS:-30}"
python scripts/cells.py run --kinds "$KINDS" $SHARD_ARG --workers "${WORKERS:-32}" --max-hours "${MAX_HOURS:-30}"
if [[ ",$KINDS," == *",shap,"* ]]; then python scripts/cells.py run --kinds brr --workers "${WORKERS:-8}"; fi
python scripts/cells.py run --kinds "$KINDS" $SHARD_ARG --workers "${WORKERS:-32}" --max-hours 2   # retry failures once
python scripts/cells.py reconcile
bash scripts/runpod_publish.sh
echo "ALL_DONE $(date -u +%FT%TZ)"
sleep infinity
