#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

exec "${PHYSFLOW_PYTHON:-python}" scripts/scan_letter_local_E_nu.py \
  --output "${OUTPUT:-outputs/letter_local_E_nu}" \
  --max-new "${MAX_NEW:-10}" \
  --gpu "$CUDA_VISIBLE_DEVICES" \
  "$@"
