#!/usr/bin/env bash
# Usage: nohup bash scripts/run_pacnerf_gradient_gpu3.sh [NEW_OUTPUT_ROOT] > runner.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export FLOW_P2G_PRIMAL=fp64_atomic
export FLOW_G2P_REDUCTION=fp64_atomic
export FLOW_ANALYTIC_G2P=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
task_root="${1:-outputs/pacnerf_gradient_$(date +%Y%m%d_%H%M%S)}"
exec "${PYTHON:-python}" scripts/run_pacnerf_gradient_comparison.py \
  --root "$task_root" --epochs 50
