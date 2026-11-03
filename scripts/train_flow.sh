#!/usr/bin/env bash
# GPU defaults to physical 0; override GPU or CUDA_VISIBLE_DEVICES on any host.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export FLOW_P2G_PRIMAL=fp64_atomic
export FLOW_G2P_REDUCTION=fp64_atomic
export FLOW_ANALYTIC_G2P=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1
task_python="${PYTHON:-python}"
exec "$task_python" scripts/train_projected_flow_gradient.py --check-after-training "$@"
