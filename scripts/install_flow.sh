#!/usr/bin/env bash
# Run inside a fresh Python 3.9 environment, with a CUDA 11.8 toolkit installed.
set -euo pipefail
cd "$(dirname "$0")/.."
task_python="${PYTHON:-python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export MAX_JOBS="${MAX_JOBS:-2}"
"$task_python" -c 'import sys; assert sys.version_info[:2] == (3,9), "Use Python 3.9"'
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  printf 'CUDA 11.8 toolkit required; set CUDA_HOME to its directory. Missing: %s/bin/nvcc\n' "$CUDA_HOME" >&2
  exit 2
fi
"$task_python" -m pip install pip==24.0 setuptools==68.0.0 wheel==0.43.0
"$task_python" -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
"$task_python" -m pip install -r requirements-flow.txt
"$task_python" -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@fdaf9bd6fed7977e4c2056e7c77c640781e58fcd'
"$task_python" -m pip install --no-build-isolation ./submodules/simple-knn ./submodules/diff-plane-rasterization
"$task_python" scripts/check_flow_install.py
