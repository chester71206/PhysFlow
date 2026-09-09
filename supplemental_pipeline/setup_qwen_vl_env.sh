#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT=/home/ccchen/miniconda3
ENV_NAME=qwen_vl
ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"

source "$CONDA_ROOT/etc/profile.d/conda.sh"

export CONDA_NO_PLUGINS=true
export CONDA_OVERRIDE_CUDA=12.1

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    conda create --solver classic -y -p "$ENV_PREFIX" python=3.11 pip
fi

"$ENV_PREFIX/bin/python" -m pip install --upgrade pip
"$ENV_PREFIX/bin/python" -m pip install "numpy<2"
"$ENV_PREFIX/bin/python" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
"$ENV_PREFIX/bin/python" -m pip install \
    "accelerate>=1.5" \
    "safetensors>=0.4.3" \
    pillow \
    sentencepiece \
    qwen-vl-utils \
    git+https://github.com/huggingface/transformers
