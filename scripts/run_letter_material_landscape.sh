#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

reuse_root="${REUSE_SWEEP:-outputs/letter_material_to_gt}"
reuse_args=()
if [[ -f "$reuse_root/manifest.json" && -f "$reuse_root/status.json" ]]; then
  reuse_args=(--reuse-sweep "$reuse_root")
fi

exec "${PHYSFLOW_PYTHON:-python}" scripts/scan_letter_material_landscape.py \
  --output "${OUTPUT:-outputs/letter_material_landscape}" \
  "${reuse_args[@]}" \
  --path-points "${PATH_POINTS:-25}" \
  --ray-points "${RAY_POINTS:-9}" \
  --grid-points "${GRID_POINTS:-5}" \
  --fold "${FOLD:-2}" \
  --max-new "${MAX_NEW:-10}" \
  --gpu "$CUDA_VISIBLE_DEVICES" \
  "$@"
