#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PHYSFLOW_PYTHON:-python}"
gpu="${GPU:-0}"
output="${OUTPUT:-outputs/letter_material_to_gt}"

exec "$python_bin" scripts/sweep_letter_material_to_gt.py \
  --start original=data/pacnerf/letter_material_starts/original.json \
  --start projected=data/pacnerf/letter_material_starts/projected_flow_ad.json \
  --physics-config config/gradient_repair/letter_flow_ad.json \
  --kinematics-manifest data/pacnerf/letter_material_starts/kinematics_manifest.json \
  --model-path model/pacnerf/letter \
  --output "$output" \
  --points "${POINTS:-7}" \
  --spacing "${SPACING:-log}" \
  --gpu "$gpu" \
  "$@"
