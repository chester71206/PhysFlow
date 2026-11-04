#!/usr/bin/env bash
# Serial 50-update Letter original vs projected-flow AD, independent evaluation, plots.
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

task_python="${PHYSFLOW_PYTHON:-python}"
root="${1:-outputs/letter_two_methods/letter}"
if [[ -e "$root" ]]; then
  printf 'Output already exists; choose a new path: %s\n' "$root" >&2
  exit 2
fi
"$task_python" scripts/check_flow_data.py --scene letter
mkdir -p "$root/original" "$root/projected_flow_AD"
config=config/gradient_repair/letter_flow_ad.json
cp "$config" "$root/original/physics_config.json"
cp "$config" "$root/projected_flow_AD/physics_config.json"

original="$root/original"
"$task_python" simulation_memory_safe.py \
  --config config/run_memory_safe.json \
  --model_path model/pacnerf/letter \
  --physics_config "$original/physics_config.json" \
  --output_path "$original/train" \
  --n_epoch 50 --n_key_frame 15 --stop-after-batches 50 \
  --skip-video --guidance-mode raft_cpu \
  > "$original/train.log" 2>&1
"$task_python" scripts/evaluate_universal_reference.py \
  --model-path model/pacnerf/letter \
  --physics-config "$original/physics_config.json" \
  --checkpoint-json "$original/train_ds1.0_ep50/ckpt/sim.json" \
  --key-frames 15 --output "$original/evaluation" --dataset pacnerf \
  > "$original/evaluation.log" 2>&1

projected="$root/projected_flow_AD"
PYTHON="$task_python" bash scripts/train_flow.sh \
  --model-path model/pacnerf/letter \
  --physics-config "$projected/physics_config.json" \
  --output "$projected/train" \
  --epochs 50 --interval 32 --lr 0.003 \
  > "$projected/train.log" 2>&1
"$task_python" scripts/evaluate_universal_reference.py \
  --model-path model/pacnerf/letter \
  --physics-config "$projected/physics_config.json" \
  --checkpoint-json "$projected/train_ds1.0_ep50/ckpt/sim.json" \
  --key-frames 15 --output "$projected/evaluation" --dataset pacnerf \
  > "$projected/evaluation.log" 2>&1

"$task_python" scripts/report_letter_two_methods.py --base "$root"
printf 'DONE: %s/detailed_comparison/REPORT.md\n' "$root"
