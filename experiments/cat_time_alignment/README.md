# Cat: original versus time-aligned experiment

## Preserved baseline

- Existing run: `outputs/pacnerf/cat_ds1.0_ep50/` (left in place).
- Existing entry point: `simulation.py` (unchanged by this experiment).
- Author configuration: `config/pacnerf/cat_config.json` (unchanged).
- `baseline/simulation_runtime_baseline.py` and `cogv_guidance_runtime_baseline.py` snapshot the local runtime used before this experiment, including previous compatibility, CPU RAFT and numeric sorting changes.
- `baseline/simulation_upstream.py` and `cat_config_upstream.json` preserve the pristine files from the Git commit recorded in `baseline/manifest.json`.

The existing run is a local-runtime baseline, not a claim of bitwise reproduction of the paper. Snapshot scripts are archival references, not standalone launchers.

## Independent experiment

Entry point: `simulation_time_aligned.py`.
Configuration: `config/pacnerf/cat_time_aligned_config.json`.
Output: `outputs/pacnerf/cat_time_aligned_ds1.0_ep50/`.

Only training timing changes: 13 target frames use the explicit timestamps from PAC-NeRF view 10 (1/24 through 13/24 seconds). Absolute target times are rounded to MPM substeps, with maximum error half a substep. The baseline's batch-dependent time offset is removed. The baseline final-substep-only gradient approximation is retained.

Material parameters, learning rates, initialization, gravity, initial velocity, scene normalization, objective weights, geometry supervision, and rendering cadence remain the same. `opt_frame_dt` remains in the config for comparison but the aligned entry point uses `experiment.reference_frame_times` instead. This experiment does NOT implement the proposed initial-velocity/gravity corrections.

The aligned script also logs the separate weighted loss components, snapshots its config and timing schedule in the run directory, and rejects nonempty output directories during training. It explicitly rejects the original run path. Evaluation uses the existing checkpoint and does not create a new training run.

```bash
cd /home/ccchen/PhysFlow
conda activate physflow
env -u LD_LIBRARY_PATH WANDB_MODE=offline OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python simulation_time_aligned.py \
  --model_path ./model/pacnerf/cat \
  --physics_config ./config/pacnerf/cat_time_aligned_config.json \
  --output_path ./outputs/pacnerf/cat_time_aligned \
  --n_epoch 50 --n_key_frame 13 --guidance-mode raft_cpu
```

Add `--dry-run` to validate and print the schedule without starting training or creating a run directory. Importing the script still initializes the CUDA libraries.

For a fresh baseline repeat, use `simulation.py` with `cat_config.json` and a NEW output prefix such as `./outputs/pacnerf/cat_original_repeat`; the baseline entry point itself does not reject overwriting existing outputs.

## Comparison

Compare both checkpoints and videos using the same physical timestamps. Rendering cadence is unchanged (0.005 seconds per saved frame), so corresponding video frame indices have the same simulation time, but they are not reference image indices (1/24 seconds).

The loss curves use different training timestamps and cannot alone establish improvement. Evaluate both checkpoints at the same reference timestamps for geometry and optical flow. Keep initial-velocity and gravity experiments separate.

This change was checked with unit tests for timestamp rounding/validation, baseline/config identity checks, and the aligned entry point's dry run. A full aligned training run has not yet been performed.
