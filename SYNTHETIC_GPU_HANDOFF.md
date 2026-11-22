# PhysFlow / WonderPlay GPU handoff (2026-11-22)

This branch contains the code and compact source data for testing whether WonderPlay optimization recovers synthetic material GT when initial `E` differs by 10× or 100×. It also includes the reviewed research records and selected three-view simulation videos. The full local `outputs/` tree is much larger and is not in Git.

## On the other GPU machine

```bash
git clone --branch research/synthetic-wonderplay-transfer-20261122 --recurse-submodules \
  https://github.com/chester71206/PhysFlow.git
cd PhysFlow
conda env create -f environment.yml
conda activate physflow
CUDA_HOME=/path/to/cuda-11.8 bash scripts/install_flow.sh
CUDA_HOME=/path/to/cuda-11.8 python -m pip install --no-build-isolation \
  -e WonderPlay/submodules/depth_diff_gaussian_rasterization_min
python -m pip install matplotlib trimesh==4.4.0
python scripts/verify_synthetic_transfer.py
python scripts/run_synthetic_wonderplay_order_magnitudes.py
```

If an available GPU has persistent memory allocations and is rejected by the conservative automatic idle rule, specify its physical `nvidia-smi` index explicitly, for example `python scripts/run_synthetic_wonderplay_order_magnitudes.py --gpu 0`. Do this only for a GPU you are free to use. The runner prints which GPU it selected and where each trial logs its updates.

The validated local environment used Python 3.9, PyTorch 2.2.0+cu118, `warp-lang` 0.10.1, `trimesh` 4.4.0 and `opencv-python` 4.8.1.78. The WonderPlay repository is pinned as a Git submodule. `scripts/install_flow.sh` needs a CUDA 11.8 toolkit and compiles CUDA extensions. Set `CUDA_HOME` to the toolkit on that machine. The runner uses the current Python interpreter, picks an idle GPU after three checks, and never assumes GPU 8. Run it from the repository root.

For a quick data and numerical preflight before the full queue:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 240s python scripts/train_synthetic_wonderplay_multistart.py \
  --scene outputs/synthetic_physics_batch_20261121/scenes/baseball_drop \
  --output outputs/synthetic_wonderplay_order_magnitudes_20261122/baseball \
  --start-name E_x10 --start-E-ratio 10 --start-nu .32 \
  --E-parameterization log --E-min-ratio .001 --E-max-ratio 1000 \
  --start-probe --epochs 0
```

The full queue tests both baseball and deflated football at `E/GT = 10, 0.1, 100, 0.01`, with `ν` initialized at GT=.32 and both parameters optimized. It retains the original Warp MPM, `RepairedTemporalRollout` reverse AD, WonderPlay loss, three views, 128×128 images and frames 0/2/4/6/8. Log-E Adam makes steps comparable over orders of magnitude; this is a deliberate optimizer parameterization change from earlier linear-E tests. Centered differences are used only to check initial gradients. Every run has a finite/stability probe, timeouts, per-update history, and atomic optimizer checkpoints. A high-E forward failure under the fixed timestep is recorded as a numerical failure, not a local minimum.

## Included data and records

- Two GT scenes include `config.json`, `trajectory.npz`, camera metadata, initial-window masks and visible-surface flow. These exact relative paths are consumed by the trainer. The corresponding asset meshes and PBR sources are included under `outputs/synthetic_physics_batch_20261121/assets/`.
- The rubber duck scene includes its trajectory, parameters, QA and three-view video. All three selected scenes have camera 0/1/2 MP4s. Blender can regenerate full RGB/mask/depth passes from trajectories and source `.blend` assets; the full frame-by-frame render passes are omitted from this compact Git transfer.
- `outputs/synthetic_physics_batch_20261121/README.md` and `quality_report.md` document the physics/render QA. PASS means automated checks passed, not human certification of naturalness.
- `outputs/synthetic_wonderplay_multistart_20261122/REPORT.md` records earlier 0.7–1.35× starts; `outputs/synthetic_wonderplay_farstart_20261122/REPORT.md` records 0.5×/2× starts and the football continuation to 270 updates. These do **not** answer the 10×/100× question.
- `outputs/synthetic_wonderplay_blender_bias_20261122/REPORT.md` shows that independent Blender visible-mesh flow has nonzero loss and gradient at GT. Gaussian self-consistency recovery does not prove recovery from Blender video.
- `outputs/synthetic_wonderplay_order_magnitudes_20261122/` is the destination for the new 10×/100× results. Its `REPORT.md` is generated after the experiments run.

After running, review `runner_state.json`, `runner_summary.csv`, `REPORT.md`, the per-start `start_probe_*.json` and `gradient_gate_*.json`, and each start's `history.csv` and `final.json`. The full local 2026-11-22 queue was waiting for idle GPU 8/9 at handoff; do not interpret the absence of results as optimizer failure.
