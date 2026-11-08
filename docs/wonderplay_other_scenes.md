# WonderPlay rasterized-flow training on other PAC-NeRF scenes

Pull `projected-flow-final`, then run `git submodule update --init WonderPlay`.
Install the pinned rasterizer in the active `physflow` environment:

```bash
CUDA_HOME=/actual/path/to/cuda-11.8 python -m pip install --no-build-isolation -e WonderPlay/submodules/depth_diff_gaussian_rasterization_min
```

`scripts/run_pacnerf_projected.py --plan-only` verifies the scene assets and
backend. `--flow-objective wonderplay` currently supports reverse AD on Letter,
Droplet, Torus, and Bird. The other five PAC-NeRF scenes use finite difference
with the older point-flow loss; the launcher rejects a WonderPlay request for
those materials instead of silently changing the objective.

Run one update on each new AD scene first:

```bash
python scripts/run_pacnerf_projected.py --scene droplet --scene torus --scene bird \
  --flow-objective wonderplay --max-points 1000 --epochs 1 --gpu 8 \
  --root outputs/wonderplay_other_ad_smoke
```

Inspect `manifest.json`, each `train.log`, `gradient_check.json`, and the
one-update `training.jsonl`. If all three gradient gates pass, train 200
updates using a new output root:

```bash
python scripts/run_pacnerf_projected.py --scene droplet --scene torus --scene bird \
  --flow-objective wonderplay --max-points 1000 --epochs 200 --gpu 8 \
  --root outputs/wonderplay_other_ad_200
```

The runner executes scenes serially. A per-scene checkpoint is written after
each update. For interrupted runs, call `scripts/train_projected_flow_gradient.py`
with `--resume path/to/optimizer.pt`, `--flow-objective wonderplay`, the same
`--max-points`, scene/config, and a fresh output path. `--epochs` then means
additional updates. On a different machine, checkpoint signatures include
absolute model paths and input frame paths; resume a checkpoint only if those
paths match, or start a fresh 200-update run there.

Training loss values differ between point and rasterized objectives. Compare
checkpoint quality with `scripts/evaluate_universal_reference.py` using the
same scene config, selected frames, and native renderer for both methods.
