# Far-start exact-flow optimization

See [REPORT.md](REPORT.md) and [summary.csv](summary.csv). This extends the November 22 Gaussian self-consistency experiment with E=2×GT, ν=.16 and E=.5×GT, ν=.44 for baseball and deflated football. Each start uses the saved Warp MPM GT initial particles, `RepairedTemporalRollout` reverse AD, three Blender camera masks, frames 0/2/4/6/8, 128×128 WonderPlay Gaussian exact-flow target, and the original WonderPlay loss. RAFT is absent; finite differences only check initial gradients.

The training command for a start is:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 900s python scripts/train_synthetic_wonderplay_multistart.py \
  --scene outputs/synthetic_physics_batch_20261121/scenes/baseball_drop \
  --output outputs/synthetic_wonderplay_farstart_20261122/baseball \
  --start-name highE_lowNu --start-E-ratio 2.0 --start-nu 0.16 \
  --max-frame 8 --frame-stride 2 --resolution 128 --checkpoint-interval 16 \
  --gradient-gate --fd-E-fraction 0.02 --fd-nu-step 0.005 \
  --epochs 120 --lr-decay-at 70,100
```

For `lowE_highNu`, use ratio .5 and ν .44. Change the scene and output label to `football_deflated_2s` and `football` for football. The football lowE/highNu run resumes from the same optimizer checkpoint with `--epochs 240`, then `--epochs 270`, omitting `--gradient-gate` because the original initial gate is saved. `optimizer.pt` is atomically replaced after each update. Each run has an external timeout; a partial run resumes by repeating the command with the same settings and a larger `--epochs` if needed. Individual `history.csv` files, plots and `final.json` record each path; `history_120.csv`/`final_120.json` and `history_240.csv`/`final_240.json` retain the two football intermediate endpoints.

The independent Blender-flow target bias is analyzed separately in `outputs/synthetic_wonderplay_blender_bias_20261122/REPORT.md`. A successful Gaussian self-consistency run does not establish material recovery from Blender video.
