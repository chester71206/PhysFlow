# Bird observed RAFT vs same-Warp synthetic flow: controlled diagnostic

## Experiment 1: observed WonderPlay loss near converted GT

The completed 20×20 Bird grid has 400 finite points. Its sampled best is `E=0.02434558`, `nu=0.38157895`, loss `17.43868637`. The separately evaluated converted GT point is `E=0.03884070`, `nu=0.3`, loss `17.45750999`. GT minus grid best is `0.01882362`, or `0.108%` of GT loss. The sampled best E is `0.627×` converted GT E, and its ν is `0.08158` higher. This is a finite-grid comparison, not a continuous global minimum.

The grid does **not** contain the exact GT E column. Its nearest E is `0.03584147`; at that E, the ν slice spans loss `17.44124–17.44835`, and ν near `0.3` is within that low-loss band. The separately evaluated exact `(E_GT, nu_GT)` has loss `17.45751`, so substituting the nearest E column for the exact-E slice would be misleading. The new observed-lines job evaluates all 20 ν grid values at **exact** initial E and **exact** converted GT E, plus exact ν initial and GT points. Its output is [bird_observed_exact_E_lines_20261119](../outputs/bird_observed_exact_E_lines_20261119).

## Experiment 2: synthetic motion target

The reference target comes from a new forward rollout at converted `E=0.03884069931590986`, `nu=0.3`, using the same Bird geometry, initial particles, gravity, boundaries, cameras, target steps, Gaussian opacities/covariances, and 20,000-point WonderPlay rasterizer. The target is the rasterizer's **physical pixel-flow field**, captured during the unchanged WonderPlay objective call. The foreground mask remains the existing fixed observed-video WonderPlay mask for both target arms. No RAFT inference is run on rendered synthetic RGB, and this target is **not** PAC-NeRF ground-truth optical flow.

Each candidate E,ν runs a fresh `RepairedTemporalRollout.forward` from the same engine initial state. One rollout yields both observed-RAFT WonderPlay loss and synthetic-target WonderPlay loss. The script runs no backward or optimizer. The target flow tensor and fixed-state fingerprints are saved, so resume uses the exact same target. [Source script](../scripts/bird_synthetic_flow_landscape.py); [launcher](../scripts/run_bird_synthetic_diagnostic.sh).

The smoke test reproduced the old observed loss exactly at grid point `(E,nu)=(5e-5,0.05)`: `17.617872238`. At the converted GT rollout, observed loss was `17.457509995`, matching the old GT evaluation. A separate GT rollout compared to the stored synthetic flow gave synthetic loss `0.0`. These checks verify routing and repeatability at tested points.

**Interpretation limit:** Since target and prediction use the same forward model and rasterizer, GT synthetic loss is zero *by construction*. It cannot independently validate the physical model or flow implementation. Its useful result is the **shape and width** of near-zero regions: if many materially different pairs also have tiny synthetic loss, the chosen observation is not identifying E,ν well under this model and mask. If synthetic loss rises sharply away from GT while observed loss prefers another region, that supports a target/data/model mismatch, but the two cannot be separated without an independent target construction or further controls.

The complete 20×20 synthetic grid is running in two disjoint shards on GPU 9 and, after the exact-E lines finish, GPU 0. Results are saved point by point in [bird_synthetic_flow_full_20261119](../outputs/bird_synthetic_flow_full_20261119). The launcher will write [summary.json](../outputs/bird_synthetic_flow_diagnostic_20261119/summary.json), [REPORT.md](../outputs/bird_synthetic_flow_diagnostic_20261119/REPORT.md), and [PNG](../outputs/bird_synthetic_flow_diagnostic_20261119/bird_observed_vs_synthetic_diagnostic.png) when each stage completes. A `COMPLETE` marker is written only after both shards and the final report succeed.

Experiment 3 (initialization strategies) is deferred until the observed exact-E slice and synthetic ambiguity width are measured. The current data already show a local barrier near initial E, but a synthetic low-loss valley or weak ν sensitivity could change which follow-up is appropriate.
