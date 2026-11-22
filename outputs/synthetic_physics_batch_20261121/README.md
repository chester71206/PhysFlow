# Synthetic Physics Dataset batch, 2026-11-21

This batch uses the existing PhysFlow Warp `MPM_Simulator_WARP.p2g2p` elastic forward and a static sticky ground collider. A licensed Poly Haven Blender asset supplies the tracked mesh, UVs, and PBR textures. The render script replaces each source mesh vertex with its corresponding MPM surface particle position. Blender adds lighting, shadows, cameras, floor, depth, and object mask; it does not add motion or deformation.

## Completed result

Nine candidate runs were recorded: **3 automated PASS**, **3 NEEDS_REVIEW**, **3 FAIL**. The PASS scenes are [baseball drop](scenes/baseball_drop/render_full/cam_0.mp4), [deflated football drop](scenes/football_deflated_2s/render_full/cam_0.mp4), and [low-drop rubber duck](scenes/duck_low_drop/render_full/cam_0.mp4). All have three fixed perspective cameras at 24 FPS, 480×480 RGB PNG and MP4 per view, Blender object masks and depth EXR, saved MPM particle/mesh trajectories, geometric visible-surface flow, repeatability reports, and short-path reverse AD checks. Baseball and football have 49 frames over 2 seconds; the duck has 25 frames over 1 second. [comparison.mp4](comparison.mp4) concatenates all three views of the three PASS scenes.

The 1-second `football_deflated_drop` run was the Pilot for the completed 2-second version. The duck pilot was reviewed at impact and then formally rendered; its rebound is only 5.14 mm, so naturalness remains open for human review. Other NEEDS_REVIEW scenes passed inexpensive Pilot gates but were not formally rendered. The baseball with sideways velocity left the fixed camera view and is FAIL; the two pillows failed mesh deformation gates. See [scene_manifest.csv](scene_manifest.csv) and [quality_report.md](quality_report.md) for exact reasons and metrics. PASS is an automated result and does not mean a human has certified natural motion.

| PASS scene | Camera 0 | Camera 1 | Camera 2 | Dataset |
|---|---|---|---|---|
| Baseball drop | [MP4](scenes/baseball_drop/render_full/cam_0.mp4) | [MP4](scenes/baseball_drop/render_full/cam_1.mp4) | [MP4](scenes/baseball_drop/render_full/cam_2.mp4) | [Files](scenes/baseball_drop/) |
| Deflated football drop | [MP4](scenes/football_deflated_2s/render_full/cam_0.mp4) | [MP4](scenes/football_deflated_2s/render_full/cam_1.mp4) | [MP4](scenes/football_deflated_2s/render_full/cam_2.mp4) | [Files](scenes/football_deflated_2s/) |
| Low-drop rubber duck | [MP4](scenes/duck_low_drop/render_full/cam_0.mp4) | [MP4](scenes/duck_low_drop/render_full/cam_1.mp4) | [MP4](scenes/duck_low_drop/render_full/cam_2.mp4) | [Files](scenes/duck_low_drop/) |

## Quick index

- `scene_manifest.csv`: every candidate, physical parameters, current quality status, and output path.
- `quality_report.md`: fixed numeric gates and unresolved visual issues.
- `contact_sheet.png`: representative RGB frames, including pilots.
- `comparison.mp4`: three-view compilation of scenes with completed formal QA.
- `scenes/<name>/config.json`: geometry, units, E, ν, density, gravity, ground, timestep, initial velocity, and solver internal E conversion.
- `scenes/<name>/trajectory.npz`: full particle positions and velocities at each saved frame, tracked surface vertex count, faces, particle volumes, timestamps.
- `scenes/<name>/physics.json`: particle motion, contact, penetration, deformation, stability metrics and gates.
- `scenes/<name>/reverse_ad_gate.json`: short-path existing Warp temporal reverse AD execution and forward parity. This does not certify long-rollout gradient accuracy.
- `scenes/<name>/render_pilot/`: 256² five-frame three-view RGB, masks and depth EXR.
- `scenes/<name>/render_full/`: only quality-selected formal 480², 24 FPS RGB, masks and depth EXR; camera metadata and MP4s.
- `scenes/<name>/mesh_visible_flow/`: visible-surface geometric flow, when generated. This is **not** WonderPlay Gaussian rasterized flow.
- `assets/<name>/source.json`: Poly Haven source URL, CC0 declaration, exact download URLs and MD5 checksums. Original `.blend` and PBR textures are retained.

## Reproduce

Use only a genuinely idle GPU. `CUDA_VISIBLE_DEVICES=8` was selected after checking this machine; do not assume GPU 8 is free on another run. The PhysFlow environment is `/disk2/ccchen/miniconda3/envs/physflow/bin/python`.

```bash
SYNTH_GPU=8 /disk2/ccchen/miniconda3/envs/physflow/bin/python scripts/run_synthetic_batch.py
```

The runner uses one GPU, processes scenes sequentially, records per-scene logs, sets a 600-second simulation/render limit and 300-second AD gate limit, and skips completed trajectories/renders. The simulator writes an atomic particle-state checkpoint after every saved frame; rerunning the command resumes a partial scene. A failed candidate remains in the manifest.

For parameter comparisons, pass `--initial-particles scenes/<GT>/trajectory.npz` to `scripts/synthetic_mpm_pilot.py`. This reuses the exact saved initial particle positions, volumes, surface mapping, and mesh topology. New particleization seeds NumPy at 0, but the saved initial state is the authoritative GT/Candidate input.

To rerender one selected scene without changing its simulation:

```bash
CUDA_VISIBLE_DEVICES=8 timeout 900s .tools/blender/blender-4.5.3-linux-x64/blender -b --factory-startup --python scripts/render_synthetic_mpm.py -- \
  outputs/synthetic_physics_batch_20261121/assets/baseball_01/baseball_01_2k.blend \
  outputs/synthetic_physics_batch_20261121/scenes/baseball_drop/trajectory.npz \
  outputs/synthetic_physics_batch_20261121/scenes/baseball_drop/render_full full baseball_01
/disk2/ccchen/miniconda3/envs/physflow/bin/python scripts/assess_synthetic_scene.py \
  outputs/synthetic_physics_batch_20261121/scenes/baseball_drop --stage full
```

Refresh the overview after QA:

```bash
/disk2/ccchen/miniconda3/envs/physflow/bin/python scripts/finalize_synthetic_batch.py
```

## Physics interpretation and limits

`E_Pa = E_solver_internal × 10^7` in this Warp implementation. The source materials are visual references. A stitched baseball, a pressurized soccer ball, and a toy duck are **modeled here as homogeneous elastic MPM solids**; their real material construction and internal air pressure are not modeled. GT parameters are therefore exact truth for this synthetic solver, not measured physical properties of the real products.

The original PhysFlow `RepairedTemporalRollout` supports the selected elastic material and static surface collider. Each quality candidate executes a short forward parity and reverse AD gate. Complete end-to-end WonderPlay training, long-rollout AD accuracy, and matched Gaussian initialization are separate downstream checks and are not claimed here. Depth EXR comes from Blender. Object masks use Blender's object-index pass. Mesh visible-surface flow is a geometric correspondence with explicit occlusion and differs from both photometric optical flow and the Gaussian rasterizer's flow.

All Poly Haven assets are [CC0](https://polyhaven.com/license). Asset pages: [Rubber Duck Toy](https://polyhaven.com/a/rubber_duck_toy), [Baseball 01](https://polyhaven.com/a/baseball_01), [Football](https://polyhaven.com/a/football), [Dirty Football](https://polyhaven.com/a/dirty_football), and [Throw Pillows 01](https://polyhaven.com/a/throw_pillows_01).
