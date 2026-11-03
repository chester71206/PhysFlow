# Release validation — 2026-11-03

This release was prepared in an isolated worktree. The existing experiment
workspace and its running GPU 3 comparison were not edited.

Completed checks:

- Clean clone of the release branch, with no `outputs/`, raw PAC-NeRF data,
  external PGSR checkout or preprocessed models initially present.
- Python syntax and shell syntax; staged Git whitespace check. Frozen kernel
  bytes are retained exactly, including their inherited whitespace.
- All ten immutable MPM source hashes match the existing validated backend.
- Import checks from the clean clone, using the existing Python 3.9 environment.
  Torch 2.2.0+cu118, torchvision 0.17.0+cu118, Warp 0.10.1, Taichi 1.5.0.
- Observation-only CPU float64 AD/central-FD gradcheck: 25 points, two pairs.
- Independent asset archive: 449 files across nine scenes, 1,096,124,236 bytes
  compressed, with archive checksum and per-file SHA256 manifest.
- Letter restored into the clean clone from that archive; all 54 file hashes
  match. Data preflight reproduces 16 frames, dt=0.00005, final step=12,500,
  using only committed compact metadata.
- All nine scenes prepare successfully for the serial comparison: 9 original
  jobs, 4 supported projected-flow jobs, 5 explicitly unsupported rows.
- Training and evaluation CLI help works from the clean clone.
- Both CUDA extensions rebuilt successfully from that clean clone, with CUDA
  11.8, Torch 2.2.0+cu118, MAX_JOBS=2 and sm_86 (the test GPU is an RTX 3090).
  No precompiled repo `.so` was used.
- Import audit repeated with the newly built wheels ahead of the existing
  site-packages. Both native extensions load from the extracted new wheels.
- Minimal GPU forward with those new wheels: four-point nearest-neighbor
  distances are finite/positive; the 32x32 native renderer returns five
  outputs, positive radii and finite non-background RGB (sum=2976.74609375).
  This smoke test does not claim correctness of native-renderer backward.
- The complete asset archive also passes `sha256sum -c`.

The import audit revealed that the existing machine uses a PGSR rasterizer
outside the original repo submodule. This release vendors that actual source
and fingerprints it, so a fresh installation builds the same implementation.
The imported installed wrapper was checked against the vendored wrapper.

Limits: this is a clean-checkout validation with an existing Python environment,
not a completed installation on a second physical machine. No new 50-epoch
training was launched as part of packaging. The existing full-training evidence
belongs to the unchanged validated MPM and projected-flow objective; it is not
a claim that every scene or material passes. Each training run performs its
own initial and final derivative gates.
