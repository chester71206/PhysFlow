# Synthetic physics batch quality report

PASS means automated physics, data, and rendering gates passed. It does not certify fully natural motion by human inspection.

| Scene | Status | E (Pa) | ν | Density (kg/m³) | Max penetration (mm) | Max relative deformation (mm) | Rebound gap (mm) | Max edge strain | Reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseball_drop | PASS | 15000 | 0.32 | 950 | 3.00 | 16.09 | 21.73 | 0.259 | Automated physics/data/render gates passed; human visual review still advised |
| pillow01_drop | FAIL | 18000 | 0.32 | 220 | 0.18 | 550.23 | 2.99 | 3.637 | no_severe_mesh_strain,rebound,no_face_collapse |
| pillow02_drop | FAIL | 18000 | 0.32 | 220 | 0.81 | 467.71 | 7.91 | 10.209 | no_severe_mesh_strain,no_face_collapse |
| football_deflated_drop | NEEDS_REVIEW | 12000 | 0.32 | 850 | 2.48 | 75.67 | 25.85 | 0.637 | 1-second Pilot superseded by completed 2-second PASS version |
| football_deflated_2s | PASS | 12000 | 0.32 | 850 | 2.48 | 75.69 | 25.91 | 0.637 | Automated physics/data/render gates passed; human visual review still advised |
| football_inflated_drop | NEEDS_REVIEW | 24000 | 0.32 | 850 | 2.14 | 65.56 | 53.03 | 0.204 | Pilot gates pass; formal render deferred because inflated shell pressure is absent from homogeneous-solid model |
| dirty_football_drop | NEEDS_REVIEW | 23000 | 0.32 | 850 | 3.56 | 95.72 | 40.47 | 0.297 | Pilot gates pass; visually similar football drop deferred to prioritize distinct motion |
| baseball_oblique_drop | FAIL | 15000 | 0.32 | 950 | 2.86 | 142.35 | 20.20 | 0.275 | all_render_files,opaque_background |
| duck_low_drop | PASS | 50000 | 0.32 | 950 | 0.00 | 134.64 | 5.14 | 0.566 | Automated physics/data/render gates passed; human visual review still advised |

## Fixed numeric gates

- All particle positions and velocities finite.
- Maximum speed < 8 m/s and particle displacement < 0.8 m.
- Maximum ground penetration ≤ one grid cell (1/64 m = 15.625 mm).
- Nonrigid relative displacement ≥ 5 mm and maximum tracked mesh edge strain < 1.5.
- Tracked mesh triangle area remains between 0.25× and 2.5× its initial area.
- Precontact center-of-mass ballistic error ≤ 3 mm.
- Postcontact bottom separation ≥ 5 mm and upward center-of-mass speed ≥ 0.1 m/s.
- Three fixed views, each frame 480×480 for formal output; masks cover 1–70% of frame with ≥1.5% border.
- Exact asset-to-MPM surface vertex count, matching timestamps, reverse AD short-path finite gradients and forward parity.
- Complete replay uses the saved initial particles and differs by ≤ 0.01 mm at every saved particle position.
- Geometric visible-surface flow has Blender-mask IoU ≥ 0.90 in every sampled pair and mean valid foreground ≥ 0.80.

## Remaining human review

- Inspect full videos for plausible deformation, texture sliding, shadow behavior, and impact/rebound timing.
- Homogeneous elastic MPM is an approximation for stitched or inflated real-world assets.
- The first football replay re-voxelized the source mesh and produced a different interior particle count. GT/Candidate replays now load the saved initial positions and particle volumes, which is the authoritative initial state; the failed replay remains archived.
- Blender visible-surface geometric flow differs from WonderPlay Gaussian rasterized flow.
- Full-length reverse AD gradient quality and end-to-end PhysFlow training are not established by the short-path gate.
- Poly Haven American Football was rejected at geometry preflight (28 nonmanifold edges), so it was not counted as a Pilot.
- A preliminary rubber-duck high-drop run exceeded the mesh strain gate. The lower-drop candidate passes automated full QA; its small rebound and perceived naturalness still need human review.
