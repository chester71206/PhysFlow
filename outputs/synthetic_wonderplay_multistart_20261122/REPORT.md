# Exact-flow WonderPlay multi-start report

The target is the unchanged WonderPlay Gaussian rasterizer applied to saved native Warp MPM GT particle motion. Blender object masks define foreground. RAFT is unused. Training uses three views, 128×128 images, and frames 0, 2, 4, 6, 8 (0–0.333 s); 100 Adam updates per start, with learning rates halved at updates 50 and 75. GT is E=15,000 Pa, ν=0.32 for baseball and E=12,000 Pa, ν=0.32 for football.

| Scene | Start | Updates | Initial loss | Final evaluated loss | Final E (Pa) | Final ν | E relative error | ν absolute error | Particle EPE (mm) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseball | highE_lowNu | 100 | 0.54609 | 4.8595e-06 | 15002 | 0.31973 | 0.0001 | 0.0003 | 0.004 |
| baseball | lowE_highNu | 100 | 1.1284 | 1.5756e-05 | 15019 | 0.31948 | 0.0013 | 0.0005 | 0.011 |
| football | highE_lowNu | 100 | 1.3629 | 1.0887e-05 | 11990 | 0.32103 | 0.0009 | 0.0010 | 0.022 |
| football | lowE_highNu | 100 | 3.0894 | 1.3051e-05 | 11994 | 0.32151 | 0.0005 | 0.0015 | 0.030 |

Loss minima along these optimizer paths are not proof of the global or continuous minimum. Parameter recovery requires both E and ν errors to decrease, not just loss.

The self-consistent Gaussian target checks the original Warp rollout and WonderPlay loss/gradient. Independent Blender visible-mesh flow is a separate observation target and may differ because of representation and visibility.

## Independent Blender visible-surface flow at GT

| Scene | Gaussian target GT loss | Blender target GT loss | Blender target ∂loss/∂log E | Blender target ∂loss/∂ν × 0.05 |
|---|---:|---:|---:|---:|
| baseball | 2.329e-10 | 0.4959 | -0.5809 | -0.0963 |
| football | 1.739e-11 | 0.3658 | 0.1783 | -0.0006576 |

The Blender target differs from the Gaussian target even for identical GT MPM motion. Thus near-zero Gaussian-target GT loss does not establish that GT minimizes the independent Blender observation loss.


At GT, the independent Blender-flow gradients can be checked against centered finite differences:

| Scene | E probe ± fraction | E AD / FD (solver units) | ν AD / FD | Gate |
|---|---:|---:|---:|---|
| baseball | 5% | -387.3 / -419.4 | -1.926 / -1.891 | PASS |
| football | 1% | 148.6 / 153.5 | -0.01315 / -0.01035 | PASS |

## Local parameter sensitivity at GT

Each point uses the same initial particles, timestep, gravity, collider, and cameras. E changes by ±10%; ν changes by ±0.05. Values are the mean excess loss over GT for the two signs.

| Scene | Target | E ±10% excess loss | ν ±0.05 excess loss | E / ν ratio |
|---|---|---:|---:|---:|
| baseball | Gaussian | 0.10501 | 0.19148 | 0.548 |
| baseball | Blender mesh | 0.10904 | 0.17279 | 0.631 |
| football | Gaussian | 0.22197 | 0.024821 | 8.94 |
| football | Blender mesh | 0.21008 | 0.020152 | 10.4 |

These step sizes differ in relative magnitude; the ratio is descriptive for these stated perturbations, not a dimensionless material identifiability theorem.


For football, the ν perturbation produces substantially less flow-loss change than the E perturbation in this early impact window. Baseball shows the opposite ordering for the same stated perturbations. The scenes differ in geometry, density, E, and contact motion, so this pair alone cannot isolate which factor causes the sensitivity difference.

## Interpretation and next work

All four starts recover both parameters closely under the Gaussian self-consistent target, and the off-GT reverse-AD derivatives passed centered-difference checks. This demonstrates a working differentiable path for these two saved native-MPM scenes. It does not prove a global minimum or recovery from independently rendered video.

The independent Blender visible-mesh target has a material GT loss and nonzero GT gradient. Its residual maps show both silhouette and interior errors during impact. Before adding further scenes or claiming video-based recovery, align Gaussian appearance/visibility with the Blender surface and recheck whether GT becomes stationary under the external target. Automated PASS for source videos does not mean human-certified visual naturalness.
