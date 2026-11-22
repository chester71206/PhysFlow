# Independent Blender visible-flow target bias

The saved native Warp MPM GT particles, three Blender cameras and frames 0/2/4/6/8 (0–0.333 s) are the same as in the Gaussian self-consistency experiments. The target here is composed Blender visible-mesh geometric flow. Candidate motion uses `RepairedTemporalRollout` reverse AD and the unchanged `WonderPlayFlowObjective`; RAFT and finite-difference optimization are absent. Each run starts at GT, saves every update and an optimizer checkpoint, and has a 900 s external timeout. Adam ran 40 updates with initial E step 0.005×GT and ν step .001, halved after updates 25 and 35.

| Scene | GT loss | GT ∂L/∂log E | GT ∂L/∂ν × .05 | Loss after 40 | E after 40 | ν after 40 |
|---|---:|---:|---:|---:|---:|---:|
| Baseball | 0.495940 | −0.580883 | −0.096296 | 0.482057 | 15,179.8 Pa | .331415 |
| Football | 0.365830 | +0.178292 | −0.000658 | 0.365024 | 11,892.2 Pa | .331759 |

The GT gradients are nonzero, and both short trajectories lower the independent target loss while moving away from GT. Baseball's decrease is 0.01388; football's is only 0.000806, so the latter does not support a precise ν optimum. The previous centered-difference gates at GT passed for both scenes; see `outputs/synthetic_wonderplay_multistart_20261122/REPORT.md`.

Saved Gaussian-versus-Blender target residuals on common valid pixels place about 99% of baseball's squared difference and 83% of football's in a four-pixel boundary band; football also has interior residual. See `outputs/synthetic_wonderplay_farstart_20261122/blender_bias_residual.json`. This motivates matching Gaussian support, opacity and visibility to the Blender surface before claiming material recovery from externally rendered video. The short optimizer trajectories do not identify a global or continuous minimum.

Per-scene `gate.json`, `gt_gradient.json`, `exact_gt_targets.pt`, `gt_start/history.csv`, `gt_start/final.json`, `gt_start/optimizer.pt` and plots are retained under `baseball/` and `football/`.
