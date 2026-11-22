# Far-start WonderPlay experiment

Same saved Warp MPM scenes, RepairedTemporalRollout and WonderPlay Gaussian exact-flow objective as the prior multi-start experiment. Three views, 128×128, frames 0/2/4/6/8. No RAFT or finite-difference optimization. Centered differences only check the initial AD gradient. Each update saves an optimizer checkpoint and per-update CSV; each run has a 900 s external timeout and can resume.

The starts are E=2×GT, ν=.16 and E=.5×GT, ν=.44. Adam uses initial E learning rate .04×GT and ν learning rate .008, halved at updates 70 and 100; initial budget is 120 updates, with continuation from checkpoint where needed. Parameter bounds are E∈[.2,3]×GT, ν∈[.05,.48].

| Scene | Start | Updates | Status | Initial loss | Final loss | E relative error | ν absolute error | Initial AD/FD E/ν relative error | Bound hit |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| baseball | highE_lowNu | 120 | GT_RECOVERED | 2.2792 | 1.9865e-05 | 0.0016018 | 0.0005151 | 0.00034755 / 0.00098521 (True) | E=False, ν=False |
| baseball | lowE_highNu | 120 | GT_RECOVERED | 5.0244 | 2.8072e-06 | 0.0006374 | 0.00013205 | 0.021237 / 0.0022222 (True) | E=False, ν=False |
| football | highE_lowNu | 120 | GT_RECOVERED | 5.5557 | 9.7733e-06 | 0.00070698 | 8.9295e-05 | 0.0041877 / 0.015298 (True) | E=False, ν=False |
| football | lowE_highNu | 270 | GT_RECOVERED | 10.404 | 0.00043798 | 0.0038108 | 0.0086495 | 0.0039531 / 0.01015 (True) | E=False, ν=True |

Football lowE/highNu at the original 120-update checkpoint: loss 0.00380637, E error 0.0127, ν error 0.02398; ν hit its upper bound earlier in the path. The saved `final_120.json` and `history_120.csv` preserve this outcome before continuation.
At 240 updates this same path had loss 0.0007078, E error 0.004868, ν error 0.01092. The final continuation reached 270 updates. Its ν upper-bound contact occurred at updates 6–17 and it subsequently moved into the interior; the declining loss and nonzero gradient before continuation do not support calling the 120-step point a local minimum.

GT_RECOVERED requires evaluated final loss ≤10⁻³, E relative error ≤2%, ν absolute error ≤.01, and at least 120 updates. LOW_LOSS_OFF_GT is an operational label for loss ≤.05 with off-GT parameters; it does not establish a mathematical local minimum. BOUNDARY_LIMITED means the final parameters sit at a bound; a transient bound hit is reported in its own column. Incomplete runs and failed gradients must not be counted as convergence.

## Independent Blender flow bias

The independent Blender visible-mesh flow remains a separate observation target. The prior GT evaluation found loss 0.4959 for baseball and 0.3658 for football, with nonzero GT gradients. Gaussian-target recovery therefore does not establish recovery from Blender video.

The following comparison uses saved Gaussian and Blender GT flow targets on their common valid pixels. A four-pixel erosion of the common mask defines the interior; the remainder is the boundary band. These are residual diagnostics, not the WonderPlay training loss.

| Scene | Interior mean EPE (px) | Boundary mean EPE (px) | Boundary share of squared residual |
|---|---:|---:|---:|
| baseball | 0.122 | 0.972 | 99.1% |
| football | 0.424 | 1.085 | 83.4% |

The boundary band dominates the saved squared flow discrepancy, especially for baseball; football also retains a measurable interior discrepancy. This points to Gaussian support, opacity and visibility as alignment targets, but does not yet prove which renderer difference causes the nonzero GT gradient.
