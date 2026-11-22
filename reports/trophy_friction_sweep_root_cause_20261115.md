# Trophy friction-angle sweep failure: root cause

## Finding

The old one-parameter Trophy sweep did **not** change friction angle. It passed a
one-element vector (`[sampled_angle]`) to `MaterialTemporalRollout.model`, whose
fixed positional contract is `(E, nu, yield_stress, plastic_viscosity,
friction_angle)`. The value labeled 40° therefore became `E=40`, while
`friction_angle` stayed at its initial 10°. The scene's intended E is 0.01, so
the mistaken 40° probe increased solver E by a factor of 4000. Even the sole
finite 0.1° sample actually used `E=0.1`, friction angle 10°; its loss and image
metrics cannot be paired. Pass B independently built a checkpoint with
`E=0.01`, friction angle 0.1°, so the old row combines a WonderPlay loss from
one material with EPE/PSNR from a different material.

This was a parameter-vector bug in `scripts/pacnerf_material_landscape.py`, not
a degrees/radians mismatch. The configured and GT friction angles are in degrees.
`sand_return_mapping` converts `angle / 180 * pi` inside the constitutive law.
For the intended 10° and 40°, the denominators in the angle expression are
nonzero; the elastic constants have positive E and nu=0.1.

## Minimal probes, same Trophy scene and dt=0.0002

| Input requested | Actual model parameters | First nonfinite | Result |
| --- | --- | --- | --- |
| Legacy "10°" | E=10, friction angle=10° | stress, step 10 (t=0.002 s) | Stopped after 0.42 s of rollout |
| Legacy "40°" | E=40, friction angle=10° | stress, step 9 (t=0.0018 s); particle position also nonfinite in step 9 | Stopped after 1.41 s of rollout |
| Correct 10° | E=0.01, friction angle=10° | none | 2708/2708 steps, 69.75 s |
| Correct 40° | E=0.01, friction angle=40° | none | 2708/2708 steps, 69.78 s |

The corrected formal WonderPlay forward objective was also evaluated at these
two angles in a fresh output directory: loss `11.63649` at 10° and `9.87578`
at 40°. Both were finite. These two values verify the corrected parameter
transmission; they are not a reconstructed friction-angle landscape.

For the legacy "40°" case, by step 8 the maximum particle speed was
1,182,054 and the maximum absolute particle position was 240 in the simulation
coordinates. At step 9, one particle had `det(F)=-0.265`. Warp SVD returned
signed singular values approximately `(0.692, 0.630, -0.608)`. The
Drucker–Prager stress takes `log(sig[i])` and divides by `sig[i]`, so the
negative third value made `particle_stress` nonfinite. The G2P phase then made
`particle_x` nonfinite in the **same** step. This is the observed first-failure
chain. The constitutive function has an inversion/zero-singular-value hazard,
but the tested 10° and 40° parameterization never reached it.

The original sweep checked positions only *after* `engine.forward` completed
all 2708 substeps. Thus it kept simulating a broken state long after step 9/10,
then reported the position error at the end. This explains why it could not
fail fast. A bounded 20-step timing probe confirmed that the corrupt rollout
also became much slower: steps 2–6 had median 0.040 s/step, step 9 took 0.514 s,
and steps 16–20 had median 1.52 s/step. This explains the order of magnitude
of the 14–18 minute failures when the old loop kept running. The probe did not
isolate which GPU kernel accounted for the slowdown. A phase-level probe now
stops at the first nonfinite value.

## Classification and changes

- **Parameter transmission:** confirmed root cause. The sweep now passes the
  full five-scalar vector in the rollout's required order.
- **Units:** degrees throughout the scene and metadata; radians used only
  inside the `sin` formula. No unit conversion bug found.
- **Constitutive model:** `log(sig)`/`1/sig` is undefined for a nonpositive
  signed singular value. This is the immediate NaN mechanism after the wrong
  stiffness inverted F. No constitutive equation was changed.
- **Timestep:** dt=0.0002 in both the old and diagnostic runs. The wrong E
  makes this dt numerically unstable; with E=0.01 both angles completed.
- **Scene conditions:** current formal and baseline Trophy configs agree on
  E, nu, angle, dt, density, gravity and damping. The diagnostic reused the
  formal scene setup and initial state.

The old Trophy sweep output is preserved for audit but **invalid as an angle
landscape**. The Pass A and Pass B scripts now reject reuse of that legacy
directory. A corrected sweep must use a new output directory. No full sweep
was started as part of this investigation. The corrected Trophy Pass A also
checks stress and particle state after each substep, so future unstable samples
stop on the first nonfinite state and record its substep rather than running
the remaining rollout.

## Reproduction data

- Correct 10°/40° finite traces:
  `outputs/trophy_friction_diagnostic_20261115/correct_10_40/diagnostic.json`
- Legacy-mapping stress and position trace:
  `outputs/trophy_friction_diagnostic_20261115/legacy_40_position/diagnostic.json`
- Legacy-mapping 10° trace:
  `outputs/trophy_friction_diagnostic_20261115/legacy_10_v2/diagnostic.json`
- Probe implementation: `scripts/diagnose_trophy_friction.py`
- Bounded slowdown profile:
  `outputs/trophy_friction_diagnostic_20261115/legacy_40_profile20/diagnostic.json`
- Corrected two-point WonderPlay output:
  `outputs/trophy_friction_diagnostic_20261115/corrected_sweep_gt40/results.jsonl`

These probes are forward-only and contain no optimizer or backward pass.
