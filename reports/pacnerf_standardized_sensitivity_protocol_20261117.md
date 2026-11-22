# PAC-NeRF standardized material sensitivity protocol (2026-11-17)

## Formal metric audit

The source of truth is `scripts/evaluate_universal_reference.py`.

- The evaluator renders each selected PAC-NeRF frame from the time-aligned MPM rollout. It compares the rendered RGB image with `guidance.guidance[i]`, the supplied white-background RGBA GT image. If sizes differ, it resizes the target by bilinear interpolation to the rendered size. PSNR is computed immediately from these same rendered frames as `-10 log10(mean((clamp(render)-clamp(target))²))`, and the reported mean omits frame 0. The sensitivity pipeline runs this evaluator once per parameter point; it does not run another simulation for PSNR.
- Both EPE metrics use the *same* RAFT Large predictions: rendered frame pair versus rendered frame pair, and aligned GT frame pair versus GT frame pair. Both use the same 256×480 preprocessing (RGB from `[0,1]` to `[-1,1]`, bilinear resize; no crop) and pair indices. The flow error is the Euclidean norm of the two-channel difference, per pixel.
- The foreground mask is the **supplied GT RGBA alpha exactly equal to 255** at the source frame of each pair, resized by nearest-neighbor to 256×480. It is neither a rendered mask nor a GT/render union or intersection. `FG EPE` averages errors over these GT foreground pixels across all pairs. `Whole EPE` averages all 256×480 pixels across all pairs. Background flow is not zeroed; RAFT predicts it from the full frames. The evaluator now also records background EPE and foreground pixel fraction so the weighting can be checked directly.

## Bird sanity visualization

Four independent evaluator runs used Bird initial, GT-converted material, a previously measured low-FG-EPE sample, and a high-FG-EPE sample from the completed Bird 2D sweep. The saved tensors and rendered frames were used to draw the late pair 12 without another rollout. Each PNG contains GT RGB, rendered RGB, supplied GT alpha, a rendered dark-pixel silhouette **for diagnosis only**, the exact EPE mask, observed/rendered RAFT magnitudes, and the per-pixel EPE map.

The GT alpha tracks the GT bird silhouette in the displayed pair. The rendered bird moves to different positions for different material samples; that displacement is visible in the rendered frames and RAFT maps. On pair 12 the FG mask occupies 3.491% of pixels. Whole EPE exactly equals `0.0349121 × FG EPE + 0.965088 × background EPE` (within floating-point error). Thus the different FG/Whole trends in this example follow the region weighting, including nonzero background RAFT flow; no mask-index or frame-index discrepancy was observed. This does **not** prove the optical-flow estimator is unbiased or that every scene's masks are correct.

There is also a concrete ranking reversal among the saved Bird 2D samples. At `(E, nu)=(0.000749427, 0.197368)`, FG/Whole EPE are `8.571/7.305`; at `(0.0112328, 0.0684211)` they are `8.772/7.078`. The first has better FG EPE but worse Whole EPE. With the same 3.45997% foreground fraction, the implied background EPE is `7.260` versus `7.017`. The reversal is exactly explained by the background contribution. This is evidence about the evaluator's weighting, not a claim that background RAFT estimates are physically meaningful.

Example: [Bird GT pair 12 diagnostic](../outputs/pacnerf_standardized_sensitivity_20261117/bird_epe_sanity/GT/evaluation_v2/epe_sanity_pair12.png). The other three categories are sibling paths under `bird_epe_sanity/`.

## Sweep design

The queue plan is `outputs/pacnerf_standardized_sensitivity_20261117/plan.json`. It contains 23 one-parameter curves, seven samples each, deduplicated into 147 forward points. Each point runs the existing 20,000-point WonderPlay objective and the official independent evaluator. It uses SQLite claims, separate point directories, per-point logs, failure records, timeouts, resume, and one worker per allowed GPU. No optimizer or backward pass is used.

For Letter, Droplet, Torus, Bird, Playdoh, Cat, and Trophy, active material parameters are fixed at the existing converted GT reference. For Cream and Toothpaste, the physical-to-solver viscosity conversion is unverified, so **every** material parameter is fixed at the full initial reference; the plan marks the partial GT status. Parameters other than the one being swept remain exactly at the declared reference. E, yield stress, and viscosity use `[0.5, 0.7, 0.85, 1, 1.15, 1.3, 2] × reference`. Poisson ratio uses seven absolute offsets scaled to stay below 0.5. Friction angle uses absolute offsets around the reference 40°, spanning 20°–60°. Exact values, solver vectors, physical values, and GT conversion status are stored in the plan and point results.

For metric `M`, local normalized sensitivity is `abs(p0/M0 × (M(p+)−M(p−))/(p+−p−))`, using the 0.85×/1.15× or corresponding ±0.5 absolute-offset pair. The denominator uses `max(abs(M0), 1e-8)` to prevent division blow-up near zero. The report also records the signed local slope, relative sampled range, `1/(1+range)` flatness score, finite count, monotonicity, and stability. These are finite-sample 1D slice descriptors. Threshold-based classifications are exploratory, not proof of identifiability or a global optimum.

Near `nu=0.5`, a tiny absolute change in `nu` is a tiny *relative parameter* change, so the elasticity formula can make `S_nu` large even when the observed metric moves little over the chosen perturbation. We therefore record `standardized_local_effect = |M(p+)−M(p−)|/max(|M0|,1e-8)` and classify flatness/proxy gaps from the **relative sampled metric range**: flat if all four are below 1%; proxy gap if WP is below 2%, FG EPE is at least 2%, and the FG/WP range ratio is at least 5. This keeps the mathematical sensitivity and the practically observed change distinct. These cutoffs are exploratory and explicitly stored in the plan/summary.

## Commands

```bash
python scripts/pacnerf_standardized_sensitivity.py status --root outputs/pacnerf_standardized_sensitivity_20261117
python scripts/pacnerf_standardized_sensitivity.py render --root outputs/pacnerf_standardized_sensitivity_20261117
python scripts/pacnerf_standardized_sensitivity.py retry --root outputs/pacnerf_standardized_sensitivity_20261117
python scripts/pacnerf_standardized_sensitivity.py resume --root outputs/pacnerf_standardized_sensitivity_20261117 --gpus 3,4,5,6,7,8,9
```

The queue result `summary.json`, `sensitivity.csv`, `samples.csv`, `material_sensitivity.csv`, and PNG heatmaps are generated by `render`. A 2D coupling scan is not part of the automatic queue; it should be selected only after the completed 1D findings justify a specific pair.
