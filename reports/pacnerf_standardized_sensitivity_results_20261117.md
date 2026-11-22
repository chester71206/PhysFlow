# PAC-NeRF nine-scene standardized material sensitivity results

Date: 2026-11-17. The exact plan and results are under `outputs/pacnerf_standardized_sensitivity_20261117/`.

## Scope and validity

Twenty-three seven-point one-parameter slices were evaluated, deduplicated to 147 forward tasks. 146 tasks completed with finite WonderPlay loss, independent FG EPE, Whole EPE, and PSNR. Torus at `E=0.3705898855` (2× the converted GT E) produced `FloatingPointError: Nonfinite state at frame 13` in the independent evaluator; its forward loss was finite. That point is excluded from metric ranges, and Torus E is marked unstable. Its local central secant around the reference is finite and included in material medians with an explicit stable count of one of two Elastic/E scenes.

The 1D slices fix all other solver parameters at the **converted GT active material set** for Letter, Droplet, Torus, Bird, Playdoh, Cat, and Trophy. Cream and Toothpaste instead use the **full initial set**, because solver-to-GT viscosity conversion has not been certified. Thus comparisons are conditional on different declared reference points; the result is not a claim of parameter identifiability everywhere. The exact seven values and conversions are in `plan.json`.

The following percentages are the full seven-point sampled `(max M − min M)/|M(reference)| × 100`, a practical complement to the mathematically requested local `S_p(M)=|(p/M)dM/dp|`. PSNR percentages refer to the *dB number* and should be interpreted together with the raw PSNR values in `samples.csv`. The complete requested sensitivity table and material medians are in `analysis.md`, `sensitivity.csv`, and `material_sensitivity.csv`.

## Direct answers

1. **WonderPlay's strongest effects depend on material and reference.** Newtonian Letter/Droplet E changes WP by 28.78%/30.01%; Cat/Playdoh yield stress by 32.74%/11.42%; Cream E and ν by 24.32%/27.31% at the full initial reference; Trophy friction angle by 11.13%. Therefore E dominates Letter/Droplet, while yield dominates Plasticine, and Cream has a strong ν effect. E alone does not dominate all nine scenes.
2. **Parameters WP sees weakly over these ranges:** Letter/Droplet/Bird/Torus ν (<0.25% WP range); Playdoh/Cat E (1.33%/0.46%); Playdoh/Cat ν (0.20%/0.07%); Cream/Toothpaste viscosity (0.89%/0.21%); Toothpaste yield (0.10%). The near-zero WP response is conditional on the reference, seven values, fixed camera and frame sequence.
3. **FG EPE sometimes sees effects WP nearly loses.** Letter ν gives 0.06% WP versus 2.12% FG EPE; Bird E gives 1.67% WP versus 11.29% FG EPE. They meet the declared exploratory proxy-gap rule. Bird ν also shows 0.04% versus 1.01%, a smaller but measurable difference. This is objective/observation sensitivity separation, not evidence that a 1D minimum is global.
4. **FG/Whole differences here follow the formal evaluator's region weighting.** Both use identical RAFT flow tensors and differ only in averaging pixels. FG uses supplied GT RGBA alpha exactly 255, nearest-resized to RAFT 256×480; Whole averages the full flow grid. Background RAFT flow is not zeroed. In Bird the foreground occupies ~3.46% of flow pixels. Two stored Bird 2D points reverse ranking: FG `8.571 < 8.772` but Whole `7.305 > 7.078`; the implied background EPE is `7.260 > 7.017`. The saved RGB/mask/flow/EPE panels show the mask follows the GT bird. No evaluator mask or frame-index defect was found in this diagnostic; other scenes were not individually visual-audited.
5. **PSNR is not universally E-dominated.** E has the largest PSNR range in Letter (2.34%), Droplet (1.39%), and Bird (0.84%) among their material axes. In Plasticine, yield stress dominates PSNR: Playdoh 11.66% and Cat 0.95%, versus E 0.22% and 0.09%. Trophy friction angle changes PSNR by 3.79%. Cream E and ν both matter (1.39% and 0.94%). The PSNR is computed from the same rendered frames as the independent evaluator, with no extra PSNR-only simulation.
6. **All four insensitive means low conditional observation sensitivity, not proof of a globally ill-posed inverse problem.** Playdoh ν has WP/FG/Whole/PSNR relative ranges `0.20/0.10/0.02/0.11%`. Toothpaste viscosity has `0.21/0.42/0.94/0.03%`. These are below the exploratory 1% flatness cutoff across the tested 1D slice. Longer motion, another camera, joint parameter changes, or better independent supervision could still reveal an effect. Toothpaste viscosity uses an initial reference because GT conversion is unverified.
7. **2D coupling priority:** inspect the **existing** Letter E×ν and Bird E×ν grids first, because those scenes have the clearest WP-versus-FG sensitivity gaps. No new 2D task is justified automatically from 1D data alone. If those grids fail to explain the gap and joint training moves both axes, a new 10×10 or 12×12 Cream E×ν grid is the next candidate; both parameters show large effects at Cream's initial reference. Do not treat the weak Playdoh ν or Toothpaste viscosity slices alone as proof of compensation.

## Reproduction and artifacts

- `scripts/pacnerf_standardized_sensitivity.py` builds and runs the SQLite queue, computes sensitivity, and renders plots.
- `scripts/evaluate_universal_reference.py` now optionally saves exact pixel sanity panels and records foreground fraction/background EPE; metric definitions are unchanged.
- `scripts/render_saved_epe_sanity.py` draws another pair from already saved RAFT tensors and rendered frames, without rerunning physics.
- `scripts/test_pacnerf_standardized_sensitivity.py` checks all 23 curves, one-parameter routing, reference provenance, and the sensitivity formula.
- `outputs/pacnerf_standardized_sensitivity_20261117/analysis.md`: full requested scene/parameter sensitivity table and material medians.
- `.../scene_sensitivity_heatmap.png`, `.../material_sensitivity_heatmap.png`: normalized local sensitivity.
- `.../scene_relative_range_heatmap.png`, `.../material_relative_range_heatmap.png`: practical seven-point metric changes.
- `.../proxy_gap_normalized_curves.png`: strongest FG-versus-WP gap.
- `.../bird_epe_sanity/{initial,GT,better_fg,worse_fg}/evaluation_v2/epe_sanity_pair12.png`: exact-region visual audit.

The experiment did not modify an optimizer or a formal training checkpoint. It did not start a new 2D grid.
