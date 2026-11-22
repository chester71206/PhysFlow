# Order-of-magnitude E initialization test

This is the experiment requested after the earlier 0.5×/2× starts proved too close to GT. It tests **E/GT = 10, 0.1, 100, 0.01** for both `baseball_drop` and `football_deflated_2s`, with ν initialized at its GT value .32. Both E and ν are optimized after initialization.

The existing Warp MPM, `RepairedTemporalRollout` reverse AD, WonderPlay Gaussian self-consistent exact-flow target and loss, three views, frames 0/2/4/6/8 and 128×128 resolution are retained. E is optimized as `log(E/GT)` so an Adam step has comparable meaning at 0.01× and 100×. Bounds are 0.001–1000×GT. The original linear-E trainer's 0.2–3× bound cannot answer this question. ν remains bounded to .05–.48. This change in parameterization is recorded and must be considered when comparing with the previous Adam paths.

`scripts/run_synthetic_wonderplay_order_magnitudes.py` waits for a genuinely idle GPU, requires three idle checks, and then runs a forward probe, initial AD-versus-centered-difference gradient check, and checkpointed training for each start. It never interrupts a GPU user. The forward probe and gradient gate have per-run timeouts; training has an 1800-second timeout and can resume. See `runner_state.json`, per-start logs, `history.csv`, `optimizer.pt`, and the generated `REPORT.md` as trials complete. A high-E start that is nonfinite under the fixed solver timestep is recorded as a numerical validity failure, not silently clamped or counted as an optimizer minimum.

If you have confirmed a specific physical GPU is available, run `python scripts/run_synthetic_wonderplay_order_magnitudes.py --gpu N` with its `nvidia-smi` index. This bypasses the runner's conservative idle rule. `CUDA_VISIBLE_DEVICES=N` alone does not select the runner's GPU because the runner chooses and sets that variable for each child process. The terminal prints waiting and per-trial status; detailed updates remain in the per-trial logs.

At launch, every GPU was occupied by other processes; `availability_at_launch.json` records the snapshot. The detached runner has a 12-hour idle-GPU wait limit, and its outer process has a 13-hour timeout. No 10×/100× result should be inferred until a trial's probe and final files exist.

Recovery is assessed by final evaluated Gaussian target loss ≤10⁻³, E relative error ≤2%, and ν absolute error ≤.01. This test does not establish recovery from independent Blender flow; that target has a nonzero loss and gradient at GT.
