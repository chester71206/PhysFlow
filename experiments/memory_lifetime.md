# Independent memory-lifetime variant

Use `simulation_memory_safe.py` with the same model, physics config, frame count,
epochs and `--guidance-mode raft_cpu` as the original run. Use a NEW output prefix,
for example `outputs/phys_dreamer/alocasia_memory_safe_<timestamp>` or
`outputs/realworld/jam_memory_safe_<timestamp>`.

`simulation.py`, original configs, original runs and the separate time-alignment
experiment remain unchanged. This variant keeps baseline timing, loss weights,
learning rates, image resolution, Taichi's 8 GB allocation and the existing
truncated gradient approximation. It is not a time-alignment fix.

Changes:

- Build the fixed initial prompt without an autograd graph; it is only consumed
  by the detached previous-frame branch in the baseline guidance.
- Render saved videos and diagnostic optical flows under `torch.no_grad()`.
- After training backward/update completes, release per-batch loss/image graph
  references and reset the completed Warp tape before visualization.
- Clear RAFT's correlation pyramid after backward. The installed torchvision
  RAFT rebuilds this cache on every forward; retaining it can retain the preceding
  input graph even when external references have been deleted.
- Release visualization image/flow lists after logging.
- Refuse to overwrite nonempty training outputs; write `memory_experiment.json`
  and per-batch allocator measurements to `memory.jsonl`.

`--stop-after-batches 2` is for a bounded smoke test: keep `--n_epoch 10` to retain
the configured schedule, run two optimization batches including batch-0 video
export, then stop. It is not a resume checkpoint feature. Omit it for training.

Validation: the timed training, objective, backward and Warp propagation block
was checked against the baseline. A CPU RAFT repeated forward/backward test
confirmed correlation-cache clearing preserves outputs and input gradients.
The 1080p alocasia smoke-test log is `/tmp/alocasia_memory_smoke.log`; durable
measurements are under `outputs/diagnostics/alocasia_memory_smoke_ds1.0_ep10/`.

The smoke test covers the no-grad visualization and graph-release changes; the
additional correlation-cache clear was validated separately. It does not prove
bitwise equivalence across full training runs or guarantee every scene fits in
24 GB. GPU kernels can produce numerical differences between runs.

The FFmpeg 1080-to-1088 macroblock warning affects encoded video dimensions, not
the training input tensors or CUDA allocation failure.
