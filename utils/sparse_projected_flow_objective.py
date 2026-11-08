"""Projected-flow objective that excludes pairs with no initial foreground match.

Each accepted pair uses FrozenVisibilityProjectedFlow itself to construct its
mask, so its loss term is identical. The accepted pair set is fixed before
optimization and must be reported with the run.
"""
import torch
import torch.nn.functional as F

from utils.projected_flow_objective import FrozenVisibilityProjectedFlow, project


class SparseFrozenVisibilityProjectedFlow:
    def __init__(self, reference_positions, cameras, observed_flow, observed_images,
                 opacity, max_points=20000, min_pairs=3):
        self.cameras = cameras
        self.flow = observed_flow.detach()
        self.height, self.width = observed_flow.shape[-2:]
        self.active_pairs = []
        self.skipped_pairs = []
        self.masks = []
        self.ids = None
        for t in range(len(observed_flow)):
            try:
                pair = FrozenVisibilityProjectedFlow(reference_positions[t:t+2],
                    cameras[t:t+2], observed_flow[t:t+1], observed_images[t:t+2],
                    opacity, max_points=max_points)
            except ValueError as exc:
                if 'Insufficient fixed visible foreground points' not in str(exc):
                    raise
                self.skipped_pairs.append(t)
                continue
            if self.ids is not None and not torch.equal(self.ids, pair.ids):
                raise RuntimeError('Projected-flow pair selection changed particle ids')
            self.ids = pair.ids
            self.active_pairs.append(t)
            self.masks.append(pair.masks[0])
        if len(self.active_pairs) < min_pairs or len(self.active_pairs) * 10 < len(observed_flow) * 3:
            raise ValueError(f'Only {len(self.active_pairs)}/{len(observed_flow)} flow pairs have enough visible foreground points')
        self.counts = [int(mask.sum()) for mask in self.masks]

    def __call__(self, positions):
        terms = []
        scale = positions.new_tensor([self.width / 2, self.height / 2])
        for t, mask in zip(self.active_pairs, self.masks):
            source, _ = project(positions[t, self.ids], self.cameras[t])
            target, _ = project(positions[t+1, self.ids], self.cameras[t+1])
            predicted = (target - source) * scale
            observed = F.grid_sample(self.flow[t:t+1].to(positions),
                source.reshape(1, 1, -1, 2), align_corners=False,
                padding_mode='zeros')[0, :, 0].T
            terms.append(.5 * (predicted[mask] - observed[mask]).square().mean())
        return torch.stack(terms).mean()
