"""Optical-flow supervision of projected material trajectories.

This is explicitly a different loss from RAFT(rendered RGB). RAFT supplies a
fixed observed 2D field; prediction is projected physical displacement. The
point selection is frozen at the initial trajectory so visibility decisions
do not masquerade as differentiable operations. Bilinear field sampling at
the live source position remains differentiable.
"""
import torch
import torch.nn.functional as F


def project(points, camera):
    """Return Gaussian-rasterizer NDC and camera depth (row-vector convention)."""
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    clip = homogeneous @ camera.full_proj_transform.to(points)
    view = homogeneous @ camera.world_view_transform.to(points)
    return clip[..., :2] / (clip[..., 3:4] + 1e-7), view[..., 2]


class FrozenVisibilityProjectedFlow:
    def __init__(self, reference_positions, cameras, observed_flow, observed_images,
                 opacity, max_points=20000):
        # reference_positions: T+1,N,3 in original world coordinates.
        self.cameras = cameras
        self.flow = observed_flow.detach()
        self.height, self.width = observed_flow.shape[-2:]
        n = reference_positions.shape[1]
        stride = max(1, (n + max_points - 1)//max_points)
        self.ids = torch.arange(0, n, stride, device=reference_positions.device)
        self.masks = []
        with torch.no_grad():
            for t in range(len(observed_flow)):
                points = reference_positions[t]
                ndc, depth = project(points, cameras[t])
                xy = (ndc + 1) * ndc.new_tensor([self.width, self.height])/2 - .5
                px = xy.round().long()
                valid = (depth > 0) & (ndc.abs().amax(-1) < .99)
                valid &= opacity.reshape(-1)[:n] > .02
                flat = px[:, 1].clamp(0, self.height-1)*self.width + px[:, 0].clamp(0, self.width-1)
                zbuf = depth.new_full((self.width*self.height,), float('inf'))
                zbuf.scatter_reduce_(0, flat[valid], depth[valid], reduce='amin', include_self=True)
                tolerance = depth[valid].median().abs()*.005 if valid.any() else depth.new_tensor(0.)
                valid &= depth <= zbuf[flat] + tolerance
                colors = F.grid_sample(observed_images[t:t+1].to(points), ndc.reshape(1,1,-1,2),
                                       align_corners=False, padding_mode='zeros')[0,:,0].T
                valid &= colors.amin(-1) < .95
                mask = valid[self.ids]
                if mask.sum() < 16:
                    raise ValueError(f'Insufficient fixed visible foreground points in pair {t}: {int(mask.sum())}')
                self.masks.append(mask)
        self.counts = [int(m.sum()) for m in self.masks]

    def __call__(self, positions):
        terms = []
        scale = positions.new_tensor([self.width/2, self.height/2])
        for t, mask in enumerate(self.masks):
            source, _ = project(positions[t, self.ids], self.cameras[t])
            target, _ = project(positions[t+1, self.ids], self.cameras[t+1])
            predicted = (target-source)*scale
            observed = F.grid_sample(self.flow[t:t+1].to(positions), source.reshape(1,1,-1,2),
                                     align_corners=False, padding_mode='zeros')[0,:,0].T
            terms.append(.5*(predicted[mask]-observed[mask]).square().mean())
        return torch.stack(terms).mean()
