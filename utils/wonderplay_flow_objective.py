"""Opacity-composited pixel flow from Gaussian motion, following WonderPlay.

The WonderPlay/HUGS rasterizer receives per-Gaussian projected displacement as
``delta`` and composites it with Gaussian opacity and covariance. The resulting
pixel flow is compared to fixed observed RAFT flow on a fixed foreground mask.
"""
import math

import torch
import torch.nn.functional as F
from depth_diff_gaussian_rasterization_min import GaussianRasterizationSettings, GaussianRasterizer

from utils.projected_flow_objective import project


class WonderPlayFlowObjective:
    def __init__(self, cameras, observed_flow, observed_images, opacity, initial_cov,
                 scale, center, rotations, transform_cov, max_points=20000):
        self.cameras = cameras
        self.flow = observed_flow.detach()
        self.height, self.width = self.flow.shape[-2:]
        n = opacity.shape[0]
        stride = max(1, math.ceil(n / max_points))
        self.ids = torch.arange(0, n, stride, device=opacity.device)
        self.opacity = opacity.detach()[self.ids].contiguous()
        self.initial_cov = initial_cov.detach()[self.ids].contiguous()
        self.scale, self.center, self.rotations = scale, center, rotations
        self.transform_cov = transform_cov
        with torch.no_grad():
            visible = F.interpolate(observed_images.detach(), size=(self.height, self.width),
                                    mode='bilinear', align_corners=False)
            self.masks = [(frame.amin(0) < .95).detach() for frame in visible[:-1]]
        self.counts = [int(mask.sum()) for mask in self.masks]
        self.rasterizers = []
        for camera in cameras[:-1]:
            settings = GaussianRasterizationSettings(
                image_height=self.height, image_width=self.width,
                tanfovx=math.tan(camera.FoVx * .5), tanfovy=math.tan(camera.FoVy * .5),
                bg=torch.zeros(3, device=opacity.device), scale_modifier=1.,
                viewmatrix=camera.world_view_transform.detach(),
                projmatrix=camera.full_proj_transform.detach(), sh_degree=0,
                campos=camera.camera_center.detach(), prefiltered=False, debug=False)
            self.rasterizers.append(GaussianRasterizer(settings))

    def __call__(self, positions, outputs):
        covariances = outputs[1]
        terms = []
        scale = positions.new_tensor([self.width / 2, self.height / 2])
        for t, (rasterizer, mask) in enumerate(zip(self.rasterizers, self.masks)):
            source = positions[t, self.ids].contiguous()
            target = positions[t+1, self.ids]
            source_ndc, _ = project(source, self.cameras[t])
            target_ndc, _ = project(target, self.cameras[t+1])
            delta = torch.cat([(target_ndc - source_ndc) * scale,
                               torch.zeros_like(source[:, :1])], dim=-1).contiguous()
            raw_cov = self.initial_cov if t == 0 else covariances[t-1].reshape(-1, 6)[self.ids]
            cov = self.transform_cov(raw_cov / (self.scale * self.scale), self.rotations).contiguous()
            zeros = torch.zeros_like(source)
            black = torch.zeros_like(source)
            features = source.new_zeros((len(source), 20))
            _, _, _, _, pixel_flow = rasterizer(
                means3D=source, means2D=zeros, colors_precomp=black,
                opacities=self.opacity, cov3D_precomp=cov, feats3D=features, delta=delta)
            predicted = pixel_flow[:2]
            observed = self.flow[t]
            terms.append(.5 * (predicted[:, mask] - observed[:, mask]).square().mean())
        return torch.stack(terms).mean()
