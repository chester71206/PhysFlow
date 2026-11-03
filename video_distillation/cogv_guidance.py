from typing import Union, List
from jaxtyping import Float, Int

import os
import re

import PIL
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from torch import Tensor
from threestudio_utils import get_device
from cv2_compat import COLOR_BGR2RGB, cvtColor, imread, resize

FLOW_HEIGHT = 256
FLOW_WIDTH = 480

def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
    """
    Convert a NumPy image to a PyTorch tensor.
    """
    if images.ndim == 3:
        images = images[None, ...]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images

def preprocess(batch, height=FLOW_HEIGHT, width=FLOW_WIDTH, dtype=torch.float32):
    batch = (batch * 2. - 1.).to(dtype).clamp(-1.0, 1.0)
    batch = F.interpolate(
            batch, (height, width), mode="bilinear", align_corners=False
        )
    return batch


def load_guidance_frames(guidance_path, downsample=1.0, num_frames=8, frame_names=None,
                         background_mode='source'):
    if background_mode not in ('source', 'white_masked'):
        raise ValueError('guidance background must be source or white_masked')
    fnames = sorted((
        fn for fn in os.listdir(guidance_path)
        if os.path.splitext(fn)[1].lower() in {".png", ".jpg", ".jpeg"}
    ), key=lambda name: [int(part) if re.fullmatch(r"-?\d+", part) else part
                         for part in re.split(r"(-?\d+)", name)])
    if not fnames:
        raise FileNotFoundError(f"No guidance images found under {guidance_path}")
    frame_per_stage = max(1, len(fnames) // num_frames)

    # Official aligned entry point supplies its exact preflight selection.
    # No argument preserves historical sampling for archived experiments.
    if frame_names is not None:
        if len(frame_names) != num_frames + 1 or len(set(frame_names)) != len(frame_names):
            raise ValueError("Aligned guidance needs n_key_frame+1 distinct images")
        if any(name not in fnames for name in frame_names):
            raise ValueError("Selected guidance image is missing")
        fnames = list(frame_names)
        frame_per_stage = 1

    frames = []
    for i, fn in enumerate(fnames):
        if i % frame_per_stage != 0:
            continue
        if background_mode == 'white_masked':
            if not re.fullmatch(r'r_\d+_\d+\.(png|jpg|jpeg)', fn, re.IGNORECASE):
                raise ValueError('white_masked guidance requires PAC-NeRF frame identities')
            mask_name = 'a_' + os.path.splitext(fn)[0][2:] + '.png'
            mask_path = os.path.join(os.path.dirname(guidance_path), 'masks', mask_name)
            with PIL.Image.open(mask_path) as supplied:
                if supplied.mode != 'RGBA':
                    raise ValueError(f'Expected supplied RGBA image: {mask_path}')
                rgba = np.asarray(supplied, dtype=np.float32)
            alpha = rgba[..., 3:4] / 255.
            rgb = np.rint(rgba[..., :3] * alpha + 255. * (1. - alpha)).astype(np.uint8)
            frame = rgb[..., ::-1].copy()
        else:
            frame = imread(os.path.join(guidance_path, fn))
        if downsample != 1.:
            frame = resize(frame, (int(frame.shape[1] * downsample), int(frame.shape[0] * downsample)))
        frame = cvtColor(frame, COLOR_BGR2RGB)
        frame = numpy_to_pt(frame / 255.)
        frames.append(frame)
    return torch.cat(frames, dim=0)


class CogVideoGuidance:
    def __init__(self, guidance_path, downsample=1., num_frames=8, flow_device=None, frame_names=None,
                 background_mode='source'):
        self.device = get_device()
        self.flow_device = flow_device or self.device
        self.weights_dtype = torch.float32
        self.FLOW_HEIGHT = 256
        self.FLOW_WIDTH = 480
        
        self.guidance = load_guidance_frames(
            guidance_path, downsample=downsample, num_frames=num_frames, frame_names=frame_names,
            background_mode=background_mode
        ).to(self.flow_device)
        print("guidance shape:",self.guidance.shape)
        print("guidance flow device:", self.flow_device)

        # predict optical flow
        self.model = raft_large(pretrained=True, progress=False).to(self.flow_device)
        self.model = self.model.eval()
        # RAFT is fixed guidance; keep gradients to rendered inputs only.
        self.model.requires_grad_(False)
        # This process also initializes Warp/Taichi CUDA contexts. cuDNN's
        # convolution path has segfaulted inside RAFT on the shared RTX 3090s;
        # the native PyTorch convolution path is stable in reference evaluation.
        torch.backends.cudnn.enabled = False

        with torch.no_grad():
            # A full sequence in one RAFT call can exhaust a shared GPU (or
            # crash in native CUDA code before PyTorch reports an OOM).
            # Each pair is independent, so batching here does not change the
            # optical-flow objective.
            self.guidance_flows = torch.cat([
                self.model(preprocess(self.guidance[i:i + 1]),
                           preprocess(self.guidance[i + 1:i + 2]))[-1]
                for i in range(len(self.guidance) - 1)
            ], dim=0)
        
        # load masks -- PAC-NeRF
        # self.guidance_masks = None
        # mask_path = guidance_path.replace(guidance_path.split('/')[-1], 'masks')
        # if os.path.exists(mask_path):
        #     fnames = os.listdir(mask_path)
        #     fnames.sort()
            
        #     frames = []
        #     for i, fn in enumerate(fnames):
        #         if i % frame_per_stage != 0:
        #             continue
        #         frame = cv2.imread(os.path.join(mask_path, fn), cv2.IMREAD_UNCHANGED)
        #         if downsample != 1.:
        #             frame = cv2.resize(frame, (frame.shape[1]*downsample, frame.shape[0]*downsample)) 
        #         frame = numpy_to_pt(frame[..., -1:] / 255.)
        #         frames.append(frame)
        #     self.guidance_masks = torch.cat(frames, dim=0).to(self.device) # (BCHW)
        #     print("guidance_masks shape:",self.guidance_masks.shape)


    def predict_flow(self, rgb_BCHW, rgb_BCHW_init):
        # predict optical flow
        flow_dtype = next(self.model.parameters()).dtype
        prev_batch = preprocess(
            torch.concat([rgb_BCHW_init, rgb_BCHW[:-1]], dim=0).to(self.flow_device), dtype=flow_dtype
        )
        curr_batch = preprocess(rgb_BCHW.to(self.flow_device), dtype=flow_dtype)

        # Both rendered frames depend on material parameters. Detaching the
        # previous frame differentiates a different objective from full-video FD.
        # Recompute one pair at a time in backward to bound RAFT activation RAM.
        def pair_flow(previous, current):
            # torchvision's training-time coordinate detach is also a surrogate
            # gradient. Keep the recurrent sampling-coordinate derivative when
            # differentiating a fixed RAFT as an image-space loss.
            from utils.raft_coordinate_adjoint_20261011 import coordinate_adjoint_forward
            flow = coordinate_adjoint_forward(self.model, previous, current)[-1]
            self.model.corr_block.corr_pyramid.clear()
            return flow
        flows = []
        for i in range(len(curr_batch)):
            previous, current = prev_batch[i:i+1], curr_batch[i:i+1]
            if torch.is_grad_enabled() and (previous.requires_grad or current.requires_grad):
                flows.append(checkpoint(pair_flow, previous, current, use_reentrant=False))
            else:
                flows.append(pair_flow(previous, current))
        rgb_flows = torch.cat(flows)
        ## debug
        flow_imgs = flow_to_image(rgb_flows.detach().float())

        return rgb_flows, flow_imgs

    def normalize_flow(self, flow):
        max_norm = torch.sum(flow**2, dim=1).sqrt().max()
        epsilon = torch.finfo((flow).dtype).eps
        normalized_flow = flow / (max_norm + epsilon)
        return normalized_flow

    def __call__(
        self,
        rgb_BCHW: Float[Tensor, "B C H W"],
        rgb_BCHW_init: Float[Tensor, "B C H W"],
        bbox_2d=None,
        reduction="sum"
    ):
        rgb_flows, _ = self.predict_flow(rgb_BCHW, rgb_BCHW_init)

        loss = {}
        if bbox_2d is None:
            bbox_2d = [0, rgb_BCHW.shape[2], 0, rgb_BCHW.shape[3]]
        
        # Downsampling produces float coordinates. Scale in floating point,
        # then truncate to pixel indices (matching the original integer case).
        # Copy so an ndarray supplied by the caller is not modified in place.
        bbox_2d = np.array(bbox_2d, dtype=np.float64, copy=True)
        if bbox_2d.shape != (4,) or not np.isfinite(bbox_2d).all():
            raise ValueError("bbox_2d must contain four finite coordinates: y0, y1, x0, x1")
        bbox_2d[:2] = bbox_2d[:2] / rgb_BCHW.shape[2] * self.FLOW_HEIGHT
        bbox_2d[2:] = bbox_2d[2:] / rgb_BCHW.shape[3] * self.FLOW_WIDTH
        bbox_2d[:2] = np.clip(bbox_2d[:2], 0, self.FLOW_HEIGHT)
        bbox_2d[2:] = np.clip(bbox_2d[2:], 0, self.FLOW_WIDTH)
        bbox_2d = bbox_2d.astype(np.int64)
        if bbox_2d[0] >= bbox_2d[1] or bbox_2d[2] >= bbox_2d[3]:
            raise ValueError(f"bbox_2d is empty after mapping to flow pixels: {bbox_2d.tolist()}")

        loss_flow = 0.5 * F.mse_loss(rgb_flows[..., bbox_2d[0]:bbox_2d[1], bbox_2d[2]:bbox_2d[3]],
                                        self.guidance_flows[..., bbox_2d[0]:bbox_2d[1], bbox_2d[2]:bbox_2d[3]], 
                                        reduction=reduction)
        loss['loss_flow'] = loss_flow.to(rgb_BCHW.device)

        return loss


class DummyGuidance:
    def __init__(self, guidance_path, downsample=1., num_frames=8, frame_names=None,
                 background_mode='source'):
        self.device = get_device()
        self.guidance = load_guidance_frames(
            guidance_path,
            downsample=downsample,
            num_frames=num_frames,
            frame_names=frame_names,
            background_mode=background_mode,
        ).to(self.device)
        print("guidance shape:", self.guidance.shape)
        print("Using dummy guidance: skipping RAFT optical flow and returning zero flow loss.")

    def predict_flow(self, rgb_BCHW, rgb_BCHW_init):
        batch = rgb_BCHW.shape[0]
        _, _, height, width = rgb_BCHW.shape
        flow = torch.zeros((batch, 2, FLOW_HEIGHT, FLOW_WIDTH), dtype=rgb_BCHW.dtype, device=rgb_BCHW.device)
        flow_imgs = torch.zeros((batch, 3, height, width), dtype=rgb_BCHW.dtype, device=rgb_BCHW.device)
        return flow, flow_imgs

    def __call__(
        self,
        rgb_BCHW: Float[Tensor, "B C H W"],
        rgb_BCHW_init: Float[Tensor, "B C H W"],
        bbox_2d=None,
        reduction="sum"
    ):
        return {"loss_flow": rgb_BCHW.sum() * 0.0}
