from typing import Union, List
from jaxtyping import Float, Int

import os
import re

import PIL
import numpy as np
import torch
import torch.nn.functional as F
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


def load_guidance_frames(guidance_path, downsample=1.0, num_frames=8):
    fnames = sorted((
        fn for fn in os.listdir(guidance_path)
        if os.path.splitext(fn)[1].lower() in {".png", ".jpg", ".jpeg"}
    ), key=lambda name: [int(part) if re.fullmatch(r"-?\d+", part) else part
                         for part in re.split(r"(-?\d+)", name)])
    if not fnames:
        raise FileNotFoundError(f"No guidance images found under {guidance_path}")
    frame_per_stage = max(1, len(fnames) // num_frames)

    frames = []
    for i, fn in enumerate(fnames):
        if i % frame_per_stage != 0:
            continue
        frame = imread(os.path.join(guidance_path, fn))
        if downsample != 1.:
            frame = resize(frame, (int(frame.shape[1] * downsample), int(frame.shape[0] * downsample)))
        frame = cvtColor(frame, COLOR_BGR2RGB)
        frame = numpy_to_pt(frame / 255.)
        frames.append(frame)
    return torch.cat(frames, dim=0)


class CogVideoGuidance:
    def __init__(self, guidance_path, downsample=1., num_frames=8, flow_device=None):
        self.device = get_device()
        self.flow_device = flow_device or self.device
        self.weights_dtype = torch.float32
        self.FLOW_HEIGHT = 256
        self.FLOW_WIDTH = 480
        
        self.guidance = load_guidance_frames(
            guidance_path, downsample=downsample, num_frames=num_frames
        ).to(self.flow_device)
        print("guidance shape:",self.guidance.shape)
        print("guidance flow device:", self.flow_device)

        # predict optical flow
        prev_batch = preprocess(self.guidance[:-1])
        curr_batch = preprocess(self.guidance[1:])
        self.model = raft_large(pretrained=True, progress=False).to(self.flow_device)
        self.model = self.model.eval()
        # RAFT is fixed guidance; keep gradients to rendered inputs only.
        self.model.requires_grad_(False)

        with torch.no_grad():
            self.guidance_flows = self.model(prev_batch, curr_batch)[-1]
        
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
        prev_batch = preprocess(
            torch.concat([rgb_BCHW_init, rgb_BCHW[:-1]], dim=0).to(self.flow_device)
        ).detach()
        curr_batch = preprocess(rgb_BCHW.to(self.flow_device))

        rgb_flows = self.model(prev_batch, curr_batch)[-1]
        ## debug
        flow_imgs = flow_to_image(rgb_flows)

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
        
        bbox_2d = np.asarray(bbox_2d)
        bbox_2d[:2] = bbox_2d[:2] / rgb_BCHW.shape[2] * self.FLOW_HEIGHT
        bbox_2d[2:] = bbox_2d[2:] / rgb_BCHW.shape[3] * self.FLOW_WIDTH

        loss_flow = 0.5 * F.mse_loss(rgb_flows[..., bbox_2d[0]:bbox_2d[1], bbox_2d[2]:bbox_2d[3]],
                                        self.guidance_flows[..., bbox_2d[0]:bbox_2d[1], bbox_2d[2]:bbox_2d[3]], 
                                        reduction=reduction)
        loss['loss_flow'] = loss_flow.to(rgb_BCHW.device)

        return loss


class DummyGuidance:
    def __init__(self, guidance_path, downsample=1., num_frames=8):
        self.device = get_device()
        self.guidance = load_guidance_frames(
            guidance_path,
            downsample=downsample,
            num_frames=num_frames,
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
