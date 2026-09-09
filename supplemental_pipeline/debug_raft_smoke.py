#!/usr/bin/env python3
from __future__ import annotations

import argparse
import faulthandler

import torch
from torchvision.models.optical_flow import raft_large


def maybe_init_warp() -> None:
    import warp as wp

    print("[step] init warp")
    wp.init()
    wp.config.verify_cuda = True
    print("[ok] warp initialized")


def maybe_init_taichi() -> None:
    import taichi as ti

    print("[step] init taichi")
    ti.init(arch=ti.cuda, device_memory_GB=2.0)
    print("[ok] taichi initialized")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal CUDA smoke test for torchvision RAFT.")
    parser.add_argument("--device", default="cuda", help="Expected to be cuda for crash reproduction.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--init-warp", action="store_true")
    parser.add_argument("--init-taichi", action="store_true")
    args = parser.parse_args()

    faulthandler.enable()

    print("[env] torch", torch.__version__)
    print("[env] cuda", torch.version.cuda)
    print("[env] cuda available", torch.cuda.is_available())
    print("[env] device", args.device)

    if args.init_warp:
        maybe_init_warp()
    if args.init_taichi:
        maybe_init_taichi()

    print("[step] allocate dummy frames")
    prev_batch = torch.rand(args.batch_size, 3, args.height, args.width, device=args.device, dtype=torch.float32)
    curr_batch = torch.rand(args.batch_size, 3, args.height, args.width, device=args.device, dtype=torch.float32)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    print("[ok] dummy frames allocated")

    print("[step] build raft_large(pretrained=True)")
    model = raft_large(pretrained=True, progress=False).to(args.device).eval()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    print("[ok] raft model ready")

    print("[step] forward")
    with torch.no_grad():
        flows = model(prev_batch, curr_batch)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    print("[ok] forward complete")

    final_flow = flows[-1]
    print("[result] num_updates", len(flows))
    print("[result] final_flow_shape", tuple(final_flow.shape))
    print("[result] final_flow_dtype", final_flow.dtype)


if __name__ == "__main__":
    main()
