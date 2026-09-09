#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import REPOSITORIES, ensure_dir, list_sorted_files


def load_segment_module():
    repo_root = REPOSITORIES["SAM2Point"]["path"]
    sys.path.insert(0, str(repo_root))
    import segment as sam2_segment
    return sam2_segment


def save_masks(mask_array: np.ndarray, frame_paths: list[Path], output_dir: Path) -> None:
    ensure_dir(output_dir)
    for frame_path, mask in zip(frame_paths, mask_array):
        img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        img.save(output_dir / f"{frame_path.stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM2Point's video predictor on a folder of frames.")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-type", choices=["point", "box", "mask"], required=True)
    parser.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--box", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--n-seed-frames", type=int, default=1, help="Used by the box prompt wrapper.")
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear output-dir before writing masks.")
    args = parser.parse_args()

    frame_paths = list_sorted_files(args.frames_dir, [".png", ".jpg", ".jpeg"])
    if not frame_paths:
        raise FileNotFoundError(f"No frames found under {args.frames_dir}")
    if args.output_dir.exists() and not args.keep_existing:
        shutil.rmtree(args.output_dir)

    sam2_segment = load_segment_module()
    frame_strs = [str(p) for p in frame_paths]

    if args.prompt_type == "point":
        if not args.point:
            raise ValueError("--point is required for point prompt.")
        masks = sam2_segment.segment_point(frame_strs, list(args.point))
    elif args.prompt_type == "box":
        if not args.box:
            raise ValueError("--box is required for box prompt.")
        masks = sam2_segment.segment_box(frame_strs, list(args.box), args.n_seed_frames)
    else:
        if not args.point:
            raise ValueError("--point is required for mask prompt.")
        masks, _ = sam2_segment.segment_mask(frame_strs, list(args.point))

    save_masks(masks.detach().cpu().numpy(), frame_paths, args.output_dir)
    print(f"Saved {len(frame_paths)} masks to {args.output_dir}")


if __name__ == "__main__":
    main()
