#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sequential frames from a video for the supplemental Grid4D pipeline.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=1, help="Keep one frame every N input frames.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--target-fps", type=float, default=None, help="Alternative to stride; keeps frames close to this fps.")
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument("--ext", choices=["png", "jpg"], default="png")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(args.video)
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", 0.0) or 0.0)
    stride = max(args.frame_stride, 1)
    if args.target_fps and src_fps > 0:
        stride = max(int(round(src_fps / args.target_fps)), 1)

    written = 0
    for idx, frame in enumerate(reader):
        if idx % stride != 0:
            continue

        image = Image.fromarray(frame)
        if args.resize_width and args.resize_height:
            image = image.resize((args.resize_width, args.resize_height), resample=Image.Resampling.LANCZOS)

        out_path = args.output_dir / f"{written:05d}.{args.ext}"
        imageio.imwrite(out_path, image)

        written += 1
        if args.max_frames is not None and written >= args.max_frames:
            break

    reader.close()
    print(f"Extracted {written} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
