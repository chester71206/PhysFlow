#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import WORKSPACE_ROOT, ensure_dir, list_sorted_files, run_command


def copy_frames(
    src_dir: Path,
    dst_dir: Path,
    ext: str = ".png",
    frame_stride: int = 1,
    max_frames: int | None = None,
    resize_width: int | None = None,
    resize_height: int | None = None,
) -> int:
    ensure_dir(dst_dir)
    frames = list_sorted_files(src_dir, [".png", ".jpg", ".jpeg"])
    written = 0
    stride = max(frame_stride, 1)
    for idx, frame in enumerate(frames):
        if idx % stride != 0:
            continue
        with Image.open(frame) as image:
            if resize_width and resize_height:
                image = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)
            if ext.lower() in {".jpg", ".jpeg"} and image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            image.save(dst_dir / f"{written:05d}{ext}")
        written += 1
        if max_frames is not None and written >= max_frames:
            break
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Grid4D workspace from a raw dynamic video or a directory of frames.")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument("--extract-python", default=sys.executable)
    parser.add_argument("--colmap-python", default=sys.executable)
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    if not args.video and not args.frames_dir:
        raise ValueError("Provide --video or --frames-dir.")
    if args.frames_dir and args.target_fps is not None:
        raise ValueError("--target-fps is only supported together with --video.")

    workspace = WORKSPACE_ROOT / "grid4d_video" / args.scene_name
    if workspace.exists():
        shutil.rmtree(workspace)
    input_dir = workspace / "input"
    ensure_dir(input_dir)

    if args.video:
        cmd = [
            args.extract_python,
            str(THIS_DIR / "extract_video_frames.py"),
            "--video", str(args.video),
            "--output-dir", str(input_dir),
            "--frame-stride", str(args.frame_stride),
        ]
        if args.max_frames is not None:
            cmd.extend(["--max-frames", str(args.max_frames)])
        if args.target_fps is not None:
            cmd.extend(["--target-fps", str(args.target_fps)])
        if args.resize_width and args.resize_height:
            cmd.extend(["--resize-width", str(args.resize_width), "--resize-height", str(args.resize_height)])
        run_command(cmd)
    else:
        copied = copy_frames(
            args.frames_dir,
            input_dir,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            resize_width=args.resize_width,
            resize_height=args.resize_height,
        )
        print(f"Prepared {copied} source frames into {input_dir}")

    colmap_cmd = [
        args.colmap_python,
        str(THIS_DIR / "prepare_colmap_scene.py"),
        "--scene-dir", str(workspace),
        "--colmap-bin", args.colmap_bin,
        "--camera-model", args.camera_model,
    ]
    if args.no_gpu:
        colmap_cmd.append("--no-gpu")
    run_command(colmap_cmd)

    images_dir = workspace / "images"
    guidance_dir = workspace / "guidance_frames"
    if guidance_dir.exists():
        shutil.rmtree(guidance_dir)
    shutil.copytree(images_dir, guidance_dir)
    print(f"Prepared Grid4D workspace at {workspace}")
    print(f"Use {images_dir} as 3DGS training frames and {guidance_dir} as PhysFlow guidance frames.")


if __name__ == "__main__":
    main()
