#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

from PIL import Image
from plyfile import PlyData

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import copy_or_link, ensure_dir, list_sorted_files


IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".bmp"]


def newest_iteration_dir(point_cloud_root: Path) -> Path:
    candidates = [p for p in point_cloud_root.iterdir() if p.is_dir() and p.name.startswith("iteration_")]
    if not candidates:
        raise FileNotFoundError(f"No iteration_* dir found under {point_cloud_root}")

    def key_fn(path: Path) -> tuple[int, str]:
        tail = path.name.replace("iteration_", "")
        if tail == "best":
            return (10**9, tail)
        try:
            return (int(tail), tail)
        except ValueError:
            return (-1, tail)

    return sorted(candidates, key=key_fn)[-1]


def make_default_camera_json(point_cloud_ply: Path, reference_image: Path, output_path: Path) -> None:
    ply = PlyData.read(str(point_cloud_ply))
    xyz = list(zip(ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]))
    xs = [p[0] for p in xyz]
    ys = [p[1] for p in xyz]
    zs = [p[2] for p in xyz]
    center = [
        0.5 * (min(xs) + max(xs)),
        0.5 * (min(ys) + max(ys)),
        0.5 * (min(zs) + max(zs)),
    ]
    radius = max(
        math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
        for x, y, z in xyz
    )
    radius = max(radius * 3.0, 1.0)

    with Image.open(reference_image) as img:
        width, height = img.size

    fov_deg = 60.0
    focal = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
    camera = {
        "id": 0,
        "img_name": reference_image.stem,
        "width": width,
        "height": height,
        "position": [center[0], center[1], center[2] - radius],
        "rotation": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "fy": focal,
        "fx": focal,
    }
    output_path.write_text(json.dumps([camera], indent=4) + "\n", encoding="utf-8")


def write_cfg_args(scene_dir: Path) -> None:
    cfg_path = scene_dir / "cfg_args"
    cfg_text = (
        f"Namespace(source_path='{scene_dir}', model_path='{scene_dir}', images='images', "
        "resolution=-1, white_background=False, data_device='cuda', eval=True, "
        "load2gpu_on_the_fly=False, is_blender=False, dxyz_scale=1.0)\n"
    )
    cfg_path.write_text(cfg_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert external 3DGS output into a PhysFlow scene folder.")
    parser.add_argument("--source-type", choices=["pgsr", "grid4d", "splatt3r", "generic"], required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--gaussian-ply", type=Path, default=None)
    parser.add_argument("--camera-json", type=Path, default=None)
    parser.add_argument("--input-ply", type=Path, default=None)
    parser.add_argument("--link", action="store_true")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    ensure_dir(scene_dir)
    ensure_dir(scene_dir / "point_cloud")

    if args.images_dir:
        copy_or_link(args.images_dir, scene_dir / "images", link=args.link)

    if args.source_type in {"pgsr", "grid4d"}:
        iteration_dir = newest_iteration_dir(args.source_dir / "point_cloud")
        target_iteration = scene_dir / "point_cloud" / iteration_dir.name
        copy_or_link(iteration_dir, target_iteration, link=args.link)
        copy_or_link(args.source_dir / "input.ply", scene_dir / "input.ply", link=args.link)
        copy_or_link(args.source_dir / "cameras.json", scene_dir / "cameras.json", link=args.link)
        if (args.source_dir / "cfg_args").exists():
            copy_or_link(args.source_dir / "cfg_args", scene_dir / "cfg_args", link=args.link)
        else:
            write_cfg_args(scene_dir)
        print(f"Converted {args.source_type} scene into {scene_dir}")
        return

    gaussian_ply = args.gaussian_ply or args.source_dir
    if gaussian_ply.is_dir():
        candidates = list(gaussian_ply.rglob("*.ply"))
        if not candidates:
            raise FileNotFoundError(f"No PLY found under {gaussian_ply}")
        gaussian_ply = candidates[0]

    target_iteration = scene_dir / "point_cloud" / "iteration_1"
    ensure_dir(target_iteration)
    shutil.copy2(gaussian_ply, target_iteration / "point_cloud.ply")

    source_input_ply = args.input_ply or gaussian_ply
    shutil.copy2(source_input_ply, scene_dir / "input.ply")

    if args.camera_json:
        shutil.copy2(args.camera_json, scene_dir / "cameras.json")
    else:
        if not args.images_dir:
            raise ValueError("Provide --images-dir when camera json is missing.")
        ref_images = list_sorted_files(args.images_dir, IMAGE_SUFFIXES)
        if not ref_images:
            raise FileNotFoundError(f"No images found under {args.images_dir}")
        make_default_camera_json(target_iteration / "point_cloud.ply", ref_images[0], scene_dir / "cameras.json")

    write_cfg_args(scene_dir)
    print(f"Converted {args.source_type} scene into {scene_dir}")


if __name__ == "__main__":
    main()
