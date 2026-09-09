#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from common import ensure_dir


def run(command: list[str]) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, check=True)


def clear_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a Colmap-style scene folder from extracted video frames.")
    parser.add_argument("--scene-dir", type=Path, required=True, help="Folder containing input/ frames.")
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--skip-undistort", action="store_true")
    args = parser.parse_args()

    scene_dir = args.scene_dir
    input_dir = scene_dir / "input"
    if not input_dir.exists():
        raise FileNotFoundError(f"{input_dir} does not exist")

    sparse_dir = scene_dir / "sparse"
    database_path = scene_dir / "database.db"
    images_dir = scene_dir / "images"
    sparse_zero = sparse_dir / "0"

    ensure_dir(sparse_dir)
    if sparse_zero.exists():
        shutil.rmtree(sparse_zero)
    if database_path.exists():
        database_path.unlink()
    if images_dir.exists():
        shutil.rmtree(images_dir)

    gpu_flag = "0" if args.no_gpu else "1"
    run([
        args.colmap_bin,
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(input_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", args.camera_model,
        "--SiftExtraction.use_gpu", gpu_flag,
    ])
    run([
        args.colmap_bin,
        "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", gpu_flag,
    ])
    run([
        args.colmap_bin,
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(input_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_global_function_tolerance=0.000001",
    ])

    if not sparse_zero.exists():
        raise RuntimeError(f"COLMAP mapper did not create {sparse_zero}")

    if args.skip_undistort:
        images_dir.mkdir(parents=True, exist_ok=True)
        for image in sorted(input_dir.iterdir()):
            if image.is_file():
                shutil.copy2(image, images_dir / image.name)
        print(f"Copied input frames to {images_dir} without undistortion")
        return

    run([
        args.colmap_bin,
        "image_undistorter",
        "--image_path", str(input_dir),
        "--input_path", str(sparse_zero),
        "--output_path", str(scene_dir),
        "--output_type", "COLMAP",
    ])
    print(f"Prepared COLMAP scene at {scene_dir}")


if __name__ == "__main__":
    main()
