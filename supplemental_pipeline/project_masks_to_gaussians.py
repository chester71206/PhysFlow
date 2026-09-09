#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import list_sorted_files


def newest_iteration_dir(point_cloud_root: Path) -> Path:
    candidates = [p for p in point_cloud_root.iterdir() if p.is_dir() and p.name.startswith("iteration_")]
    if not candidates:
        raise FileNotFoundError(f"No point_cloud/iteration_* found under {point_cloud_root}")

    def key_fn(path: Path) -> tuple[int, str]:
        tail = path.name.replace("iteration_", "")
        if tail == "best":
            return (10**9, tail)
        try:
            return (int(tail), tail)
        except ValueError:
            return (-1, tail)

    return sorted(candidates, key=key_fn)[-1]


def load_xyz(point_cloud_ply: Path) -> np.ndarray:
    ply = PlyData.read(str(point_cloud_ply))
    x = np.asarray(ply["vertex"]["x"])
    y = np.asarray(ply["vertex"]["y"])
    z = np.asarray(ply["vertex"]["z"])
    return np.stack([x, y, z], axis=1)


def write_points_ply(points: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    verts = np.empty(points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts["x"] = points[:, 0]
    verts["y"] = points[:, 1]
    verts["z"] = points[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")]).write(str(output_path))


def build_camera_lookup(camera_json: Path) -> list[dict]:
    with camera_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def camera_world_to_image(raw_camera: dict, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.asarray(raw_camera["rotation"], dtype=np.float64)
    c2w[:3, 3] = np.asarray(raw_camera["position"], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    cam_xyz = (w2c @ xyz_h.T).T[:, :3]

    z = cam_xyz[:, 2]
    x = cam_xyz[:, 0]
    y = cam_xyz[:, 1]
    u = raw_camera["fx"] * (x / z) + raw_camera["width"] * 0.5
    v = raw_camera["fy"] * (y / z) + raw_camera["height"] * 0.5
    return u, v, z


def match_cameras_and_masks(cameras: list[dict], masks: list[Path]) -> list[tuple[dict, Path]]:
    mask_by_stem = {p.stem: p for p in masks}
    pairs: list[tuple[dict, Path]] = []
    missing: list[str] = []
    for camera in cameras:
        stem = Path(str(camera["img_name"])).stem
        mask_path = mask_by_stem.get(stem)
        if mask_path is None:
            missing.append(stem)
            continue
        pairs.append((camera, mask_path))

    if pairs:
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            print(f"Warning: skipped {len(missing)} cameras without matching masks: {preview}{suffix}")
        return pairs

    if len(cameras) == len(masks):
        print("Warning: no camera names matched mask stems; falling back to sequential camera/mask pairing.")
        return list(zip(cameras, masks))

    raise RuntimeError(
        "Could not match any masks to cameras by stem name. "
        "Expected mask files named like cameras.json img_name entries."
    )


def select_visible_points(valid_idx: np.ndarray, u: np.ndarray, v: np.ndarray, z: np.ndarray, width: int) -> np.ndarray:
    if valid_idx.size == 0:
        return valid_idx

    uu = np.round(u[valid_idx]).astype(np.int32)
    vv = np.round(v[valid_idx]).astype(np.int32)
    linear = vv.astype(np.int64) * width + uu.astype(np.int64)
    z_vals = z[valid_idx]

    # Approximate a z-buffer by keeping the nearest Gaussian center per pixel.
    order = np.lexsort((z_vals, linear))
    sorted_linear = linear[order]
    keep = np.ones(order.shape[0], dtype=bool)
    keep[1:] = sorted_linear[1:] != sorted_linear[:-1]
    return valid_idx[order[keep]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 2D masks back to a Gaussian point cloud.")
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, default=None)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--min-votes", type=int, default=1)
    parser.add_argument("--vote-frac", type=float, default=0.3)
    args = parser.parse_args()

    point_cloud_root = args.scene_dir / "point_cloud"
    point_cloud_ply = newest_iteration_dir(point_cloud_root) / "point_cloud.ply"
    xyz = load_xyz(point_cloud_ply)

    cameras = build_camera_lookup(args.scene_dir / "cameras.json")
    masks = list_sorted_files(args.masks_dir, [".png", ".jpg", ".jpeg"])
    if not masks:
        raise FileNotFoundError(f"No masks found under {args.masks_dir}")

    camera_mask_pairs = match_cameras_and_masks(cameras, masks)

    votes = np.zeros((xyz.shape[0],), dtype=np.int32)
    used = 0
    for raw_camera, mask_path in camera_mask_pairs:
        mask = np.asarray(Image.open(mask_path).convert("L"))
        u, v, z = camera_world_to_image(raw_camera, xyz)
        valid = (
            (z > 1e-6)
            & (u >= 0)
            & (u < mask.shape[1])
            & (v >= 0)
            & (v < mask.shape[0])
        )
        if not np.any(valid):
            continue
        valid_idx = np.nonzero(valid)[0]
        visible_idx = select_visible_points(valid_idx, u, v, z, mask.shape[1])
        if visible_idx.size == 0:
            continue
        uu = np.round(u[visible_idx]).astype(np.int32)
        vv = np.round(v[visible_idx]).astype(np.int32)
        uu = np.clip(uu, 0, mask.shape[1] - 1)
        vv = np.clip(vv, 0, mask.shape[0] - 1)
        selected = mask[vv, uu] > args.mask_threshold
        votes[visible_idx[selected]] += 1
        used += 1

    if used == 0:
        raise RuntimeError("No valid camera/mask pairs were usable for projection.")

    required_votes = max(args.min_votes, int(np.ceil(used * args.vote_frac)))
    selected_points = xyz[votes >= required_votes]
    if selected_points.size == 0:
        raise RuntimeError("No Gaussian centers survived the voting threshold.")

    output_ply = args.output_ply or (args.scene_dir / "moving_part_points.ply")
    write_points_ply(selected_points, output_ply)
    print(f"Saved {selected_points.shape[0]} moving points to {output_ply}")


if __name__ == "__main__":
    main()
