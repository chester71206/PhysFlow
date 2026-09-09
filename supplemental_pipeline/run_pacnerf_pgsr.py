#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import REPOSITORIES, WORKSPACE_ROOT, ensure_dir, run_command
from prepare_pacnerf_multiview import detect_pacnerf_scene_dir, prepare_pacnerf_pgsr_scene


def cli_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def compute_neighbor_report(
    scene_dir: Path,
    multi_view_num: int,
    multi_view_max_angle: float,
    multi_view_max_dis: float,
    multi_view_min_dis: float = 0.01,
) -> list[dict]:
    with (scene_dir / "transforms_train.json").open("r", encoding="utf-8") as f:
        contents = json.load(f)

    cameras = []
    for frame in contents["frames"]:
        c2w = [row[:] for row in frame["transform_matrix"]]
        c2w = [[float(v) for v in row] for row in c2w]
        for row in c2w[:3]:
            row[1] *= -1.0
            row[2] *= -1.0

        rot = [row[:3] for row in c2w[:3]]
        center = [c2w[0][3], c2w[1][3], c2w[2][3]]
        center_ray = [rot[2][0], rot[2][1], rot[2][2]]
        ray_norm = math.sqrt(sum(v * v for v in center_ray))
        center_ray = [v / ray_norm for v in center_ray]

        cameras.append(
            {
                "image_name": Path(frame["file_path"]).stem,
                "center": center,
                "center_ray": center_ray,
            }
        )

    report = []
    for ref_idx, ref_cam in enumerate(cameras):
        ranked = []
        for cand_idx, cand_cam in enumerate(cameras):
            dx = ref_cam["center"][0] - cand_cam["center"][0]
            dy = ref_cam["center"][1] - cand_cam["center"][1]
            dz = ref_cam["center"][2] - cand_cam["center"][2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            dot = sum(a * b for a, b in zip(ref_cam["center_ray"], cand_cam["center_ray"]))
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(math.acos(dot))
            ranked.append(
                {
                    "idx": cand_idx,
                    "image_name": cand_cam["image_name"],
                    "distance": distance,
                    "angle": angle,
                }
            )

        ranked.sort(key=lambda item: (item["distance"], item["angle"]))
        filtered = [
            item for item in ranked
            if item["angle"] < multi_view_max_angle
            and item["distance"] > multi_view_min_dis
            and item["distance"] < multi_view_max_dis
        ]
        report.append(
            {
                "ref_name": ref_cam["image_name"],
                "neighbor_count": min(multi_view_num, len(filtered)),
                "neighbors": filtered[:multi_view_num],
            }
        )

    return report


def prepare_scene(
    scene_name: str,
    input_dir: Path,
    output_dir: Path,
    time_value: float | None,
    time_index: int | None,
    link: bool,
) -> tuple[Path, dict]:
    pacnerf_scene_dir = detect_pacnerf_scene_dir(input_dir, scene_name)
    if pacnerf_scene_dir is None:
        raise FileNotFoundError(
            f"Could not find a PAC-NeRF scene named {scene_name!r} under {input_dir}. "
            "Expected either <input-dir>/all_data.json or <input-dir>/<scene-name>/all_data.json."
        )

    prepared_dir, metadata = prepare_pacnerf_pgsr_scene(
        scene_dir=pacnerf_scene_dir,
        output_dir=output_dir,
        time_value=time_value,
        time_index=time_index,
        link=link,
    )
    return prepared_dir, metadata


def train_pgsr(
    scene_dir: Path,
    model_dir: Path,
    python_bin: str,
    iterations: int,
    multi_view_num: int,
    multi_view_max_angle: float,
    multi_view_max_dis: float,
) -> None:
    pgsr_root = REPOSITORIES["PGSR"]["path"]
    ensure_dir(model_dir.parent)
    cmd = [
        python_bin,
        "train.py",
        "-s",
        str(scene_dir),
        "-m",
        str(model_dir),
        "--iterations",
        str(iterations),
        "--multi_view_num",
        str(multi_view_num),
        "--multi_view_max_angle",
        cli_number(multi_view_max_angle),
        "--multi_view_max_dis",
        cli_number(multi_view_max_dis),
        "--max_abs_split_points",
        "0",
        "--opacity_cull_threshold",
        "0.05",
    ]
    run_command(cmd, cwd=pgsr_root)


def render_pgsr(
    scene_dir: Path,
    model_dir: Path,
    python_bin: str,
    max_depth: float,
    voxel_size: float,
    num_cluster: int,
) -> None:
    pgsr_root = REPOSITORIES["PGSR"]["path"]
    cmd = [
        python_bin,
        "render.py",
        "-s",
        str(scene_dir),
        "-m",
        str(model_dir),
        "--iteration",
        "-1",
        "--skip_test",
        "--max_depth",
        str(max_depth),
        "--voxel_size",
        str(voxel_size),
        "--num_cluster",
        str(num_cluster),
    ]
    run_command(cmd, cwd=pgsr_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Focused PAC-NeRF -> PGSR branch runner. "
            "This uses PAC-NeRF GT cameras as a synthetic-scene adapter/workaround, not the PhysFlow paper's official synthetic benchmark path."
        )
    )
    parser.add_argument("--scene-name", required=True, help="PAC-NeRF scene name, e.g. torus")
    parser.add_argument("--input-dir", type=Path, required=True, help="PAC-NeRF scene dir or a root containing <scene-name>/all_data.json")
    parser.add_argument("--pacnerf-time", type=float, default=None, help="Exact PAC-NeRF time value to slice")
    parser.add_argument("--pacnerf-time-index", type=int, default=None, help="Index into the sorted PAC-NeRF time list")
    parser.add_argument(
        "--prepared-scene-dir",
        type=Path,
        default=None,
        help="Where to write the GT-camera PGSR synthetic scene. Defaults to supplemental_pipeline/workspaces/pacnerf_multiview/<scene-name>/pgsr_scene",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where PGSR writes the reconstructed Gaussian model. Defaults to supplemental_pipeline/workspaces/reconstruction/<scene-name>_pacnerf_pgsr",
    )
    parser.add_argument("--pgsr-python", default=sys.executable, help="Python executable with PGSR dependencies installed")
    parser.add_argument("--link", action="store_true", help="Symlink PAC-NeRF images instead of copying them")
    parser.add_argument("--prepare-only", action="store_true", help="Only write the synthetic PGSR scene and stop")
    parser.add_argument("--skip-render", action="store_true", help="Stop after PGSR training")
    parser.add_argument("--iterations", type=int, default=30000, help="PGSR training iterations")
    parser.add_argument("--multi-view-num", type=int, default=8, help="PGSR nearest-view count")
    parser.add_argument(
        "--multi-view-max-angle",
        type=float,
        default=30.0,
        help="PGSR upstream default. Override only after checking the reported neighbor statistics.",
    )
    parser.add_argument(
        "--multi-view-max-dis",
        type=float,
        default=1.5,
        help="PGSR upstream default. Override only after checking the reported neighbor statistics.",
    )
    parser.add_argument("--multi-view-min-dis", type=float, default=0.01, help="PGSR upstream default minimum camera distance")
    parser.add_argument("--report-neighbors-only", action="store_true", help="Prepare the scene, print PGSR-equivalent neighbor stats, and stop")
    parser.add_argument("--render-max-depth", type=float, default=5.0)
    parser.add_argument("--render-voxel-size", type=float, default=0.002)
    parser.add_argument("--render-num-cluster", type=int, default=1)
    args = parser.parse_args()

    if args.pacnerf_time is not None and args.pacnerf_time_index is not None:
        raise ValueError("Specify only one of --pacnerf-time or --pacnerf-time-index")

    prepared_scene_dir = args.prepared_scene_dir or (WORKSPACE_ROOT / "pacnerf_multiview" / args.scene_name / "pgsr_scene")
    output_dir = args.output_dir or (WORKSPACE_ROOT / "reconstruction" / f"{args.scene_name}_pacnerf_pgsr")

    scene_dir, metadata = prepare_scene(
        scene_name=args.scene_name,
        input_dir=args.input_dir,
        output_dir=prepared_scene_dir,
        time_value=args.pacnerf_time,
        time_index=args.pacnerf_time_index,
        link=args.link,
    )
    print(
        f"Prepared PAC-NeRF scene {args.scene_name} with {metadata['num_views']} views "
        f"at time index {metadata['selected_time_index']} (time={metadata['selected_time']}) into {scene_dir}"
    )
    print(
        "Using GT-camera synthetic-scene adapter for PGSR. "
        "This is a reconstruction workaround for the multi-view branch, not the paper's official PAC-NeRF synthetic benchmark pipeline."
    )
    neighbor_report = compute_neighbor_report(
        scene_dir=scene_dir,
        multi_view_num=args.multi_view_num,
        multi_view_max_angle=args.multi_view_max_angle,
        multi_view_max_dis=args.multi_view_max_dis,
        multi_view_min_dis=args.multi_view_min_dis,
    )
    print(
        f"Neighbor stats under PGSR thresholds angle<{args.multi_view_max_angle}, "
        f"distance in ({args.multi_view_min_dis}, {args.multi_view_max_dis}):"
    )
    for row in neighbor_report:
        neighbor_names = [item["image_name"] for item in row["neighbors"]]
        print(f"  {row['ref_name']}: {row['neighbor_count']} -> {neighbor_names}")

    if args.prepare_only or args.report_neighbors_only:
        return

    train_pgsr(
        scene_dir=scene_dir,
        model_dir=output_dir,
        python_bin=args.pgsr_python,
        iterations=args.iterations,
        multi_view_num=args.multi_view_num,
        multi_view_max_angle=args.multi_view_max_angle,
        multi_view_max_dis=args.multi_view_max_dis,
    )

    if args.skip_render:
        return

    render_pgsr(
        scene_dir=scene_dir,
        model_dir=output_dir,
        python_bin=args.pgsr_python,
        max_depth=args.render_max_depth,
        voxel_size=args.render_voxel_size,
        num_cluster=args.render_num_cluster,
    )


if __name__ == "__main__":
    main()
