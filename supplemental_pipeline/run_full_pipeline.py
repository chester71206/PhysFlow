#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import REPO_ROOT, REPOSITORIES, WORKSPACE_ROOT, ensure_dir, list_sorted_files, run_command
from prepare_pacnerf_multiview import detect_pacnerf_scene_dir, prepare_pacnerf_multiview_scene, prepare_pacnerf_pgsr_scene


def build_scene_paths(scene_name: str) -> tuple[Path, Path]:
    scene_dir = REPO_ROOT / "model" / "custom" / scene_name
    config_dir = REPO_ROOT / "config" / "custom"
    ensure_dir(scene_dir)
    ensure_dir(config_dir)
    return scene_dir, config_dir / f"{scene_name}_config.json"


def copy_guidance_frames(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def run_pgsr(
    scene_name: str,
    input_dir: Path,
    output_dir: Path,
    python_bin: str,
    colmap_bin: str,
    no_gpu: bool = False,
) -> Path:
    pgsr_root = REPOSITORIES["PGSR"]["path"]
    workspace_scene = WORKSPACE_ROOT / "pgsr" / scene_name
    synthetic_scene = (input_dir / "transforms_train.json").exists()
    if synthetic_scene:
        if workspace_scene.exists():
            shutil.rmtree(workspace_scene)
        shutil.copytree(input_dir, workspace_scene)
        images_dir = workspace_scene / "images"
        if not images_dir.exists():
            raise RuntimeError(f"Synthetic PGSR scene is missing {images_dir}")
    else:
        ensure_dir(workspace_scene.parent)
        if workspace_scene.exists():
            shutil.rmtree(workspace_scene)
        ensure_dir(workspace_scene / "input")
        for image in list_sorted_files(input_dir, [".png", ".jpg", ".jpeg"]):
            shutil.copy2(image, workspace_scene / "input" / image.name)

        if no_gpu:
            convert_cmd = [
                python_bin,
                "scripts/preprocess/convert.py",
                "--data_path",
                str(workspace_scene.parent),
                "--colmap_executable",
                colmap_bin,
                "--no_gpu",
            ]
        else:
            convert_cmd = [
                python_bin,
                "scripts/preprocess/convert.py",
                "--data_path",
                str(workspace_scene.parent),
                "--colmap_executable",
                colmap_bin,
            ]
        run_command(convert_cmd, cwd=pgsr_root, env={"QT_QPA_PLATFORM": "offscreen"})
        sparse_root = workspace_scene / "sparse"
        images_dir = workspace_scene / "images"
        sparse_images = sparse_root / "images.bin"
        sparse_cameras = sparse_root / "cameras.bin"
        sparse_points = sparse_root / "points3D.bin"
        if not images_dir.exists() or not sparse_images.exists() or not sparse_cameras.exists() or not sparse_points.exists():
            raise RuntimeError(
                "PGSR preprocess did not produce a valid COLMAP scene. "
                f"Expected {images_dir}, {sparse_images}, {sparse_cameras}, and {sparse_points}. "
                "If COLMAP is running over SSH or without a display, try QT_QPA_PLATFORM=offscreen."
            )
    run_command(
        [python_bin, "train.py", "-s", str(workspace_scene), "-m", str(output_dir), "--max_abs_split_points", "0", "--opacity_cull_threshold", "0.05"],
        cwd=pgsr_root,
    )
    return workspace_scene / "images"


def run_grid4d(source_dir: Path, output_dir: Path, conf_path: Path, python_bin: str) -> None:
    grid4d_root = REPOSITORIES["Grid4D"]["path"]
    run_command([python_bin, "train.py", "-s", str(source_dir), "-m", str(output_dir), "--conf", str(conf_path)], cwd=grid4d_root)


def run_splatt3r(image_dir: Path, output_ply: Path, python_bin: str) -> list[Path]:
    images = list_sorted_files(image_dir, [".png", ".jpg", ".jpeg"])
    if not images:
        raise FileNotFoundError(f"No images found under {image_dir}")
    cmd = [python_bin, str(THIS_DIR / "run_splatt3r_infer.py"), "--output", str(output_ply)]
    for image in images[:2]:
        cmd.extend(["--image", str(image)])
    run_command(cmd, cwd=REPO_ROOT)
    return images[:2]


def prepare_multiview_input(scene_name: str, input_dir: Path, pacnerf_time: float | None, pacnerf_time_index: int | None) -> Path:
    pacnerf_scene_dir = detect_pacnerf_scene_dir(input_dir, scene_name)
    if pacnerf_scene_dir is None:
        return input_dir

    prepared_dir = WORKSPACE_ROOT / "pacnerf_multiview" / scene_name / "pgsr_scene"
    _, metadata = prepare_pacnerf_pgsr_scene(
        scene_dir=pacnerf_scene_dir,
        output_dir=prepared_dir,
        time_value=pacnerf_time,
        time_index=pacnerf_time_index,
    )
    print(
        f"Detected PAC-NeRF dataset input under {pacnerf_scene_dir}. "
        f"Using time index {metadata['selected_time_index']} (time={metadata['selected_time']}) "
        f"with {metadata['num_views']} views and preparing a PGSR synthetic scene."
    )
    return prepared_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplemental end-to-end PhysFlow pipeline.")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--mode", choices=["multiview", "video", "single_image"], required=True)
    parser.add_argument("--reconstruction-method", choices=["pgsr", "grid4d", "splatt3r"], required=True)
    parser.add_argument("--input-dir", type=Path, default=None, help="Images, frames, or prepared dataset folder depending on method.")
    parser.add_argument("--pacnerf-time", type=float, default=None, help="When input-dir points to a PAC-NeRF scene, select this time value.")
    parser.add_argument("--pacnerf-time-index", type=int, default=None, help="When input-dir points to a PAC-NeRF scene, select this sorted time index.")
    parser.add_argument("--grid4d-conf", type=Path, default=None)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument("--prepare-video-workspace", action="store_true")
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--camera-model", default="OPENCV")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU use for COLMAP preprocessing.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--target-fps", type=float, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument("--sam2-prompt-type", choices=["point", "box", "mask"], default=None)
    parser.add_argument("--sam2-point", nargs=2, type=float, default=None)
    parser.add_argument("--sam2-box", nargs=4, type=float, default=None)
    parser.add_argument("--video-prompt", default=None)
    parser.add_argument("--material-json", type=Path, default=None)
    parser.add_argument("--material-image", type=Path, default=None)
    parser.add_argument(
        "--material-backend",
        choices=["openai_compatible", "hf_local"],
        default=None,
    )
    parser.add_argument("--material-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--material-api-base", default="https://api.openai.com/v1")
    parser.add_argument("--material-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--material-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--material-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--openai-material", action="store_true")
    parser.add_argument("--n-epoch", type=int, default=50)
    parser.add_argument("--n-key-frame", type=int, default=13)
    parser.add_argument("--pgsr-python", default=sys.executable)
    parser.add_argument("--grid4d-python", default=sys.executable)
    parser.add_argument("--splatt3r-python", default=sys.executable)
    parser.add_argument("--sam2point-python", default=sys.executable)
    parser.add_argument("--physflow-python", default=sys.executable)
    args = parser.parse_args()

    if args.reconstruction_method in {"pgsr", "splatt3r"} and not args.input_dir:
        raise ValueError(f"--input-dir is required for {args.reconstruction_method}.")
    if args.reconstruction_method == "grid4d":
        if args.mode == "video":
            if not args.input_dir and not args.video_path:
                raise ValueError("Provide --video-path or --input-dir for the dynamic video Grid4D route.")
        elif not args.input_dir:
            raise ValueError("--input-dir is required for Grid4D outside video mode.")
    if args.pacnerf_time is not None and args.pacnerf_time_index is not None:
        raise ValueError("Specify only one of --pacnerf-time or --pacnerf-time-index.")

    scene_dir, output_config = build_scene_paths(args.scene_name)
    recon_output = WORKSPACE_ROOT / "reconstruction" / args.scene_name
    ensure_dir(recon_output.parent)

    if args.reconstruction_method == "pgsr":
        pgsr_input_dir = prepare_multiview_input(args.scene_name, args.input_dir, args.pacnerf_time, args.pacnerf_time_index)
        images_dir = run_pgsr(
            args.scene_name,
            pgsr_input_dir,
            recon_output,
            args.pgsr_python,
            args.colmap_bin,
            no_gpu=args.no_gpu,
        )
        run_command(
            [
                args.physflow_python,
                str(THIS_DIR / "convert_to_physflow_scene.py"),
                "--source-type",
                "pgsr",
                "--source-dir",
                str(recon_output),
                "--scene-dir",
                str(scene_dir),
                "--images-dir",
                str(images_dir),
            ],
            cwd=REPO_ROOT,
        )
    elif args.reconstruction_method == "grid4d":
        grid4d_source = args.input_dir
        if args.mode == "video" and (args.video_path or args.prepare_video_workspace):
            prep_cmd = [
                args.physflow_python,
                str(THIS_DIR / "prepare_dynamic_video_workspace.py"),
                "--scene-name",
                args.scene_name,
                "--extract-python",
                args.physflow_python,
                "--colmap-python",
                args.physflow_python,
                "--colmap-bin",
                args.colmap_bin,
                "--camera-model",
                args.camera_model,
                "--frame-stride",
                str(args.frame_stride),
            ]
            if args.no_gpu:
                prep_cmd.append("--no-gpu")
            if args.video_path:
                prep_cmd.extend(["--video", str(args.video_path)])
            else:
                prep_cmd.extend(["--frames-dir", str(args.input_dir)])
            if args.max_frames is not None:
                prep_cmd.extend(["--max-frames", str(args.max_frames)])
            if args.target_fps is not None:
                prep_cmd.extend(["--target-fps", str(args.target_fps)])
            if args.resize_width and args.resize_height:
                prep_cmd.extend(["--resize-width", str(args.resize_width), "--resize-height", str(args.resize_height)])
            run_command(prep_cmd, cwd=REPO_ROOT)
            grid4d_source = WORKSPACE_ROOT / "grid4d_video" / args.scene_name

        grid4d_cmd_conf = args.grid4d_conf
        run_grid4d(grid4d_source, recon_output, grid4d_cmd_conf, args.grid4d_python) if grid4d_cmd_conf else run_command(
            [args.grid4d_python, "train.py", "-s", str(grid4d_source), "-m", str(recon_output)],
            cwd=REPOSITORIES["Grid4D"]["path"],
        )
        run_command(
            [
                args.physflow_python,
                str(THIS_DIR / "convert_to_physflow_scene.py"),
                "--source-type",
                "grid4d",
                "--source-dir",
                str(recon_output),
                "--scene-dir",
                str(scene_dir),
                "--images-dir",
                str(grid4d_source / "images"),
            ],
            cwd=REPO_ROOT,
        )
        guidance_src = grid4d_source / "guidance_frames"
        if not guidance_src.exists():
            guidance_src = grid4d_source / "images"
        if guidance_src.exists():
            copy_guidance_frames(guidance_src, scene_dir / "images_generated")
    else:
        output_ply = WORKSPACE_ROOT / "splatt3r" / args.scene_name / "gaussians.ply"
        ensure_dir(output_ply.parent)
        run_splatt3r(args.input_dir, output_ply, args.splatt3r_python)
        run_command(
            [
                args.physflow_python,
                str(THIS_DIR / "convert_to_physflow_scene.py"),
                "--source-type",
                "splatt3r",
                "--source-dir",
                str(output_ply),
                "--gaussian-ply",
                str(output_ply),
                "--scene-dir",
                str(scene_dir),
                "--images-dir",
                str(args.input_dir),
            ],
            cwd=REPO_ROOT,
        )

    if args.sam2_prompt_type:
        masks_dir = scene_dir / "masks_generated"
        mask_cmd = [
            args.sam2point_python,
            str(THIS_DIR / "run_sam2_masks.py"),
            "--frames-dir",
            str(scene_dir / "images"),
            "--output-dir",
            str(masks_dir),
            "--prompt-type",
            args.sam2_prompt_type,
        ]
        if args.sam2_prompt_type in {"point", "mask"}:
            if not args.sam2_point:
                raise ValueError("--sam2-point is required for point/mask prompt.")
            mask_cmd.extend(["--point", str(args.sam2_point[0]), str(args.sam2_point[1])])
        else:
            if not args.sam2_box:
                raise ValueError("--sam2-box is required for box prompt.")
            mask_cmd.extend(["--box", *(str(v) for v in args.sam2_box)])
        run_command(mask_cmd, cwd=REPO_ROOT)
        run_command(
            [
                args.physflow_python,
                str(THIS_DIR / "project_masks_to_gaussians.py"),
                "--scene-dir",
                str(scene_dir),
                "--masks-dir",
                str(masks_dir),
            ],
            cwd=REPO_ROOT,
        )

    if args.video_prompt:
        image_candidates = list_sorted_files(scene_dir / "images", [".png", ".jpg", ".jpeg"])
        if not image_candidates:
            raise FileNotFoundError(f"No source image available under {scene_dir / 'images'}")
        run_command(
            [
                args.physflow_python,
                str(THIS_DIR / "run_cogvideo_cli.py"),
                "--image",
                str(image_candidates[0]),
                "--prompt",
                args.video_prompt,
                "--output-dir",
                str(scene_dir / "images_generated"),
            ],
            cwd=REPO_ROOT,
        )

    material_json = args.material_json or (WORKSPACE_ROOT / "materials" / f"{args.scene_name}.json")
    material_backend = args.material_backend
    if args.openai_material and material_backend is None:
        material_backend = "openai_compatible"

    if material_backend:
        if args.material_image:
            material_image = args.material_image
        else:
            material_images = list_sorted_files(scene_dir / "images", [".png", ".jpg", ".jpeg"])
            if not material_images:
                raise FileNotFoundError(f"No source image available under {scene_dir / 'images'} for material initialization")
            material_image = material_images[0]
        ensure_dir(material_json.parent)
        run_command(
            [
                args.physflow_python,
                str(THIS_DIR / "query_material.py"),
                "--image",
                str(material_image),
                "--output",
                str(material_json),
                "--backend",
                material_backend,
                "--model",
                args.material_model,
                "--api-base",
                args.material_api_base,
                "--api-key-env",
                args.material_api_key_env,
                "--device",
                args.material_device,
                "--dtype",
                args.material_dtype,
            ],
            cwd=REPO_ROOT,
        )

    if material_json.exists():
        run_command(
            [
                args.physflow_python,
                str(THIS_DIR / "create_material_config.py"),
                "--material-json",
                str(material_json),
                "--output-config",
                str(output_config),
            ],
            cwd=REPO_ROOT,
        )

    if output_config.exists() and (scene_dir / "images_generated").exists():
        run_command(
            [
                args.physflow_python,
                "simulation.py",
                "--model_path",
                str(scene_dir),
                "--output_path",
                str(REPO_ROOT / "outputs" / "custom" / args.scene_name),
                "--physics_config",
                str(output_config),
                "--n_epoch",
                str(args.n_epoch),
                "--n_key_frame",
                str(args.n_key_frame),
            ],
            cwd=REPO_ROOT,
        )
    else:
        print("Skipped simulation because config or images_generated is missing.")


if __name__ == "__main__":
    main()
