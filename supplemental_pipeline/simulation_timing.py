"""CPU-only preflight: one source of truth for images, time and geometry IDs."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from supplemental_pipeline.aligned_timing import keyframe_steps

ROOT = Path(__file__).resolve().parents[1]


def natural_key(name):
    return [int(p) if re.fullmatch(r"-?\d+", p) else p for p in re.split(r"(-?\d+)", name)]


def image_names(folder):
    return sorted([p.name for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}], key=natural_key)


def selected_indices(count, intervals):
    if intervals < 1 or count < intervals + 1:
        raise ValueError(f"Need at least {intervals + 1} guidance images, found {count}")
    return [round(i * (count - 1) / intervals) for i in range(intervals + 1)]


def pac_id(name):
    match = re.fullmatch(r"r_(-?\d+)_(-?\d+)", Path(name).stem)
    if not match:
        raise ValueError(f"Not a PAC-NeRF image name: {name}")
    return tuple(map(int, match.groups()))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_schedule(model_path, config, n_key_frame, reference_fps=None,
                     reference_times=None, reference_metadata=None):
    model_path = Path(model_path).resolve()
    names = image_names(model_path / "images_generated")
    indices = selected_indices(len(names), n_key_frame)
    timing = config.get("timing", {})
    geometry = timing.get("geometry_frame_ids")
    metadata = Path(reference_metadata).resolve() if reference_metadata else None
    if metadata is None and model_path.parent.name == "pacnerf":
        possible = ROOT / "PAC_NeRF_data" / model_path.name / "all_data.json"
        if not possible.exists():
            possible = ROOT / "data" / "pacnerf" / "timing" / (model_path.name + ".json")
        if possible.exists():
            metadata = possible
    configured_times = timing.get("reference_frame_times", config.get("experiment", {}).get("reference_frame_times"))
    # Explicit CLI source overrides config. Multiple CLI sources are parser errors.
    if reference_times is not None:
        payload = json.loads(Path(reference_times).read_text())
        full_times = [payload[n] for n in names] if isinstance(payload, dict) else payload
        source = {"kind": "timestamps_file", "path": str(Path(reference_times).resolve()), "sha256": sha(reference_times)}
    elif reference_fps is not None:
        if not math.isfinite(reference_fps) or reference_fps <= 0:
            raise ValueError("reference-fps must be positive and finite")
        full_times = [i / reference_fps for i in range(len(names))]
        source = {"kind": "explicit_source_fps", "fps": reference_fps}
    elif reference_metadata is not None or (configured_times is None and timing.get("reference_fps") is None and metadata is not None):
        entries = json.loads(metadata.read_text())
        table = {}
        for entry in entries:
            key = pac_id(entry["file_path"])
            if key in table:
                raise ValueError(f"Duplicate PAC-NeRF metadata identity {key}")
            table[key] = float(entry["time"])
        keys = [pac_id(n) for n in names]
        if len({k[0] for k in keys}) != 1 or len(set(keys)) != len(keys):
            raise ValueError("Guidance must contain one PAC-NeRF view without duplicate frames")
        full_times = [table[k] for k in keys]
        geometry = [k[1] for k in keys]
        source = {"kind": "pacnerf_metadata", "path": str(metadata), "sha256": sha(metadata)}
    elif configured_times is not None:
        full_times = configured_times
        source = {"kind": "config_timestamps"}
    elif timing.get("reference_fps") is not None:
        fps = float(timing["reference_fps"])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("timing.reference_fps must be positive and finite")
        full_times = [i / fps for i in range(len(names))]
        source = {"kind": "config_source_fps", "fps": fps}
    else:
        raise ValueError("No reliable reference timing. For this scene supply --reference-fps (FPS of all images_generated before subsampling), --reference-times JSON, or config timing.reference_frame_times/reference_fps. frame_dt and opt_frame_dt are NOT reference FPS.")
    if len(full_times) != len(names):
        raise ValueError("Provide one timestamp per source guidance image, before keyframe subsampling")
    full_times = [float(t) for t in full_times]
    if any(not math.isfinite(t) for t in full_times) or any(b <= a for a, b in zip(full_times, full_times[1:])):
        raise ValueError("Source timestamps must be finite and strictly increasing")
    times = [full_times[i] - full_times[0] for i in indices]
    dt = float(config.get("substep_dt", 1e-4))
    steps = keyframe_steps(times, dt, n_key_frame)
    # Explicit timestamps still use PAC filename geometry identity, never frame+1.
    if geometry is None and all(re.fullmatch(r"r_\d+_-?\d+", Path(n).stem) for n in names):
        if len({pac_id(n)[0] for n in names}) != 1:
            raise ValueError("Mixed PAC-NeRF views in guidance")
        geometry = [pac_id(n)[1] for n in names]
    if geometry is None:
        if (model_path / "pcd").exists():
            raise ValueError("Non-PAC scene with pcd requires timing.geometry_frame_ids, one per source image")
        geometry = [None] * len(names)
    if len(geometry) != len(names) or any(g is not None and (not isinstance(g, int) or g < 0) for g in geometry):
        raise ValueError("geometry_frame_ids must be nonnegative integers/null, one per source image")
    selected = []
    for i, t, step in zip(indices, times, steps):
        pcd = model_path / "pcd" / f"{geometry[i]}.ply" if geometry[i] is not None else None
        selected.append({"source_index": i, "filename": names[i], "source_time": full_times[i],
                         "target_time": t, "target_step": step, "actual_time": step * dt,
                         "image_sha256": sha(model_path / "images_generated" / names[i]),
                         "geometry_frame_id": geometry[i],
                         "geometry_path": str(pcd) if pcd is not None and pcd.exists() else None})
    render_dt = float(config.get("frame_dt", 1e-2))
    if not math.isfinite(render_dt) or render_dt < dt:
        raise ValueError("frame_dt must be finite and >= substep_dt")
    return {"version": 1, "source": source, "source_count": len(names), "source_filenames": names,
            "substep_dt": dt, "reference_time_origin": full_times[0], "selected_frames": selected,
            "reference_times": times, "target_steps": steps, "render_frame_dt": render_dt,
            "render_note": "Rendering uses config frame_dt; video playback FPS is separate. Training timestamps are fixed across epochs.",
            "gradient_note": "Scene timing only; gradient implementation is selected by the calling trainer."}


def prepare_run(argv=None, per_gaussian=False):
    parser = argparse.ArgumentParser(description="PhysFlow official time-aligned simulation/training entry point")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--physics_config", required=True)
    parser.add_argument("--output_path")
    parser.add_argument("--white_bg", type=lambda x: x.lower() in {"true", "1", "yes"}, default=True)
    for flag in ["output_ply", "output_h5", "debug", "eval"]:
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--downsample", type=float, default=1.)
    parser.add_argument("--n_epoch", type=int, default=10)
    parser.add_argument("--n_key_frame", type=int, default=8)
    parser.add_argument("--guidance-mode", choices=["raft", "raft_cpu", "dummy"], default="raft")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--reference-fps", type=float)
    sources.add_argument("--reference-times", help="JSON list or filename->seconds mapping for ALL source guidance images")
    sources.add_argument("--reference-metadata", help="PAC-NeRF all_data.json override")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after-batches", type=int)
    parser.add_argument("--skip-video", action="store_true")
    if not per_gaussian:
        parser.add_argument('--material-optimizer', choices=['legacy'], default='legacy', help='Compatibility flag for scene initialization; optimization is owned by the AD trainer')
        parser.add_argument('--taichi-memory-gb', type=float, default=2.)
    if per_gaussian:
        parser.add_argument('--E-mode', choices=['per_gaussian', 'shared'], default='per_gaussian')
        parser.add_argument('--E-gradient-normalization', choices=['raw', 'particle_mean'], default='raw')
    args = parser.parse_args(argv)
    if args.n_epoch < 1 or not math.isfinite(args.downsample) or args.downsample <= 0:
        parser.error("n_epoch and downsample must be positive")
    if not per_gaussian and (not math.isfinite(args.taichi_memory_gb) or args.taichi_memory_gb <= 0):
        parser.error("Taichi memory must be positive and finite")
    config = json.loads(Path(args.physics_config).read_text())
    plan = resolve_schedule(args.model_path, config, args.n_key_frame, args.reference_fps,
                            args.reference_times, args.reference_metadata)
    from utils.rollout_objective import RolloutObjective
    objective = RolloutObjective(config)
    if not objective.needs_flow and any(frame['geometry_path'] is None for frame in plan['selected_frames'][1:]):
        raise ValueError('Geometry objective requires a matched PCD for every selected supervised frame')
    background_mode = config.get('guidance_background', 'source')
    if background_mode not in ('source', 'white_masked'):
        raise ValueError('guidance_background must be source or white_masked')
    if background_mode == 'white_masked':
        if not args.white_bg:
            raise ValueError('white_masked guidance requires white_bg=true')
        for frame in plan['selected_frames']:
            name = frame['filename']
            if not re.fullmatch(r'r_\d+_\d+\.(png|jpg|jpeg)', name, re.IGNORECASE):
                raise ValueError('white_masked guidance requires PAC-NeRF frame identities')
            mask = Path(args.model_path) / 'masks' / ('a_' + Path(name).stem[2:] + '.png')
            if not mask.is_file():
                raise ValueError(f'Missing supplied RGBA guidance image: {mask}')
            frame['guidance_rgba_path'] = str(mask.resolve())
            frame['guidance_rgba_sha256'] = sha(mask)
    plan["config_sha256"] = sha(args.physics_config)
    plan["model_path"] = str(Path(args.model_path).resolve())
    if not args.dry_run:
        if args.output_path is None:
            parser.error("Use a separate --output_path; historical runs must be preserved")
        output = Path(args.output_path + f"_ds{args.downsample}_ep{args.n_epoch}")
        if args.eval:
            saved = json.loads((output / "timing_manifest.json").read_text())
            if any(saved.get(k) != plan[k] for k in ["config_sha256", "model_path", "selected_frames"]):
                raise ValueError("Evaluation timing/config/frame mapping differs from the saved run")
        elif output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing run: {output}")
    return args, plan
