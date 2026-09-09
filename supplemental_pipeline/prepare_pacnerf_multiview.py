#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import copy_or_link, ensure_dir, write_json


PACNERF_JSON = "all_data.json"
PACNERF_IMAGE_DIR = "data"
TIME_TOL = 1e-8


def is_pacnerf_scene_dir(path: Path) -> bool:
    return (path / PACNERF_JSON).is_file() and (path / PACNERF_IMAGE_DIR).is_dir()


def detect_pacnerf_scene_dir(input_dir: Path, scene_name: str) -> Path | None:
    if is_pacnerf_scene_dir(input_dir):
        return input_dir

    candidate = input_dir / scene_name
    if is_pacnerf_scene_dir(candidate):
        return candidate

    return None


def load_entries(scene_dir: Path) -> list[dict]:
    with (scene_dir / PACNERF_JSON).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{scene_dir / PACNERF_JSON} must contain a list of entries")
    return data


def unique_times(entries: list[dict]) -> list[float]:
    return sorted({float(entry["time"]) for entry in entries})


def choose_time_value(entries: list[dict], time_value: float | None, time_index: int | None) -> tuple[float, int]:
    times = unique_times(entries)
    if not times:
        raise ValueError("PAC-NeRF scene contains no time entries")
    if time_value is not None and time_index is not None:
        raise ValueError("Specify only one of --time or --time-index")

    if time_index is not None:
        if time_index < 0 or time_index >= len(times):
            raise IndexError(f"time-index {time_index} is out of range for {len(times)} time steps")
        return times[time_index], time_index

    if time_value is None:
        nearest = min(range(len(times)), key=lambda idx: abs(times[idx]))
        return times[nearest], nearest

    nearest = min(range(len(times)), key=lambda idx: abs(times[idx] - time_value))
    if abs(times[nearest] - time_value) > TIME_TOL:
        print(f"Warning: requested time {time_value} not found exactly; using nearest available time {times[nearest]}")
    return times[nearest], nearest


def parse_view_index(file_path: str) -> tuple[int, str]:
    stem = Path(file_path).stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "r":
        try:
            return int(parts[1]), stem
        except ValueError:
            pass
    return 10**9, stem


def select_entries_for_time(entries: list[dict], selected_time: float) -> list[dict]:
    matches = [entry for entry in entries if math.isclose(float(entry["time"]), selected_time, abs_tol=TIME_TOL)]
    if not matches:
        raise RuntimeError(f"No PAC-NeRF frames found for time {selected_time}")
    return sorted(matches, key=lambda entry: parse_view_index(str(entry["file_path"])))


def prepare_pacnerf_multiview_scene(
    scene_dir: Path,
    output_dir: Path,
    time_value: float | None = None,
    time_index: int | None = None,
    link: bool = False,
) -> tuple[Path, dict]:
    entries = load_entries(scene_dir)
    selected_time, selected_index = choose_time_value(entries, time_value, time_index)
    selected_entries = select_entries_for_time(entries, selected_time)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    prepared_entries = []
    for entry in selected_entries:
        relative_path = Path(str(entry["file_path"]).replace("./", "", 1))
        src = scene_dir / relative_path
        if not src.is_file():
            raise FileNotFoundError(f"Referenced PAC-NeRF image does not exist: {src}")
        dst = output_dir / src.name
        copy_or_link(src, dst, link=link)
        prepared_entries.append(
            {
                "image_name": dst.name,
                "source_file": str(src),
                "time": float(entry["time"]),
                "c2w": entry.get("c2w"),
                "intrinsic": entry.get("intrinsic"),
            }
        )

    metadata = {
        "source_scene_dir": str(scene_dir),
        "selected_time": selected_time,
        "selected_time_index": selected_index,
        "available_times": unique_times(entries),
        "num_views": len(prepared_entries),
        "entries": prepared_entries,
    }
    write_json(output_dir / "pacnerf_import_metadata.json", metadata)
    return output_dir, metadata


def read_image_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as f:
        header = f.read(24)

    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        width, height = struct.unpack(">II", header[16:24])
        return width, height

    raise ValueError(f"Unsupported image format for size detection: {image_path}")


def build_transform_matrix(c2w_rows: list[list[float]]) -> list[list[float]]:
    if len(c2w_rows) == 4:
        return [[float(v) for v in row] for row in c2w_rows]
    if len(c2w_rows) == 3 and all(len(row) == 4 for row in c2w_rows):
        matrix = [[float(v) for v in row] for row in c2w_rows]
        matrix.append([0.0, 0.0, 0.0, 1.0])
        return matrix
    raise ValueError("PAC-NeRF c2w must be either 3x4 or 4x4")


def build_transforms_payload(selected_entries: list[dict]) -> dict:
    if not selected_entries:
        raise ValueError("selected_entries must not be empty")

    first_image = selected_entries[0]
    image_path = Path(first_image["source_file"])
    width, height = read_image_size(image_path)

    intrinsic = first_image["intrinsic"]
    fx = float(intrinsic[0][0])
    fy = float(intrinsic[1][1])
    cx = float(intrinsic[0][2])
    cy = float(intrinsic[1][2])
    camera_angle_x = 2.0 * math.atan(width / (2.0 * fx))
    camera_angle_y = 2.0 * math.atan(height / (2.0 * fy))

    for entry in selected_entries[1:]:
        cur_intrinsic = entry["intrinsic"]
        if (
            float(cur_intrinsic[0][0]) != fx
            or float(cur_intrinsic[1][1]) != fy
            or float(cur_intrinsic[0][2]) != cx
            or float(cur_intrinsic[1][2]) != cy
        ):
            raise ValueError("PGSR synthetic scene adapter currently requires shared intrinsics across all selected views")

        cur_width, cur_height = read_image_size(Path(entry["source_file"]))
        if cur_width != width or cur_height != height:
            raise ValueError("PGSR synthetic scene adapter currently requires all selected views to share the same image size")

    frames = []
    for entry in selected_entries:
        relative_no_ext = f"./images/{Path(entry['image_name']).stem}"
        c2w = build_transform_matrix(entry["c2w"])
        frames.append(
            {
                "file_path": relative_no_ext,
                "transform_matrix": c2w,
            }
        )

    return {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "w": width,
        "h": height,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "frames": frames,
    }


def prepare_pacnerf_pgsr_scene(
    scene_dir: Path,
    output_dir: Path,
    time_value: float | None = None,
    time_index: int | None = None,
    link: bool = False,
) -> tuple[Path, dict]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    images_dir = output_dir / "images"
    _, metadata = prepare_pacnerf_multiview_scene(
        scene_dir=scene_dir,
        output_dir=images_dir,
        time_value=time_value,
        time_index=time_index,
        link=link,
    )

    transforms = build_transforms_payload(metadata["entries"])
    write_json(output_dir / "transforms_train.json", transforms)
    write_json(output_dir / "transforms_test.json", {"camera_angle_x": transforms["camera_angle_x"], "frames": []})
    metadata["prepared_pgsr_scene"] = str(output_dir)
    write_json(output_dir / "pacnerf_import_metadata.json", metadata)
    return output_dir, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one multiview time slice from a PAC-NeRF scene.")
    parser.add_argument("--scene-dir", type=Path, required=True, help="Path like PAC_NeRF_data/torus containing all_data.json and data/.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time", type=float, default=None, help="Target time value from all_data.json. Defaults to the time closest to 0.")
    parser.add_argument("--time-index", type=int, default=None, help="Index into the sorted unique time list.")
    parser.add_argument("--link", action="store_true", help="Symlink source images instead of copying them.")
    parser.add_argument("--pgsr-scene", action="store_true", help="Write a PGSR-ready synthetic scene with transforms_train/test.json.")
    args = parser.parse_args()

    if args.pgsr_scene:
        output_dir, metadata = prepare_pacnerf_pgsr_scene(
            scene_dir=args.scene_dir,
            output_dir=args.output_dir,
            time_value=args.time,
            time_index=args.time_index,
            link=args.link,
        )
    else:
        output_dir, metadata = prepare_pacnerf_multiview_scene(
            scene_dir=args.scene_dir,
            output_dir=args.output_dir,
            time_value=args.time,
            time_index=args.time_index,
            link=args.link,
        )
    print(
        f"Prepared {metadata['num_views']} PAC-NeRF views from time index {metadata['selected_time_index']} "
        f"(time={metadata['selected_time']}) into {output_dir}"
    )


if __name__ == "__main__":
    main()
