from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable


SUPPLEMENTAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SUPPLEMENTAL_ROOT.parent
EXTERNAL_ROOT = SUPPLEMENTAL_ROOT / "external"
WORKSPACE_ROOT = SUPPLEMENTAL_ROOT / "workspaces"

REPOSITORIES = {
    "PGSR": {
        "repo": "zju3dv/PGSR",
        "branch": "main",
        "path": EXTERNAL_ROOT / "PGSR",
    },
    "Grid4D": {
        "repo": "JiaweiXu8/Grid4D",
        "branch": "main",
        "path": EXTERNAL_ROOT / "Grid4D",
    },
    "splatt3r": {
        "repo": "btsmart/splatt3r",
        "branch": "main",
        "path": EXTERNAL_ROOT / "splatt3r",
    },
    "SAM2Point": {
        "repo": "ZiyuGuo99/SAM2Point",
        "branch": "main",
        "path": EXTERNAL_ROOT / "SAM2Point",
    },
}

MATERIAL_CONFIG_TEMPLATES = {
    "elastic": REPO_ROOT / "config/pacnerf/torus_config.json",
    "plasticine": REPO_ROOT / "config/pacnerf/playdoh_config.json",
    "metal": REPO_ROOT / "config/realworld/plane_960_config.json",
    "foam": REPO_ROOT / "config/phys_dreamer/hat_config.json",
    "sand": REPO_ROOT / "config/pacnerf/trophy_config.json",
    "newtonian": REPO_ROOT / "config/pacnerf/droplet_config.json",
    "non_newtonian": REPO_ROOT / "config/pacnerf/toothpaste_config.json",
}

MATERIAL_ALIASES = {
    "jelly": "elastic",
    "plastic": "plasticine",
    "plasticine": "plasticine",
    "metal": "metal",
    "foam": "foam",
    "sand": "sand",
    "fluid": "newtonian",
    "newtonian_fluid": "newtonian",
    "newtonian": "newtonian",
    "non_newtonian_fluid": "non_newtonian",
    "non-newtonian": "non_newtonian",
    "non_newtonian": "non_newtonian",
    "elastic": "elastic",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def run_command(command: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=full_env, check=True)


def copy_or_link(src: Path, dst: Path, link: bool = False) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if link:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
        return

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def list_sorted_files(path: Path, suffixes: Iterable[str]) -> list[Path]:
    allowed = {s.lower() for s in suffixes}
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in allowed
    )


def normalize_material_type(material_type: str) -> str:
    key = material_type.strip().lower().replace(" ", "_")
    if key not in MATERIAL_ALIASES:
        raise ValueError(f"Unsupported material type: {material_type}")
    return MATERIAL_ALIASES[key]


def download_github_tarball(repo: str, branch: str, destination: Path) -> Path:
    if destination.exists():
        return destination

    ensure_dir(destination.parent)
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
    archive_name = f"{destination.name}-{branch}.tar.gz"
    archive_path = destination.parent / archive_name
    extracted_root = destination.parent / f"{destination.name}-{branch}"

    print(f"Downloading {repo}@{branch} -> {destination}")
    with urllib.request.urlopen(url) as response, archive_path.open("wb") as f:
        f.write(response.read())

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(destination.parent)

    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    shutil.move(str(extracted_root), str(destination))
    archive_path.unlink()
    return destination
