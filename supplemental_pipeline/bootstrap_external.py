#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import EXTERNAL_ROOT, REPOSITORIES, download_github_tarball, ensure_dir, run_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Download external repos used by the PhysFlow supplemental pipeline.")
    parser.add_argument(
        "--repos",
        nargs="*",
        choices=sorted(REPOSITORIES.keys()),
        default=sorted(REPOSITORIES.keys()),
        help="Subset of external repos to download.",
    )
    parser.add_argument(
        "--download-sam2-checkpoints",
        action="store_true",
        help="Run SAM2Point's checkpoint downloader after the source code is present.",
    )
    args = parser.parse_args()

    ensure_dir(EXTERNAL_ROOT)
    for name in args.repos:
        spec = REPOSITORIES[name]
        download_github_tarball(spec["repo"], spec["branch"], spec["path"])

    if args.download_sam2_checkpoints:
        sam2_root = REPOSITORIES["SAM2Point"]["path"]
        run_command(["bash", "download_ckpts.sh"], cwd=sam2_root / "checkpoints")


if __name__ == "__main__":
    main()
