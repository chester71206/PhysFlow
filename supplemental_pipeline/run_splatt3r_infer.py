#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import REPOSITORIES, ensure_dir, list_sorted_files


def load_splatt3r_modules(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src/mast3r_src"))
    sys.path.insert(0, str(repo_root / "src/mast3r_src/dust3r"))
    sys.path.insert(0, str(repo_root / "src/pixelsplat_src"))
    from dust3r.utils.image import load_images
    import main as splatt3r_main
    import utils.export as splatt3r_export
    return load_images, splatt3r_main, splatt3r_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Splatt3R inference and export a Gaussian PLY.")
    parser.add_argument("--image", dest="images", action="append", help="Input image. Provide one or two times.", required=False)
    parser.add_argument("--image-dir", type=Path, help="Directory to auto-pick one or two images from.", default=None)
    parser.add_argument("--output", type=Path, required=True, help="Output PLY path.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--repo-id", default="brandonsmart/splatt3r_v1.0")
    parser.add_argument("--checkpoint", default="epoch=19-step=1200.ckpt")
    args = parser.parse_args()

    image_paths: list[Path] = []
    if args.images:
        image_paths.extend(Path(p) for p in args.images)
    if args.image_dir:
        picked = list_sorted_files(args.image_dir, [".png", ".jpg", ".jpeg"])
        image_paths.extend(picked[:2])
    if not image_paths:
        raise ValueError("Provide --image or --image-dir.")
    if len(image_paths) > 2:
        raise ValueError("Splatt3R wrapper only supports one or two images.")
    if len(image_paths) == 1:
        image_paths = [image_paths[0], image_paths[0]]

    repo_root = REPOSITORIES["splatt3r"]["path"]
    load_images, splatt3r_main, splatt3r_export = load_splatt3r_modules(repo_root)

    weights_path = hf_hub_download(repo_id=args.repo_id, filename=args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = splatt3r_main.MAST3RGaussians.load_from_checkpoint(weights_path, device)

    imgs = load_images([str(p) for p in image_paths], size=args.image_size, verbose=True)
    for img in imgs:
        img["img"] = img["img"].to(device)
        img["original_img"] = img["original_img"].to(device)
        img["true_shape"] = torch.from_numpy(img["true_shape"])

    pred1, pred2 = model(imgs[0], imgs[1])
    ensure_dir(args.output.parent)
    splatt3r_export.save_as_ply(pred1, pred2, str(args.output))
    print(f"Saved Splatt3R Gaussian PLY to {args.output}")


if __name__ == "__main__":
    main()
