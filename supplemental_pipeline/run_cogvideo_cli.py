#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI wrapper around CogVideoX image-to-video generation for PhysFlow.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="THUDM/CogVideoX-5b-I2V")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seconds", type=int, default=6)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true", help="Do not clear output-dir before writing frames.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    num_frames = args.seconds * args.fps + 1
    image = load_image(image=str(args.image))
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if generator_device == "cuda" else torch.float32
    if generator_device != "cuda":
        print("Warning: CUDA is not available; CogVideo generation may be very slow or unsupported on this machine.")

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model_id, torch_dtype=torch_dtype)
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=image,
        num_videos_per_prompt=1,
        num_inference_steps=args.num_inference_steps,
        num_frames=num_frames,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=generator_device).manual_seed(seed),
    ).frames[0]

    width, height = image.width, image.height
    resized = []
    for idx, frame in enumerate(video):
        frame = frame.resize((width, height))
        resized.append(frame)
        frame.save(output_dir / f"frame_{idx:05d}.png")

    export_to_video(resized, str(output_dir / f"cogvideo_seed{seed}.mp4"), fps=args.fps)
    print(f"Saved {len(resized)} frames to {output_dir}")


if __name__ == "__main__":
    main()
