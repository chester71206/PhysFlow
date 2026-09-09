#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import write_json

PROMPT = (
    "Identify the main object in this image and infer the closest PhysFlow material family. "
    "Supported material_type values: elastic, plasticine, metal, foam, sand, newtonian, non_newtonian. "
    "Return JSON only with a required string field named material_type. "
    "Also include object_name and density when possible. "
    "Use E and nu for elastic; "
    "E, nu, yield_stress for plasticine and metal; "
    "E, nu, yield_stress, plastic_viscosity for foam; "
    "friction_angle for sand; "
    "mu and kappa for newtonian; "
    "mu, kappa, yield_stress, plastic_viscosity for non_newtonian. "
    "Use those exact JSON key names. "
    "Do not wrap the JSON in markdown."
)


def normalize_material_payload(payload: dict) -> dict:
    normalized = dict(payload)
    if "material_type" not in normalized and "material" in normalized:
        normalized["material_type"] = normalized.pop("material")
    if "material_type" not in normalized:
        raise ValueError(
            "Model response did not include 'material_type'. "
            f"Received keys: {sorted(normalized.keys())}"
        )
    normalized["material_type"] = str(normalized["material_type"]).strip()
    return normalized


def extract_json_text(raw_text: str) -> str:
    stripped = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    start = stripped.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model response: {stripped[:200]!r}")

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(stripped[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start:index + 1]

    raise ValueError(f"Unterminated JSON object in model response: {stripped[:200]!r}")


def resolve_dtype(dtype_name: str):
    import torch

    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def generate_material_json(
    *,
    image_path: Path,
    model_name: str,
    prompt: str,
    dtype_name: str,
    attn_implementation: str,
    max_new_tokens: int,
    offload_folder: Path | None,
) -> dict:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch_dtype = resolve_dtype(dtype_name)
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }
    if offload_folder is not None:
        model_kwargs["offload_folder"] = str(offload_folder)

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    payload = json.loads(extract_json_text(output_text))
    return normalize_material_payload(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL material initialization in an isolated environment.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--offload-folder", type=Path, default=None)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(args.prompt)
        return

    material_json = generate_material_json(
        image_path=args.image,
        model_name=args.model,
        prompt=args.prompt,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        offload_folder=args.offload_folder,
    )
    write_json(args.output, material_json)
    print(f"Saved material initialization to {args.output}")


if __name__ == "__main__":
    main()
