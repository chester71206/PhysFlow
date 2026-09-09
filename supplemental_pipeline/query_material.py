#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import write_json

PROMPT = (
    "Identify the object and infer the closest material family for PhysFlow. "
    "Supported material_type values: elastic, plasticine, metal, foam, sand, newtonian, non_newtonian. "
    "Return JSON only, with a required string field named material_type. "
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


def validate_model_choice(model_name: str) -> None:
    lowered = model_name.lower()
    if "qwen3-8b-instruct" in lowered and "vl" not in lowered:
        raise ValueError(
            "Qwen3-8B-Instruct is a text-only model and cannot replace image-based material initialization. "
            "Use a vision-language checkpoint such as Qwen/Qwen3-VL-8B-Instruct "
            "or Qwen/Qwen2.5-VL-7B-Instruct."
        )


def request_openai_compatible(
    *,
    image_path: Path,
    model_name: str,
    prompt: str,
    api_base: str,
    api_key: str,
) -> dict:
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    data_uri = f"data:{mime};base64," + base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        data = json.load(response)

    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return normalize_material_payload(json.loads(content))


def resolve_torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def ensure_torch_transformers_compat() -> None:
    import torch

    # transformers 4.57 may call torch.compiler.is_compiling(), but older
    # torch builds in this environment do not expose that helper yet.
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and not hasattr(compiler, "is_compiling"):
        compiler.is_compiling = lambda: False


def extract_generated_text(output: object) -> str:
    if isinstance(output, list):
        return "".join(
            part.get("text", "")
            for part in output
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    if isinstance(output, str):
        return output.strip()
    raise TypeError(f"Unsupported generated output type: {type(output)!r}")


def request_hf_local(
    *,
    image_path: Path,
    model_name: str,
    prompt: str,
    device: str,
    dtype_name: str,
    max_new_tokens: int,
    offload_folder: Path | None,
) -> dict:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed in the selected Python environment."
        ) from exc

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is not installed in the selected Python environment.") from exc

    ensure_torch_transformers_compat()
    torch_dtype = resolve_torch_dtype(dtype_name)
    pipe_kwargs = {
        "task": "image-text-to-text",
        "model": model_name,
        "trust_remote_code": True,
        "dtype": torch_dtype,
    }
    if device == "auto":
        pipe_kwargs["device_map"] = "auto"
        if offload_folder is not None:
            pipe_kwargs["offload_folder"] = str(offload_folder)
    elif device == "cpu":
        pipe_kwargs["device"] = -1
        if torch_dtype in {torch.float16, torch.bfloat16}:
            pipe_kwargs["dtype"] = torch.float32
    elif device == "cuda":
        pipe_kwargs["device"] = 0
    else:
        raise ValueError(f"Unsupported device: {device}")

    try:
        pipe = pipeline(**pipe_kwargs)
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
            raise RuntimeError(
                "Local VLM initialization ran out of GPU memory. "
                "Try --device auto to enable device_map=auto offload, "
                "--dtype bfloat16 or --dtype float16 if your GPU stack supports it, "
                "or switch to a smaller checkpoint such as Qwen/Qwen2.5-VL-3B-Instruct."
            ) from exc
        raise RuntimeError(
            "Failed to initialize the local multimodal pipeline. "
            "If you are loading a recent Qwen VL checkpoint, upgrade transformers first, "
            "for example: pip install -U 'transformers>=4.57' accelerate"
        ) from exc

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        outputs = pipe(text=messages, max_new_tokens=max_new_tokens, return_full_text=False)
    except RuntimeError as exc:
        message = str(exc)
        if "unable to find an engine to execute this computation" in message.lower():
            raise RuntimeError(
                "Qwen VL local inference failed inside the visual Conv3d path. "
                "This is usually a GPU/kernel compatibility issue with the current torch/CUDA/cuDNN stack. "
                "Try rerunning with --dtype float32 first. "
                "If that still fails, use an OpenAI-compatible Qwen server or switch to a smaller fallback such as "
                "Qwen/Qwen2.5-VL-3B-Instruct."
            ) from exc
        raise
    generated = outputs[0]["generated_text"]
    return normalize_material_payload(json.loads(extract_generated_text(generated)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a VLM for PhysFlow material initialization.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=["openai_compatible", "hf_local"],
        default="hf_local",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--offload-folder", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate_model_choice(args.model)

    if args.dry_run:
        print(args.prompt)
        return

    if args.backend == "openai_compatible":
        api_key = os.environ.get(args.api_key_env, "EMPTY")
        material_json = request_openai_compatible(
            image_path=args.image,
            model_name=args.model,
            prompt=args.prompt,
            api_base=args.api_base,
            api_key=api_key,
        )
    else:
        material_json = request_hf_local(
            image_path=args.image,
            model_name=args.model,
            prompt=args.prompt,
            device=args.device,
            dtype_name=args.dtype,
            max_new_tokens=args.max_new_tokens,
            offload_folder=args.offload_folder,
        )

    write_json(args.output, material_json)
    print(f"Saved material initialization to {args.output}")


if __name__ == "__main__":
    main()
