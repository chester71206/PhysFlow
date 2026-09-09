#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import MATERIAL_CONFIG_TEMPLATES, normalize_material_type, read_json, write_json


SUPPORTED_FIELDS = {
    "elastic": {"material", "density", "E", "nu"},
    "plasticine": {"material", "density", "E", "nu", "yield_stress"},
    "metal": {"material", "density", "E", "nu", "yield_stress", "hardening", "xi"},
    "foam": {"material", "density", "E", "nu", "yield_stress", "plastic_viscosity"},
    "sand": {"material", "density", "friction_angle"},
    "newtonian": {"material", "density", "E", "nu", "mu", "kappa"},
    "non_newtonian": {"material", "density", "E", "nu", "mu", "kappa", "yield_stress", "plastic_viscosity"},
}


def extract_material_type(material_init: dict) -> str:
    raw_material_type = material_init.get("material_type", material_init.get("material"))
    if raw_material_type is None:
        raise ValueError(
            "material json must contain 'material_type' or 'material'. "
            f"Received keys: {sorted(material_init.keys())}"
        )
    return normalize_material_type(str(raw_material_type))


def mu_kappa_to_E_nu(mu: float, kappa: float) -> tuple[float, float]:
    if mu <= 0 or kappa <= 0:
        raise ValueError(f"mu and kappa must be positive, got mu={mu}, kappa={kappa}")
    e_value = 9.0 * kappa * mu / (3.0 * kappa + mu)
    nu_value = (3.0 * kappa - 2.0 * mu) / (2.0 * (3.0 * kappa + mu))
    # Guard against values outside the stable range caused by poor VLM guesses.
    nu_value = min(max(nu_value, -0.499), 0.4995)
    return e_value, nu_value


def enrich_fluid_parameters(material_type: str, material_init: dict) -> dict:
    normalized = dict(material_init)
    if material_type not in {"newtonian", "non_newtonian"}:
        return normalized

    has_e_nu = "E" in normalized and "nu" in normalized
    has_mu_kappa = "mu" in normalized and "kappa" in normalized

    if has_mu_kappa:
        mu_value = float(normalized["mu"])
        kappa_value = float(normalized["kappa"])
        e_value, nu_value = mu_kappa_to_E_nu(mu_value, kappa_value)
        normalized.setdefault("E", e_value)
        normalized.setdefault("nu", nu_value)
        normalized["_converted_from_mu_kappa"] = True
    elif not has_e_nu:
        raise ValueError(
            f"{material_type} material json must contain either (mu, kappa) following the paper "
            f"or legacy (E, nu) fields. Received keys: {sorted(material_init.keys())}"
        )

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a PhysFlow config from an LLM material estimate.")
    parser.add_argument("--material-json", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--default-camera-index", type=int, default=None)
    parser.add_argument("--bbox-2d", nargs=4, type=int, default=None)
    args = parser.parse_args()

    material_init = read_json(args.material_json)
    material_type = extract_material_type(material_init)
    material_init = enrich_fluid_parameters(material_type, material_init)
    template_path = args.template or MATERIAL_CONFIG_TEMPLATES[material_type]
    config = copy.deepcopy(read_json(template_path))

    config["material"] = material_type
    keep = SUPPORTED_FIELDS[material_type]
    for field in list(config.keys()):
        if field in {"opacity_threshold", "rotation_degree", "rotation_axis", "substep_dt", "frame_dt",
                     "opt_frame_dt", "frame_num", "g", "grid_v_damping_scale", "rpic_damping",
                     "boundary_conditions", "mpm_space_vertical_upward_axis", "mpm_space_viewpoint_center",
                     "default_camera_index", "show_hint", "move_camera", "delta_a", "delta_e",
                     "delta_r", "line", "bbox_2d", "reduction", "lr", "n_grid"}:
            continue
        if field not in keep:
            config.pop(field, None)

    for field in keep:
        if field == "material":
            continue
        if field in {"mu", "kappa"}:
            continue
        if field in material_init:
            config[field] = material_init[field]

    required_for_config = {field for field in keep if field not in {"material", "mu", "kappa"}}
    missing_fields = sorted(field for field in required_for_config if field not in material_init)
    if missing_fields:
        print(
            "Warning: material json is missing fields "
            f"{missing_fields}; keeping template defaults from {template_path.name}."
        )

    if material_init.get("_converted_from_mu_kappa"):
        print(
            "Warning: converted paper-style fluid parameters (mu, kappa) into legacy solver config fields "
            f"(E={config.get('E')}, nu={config.get('nu')}). "
            "This is a compatibility workaround for the current PhysFlow config path, not the paper's native fluid parameterization."
        )

    if args.default_camera_index is not None:
        config["default_camera_index"] = args.default_camera_index
    if args.bbox_2d is not None:
        config["bbox_2d"] = list(args.bbox_2d)

    print(
        "Warning: the selected template still provides scene-specific settings such as "
        "boundary_conditions, camera placement, and bbox_2d. Review them before trusting the result."
    )
    write_json(args.output_config, config)
    print(f"Saved PhysFlow config to {args.output_config}")


if __name__ == "__main__":
    main()
