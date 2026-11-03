"""Runtime for simulation.py; launch through the CPU-only timing preflight."""
# Retains memory-lifetime fixes; training times come from the official preflight.
from supplemental_pipeline.aligned_timing import steps_to_target
from utils.initial_kinematics import configure_initial_kinematics
from utils.rollout_objective import RolloutObjective
import sys
from pathlib import Path
import hashlib
import wandb
import argparse
import math
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import json
from tqdm import tqdm
from omegaconf import OmegaConf
import point_cloud_utils as pcu
from pytorch3d.loss import chamfer_distance

# Gaussian splatting dependencies
sys.path.append("gs")
from scene.gaussian_model import GaussianModel
from diff_plane_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from gaussian_renderer import render, GaussianModel

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.mpm_utils import sum_array, sum_mat33, sum_vec3, wp_clamp, update_param
from mpm_solver_warp.warp_utils import torch2warp_float
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
sys.path.append("utils")
from decode_param import *
from transformation_utils import *
from camera_view_utils import *
from render_utils import *
from cv2_compat import COLOR_BGR2RGB, COLOR_RGB2BGR, cvtColor as cv2_cvtColor, imwrite as cv2_imwrite, resize as cv2_resize
from save_video import save_video
from threestudio_utils import cleanup
from update_grad import update_grad_param, lr_scheduler

from video_distillation.cogv_guidance import CogVideoGuidance, DummyGuidance

torch.manual_seed(0)

wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=_prepared_args.taichi_memory_gb)

SCALE_E = 1e7

class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def unpack_rasterize_output(result):
    if not isinstance(result, tuple):
        raise TypeError(f"Unexpected rasterize output type: {type(result)!r}")
    if len(result) < 4:
        raise ValueError(f"Unexpected rasterize output length: {len(result)}")
    return result[0], result[1], result[2], result[3]


def load_checkpoint(model_path, iteration=-1, material=None):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )

    # sh_degree=0, if you use a 3D asset without spherical harmonics
    from plyfile import PlyData
    plydata = PlyData.read(checkpt_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

    # Load guassians
    sh_degree = int(math.sqrt((len(extra_f_names)+3) // 3)) - 1
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path, material)
    return gaussians


def load_inpaint_gs(model_path):
    checkpt_path = os.path.join(
        model_path, "inpaint_points.ply"
    )
    if not os.path.exists(checkpt_path):
        return None

    # sh_degree=0, if you use a 3D asset without spherical harmonics
    from plyfile import PlyData
    plydata = PlyData.read(checkpt_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

    # Load guassians
    sh_degree = int(math.sqrt((len(extra_f_names)+3) // 3)) - 1
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def render_frame(mpm_solver, gs_num, init_len, moving_pts_path,
                current_camera, gaussians, params_inpaint,
                opacity, shs,
                unselected_pos, unselected_cov, unselected_opacity, unselected_shs):

    pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
    cov3D = mpm_solver.export_particle_cov_to_torch()
    rot = mpm_solver.export_particle_R_to_torch()

    cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
    rot = rot.view(-1, 3, 3)[:gs_num].to(device)

    pos = pos[:init_len,:]
    pos = apply_inverse_rotations(
        undotransform2origin(
            undoshift2center111(pos), scale_origin, original_mean_pos
        ),
        rotation_matrices,
    )
    cov3D = cov3D / (scale_origin * scale_origin)
    cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
    if os.path.exists(moving_pts_path):
        # print("---select moving points")
        pos = torch.cat([pos, unselected_pos], dim=0)
        cov3D = torch.cat([cov3D, unselected_cov], dim=0)
        opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
        shs = torch.cat([shs_render, unselected_shs], dim=0)
    if params_inpaint is not None:
        pos = torch.cat([pos, params_inpaint['pos']], dim=0)
        cov3D = torch.cat([cov3D, params_inpaint['cov3D_precomp']], dim=0)
        opacity = torch.cat([opacity, params_inpaint['opacity']], dim=0)
        shs = torch.cat([shs, params_inpaint['shs']], dim=0)

    colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
    rendering, _, _, _ = unpack_rasterize_output(rasterize(
        means3D=pos,
        means2D=pos,
        means2D_abs=pos,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=None,
        rotations=None,
        cov3D_precomp=cov3D,
    ))
    return rendering


if __name__ == "__main__":
    if "_prepared_args" not in globals():
        raise RuntimeError("Launch scripts/train_projected_flow_gradient.py; this module only initializes a scene")
    args = _prepared_args
    timing_plan = _timing_plan
    selected_frames = timing_plan["selected_frames"]
    selected_names = [entry["filename"] for entry in selected_frames]
    aligned_steps = timing_plan["target_steps"]

    if not os.path.exists(args.model_path):
        raise AssertionError("Model path does not exist!")
    if not os.path.exists(args.physics_config):
        raise AssertionError("Scene config does not exist!")

    guidance_path = os.path.join(args.model_path, 'images_generated')
    if not os.path.exists(guidance_path):
        raise AssertionError("Guidance frames do not exist!")

    if args.output_path is not None:
        args.output_path = args.output_path + f'_ds{args.downsample}_ep{args.n_epoch}'
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

    if args.debug:
        if not os.path.exists(f"{args.output_path}/log"):
            os.makedirs(f"{args.output_path}/log")

    if not args.eval:
        with open(os.path.join(args.output_path, "timing_manifest.json"), "w") as out:
            json.dump(timing_plan, out, indent=2)
        with open(os.path.join(args.output_path, "run_config.json"), "w") as out:
            json.dump({"arguments": vars(args), "mode": "time_aligned",
                       "runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "physics_config": json.load(open(args.physics_config))}, out, indent=2)
    print("TIME ALIGNMENT", json.dumps(timing_plan), flush=True)

    # Create experiment logger
    wandb.init(project="Phys3D", name=Path(args.output_path).name)


    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
        optimize_params
    ) = decode_param_json(args.physics_config)

    if args.downsample != 1.:
        # Only pixel coordinates scale; axis is a direction selector.
        for k in ("pos", "start", "end"):
            optimize_params["line"][k] = int(optimize_params["line"][k] * args.downsample)
        if optimize_params["bbox_2d"] is not None:
            optimize_params["bbox_2d"] = [v*args.downsample for v in optimize_params["bbox_2d"]]

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path, material=material_params["material"])
    gaussians_inpaint = load_inpaint_gs(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)
    params_inpaint = None
    if gaussians_inpaint is not None:
        params_inpaint = load_params_from_gs(gaussians_inpaint, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]

    # optimize moving parts only
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )

    moving_pts_path = os.path.join(model_path, "moving_part_points.ply")
    if os.path.exists(moving_pts_path):
        import point_cloud_utils as pcu
        moving_pts = pcu.load_mesh_v(moving_pts_path)
        moving_pts = torch.from_numpy(moving_pts).float().to("cuda")
        thres = 0.5 / material_params["n_grid"]
        if "playdoh" in model_path:
            thres = 1.0 / material_params["n_grid"]
        freeze_mask = find_far_points(
            init_pos, moving_pts, thres=thres
        ).bool()
        moving_pts.to("cpu")
        unselected_pos = init_pos[freeze_mask, :]
        unselected_cov = init_cov[freeze_mask, :]
        unselected_opacity = init_opacity[freeze_mask, :]
        unselected_shs = init_shs[freeze_mask, :]

        init_pos = init_pos[~freeze_mask, :]
        init_cov = init_cov[~freeze_mask, :]
        init_opacity = init_opacity[~freeze_mask, :]
        init_shs = init_shs[~freeze_mask, :]


    # rorate and translate object
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)


    scaling = 1.0
    if 'cat' in model_path:
        scaling = 0.7
    if 'letter' in model_path:
        scaling = 2.0
    if 'cream' in model_path:
        scaling = 0.8
    if 'toothpaste' in model_path:
        scaling = 0.6
    if 'playdoh' in model_path:
        scaling = 0.75
    scaling = float(json.loads(Path(args.physics_config).read_text()).get('mpm_scaling', scaling))
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError('mpm_scaling must be positive and finite')
    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos, scaling=scaling)
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            f"{args.output_path}/log/transformed_particles.ply",
        )

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, f"{args.output_path}/log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        _pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(mpm_init_pos[gs_num:]), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
        print("gs.xyz", gaussians._xyz.shape)
        gaussians._xyz = nn.Parameter(torch.tensor(torch.cat([gaussians._xyz, _pos], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _opacity = torch.zeros((_pos.shape[0], 1)).to("cuda:0")
        gaussians._opacity = nn.Parameter(torch.tensor(torch.cat([gaussians._opacity, _opacity], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _scaling = torch.zeros((_pos.shape[0], 3)).to("cuda:0")
        gaussians._scaling = nn.Parameter(torch.tensor(torch.cat([gaussians._scaling, _scaling], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _rotation = torch.zeros((_pos.shape[0], 4)).to("cuda:0")
        gaussians._rotation = nn.Parameter(torch.tensor(torch.cat([gaussians._rotation, _rotation], 0), dtype=torch.float, device="cuda").requires_grad_(True))

        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity


    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)
    kinematics = configure_initial_kinematics(mpm_solver,
        json.loads(Path(args.physics_config).read_text()), rotation_matrices, scale_origin)
    if not args.eval:
        Path(args.output_path, 'kinematics_manifest.json').write_text(json.dumps(kinematics, indent=2))

    set_boundary_conditions(mpm_solver, bc_params, time_params)

    tape = wp.Tape()


    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    # run the simulation
    rollout_objective = RolloutObjective(json.loads(Path(args.physics_config).read_text()))
    guidance_background = json.loads(Path(args.physics_config).read_text()).get('guidance_background', 'source')
    if args.guidance_mode == "dummy" or not rollout_objective.needs_flow:
        guidance = DummyGuidance(
            guidance_path, downsample=args.downsample, num_frames=args.n_key_frame, frame_names=selected_names,
            background_mode=guidance_background
        )
    else:
        flow_device = "cpu" if args.guidance_mode == "raft_cpu" else None
        guidance = CogVideoGuidance(
            guidance_path,
            downsample=args.downsample,
            num_frames=args.n_key_frame,
            flow_device=flow_device,
            frame_names=selected_names,
            background_mode=guidance_background,
        )

    current_camera, camera_view_info = get_camera_view(
        model_path,
        center_view_world_space=viewpoint_center_worldspace,
        observant_coordinates=observant_coordinates,
        default_camera_index=camera_params["default_camera_index"],
        downsample=args.downsample
    )
    rasterize = initialize_resterize(
        current_camera, gaussians, pipeline, background
    )

    ## To render the first frame as image prompt
    opacity_render = opacity
    shs_render = shs
    init_len = mpm_init_pos.shape[0]
    mpm_solver.reset_pos_from_torch(mpm_init_pos, mpm_init_vol, mpm_init_cov,
                                    device=device, complete=True)
    with torch.no_grad():
        image_prompt = render_frame(mpm_solver, gs_num, init_len, moving_pts_path,
                                    current_camera, gaussians, params_inpaint,
                                    opacity_render, shs_render,
                                    unselected_pos, unselected_cov, unselected_opacity, unselected_shs)

    # Allow evaluation-only callers to reuse the aligned scene setup and render
    # at the exact source timestamps without entering the optimizer.
    if "_inference_callback" in globals():
        _inference_callback(globals())
        sys.exit(0)


    raise RuntimeError("A training/evaluation callback is required")
