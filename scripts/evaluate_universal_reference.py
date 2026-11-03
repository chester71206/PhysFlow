"""Complete time-aligned reference rollout and explicitly defined image metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'utils')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paper_ecms(flows):
    """Requested ECMS: sum of temporal L2², spatial Laplacian L2², 0.1 L1."""
    import torch
    import torch.nn.functional as F
    v = torch.stack(flows).float()
    temporal = (v[1:] - v[:-1]).square().sum().item()
    p = F.pad(v, (1, 1, 1, 1), mode='replicate')
    lap = p[..., 1:-1, :-2] + p[..., 1:-1, 2:] + p[..., :-2, 1:-1] + p[..., 2:, 1:-1] - 4 * v
    spatial = lap.square().sum().item()
    speed_l1 = v.abs().sum().item()
    alpha = .1
    return dict(ecms=temporal + spatial + alpha*speed_l1, temporal=temporal,
                spatial=spatial, motion_l1=alpha*speed_l1, alpha=alpha,
                formula='sum_t (||v[t+1]-v[t]||_2^2 + ||Laplacian(v[t])||_2^2 + 0.1||v[t]||_1)',
                discretization='RAFT 256x480 pixels; 5-point Laplacian with replicate padding; all norms sum over channels and pixels',
                caution='Formula requested by user; do not compare numerical values with the previous inverse-motion ECMS implementation.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--physics-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-json', type=Path)
    p.add_argument('--reference-fps', type=float)
    p.add_argument('--key-frames', type=int)
    p.add_argument('--downsample', type=float, default=1.)
    p.add_argument('--dataset', choices=('pacnerf', 'phys_dreamer', 'realworld'), required=True)
    a = p.parse_args()
    model = a.model_path.resolve()
    config_path = a.physics_config.resolve()
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    config = json.loads(config_path.read_text())
    source_count = len([x for x in (model/'images_generated').iterdir()
                        if x.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    nframes = a.key_frames or source_count - 1
    command = ['--model_path', str(model), '--physics_config', str(config_path),
               '--output_path', str(out), '--n_epoch', '1', '--n_key_frame', str(nframes),
               '--guidance-mode', 'dummy', '--material-optimizer', 'legacy', '--skip-video',
               '--downsample', str(a.downsample)]
    if a.reference_fps is not None:
        command.extend(['--reference-fps', str(a.reference_fps)])
    from supplemental_pipeline.simulation_timing import prepare_run
    args, plan = prepare_run(command)
    import numpy as np
    import torch
    import torch.nn.functional as F
    import warp as wp
    from PIL import Image
    from torchvision.models.optical_flow import raft_large
    from video_distillation.cogv_guidance import preprocess
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = json.loads(a.checkpoint_json.read_text()) if a.checkpoint_json else None

    def evaluate(g):
        out.mkdir(parents=True)
        frame_dir = out/'frames'
        frame_dir.mkdir()
        solver = g['mpm_solver']
        device = g['device']
        if checkpoint:
            with torch.no_grad():
                for key, value in checkpoint.items():
                    arr = wp.to_torch(getattr(solver.mpm_model, key))
                    if arr.numel() != 1:
                        raise ValueError(f'Expected shared scalar {key}')
                    arr.fill_(float(value))
        solver.reset_pos_from_torch(g['mpm_init_pos'], g['mpm_init_vol'], g['mpm_init_cov'],
                                    device=device, complete=True)
        solver.finalize_mu_lam(device=device)
        images, targets, psnrs, areas = [], [], [], []
        for i, entry in enumerate(plan['selected_frames']):
            if i:
                steps = g['steps_to_target'](entry['target_step'], solver.time, plan['substep_dt'])
                with torch.no_grad():
                    for _ in range(steps):
                        solver.p2g2p(i-1, plan['substep_dt'], device=device)
            if abs(solver.time - entry['actual_time']) > plan['substep_dt'] * 1e-3:
                raise RuntimeError('Reference solver missed source timestamp')
            position = solver.export_particle_x_to_torch()[:g['gs_num']]
            if not torch.isfinite(position).all():
                raise FloatingPointError(f'Nonfinite state at frame {i}')
            camera, _ = g['get_camera_view'](str(model),
                default_camera_index=g['camera_params']['default_camera_index'],
                center_view_world_space=g['viewpoint_center_worldspace'],
                observant_coordinates=g['observant_coordinates'], current_frame=max(i-1, 0),
                downsample=args.downsample)
            g['rasterize'] = g['initialize_resterize'](camera, g['gaussians'], g['pipeline'], g['background'])
            with torch.no_grad():
                image = g['render_frame'](solver, g['gs_num'], g['init_len'],
                    g['moving_pts_path'], camera, g['gaussians'], g['params_inpaint'],
                    g['opacity_render'], g['shs_render'], g['unselected_pos'],
                    g['unselected_cov'], g['unselected_opacity'], g['unselected_shs']).detach()
            if not torch.isfinite(image).all():
                raise FloatingPointError(f'Nonfinite render at frame {i}')
            target = g['guidance'].guidance[i].detach()
            if target.shape[-2:] != image.shape[-2:]:
                target = F.interpolate(target[None], size=image.shape[-2:],
                                       mode='bilinear', align_corners=False)[0]
            images.append(image)
            targets.append(target)
            pixels = image.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(pixels).save(frame_dir/f'{i:04d}.png')
            err = ((image.clamp(0,1) - target.clamp(0,1))**2).mean().item()
            psnrs.append(float(-10*math.log10(max(err, 1e-12))))
            areas.append(float(np.any(pixels < 250, axis=2).mean()))
            print('REFERENCE FRAME', a.dataset, model.name, i, '/', nframes, flush=True)
        raft = raft_large(pretrained=True, progress=False).eval().to(device)
        raft.requires_grad_(False)
        simulated_flows, target_flows = [], []
        with torch.no_grad():
            for i in range(len(images)-1):
                simulated_flows.append(raft(preprocess(images[i][None]),
                                            preprocess(images[i+1][None]))[-1][0].detach())
                if a.dataset == 'pacnerf':
                    target_flows.append(raft(preprocess(targets[i][None]),
                                             preprocess(targets[i+1][None]))[-1][0].detach())
        ecms = paper_ecms(simulated_flows) if a.dataset != 'pacnerf' else None
        epe = None
        if target_flows:
            errors = torch.stack([(x-y).square().sum(0).sqrt() for x,y in zip(simulated_flows,target_flows)])
            masks = []
            for entry in plan['selected_frames'][:-1]:
                with Image.open(entry['guidance_rgba_path']) as im:
                    mask = torch.from_numpy((np.asarray(im.convert('RGBA'))[...,3] == 255).copy()).float().to(device)
                masks.append(F.interpolate(mask[None,None], size=errors.shape[-2:],mode='nearest')[0,0].bool())
            foreground = torch.stack(masks)
            flow_residual = torch.stack([x-y for x,y in zip(simulated_flows,target_flows)])
            epe = dict(foreground_px=float(errors[foreground].mean()),
                       whole_px=float(errors.mean()),
                       half_mse=float(0.5 * flow_residual.square().mean()),
                       flow_hw=list(errors.shape[-2:]))
            # Retain the exact fields and mask used by the metric so later
            # optical-flow comparisons do not need another physics rollout.
            torch.save(dict(simulated_flows=torch.stack(simulated_flows).cpu(),
                            target_flows=torch.stack(target_flows).cpu(),
                            foreground=foreground.cpu(),
                            selected_source_frames=[x['filename'] for x in plan['selected_frames']]),
                       out/'flow_fields.pt')
        params = {k:float(wp.to_torch(getattr(solver.mpm_model,k))[0])
                  for k in ('E','nu','yield_stress','plastic_viscosity') if k in config}
        result = dict(dataset=a.dataset, scene=model.name, status='complete',
            protocol='Independent complete time-aligned reference rollout at selected source timestamps',
            parameters=params,checkpoint_sha256=sha(a.checkpoint_json) if a.checkpoint_json else None,
            config_sha256=sha(config_path),source_timing=plan['source'],
            target_steps=plan['target_steps'],selected_source_frames=[x['filename'] for x in plan['selected_frames']],
            num_frames=len(images),downsample=a.downsample,
            psnr_db=float(np.mean(psnrs[1:])),psnr_by_frame=psnrs,
            psnr_target='supplied white RGBA GT' if a.dataset=='pacnerf' else 'diffusion guidance (no aligned dynamic GT)',
            ecms=ecms,epe=epe,finite=True,min_dark_pixel_fraction=min(areas),
            metric_notes='RAFT flow at 256x480; frame 0 excluded from mean PSNR; ECMS uses simulated frames only')
        (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
        print('REFERENCE EVALUATION',json.dumps({k:v for k,v in result.items() if k!='psnr_by_frame'}),flush=True)

    runpy.run_path(str(ROOT/'flow_scene.py'), run_name='__main__',
                   init_globals={'_prepared_args':args,'_timing_plan':plan,'_inference_callback':evaluate})


if __name__ == '__main__':
    main()
