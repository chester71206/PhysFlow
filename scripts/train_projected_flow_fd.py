"""Projected-flow training for materials without a validated MPM adjoint.

Uses the original forward MPM and central finite differences in material
coordinates. This is slower than projected-flow AD, but keeps the same fixed
observed RAFT field and frozen initial visibility definition.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'utils')]

MATERIAL_PARAMETERS = {
    'metal': ('E', 'nu', 'yield_stress'),
    'foam': ('E', 'nu', 'yield_stress', 'plastic_viscosity'),
    'plasticine': ('E', 'nu', 'yield_stress'),
    'non_newtonian': ('E', 'nu', 'yield_stress', 'plastic_viscosity'),
    'sand': ('friction_angle',),
}


def encode(name, value):
    if name == 'nu':
        if not 0 < value < .5:
            raise ValueError('nu must lie strictly between 0 and 0.5')
        return math.log(value / (.5 - value))
    if name == 'friction_angle':
        if not 0 < value < 89:
            raise ValueError('friction_angle must lie between 0 and 89 degrees')
        return value
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return math.log10(value)


def decode(name, coordinate):
    if name == 'nu':
        return .5 / (1 + math.exp(-coordinate))
    if name == 'friction_angle':
        return max(.01, min(89., coordinate))
    return 10 ** max(-12., min(4., coordinate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--physics-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, help='Default 0.003; preserve checkpoint learning rate on resume')
    parser.add_argument('--interval', type=int, default=32, help='Accepted for CLI parity; forward solver does not checkpoint')
    parser.add_argument('--max-points', type=int, default=20000)
    parser.add_argument('--fd-step', type=float, help='Override central-difference step; default 0.003, or 0.1 degree for sand')
    parser.add_argument('--guidance-mode', choices=('raft', 'raft_cpu'), default='raft_cpu')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--resume', type=Path, help='optimizer.pt from a prior FD run; epochs means additional updates')
    a = parser.parse_args()
    if a.epochs < 1 or (a.lr is not None and a.lr <= 0) or (a.fd_step is not None and a.fd_step <= 0) or a.max_points < 1:
        parser.error('epochs, optional lr, fd-step, and max-points must be positive')
    config = json.loads(Path(a.physics_config).read_text())
    kind = config['material']
    if kind not in MATERIAL_PARAMETERS:
        parser.error(f'FD backend covers {tuple(MATERIAL_PARAMETERS)}; use train_flow.sh for elastic/Newtonian')
    if config.get('training_objective') != 'flow' or config.get('guidance_background') != 'white_masked':
        parser.error('Use the PAC-NeRF projected-flow config: training_objective=flow, guidance_background=white_masked')
    names = MATERIAL_PARAMETERS[kind]
    if a.plan_only:
        print(json.dumps(dict(material=kind,backend='central_finite_difference',parameters=names,
                              model_path=a.model_path,physics_config=a.physics_config,epochs=a.epochs),indent=2))
        return

    import torch
    import warp as wp
    from supplemental_pipeline.simulation_timing import prepare_run
    from utils.projected_flow_objective import FrozenVisibilityProjectedFlow
    from utils.sparse_projected_flow_objective import SparseFrozenVisibilityProjectedFlow

    count = len(list((Path(a.model_path) / 'images_generated').glob('*.png')))
    if count < 2:
        raise ValueError(f'At least two source images are needed: {a.model_path}')
    args, plan = prepare_run(['--model_path', a.model_path, '--physics_config', a.physics_config,
                              '--output_path', a.output, '--n_epoch', str(a.epochs),
                              '--n_key_frame', str(count - 1), '--material-optimizer', 'legacy',
                              '--guidance-mode', a.guidance_mode, '--skip-video'])
    signature = dict(model_path=str(Path(a.model_path).resolve()),
                     config_sha256=hashlib.sha256(Path(a.physics_config).read_bytes()).hexdigest(),
                     selected_frames=plan['selected_frames'],parameters=names)

    def callback(g):
        out = Path(g['args'].output_path)
        solver = g['mpm_solver']
        device = g['device']
        initial = g['mpm_init_pos'].detach()
        n = min(g['gs_num'], g['init_len'])
        steps = tuple(plan['target_steps'][1:])
        dt = plan['substep_dt']
        cameras = [g['get_camera_view'](args.model_path,
                    default_camera_index=g['camera_params']['default_camera_index'],
                    center_view_world_space=g['viewpoint_center_worldspace'],
                    observant_coordinates=g['observant_coordinates'], current_frame=i,
                    downsample=args.downsample)[0] for i in range(len(steps) + 1)]
        initial_values = {name: float(wp.to_torch(getattr(solver.mpm_model, name))[0]) for name in names}
        z0 = [encode(name, initial_values[name]) for name in names]
        resume = torch.load(a.resume, map_location='cpu') if a.resume else None
        if resume is not None and resume['signature'] != signature:
            raise ValueError('FD resume model/config/frames/parameters differ from checkpoint')
        scale = g['scale_origin'].detach() if torch.is_tensor(g['scale_origin']) else g['scale_origin']
        center = g['original_mean_pos'].detach()
        rotations = [r.detach() for r in g['rotation_matrices']]

        def rollout(coordinates):
            values = {name: decode(name, float(z)) for name, z in zip(names, coordinates)}
            with torch.no_grad():
                solver.reset_pos_from_torch(g['mpm_init_pos'], g['mpm_init_vol'], g['mpm_init_cov'],
                                             device=device, complete=True)
                for name, value in values.items():
                    wp.to_torch(getattr(solver.mpm_model, name)).fill_(value)
                solver.finalize_mu_lam(device=device)
                pos = [initial[:n].clone()]
                selected = set(steps)
                for t in range(1, steps[-1] + 1):
                    solver.p2g2p(t - 1, dt, device=device)
                    if t in selected:
                        pos.append(solver.export_particle_x_to_torch()[:n].detach().clone())
                if len(pos) != len(steps) + 1:
                    raise RuntimeError('Rollout missed a source frame')
                x = torch.stack(pos)
                world = g['apply_inverse_rotations'](g['undotransform2origin'](
                    g['undoshift2center111'](x.reshape(-1, 3)), scale, center),
                    rotations).reshape(len(steps) + 1, n, 3)
                if not torch.isfinite(world).all():
                    raise FloatingPointError('Nonfinite material rollout')
                return world, values

        reference, _ = rollout(z0)
        observed_flow = g['guidance'].guidance_flows.to(device)
        observed_images = g['guidance'].guidance.to(device)
        try:
            objective = FrozenVisibilityProjectedFlow(reference, cameras,
                observed_flow, observed_images, g['opacity_render'], max_points=a.max_points)
            active_pairs = list(range(len(observed_flow)))
            skipped_pairs = []
        except ValueError as exc:
            if 'Insufficient fixed visible foreground points' not in str(exc):
                raise
            objective = SparseFrozenVisibilityProjectedFlow(reference, cameras,
                observed_flow, observed_images, g['opacity_render'], max_points=a.max_points)
            active_pairs = objective.active_pairs
            skipped_pairs = objective.skipped_pairs
            print('PROJECTED_FLOW_FD_VISIBLE_PAIRS',json.dumps(dict(active=active_pairs,
                skipped=skipped_pairs,counts=objective.counts)),flush=True)
        g['guidance'].model.cpu()
        torch.cuda.empty_cache()

        def loss_at(z):
            positions, values = rollout(z)
            with torch.no_grad():
                loss = float(objective(positions))
            if not math.isfinite(loss):
                raise FloatingPointError(f'Nonfinite projected-flow loss at {values}')
            return loss, values

        first_loss, _ = loss_at(z0)
        repeated, _ = loss_at(z0)
        if abs(first_loss - repeated) > max(1e-5, abs(first_loss) * 1e-4):
            raise RuntimeError('Forward rollout is not repeatable; finite differences are unsafe')
        (out / 'backend.json').write_text(json.dumps(dict(backend='central_finite_difference',
            material=kind,parameters=names,initial_loss=first_loss,
            repeat_loss=repeated,fd_step_override=a.fd_step,
            active_flow_pairs=active_pairs,skipped_flow_pairs=skipped_pairs,
            objective='projected physical flow vs fixed observed RAFT flow',
            warning='Finite-difference optimizer; not reverse AD'),indent=2) + '\n')
        z = torch.nn.Parameter(torch.tensor(z0, dtype=torch.float64))
        optimizer = torch.optim.Adam([z], lr=.003 if a.lr is None else a.lr)
        first_epoch = 0
        if resume is not None:
            with torch.no_grad():
                z.copy_(resume['coordinates'])
            optimizer.load_state_dict(resume['optimizer'])
            if a.lr is not None:
                for group in optimizer.param_groups:
                    group['lr'] = a.lr
            first_epoch = int(resume['epoch'])
        for epoch in range(first_epoch + 1, first_epoch + a.epochs + 1):
            before = z.detach().tolist()
            loss_before, params_before = loss_at(before)
            fd = []
            for j in range(len(names)):
                step = a.fd_step if a.fd_step is not None else (.1 if names[j] == 'friction_angle' else .003)
                plus, minus = before.copy(), before.copy()
                plus[j] += step
                minus[j] -= step
                high, _ = loss_at(plus)
                low, _ = loss_at(minus)
                fd.append((high - low) / (2 * step))
            if not all(math.isfinite(v) for v in fd):
                raise FloatingPointError('Nonfinite finite-difference material gradient')
            optimizer.zero_grad(set_to_none=True)
            z.grad = torch.tensor(fd, dtype=torch.float64)
            optimizer.step()
            after = z.detach().tolist()
            loss_after, params_after = loss_at(after)
            row = dict(epoch=epoch,backend='central_finite_difference',learning_rate=optimizer.param_groups[0]['lr'],
                before=params_before,after=params_after,loss_before=loss_before,
                loss_after=loss_after,coordinate_fd=fd)
            with (out / 'training.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            (out / 'ckpt').mkdir(exist_ok=True)
            (out / 'ckpt/sim.json').write_text(json.dumps(params_after,indent=2) + '\n')
            temporary = out / 'ckpt/optimizer.pt.tmp'
            torch.save(dict(epoch=epoch,coordinates=z.detach(),optimizer=optimizer.state_dict(),
                            parameters=params_after,signature=signature),temporary)
            temporary.replace(out / 'ckpt/optimizer.pt')
            print('PROJECTED_FLOW_FD_TRAIN',json.dumps(row),flush=True)

    runpy.run_path(str(ROOT / 'flow_scene.py'), run_name='__main__',
                   init_globals={'_prepared_args':args,'_timing_plan':plan,'_inference_callback':callback})


if __name__ == '__main__':
    main()
