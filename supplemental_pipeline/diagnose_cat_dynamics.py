"""Read-only cat trajectory audit; writes isolated measurements, not training files.

Run from the repository root with the physflow Python and LD_LIBRARY_PATH unset.
Reuses simulation.py initialization so particle preprocessing stays identical.
"""
from pathlib import Path
import sys
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs/diagnostics/cat_dynamics'
OUT.mkdir(parents=True, exist_ok=True)
sys.argv = ['simulation.py', '--model_path', str(ROOT / 'model/pacnerf/cat'),
            '--physics_config', str(ROOT / 'config/pacnerf/cat_config.json'),
            '--output_path', str(OUT / 'init'), '--n_epoch', '1', '--n_key_frame', '13']
source = (ROOT / 'simulation.py').read_text()
source = source.replace('    wandb.init(project="Phys3D")', '    pass  # No training logger in the audit')
source = source.replace("    wandb.run.name = args.output_path.split('/')[-1]", '    pass')
exec(compile(source.split('    # run the simulation\n')[0], str(ROOT / 'simulation.py'), 'exec'), globals())

gt = [pcu.load_mesh_v(str(ROOT / f'model/pacnerf/cat/pcd/{i}.ply')) for i in range(14)]
gt_tensors = [torch.as_tensor(p, dtype=torch.float32, device='cuda')[None] for p in gt]
metadata = json.loads((ROOT / 'PAC_NeRF_data/cat/all_data.json').read_text())
times = sorted(e['time'] for e in metadata if Path(e['file_path']).stem.startswith('r_10_') and e['time'] >= 0)
dt_ref = times[1] - times[0]
# Fit the rigid free-fall portion, before contact. This is a hypothesis, not a GT velocity label.
coef = np.polyfit(np.arange(5) * dt_ref, [p[:, 1].mean() for p in gt[:5]], 2)
scale_value = float(scale_origin)
initial_params = {k: material_params[k] for k in ('E', 'nu', 'yield_stress')}
trained_params = json.loads((ROOT / 'outputs/pacnerf/cat_ds1.0_ep50/ckpt/sim.json').read_text())

def world_positions():
    pos = mpm_solver.export_particle_x_to_torch()[:gs_num]
    return apply_inverse_rotations(undotransform2origin(undoshift2center111(pos), scale_origin, original_mean_pos), rotation_matrices).detach().cpu().numpy()

def measure(pos, i):
    with torch.no_grad():
        chamfer = float(chamfer_distance(torch.as_tensor(pos, dtype=torch.float32, device='cuda')[None], gt_tensors[i])[0])
    return {'frame': i, 'sim_time': mpm_solver.time, 'target_time': times[i],
            'chamfer': chamfer, 'mean_y': float(pos[:, 1].mean()),
            'target_mean_y': float(gt[i][:, 1].mean()), 'min_y': float(pos[:, 1].min())}

results = {'scale': scale_value, 'freefall_fit': {'world_acceleration': float(2*coef[0]),
           'world_velocity_at_pcd0': float(coef[1])}, 'initial_params': initial_params,
           'trained_params': trained_params, 'runs': {}}
variants = [('original_initial', initial_params, .0022, False),
            ('original_trained', trained_params, .0022, False),
            ('time_aligned_trained', trained_params, dt_ref, False),
            ('time_scale_velocity_trained', trained_params, dt_ref, True)]
for label, params, interval, correct_motion in variants:
    # Fresh state avoids carrying affine velocity or other internal state across variants.
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(mpm_init_pos, mpm_init_vol, mpm_init_cov,
                                          n_grid=material_params['n_grid'], grid_lim=material_params['grid_lim'])
    mpm_solver.set_parameters_dict(material_params)
    set_boundary_conditions(mpm_solver, bc_params, time_params)
    for key, value in params.items():
        setattr(mpm_solver.mpm_model, key, torch2warp_float(torch.tensor([value], dtype=torch.float32, device='cuda')))
    mpm_solver.mpm_model.gravitational_accelaration = wp.vec3(0., -9.8 * (scale_value if correct_motion else 1.), 0.)
    if correct_motion:
        vel = wp.to_torch(mpm_solver.mpm_state.particle_v)
        with torch.no_grad():
            vel[:, 1] = float(coef[1]) * scale_value
    mpm_solver.finalize_mu_lam()
    rows = [measure(world_positions(), 0)]
    for i in range(1, 14):
        # Reach absolute timestamps with nearest substeps; avoid accumulating rounding error.
        target = i * interval
        steps = round((target - mpm_solver.time) / time_params['substep_dt'])
        for _ in range(steps):
            mpm_solver.p2g2p(i - 1, time_params['substep_dt'], device=device)
        pos = world_positions()
        if not np.isfinite(pos).all():
            raise RuntimeError(f'Non-finite trajectory: {label} frame {i}')
        rows.append(measure(pos, i))
    results['runs'][label] = rows
    print(label, 'mean Chamfer', np.mean([r['chamfer'] for r in rows[1:]]), 'last', rows[-1], flush=True)
    (OUT / 'measurements.json').write_text(json.dumps(results, indent=2))
print('Saved', OUT / 'measurements.json')
