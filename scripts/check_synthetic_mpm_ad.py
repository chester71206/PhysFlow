"""Check original Warp forward parity and existing reverse AD on a saved pilot state."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.repaired_temporal_rollout import RepairedTemporalRollout

root=Path(sys.argv[1])
cfg=json.loads((root/'config.json').read_text())
traj=np.load(root/'trajectory.npz')
x=traj['positions'][0].astype(np.float32)
vol=traj['initial_volume'].astype(np.float32)
device='cuda:0'
wp.init()
solver=MPM_Simulator_WARP(len(x),n_grid=cfg['grid_n'],grid_lim=cfg['grid_lim_m'],device=device)
solver.load_initial_data_from_torch(torch.as_tensor(x,device=device),torch.as_tensor(vol,device=device),
                                    n_grid=cfg['grid_n'],grid_lim=cfg['grid_lim_m'],device=device)
if any(cfg['initial_velocity_m_s']):
    solver.set_initial_velocity_from_torch(torch.as_tensor(np.tile(cfg['initial_velocity_m_s'],(len(x),1)),
                                                           dtype=torch.float32,device=device))
solver.set_parameters_dict(dict(material='elastic',density=cfg['density_kg_m3'],
                                E=cfg['E_solver_internal'],nu=cfg['nu'],
                                g=cfg['gravity_m_s2'],grid_v_damping_scale=1.0),device=device)
solver.finalize_mu_lam(device=device)
solver.add_surface_collider((0,0,cfg['ground_z_m']),(0,0,1),'sticky',0)
engine=RepairedTemporalRollout(solver)
steps=(2,8)
E=torch.tensor(cfg['E_solver_internal'],device=device,requires_grad=True)
nu=torch.tensor(cfg['nu'],device=device,requires_grad=True)
ys=torch.tensor(0.,device=device,requires_grad=True)
out=engine.apply(E,nu,ys,steps,cfg['dt_s'],4)[0]
objective=(out[-1,:int(traj['n_surface']),2]**2).sum()
objective.backward()
ad=[float(E.grad),float(nu.grad)]
for i in range(8):solver.p2g2p(i,cfg['dt_s'],device=device)
parity=float((out[-1]-solver.export_particle_x_to_torch()).abs().max())
report=dict(steps=steps,forward_max_abs_error_m=parity,gradient_E_internal=ad[0],
            gradient_nu=ad[1],finite_gradients=all(math.isfinite(v) for v in ad),
            reverse_ad_executed=True,
            pass_gate=parity<1e-4 and all(math.isfinite(v) for v in ad))
(root/'reverse_ad_gate.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
if not report['pass_gate']:raise SystemExit(1)
