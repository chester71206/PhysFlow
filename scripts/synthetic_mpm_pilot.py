"""Run a resumable, bounded PhysFlow Warp MPM drop pilot for a tracked asset mesh."""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.repaired_temporal_rollout import RepairedTemporalRollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--E', type=float, default=25000)
    ap.add_argument('--nu', type=float, default=.32)
    ap.add_argument('--density', type=float, default=950)
    ap.add_argument('--height', type=float, default=.09)
    ap.add_argument('--duration', type=float, default=.55)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--dt', type=float, default=.0005)
    ap.add_argument('--spacing', type=float, default=.018)
    ap.add_argument('--grid', type=int, default=64)
    ap.add_argument('--ground', type=float, default=.12)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--velocity', type=float, nargs=3, default=(0.,0.,0.))
    ap.add_argument('--initial-particles',type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    result_file = a.out/'trajectory.npz'
    if result_file.exists() and (a.out/'physics.json').exists():
        print('RESUME complete', a.out, flush=True)
        return
    raw = np.load(a.mesh)
    surface = raw['vertices'].astype(np.float32)
    faces = raw['faces'].astype(np.int32)
    mesh = trimesh.Trimesh(vertices=surface, faces=faces, process=False)
    if not mesh.is_watertight:
        raise RuntimeError('Non-watertight asset is ineligible for volume MPM')
    # Use deterministic voxel-center occupancy. Surface vertices are included as tracked particles.
    lo, hi = surface.min(0), surface.max(0)
    axes = [np.arange(lo[k]+a.spacing/2, hi[k], a.spacing) for k in range(3)]
    xx = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    np.random.seed(0)
    inside = xx[mesh.contains(xx)].astype(np.float32)
    if len(inside) < 50:
        raise RuntimeError(f'Insufficient interior particles: {len(inside)}')
    shift = np.asarray([.5,.5,a.ground+a.height-float(lo[2])], dtype=np.float32)
    x0 = np.concatenate((surface, inside), axis=0)+shift
    volume = float(mesh.volume)
    if volume <= 0:
        raise RuntimeError('Mesh orientation or volume invalid')
    # Surface samples have lower mass than occupied voxel centers; normalize to exact mesh volume.
    weights = np.ones(len(x0),dtype=np.float32)
    weights *= volume/weights.sum()
    if a.initial_particles:
        source=np.load(a.initial_particles)
        if int(source['n_surface'])!=len(surface) or not np.array_equal(source['faces'],faces):
            raise RuntimeError('Initial particle source surface mesh/topology differs')
        x0=source['positions'][0].astype(np.float32).copy()
        weights=source['initial_volume'].astype(np.float32).copy()
        inside=x0[len(surface):]
    E_internal=a.E/1e7  # mpm_utils.SCALE_E; exposed CLI E is in Pa.
    init = dict(asset=str(a.mesh), E_Pa=a.E, E_solver_internal=E_internal,
                solver_E_scale_Pa=1e7, nu=a.nu, density_kg_m3=a.density,
                solver_forward='mpm_solver_warp.mpm_solver_warp.MPM_Simulator_WARP.p2g2p',
                candidate_forward='same MPM_Simulator_WARP.p2g2p with altered E/nu',
                reverse_ad_engine='mpm_solver_warp.repaired_temporal_rollout.RepairedTemporalRollout',
                gravity_m_s2=[0,0,-9.8], height_m=a.height, ground_z_m=a.ground,
                ground_contact='Warp add_surface_collider sticky static', dt_s=a.dt,
                fps=a.fps, duration_s=a.duration, grid_n=a.grid, grid_lim_m=1.0,
                particle_spacing_m=a.spacing, n_surface=len(surface),n_interior=len(inside),
                mesh_volume_m3=volume, initial_velocity_m_s=list(a.velocity),
                particle_volume_rule='uniform per particle, normalized to mesh volume',
                random_seed=0,initial_particles_source=str(a.initial_particles) if a.initial_particles else None)
    (a.out/'config.json').write_text(json.dumps(init,indent=2))
    wp.init()
    device=a.device
    solver=MPM_Simulator_WARP(len(x0),n_grid=a.grid,grid_lim=1.0,device=device)
    tx=torch.as_tensor(x0,device=device)
    tv=torch.as_tensor(weights,device=device)
    solver.load_initial_data_from_torch(tx,tv,n_grid=a.grid,grid_lim=1.0,device=device)
    if any(a.velocity):
        solver.set_initial_velocity_from_torch(torch.as_tensor(np.tile(a.velocity,(len(x0),1)),
                                                               dtype=torch.float32,device=device))
    solver.set_parameters_dict(dict(material='elastic',density=a.density,E=E_internal,nu=a.nu,
                                    g=[0.,0.,-9.8],grid_v_damping_scale=1.0),device=device)
    solver.finalize_mu_lam(device=device)
    solver.add_surface_collider((0,0,a.ground),(0,0,1),'sticky',0)
    # Construction gate verifies this exact boundary can be represented by the existing reverse rollout.
    try:
        RepairedTemporalRollout(solver)
        init['reverse_ad_path_constructed']=True
    except Exception as e:
        init['reverse_ad_path_constructed']=False
        init['reverse_ad_error']=repr(e)
    (a.out/'config.json').write_text(json.dumps(init,indent=2))
    frame_steps=np.round(np.arange(math.floor(a.duration*a.fps)+1)/(a.fps*a.dt)).astype(int)
    total=int(frame_steps[-1])
    checkpoint=a.out/'trajectory.partial.npz'
    state_names=('particle_x','particle_v','particle_C','particle_F_trial','particle_F_N_trial','particle_Jp')
    if checkpoint.exists():
        saved=np.load(checkpoint)
        if not np.array_equal(saved['positions'][0],x0) or float(saved['dt_s'])!=a.dt:
            raise RuntimeError('Partial checkpoint does not match initial state or dt')
        for name in state_names:
            target=getattr(solver.mpm_state,name)
            wp.copy(target,wp.from_numpy(saved['state_'+name],dtype=target.dtype,device=device))
        start_step=int(saved['step'])
        solver.time=start_step*a.dt
        positions=list(saved['positions']);velocities=list(saved['velocities']);stamps=list(saved['times'])
        print('RESUME from step',start_step,flush=True)
    else:
        start_step=0
        positions=[x0.copy()]
        velocities=[np.broadcast_to(np.asarray(a.velocity,dtype=np.float32),x0.shape).copy()]
        stamps=[0.0]
    start=time.monotonic()
    for step in range(start_step+1,total+1):
        solver.p2g2p(step-1,a.dt,device=device)
        if step in frame_steps:
            positions.append(solver.mpm_state.particle_x.numpy().copy())
            velocities.append(solver.mpm_state.particle_v.numpy().copy())
            stamps.append(step*a.dt)
            x=positions[-1]
            if not np.isfinite(x).all() or abs(x).max()>2:
                print('BAD_FRAME',step,'range',np.nanmin(x,axis=0),np.nanmax(x,axis=0),'speed',np.nanmax(np.linalg.norm(velocities[-1],axis=-1)),flush=True)
                raise RuntimeError(f'Numerical explosion at step {step}')
            checkpoint_data=dict(positions=np.asarray(positions),velocities=np.asarray(velocities),
                                 times=np.asarray(stamps),faces=faces,n_surface=len(surface),
                                 initial_volume=weights,dt_s=a.dt,step=step)
            checkpoint_data.update({'state_'+name:getattr(solver.mpm_state,name).numpy().copy()
                                    for name in state_names})
            temp=a.out/'trajectory.partial.tmp.npz'
            np.savez_compressed(temp,**checkpoint_data)
            os.replace(temp,checkpoint)
            print(f'FRAME {len(stamps)-1}/{len(frame_steps)-1} t={stamps[-1]:.4f} minz={x[:,2].min():.4f}',flush=True)
    P=np.asarray(positions); V=np.asarray(velocities)
    np.savez_compressed(result_file,positions=P,velocities=V,times=np.asarray(stamps),
                        faces=faces,n_surface=len(surface),initial_volume=weights)
    checkpoint.unlink(missing_ok=True)
    cm=P.mean(axis=1)
    relative=P-P[:,0:1]-(x0-x0[0:1])[None,:,:]
    minz=P[:,:,2].min(axis=1)
    stretch=np.linalg.norm(P[:,:len(surface),:][:,faces[:,0]]-P[:,:len(surface),:][:,faces[:,1]],axis=-1)
    e0=np.maximum(stretch[0],1e-8)
    metrics=dict(finite=bool(np.isfinite(P).all() and np.isfinite(V).all()),
                 max_speed_m_s=float(np.linalg.norm(V,axis=-1).max()),
                 max_particle_displacement_m=float(np.linalg.norm(P-P[0],axis=-1).max()),
                 max_relative_displacement_m=float(np.linalg.norm(relative,axis=-1).max()),
                 max_ground_penetration_m=float(np.maximum(a.ground-minz,0).max()),
                 min_z_m=float(minz.min()), initial_bottom_gap_m=float(minz[0]-a.ground),
                 final_mean_speed_m_s=float(np.linalg.norm(V[-1],axis=-1).mean()),
                 max_edge_strain=float(np.abs(stretch/e0-1).max()),
                 center_of_mass_z_m=cm[:,2].tolist(),
                 bottom_z_m=minz.tolist(), times_s=stamps,
                 runtime_s=time.monotonic()-start)
    dx=1/a.grid
    # Physics gates intentionally fixed before seeing pilot outcomes.
    gates=dict(finite=metrics['finite'],penetration=metrics['max_ground_penetration_m']<=dx,
               no_explosion=metrics['max_speed_m_s']<8 and metrics['max_particle_displacement_m']<.8,
               deformation=metrics['max_relative_displacement_m']>=.005,
               no_severe_mesh_strain=metrics['max_edge_strain']<1.5)
    metrics['thresholds']=dict(ground_penetration_m=dx,max_speed_m_s=8,
                              max_displacement_m=.8,min_relative_deformation_m=.005,
                              max_edge_strain=1.5)
    metrics['gates']=gates
    metrics['status']='PASS_PHYSICS' if all(gates.values()) else 'FAIL_PHYSICS'
    (a.out/'physics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(dict(status=metrics['status'],gates=gates,metrics={k:metrics[k] for k in ('max_ground_penetration_m','max_relative_displacement_m','max_edge_strain','max_speed_m_s')})),flush=True)

if __name__=='__main__': main()
