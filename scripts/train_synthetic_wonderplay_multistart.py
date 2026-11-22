"""Exact-flow WonderPlay multi-start with the existing PhysFlow Warp reverse AD.

The observed target is captured from the unchanged WonderPlay Gaussian-flow
rasterizer using saved native-MPM GT positions. Blender masks select foreground;
RAFT is never initialized. Each start saves an optimizer checkpoint per update.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import warp as wp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gs.utils.graphics_utils import getProjectionMatrix
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.repaired_temporal_rollout import RepairedTemporalRollout
from utils.projected_flow_objective import project
from utils.wonderplay_flow_objective import WonderPlayFlowObjective


def sha(data):return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def atomic_json(path,value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def camera_from_blender(record,width,height,device):
    c2w=np.asarray(record['c2w'],np.float32)
    W=np.linalg.inv(c2w)
    # Blender camera is +X right,+Y up,-Z forward. 3DGS is +X right,+Y down,+Z forward.
    flip=np.diag([1.,-1.,-1.,1.]).astype(np.float32)
    view=flip@W
    K=np.asarray(record['intrinsic'],np.float32)*(width/480)
    fx,fy=float(K[0,0]),float(K[1,1])
    fovx=2*math.atan(width/(2*fx));fovy=2*math.atan(height/(2*fy))
    wv=torch.as_tensor(view.T.copy(),device=device)
    proj=getProjectionMatrix(.01,100.,fovx,fovy).T.to(device)
    return SimpleNamespace(FoVx=fovx,FoVy=fovy,world_view_transform=wv,
                           full_proj_transform=wv@proj,camera_center=torch.as_tensor(c2w[:3,3],device=device))


def capture_flow(objective,positions,outputs):
    original=objective.rasterizers
    captured=[]
    class Capture:
        def __init__(self,rasterizer):self.rasterizer=rasterizer
        def __call__(self,*args,**kwargs):
            result=self.rasterizer(*args,**kwargs)
            captured.append(result[4][:2].detach())
            return result
    objective.rasterizers=[Capture(r) for r in original]
    try:objective(positions,outputs)
    finally:objective.rasterizers=original
    return torch.stack(captured)


def blender_interval_flow(scene,view,first,last,resolution):
    height=width=480
    yy,xx=np.mgrid[:height,:width].astype(np.float32)
    total=np.zeros((height,width,2),np.float32)
    valid=np.ones((height,width),bool)
    for frame in range(first,last):
        item=np.load(scene/'mesh_visible_flow'/f'cam_{view}'/f'{frame:04d}.npz')
        mapx=xx+total[:,:,0];mapy=yy+total[:,:,1]
        field=cv2.remap(item['flow_px'],mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        visibility=cv2.remap(item['valid'],mapx,mapy,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)>0
        valid &= visibility
        total+=field
    factor=resolution/480
    small=cv2.resize(total,(resolution,resolution),interpolation=cv2.INTER_AREA)*factor
    mask=cv2.resize(valid.astype(np.uint8),(resolution,resolution),interpolation=cv2.INTER_NEAREST)>0
    return torch.as_tensor(small.transpose(2,0,1).copy()),torch.as_tensor(mask)


def plot(history,out,gt_e,gt_nu):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not history:return
    x=[r['epoch'] for r in history]
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    axes[0].plot(x,[r['loss'] for r in history],marker='o',ms=2)
    axes[0].set_yscale('log');axes[0].set_title('Exact-flow WonderPlay loss')
    axes[1].plot(x,[r['E_Pa'] for r in history],label='E');axes[1].axhline(gt_e,color='k',ls='--',label='GT')
    axes[1].set_title('E (Pa)');axes[1].legend()
    axes[2].plot(x,[r['nu'] for r in history],label='ν');axes[2].axhline(gt_nu,color='k',ls='--',label='GT')
    axes[2].set_title('ν');axes[2].legend()
    for ax in axes:ax.set_xlabel('optimizer update');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'loss_parameters.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5))
    axes[0].plot(x,[r['E_relative_error'] for r in history]);axes[0].set_title('E relative error')
    axes[1].plot(x,[r['nu_absolute_error'] for r in history]);axes[1].set_title('ν absolute error')
    for ax in axes:ax.set_xlabel('optimizer update');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'accuracy.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--start-name',default='probe')
    ap.add_argument('--start-E-ratio',type=float,default=1.35)
    ap.add_argument('--start-nu',type=float,default=.27)
    ap.add_argument('--epochs',type=int,default=0)
    ap.add_argument('--max-frame',type=int,default=8)
    ap.add_argument('--frame-stride',type=int,default=2)
    ap.add_argument('--resolution',type=int,default=128)
    ap.add_argument('--views',default='0,1,2')
    ap.add_argument('--checkpoint-interval',type=int,default=16)
    ap.add_argument('--lr-E-ratio',type=float,default=.04)
    ap.add_argument('--E-parameterization',choices=('linear','log'),default='linear',
                    help='log optimizes log(E/GT_E), suitable for orders-of-magnitude starts')
    ap.add_argument('--lr-log-E',type=float,default=.08)
    ap.add_argument('--E-min-ratio',type=float,default=.2)
    ap.add_argument('--E-max-ratio',type=float,default=3.)
    ap.add_argument('--start-probe',action='store_true',
                    help='save a finite forward/loss check at the initial parameters')
    ap.add_argument('--lr-nu',type=float,default=.008)
    ap.add_argument('--lr-decay-at',default='50,75',help='optimizer update numbers at which to halve learning rates')
    ap.add_argument('--lr-decay-factor',type=float,default=.5)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--gradient-gate',action='store_true')
    ap.add_argument('--fd-E-fraction',type=float,default=.05)
    ap.add_argument('--fd-nu-step',type=float,default=.01)
    ap.add_argument('--target-kind',choices=('gaussian-exact','blender-mesh'),default='gaussian-exact')
    ap.add_argument('--allow-nonzero-gt',action='store_true')
    ap.add_argument('--gt-gradient',action='store_true')
    ap.add_argument('--local-sensitivity',action='store_true',help='evaluate equal normalized E and nu perturbations around GT')
    a=ap.parse_args()
    if a.resolution<32 or a.epochs<0 or a.frame_stride<1 or a.max_frame<1:ap.error('invalid resolution/epochs/frames')
    if not 0<a.E_min_ratio<1<a.E_max_ratio or not a.E_min_ratio<a.start_E_ratio<a.E_max_ratio or not .05<a.start_nu<.48:
        ap.error('start must lie strictly inside the optimizer parameter bounds')
    if a.lr_log_E<=0:ap.error('lr-log-E must be positive')
    decay_at={int(item) for item in a.lr_decay_at.split(',') if item.strip()}
    if min(decay_at,default=1)<1 or not 0<a.lr_decay_factor<=1:ap.error('invalid learning-rate decay schedule')
    if not 0<a.fd_E_fraction<.5 or not 0<a.fd_nu_step<.1:ap.error('invalid finite-difference probe step')
    a.output.mkdir(parents=True,exist_ok=True)
    torch.backends.cudnn.enabled=False
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((a.scene/'config.json').read_text())
    native=np.load(a.scene/'trajectory.npz')
    all_positions=native['positions'].astype(np.float32)
    initial=all_positions[0]
    volume=native['initial_volume'].astype(np.float32)
    ns=int(native['n_surface'])
    frames=list(range(0,a.max_frame+1,a.frame_stride))
    if frames[-1]!=a.max_frame:frames.append(a.max_frame)
    if frames[-1]>=len(all_positions):ap.error('max-frame exceeds saved trajectory')
    steps=tuple(int(round(float(native['times'][i])/cfg['dt_s'])) for i in frames[1:])
    device=a.device
    wp.init()
    solver=MPM_Simulator_WARP(len(initial),n_grid=cfg['grid_n'],grid_lim=cfg['grid_lim_m'],device=device)
    raw=np.load(cfg['asset']);f=raw['faces'];v=raw['vertices']
    edges=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]))
    median_edge=float(np.median(np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)))
    sigma=median_edge*.75
    cov=np.zeros((len(initial),6),np.float32)
    cov[:,0]=cov[:,3]=cov[:,5]=sigma*sigma
    solver.load_initial_data_from_torch(torch.as_tensor(initial,device=device),
        torch.as_tensor(volume,device=device),torch.as_tensor(cov,device=device),
        n_grid=cfg['grid_n'],grid_lim=cfg['grid_lim_m'],device=device)
    if any(cfg['initial_velocity_m_s']):
        vv=np.tile(cfg['initial_velocity_m_s'],(len(initial),1)).astype(np.float32)
        solver.set_initial_velocity_from_torch(torch.as_tensor(vv,device=device))
    solver.set_parameters_dict(dict(material='elastic',density=cfg['density_kg_m3'],
        E=cfg['E_solver_internal'],nu=cfg['nu'],g=cfg['gravity_m_s2'],
        grid_v_damping_scale=1.),device=device)
    solver.finalize_mu_lam(device=device)
    solver.add_surface_collider((0,0,cfg['ground_z_m']),(0,0,1),'sticky',0)
    engine=RepairedTemporalRollout(solver)
    gt=[cfg['E_solver_internal'],cfg['nu'],0.]
    t0=time.monotonic()
    with torch.no_grad():gt_outputs,_=engine.forward(gt,steps,cfg['dt_s'],a.checkpoint_interval)
    model_positions=torch.cat((torch.as_tensor(initial[None],device=device),gt_outputs[0]),0)
    native_selected=torch.as_tensor(all_positions[frames],device=device)
    parity=float((model_positions-native_selected).abs().max())
    print('FORWARD_PARITY',parity,'seconds',time.monotonic()-t0,flush=True)
    if not math.isfinite(parity) or parity>1e-4:raise RuntimeError(f'Full selected horizon forward mismatch: {parity} m')
    names=[int(x) for x in a.views.split(',')]
    cameras_json=json.loads((a.scene/'render_full/cameras.json').read_text())['cameras']
    masks=[];objectives=[];target_stats=[];projection_errors=[]
    for ci in names:
        cam=camera_from_blender(cameras_json[ci],a.resolution,a.resolution,device)
        # Check our 3DGS projection against Blender's saved perspective K on initial surface points.
        subset=torch.as_tensor(initial[:ns:31],device=device)
        ndc,_=project(subset,cam)
        K=np.asarray(cameras_json[ci]['intrinsic'],np.float32)*(a.resolution/480)
        c2w=np.asarray(cameras_json[ci]['c2w'],np.float32)
        q=np.linalg.inv(c2w)@np.concatenate((subset.cpu().numpy(),np.ones((len(subset),1))),axis=1).T
        px=K[0,0]*q[0]/-q[2]+K[0,2];py=K[1,2]-K[1,1]*q[1]/-q[2]
        render_xy=np.stack((px,py),axis=1)
        gs_xy=np.stack(((ndc[:,0].cpu().numpy()+1)*a.resolution/2,
                        (ndc[:,1].cpu().numpy()+1)*a.resolution/2),axis=1)
        projection_errors.append(float(np.abs(render_xy-gs_xy).max()))
        img=[]
        for fi in frames:
            path=a.scene/'render_full/passes'/f'cam_{ci}'/f'mask_{fi:04d}_{fi+1:04d}.png'
            m=np.asarray(Image.open(path).convert('L').resize((a.resolution,a.resolution),Image.Resampling.NEAREST))>127
            img.append(torch.as_tensor(np.where(m,0.,1.),device=device,dtype=torch.float32).expand(3,-1,-1))
        observed_images=torch.stack(img)
        placeholder=torch.zeros((len(steps),2,a.resolution,a.resolution),device=device)
        opacity=torch.full((ns,1),.95,device=device)
        initial_cov=torch.as_tensor(cov[:ns],device=device)
        objective=WonderPlayFlowObjective([cam]*len(frames),placeholder,observed_images,
            opacity,initial_cov,torch.tensor(1.,device=device),torch.zeros(3,device=device),
            [torch.eye(3,device=device)],lambda c,r:c,max_points=ns)
        if a.target_kind=='gaussian-exact':
            with torch.no_grad():exact=capture_flow(objective,native_selected,gt_outputs)
        else:
            pairs=[blender_interval_flow(a.scene,ci,f0,f1,a.resolution)
                   for f0,f1 in zip(frames[:-1],frames[1:])]
            exact=torch.stack([p[0] for p in pairs]).to(device)
            objective.masks=[m & p[1].to(device) for m,p in zip(objective.masks,pairs)]
            objective.counts=[int(m.sum()) for m in objective.masks]
        objective.flow=exact.detach()
        objectives.append(objective)
        masks.append([int(m.sum()) for m in objective.masks])
        target_stats.append(dict(view=ci,mean_flow_px=float(exact.norm(dim=1).mean()),
                                 max_flow_px=float(exact.norm(dim=1).max()),
                                 foreground_pixels=masks[-1]))
    print('PROJECTION_MAX_ERROR_PX',projection_errors,flush=True)
    if max(projection_errors)>1.:raise RuntimeError('Blender to WonderPlay camera projection mismatch')
    def loss_of(outputs):
        positions=torch.cat((torch.as_tensor(initial[None],device=device),outputs[0]),0)
        return torch.stack([o(positions,outputs) for o in objectives]).mean(),positions
    with torch.no_grad():gt_loss=float(loss_of(gt_outputs)[0])
    gate=dict(scene=str(a.scene),frames=frames,steps=steps,dt_s=cfg['dt_s'],views=names,
        resolution=a.resolution,gt_E_Pa=cfg['E_Pa'],gt_nu=cfg['nu'],
        native_vs_repaired_max_position_error_m=parity,
        max_camera_projection_error_px=max(projection_errors),gaussian_sigma_m=sigma,
        target=target_stats,gt_loss=gt_loss,target_kind=a.target_kind,
        target_definition=('same WonderPlay Gaussian rasterizer; source native GT MPM particle positions; source covariances from GT repaired rollout; fixed Blender object mask'
                           if a.target_kind=='gaussian-exact' else
                           'composed Blender visible-mesh geometric flow with source/destination visibility; fixed Blender object mask; independent of WonderPlay rasterizer'))
    atomic_json(a.output/'gate.json',gate)
    target_file=a.output/'exact_gt_targets.pt'
    if not target_file.exists():
        torch.save(dict(frames=frames,views=names,resolution=a.resolution,
                        flows=[o.flow.detach().cpu() for o in objectives],
                        masks=[[m.detach().cpu() for m in o.masks] for o in objectives],
                        note=gate['target_definition']),target_file)
    print('GT_LOSS',gt_loss,flush=True)
    if not math.isfinite(gt_loss) or (a.target_kind=='gaussian-exact' and gt_loss>1e-3 and not a.allow_nonzero_gt):
        raise RuntimeError('GT loss not near zero; stop before multi-start')
    if a.target_kind=='blender-mesh' and a.epochs and not a.allow_nonzero_gt:
        raise RuntimeError('Blender target has renderer mismatch; explicit --allow-nonzero-gt required before training')
    if a.start_probe:
        probe_path=a.output/f'start_probe_{a.start_name}.json'
        try:
            with torch.no_grad():
                probe_outputs,_=engine.forward([cfg['E_solver_internal']*a.start_E_ratio,a.start_nu,0.],
                                               steps,cfg['dt_s'],a.checkpoint_interval)
                probe_loss,probe_positions=loss_of(probe_outputs)
                finite=bool(torch.isfinite(probe_loss) and torch.isfinite(probe_positions).all())
                max_abs=float(probe_positions.abs().max()) if finite else None
                stable=bool(finite and max_abs<2. and float(probe_loss)<1e8)
                probe=dict(E_Pa=cfg['E_Pa']*a.start_E_ratio,nu=a.start_nu,
                           finite=finite,stable=stable,loss=float(probe_loss) if finite else None,
                           max_abs_position_m=max_abs)
        except Exception as exc:
            probe=dict(E_Pa=cfg['E_Pa']*a.start_E_ratio,nu=a.start_nu,finite=False,error=repr(exc))
        atomic_json(probe_path,probe)
        print('START_PROBE',json.dumps(probe),flush=True)
        if not probe.get('stable',False):raise RuntimeError('Initial forward probe is nonfinite or unstable')
    if a.local_sensitivity:
        # E +/-10% and nu +/-0.05 are comparable controlled perturbations;
        # each candidate uses the same saved initial state, forces, and timestep.
        trial_specs=[('E_minus_10pct',.9*cfg['E_solver_internal'],cfg['nu']),
                     ('E_plus_10pct',1.1*cfg['E_solver_internal'],cfg['nu']),
                     ('nu_minus_0p05',cfg['E_solver_internal'],cfg['nu']-.05),
                     ('nu_plus_0p05',cfg['E_solver_internal'],cfg['nu']+.05)]
        sensitivity={'gt_loss':gt_loss,'target_kind':a.target_kind,'points':{}}
        for name,e_value,nu_value in trial_specs:
            with torch.no_grad():
                prediction,_=engine.forward([e_value,nu_value,0.],steps,cfg['dt_s'],a.checkpoint_interval)
                trial_loss=float(loss_of(prediction)[0])
            sensitivity['points'][name]=dict(E_Pa=e_value*1e7,nu=nu_value,loss=trial_loss,
                excess_loss_over_gt=trial_loss-gt_loss)
        atomic_json(a.output/'local_sensitivity.json',sensitivity)
        print('LOCAL_SENSITIVITY',json.dumps(sensitivity),flush=True)
    if a.gt_gradient:
        gt_e=torch.tensor(cfg['E_solver_internal'],device=device,requires_grad=True)
        gt_nu=torch.tensor(cfg['nu'],device=device,requires_grad=True)
        gt_yield=torch.tensor(0.,device=device)
        gt_ad=engine.apply(gt_e,gt_nu,gt_yield,steps,cfg['dt_s'],a.checkpoint_interval)
        gt_ad_loss=loss_of(gt_ad)[0]
        gt_ad_loss.backward()
        probe=dict(target_kind=a.target_kind,loss=float(gt_ad_loss),
                   gradient_E_internal=float(gt_e.grad),gradient_nu=float(gt_nu.grad),
                   normalized_gradient_logE=float(gt_e.grad)*cfg['E_solver_internal'],
                   normalized_gradient_nu_0p05=float(gt_nu.grad)*.05)
        atomic_json(a.output/'gt_gradient.json',probe)
        print('GT_GRADIENT',json.dumps(probe),flush=True)
    if a.gradient_gate:
        base=[cfg['E_solver_internal']*a.start_E_ratio,a.start_nu,0.]
        variables=[torch.tensor(v,device=device,requires_grad=i<2) for i,v in enumerate(base)]
        ad_outputs=engine.apply(*variables,steps,cfg['dt_s'],a.checkpoint_interval)
        ad_loss=loss_of(ad_outputs)[0]
        ad_loss.backward()
        rows={}
        for j,name in enumerate(('E_internal','nu')):
            h=base[0]*a.fd_E_fraction if j==0 else a.fd_nu_step
            high=base.copy();low=base.copy();high[j]+=h;low[j]-=h
            with torch.no_grad():
                hi=float(loss_of(engine.forward(high,steps,cfg['dt_s'],a.checkpoint_interval)[0])[0])
                lo=float(loss_of(engine.forward(low,steps,cfg['dt_s'],a.checkpoint_interval)[0])[0])
            fd=(hi-lo)/(2*h);ad=float(variables[j].grad)
            rows[name]=dict(ad=ad,fd=fd,plus_loss=hi,minus_loss=lo,
                            same_sign=ad*fd>0,relative_error=abs(ad-fd)/max(abs(ad),abs(fd),1e-12))
        gradient_report=dict(parameters=dict(E_Pa=base[0]*1e7,nu=base[1]),
                             loss=float(ad_loss),derivatives=rows,
                             pass_gate=all(r['same_sign'] and r['relative_error']<=.25 for r in rows.values()))
        atomic_json(a.output/f'gradient_gate_{a.start_name}.json',gradient_report)
        print('GRADIENT_GATE',json.dumps(gradient_report),flush=True)
        if not gradient_report['pass_gate']:raise RuntimeError('Full WonderPlay gradient gate failed')
    if a.epochs==0:return
    start=a.output/a.start_name;start.mkdir(exist_ok=True)
    checkpoint=start/'optimizer.pt'
    signature=dict(scene_sha256=sha(initial),frames=frames,views=names,resolution=a.resolution,
                   sigma=sigma,start_E_ratio=a.start_E_ratio,start_nu=a.start_nu,target_kind=a.target_kind)
    if a.E_parameterization=='log':
        signature.update(E_parameterization='log',E_min_ratio=a.E_min_ratio,
                         E_max_ratio=a.E_max_ratio,lr_log_E=a.lr_log_E)
        E_variable=torch.tensor(math.log(a.start_E_ratio),device=device,requires_grad=True)
        def current_E():return cfg['E_solver_internal']*torch.exp(E_variable)
        E_lr=a.lr_log_E
    else:
        E_variable=torch.tensor(cfg['E_solver_internal']*a.start_E_ratio,device=device,requires_grad=True)
        def current_E():return E_variable
        E_lr=cfg['E_solver_internal']*a.lr_E_ratio
    nu=torch.tensor(a.start_nu,device=device,requires_grad=True)
    ys=torch.tensor(0.,device=device)
    optim=torch.optim.Adam([{'params':[E_variable],'lr':E_lr},
                            {'params':[nu],'lr':a.lr_nu}])
    history=[];begin=0
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location=device)
        old_signature=dict(state['signature'])
        old_signature.setdefault('target_kind','gaussian-exact')
        if old_signature!=signature:raise RuntimeError('Checkpoint signature mismatch')
        with torch.no_grad():
            E_variable.copy_(state['log_E_ratio'] if a.E_parameterization=='log' else state['E'])
            nu.copy_(state['nu'])
        optim.load_state_dict(state['optimizer'])
        history=state['history'];begin=len(history)
        print('RESUME',a.start_name,begin,flush=True)
    for epoch in range(begin,a.epochs):
        if epoch in decay_at:
            for group in optim.param_groups:group['lr']*=a.lr_decay_factor
            print('LR_DECAY',epoch,[group['lr'] for group in optim.param_groups],flush=True)
        tic=time.monotonic();optim.zero_grad(set_to_none=True)
        E_now=current_E()
        outputs=engine.apply(E_now,nu,ys,steps,cfg['dt_s'],a.checkpoint_interval)
        loss,positions=loss_of(outputs)
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {epoch}')
        loss.backward()
        grad_log_E=float(E_variable.grad) if a.E_parameterization=='log' else float(E_variable.grad*E_now)
        grad_E=float(E_variable.grad/E_now) if a.E_parameterization=='log' else float(E_variable.grad)
        grad_nu=float(nu.grad)
        if not all(map(math.isfinite,(grad_E,grad_nu))):raise RuntimeError(f'Nonfinite gradient at {epoch}')
        loss_before=float(loss.detach())
        particle_epe=float((positions.detach()-native_selected).norm(dim=-1).mean())
        optim.step()
        with torch.no_grad():
            E_before_clamp=float(current_E())
            boundary_hit_E=bool(E_before_clamp<=cfg['E_solver_internal']*a.E_min_ratio or
                                E_before_clamp>=cfg['E_solver_internal']*a.E_max_ratio)
            boundary_hit_nu=bool(nu<=.05 or nu>=.48)
            if a.E_parameterization=='log':
                E_variable.clamp_(math.log(a.E_min_ratio),math.log(a.E_max_ratio))
            else:
                E_variable.clamp_(cfg['E_solver_internal']*a.E_min_ratio,
                                  cfg['E_solver_internal']*a.E_max_ratio)
            nu.clamp_(.05,.48)
        E_after=float(current_E().detach())
        row=dict(epoch=epoch+1,loss=loss_before,E_Pa=E_after*1e7,
                 nu=float(nu.detach()),E_relative_error=abs(E_after/cfg['E_solver_internal']-1),
                 nu_absolute_error=abs(float(nu.detach())-cfg['nu']),
                 particle_epe_m=particle_epe,grad_E_internal=grad_E,grad_log_E=grad_log_E,grad_nu=grad_nu,
                 boundary_hit_E=boundary_hit_E,boundary_hit_nu=boundary_hit_nu,
                 update_seconds=time.monotonic()-tic)
        history.append(row)
        tmp=checkpoint.with_suffix('.tmp')
        torch.save(dict(signature=signature,E=current_E().detach(),
                        log_E_ratio=E_variable.detach() if a.E_parameterization=='log' else None,
                        nu=nu.detach(),optimizer=optim.state_dict(),history=history),tmp)
        os.replace(tmp,checkpoint)
        with (start/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerows(history)
        plot(history,start,cfg['E_Pa'],cfg['nu'])
        print('UPDATE',a.start_name,json.dumps(row),flush=True)
    if history:
        with torch.no_grad():
            final_outputs,_=engine.forward([float(current_E()),float(nu),0.],steps,cfg['dt_s'],a.checkpoint_interval)
            final_loss,final_positions=loss_of(final_outputs)
        final_E=float(current_E())
        summary=dict(start_name=a.start_name,target_kind=a.target_kind,updates=len(history),
            initial_loss=history[0]['loss'],final_loss=float(final_loss),
            initial_E_Pa=cfg['E_Pa']*a.start_E_ratio,initial_nu=a.start_nu,
            final_E_Pa=final_E*1e7,final_nu=float(nu),
            final_E_relative_error=abs(final_E/cfg['E_solver_internal']-1),
            final_nu_absolute_error=abs(float(nu)-cfg['nu']),
            any_boundary_hit_E=any(r.get('boundary_hit_E',False) for r in history),
            any_boundary_hit_nu=any(r.get('boundary_hit_nu',False) for r in history),
            final_particle_epe_m=float((final_positions-native_selected).norm(dim=-1).mean()))
        atomic_json(start/'final.json',summary)
        print('FINAL',json.dumps(summary),flush=True)


if __name__=='__main__':main()
