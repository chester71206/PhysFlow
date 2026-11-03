"""Full-rollout reverse AD with observed-flow / projected-motion supervision.

An explicitly separate optical-flow objective, NOT a repair claim for
RAFT(rendered RGB). Observed RAFT fields and reference visibility are fixed;
material -> complete MPM -> projected 2D motion is differentiated exactly.
Finite differences only gate correctness; Adam alone updates parameters.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'utils')]
import torch
import warp as wp
from supplemental_pipeline.simulation_timing import prepare_run
from mpm_solver_warp.repaired_temporal_rollout import RepairedTemporalRollout
from utils.projected_flow_objective import FrozenVisibilityProjectedFlow

p = argparse.ArgumentParser()
p.add_argument('--model-path', required=True)
p.add_argument('--physics-config', required=True)
p.add_argument('--output', required=True)
p.add_argument('--epochs', type=int, default=3)
p.add_argument('--interval', type=int, default=32)
p.add_argument('--lr', type=float, help='Default 0.003 for a new run; preserve checkpoint lr on resume unless specified')
p.add_argument('--max-points', type=int, default=20000)
p.add_argument('--check-only', action='store_true', help='Run AD/FD checks without any parameter updates')
p.add_argument('--resume', type=Path, help='Resume optimizer.pt into a new output; epochs means additional updates')
p.add_argument('--check-after-training', action='store_true', help='Repeat three-scale material AD/FD checks at the final checkpoint')
p.add_argument('--capture-observations', type=Path, help='Save fixed projected states/targets for observation-only replay')
a = p.parse_args()
if a.lr is not None and (not math.isfinite(a.lr) or a.lr<=0):
    p.error('lr must be finite and positive')
count = len(list((Path(a.model_path)/'images_generated').glob('*.png')))
args, plan = prepare_run(['--model_path', a.model_path, '--physics_config', a.physics_config,
    '--output_path', a.output, '--n_epoch', str(a.epochs), '--n_key_frame', str(count-1),
    '--material-optimizer', 'legacy', '--guidance-mode', 'raft', '--skip-video'])
plan['gradient_note'] = 'Full temporal reverse AD, projected physical flow vs fixed observed RAFT; no search.'


def callback(g):
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(g['args'].output_path)
    frozen=json.loads((ROOT/'release/frozen_mpm.json').read_text())
    extended={}
    def check_frozen():
        for path,digest in {**frozen,**extended}.items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Frozen MPM implementation changed: '+path)
    check_frozen()
    engine = RepairedTemporalRollout(g['mpm_solver'])
    steps, dt = tuple(plan['target_steps'][1:]), plan['substep_dt']
    if g['mpm_solver'].mpm_model.material not in (0,5):
        raise ValueError('Projected-flow horizon gates cover elastic and Newtonian models; other materials remain unvalidated')
    if g['mpm_solver'].mpm_model.material==5 and g['mpm_solver'].mpm_model.newtonian_rate_model!=1:
        raise ValueError('Long-horizon Newtonian gates cover the strain-rate model; legacy fluid stress is not validated here')
    names = ('E', 'nu', 'yield_stress')
    base = [float(wp.to_torch(getattr(g['mpm_solver'].mpm_model,n))[0]) for n in names]
    resume=torch.load(a.resume,map_location=g['device']) if a.resume else None
    signature=dict(model_path=str(Path(a.model_path).resolve()),config_sha256=plan['config_sha256'],
        frames=plan['selected_frames'],frozen_mpm=frozen,
        objective_sha256=hashlib.sha256((ROOT/'utils/projected_flow_objective.py').read_bytes()).hexdigest())
    if resume is not None and resume['signature']!=signature:
        raise ValueError('Resume model/config/frames/physics/objective differ from the checkpoint')
    n = min(g['gs_num'],g['init_len'])
    # These are fixed scene calibration values, originally computed from the
    # Gaussian initialization. Reusing their construction graph across epochs
    # caused the second-backward exception even with correct MPM adjoints.
    scale = g['scale_origin'].detach() if torch.is_tensor(g['scale_origin']) else g['scale_origin']
    center = g['original_mean_pos'].detach()
    rotations = [r.detach() for r in g['rotation_matrices']]
    cameras = [g['get_camera_view'](args.model_path,
        default_camera_index=g['camera_params']['default_camera_index'],
        center_view_world_space=g['viewpoint_center_worldspace'],
        observant_coordinates=g['observant_coordinates'], current_frame=i,
        downsample=args.downsample)[0] for i in range(len(steps)+1)]
    initial = wp.to_torch(engine.restore(engine.initial).particle_x).detach().clone()

    def world_positions(outputs):
        x = torch.cat([initial[None,:n],outputs[0][:,:n]],dim=0)
        return g['apply_inverse_rotations'](g['undotransform2origin'](
            g['undoshift2center111'](x.reshape(-1,3)),scale,center),
            rotations).reshape(len(steps)+1,n,3)

    baseline=None
    if resume is None:
        with torch.no_grad():
            # No reverse is requested for the reference/FD forwards. Keep
            # no intermediate CPU checkpoints for these untaped rollouts.
            baseline, _ = engine.forward(base,steps,dt,steps[-1])
            objective = FrozenVisibilityProjectedFlow(world_positions(baseline), cameras,
                g['guidance'].guidance_flows, g['guidance'].guidance,
                g['opacity_render'], max_points=a.max_points)
    else:
        # Restore the fixed observations directly. Re-simulating a reference
        # just to overwrite its visibility wastes a complete physics rollout.
        objective=FrozenVisibilityProjectedFlow.__new__(FrozenVisibilityProjectedFlow)
        objective.cameras=cameras
        objective.flow=g['guidance'].guidance_flows.detach()
        objective.height,objective.width=objective.flow.shape[-2:]
        objective.ids=resume['visibility']['ids'].to(g['device'])
        objective.masks=[m.to(g['device']) for m in resume['visibility']['masks']]
        objective.counts=[int(m.sum()) for m in objective.masks]
        base=[float(x) for x in resume['parameters']]
    if a.capture_observations:
        a.capture_observations.mkdir(parents=True,exist_ok=False)
        torch.save(dict(flow=objective.flow.cpu(),masks=[m.cpu() for m in objective.masks],
            original_particle_ids=objective.ids.cpu(),projection=[c.full_proj_transform.cpu() for c in cameras],
            view=[c.world_view_transform.cpu() for c in cameras],signature=signature,
            note='Position files already select original_particle_ids; replay with local ids arange(N).'),a.capture_observations/'metadata.pt')
    def capture(positions,values,value,adjoint=None):
        if not a.capture_observations:return
        parameters=[float(x) for x in values]
        key=hashlib.sha256(json.dumps(parameters).encode()).hexdigest()[:16]
        target=a.capture_observations/(key+'.pt')
        if target.exists():return
        temporary=target.with_suffix('.tmp')
        torch.save(dict(parameters=parameters,positions=positions[:,objective.ids].detach().cpu(),loss=value,
            position_adjoint=adjoint[:,objective.ids].detach().cpu() if adjoint is not None else None),temporary)
        temporary.replace(target)
    # Target extraction is finished. Release RAFT; it is never differentiated
    # by this explicitly named alternative objective.
    g['guidance'].model.cpu()
    torch.cuda.empty_cache()

    def forward(values):
        with torch.no_grad():
            outputs,_ = engine.forward(values,steps,dt,steps[-1])
            positions=world_positions(outputs);value=float(objective(positions))
            capture(positions,values,value)
            return value

    params = [torch.tensor(v,device=g['device'],requires_grad=True) for v in base]
    ad_outputs=engine.apply(*params,steps,dt,a.interval)
    ad_positions=world_positions(ad_outputs)
    if a.capture_observations:ad_positions.retain_grad()
    loss = objective(ad_positions)
    loss.backward()
    capture(ad_positions,base,float(loss),ad_positions.grad if a.capture_observations else None)
    ad = [float(v.grad) if v.grad is not None else 0. for v in params]
    report = dict(objective='projected physical flow vs observed fixed RAFT flow',
        check_only=a.check_only,optimizer_updates_requested=0 if a.check_only else a.epochs,
        resumed_from=str(a.resume) if a.resume else None,
        checkpoint_epoch=int(resume['epoch']) if resume is not None else 0,
        original_rendered_raft_objective=False, parameters=dict(zip(names,base)),
        loss=float(loss), ad=dict(zip(names,ad)), steps=steps, dt=dt,
        visibility='frozen initial-trajectory foreground and nearest-depth selection',
        visible_points_per_pair=objective.counts, finite_difference={},
        optimizer='torch.optim.Adam; reverse AD only; no search or FD updates',
        p2g_primal=os.environ.get('FLOW_P2G_PRIMAL','fixed'),
        g2p_reduction=os.environ.get('FLOW_G2P_REDUCTION','fixed'),
        sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['scripts/train_projected_flow_gradient.py','utils/projected_flow_objective.py',
             'mpm_solver_warp/repaired_temporal_rollout.py','mpm_solver_warp/analytic_p2g_adjoint.py',
             'mpm_solver_warp/analytic_g2p_adjoint.py']})

    def save():
        (out/'gradient_check.json').write_text(json.dumps(report,indent=2)+'\n')

    print('PROJECTED_FLOW_AD',report,flush=True)
    save()
    for j,name in enumerate(names[:2]):
        h = base[j]*.01 if j==0 else min(.001,(.5-base[j])*.05)
        rows = []
        for factor in (2.,1.,.5):
            plus,minus = base.copy(),base.copy()
            plus[j] += h*factor
            minus[j] -= h*factor
            lp,lm = forward(plus),forward(minus)
            fd = (lp-lm)/(2*h*factor)
            rows.append(dict(h=h*factor,plus=lp,minus=lm,ad=ad[j],fd=fd,
                same_sign=fd*ad[j]>0,relative_error=abs(fd-ad[j])/max(abs(fd),abs(ad[j]),1e-20)))
            report['finite_difference'][name] = rows
            save()
            print('PROJECTED_FLOW_FD',name,rows[-1],flush=True)
    stencils = [r for rows in report['finite_difference'].values() for r in rows]
    report['passed'] = all(r['same_sign'] and r['relative_error']<=.1 for r in stencils) and __import__('statistics').median(r['relative_error'] for r in stencils)<=.05
    save()
    if not report['passed']:
        raise RuntimeError('Projected-flow material gradient gate failed; no updates applied')
    if a.check_only:
        return
    del loss,params,baseline,ad_outputs,ad_positions
    z = torch.nn.Parameter(torch.tensor([math.log10(base[0]),math.log10(1-2*base[1])],device=g['device']))
    optimizer = torch.optim.Adam([z],lr=.003 if a.lr is None else a.lr)
    first_epoch=0
    if resume is not None:
        with torch.no_grad():z.copy_(resume['log_parameters'])
        optimizer.load_state_dict(resume['optimizer'])
        if a.lr is not None:
            for group in optimizer.param_groups:group['lr']=a.lr
        first_epoch=int(resume['epoch'])
    for epoch in range(a.epochs):
        check_frozen()
        optimizer.zero_grad(set_to_none=True)
        vals = [10**z[0],(1-10**z[1])/2,torch.tensor(base[2],device=g['device'])]
        loss = objective(world_positions(engine.apply(*vals,steps,dt,a.interval)))
        loss.backward()
        if not torch.isfinite(z.grad).all():
            raise FloatingPointError('Nonfinite reverse AD; no update applied')
        grad = z.grad.detach().cpu().tolist()
        before = [float(v.detach()) for v in vals]
        optimizer.step()
        with torch.no_grad():
            z[0].clamp_(-7,-.4)
            z[1].clamp_(-6,-.0001)
        after = [float(10**z[0]),float((1-10**z[1])/2),base[2]]
        row = dict(epoch=first_epoch+epoch+1,learning_rate=optimizer.param_groups[0]['lr'],before=dict(zip(names,before)),after=dict(zip(names,after)),
            loss_before=float(loss),loss_after=forward(after),log_coordinate_ad=grad)
        with (out/'training.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        (out/'ckpt').mkdir(exist_ok=True)
        (out/'ckpt/sim.json').write_text(json.dumps(dict(zip(names,after)),indent=2)+'\n')
        temporary=out/'ckpt/optimizer.pt.tmp'
        torch.save(dict(epoch=first_epoch+epoch+1,parameters=after,log_parameters=z.detach(),optimizer=optimizer.state_dict(),
            signature=signature,visibility=dict(ids=objective.ids,masks=objective.masks)),temporary)
        temporary.replace(out/'ckpt/optimizer.pt')
        print('PROJECTED_FLOW_TRAIN',row,flush=True)
        del loss
        torch.cuda.empty_cache()
    check_frozen()
    if a.check_after_training:
        test_params=[torch.tensor(v,device=g['device'],requires_grad=True) for v in after]
        test_loss=objective(world_positions(engine.apply(*test_params,steps,dt,a.interval)))
        gradients=[float(x) for x in torch.autograd.grad(test_loss,test_params)]
        final_check=dict(parameters=dict(zip(names,after)),loss=float(test_loss),ad=dict(zip(names,gradients)),finite_difference={})
        for j,name in enumerate(names[:2]):
            h=after[j]*.01 if j==0 else min(.001,(.5-after[j])*.05)
            rows=[]
            for factor in (2.,1.,.5):
                plus,minus=after.copy(),after.copy();plus[j]+=h*factor;minus[j]-=h*factor
                lp,lm=forward(plus),forward(minus);fd=(lp-lm)/(2*h*factor)
                rows.append(dict(h=h*factor,AD=gradients[j],FD=fd,same_sign=gradients[j]*fd>0,relative_error=abs(gradients[j]-fd)/max(abs(gradients[j]),abs(fd),1e-20)))
                final_check['finite_difference'][name]=rows
                (out/'post_gradient_check.json').write_text(json.dumps(final_check,indent=2)+'\n')
                print('POST_TRAINING_FD',name,rows[-1],flush=True)
        stencils=[r for rows in final_check['finite_difference'].values() for r in rows]
        final_check['passed']=all(r['same_sign'] and r['relative_error']<=.1 for r in stencils) and __import__('statistics').median(r['relative_error'] for r in stencils)<=.05
        (out/'post_gradient_check.json').write_text(json.dumps(final_check,indent=2)+'\n')
        if not final_check['passed']:
            raise RuntimeError('Final-checkpoint gradient check failed; checkpoint retained for diagnosis')


runpy.run_path(str(ROOT/'flow_scene.py'),run_name='__main__',
    init_globals={'_prepared_args':args,'_timing_plan':plan,'_inference_callback':callback})
