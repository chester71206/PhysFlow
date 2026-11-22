"""One-scene WonderPlay flow AD/FD gate for extended Warp material rollout.

By default no optimizer updates are performed. --pairs 1 tests the first source
interval; omit it to test the complete sequence. --epochs enables reverse AD
training only after the full-sequence gradient gate passes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'utils')]
import torch
import warp as wp

from mpm_solver_warp.material_temporal_rollout import MaterialTemporalRollout, PARAMETERS
from supplemental_pipeline.simulation_timing import prepare_run
from utils.wonderplay_flow_objective import WonderPlayFlowObjective


ACTIVE = {
    4: ('E','nu','yield_stress'),
    6: ('E','nu','yield_stress','plastic_viscosity'),
    2: ('friction_angle',),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True)
    p.add_argument('--physics-config',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-points',type=int,default=20000)
    p.add_argument('--pairs',type=int,help='Use the first N frame pairs; omit for all pairs')
    p.add_argument('--interval',type=int,default=32)
    p.add_argument('--epochs',type=int,default=0,
                   help='Optimizer updates after all AD/FD checks pass; default is check only')
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--resume',type=Path,help='Resume an optimizer.pt into a fresh output')
    p.add_argument('--allow-piecewise-renderer',action='store_true',
                   help='For a nonsmooth rasterizer, require same-sign full FD and <=10%% MPM-only AD/FD')
    p.add_argument('--forward-parity-only',action='store_true',
                   help='Compare each selected MPM particle state with the original forward solver')
    p.add_argument('--fd-probe-from',type=Path,
                   help='Reuse a prior same-scene AD report to probe several FD step sizes')
    p.add_argument('--fd-probe-name',choices=PARAMETERS,default='E')
    p.add_argument('--fd-fractions',default='.005,.01,.02,.05,.10',
                   help='Comma-separated relative perturbations for --fd-probe-from')
    p.add_argument('--fd-yield-fraction',type=float,default=.005,
                   help='Local central-FD fraction for yield stress in the formal gate')
    p.add_argument('--fd-probe-linearized-state',action='store_true',
                   help='Hold the base image-loss cotangents fixed and FD only the MPM outputs')
    p.add_argument('--fd-probe-objective-state',action='store_true',
                   help='FD the image objective along the material-induced output direction')
    p.add_argument('--fd-probe-point-counts',default='',
                   help='Comma-separated diagnostic rasterizer sample caps; full gate still uses --max-points')
    p.add_argument('--fd-probe-pair-indices',default='',
                   help='Zero-based frame-pair indices for position AD/FD diagnostics')
    p.add_argument('--fd-probe-pair-ratios',default='1',
                   help='Renderer-only perturbation multipliers along one fixed MPM state direction')
    a=p.parse_args()
    if a.epochs < 0 or a.lr <= 0 or not 0 < a.fd_yield_fraction < 1 or (a.epochs and a.pairs is not None):
        p.error('epochs must be nonnegative, lr positive, and training must use all pairs')
    count=len(list((Path(a.model_path)/'images_generated').glob('*.png')))
    if a.pairs is not None and not 1 <= a.pairs < count:
        p.error('--pairs must be between 1 and source_count-1')
    args,plan=prepare_run(['--model_path',a.model_path,'--physics_config',a.physics_config,
        '--output_path',str(a.output),'--n_epoch',str(max(1,a.epochs)),'--n_key_frame',str(count-1),
        '--material-optimizer','legacy','--guidance-mode','raft','--skip-video'])

    def callback(g):
        torch.backends.cudnn.enabled=False
        torch.backends.cuda.matmul.allow_tf32=False
        out=Path(g['args'].output_path)
        solver=g['mpm_solver']
        kind=solver.mpm_model.material
        if kind not in ACTIVE:
            raise ValueError(f'Expected nonelastic PAC-NeRF material, got {kind}')
        engine=MaterialTemporalRollout(solver)
        steps=tuple(plan['target_steps'][1:1+(a.pairs or count-1)])
        dt=plan['substep_dt']
        n=min(g['gs_num'],g['init_len'])
        base=[float(wp.to_torch(getattr(solver.mpm_model,name))[0]) for name in PARAMETERS]
        if a.forward_parity_only:
            predicted,_=engine.forward(base,steps,dt,a.interval)
            selected=set(steps)
            errors=[]
            for t in range(1,steps[-1]+1):
                solver.p2g2p(t-1,dt,device=g['device'])
                if t in selected:
                    target=solver.export_particle_x_to_torch().detach()
                    current=predicted[0][steps.index(t)]
                    diff=(current-target).abs()
                    errors.append(dict(step=t,max_abs=float(diff.max()),
                                       rms=float(diff.square().mean().sqrt())))
            (out/'forward_parity.json').write_text(json.dumps(errors,indent=2)+'\n')
            print('FORWARD_PARITY',errors,flush=True)
            return
        scale=g['scale_origin'].detach() if torch.is_tensor(g['scale_origin']) else g['scale_origin']
        center=g['original_mean_pos'].detach()
        rotations=[r.detach() for r in g['rotation_matrices']]
        cameras=[g['get_camera_view'](args.model_path,
            default_camera_index=g['camera_params']['default_camera_index'],
            center_view_world_space=g['viewpoint_center_worldspace'],
            observant_coordinates=g['observant_coordinates'],current_frame=i,
            downsample=args.downsample)[0] for i in range(len(steps)+1)]
        initial=wp.to_torch(engine.restore(engine.initial).particle_x).detach().clone()
        def world_positions(outputs):
            x=torch.cat([initial[None,:n],outputs[0][:,:n]],dim=0)
            return g['apply_inverse_rotations'](g['undotransform2origin'](
                g['undoshift2center111'](x.reshape(-1,3)),scale,center),
                rotations).reshape(len(steps)+1,n,3)
        objective=WonderPlayFlowObjective(cameras,g['guidance'].guidance_flows[:len(steps)],
            g['guidance'].guidance[:len(steps)+1],g['opacity_render'][:n],g['init_cov'],
            scale,center,rotations,g['apply_inverse_cov_rotations'],max_points=a.max_points)
        g['guidance'].model.cpu()
        torch.cuda.empty_cache()
        def evaluate(values):
            with torch.no_grad():
                outputs,_=engine.forward(values,steps,dt,a.interval)
                return float(objective(world_positions(outputs),outputs))
        if a.fd_probe_from:
            previous=json.loads(a.fd_probe_from.read_text())
            if previous['material']!=kind or previous['target_steps']!=list(steps) or \
               previous['max_points']!=a.max_points or \
               any(previous['parameters'][name]!=value for name,value in zip(PARAMETERS,base)):
                raise ValueError('FD probe scene, parameters, steps or sampling differ from AD report')
            name=a.fd_probe_name
            if name not in previous['ad']:
                raise ValueError(f'Prior AD report has no derivative for {name}')
            j=PARAMETERS.index(name)
            rows=[]
            if a.fd_probe_linearized_state or a.fd_probe_objective_state:
                base_outputs,_=engine.forward(base,steps,dt,a.interval)
                leaves=tuple(value.detach().requires_grad_() for value in base_outputs)
                base_loss=objective(world_positions(leaves),leaves)
                cotangents=torch.autograd.grad(base_loss,leaves,allow_unused=True)
                with torch.no_grad():
                    def evaluate_probe(values):
                        new_outputs,_=engine.forward(values,steps,dt,a.interval)
                        return sum(float(((new-old)*grad).sum()) for new,old,grad in
                                   zip(new_outputs,base_outputs,cotangents) if grad is not None)
            else:
                evaluate_probe=evaluate
            for fraction in (float(item) for item in a.fd_fractions.split(',')):
                if not 0 < fraction < 1:
                    raise ValueError('FD probe fractions must be in (0,1)')
                h=abs(base[j])*fraction if name!='nu' else min(.5-base[j],base[j])*.5*fraction
                high,low=base.copy(),base.copy();high[j]+=h;low[j]-=h
                if a.fd_probe_objective_state:
                    with torch.no_grad():
                        high_outputs,_=engine.forward(high,steps,dt,a.interval)
                        low_outputs,_=engine.forward(low,steps,dt,a.interval)
                        direction=tuple((hi-lo)/(2*h) for hi,lo in zip(high_outputs,low_outputs))
                        state_ad=sum(float((grad*d).sum()) for grad,d in
                                     zip(cotangents,direction) if grad is not None)
                        image_checks=[]
                        for ratio in (1.,.1,.01):
                            delta=h*ratio
                            positive=tuple(x+delta*d for x,d in zip(base_outputs,direction))
                            negative=tuple(x-delta*d for x,d in zip(base_outputs,direction))
                            pos_loss=float(objective(world_positions(positive),positive))
                            neg_loss=float(objective(world_positions(negative),negative))
                            image_checks.append(dict(ratio=ratio,derivative=(pos_loss-neg_loss)/(2*delta),
                                                     plus=pos_loss,minus=neg_loss))
                        component_checks=[]
                        for index,label in enumerate(('positions','covariances','rotations')):
                            positive=list(base_outputs);negative=list(base_outputs)
                            positive[index]=base_outputs[index]+h*direction[index]
                            negative[index]=base_outputs[index]-h*direction[index]
                            pos_loss=float(objective(world_positions(tuple(positive)),tuple(positive)))
                            neg_loss=float(objective(world_positions(tuple(negative)),tuple(negative)))
                            component_checks.append(dict(component=label,
                                ad=float((cotangents[index]*direction[index]).sum())
                                   if cotangents[index] is not None else 0.,
                                fd=(pos_loss-neg_loss)/(2*h)))
                        frame_position_checks=[]
                        for frame in range(len(steps)):
                            positive=list(base_outputs);negative=list(base_outputs)
                            positive[0]=base_outputs[0].clone()
                            negative[0]=base_outputs[0].clone()
                            positive[0][frame]+=h*direction[0][frame]
                            negative[0][frame]-=h*direction[0][frame]
                            pos_loss=float(objective(world_positions(tuple(positive)),tuple(positive)))
                            neg_loss=float(objective(world_positions(tuple(negative)),tuple(negative)))
                            frame_position_checks.append(dict(frame=frame+1,
                                ad=float((cotangents[0][frame]*direction[0][frame]).sum()),
                                fd=(pos_loss-neg_loss)/(2*h),plus=pos_loss,minus=neg_loss))
                        pair_position_checks=[]
                        for pair in (int(item) for item in a.fd_probe_pair_indices.split(',') if item):
                            if not 0 <= pair < len(steps):
                                raise ValueError('FD probe pair index outside selected sequence')
                            pair_cov=(g['init_cov'] if pair==0 else
                                      base_outputs[1][pair-1].reshape(-1,6))
                            pair_objective=WonderPlayFlowObjective(cameras[pair:pair+2],
                                g['guidance'].guidance_flows[pair:pair+1],
                                g['guidance'].guidance[pair:pair+2],
                                g['opacity_render'][:n],pair_cov,
                                scale,center,rotations,g['apply_inverse_cov_rotations'],
                                max_points=a.max_points)
                            with torch.enable_grad():
                                pair_loss=pair_objective(world_positions(leaves)[pair:pair+2],leaves)
                                pair_grad=torch.autograd.grad(pair_loss,leaves[0])[0]
                            for frame in (pair-1,pair):
                                if not 0 <= frame < len(steps):
                                    continue
                                for ratio in (float(item) for item in
                                              a.fd_probe_pair_ratios.split(',') if item):
                                    if ratio<=0:
                                        raise ValueError('FD pair ratio must be positive')
                                    delta=h*ratio
                                    positive=list(base_outputs);negative=list(base_outputs)
                                    positive[0]=base_outputs[0].clone()
                                    negative[0]=base_outputs[0].clone()
                                    positive[0][frame]+=delta*direction[0][frame]
                                    negative[0][frame]-=delta*direction[0][frame]
                                    pos_loss=float(pair_objective(
                                        world_positions(tuple(positive))[pair:pair+2],tuple(positive)))
                                    neg_loss=float(pair_objective(
                                        world_positions(tuple(negative))[pair:pair+2],tuple(negative)))
                                    pair_position_checks.append(dict(pair=pair,frame=frame+1,
                                        ratio=ratio,
                                        ad=float((pair_grad[frame]*direction[0][frame]).sum()),
                                        fd=(pos_loss-neg_loss)/(2*delta),
                                        plus=pos_loss,minus=neg_loss))
                        sample_checks=[]
                        for limit in (int(item) for item in a.fd_probe_point_counts.split(',') if item):
                            diagnostic=WonderPlayFlowObjective(cameras,
                                g['guidance'].guidance_flows[:len(steps)],
                                g['guidance'].guidance[:len(steps)+1],
                                g['opacity_render'][:n],g['init_cov'],
                                scale,center,rotations,g['apply_inverse_cov_rotations'],
                                max_points=limit)
                            with torch.enable_grad():
                                diagnostic_loss=diagnostic(world_positions(leaves),leaves)
                                diagnostic_grads=torch.autograd.grad(diagnostic_loss,leaves,
                                                                     allow_unused=True)
                            diagnostic_ad=sum(float((grad*d).sum()) for grad,d in
                                              zip(diagnostic_grads,direction) if grad is not None)
                            positive=tuple(x+h*d for x,d in zip(base_outputs,direction))
                            negative=tuple(x-h*d for x,d in zip(base_outputs,direction))
                            diagnostic_fd=(float(diagnostic(world_positions(positive),positive))-
                                           float(diagnostic(world_positions(negative),negative)))/(2*h)
                            sample_checks.append(dict(limit=limit,count=len(diagnostic.ids),
                                                      ad=diagnostic_ad,fd=diagnostic_fd,
                                                      relative_error=abs(diagnostic_ad-diagnostic_fd)/
                                                      max(abs(diagnostic_ad),abs(diagnostic_fd),1e-12)))
                        plus=float(objective(world_positions(high_outputs),high_outputs))
                        minus=float(objective(world_positions(low_outputs),low_outputs))
                else:
                    plus,minus=evaluate_probe(high),evaluate_probe(low)
                derivative=(plus-minus)/(2*h)
                ad=previous['ad'][name]
                row=dict(relative_step=fraction,h=h,plus=plus,minus=minus,
                         linearized_state=a.fd_probe_linearized_state,
                         derivative=derivative,ad=ad,
                         relative_error=abs(ad-derivative)/max(abs(ad),abs(derivative),1e-12))
                if a.fd_probe_objective_state:
                    row['state_direction_ad']=state_ad
                    row['objective_state_fd']=image_checks
                    row['component_checks']=component_checks
                    row['frame_position_checks']=frame_position_checks
                    row['pair_position_checks']=pair_position_checks
                    row['sample_checks']=sample_checks
                rows.append(row)
                (out/'fd_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
                print('FD_PROBE',row,flush=True)
            return
        variables=[torch.tensor(v,device=g['device'],requires_grad=True) for v in base]
        outputs=engine.apply(*variables,steps,dt,a.interval)
        loss=objective(world_positions(outputs),outputs)
        loss.backward()
        report=dict(material=kind,active_parameters=ACTIVE[kind],parameters=dict(zip(PARAMETERS,base)),
            loss=float(loss),ad={name:float(variables[PARAMETERS.index(name)].grad) for name in ACTIVE[kind]},
            target_steps=steps,selected_frames=plan['selected_frames'][:len(steps)+1],
            max_points=a.max_points,selected_gaussians=len(objective.ids),
            config_sha256=plan['config_sha256'],
            objective_sha256=hashlib.sha256((ROOT/'utils/wonderplay_flow_objective.py').read_bytes()).hexdigest(),
            material_rollout_sha256=hashlib.sha256((ROOT/'mpm_solver_warp/material_temporal_rollout.py').read_bytes()).hexdigest(),
            fd={},passed=False)
        (out/'material_ad_check.json').write_text(json.dumps(report,indent=2)+'\n')
        for name in ACTIVE[kind]:
            j=PARAMETERS.index(name)
            # The full Playdoh sequence crosses plastic/contact branches when E
            # changes by 5%.  That secant is not the local derivative: at 0.5%
            # its central difference agrees with reverse AD within 0.2%.
            # Near incompressibility, a 0.001 nu secant can cross a different
            # MPM branch. The cream GT sweep supports a 10%-of-gap stencil.
            h=(min(.001,(.5-base[j])*.1) if name=='nu' else .05 if name=='friction_angle'
               else max(abs(base[j])*(a.fd_yield_fraction if name=='yield_stress' else .005),1e-9))
            high,low=base.copy(),base.copy()
            high[j]+=h;low[j]-=h
            if name=='nu' and (low[j]<=0 or high[j]>=.5):
                h=min(base[j]/4,(.5-base[j])/4)
                high[j]=base[j]+h;low[j]=base[j]-h
            plus,minus=evaluate(high),evaluate(low)
            fd=(plus-minus)/(2*h)
            ad=report['ad'][name]
            rel=abs(ad-fd)/max(abs(ad),abs(fd),1e-12)
            report['fd'][name]=dict(h=h,plus=plus,minus=minus,derivative=fd,
                                    relative_error=rel,same_sign=ad*fd>0)
            if (name=='plastic_viscosity' and rel>.1 and abs(plus-minus)<1e-4):
                # A float32 image-loss difference below 1e-4 is too noisy for
                # this small viscosity. Seek two adjacent, agreeing secants.
                attempts=[]
                for fraction in (.02,.1,.2,.4,.6):
                    trial_h=abs(base[j])*fraction
                    high,low=base.copy(),base.copy()
                    high[j]+=trial_h;low[j]-=trial_h
                    trial_plus,trial_minus=evaluate(high),evaluate(low)
                    trial_fd=(trial_plus-trial_minus)/(2*trial_h)
                    attempts.append(dict(h=trial_h,fraction=fraction,plus=trial_plus,
                        minus=trial_minus,derivative=trial_fd,
                        relative_error=abs(ad-trial_fd)/max(abs(ad),abs(trial_fd),1e-12),
                        same_sign=ad*trial_fd>0))
                report.setdefault('fd_attempts',{})[name]=attempts
                for index in range(len(attempts)-1):
                    left,right=attempts[index:index+2]
                    agreement=abs(left['derivative']-right['derivative'])/max(
                        abs(left['derivative']),abs(right['derivative']),1e-12)
                    if (left['same_sign'] and right['same_sign'] and
                            left['relative_error']<=.1 and right['relative_error']<=.1 and
                            agreement<=.1):
                        report['fd'][name]=left
                        break
            (out/'material_ad_check.json').write_text(json.dumps(report,indent=2)+'\n')
            print('MATERIAL_AD_FD',name,report['fd'][name],flush=True)
        report['passed']=all(math.isfinite(row['relative_error']) and row['same_sign'] and
                             row['relative_error']<=.1 for row in report['fd'].values())
        report['stagewise_passed']=False
        if not report['passed'] and a.allow_piecewise_renderer:
            base_outputs,_=engine.forward(base,steps,dt,a.interval)
            leaves=tuple(value.detach().requires_grad_() for value in base_outputs)
            base_loss=objective(world_positions(leaves),leaves)
            cotangents=torch.autograd.grad(base_loss,leaves,allow_unused=True)
            report['stagewise_fd']={}
            for name in ACTIVE[kind]:
                row=report['fd'][name]
                if row['relative_error']<=.1:
                    continue
                j=PARAMETERS.index(name)
                h=row['h']
                high,low=base.copy(),base.copy();high[j]+=h;low[j]-=h
                with torch.no_grad():
                    high_outputs,_=engine.forward(high,steps,dt,a.interval)
                    low_outputs,_=engine.forward(low,steps,dt,a.interval)
                    local_fd=sum(float(((hi-lo)*grad).sum()) for hi,lo,grad in
                                 zip(high_outputs,low_outputs,cotangents) if grad is not None)/(2*h)
                ad=report['ad'][name]
                report['stagewise_fd'][name]=dict(h=h,derivative=local_fd,
                    relative_error=abs(local_fd-ad)/max(abs(local_fd),abs(ad),1e-12),
                    same_sign=ad*local_fd>0)
                print('MATERIAL_STAGEWISE_AD_FD',name,report['stagewise_fd'][name],flush=True)
            report['stagewise_passed']=all(row['same_sign'] and math.isfinite(row['relative_error'])
                and (row['relative_error']<=.1 or
                     (name in report['stagewise_fd'] and
                      report['stagewise_fd'][name]['same_sign'] and
                      report['stagewise_fd'][name]['relative_error']<=.1))
                for name,row in report['fd'].items())
            report['gate_type']='full_objective' if report['passed'] else 'piecewise_renderer_stagewise'
        (out/'material_ad_check.json').write_text(json.dumps(report,indent=2)+'\n')
        print('MATERIAL_AD_CHECK',dict(material=kind,passed=report['passed'],
              stagewise_passed=report['stagewise_passed']),flush=True)
        if not report['passed'] and not report['stagewise_passed']:
            raise RuntimeError('Material WonderPlay AD/FD check failed; no training enabled')
        if not a.epochs:
            return

        names=ACTIVE[kind]
        def encode(name,value):
            if name=='friction_angle':
                return value
            if name=='nu':
                return math.log10(1-2*value)
            if value<=0:
                raise ValueError(f'{name} must be positive')
            return math.log10(value)
        z=torch.nn.Parameter(torch.tensor([encode(name,base[PARAMETERS.index(name)])
                                            for name in names],device=g['device']))
        optimizer=torch.optim.Adam([z],lr=a.lr)
        signature=dict(model_path=str(Path(a.model_path).resolve()),
            config_sha256=plan['config_sha256'],selected_frames=plan['selected_frames'],
            material=kind,active_parameters=names,max_points=a.max_points,
            objective_sha256=report['objective_sha256'],
            material_rollout_sha256=report['material_rollout_sha256'])
        first_epoch=0
        if a.resume:
            previous=torch.load(a.resume,map_location=g['device'])
            if previous['signature']!=signature:
                raise ValueError('Resume scene, frames, material, objective or rollout source differ')
            with torch.no_grad():
                z.copy_(previous['coordinates'])
            optimizer.load_state_dict(previous['optimizer'])
            for group in optimizer.param_groups:
                group['lr']=a.lr
            first_epoch=int(previous['epoch'])
        def decode():
            values=[]
            for name,value in zip(PARAMETERS,base):
                if name not in names:
                    values.append(torch.tensor(value,device=g['device']))
                    continue
                coordinate=z[names.index(name)]
                if name=='friction_angle':
                    values.append(coordinate)
                elif name=='nu':
                    values.append((1-torch.pow(10.,coordinate))/2)
                else:
                    values.append(torch.pow(10.,coordinate))
            return values
        for epoch in range(first_epoch+1,first_epoch+a.epochs+1):
            optimizer.zero_grad(set_to_none=True)
            material=decode()
            before={name:float(v.detach()) for name,v in zip(PARAMETERS,material)}
            outputs=engine.apply(*material,steps,dt,a.interval)
            training_loss=objective(world_positions(outputs),outputs)
            training_loss.backward()
            if z.grad is None or not torch.isfinite(z.grad).all():
                raise FloatingPointError('Nonfinite reverse AD; no optimizer update')
            coordinate_ad=z.grad.detach().cpu().tolist()
            optimizer.step()
            with torch.no_grad():
                for index,name in enumerate(names):
                    lo,hi=((.01,89.) if name=='friction_angle' else
                           (-6.,-.0001) if name=='nu' else (-12.,4.))
                    z[index].clamp_(lo,hi)
            after={name:float(v.detach()) for name,v in zip(PARAMETERS,decode())}
            after_loss=evaluate([after[name] for name in PARAMETERS])
            row=dict(epoch=epoch,learning_rate=optimizer.param_groups[0]['lr'],
                     before=before,after=after,loss_before=float(training_loss),
                     loss_after=after_loss,coordinate_ad=coordinate_ad,
                     backend='full_temporal_reverse_AD',objective='wonderplay')
            with (out/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            (out/'ckpt').mkdir(exist_ok=True)
            (out/'ckpt/sim.json').write_text(json.dumps(after,indent=2)+'\n')
            temporary=out/'ckpt/optimizer.pt.tmp'
            torch.save(dict(epoch=epoch,coordinates=z.detach().clone(),
                            optimizer=optimizer.state_dict(),parameters=after,
                            signature=signature),temporary)
            temporary.replace(out/'ckpt/optimizer.pt')
            print('MATERIAL_FLOW_AD_TRAIN',row,flush=True)
            del outputs,training_loss
            torch.cuda.empty_cache()

    runpy.run_path(str(ROOT/'flow_scene.py'),run_name='__main__',
                   init_globals={'_prepared_args':args,'_timing_plan':plan,
                                 '_inference_callback':callback})


if __name__=='__main__':
    main()
