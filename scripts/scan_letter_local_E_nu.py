"""Measure image metrics around one fixed Letter checkpoint in solver E/nu.

The first axis varies E with nu fixed. The second varies nu with E fixed.
An optional local grid varies both. Every point uses the same independent
full-frame native-renderer and RAFT evaluation; there is no training update.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.score_universal_pacnerf_re import score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,content):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def centered(count):
    middle=count//2
    return sorted(range(count),key=lambda i:(abs(i-middle),i))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'data/pacnerf/letter_material_starts/projected_flow_ad.json')
    p.add_argument('--physics-config',type=Path,default=ROOT/'config/gradient_repair/letter_flow_ad.json')
    p.add_argument('--kinematics-manifest',type=Path,default=ROOT/'data/pacnerf/letter_material_starts/kinematics_manifest.json')
    p.add_argument('--model-path',type=Path,default=ROOT/'model/pacnerf/letter')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/letter_local_E_nu')
    p.add_argument('--reuse-sweep',type=Path,default=ROOT/'outputs/letter_material_to_gt')
    p.add_argument('--e-points',type=int,default=41,help='Odd count, evaluated center outward')
    p.add_argument('--e-span',type=float,default=.25,help='E/E0 from 1-span to 1+span')
    p.add_argument('--nu-points',type=int,default=21,help='Odd count, evaluated center outward')
    p.add_argument('--nu-gap-fold',type=float,default=4.,
                   help='0.5-nu varies by 1/fold to fold around the current gap')
    p.add_argument('--grid-points',type=int,default=7,help='Odd count for local E/nu grid')
    p.add_argument('--grid-e-span',type=float,default=.15)
    p.add_argument('--grid-nu-gap-fold',type=float,default=2.)
    p.add_argument('--max-new',type=int,default=10,help='New GPU evaluations this invocation; zero imports only')
    p.add_argument('--retry-failed',action='store_true',help='Retry points that failed in an earlier invocation')
    p.add_argument('--gpu',default=os.environ.get('CUDA_VISIBLE_DEVICES','0'))
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    if any(n<3 or n%2!=1 for n in (a.e_points,a.nu_points,a.grid_points)):
        p.error('All point counts must be odd and at least three')
    if not (0<a.e_span<1 and 0<a.grid_e_span<1 and a.nu_gap_fold>1 and a.grid_nu_gap_fold>1 and a.max_new>=0):
        p.error('Invalid scan range or --max-new')
    checkpoint,config,kin,model,out=(x.resolve() for x in
        (a.checkpoint,a.physics_config,a.kinematics_manifest,a.model_path,a.output))
    start=json.loads(checkpoint.read_text())
    E0,nu0=float(start['E']),float(start['nu'])
    if E0<=0 or not 0<nu0<.5:
        p.error('Expected E>0 and 0<nu<0.5 in the checkpoint')
    scale=float(json.loads(kin.read_text())['scale_origin'])
    signature=dict(checkpoint_sha256=sha(checkpoint),config_sha256=sha(config),
        evaluator_sha256=sha(ROOT/'scripts/evaluate_universal_reference.py'),
        model_path=str(model),kinematics_sha256=sha(kin),scale_origin=scale,
        e_points=a.e_points,e_span=a.e_span,nu_points=a.nu_points,
        nu_gap_fold=a.nu_gap_fold,grid_points=a.grid_points,
        grid_e_span=a.grid_e_span,grid_nu_gap_fold=a.grid_nu_gap_fold)
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:
        raise ValueError('Existing output uses a different checkpoint or scan design; change --output')
    atomic(manifest,json.dumps(signature,indent=2)+'\n')
    points={}

    def add(E,nu,axis,index,x):
        if E<=0 or not 0<nu<.5:
            raise ValueError(f'Invalid E/nu point: {E},{nu}')
        key=(round(E,16),round(nu,12))
        member=dict(axis=axis,index=index,x=x)
        if key in points:
            points[key]['memberships'].append(member)
            return
        params=start.copy();params.update(E=E,nu=nu)
        # Preserve the exact saved checkpoint at the center for reuse.
        if abs(E-E0)<1e-16 and abs(nu-nu0)<1e-12:params=start.copy()
        ident=f'p{len(points):04d}'
        folder=out/ident;folder.mkdir(exist_ok=True)
        material=folder/'parameters.json'
        content=json.dumps(params,indent=2)+'\n'
        if material.exists() and material.read_text()!=content:
            raise ValueError(f'Existing material differs: {material}')
        atomic(material,content)
        re=score('letter',params,scale)
        points[key]=dict(id=ident,E=E,nu=nu,mu=re['physical']['mu'],kappa=re['physical']['kappa'],
            mu_RE_pct=100*re['per_parameter_RE']['mu'],
            kappa_RE_pct=100*re['per_parameter_RE']['kappa'],
            mean_RE_pct=100*re['mean_RE'],memberships=[member],status='planned')

    # Center and immediate ±E changes are evaluated first.
    for i in centered(a.e_points):
        fraction=-a.e_span+2*a.e_span*i/(a.e_points-1)
        add(E0*(1+fraction),nu0,'E_only',i,fraction)
    gap=.5-nu0
    for i in centered(a.nu_points):
        t=-1+2*i/(a.nu_points-1)
        add(E0,.5-gap*a.nu_gap_fold**t,'nu_only',i,t)
    grid=[]
    for i in range(a.grid_points):
        for j in range(a.grid_points):
            grid.append((i,j))
    middle=a.grid_points//2
    for i,j in sorted(grid,key=lambda ij:((ij[0]-middle)**2+(ij[1]-middle)**2,ij)):
        e_fraction=-a.grid_e_span+2*a.grid_e_span*i/(a.grid_points-1)
        t=-1+2*j/(a.grid_points-1)
        add(E0*(1+e_fraction),.5-gap*a.grid_nu_gap_fold**t,
            'grid',i*a.grid_points+j,f'{i},{j}')
    items=list(points.values())

    previous=[]
    if (a.reuse_sweep/'status.json').exists():
        prior=a.reuse_sweep.resolve()
        old=json.loads((prior/'manifest.json').read_text())
        if old['config_sha256']!=signature['config_sha256'] or abs(old['scale_origin']-scale)>1e-12:
            raise ValueError('Prior sweep used a different config or physical scale')
        for row in json.loads((prior/'status.json').read_text())['samples']:
            if row['status']!='complete':continue
            folder=Path(row['path'])
            evaluation=Path(row.get('reused_evaluation') or row.get('evaluation_path') or folder/'evaluation')
            if (evaluation/'evaluation.json').exists():
                previous.append((json.loads((folder/'parameters.json').read_text()),evaluation))

    def accept(item,evaluation):
        result=json.loads((evaluation/'evaluation.json').read_text())
        if (result.get('status')!='complete' or not result.get('finite') or
            result.get('scene')!='letter' or result.get('dataset')!='pacnerf' or
            result.get('config_sha256')!=signature['config_sha256']):
            raise ValueError(f'Incompatible evaluation: {evaluation}')
        params=json.loads((out/item['id']/'parameters.json').read_text())
        if any(not math.isclose(result['parameters'][name],params[name],rel_tol=1e-5,abs_tol=1e-10)
               for name in ('E','nu')):
            raise ValueError(f'Evaluation parameters differ: {evaluation}')
        item.update(status='complete',evaluation_path=str(evaluation),
            foreground_EPE_px=result['epe']['foreground_px'],whole_EPE_px=result['epe']['whole_px'],
            flow_half_MSE=result['epe']['half_mse'],PSNR_db=result['psnr_db'])

    def publish():
        fields=['id','status','E','nu','mu','kappa','mu_RE_pct','kappa_RE_pct','mean_RE_pct',
                'foreground_EPE_px','whole_EPE_px','flow_half_MSE','PSNR_db','evaluation_path','error']
        with (out/'results.csv.tmp').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
            writer.writeheader();writer.writerows(items)
        (out/'results.csv.tmp').replace(out/'results.csv')
        atomic(out/'status.json',json.dumps(items,indent=2)+'\n')
        lines=['# Letter local E/nu scan','',
            f'Fixed center: E={E0:.9g}, nu={nu0:.9g}. Same native-renderer/RAFT evaluation for every point.',
            f'Completed {sum(x["status"]=="complete" for x in items)}/{len(items)} unique points.',
            'E_only holds nu fixed. nu_only holds E fixed. The grid changes both.',
            '', '| ID | Axis | Coordinate | E/E0 | nu | μ RE % | κ RE % | FG EPE | Whole EPE | PSNR | Status |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for item in items:
            for m in item['memberships']:
                lines.append(f"| {item['id']} | {m['axis']} | {m['x']} | {item['E']/E0:.5g} | {item['nu']:.8g} | "
                    f"{item['mu_RE_pct']:.4g} | {item['kappa_RE_pct']:.4g} | "
                    f"{item.get('foreground_EPE_px',float('nan')):.4g} | "
                    f"{item.get('whole_EPE_px',float('nan')):.4g} | "
                    f"{item.get('PSNR_db',float('nan')):.4g} | {item['status']} |")
        atomic(out/'results.md','\n'.join(lines)+'\n')

    publish()
    if a.plan_only:
        print(f'PLANNED {len(items)} unique points; {out}/results.md')
        return
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),WANDB_MODE='disabled',
        PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    launched=0
    for item in items:
        folder=out/item['id']
        completed=[candidate for candidate in [folder/'evaluation',*sorted(folder.glob('evaluation_retry*'))]
                   if (candidate/'evaluation.json').exists()]
        if completed:
            accept(item,completed[-1]);publish();continue
        params=json.loads((folder/'parameters.json').read_text())
        match=next((evaluation for old,evaluation in previous if
                    all(math.isclose(old[name],params[name],rel_tol=1e-10,abs_tol=1e-14)
                        for name in ('E','nu'))),None)
        if match is not None:
            accept(item,match);publish();continue
        failure=folder/'failure.json'
        if failure.exists() and not a.retry_failed:
            item.update(status='failed',error=json.loads(failure.read_text())['error'])
            publish();continue
        if launched>=a.max_new:continue
        evaluation=folder/'evaluation'
        if evaluation.exists():
            n=1
            while (folder/f'evaluation_retry{n:02d}').exists():n+=1
            evaluation=folder/f'evaluation_retry{n:02d}'
        command=[sys.executable,str(ROOT/'scripts/evaluate_universal_reference.py'),
            '--model-path',str(model),'--physics-config',str(config),'--output',str(evaluation),
            '--checkpoint-json',str(folder/'parameters.json'),'--dataset','pacnerf']
        item['status']='evaluating';publish()
        log_path=folder/(evaluation.name+'.log')
        with log_path.open('w') as log:
            exit_code=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        launched+=1
        if exit_code:
            item.update(status='failed',error=f'exit={exit_code}; see {log_path}')
            atomic(failure,json.dumps(dict(error=item['error'],command=command),indent=2)+'\n')
            publish();continue
        accept(item,evaluation);publish()
        if failure.exists():failure.unlink()
        print('COMPLETE',item['id'],f"E/E0={item['E']/E0:.5g}",
              f"EPE={item['foreground_EPE_px']:.5g}",f"PSNR={item['PSNR_db']:.5g}",flush=True)

    finished=[item for item in items if item['status']=='complete']
    if finished:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        metrics=(('foreground_EPE_px','Foreground EPE ↓'),('whole_EPE_px','Whole EPE ↓'),
                 ('PSNR_db','PSNR ↑'),('mean_RE_pct','Mean material RE % ↓'))
        for axis,filename in (('E_only','E_only_curves.png'),('nu_only','nu_only_curves.png')):
            for ax,(name,title) in zip(axes.flat,metrics):
                points=sorted(((s['E']/E0 if axis=='E_only' else s['nu'],s)
                               for s in finished if any(m['axis']==axis for m in s['memberships'])),
                              key=lambda row:row[0])
                ax.clear()
                if points:ax.plot([x for x,_ in points],[s[name] for _,s in points],'.-')
                ax.axvline(1. if axis=='E_only' else nu0,color='black',alpha=.35)
                ax.set(title=title,xlabel='E/E0 (nu fixed)' if axis=='E_only' else 'nu (E fixed)')
                ax.grid(alpha=.25)
            fig.savefig(out/filename,dpi=180)
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
        for ax,(name,title) in zip(axes,(('foreground_EPE_px','FG EPE ↓'),('PSNR_db','PSNR ↑'))):
            field=np.full((a.grid_points,a.grid_points),np.nan)
            for s in finished:
                for member in s['memberships']:
                    if member['axis']=='grid':
                        i,j=map(int,member['x'].split(','));field[i,j]=s[name]
            im=ax.imshow(field,origin='lower',aspect='auto')
            ax.set(title=title,xlabel='nu gap grid index',ylabel='E grid index')
            fig.colorbar(im,ax=ax)
        fig.savefig(out/'local_grid.png',dpi=180);plt.close(fig)
        center=next((s for s in finished if math.isclose(s['E'],E0,rel_tol=0,abs_tol=1e-16)
                    and math.isclose(s['nu'],nu0,rel_tol=0,abs_tol=1e-12)),None)
        report=['# Local E/nu response','',
            f'Center: E={E0:.9g}, nu={nu0:.9g}. Completed {len(finished)}/{len(items)} points.',
            'E-only changes hold nu and all non-material settings fixed.',
            'Measured points are connected by straight lines; unmeasured intervals are not proven smooth.',
            '', '| E/E0 | ΔPSNR dB | ΔFG EPE px | Δmean RE % |',
            '|---:|---:|---:|---:|']
        if center:
            e_axis=sorted((s for s in finished if any(m['axis']=='E_only' for m in s['memberships'])),key=lambda s:s['E'])
            for s in e_axis:
                report.append(f"| {s['E']/E0:.5g} | {s['PSNR_db']-center['PSNR_db']:+.5g} | "
                    f"{s['foreground_EPE_px']-center['foreground_EPE_px']:+.5g} | "
                    f"{s['mean_RE_pct']-center['mean_RE_pct']:+.5g} |")
            improves=[s for s in e_axis if s['E']>E0 and (
                s['PSNR_db']>center['PSNR_db']+.05 or
                s['foreground_EPE_px']<center['foreground_EPE_px']-.01)]
            report += ['',f'Sampled points with E>E0 and PSNR gain >0.05 dB or FG EPE drop >0.01 px: {len(improves)}.']
        atomic(out/'analysis.md','\n'.join(report)+'\n')
    print(f'COMPLETE {len(finished)}/{len(items)}; new this call={launched}; {out}/analysis.md')


if __name__=='__main__':
    main()
