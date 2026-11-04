"""Serial, resumable Letter material landscape scan using the independent evaluator.

This is a diagnostic intervention, not optimizer training. All metric comparisons
use the same full-frame native renderer + RAFT protocol at a fixed scene/config.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.score_universal_pacnerf_re import score
from scripts.sweep_letter_material_to_gt import internal


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(value)
    tmp.replace(path)


def linspace(start, stop, count):
    return [start + (stop-start)*i/(count-1) for i in range(count)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = ROOT/'data/pacnerf/letter_material_starts'
    p.add_argument('--original', type=Path, default=base/'original.json')
    p.add_argument('--projected', type=Path, default=base/'projected_flow_ad.json')
    p.add_argument('--start', action='append', default=[], metavar='LABEL=MU,KAPPA',
                   help='Additional physical-parameter start; may be repeated')
    p.add_argument('--physics-config', type=Path, default=ROOT/'config/gradient_repair/letter_flow_ad.json')
    p.add_argument('--kinematics-manifest', type=Path, default=base/'kinematics_manifest.json')
    p.add_argument('--model-path', type=Path, default=ROOT/'model/pacnerf/letter')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/letter_material_landscape')
    p.add_argument('--reuse-sweep', type=Path, help='Import matching evaluations from the prior seven-point sweep')
    p.add_argument('--path-points', type=int, default=25,
                   help='25 includes every point of the previous 7-point paths')
    p.add_argument('--ray-points', type=int, default=9)
    p.add_argument('--grid-points', type=int, default=5)
    p.add_argument('--fold', type=float, default=2., help='Rays/grid span each parameter by ×1/fold to ×fold')
    p.add_argument('--max-new', type=int, default=10, help='New GPU evaluations per invocation; 0 imports only')
    p.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if min(a.path_points, a.ray_points, a.grid_points) < 3 or min(a.path_points, a.ray_points, a.grid_points)%2 != 1:
        p.error('Point counts must be odd and at least three')
    if not (a.fold > 1 and math.isfinite(a.fold)) or a.max_new < 0:
        p.error('--fold must be >1 and --max-new must be >=0')
    config, model, kin, out = (x.resolve() for x in
                               (a.physics_config, a.model_path, a.kinematics_manifest, a.output))
    scale = float(json.loads(kin.read_text())['scale_origin'])
    gt = json.loads((ROOT/'data/pacnerf/groundtruth.json').read_text())['scenes']['letter']['parameters']
    starts = {'original': json.loads(a.original.read_text()),
              'projected': json.loads(a.projected.read_text())}
    custom={}
    for spec in a.start:
        try:
            label, values=spec.split('=',1)
            mu,kappa=map(float,values.split(','))
        except ValueError as exc:
            p.error(f'Invalid --start {spec!r}: expected LABEL=MU,KAPPA ({exc})')
        if (not label or not all(c.isalnum() or c in '_-' for c in label) or
            label in starts or label=='GT' or not all(math.isfinite(x) and x>0 for x in (mu,kappa))):
            p.error(f'Invalid start label or physical parameters: {spec!r}')
        starts[label]=internal(mu,kappa,scale)
        custom[label]=dict(mu=mu,kappa=kappa)
    physical = {name:score('letter',params,scale)['physical'] for name,params in starts.items()}
    physical['GT'] = gt
    signature = dict(config_sha256=digest(config), model_path=str(model), scale_origin=scale,
        kin_sha256=digest(kin), gt_sha256=digest(ROOT/'data/pacnerf/groundtruth.json'),
        original_sha256=digest(a.original), projected_sha256=digest(a.projected),
        evaluator_sha256=digest(ROOT/'scripts/evaluate_universal_reference.py'),
        path_points=a.path_points, ray_points=a.ray_points,
        grid_points=a.grid_points, fold=a.fold, custom_starts=custom)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Existing output has a different design; choose a new --output')
    atomic(manifest, json.dumps(signature,indent=2)+'\n')

    # Each (mu,kappa) is evaluated once even when it belongs to multiple curves.
    samples, by_key = [], {}
    def add(mu, kappa, group, anchor, coordinate, x, index):
        key = (round(math.log(mu),12),round(math.log(kappa),12))
        member = dict(group=group,anchor=anchor,coordinate=coordinate,x=x,index=index)
        if key in by_key:
            by_key[key]['memberships'].append(member)
            return
        params = (starts[anchor].copy() if anchor in starts and
                  abs(mu-physical[anchor]['mu'])/mu < 1e-12 and
                  abs(kappa-physical[anchor]['kappa'])/kappa < 1e-12
                  else internal(mu,kappa,scale))
        ident = f's{len(samples):04d}'
        item = dict(id=ident,mu=mu,kappa=kappa,E=params['E'],nu=params['nu'],
            mu_RE_pct=100*abs(mu-gt['mu'])/gt['mu'],
            kappa_RE_pct=100*abs(kappa-gt['kappa'])/gt['kappa'],
            mean_RE_pct=50*(abs(mu-gt['mu'])/gt['mu']+abs(kappa-gt['kappa'])/gt['kappa']),
            memberships=[member],status='planned')
        path = out/ident
        path.mkdir(exist_ok=True)
        checkpoint = path/'parameters.json'
        content = json.dumps(params,indent=2)+'\n'
        if checkpoint.exists() and checkpoint.read_text()!=content:
            raise ValueError(f'Existing checkpoint changed: {checkpoint}')
        atomic(checkpoint,content)
        samples.append(item);by_key[key]=item

    # Log-space interpolation gives dense, physically positive paths to GT.
    for anchor in starts:
        initial=physical[anchor]
        for i,alpha in enumerate(linspace(0.,1.,a.path_points)):
            mu=math.exp((1-alpha)*math.log(initial['mu'])+alpha*math.log(gt['mu']))
            kappa=math.exp((1-alpha)*math.log(initial['kappa'])+alpha*math.log(gt['kappa']))
            add(mu,kappa,'path',anchor,'to_GT',alpha,i)

    # Four signed directions around each start and GT; negative t moves in
    # the reverse direction. This can expose better image metrics at worse RE.
    directions={'mu_only':(1,0),'kappa_only':(0,1),'same':(1,1),'opposite':(1,-1)}
    for anchor, center in physical.items():
        for coordinate,(dmu,dkappa) in directions.items():
            for i,t in enumerate(linspace(-math.log(a.fold),math.log(a.fold),a.ray_points)):
                add(center['mu']*math.exp(dmu*t),center['kappa']*math.exp(dkappa*t),
                    'ray',anchor,coordinate,t,i)

    # Small two-dimensional neighborhood for curvature and local optima.
    for i,u in enumerate(linspace(-math.log(a.fold),math.log(a.fold),a.grid_points)):
        for j,v in enumerate(linspace(-math.log(a.fold),math.log(a.fold),a.grid_points)):
            add(gt['mu']*math.exp(u),gt['kappa']*math.exp(v),'grid','GT',
                f'{i},{j}',float(i*a.grid_points+j),i*a.grid_points+j)

    previous=[]
    if a.reuse_sweep:
        prior=a.reuse_sweep.resolve()
        old=json.loads((prior/'manifest.json').read_text())
        if old['config_sha256']!=signature['config_sha256'] or abs(old['scale_origin']-scale)>1e-12:
            raise ValueError('Prior sweep used a different config or physical scale')
        status=json.loads((prior/'status.json').read_text())
        for row in status['samples']:
            if row['status']!='complete':
                continue
            old_checkpoint=Path(row['path'])/'parameters.json'
            old_evaluation=Path(row.get('reused_evaluation') or row.get('evaluation_path') or Path(row['path'])/'evaluation')
            if (old_evaluation/'evaluation.json').exists():
                previous.append((json.loads(old_checkpoint.read_text()),old_evaluation))

    def accept(item, evaluation):
        result=json.loads((evaluation/'evaluation.json').read_text())
        if result.get('status')!='complete' or not result.get('finite') or result.get('dataset')!='pacnerf' or result.get('scene')!='letter':
            raise ValueError(f'Invalid evaluation: {evaluation}')
        if result['config_sha256']!=signature['config_sha256']:
            raise ValueError(f'Different physics config: {evaluation}')
        checkpoint=out/item['id']/'parameters.json'
        expected=json.loads(checkpoint.read_text())
        if any(abs(result['parameters'][name]-expected[name]) > max(1e-10,abs(expected[name])*1e-5)
               for name in ('E','nu')):
            raise ValueError(f'Different material parameters: {evaluation}')
        item.update(status='complete',evaluation_path=str(evaluation),
            foreground_EPE_px=result['epe']['foreground_px'],whole_EPE_px=result['epe']['whole_px'],
            flow_half_MSE=result['epe']['half_mse'],PSNR_db=result['psnr_db'])

    def publish():
        fields=['id','status','mu','kappa','E','nu','mu_RE_pct','kappa_RE_pct','mean_RE_pct',
                'foreground_EPE_px','whole_EPE_px','flow_half_MSE','PSNR_db','evaluation_path','error']
        with (out/'results.csv.tmp').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
            writer.writeheader();writer.writerows(samples)
        (out/'results.csv.tmp').replace(out/'results.csv')
        atomic(out/'status.json',json.dumps(samples,indent=2)+'\n')
        lines=['# Letter material landscape','',
            f'Unique points: {len(samples)}; complete: {sum(s["status"]=="complete" for s in samples)}.',
            'Lower EPE and RE, higher PSNR are better. Each point is a full independent evaluation.',
            '', '| ID | Group | Anchor | Direction | x | RE % | FG EPE | PSNR | Status |',
            '|---|---|---|---|---:|---:|---:|---:|---|']
        for s in samples:
            for m in s['memberships']:
                lines.append(f"| {s['id']} | {m['group']} | {m['anchor']} | {m['coordinate']} | {m['x']:.4g} | {s['mean_RE_pct']:.4g} | "
                    f"{s.get('foreground_EPE_px',float('nan')):.4g} | {s.get('PSNR_db',float('nan')):.4g} | {s['status']} |")
        atomic(out/'results.md','\n'.join(lines)+'\n')

    publish()
    if a.plan_only:
        print(f'PLANNED {len(samples)} unique full evaluations; {out}/results.md')
        return
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),WANDB_MODE='disabled',PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    launched=0
    for item in samples:
        point=out/item['id']
        existing=[path for path in [point/'evaluation',*sorted(point.glob('evaluation_retry*'))]
                  if (path/'evaluation.json').exists()]
        if existing:
            accept(item,existing[-1]);publish();continue
        intended=json.loads((point/'parameters.json').read_text())
        match=next((evaluation for params,evaluation in previous
                    if all(math.isclose(params[name],intended[name],rel_tol=1e-10,abs_tol=1e-14)
                           for name in ('E','nu'))),None)
        if match is not None:
            accept(item,match);publish();continue
        if launched>=a.max_new:
            continue
        evaluation=point/'evaluation'
        if evaluation.exists():
            n=1
            while (point/f'evaluation_retry{n:02d}').exists():n+=1
            evaluation=point/f'evaluation_retry{n:02d}'
        command=[sys.executable,str(ROOT/'scripts/evaluate_universal_reference.py'),
            '--model-path',str(model),'--physics-config',str(config),'--output',str(evaluation),
            '--checkpoint-json',str(point/'parameters.json'),'--dataset','pacnerf']
        item['status']='evaluating';publish()
        with (point/'evaluation.log').open('w') as stream:
            rc=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
        launched+=1
        if rc:
            item.update(status='failed',error=f'exit={rc}; see {point}/evaluation.log')
            publish();break
        accept(item,evaluation);publish()
        print('COMPLETE',item['id'],f"RE={item['mean_RE_pct']:.4g}%",
              f"EPE={item['foreground_EPE_px']:.4g}",f"PSNR={item['PSNR_db']:.4g}",flush=True)

    finished=[s for s in samples if s['status']=='complete']
    if finished:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        metrics=[('foreground_EPE_px','Foreground EPE ↓'),('PSNR_db','PSNR ↑'),
                 ('whole_EPE_px','Whole EPE ↓'),('mean_RE_pct','Mean material RE % ↓')]
        for anchor in starts:
            rows=sorted(((m['x'],s) for s in finished for m in s['memberships']
                         if m['group']=='path' and m['anchor']==anchor),key=lambda row:row[0])
            if not rows:continue
            for ax,(name,title) in zip(axes.flat,metrics):
                ax.plot([x for x,_ in rows],[s[name] for _,s in rows],'.-',label=anchor)
                ax.set(title=title,xlabel='Fraction of log-physical path to GT');ax.grid(alpha=.3)
        axes[0,0].legend();fig.savefig(out/'path_curves.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(len(physical),2,figsize=(12,4*len(physical)),layout='constrained')
        for row,anchor in enumerate(physical):
            for direction in directions:
                points=sorted(((m['x']/math.log(a.fold),s) for s in finished for m in s['memberships']
                               if m['group']=='ray' and m['anchor']==anchor and m['coordinate']==direction),
                              key=lambda pair:pair[0])
                if not points:continue
                for column,(name,title) in enumerate((('foreground_EPE_px','FG EPE ↓'),('PSNR_db','PSNR ↑'))):
                    axes[row,column].plot([t for t,_ in points],[s[name] for _,s in points],'.-',label=direction)
                    axes[row,column].set(title=f'{anchor}: {title}',xlabel='Signed log-fold from anchor (−1 to +1)')
                    axes[row,column].axvline(0,color='black',alpha=.25)
                    axes[row,column].grid(alpha=.25)
        axes[0,0].legend();fig.savefig(out/'direction_curves.png',dpi=180);plt.close(fig)
        import numpy as np
        fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
        for ax,(name,title) in zip(axes,(('foreground_EPE_px','FG EPE ↓'),('PSNR_db','PSNR ↑'))):
            grid=np.full((a.grid_points,a.grid_points),np.nan)
            for s in finished:
                for m in s['memberships']:
                    if m['group']=='grid':
                        i,j=map(int,m['coordinate'].split(','));grid[i,j]=s[name]
            im=ax.imshow(grid,origin='lower',extent=[-1,1,-1,1],aspect='auto')
            ax.set(title=title,xlabel='log-fold κ from GT',ylabel='log-fold μ from GT')
            fig.colorbar(im,ax=ax)
        fig.savefig(out/'gt_neighborhood.png',dpi=180);plt.close(fig)
        # Directly report counterexamples to image/material agreement.
        anchors={}
        for name in physical:
            candidates=[s for s in finished if any(m['group']=='ray' and m['anchor']==name and abs(m['x'])<1e-12
                                             for m in s['memberships'])]
            if candidates:anchors[name]=candidates[0]
        counter=[]
        for s in finished:
            for m in s['memberships']:
                if m['group']!='ray' or m['anchor'] not in anchors or abs(m['x'])<1e-12:continue
                ref=anchors[m['anchor']]
                if s['mean_RE_pct']>ref['mean_RE_pct']+1e-4 and (
                    s['PSNR_db']>ref['PSNR_db']+.05 or
                    s['foreground_EPE_px']<ref['foreground_EPE_px']-.01):
                    counter.append(dict(sample=s['id'],anchor=m['anchor'],direction=m['coordinate'],
                        x=m['x'],RE_delta_pct=s['mean_RE_pct']-ref['mean_RE_pct'],
                        PSNR_delta_db=s['PSNR_db']-ref['PSNR_db'],
                        foreground_EPE_delta_px=s['foreground_EPE_px']-ref['foreground_EPE_px']))
        atomic(out/'counterexamples.json',json.dumps(counter,indent=2)+'\n')
        report=['# Letter landscape analysis','',
            f'Completed {len(finished)} of {len(samples)} unique points. This is a finite sample; smoothness between points is not proven.',
            'Path order increases α toward GT. EPE should decrease and PSNR should increase for a monotonic image metric.',
            '', '| Start | Adjacent pairs measured | FG EPE uphill | PSNR downhill | Lowest sampled FG EPE α | Highest sampled PSNR α |',
            '|---|---:|---:|---:|---:|---:|']
        for anchor in starts:
            rows=sorted(((m['index'],m['x'],s) for s in finished for m in s['memberships']
                         if m['group']=='path' and m['anchor']==anchor),key=lambda row:row[0])
            adjacent=[(left,right) for left,right in zip(rows,rows[1:]) if right[0]==left[0]+1]
            epe_up=sum(right[2]['foreground_EPE_px']>left[2]['foreground_EPE_px']+1e-4
                       for left,right in adjacent)
            psnr_down=sum(right[2]['PSNR_db']<left[2]['PSNR_db']-1e-4
                          for left,right in adjacent)
            best_epe=min(rows,key=lambda row:row[2]['foreground_EPE_px'])[1] if rows else float('nan')
            best_psnr=max(rows,key=lambda row:row[2]['PSNR_db'])[1] if rows else float('nan')
            report.append(f'| {anchor} | {len(adjacent)} | {epe_up} | {psnr_down} | {best_epe:.3f} | {best_psnr:.3f} |')
        report += ['',f'Counterexamples with larger material RE and either PSNR > anchor + 0.05 dB or FG EPE < anchor − 0.01 px: {len(counter)}.',
                   'The thresholds exclude tiny numerical differences; see counterexamples.json for every matching point.','',
                   '| Anchor | Direction | Signed fold coordinate | RE increase % | PSNR change dB | FG EPE change px |',
                   '|---|---|---:|---:|---:|---:|']
        for row in counter:
            report.append(f"| {row['anchor']} | {row['direction']} | {row['x']/math.log(a.fold):.3f} | "
                          f"{row['RE_delta_pct']:.4g} | {row['PSNR_delta_db']:.4g} | "
                          f"{row['foreground_EPE_delta_px']:.4g} |")
        atomic(out/'analysis.md','\n'.join(report)+'\n')
    print(f'COMPLETE {len(finished)}/{len(samples)} unique points; new this call={launched}; {out}/results.md')


if __name__=='__main__':
    main()
