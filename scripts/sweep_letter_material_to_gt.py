"""Evaluate a fixed path from Letter material checkpoints to PAC-NeRF GT.

This is a diagnostic sweep: GT chooses points on the path, never gradients or
optimizer updates. Each point uses the same independent native-renderer replay.
Use --plan-only first; rerun with the same arguments to execute/resume.
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
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.score_universal_pacnerf_re import score


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def stamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def internal(mu, kappa, scale):
    # Solver: mu = 1e7 E/[2(1+nu)scale²],
    # kappa = 2mu[nu/(1-2nu)+1/3]. Invert in physical units.
    r = kappa/(2*mu) - 1/3
    nu = r/(1+2*r)
    E = 2*mu*scale**2*(1+nu)/1e7
    if not (math.isfinite(E) and E > 0 and 0 < nu < .5):
        raise ValueError(f'Unrepresentable physical material mu={mu}, kappa={kappa}')
    return {'E': E, 'nu': nu, 'yield_stress': 0.}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', action='append', required=True, metavar='LABEL=CHECKPOINT_JSON',
                   help='May repeat for original and projected-flow AD')
    p.add_argument('--physics-config', type=Path, required=True)
    p.add_argument('--kinematics-manifest', type=Path, required=True,
                   help='Use the common independent-evaluation manifest, not an original-training manifest')
    p.add_argument('--model-path', type=Path, default=ROOT/'model/pacnerf/letter')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--points', type=int, default=7, help='Includes start and GT; default seven')
    p.add_argument('--spacing', choices=('log', 'linear'), default='log',
                   help='Interpolate physical mu and kappa geometrically or linearly')
    p.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--reuse-only', action='store_true',
                   help='Import completed matching evaluations without launching GPU work')
    a = p.parse_args()
    if a.points < 2:
        p.error('--points must be at least 2')
    root = a.output.resolve(); model = a.model_path.resolve(); config = a.physics_config.resolve()
    manifest = a.kinematics_manifest.resolve()
    if model.name != 'letter' or model.parent.name != 'pacnerf':
        p.error('This script is restricted to PAC-NeRF Letter')
    scale = float(json.loads(manifest.read_text())['scale_origin'])
    if not math.isfinite(scale) or scale <= 0:
        p.error('Invalid scale_origin')
    gt_path = ROOT/'data/pacnerf/groundtruth.json'
    gt = json.loads(gt_path.read_text())['scenes']['letter']['parameters']
    starts = {}
    for spec in a.start:
        if '=' not in spec:
            p.error('--start must be LABEL=CHECKPOINT_JSON')
        label, filename = spec.split('=', 1)
        if not label or not all(c.isalnum() or c in '-_' for c in label) or label in starts:
            p.error('Labels must be unique and contain only letters, numbers, - or _')
        source = Path(filename).resolve()
        raw = json.loads(source.read_text())
        starts[label] = dict(path=str(source), sha256=sha(source), params=raw,
                             physical=score('letter', raw, scale)['physical'])
    signature = dict(model_path=str(model), config_sha256=sha(config),
                     manifest_sha256=sha(manifest), scale_origin=scale,
                     gt_sha256=sha(gt_path),
                     evaluator_sha256=sha(ROOT/'scripts/evaluate_universal_reference.py'),
                     starts=starts, points=a.points, spacing=a.spacing)
    root.mkdir(parents=True, exist_ok=True)
    existing = root/'manifest.json'
    if existing.exists() and json.loads(existing.read_text()) != signature:
        raise ValueError('Output belongs to a different sweep; choose a new --output')
    atomic(existing, json.dumps(signature, indent=2)+'\n')
    samples = []
    for label, start in starts.items():
        for index in range(a.points):
            alpha = index/(a.points-1)
            physical = {}
            for name in ('mu','kappa'):
                lo, hi = start['physical'][name], gt[name]
                physical[name] = (lo**(1-alpha)*hi**alpha if a.spacing=='log'
                                  else (1-alpha)*lo+alpha*hi)
            params = start['params'].copy() if index==0 else internal(physical['mu'],physical['kappa'],scale)
            if index==a.points-1:
                physical = dict(gt)
            returned = score('letter',params,scale)['physical']
            if max(abs(returned[k]-physical[k])/physical[k] for k in physical) > 1e-5:
                raise AssertionError('Physical-to-internal conversion mismatch')
            sample_dir = root/label/f'p{index:02d}'
            sample_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = sample_dir/'parameters.json'
            expected = json.dumps(params,indent=2)+'\n'
            if checkpoint.exists() and checkpoint.read_text()!=expected:
                raise ValueError(f'Existing material checkpoint differs: {checkpoint}')
            atomic(checkpoint, expected)
            samples.append(dict(label=label, index=index, alpha=alpha, path=str(sample_dir),
                                E=params['E'], nu=params['nu'], target_mu=physical['mu'],
                                target_kappa=physical['kappa'],
                                mu_RE_pct=100*abs(returned['mu']-gt['mu'])/gt['mu'],
                                kappa_RE_pct=100*abs(returned['kappa']-gt['kappa'])/gt['kappa'],
                                mean_RE_pct=50*(abs(returned['mu']-gt['mu'])/gt['mu']+
                                                abs(returned['kappa']-gt['kappa'])/gt['kappa']),
                                status='planned'))

    def publish():
        fields=['label','index','alpha','status','E','nu','target_mu','target_kappa','actual_mu','actual_kappa',
                'mu_RE_pct','kappa_RE_pct','mean_RE_pct','foreground_EPE_px','whole_EPE_px',
                'flow_half_MSE','PSNR_db','foreground_fraction','background_EPE_px','error']
        with (root/'results.csv.tmp').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
            writer.writeheader();writer.writerows(samples)
        (root/'results.csv.tmp').replace(root/'results.csv')
        atomic(root/'status.json',json.dumps(dict(updated=stamp(),gpu=a.gpu,plan_only=a.plan_only,samples=samples),indent=2)+'\n')
        lines=['# Letter material path to PAC-NeRF GT','',
          f'Physical path: {a.spacing}, {a.points} points per start. Common evaluation scale_origin={scale:.9g}.',
          f'GT: μ={gt["mu"]}, κ={gt["kappa"]}. GT selects path points only; no training/optimizer occurs.',
          'All points use the same native-renderer replay, supplied white RGBA target and RAFT EPE protocol.',
          'EPE is in pixels at 256×480; RE is percent. PSNR averages source frames 1..15.',
          '', '| Start | α | Status | μ RE % | κ RE % | mean RE % | FG EPE | whole EPE | half MSE | PSNR |',
          '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        def fmt(v):
            return '—' if v is None else f'{v:.4g}'
        for s in samples:
            lines.append('| '+ ' | '.join([s['label'],f"{s['alpha']:.3f}",s['status']]+[
                fmt(s.get(k)) for k in ('mu_RE_pct','kappa_RE_pct','mean_RE_pct',
                                         'foreground_EPE_px','whole_EPE_px','flow_half_MSE','PSNR_db')])+' |')
        atomic(root/'results.md','\n'.join(lines)+'\n')
    def analyze(s, evaluation):
        data=json.loads((evaluation/'evaluation.json').read_text())
        if data.get('status')!='complete' or not data.get('finite'):
            raise ValueError('Evaluation incomplete or nonfinite')
        if data['config_sha256']!=signature['config_sha256']:
            raise ValueError('Evaluation used a different physics config')
        expected_sha=sha(Path(s['path'])/'parameters.json')
        source_sha=starts[s['label']]['sha256'] if s['index']==0 else None
        if data.get('checkpoint_sha256') not in {expected_sha,source_sha}:
            raise ValueError('Evaluation used a different material checkpoint')
        if data.get('scene')!='letter' or data.get('dataset')!='pacnerf':
            raise ValueError('Evaluation used a different dataset or scene')
        r=score('letter',data['parameters'],scale)
        s.update(status='complete',actual_mu=r['physical']['mu'],actual_kappa=r['physical']['kappa'],
                 mu_RE_pct=100*r['per_parameter_RE']['mu'],kappa_RE_pct=100*r['per_parameter_RE']['kappa'],
                 mean_RE_pct=100*r['mean_RE'],foreground_EPE_px=data['epe']['foreground_px'],
                 whole_EPE_px=data['epe']['whole_px'],flow_half_MSE=data['epe']['half_mse'],PSNR_db=data['psnr_db'])
        # A saved full-field flow allows a transparent foreground/background split.
        import torch
        flow=torch.load(evaluation/'flow_fields.pt',map_location='cpu')
        err=(flow['simulated_flows']-flow['target_flows']).square().sum(1).sqrt()
        mask=flow['foreground'].bool()
        s['foreground_fraction']=float(mask.float().mean())
        s['background_EPE_px']=float(err[~mask].mean())
        if abs(float(err[mask].mean())-s['foreground_EPE_px'])>1e-6:
            raise AssertionError('Saved EPE and flow field disagree')
        details=Path(s['path'])/'frame_metrics.csv'
        with details.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['frame','PSNR_db','flow_pair','foreground_EPE_px',
                'whole_EPE_px','background_EPE_px'])
            writer.writeheader()
            for i, psnr in enumerate(data['psnr_by_frame']):
                row=dict(frame=i,PSNR_db=psnr)
                if i<len(err):
                    row.update(flow_pair=f'{i}->{i+1}',foreground_EPE_px=float(err[i][mask[i]].mean()),
                               whole_EPE_px=float(err[i].mean()),
                               background_EPE_px=float(err[i][~mask[i]].mean()))
                writer.writerow(row)

    publish()
    if a.plan_only:
        print(root/'results.md')
        return
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),WANDB_MODE='disabled',
        PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for s in samples:
        point=Path(s['path']); evaluation=point/'evaluation'
        completed=[candidate for candidate in [evaluation,*sorted(point.glob('evaluation_retry*'))]
                   if (candidate/'evaluation.json').exists()]
        if completed:
            try:
                analyze(s,completed[-1]);s['evaluation_path']=str(completed[-1]);publish();continue
            except Exception as exc:
                raise RuntimeError(f'Existing evaluation is unusable: {completed[-1]}: {exc}') from exc
        # Reuse a previously completed evaluation of the exact start checkpoint.
        if s['index']==0:
            source=Path(starts[s['label']]['path'])
            prior=source.parent.parent.parent/'evaluation'
            prior_json=prior/'evaluation.json'
            if prior_json.exists():
                old=json.loads(prior_json.read_text())
                if old.get('checkpoint_sha256')==starts[s['label']]['sha256'] and old.get('config_sha256')==signature['config_sha256']:
                    analyze(s,prior);s['reused_evaluation']=str(prior);publish();continue
        if a.reuse_only:
            continue
        # A failed attempt may have created its output before crashing. Give a
        # retry a fresh directory while keeping the failed log and frames.
        if evaluation.exists():
            attempt=1
            while (point/f'evaluation_retry{attempt:02d}').exists():
                attempt+=1
            evaluation=point/f'evaluation_retry{attempt:02d}'
        cmd=[sys.executable,str(ROOT/'scripts/evaluate_universal_reference.py'),
             '--model-path',str(model),'--physics-config',str(config),'--output',str(evaluation),
             '--checkpoint-json',str(point/'parameters.json'),'--dataset','pacnerf']
        s.update(status='evaluating',started=stamp(),command=cmd);publish()
        with (point/'evaluation.log').open('w') as log:
            rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        s['evaluation_exit']=rc
        if rc:
            s.update(status='failed',error=f'evaluation exit={rc}; see evaluation.log');publish();break
        analyze(s,evaluation);s['evaluation_path']=str(evaluation);publish()
        print(stamp(),s['label'],s['index'],'EPE',s['foreground_EPE_px'],'RE%',s['mean_RE_pct'],flush=True)
    # Render diagnostic curves from the completed rows only.
    finished=[s for s in samples if s['status']=='complete']
    if finished:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        metrics=[('foreground_EPE_px','Foreground EPE ↓'),('whole_EPE_px','Whole EPE ↓'),
                 ('background_EPE_px','Background EPE ↓'),('flow_half_MSE','Flow half MSE ↓'),
                 ('PSNR_db','PSNR ↑'),('mean_RE_pct','Mean material RE % ↓')]
        for label in starts:
            rows=[s for s in finished if s['label']==label]
            for ax,(key,title) in zip(axs.flat,metrics):
                ax.plot([s['alpha'] for s in rows],[s[key] for s in rows],'.-',label=label)
                ax.set(xlabel='Fraction of path to GT',title=title);ax.grid(alpha=.25)
        axs[0,0].legend()
        fig.suptitle('Letter: independently rendered metrics along a fixed physical material path')
        fig.savefig(root/'metric_curves.png',dpi=180)
        fig.savefig(root/'metric_curves.pdf')
        plt.close(fig)
    print(root/'results.md')


if __name__=='__main__':
    main()
