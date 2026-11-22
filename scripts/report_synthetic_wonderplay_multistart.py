"""Summarize exact-flow WonderPlay starts without confusing sampled paths with convergence."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

root=Path('outputs/synthetic_wonderplay_multistart_20261122')
rows=[]
for scene in ('baseball','football'):
    scene_dir=root/scene
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for histfile in sorted(scene_dir.glob('*/history.csv')):
        hist=[]
        with histfile.open() as f:
            for r in csv.DictReader(f):hist.append({k:float(v) for k,v in r.items()})
        if not hist:continue
        name=histfile.parent.name
        final=histfile.parent/'final.json'
        d=json.loads(final.read_text()) if final.exists() else {}
        gt_e=15000 if scene=='baseball' else 12000
        gt_nu=.32
        # A history row records loss before its update and parameters after it.
        # Put those two measurements on their actual optimizer-update indices.
        x_before=[r['epoch']-1 for r in hist]
        x_after=[0]+[r['epoch'] for r in hist]
        loss_curve=[r['loss'] for r in hist]
        if 'final_loss' in d:
            x_before.append(len(hist));loss_curve.append(d['final_loss'])
        e_curve=[abs(d.get('initial_E_Pa',gt_e)/gt_e-1)]+[r['E_relative_error'] for r in hist]
        nu_curve=[abs(d.get('initial_nu',gt_nu)-gt_nu)]+[r['nu_absolute_error'] for r in hist]
        axes[0].plot(x_before,loss_curve,label=name)
        axes[1].plot(x_after,e_curve,label=name)
        axes[2].plot(x_after,nu_curve,label=name)
        fig_start,axs=plt.subplots(1,3,figsize=(13,3.8))
        axs[0].plot(x_before,loss_curve);axs[0].set_yscale('log');axs[0].set_title('WonderPlay loss')
        axs[1].plot(x_after,[d.get('initial_E_Pa',gt_e)]+[r['E_Pa'] for r in hist]);axs[1].axhline(gt_e,color='k',ls='--',label='GT');axs[1].set_title('E (Pa)');axs[1].legend()
        axs[2].plot(x_after,[d.get('initial_nu',gt_nu)]+[r['nu'] for r in hist]);axs[2].axhline(gt_nu,color='k',ls='--',label='GT');axs[2].set_title('ν');axs[2].legend()
        for ax in axs:ax.set_xlabel('completed optimizer updates');ax.grid(alpha=.25)
        fig_start.tight_layout();fig_start.savefig(histfile.parent/'loss_parameters.png',dpi=160);plt.close(fig_start)
        fig_acc,axs=plt.subplots(1,2,figsize=(8,3.8))
        axs[0].plot(x_after,e_curve);axs[0].set_title('E relative error')
        axs[1].plot(x_after,nu_curve);axs[1].set_title('ν absolute error')
        for ax in axs:ax.set_xlabel('completed optimizer updates');ax.grid(alpha=.25)
        fig_acc.tight_layout();fig_acc.savefig(histfile.parent/'accuracy.png',dpi=160);plt.close(fig_acc)
        rows.append(dict(scene=scene,start=name,updates=len(hist),
             initial_loss=hist[0]['loss'],last_recorded_loss=hist[-1]['loss'],
             final_evaluated_loss=d.get('final_loss'),
             initial_E_Pa=d.get('initial_E_Pa'),initial_nu=d.get('initial_nu'),
             final_E_Pa=d.get('final_E_Pa',hist[-1]['E_Pa']),
             final_nu=d.get('final_nu',hist[-1]['nu']),
             final_E_relative_error=d.get('final_E_relative_error',hist[-1]['E_relative_error']),
             final_nu_absolute_error=d.get('final_nu_absolute_error',hist[-1]['nu_absolute_error']),
             final_particle_epe_m=d.get('final_particle_epe_m',hist[-1]['particle_epe_m'])))
    for ax,title in zip(axes,('WonderPlay loss','E relative error','ν absolute error')):
        ax.set_title(title);ax.set_xlabel('optimizer update');ax.grid(alpha=.25);ax.legend()
    axes[0].set_yscale('log')
    fig.tight_layout();fig.savefig(scene_dir/'multi_start_curves.png',dpi=160);plt.close(fig)
if rows:
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Exact-flow WonderPlay multi-start report','',
       'The target is the unchanged WonderPlay Gaussian rasterizer applied to saved native Warp MPM GT particle motion. Blender object masks define foreground. RAFT is unused. Training uses three views, 128×128 images, and frames 0, 2, 4, 6, 8 (0–0.333 s); 100 Adam updates per start, with learning rates halved at updates 50 and 75. GT is E=15,000 Pa, ν=0.32 for baseball and E=12,000 Pa, ν=0.32 for football.','',
       '| Scene | Start | Updates | Initial loss | Final evaluated loss | Final E (Pa) | Final ν | E relative error | ν absolute error | Particle EPE (mm) |',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    loss=r['final_evaluated_loss'] if r['final_evaluated_loss'] is not None else r['last_recorded_loss']
    lines.append(f"| {r['scene']} | {r['start']} | {r['updates']} | {r['initial_loss']:.5g} | {loss:.5g} | {r['final_E_Pa']:.5g} | {r['final_nu']:.5f} | {r['final_E_relative_error']:.4f} | {r['final_nu_absolute_error']:.4f} | {1000*r['final_particle_epe_m']:.3f} |")
lines+=['','Loss minima along these optimizer paths are not proof of the global or continuous minimum. Parameter recovery requires both E and ν errors to decrease, not just loss.','',
        'The self-consistent Gaussian target checks the original Warp rollout and WonderPlay loss/gradient. Independent Blender visible-mesh flow is a separate observation target and may differ because of representation and visibility.','',
        '## Independent Blender visible-surface flow at GT','',
        '| Scene | Gaussian target GT loss | Blender target GT loss | Blender target ∂loss/∂log E | Blender target ∂loss/∂ν × 0.05 |',
        '|---|---:|---:|---:|---:|']
for scene in ('baseball','football'):
    gaussian_gate=root/scene/'gate.json'
    blender_gate=root/(scene+'_blender_target')/'gate.json'
    grad_file=root/(scene+'_blender_target')/'gt_gradient.json'
    if all(p.exists() for p in (gaussian_gate,blender_gate,grad_file)):
        g=json.loads(gaussian_gate.read_text());b=json.loads(blender_gate.read_text());d=json.loads(grad_file.read_text())
        lines.append(f"| {scene} | {g['gt_loss']:.4g} | {b['gt_loss']:.4g} | {d['normalized_gradient_logE']:.4g} | {d['normalized_gradient_nu_0p05']:.4g} |")
    gaussian_target=root/scene/'exact_gt_targets.pt'
    blender_target=root/(scene+'_blender_target')/'exact_gt_targets.pt'
    if gaussian_target.exists() and blender_target.exists():
        a=torch.load(gaussian_target,map_location='cpu');b=torch.load(blender_target,map_location='cpu')
        fig,axes=plt.subplots(3,4,figsize=(16,9))
        for i in range(3):
            for j in range(4):
                m=(a['masks'][i][j]&b['masks'][i][j]).numpy()
                delta=(a['flows'][i][j]-b['flows'][i][j]).norm(dim=0).numpy()
                delta[~m]=np.nan
                ax=axes[i,j];im=ax.imshow(delta,vmin=0,vmax=3,cmap='magma')
                ax.set_title(f'view {i}, frames {2*j}→{2*j+2}: {np.nanmean(delta):.2f} px',fontsize=10)
                ax.axis('off')
        fig.subplots_adjust(right=.88,hspace=.16,wspace=.08)
        cax=fig.add_axes((.9,.18,.015,.64));fig.colorbar(im,cax=cax,label='flow EPE (px)')
        fig.savefig(root/(scene+'_flow_target_residual.png'),dpi=140);plt.close(fig)
lines+=['','The Blender target differs from the Gaussian target even for identical GT MPM motion. Thus near-zero Gaussian-target GT loss does not establish that GT minimizes the independent Blender observation loss.','']
lines+=['','At GT, the independent Blender-flow gradients can be checked against centered finite differences:', '',
        '| Scene | E probe ± fraction | E AD / FD (solver units) | ν AD / FD | Gate |',
        '|---|---:|---:|---:|---|']
for scene,filename,step in (('baseball','gradient_gate_GT.json','5%'),('football','gradient_gate_GT_fd1pct.json','1%')):
    p=root/(scene+'_blender_target')/filename
    if p.exists():
        d=json.loads(p.read_text());e=d['derivatives']['E_internal'];nu=d['derivatives']['nu']
        lines.append(f"| {scene} | {step} | {e['ad']:.4g} / {e['fd']:.4g} | {nu['ad']:.4g} / {nu['fd']:.4g} | {'PASS' if d['pass_gate'] else 'FAIL'} |")
lines+=['','## Local parameter sensitivity at GT','',
        'Each point uses the same initial particles, timestep, gravity, collider, and cameras. E changes by ±10%; ν changes by ±0.05. Values are the mean excess loss over GT for the two signs.','',
        '| Scene | Target | E ±10% excess loss | ν ±0.05 excess loss | E / ν ratio |',
        '|---|---|---:|---:|---:|']
for scene in ('baseball','football'):
    for target,label in ((scene,'Gaussian'),(scene+'_blender_target','Blender mesh')):
        path=root/target/'local_sensitivity.json'
        if path.exists():
            d=json.loads(path.read_text())['points']
            e=np.mean([d[k]['excess_loss_over_gt'] for k in ('E_minus_10pct','E_plus_10pct')])
            nu=np.mean([d[k]['excess_loss_over_gt'] for k in ('nu_minus_0p05','nu_plus_0p05')])
            lines.append(f'| {scene} | {label} | {e:.5g} | {nu:.5g} | {e/nu:.3g} |')
lines+=['','These step sizes differ in relative magnitude; the ratio is descriptive for these stated perturbations, not a dimensionless material identifiability theorem.','']
lines+=['','For football, the ν perturbation produces substantially less flow-loss change than the E perturbation in this early impact window. Baseball shows the opposite ordering for the same stated perturbations. The scenes differ in geometry, density, E, and contact motion, so this pair alone cannot isolate which factor causes the sensitivity difference.','',
        '## Interpretation and next work','',
        'All four starts recover both parameters closely under the Gaussian self-consistent target, and the off-GT reverse-AD derivatives passed centered-difference checks. This demonstrates a working differentiable path for these two saved native-MPM scenes. It does not prove a global minimum or recovery from independently rendered video.','',
        'The independent Blender visible-mesh target has a material GT loss and nonzero GT gradient. Its residual maps show both silhouette and interior errors during impact. Before adding further scenes or claiming video-based recovery, align Gaussian appearance/visibility with the Blender surface and recheck whether GT becomes stationary under the external target. Automated PASS for source videos does not mean human-certified visual naturalness.','']
(root/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(dict(starts=len(rows),scenes=sorted({r['scene'] for r in rows}))))
