"""Create read-only Letter loss, material RE, and image-metric plots from two runs.

Usage: python scripts/report_letter_two_methods.py --base outputs/letter_two_methods/letter
"""
import argparse, csv, json, re, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.score_universal_pacnerf_re import score
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True,help='Contains original/ and projected_flow_AD/')
p.add_argument('--output',type=Path,help='Defaults to BASE/detailed_comparison')
a=p.parse_args()
BASE=a.base.resolve()
OUT=(a.output or BASE/'detailed_comparison').resolve(); OUT.mkdir(parents=True,exist_ok=True)
methods=['original','projected_flow_AD']; colors=['#d27622','#2878aa']
truth=json.loads((ROOT/'data/pacnerf/groundtruth.json').read_text())['scenes']['letter']['parameters']
data={}
for method in methods:
 b=BASE/method;run=b/'train_ds1.0_ep50'; ev=json.loads((b/'evaluation/evaluation.json').read_text())
 cfg=json.loads((b/'physics_config.json').read_text()); scale=json.loads((run/'kinematics_manifest.json').read_text())['scale_origin']
 final=json.loads((run/'ckpt/sim.json').read_text()); flow=torch.load(b/'evaluation/flow_fields.pt',map_location='cpu')
 assert ev['status']=='complete' and ev['finite'] and len(ev['selected_source_frames'])==16
 if method=='original':
  log=(b/'train.log').read_text();loss=[float(x) for x in re.findall(r'^loss:\s*([-+\d.eE]+)',log,re.M)]
  es=[float(x) for x in re.findall(r'^- E:\s*([-+\d.eE]+)',log,re.M)]
  nus=[float(x) for x in re.findall(r'^- nu:\s*([-+\d.eE]+)',log,re.M)]
  assert len(loss)==len(es)==len(nus)==50,(len(loss),len(es),len(nus))
  epochs=np.arange(1,51);params=[dict(E=e,nu=n) for e,n in zip(es,nus)]
 else:
  rows=[json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
  assert len(rows)==50 and [r['epoch'] for r in rows]==list(range(1,51))
  epochs=np.arange(0,51);loss=[rows[0]['loss_before']]+[r['loss_after'] for r in rows]
  params=[rows[0]['before']]+[r['after'] for r in rows]
  gate=json.loads((run/'gradient_check.json').read_text());post=json.loads((run/'post_gradient_check.json').read_text())
 err=((flow['simulated_flows']-flow['target_flows'])**2).sum(1).sqrt();mask=flow['foreground'].bool()
 fg=[float(err[i][mask[i]].mean()) for i in range(15)];whole=[float(err[i].mean()) for i in range(15)]
 assert abs(float(err[mask].mean())-ev['epe']['foreground_px'])<1e-6
 assert abs(float(err.mean())-ev['epe']['whole_px'])<1e-6
 phys=[score('letter',p,scale) for p in params];initial=score('letter',cfg,scale);ending=score('letter',final,scale)
 data[method]=dict(ev=ev,epochs=epochs,loss=np.asarray(loss),params=params,phys=phys,initial=initial,ending=ending,
                   scale=scale,fg=np.asarray(fg),whole=np.asarray(whole),flow=flow,
                   gate=gate if method!='original' else None,post=post if method!='original' else None)
 assert abs(ending['mean_RE']-phys[-1]['mean_RE'])<1e-4
 if method=='original':
  assert abs(float(np.mean([params[-1]['E'],final['E']]))-final['E'])<1e-6
target_max_difference=float((data['original']['flow']['target_flows']-
                             data['projected_flow_AD']['flow']['target_flows']).abs().max())
assert torch.equal(data['original']['flow']['foreground'],data['projected_flow_AD']['flow']['foreground'])
# One CSV contains the exact plotted training values; each method has its own loss definition.
with (OUT/'training_curves.csv').open('w',newline='') as stream:
 w=csv.writer(stream);w.writerow(['method','epoch','training_loss','E_internal','nu','mu_physical','kappa_physical','mu_RE_percent','kappa_RE_percent','mean_RE_percent'])
 for m,d in data.items():
  for i,ep in enumerate(d['epochs']):
   p=d['params'][i];r=d['phys'][i]
   w.writerow([m,ep,d['loss'][i],p['E'],p['nu'],r['physical']['mu'],r['physical']['kappa'],
               100*r['per_parameter_RE']['mu'],100*r['per_parameter_RE']['kappa'],100*r['mean_RE']])
with (OUT/'frame_metrics.csv').open('w',newline='') as stream:
 w=csv.writer(stream);w.writerow(['method','frame_or_pair','foreground_EPE_px','whole_EPE_px','PSNR_dB'])
 for m,d in data.items():
  for i in range(16):w.writerow([m,i,'' if i==15 else d['fg'][i],'' if i==15 else d['whole'][i],d['ev']['psnr_by_frame'][i]])
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'figure.facecolor':'white'})
fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
for j,m in enumerate(methods):
 d=data[m];ax=axs[0,j];ax.plot(d['epochs'],d['loss'],color=colors[j],lw=1.8,marker='.',ms=3)
 ax.set(xlabel='Epoch (original: loss before update; projected: after update)',ylabel='Training loss',title=f'{m}: {d["loss"][0]:.6g} to {d["loss"][-1]:.6g}')
 ax.grid(alpha=.25)
 if j==0:
  ax.text(.02,.04,'Five repeated 10-epoch windows; x=1..50\nLoss is measured before each update.',transform=ax.transAxes,va='bottom',fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
 else: ax.text(.02,.04,'x=0 initial; x=1..50 after each update.\nSame fixed projected-flow objective.',transform=ax.transAxes,va='bottom',fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
 axs[1,j].plot(d['epochs'],d['loss']/d['loss'][0],color=colors[j],lw=1.8)
 axs[1,j].axhline(1,color='gray',lw=.8)
 axs[1,j].set(xlabel='Epoch (original: before update; projected: after update)',ylabel='Loss / first recorded loss',title=f'{m}: change {(d["loss"][-1]/d["loss"][0]-1)*100:+.2f}%')
 axs[1,j].grid(alpha=.25)
fig.suptitle('Letter | 50 epochs each | different losses; compare trends only',fontsize=15)
fig.savefig(OUT/'loss_curves.png',dpi=180);fig.savefig(OUT/'loss_curves.pdf');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
for j,(key,gt) in enumerate([('mu',truth['mu']),('kappa',truth['kappa'])]):
 ax=axs[0,j];bx=axs[1,j]
 for i,m in enumerate(methods):
  d=data[m];v=np.array([r['physical'][key] for r in d['phys']]);re=np.array([r['per_parameter_RE'][key] for r in d['phys']])*100
  ax.plot(d['epochs'],v,label=m,color=colors[i],lw=1.8);bx.plot(d['epochs'],re,label=m,color=colors[i],lw=1.8)
 ax.axhline(gt,color='black',ls='--',lw=1,label=f'GT {gt:g}');ax.set_yscale('log')
 bx.axhline(0,color='black',ls='--',lw=.8);bx.set_yscale('symlog',linthresh=10)
 ax.set(xlabel='Epoch',ylabel=f'{key} physical units (log)',title=f'{key} estimate vs GT');bx.set(xlabel='Epoch',ylabel='Relative error % (symlog)',title=f'{key} material RE')
 ax.grid(alpha=.25);bx.grid(alpha=.25);ax.legend(fontsize=8);bx.legend(fontsize=8)
fig.suptitle(f'Letter | material estimates use each trainer\'s own scale_origin '
             f'(original {data["original"]["scale"]:.6g}, projected {data["projected_flow_AD"]["scale"]:.6g})',fontsize=14)
fig.savefig(OUT/'material_accuracy.png',dpi=180);fig.savefig(OUT/'material_accuracy.pdf');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
for i,m in enumerate(methods):
 d=data[m];axs[0,0].plot(range(1,16),d['fg'],label=m,color=colors[i],lw=1.7,marker='.')
 axs[0,1].plot(range(1,16),d['whole'],label=m,color=colors[i],lw=1.7,marker='.')
 axs[1,0].plot(range(16),d['ev']['psnr_by_frame'],label=m,color=colors[i],lw=1.7,marker='.')
for ax,title,ylabel in [(axs[0,0],'Foreground EPE per flow pair','pixels ↓'),(axs[0,1],'Whole-image EPE per flow pair','pixels ↓'),(axs[1,0],'PSNR per rendered frame','dB ↑')]:
 ax.set(title=title,xlabel='Source frame 0..15' if title.startswith('PSNR') else 'Flow pair 1..15 (pair 1 = frames 0 to 1)',ylabel=ylabel);ax.grid(alpha=.25);ax.legend(fontsize=8)
labels=['FG EPE','Whole EPE','half MSE','PSNR'];orig=data['original']['ev'];new=data['projected_flow_AD']['ev']
for i,(name,a,b) in enumerate(zip(labels,[orig['epe']['foreground_px'],orig['epe']['whole_px'],orig['epe']['half_mse'],orig['psnr_db']],[new['epe']['foreground_px'],new['epe']['whole_px'],new['epe']['half_mse'],new['psnr_db']])):
 axs[1,1].text(.03,.91-i*.23,f'{name}:  original {a:.4f}   |   projected {b:.4f}',transform=axs[1,1].transAxes,fontsize=11)
axs[1,1].axis('off');axs[1,1].set_title('Independent native-renderer evaluation')
fig.suptitle('Letter | same 16 source frames, same target flow and supplied foreground mask',fontsize=14)
fig.savefig(OUT/'evaluation_accuracy.png',dpi=180);fig.savefig(OUT/'evaluation_accuracy.pdf');plt.close(fig)
summary={}
for m,d in data.items():
 g=[r['relative_error'] for rows in d['gate']['finite_difference'].values() for r in rows] if d['gate'] else []
 p=[r['relative_error'] for rows in d['post']['finite_difference'].values() for r in rows] if d['post'] else []
 summary[m]=dict(status='complete',epochs=50,scale_origin=d['scale'],material_RE_at_common_evaluation_scale=score('letter',json.loads((BASE/m/'train_ds1.0_ep50/ckpt/sim.json').read_text()),data['projected_flow_AD']['scale']),training_loss_first=float(d['loss'][0]),training_loss_last=float(d['loss'][-1]),loss_percent_change=float((d['loss'][-1]/d['loss'][0]-1)*100),
  material_initial=d['initial'],material_final=d['ending'],evaluation=dict(EPE=d['ev']['epe'],PSNR=d['ev']['psnr_db']),gradient_gate_initial_max_RE=max(g) if g else None,gradient_gate_final_max_RE=max(p) if p else None)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Letter: original vs projected-flow AD, 50 epochs','',
       'Training loss is method-specific. Original loss uses a moving 10-stage window and is logged before updates; projected flow is one fixed objective, x=0 before training and x=1..50 after updates. Their magnitudes must not be compared.',
       f'Original uses scale_origin={data["original"]["scale"]:.9g}; projected uses {data["projected_flow_AD"]["scale"]:.9g}. RE curves convert using each trainer\'s own scale. The table reports material RE under both trainer-specific and common evaluator scales. Final EPE/PSNR share a protocol; if training scales differ, this is not a controlled comparison of identical training physics.',
       '', '| Metric | Original | Projected-flow AD | Better |','|---|---:|---:|---|']
def row(name,a,b,lower=True,unit=''):
 return f'| {name} | {a:.4f}{unit} | {b:.4f}{unit} | {"original" if (a<b if lower else a>b) else "projected"} |'
a,b=data['original'],data['projected_flow_AD']
for name,fetch,lower,unit in [('Training loss, first',lambda d:d['loss'][0],True,''),('Training loss, last',lambda d:d['loss'][-1],True,''),('FG EPE px',lambda d:d['ev']['epe']['foreground_px'],True,''),('Whole EPE px',lambda d:d['ev']['epe']['whole_px'],True,''),('Flow half MSE',lambda d:d['ev']['epe']['half_mse'],True,''),('PSNR dB',lambda d:d['ev']['psnr_db'],False,''),('Mu RE (trainer scale)',lambda d:d['ending']['per_parameter_RE']['mu']*100,True,'%'),('Kappa RE (trainer scale)',lambda d:d['ending']['per_parameter_RE']['kappa']*100,True,'%'),('Mean material RE (trainer scale)',lambda d:d['ending']['mean_RE']*100,True,'%'),('Mean RE (common eval scale)',lambda d:summary['original' if d is a else 'projected_flow_AD']['material_RE_at_common_evaluation_scale']['mean_RE']*100,True,'%')]:
 if name.startswith('Training loss'):lines.append(f'| {name} (different definitions) | {fetch(a):.6g} | {fetch(b):.6g} | — |')
 else:lines.append(row(name,fetch(a),fetch(b),lower,unit))
lines+=['',f'GT: mu={truth["mu"]}, kappa={truth["kappa"]}. Original final mu={a["ending"]["physical"]["mu"]:.3f}, kappa={a["ending"]["physical"]["kappa"]:.3f}; projected final mu={b["ending"]["physical"]["mu"]:.3f}, kappa={b["ending"]["physical"]["kappa"]:.3f}.',
 f'Projected-flow loss improves {abs(summary["projected_flow_AD"]["loss_percent_change"]):.2f}% and material mean RE falls from {b["initial"]["mean_RE"]*100:.1f}% to {b["ending"]["mean_RE"]*100:.1f}%; whole-image EPE and half MSE worsen versus original.',
 f'Projected initial/final AD/FD gates: {bool(b["gate"] and b["gate"].get("passed"))} / {bool(b["post"] and b["post"].get("passed"))}. Maximum relative errors: {summary["projected_flow_AD"]["gradient_gate_initial_max_RE"]*100:.3f}% and {summary["projected_flow_AD"]["gradient_gate_final_max_RE"]*100:.3f}%. The original method has no comparable full-horizon AD/FD gate.',
 '', 'Files: loss_curves.png/.pdf, material_accuracy.png/.pdf, evaluation_accuracy.png/.pdf, training_curves.csv, frame_metrics.csv, summary.json.',
 f'Initial internal parameters: original E={a["params"][0]["E"]:.9g}, nu={a["params"][0]["nu"]:.9g}; projected E={b["params"][0]["E"]:.9g}, nu={b["params"][0]["nu"]:.9g}.',
 f'Final internal parameters: original E={a["params"][-1]["E"]:.9g}, nu={a["params"][-1]["nu"]:.9g}; projected E={b["params"][-1]["E"]:.9g}, nu={b["params"][-1]["nu"]:.9g}.',
 'The original moving-window loss repeats each 10-epoch phase; it is not 50 evaluations of one fixed loss.',
 f'Per-pair EPE comes from saved RAFT flow_fields.pt. Foreground masks match exactly; maximum target-flow difference between evaluations is {target_max_difference:.6g}. Flow is 256x480; PSNR averages frames 1..15.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(OUT/'REPORT.md')
print(json.dumps({m:{'loss':(d['loss'][0],d['loss'][-1]),'FG_EPE':d['ev']['epe']['foreground_px'],'whole_EPE':d['ev']['epe']['whole_px'],'PSNR':d['ev']['psnr_db'],'mean_RE':d['ending']['mean_RE']} for m,d in data.items()},indent=2))
