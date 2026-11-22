"""Summarize saved far-start WonderPlay paths and distinguish recovery from low loss."""
import csv
import json
from pathlib import Path

ROOT = Path('outputs/synthetic_wonderplay_farstart_20261122')
runner = (ROOT/'runner.log').read_text() if (ROOT/'runner.log').exists() else ''
GT = {'baseball': 15000., 'football': 12000.}
rows = []
for scene, gt_e in GT.items():
    for start, ratio, nu0 in (('highE_lowNu', 2., .16), ('lowE_highNu', .5, .44)):
        folder = ROOT / scene / start
        path = folder / 'history.csv'
        gate_path = ROOT / scene / f'gradient_gate_{start}.json'
        gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
        hist = list(csv.DictReader(path.open())) if path.exists() else []
        final_path = folder / 'final.json'
        final = json.loads(final_path.read_text()) if final_path.exists() else {}
        last = hist[-1] if hist else {}
        loss = final.get('final_loss')
        e = final.get('final_E_Pa', float(last['E_Pa']) if last else None)
        nu = final.get('final_nu', float(last['nu']) if last else None)
        count = len(hist)
        boundary_e = any(r.get('boundary_hit_E') == 'True' for r in hist)
        boundary_nu = any(r.get('boundary_hit_nu') == 'True' for r in hist)
        if count >= 120 and loss is not None:
            if loss <= 1e-3 and abs(e / gt_e - 1) <= .02 and abs(nu - .32) <= .01:
                status = 'GT_RECOVERED'
            elif e <= .201*gt_e or e >= 2.999*gt_e or nu <= .051 or nu >= .479:
                status = 'BOUNDARY_LIMITED'
            elif loss <= .05:
                status = 'LOW_LOSS_OFF_GT'
            else:
                status = 'OFF_GT_OR_NOT_CONVERGED'
        elif gate and not gate.get('pass_gate', False):
            status = 'GRADIENT_GATE_FAILED'
        elif count:
            status = 'INCOMPLETE_CHECKPOINTED'
        elif f'DONE ' in runner and any(line.startswith(f'DONE {"baseball_drop" if scene == "baseball" else "football_deflated_2s"} {scene} {start} ') for line in runner.splitlines()):
            status = 'FAILED_BEFORE_CHECKPOINT'
        else:
            status = 'PENDING'
        rows.append(dict(scene=scene,start=start,updates=count,status=status,
                         initial_E_Pa=gt_e*ratio,initial_nu=nu0,
                         initial_loss=float(hist[0]['loss']) if hist else gate.get('loss'),
                         final_loss=loss,final_E_Pa=e,final_nu=nu,
                         E_relative_error=abs(e/gt_e-1) if e is not None else None,
                         nu_absolute_error=abs(nu-.32) if nu is not None else None,
                         gradient_gate=gate.get('pass_gate'),
                         gradient_E_relative_error=gate.get('derivatives',{}).get('E_internal',{}).get('relative_error'),
                         gradient_nu_relative_error=gate.get('derivatives',{}).get('nu',{}).get('relative_error'),
                         boundary_hit_E=boundary_e,boundary_hit_nu=boundary_nu,
                         history=str(path),checkpoint=str(folder/'optimizer.pt')))
with (ROOT/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
lines = ['# Far-start WonderPlay experiment', '',
         'Same saved Warp MPM scenes, RepairedTemporalRollout and WonderPlay Gaussian exact-flow objective as the prior multi-start experiment. Three views, 128×128, frames 0/2/4/6/8. No RAFT or finite-difference optimization. Centered differences only check the initial AD gradient. Each update saves an optimizer checkpoint and per-update CSV; each run has a 900 s external timeout and can resume.', '',
         'The starts are E=2×GT, ν=.16 and E=.5×GT, ν=.44. Adam uses initial E learning rate .04×GT and ν learning rate .008, halved at updates 70 and 100; initial budget is 120 updates, with continuation from checkpoint where needed. Parameter bounds are E∈[.2,3]×GT, ν∈[.05,.48].', '',
         '| Scene | Start | Updates | Status | Initial loss | Final loss | E relative error | ν absolute error | Initial AD/FD E/ν relative error | Bound hit |',
         '|---|---|---:|---|---:|---:|---:|---:|---:|---|']
for r in rows:
    fmt=lambda v: '—' if v is None else f'{v:.5g}'
    lines.append(f"| {r['scene']} | {r['start']} | {r['updates']} | {r['status']} | {fmt(r['initial_loss'])} | {fmt(r['final_loss'])} | {fmt(r['E_relative_error'])} | {fmt(r['nu_absolute_error'])} | {fmt(r['gradient_E_relative_error'])} / {fmt(r['gradient_nu_relative_error'])} ({r['gradient_gate']}) | E={r['boundary_hit_E']}, ν={r['boundary_hit_nu']} |")
snapshot = ROOT/'football/lowE_highNu/final_120.json'
if snapshot.exists():
    s=json.loads(snapshot.read_text())
    lines += ['', f"Football lowE/highNu at the original 120-update checkpoint: loss {s['final_loss']:.6g}, E error {s['final_E_relative_error']:.4g}, ν error {s['final_nu_absolute_error']:.4g}; ν hit its upper bound earlier in the path. The saved `final_120.json` and `history_120.csv` preserve this outcome before continuation."]
snapshot240 = ROOT/'football/lowE_highNu/final_240.json'
if snapshot240.exists():
    s=json.loads(snapshot240.read_text())
    lines.append(f"At 240 updates this same path had loss {s['final_loss']:.6g}, E error {s['final_E_relative_error']:.4g}, ν error {s['final_nu_absolute_error']:.4g}. The final continuation reached 270 updates. Its ν upper-bound contact occurred at updates 6–17 and it subsequently moved into the interior; the declining loss and nonzero gradient before continuation do not support calling the 120-step point a local minimum.")
lines += ['', 'GT_RECOVERED requires evaluated final loss ≤10⁻³, E relative error ≤2%, ν absolute error ≤.01, and at least 120 updates. LOW_LOSS_OFF_GT is an operational label for loss ≤.05 with off-GT parameters; it does not establish a mathematical local minimum. BOUNDARY_LIMITED means the final parameters sit at a bound; a transient bound hit is reported in its own column. Incomplete runs and failed gradients must not be counted as convergence.', '',
          '## Independent Blender flow bias', '',
          'The independent Blender visible-mesh flow remains a separate observation target. The prior GT evaluation found loss 0.4959 for baseball and 0.3658 for football, with nonzero GT gradients. Gaussian-target recovery therefore does not establish recovery from Blender video.', '']
bias_path = ROOT/'blender_bias_residual.json'
if bias_path.exists():
    bias = json.loads(bias_path.read_text())
    lines += ['The following comparison uses saved Gaussian and Blender GT flow targets on their common valid pixels. A four-pixel erosion of the common mask defines the interior; the remainder is the boundary band. These are residual diagnostics, not the WonderPlay training loss.', '',
              '| Scene | Interior mean EPE (px) | Boundary mean EPE (px) | Boundary share of squared residual |',
              '|---|---:|---:|---:|']
    for scene in GT:
        d=bias[scene]; boundary=d['boundary']['pixels']*d['boundary']['mean_sq_epe_px2']; total=d['all']['pixels']*d['all']['mean_sq_epe_px2']
        lines.append(f"| {scene} | {d['interior']['mean_epe_px']:.3f} | {d['boundary']['mean_epe_px']:.3f} | {boundary/total:.1%} |")
    lines += ['', 'The boundary band dominates the saved squared flow discrepancy, especially for baseball; football also retains a measurable interior discrepancy. This points to Gaussian support, opacity and visibility as alignment targets, but does not yet prove which renderer difference causes the nonzero GT gradient.', '']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps({'rows':len(rows),'status_counts':{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}}))
