"""Run resumable 10x/100x E starts on an idle or explicitly selected GPU."""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path('outputs/synthetic_wonderplay_order_magnitudes_20261122')
PY=sys.executable
TRAIN='scripts/train_synthetic_wonderplay_multistart.py'
SCENES={'baseball':'baseball_drop','football':'football_deflated_2s'}
STARTS=[('E_x10',10.,160),('E_div10',.1,160),('E_x100',100.,240),('E_div100',.01,240)]

def write_json(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def gpu_state():
    r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
    return [tuple(map(int,line.split(','))) for line in r.stdout.splitlines()]

def wait_idle(deadline):
    consecutive=0
    checks=0
    while time.monotonic()<deadline:
        try:
            state=gpu_state()
            free=[index for index,mem,util in state if mem<300 and util<10]
        except Exception as exc:
            state=[]
            free=[]
            if checks%3==0:print(f'GPU check failed: {exc}',flush=True)
        if checks%3==0:
            detail=', '.join(f'{index}:{mem}MiB/{util}%' for index,mem,util in state)
            print(f'Waiting for idle GPU (need <300 MiB and <10%): {detail}',flush=True)
        checks+=1
        if free:
            consecutive+=1
            if consecutive>=3:
                print(f'Selected idle GPU {free[0]}',flush=True)
                return free[0]
        else:consecutive=0
        time.sleep(20)
    return None

def select_gpu(requested,deadline):
    if requested is None:return wait_idle(deadline)
    available={index for index,_,_ in gpu_state()}
    if requested not in available:raise ValueError(f'GPU {requested} not found; available: {sorted(available)}')
    print(f'Using explicitly selected GPU {requested}; idle check bypassed',flush=True)
    return requested

def execute(command,log,timeout,gpu):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    print(f'Running on GPU {gpu}; details: {log}',flush=True)
    with log.open('a') as f:
        f.write('\nCOMMAND '+json.dumps(command)+'\n');f.flush()
        try:
            p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
            print(f'Finished {log.name}: exit {p.returncode}',flush=True)
            return p.returncode
        except subprocess.TimeoutExpired:
            f.write(f'\nTIMEOUT {timeout}s; checkpoint retained\n')
            print(f'Timed out {log.name} after {timeout}s; checkpoint retained',flush=True)
            return 124

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu',type=int,help='physical nvidia-smi GPU index; bypass automatic idle check')
    args=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    rows=[]
    deadline=time.monotonic()+12*3600
    for label,scene in SCENES.items():
        for name,ratio,epochs in STARTS:
            print(f'Starting {label} {name} (E/GT={ratio:g})',flush=True)
            out=ROOT/label;out.mkdir(exist_ok=True)
            final=out/name/'final.json'
            probe=out/f'start_probe_{name}.json'
            base=[PY,TRAIN,'--scene',f'outputs/synthetic_physics_batch_20261121/scenes/{scene}',
                  '--output',str(out),'--start-name',name,'--start-E-ratio',str(ratio),
                  '--start-nu','.32','--E-parameterization','log','--E-min-ratio','.001',
                  '--E-max-ratio','1000','--lr-log-E','.08','--lr-nu','.004',
                  '--lr-decay-at','100,160,240','--max-frame','8','--frame-stride','2',
                  '--resolution','128','--checkpoint-interval','16']
            row=dict(scene=label,start=name,E_ratio=ratio,planned_updates=epochs,status='PENDING')
            if final.exists() and json.loads(final.read_text()).get('updates',0)>=epochs:
                row['status']='FINISHED';row['final']=json.loads(final.read_text())
            else:
                gpu=select_gpu(args.gpu,deadline)
                if gpu is None:
                    row['status']='NO_IDLE_GPU_WITHIN_12H';rows.append(row);break
                row['gpu']=gpu
                if not probe.exists():
                    code=execute(base+['--start-probe','--epochs','0'],
                                 ROOT/f'{label}_{name}_probe.log',240,gpu)
                    if code or not probe.exists() or not json.loads(probe.read_text()).get('finite'):
                        row['status']='NONFINITE_OR_FAILED_FORWARD_PROBE';row['probe_exit']=code
                        rows.append(row);write_json(ROOT/'runner_state.json',rows);continue
                row['probe']=json.loads(probe.read_text())
                gate=out/f'gradient_gate_{name}.json'
                if not gate.exists():
                    code=execute(base+['--gradient-gate','--fd-E-fraction','.01',
                                       '--fd-nu-step','.005','--epochs','0'],
                                 ROOT/f'{label}_{name}_gradient.log',300,gpu)
                    row['gradient_exit']=code
                if gate.exists():row['gradient_gate']=json.loads(gate.read_text())
                # The gate is diagnostic; a finite forward is sufficient to test
                # whether the original reverse-AD optimizer can move from the start.
                for attempt in range(2):
                    gpu=select_gpu(args.gpu,deadline)
                    if gpu is None:break
                    code=execute(base+['--epochs',str(epochs)],
                                 ROOT/f'{label}_{name}_train.log',1800,gpu)
                    row[f'train_exit_{attempt+1}']=code
                    if final.exists() and json.loads(final.read_text()).get('updates',0)>=epochs:break
                    if code!=124:break
                if final.exists() and json.loads(final.read_text()).get('updates',0)>=epochs:
                    row['status']='FINISHED';row['final']=json.loads(final.read_text())
                else:
                    checkpoint=out/name/'optimizer.pt'
                    row['status']='INCOMPLETE_CHECKPOINTED' if checkpoint.exists() else 'TRAIN_FAILED_BEFORE_CHECKPOINT'
            rows.append(row)
            write_json(ROOT/'runner_state.json',rows)
        else:continue
        break
    with (ROOT/'runner_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['scene','start','E_ratio','planned_updates','status','final_loss','E_error','nu_error'])
        for r in rows:
            d=r.get('final',{})
            w.writerow([r['scene'],r['start'],r['E_ratio'],r['planned_updates'],r['status'],
                        d.get('final_loss'),d.get('final_E_relative_error'),d.get('final_nu_absolute_error')])
    lines=['# 10× and 100× initial E experiment','',
           'The target is self-consistent Gaussian exact flow, with the same Warp MPM, RepairedTemporalRollout, WonderPlay loss, three views, 128×128 pixels and 0–0.333 s window as the previous study. ν starts at GT=.32 to isolate distance in E, but both E and ν are optimized. E uses log(E/GT) Adam with initial learning rate .08; ν uses .004. Finite differences check initial reverse-AD derivatives only. No RAFT or finite-difference optimizer is used. Each run has a forward probe, checkpoint per update and external timeout. Trials run only after an idle GPU is observed three times.', '',
           '| Scene | Start E/GT | Updates | Forward probe | Gradient gate | Final loss | E error | ν error | Boundary | Classification |',
           '|---|---:|---:|---|---|---:|---:|---:|---|---|']
    for r in rows:
        d=r.get('final',{});g=r.get('gradient_gate',{});p=r.get('probe',{})
        if r['status']=='FINISHED':
            recovered=(d['final_loss']<=1e-3 and d['final_E_relative_error']<=.02 and
                       d['final_nu_absolute_error']<=.01)
            classification=('GT_RECOVERED' if recovered else
                            'LOW_LOSS_OFF_GT' if d['final_loss']<=.05 else 'NOT_RECOVERED')
        else:classification=r['status']
        fmt=lambda x:'—' if x is None else f'{x:.5g}'
        lines.append(f"| {r['scene']} | {r['E_ratio']:g} | {d.get('updates','—')} | {p.get('finite','—')} | {g.get('pass_gate','—')} | {fmt(d.get('final_loss'))} | {fmt(d.get('final_E_relative_error'))} | {fmt(d.get('final_nu_absolute_error'))} | E={d.get('any_boundary_hit_E','—')}, ν={d.get('any_boundary_hit_nu','—')} | {classification} |")
    lines+=['','A failed 100× forward probe is a numerical validity result for this timestep and solver, not evidence about the global optimization landscape. GT recovery here only applies to the matched Gaussian target. Independent Blender visible-flow GT loss and gradients remain biased, as documented in `outputs/synthetic_wonderplay_blender_bias_20261122/REPORT.md`.','']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
