"""Serial original/projected-flow comparison. No search or physics edits.

Training and independent native-renderer evaluation share a single GPU.
Each completed epoch and each stage is published atomically for live review.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.score_universal_pacnerf_re import score
from supplemental_pipeline.simulation_timing import resolve_schedule

SCENES = ['letter', 'droplet', 'torus', 'bird', 'playdoh', 'cat', 'cream', 'toothpaste', 'trophy']


def stamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def records(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def atomic(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    a = p.parse_args()
    if a.epochs < 1:
        p.error('epochs must be positive')
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    jobs = []
    state = dict(started=stamp(), status='preparing', physical_gpu=a.gpu, epochs=a.epochs, jobs=jobs,
                 protocol='Original geometry/rendered-flow loss versus projected-flow reverse AD. Same per-scene config and independent image metrics; objectives and optimization coordinates differ.',
                 projected_learning_rate=.003, groundtruth=str(ROOT/'data/pacnerf/groundtruth.json'),
                 groundtruth_usage='reporting only; never selects parameter updates', unsupported_current_materials=['plasticine', 'non_newtonian', 'sand'])

    def publish():
        state['updated'] = stamp()
        atomic(root/'status.json', json.dumps(state, indent=2, ensure_ascii=False)+'\n')
        rows = []
        for j in jobs:
            e, re = j.get('evaluation', {}), j.get('material_re', {})
            rows.append(dict(scene=j['scene'], method=j['method'], status=j['status'], epochs=j.get('completed_epochs', 0),
                foreground_EPE=e.get('epe', {}).get('foreground_px'), whole_EPE=e.get('epe', {}).get('whole_px'),
                half_MSE=e.get('epe', {}).get('half_mse'), PSNR=e.get('psnr_db'), mean_RE=re.get('mean_RE'),
                per_parameter_RE=json.dumps(re.get('per_parameter_RE', {})), physical=json.dumps(re.get('physical', {})),
                loss=j.get('last_update', {}).get('loss_after'), gradient_gate=j.get('gradient_gate', {}).get('passed'),
                final_gradient_gate=j.get('final_gradient_gate', {}).get('passed'), error=j.get('error', '')))
        if not rows:
            return
        with (root/'results.csv.tmp').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (root/'results.csv.tmp').replace(root/'results.csv')
        lines = ['# PAC-NeRF original vs projected-flow reverse AD', '',
            f'Updated: {stamp()} | Physical GPU {a.gpu} | {a.epochs} epochs per job | original then current per scene', '',
            state['protocol'], '',
            'No verified search. FD only checks gradients; Adam updates current parameters. Current LR=0.003, fixed across scenes. MPM sources remain frozen.',
            'Current must pass all six initial material AD/FD stencils (sign match, max error <=10%, median <=5%) before training. Final checkpoint is checked again.',
            'Unsupported current materials are skipped explicitly. Failed gates never fall back to another optimizer. Completed epochs count actual optimizer updates.',
            'EPE (pixels), half MSE and PSNR use the same independent native-renderer replay versus supplied GT images; they are not the projected training loss.',
            'RE below is percent; CSV stores fractional RE. RE during training uses the latest saved checkpoint; EPE/PSNR appear after final evaluation. Missing viscosity conversion prevents a full mean RE.', '',
            '| Scene | Method | Status | Epoch | FG EPE ↓ | Whole EPE ↓ | half MSE ↓ | PSNR ↑ | Mean RE % ↓ | Parameter RE % | Initial / final gate |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
        def fmt(v):
            return '—' if v is None else f'{v:.5g}'
        for j, r in zip(jobs, rows):
            re = j.get('material_re', {})
            detail = ', '.join(f'{k}: {100*v:.4g}' for k, v in re.get('per_parameter_RE', {}).items() if v is not None)
            gate = ' / '.join('—' if r[k] is None else ('PASS' if r[k] else 'FAIL') for k in ['gradient_gate', 'final_gradient_gate'])
            lines.append(f"| {r['scene']} | {r['method']} | {r['status']} | {r['epochs']} | {fmt(r['foreground_EPE'])} | {fmt(r['whole_EPE'])} | {fmt(r['half_MSE'])} | {fmt(r['PSNR'])} | {fmt(None if r['mean_RE'] is None else 100*r['mean_RE'])} | {detail} | {gate} |")
        lines += ['', 'See status.json and each job train.log/evaluation.log for commands, errors, source hashes and full precision metrics.']
        atomic(root/'results.md', '\n'.join(lines)+'\n')

    for scene in SCENES:
        source = ROOT/('config/gradient_repair/letter_flow_ad.json' if scene == 'letter' else f'config/pacnerf/{scene}_config.json')
        config = read(source)
        config.update(training_objective='flow', guidance_background='white_masked')
        if config['material'] == 'newtonian':
            config['newtonian_model'] = 'strain_rate'
        model = ROOT/'model/pacnerf'/scene
        count = len(list((model/'images_generated').glob('*.png')))
        if count < 2:
            raise ValueError(f'Insufficient source images: {model}')
        plan = resolve_schedule(model, config, count-1)
        for frame in plan['selected_frames']:
            mask = model/'masks'/('a_'+Path(frame['filename']).stem[2:]+'.png')
            if not mask.is_file():
                raise FileNotFoundError(mask)
        for method in ['original', 'projected_flow_AD']:
            base = root/scene/method
            base.mkdir(parents=True)
            atomic(base/'physics_config.json', json.dumps(config, indent=2)+'\n')
            j = dict(scene=scene, method=method, base=str(base), material=config['material'], key_frames=count-1,
                     source_config=str(source), status='queued', completed_epochs=0)
            if method != 'original' and config['material'] not in ('elastic', 'newtonian'):
                j.update(status='unsupported', error='Frozen repaired production entry only validates elastic and strain-rate Newtonian; no fallback.')
            jobs.append(j)
    sources = ['simulation_memory_safe.py', 'scripts/train_projected_flow_gradient.py', 'scripts/evaluate_universal_reference.py',
               'scripts/score_universal_pacnerf_re.py', 'scripts/run_pacnerf_gradient_comparison.py', 'utils/projected_flow_objective.py']
    frozen = read(ROOT/'release/frozen_mpm.json')
    state['frozen_mpm'] = frozen
    state['source_sha256'] = {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}
    atomic(root/'groundtruth.json', (ROOT/'data/pacnerf/groundtruth.json').read_text())
    state['status'] = 'prepared'
    publish()
    if a.prepare_only:
        return

    def inspect(j, run):
        rows = records(run/('memory.jsonl' if j['method']=='original' else 'training.jsonl'))
        j['completed_epochs'] = len(rows)
        if rows and j['method'] != 'original':
            j['last_update'] = rows[-1]
        kin, params = read(run/'kinematics_manifest.json', {}), read(run/'ckpt/sim.json')
        if kin and params:
            try:
                j['material_re'] = score(j['scene'], params, kin['scale_origin'])
            except Exception as exc:
                j['material_re_error'] = str(exc)
        for filename, key in [('gradient_check.json', 'gradient_gate'), ('post_gradient_check.json', 'final_gradient_gate')]:
            check = read(run/filename)
            if check:
                j[key] = check

    def execute(j, cmd, log, run):
        j['command'] = cmd
        for s, digest in {**state['source_sha256'], **frozen}.items():
            if hashlib.sha256((ROOT/s).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Source changed during comparison: '+s)
        with log.open('w') as stream:
            stream.write('COMMAND '+json.dumps(cmd)+'\n'); stream.flush()
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            j['pid'] = child.pid
            publish()
            while child.poll() is None:
                inspect(j, run)
                publish()
                time.sleep(15)
            j.pop('pid', None)
        inspect(j, run)
        publish()
        return child.returncode

    state['status'] = 'running'
    for j in jobs:
        if j['status'] == 'unsupported':
            continue
        base = Path(j['base']); model = ROOT/'model/pacnerf'/j['scene']
        config = base/'physics_config.json'
        run = base/f'train_ds1.0_ep{a.epochs}'
        j['run_directory'] = str(run)
        try:
            j.update(status='training', started=stamp())
            print(stamp(), j['scene'], j['method'], 'TRAIN_START', flush=True)
            if j['method'] == 'original':
                cmd = [sys.executable, str(ROOT/'simulation_memory_safe.py'), '--model_path', str(model), '--physics_config', str(config),
                       '--output_path', str(base/'train'), '--n_epoch', str(a.epochs), '--n_key_frame', str(j['key_frames']),
                       '--stop-after-batches', str(a.epochs), '--skip-video', '--guidance-mode', 'raft_cpu']
            else:
                cmd = [sys.executable, str(ROOT/'scripts/train_projected_flow_gradient.py'), '--model-path', str(model),
                       '--physics-config', str(config), '--output', str(base/'train'), '--epochs', str(a.epochs),
                       '--interval', '32', '--lr', '.003', '--check-after-training']
            rc = execute(j, cmd, base/'train.log', run)
            j['train_exit'] = rc
            atomic(base/'train.exit', str(rc)+'\n')
            checkpoint = run/'ckpt/sim.json'
            if j['completed_epochs'] != a.epochs or not checkpoint.exists():
                j.update(status='train_failed' if rc else 'train_incomplete', error=f'exit={rc}, updates={j["completed_epochs"]}/{a.epochs}; see train.log')
                continue
            j['training_validated'] = (rc==0 and (j['method']=='original' or
                (j.get('gradient_gate', {}).get('passed') is True and j.get('final_gradient_gate', {}).get('passed') is True)))
            j['status'] = 'evaluating'
            cmd = [sys.executable, str(ROOT/'scripts/evaluate_universal_reference.py'), '--model-path', str(model),
                   '--physics-config', str(config), '--output', str(base/'evaluation'), '--checkpoint-json', str(checkpoint),
                   '--key-frames', str(j['key_frames']), '--dataset', 'pacnerf']
            erc = execute(j, cmd, base/'evaluation.log', run)
            j['evaluation_exit'] = erc
            atomic(base/'evaluation.exit', str(erc)+'\n')
            j['evaluation'] = read(base/'evaluation/evaluation.json', {})
            if erc or not j['evaluation'].get('finite', False):
                j.update(status='evaluation_failed', error=f'evaluation exit={erc} or nonfinite output; see evaluation.log')
            elif not j['training_validated']:
                j.update(status='gradient_validation_failed', error='50-update checkpoint evaluated for diagnosis; final AD/FD gate failed. Not accepted as repaired.')
            else:
                j['status'] = 'complete'
        except Exception as exc:
            j.update(status='failed', error=repr(exc))
        finally:
            j['finished'] = stamp()
            publish()
            print(stamp(), j['scene'], j['method'], j['status'], flush=True)
    state.update(status='finished', finished=stamp())
    publish()
    print(stamp(), 'QUEUE_FINISHED', flush=True)


if __name__ == '__main__':
    main()
