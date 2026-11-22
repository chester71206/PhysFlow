"""Run nine PAC-NeRF WonderPlay scenes from GT, near, and distant starts.

Each physical GPU runs one job at a time. Jobs retain their configs, logs,
per-update checkpoints, and an atomic queue state. Re-running this command
skips completed jobs and resumes timed-out jobs from their last checkpoint.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supplemental_pipeline.simulation_timing import resolve_schedule

SCENES = ('letter', 'droplet', 'torus', 'bird', 'playdoh', 'cat', 'cream', 'toothpaste', 'trophy')
STATE_LOCK = threading.Lock()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def source_config(scene):
    return (ROOT / 'config/gradient_repair/letter_flow_ad.json' if scene == 'letter'
            else ROOT / 'config/pacnerf_wonderplay_ad' / f'{scene}.json')


def model_with_masks(scene, destination, config):
    model = ROOT / 'model/pacnerf' / scene
    frames = sorted((model / 'images_generated').glob('*.png'))
    if len(frames) < 2 or not (model / 'point_cloud').is_dir() or not (model / 'masks').is_dir():
        raise FileNotFoundError(f'{scene}: missing PAC-NeRF model, frames, or masks')
    aliases = {}
    for frame in resolve_schedule(model, config, len(frames) - 1)['selected_frames']:
        stem = Path(frame['filename']).stem[2:]
        name = f'a_{stem}.png'
        if (model / 'masks' / name).is_file():
            continue
        prefix, number = stem.rsplit('_', 1)
        padded = model / 'masks' / f'a_{prefix}_{int(number):02d}.png'
        if not padded.is_file():
            raise FileNotFoundError(f'{scene}: missing mask {name}')
        aliases[name] = padded.resolve()
    if not aliases:
        return model.resolve()
    overlay = destination / 'model'
    overlay.mkdir(exist_ok=True)
    for item in model.iterdir():
        if item.name != 'masks' and not (overlay / item.name).exists():
            (overlay / item.name).symlink_to(item.resolve())
    (overlay / 'masks').mkdir(exist_ok=True)
    for item in (model / 'masks').iterdir():
        link = overlay / 'masks' / item.name
        if not link.exists():
            link.symlink_to(item.resolve())
    for name, source in aliases.items():
        link = overlay / 'masks' / name
        if not link.exists():
            link.symlink_to(source)
    return overlay.resolve()


def create_jobs(root):
    jobs = []
    anchors = json.loads((ROOT / 'config/pacnerf_wonderplay_gt_anchors.json').read_text())
    for start in ('gt', 'near', 'far_E_x10'):
        for scene in SCENES:
            base = json.loads(source_config(scene).read_text())
            anchor = anchors[scene]
            kind = base['material']
            base.update(training_objective='flow', guidance_background='white_masked')
            if kind == 'newtonian':
                base['newtonian_model'] = 'strain_rate'
            active = ('friction_angle',) if kind == 'sand' else (
                ('E', 'nu', 'yield_stress', 'plastic_viscosity') if kind == 'non_newtonian' else
                ('E', 'nu', 'yield_stress') if kind == 'plasticine' else ('E', 'nu'))
            for name in active:
                base[name] = anchor[name]
            if start == 'near':
                if kind == 'sand':
                    base['friction_angle'] = min(89., anchor['friction_angle'] + 5.)
                else:
                    base['E'] *= 1.25
                    gap = .5 - anchor['nu']
                    base['nu'] = .5 - 1.25 * gap if anchor['nu'] > .45 else anchor['nu'] + .02
                    if 'yield_stress' in active:
                        base['yield_stress'] *= 1.25
                    if 'plastic_viscosity' in active:
                        base['plastic_viscosity'] *= 1.25
            elif start == 'far_E_x10':
                if kind == 'sand':
                    base['friction_angle'] = min(89., anchor['friction_angle'] + 30.)
                else:
                    base['E'] *= 10.
            if not all(math.isfinite(float(base[name])) for name in active):
                raise ValueError(f'{scene}/{start}: nonfinite parameter')
            destination = root / scene / start
            destination.mkdir(parents=True, exist_ok=True)
            model = model_with_masks(scene, destination, base)
            config = destination / 'physics_config.json'
            if config.exists():
                if json.loads(config.read_text()) != base:
                    raise ValueError(f'Existing config differs: {config}')
            else:
                atomic_json(config, base)
            jobs.append(dict(id=f'{scene}/{start}', scene=scene, start=start, material=kind,
                             active_parameters=list(active), anchor={name: anchor[name] for name in active},
                             initial={name: base[name] for name in active},
                             model_path=str(model), config_path=str(config.resolve()),
                             destination=str(destination.resolve()), status='pending',
                             segments=[]))
    return jobs


def completed_updates(job):
    for segment in reversed(job['segments']):
        checkpoint = Path(job['destination']) / segment['name'] / 'ckpt/optimizer.pt'
        if checkpoint.is_file():
            import torch
            data = torch.load(checkpoint, map_location='cpu', weights_only=False)
            return int(data['epoch']), checkpoint
    return 0, None


def run_job(job, gpu, args, state):
    directory = Path(job['destination'])
    updates, resume = completed_updates(job)
    if updates >= args.epochs:
        job['status'] = 'complete'
        with STATE_LOCK:
            atomic_json(state['path'], state['jobs'])
        return
    while updates < args.epochs:
        if len(job['segments']) >= args.max_segments:
            job['status'] = 'timeout_exhausted'
            break
        name = f'segment_{len(job["segments"]):02d}'
        output = directory / name
        # prepare_run appends the data scale and the requested update count.
        actual_output = directory / f'{name}_ds1.0_ep{args.epochs - updates}'
        if output.is_symlink():
            if output.readlink() != Path(actual_output.name):
                raise ValueError(f'Output link points to a different segment: {output}')
        elif not output.exists():
            output.symlink_to(actual_output.name, target_is_directory=True)
        script = ('check_pacnerf_material_flow_ad.py' if job['material'] in
                  ('plasticine', 'non_newtonian', 'sand') else 'train_projected_flow_gradient.py')
        command = [args.python, str(ROOT / 'scripts' / script), '--model-path', job['model_path'],
                   '--physics-config', job['config_path'], '--output', str(output),
                   '--epochs', str(args.epochs - updates), '--max-points', str(args.max_points),
                   '--allow-piecewise-renderer']
        if script == 'train_projected_flow_gradient.py':
            command += ['--flow-objective', 'wonderplay', '--check-after-training']
        if resume:
            command += ['--resume', str(resume)]
        segment = dict(name=name, gpu=gpu, command=command, start_time=time.time(),
                       status='running', updates_before=updates)
        job['segments'].append(segment)
        job['status'] = 'running'
        with STATE_LOCK:
            atomic_json(state['path'], state['jobs'])
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), GPU=str(gpu), PYTHONUNBUFFERED='1',
                   CUDA_HOME=env.get('CUDA_HOME', str(Path(sys.prefix))),
                   WANDB_MODE='disabled', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   FLOW_P2G_PRIMAL='fp64_atomic', FLOW_G2P_REDUCTION='fp64_atomic',
                   FLOW_ANALYTIC_G2P='1')
        with (directory / f'{name}.log').open('w') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                code = 124
        segment.update(end_time=time.time(), exit_code=code,
                       status='timeout' if code == 124 else 'complete' if code == 0 else 'failed')
        new_updates, new_resume = completed_updates(job)
        segment['updates_after'] = new_updates
        if code == 0 and new_updates >= args.epochs:
            job['status'] = 'complete'
        elif code == 124 and new_updates > updates:
            updates, resume = new_updates, new_resume
            job['status'] = 'pending_resume'
        else:
            job['status'] = 'timeout_no_progress' if code == 124 else 'failed'
        with STATE_LOCK:
            atomic_json(state['path'], state['jobs'])
        print(f'{job["id"]} GPU {gpu}: {job["status"]}, updates={new_updates}', flush=True)
        if job['status'] != 'pending_resume':
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/pacnerf_wonderplay_multistart_20261122'))
    parser.add_argument('--gpus', default='1,2,3,4,5,6,7')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max-points', type=int, default=20000)
    parser.add_argument('--timeout', type=int, default=10800, help='Seconds per segment')
    parser.add_argument('--max-segments', type=int, default=4)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    state_path = args.root / 'runner_state.json'
    planned = create_jobs(args.root)
    if state_path.exists():
        previous = {job['id']: job for job in json.loads(state_path.read_text())}
        for job in planned:
            if job['id'] in previous:
                old = previous[job['id']]
                if old['initial'] != job['initial'] or old['config_path'] != job['config_path']:
                    raise ValueError(f'Previous job definition differs: {job["id"]}')
                job['segments'] = old['segments']
                job['status'] = old['status']
    if args.plan_only:
        print(json.dumps([{k: j[k] for k in ('id', 'material', 'anchor', 'initial', 'status')}
                          for j in planned], indent=2))
        return
    gpus = [int(item) for item in args.gpus.split(',')]
    if len(set(gpus)) != len(gpus):
        parser.error('GPU list contains duplicates')
    state = dict(path=state_path, jobs=planned)
    atomic_json(state_path, planned)
    pending = [j for j in planned if j['status'] not in ('complete', 'failed',
               'timeout_no_progress', 'timeout_exhausted')]
    queue_lock = threading.Lock()

    def worker(gpu):
        while True:
            with queue_lock:
                if not pending:
                    return
                job = pending.pop(0)
            run_job(job, gpu, args, state)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))


if __name__ == '__main__':
    main()
