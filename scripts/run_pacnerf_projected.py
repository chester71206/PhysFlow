"""Run projected-flow optimization on the nine PAC-NeRF scenes.

Elastic/Newtonian use validated reverse AD. Plasticine, non-Newtonian and sand
use the original forward solver with a slower finite-difference optimizer.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supplemental_pipeline.simulation_timing import resolve_schedule
SCENES = ('letter', 'droplet', 'torus', 'bird', 'playdoh', 'cat', 'cream', 'toothpaste', 'trophy')
AD_MATERIALS = ('elastic', 'newtonian')
FD_MATERIALS = ('metal', 'foam', 'plasticine', 'non_newtonian', 'sand')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', choices=SCENES, action='append', help='Repeat to select scenes; default all nine')
    p.add_argument('--root', type=Path, default=Path('outputs/pacnerf_projected_all'))
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--flow-objective', choices=('points', 'wonderplay'), default='points',
                   help='WonderPlay rasterized flow is available only for reverse AD scenes')
    p.add_argument('--max-points', type=int, default=1000,
                   help='Gaussian count for WonderPlay rasterized flow; original point loss uses trainer default')
    p.add_argument('--gpu', default=os.environ.get('GPU', '0'))
    p.add_argument('--python', default=os.environ.get('PHYSFLOW_PYTHON', sys.executable),
                   help='Python in the physflow environment')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if a.epochs < 1:
        p.error('--epochs must be positive')
    jobs = []
    for scene in a.scene or SCENES:
        source = ROOT / ('config/gradient_repair/letter_flow_ad.json' if scene == 'letter'
                         else f'config/pacnerf/{scene}_config.json')
        config = json.loads(source.read_text())
        material = config['material']
        if material not in AD_MATERIALS + FD_MATERIALS:
            raise ValueError(f'No projected-flow backend for {scene}: {material}')
        config['training_objective'] = 'flow'
        config['guidance_background'] = 'white_masked'
        if material == 'newtonian':
            config['newtonian_model'] = 'strain_rate'
        model = ROOT / 'model/pacnerf' / scene
        frame_count = len(list((model / 'images_generated').glob('*.png')))
        missing = []
        if frame_count < 2:
            missing.append('images_generated/*.png')
        if not (model / 'point_cloud').is_dir():
            missing.append('point_cloud')
        if not (model / 'masks').is_dir():
            missing.append('masks')
        mask_aliases = {}
        if frame_count >= 2 and (model / 'masks').is_dir():
            plan = resolve_schedule(model, config, frame_count - 1)
            for frame in plan['selected_frames']:
                mask = model / 'masks' / ('a_' + Path(frame['filename']).stem[2:] + '.png')
                if not mask.is_file():
                    stem = Path(frame['filename']).stem[2:]
                    prefix, number = stem.rsplit('_', 1)
                    padded = model / 'masks' / f'a_{prefix}_{int(number):02d}.png'
                    if padded.is_file():
                        mask_aliases[mask.name] = str(padded.resolve())
                    else:
                        missing.append(str(mask.relative_to(model)))
        backend = 'reverse_AD' if material in AD_MATERIALS else 'central_finite_difference'
        if a.flow_objective == 'wonderplay' and backend != 'reverse_AD':
            raise ValueError(f'{scene} uses {backend}; WonderPlay loss is not implemented for this backend')
        jobs.append(dict(scene=scene,material=material,backend=backend,frames=frame_count,
                         source_config=str(source),missing_assets=missing,
                         mask_aliases=mask_aliases,config=config))
    if a.plan_only:
        print(json.dumps([{k:v for k,v in job.items() if k != 'config'} for job in jobs],indent=2))
        return
    unavailable = [j for j in jobs if j['missing_assets']]
    if unavailable:
        raise SystemExit('Missing scene assets: ' + '; '.join(
            f"{j['scene']}: {', '.join(j['missing_assets'])}" for j in unavailable))
    a.root.mkdir(parents=True, exist_ok=False)
    manifest = []
    for job in jobs:
        scene = job['scene']
        directory = a.root / scene
        directory.mkdir()
        model_path = ROOT / 'model/pacnerf' / scene
        if job['mask_aliases']:
            overlay = directory / 'model'
            overlay.mkdir()
            for item in model_path.iterdir():
                if item.name != 'masks':
                    (overlay / item.name).symlink_to(item.resolve())
            (overlay / 'masks').mkdir()
            for item in (model_path / 'masks').iterdir():
                (overlay / 'masks' / item.name).symlink_to(item.resolve())
            for alias, source in job['mask_aliases'].items():
                (overlay / 'masks' / alias).symlink_to(source)
            model_path = overlay
        config_path = directory / 'physics_config.json'
        config_path.write_text(json.dumps(job['config'],indent=2) + '\n')
        output = directory / 'train'
        if job['backend'] == 'reverse_AD':
            command = ['bash',str(ROOT / 'scripts/train_flow.sh')]
        else:
            command = [a.python,str(ROOT / 'scripts/train_projected_flow_fd.py')]
        command += ['--model-path',str(model_path.resolve()),
                    '--physics-config',str(config_path.resolve()),
                    '--output',str(output.resolve()),'--epochs',str(a.epochs)]
        if a.flow_objective == 'wonderplay':
            command += ['--flow-objective', 'wonderplay', '--max-points', str(a.max_points)]
        record = {k:v for k,v in job.items() if k != 'config'}
        record.update(command=command,status='running')
        manifest.append(record)
        (a.root / 'manifest.json').write_text(json.dumps(manifest,indent=2) + '\n')
        env = os.environ.copy()
        env.update(GPU=str(a.gpu),CUDA_VISIBLE_DEVICES=str(a.gpu),
                   CUDA_HOME=env.get('CUDA_HOME',str(Path(sys.prefix))),
                   WANDB_MODE='disabled',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                   PYTHONUNBUFFERED='1',PYTHON=a.python)
        with (directory / 'train.log').open('w') as log:
            code = subprocess.call(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record['status'] = 'complete' if code == 0 else 'failed'
        record['exit_code'] = code
        (a.root / 'manifest.json').write_text(json.dumps(manifest,indent=2) + '\n')
        if code:
            raise SystemExit(f'{scene} {job["backend"]} failed; see {directory / "train.log"}')
    print(a.root / 'manifest.json')


if __name__ == '__main__':
    main()
