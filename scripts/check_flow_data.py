"""CPU-only data/timestamp preflight; does not load RAFT or initialize CUDA."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supplemental_pipeline.simulation_timing import prepare_run

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene', default='letter')
p.add_argument('--model-root', type=Path, default=ROOT/'model/pacnerf')
a = p.parse_args()
config = ROOT/('config/gradient_repair/letter_flow_ad.json' if a.scene=='letter' else f'config/pacnerf/{a.scene}_config.json')
model = a.model_root/a.scene
count = len(list((model/'images_generated').glob('*.png')))
if count < 2:
    raise FileNotFoundError(f'Install the PAC-NeRF assets first: {model}/images_generated')
for required in [model/'cameras.json', model/'point_cloud']:
    if not required.exists():
        raise FileNotFoundError(required)
if not list((model/'point_cloud').glob('iteration_*/point_cloud.ply')):
    raise FileNotFoundError('Missing Gaussian point_cloud/iteration_*/point_cloud.ply')
# Enforce the production observation config without changing the source config.
from supplemental_pipeline.simulation_timing import resolve_schedule
config_data = json.loads(config.read_text())
plan = resolve_schedule(model, config_data, count-1)
for f in plan['selected_frames']:
    mask = model/'masks'/('a_'+Path(f['filename']).stem[2:]+'.png')
    if not mask.is_file():
        raise FileNotFoundError(f'Supplied RGBA image required: {mask}')
print(json.dumps(dict(scene=a.scene,material=config_data['material'],projected_training_supported=config_data['material'] in ['elastic','newtonian'],
    frames=count,timing=plan['source'],target_steps=plan['target_steps'],dt=plan['substep_dt']),indent=2))
print('PASS: source frames, RGBA masks, Gaussian checkpoint and explicit timestamps')
