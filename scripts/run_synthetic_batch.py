"""Sequential checkpointed synthetic MPM pilots on one selected idle GPU."""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path('outputs/synthetic_physics_batch_20261121')
PY=sys.executable
BLENDER=os.environ.get('SYNTH_BLENDER','.tools/blender/blender-4.5.3-linux-x64/blender')
SCENES=[
 dict(name='baseball_drop',asset='baseball_01',mesh='mesh.npz',obj='baseball_01',E=15000,density=950,height=.055,spacing=.0075,duration=2.0),
 dict(name='pillow01_drop',asset='throw_pillows_01',mesh='pillow01_mesh.npz',obj='throw_pillows_01_pillow01',E=18000,density=220,height=.065,spacing=.022,duration=1.0),
 dict(name='pillow02_drop',asset='throw_pillows_01',mesh='pillow02_mesh.npz',obj='throw_pillows_01_pillow02',E=18000,density=220,height=.065,spacing=.022,duration=1.0),
 dict(name='football_deflated_drop',asset='football',mesh='deflated_mesh.npz',obj='football_deflated',E=12000,density=850,height=.07,spacing=.014,duration=1.0),
 dict(name='football_deflated_2s',asset='football',mesh='deflated_mesh.npz',obj='football_deflated',E=12000,density=850,height=.07,spacing=.014,duration=2.0),
 dict(name='football_inflated_drop',asset='football',mesh='inflated_mesh.npz',obj='football_inflated',E=24000,density=850,height=.07,spacing=.014,duration=1.0),
 dict(name='dirty_football_drop',asset='dirty_football',mesh='lod3_mesh.npz',obj='dirty_football_LOD3',E=23000,density=850,height=.07,spacing=.014,duration=1.0),
 dict(name='baseball_oblique_drop',asset='baseball_01',mesh='mesh.npz',obj='baseball_01',E=15000,density=950,height=.05,spacing=.0075,duration=1.0,velocity=(.35,0,0)),
 dict(name='duck_low_drop',asset='rubber_duck_toy',mesh='mesh.npz',obj='rubber_duck_toy',E=50000,density=950,height=.035,spacing=.016,duration=1.0),
]

def run(cmd,log,seconds,env):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:
        try:
            p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=seconds)
            return None if p.returncode==0 else f'exit {p.returncode}; see {log}'
        except subprocess.TimeoutExpired:
            return f'timeout {seconds}s; see {log}'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene',action='append',choices=[s['name'] for s in SCENES],
                    help='limit processing to selected scenes; may be repeated')
    args=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=env.get('SYNTH_GPU','8')
    manifest_path=ROOT/('scene_manifest_selected.csv' if args.scene else 'scene_manifest.csv')
    complete_path=ROOT/('batch_selected_complete.json' if args.scene else 'batch_complete.json')
    rows=[]
    for s in SCENES:
        if args.scene and s['name'] not in args.scene:continue
        out=ROOT/'scenes'/s['name'];out.mkdir(parents=True,exist_ok=True)
        asset=ROOT/'assets'/s['asset']
        stem=s['asset']+'_2k.blend'
        cmd=[PY,'scripts/synthetic_mpm_pilot.py','--mesh',str(asset/s['mesh']),
             '--out',str(out),'--E',str(s['E']),'--density',str(s['density']),
             '--height',str(s['height']),'--spacing',str(s['spacing']),
             '--duration',str(s['duration'])]
        if 'velocity' in s:cmd+=['--velocity',*map(str,s['velocity'])]
        error=None if (out/'trajectory.npz').exists() and (out/'physics.json').exists() else run(cmd,out/'simulate.log',600,env)
        if error is None and (out/'trajectory.npz').exists():
            error=run([PY,'scripts/validate_synthetic_physics.py',str(out)],out/'physics_validate.log',60,env)
        physics=json.loads((out/'physics.json').read_text()) if (out/'physics.json').exists() else {}
        if error is None and physics.get('status')=='PASS_PHYSICS':
            ad_error=None if (out/'reverse_ad_gate.json').exists() else run([PY,'scripts/check_synthetic_mpm_ad.py',str(out)],out/'reverse_ad_gate.log',300,env)
            if ad_error:error=ad_error
        if error is None and physics.get('status')=='PASS_PHYSICS':
            preview=out/'render_pilot'
            if not (preview/'cameras.json').exists():
                render=[BLENDER,'-b','--factory-startup','--python','scripts/render_synthetic_mpm.py','--',
                        str(asset/stem),str(out/'trajectory.npz'),str(preview),'pilot',s['obj']]
                error=run(render,out/'render_pilot.log',600,env)
        status='FAIL' if error or physics.get('status')=='FAIL_PHYSICS' else 'NEEDS_REVIEW'
        row=dict(scene=s['name'],asset=s['asset'],object=s['obj'],E_Pa=s['E'],nu=.32,
                 density_kg_m3=s['density'],status=status,
                 reason=error or (','.join(k for k,v in physics.get('gates',{}).items() if not v) if status=='FAIL' else 'Pilot visual checks pending'),
                 path=str(out),rendered_pilot=str(out/'render_pilot') if (out/'render_pilot/cameras.json').exists() else '')
        rows.append(row)
        with manifest_path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=row.keys());w.writeheader();w.writerows(rows)
        print(json.dumps(row),flush=True)
    complete_path.write_text(json.dumps(dict(scenes=len(rows),status_counts={k:sum(r['status']==k for r in rows) for k in ('PASS','NEEDS_REVIEW','FAIL')}),indent=2))

if __name__=='__main__':main()
