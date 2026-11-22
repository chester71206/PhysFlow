"""Update quality manifest, contact sheet, and MP4s from completed scene artifacts."""
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('outputs/synthetic_physics_batch_20261121')
PILOT_NOTES={
 'football_deflated_drop':'1-second Pilot superseded by completed 2-second PASS version',
 'football_inflated_drop':'Pilot gates pass; formal render deferred because inflated shell pressure is absent from homogeneous-solid model',
 'dirty_football_drop':'Pilot gates pass; visually similar football drop deferred to prioritize distinct motion',
 'duck_low_drop':'Pilot gates pass; rebound gap only slightly above 5 mm, visual naturalness needs review',
}
manifest=ROOT/'scene_manifest.csv'
rows=list(csv.DictReader(manifest.open()))
qualified=[]
for r in rows:
    out=Path(r['path'])
    qa_path=out/'qa_full.json'
    pilot=out/'qa_pilot.json'
    if qa_path.exists():
        qa=json.loads(qa_path.read_text())
        r['status']=qa['status']
        r['reason']='Automated physics/data/render gates passed; human visual review still advised' if qa['status']=='PASS' else ','.join(k for k,v in qa['checks'].items() if not v)
        if qa['status']=='PASS': qualified.append(r)
    elif pilot.exists() and r['status']!='FAIL':
        qa=json.loads(pilot.read_text())
        r['status']='FAIL' if qa['status']=='FAIL' else 'NEEDS_REVIEW'
        r['reason']=(PILOT_NOTES.get(r['scene'],'Pilot ready; formal render pending') if qa['status']=='READY_FOR_FULL_RENDER'
                     else ','.join(k for k,v in qa['checks'].items() if not v))
    if r['status']=='PASS':r['output_path']=str(out/'render_full')
    elif r['status']=='NEEDS_REVIEW':r['output_path']=str(out/'render_pilot')
    else:r['output_path']=''
with manifest.open('w',newline='') as f:
    columns=list(rows[0])
    if 'output_path' not in columns:columns.append('output_path')
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

thumbs=[]
for r in rows:
    out=Path(r['path'])
    stage='full' if (out/'contact_full.png').exists() else 'pilot'
    path=out/f'contact_{stage}.png'
    if not path.exists():path=out/'particle_preview.png'
    if not path.exists():continue
    im=Image.open(path).convert('RGB');im.thumbnail((720,230))
    thumbs.append((r['scene'],r['status'],im))
sheet=Image.new('RGB',(750,max(1,len(thumbs))*265),'white')
draw=ImageDraw.Draw(sheet)
for i,(name,status,im) in enumerate(thumbs):
    draw.text((10,i*265+5),f'{name} — {status}',fill=(0,0,0))
    sheet.paste(im,(10,i*265+28))
sheet.save(ROOT/'contact_sheet.png')

ffmpeg='/usr/bin/ffmpeg' if Path('/usr/bin/ffmpeg').exists() else shutil.which('ffmpeg')
if ffmpeg:
    encoders=subprocess.run([ffmpeg,'-hide_banner','-encoders'],capture_output=True,text=True,check=True).stdout
    video_args=(['-c:v','libx264','-crf','18'] if 'libx264' in encoders else
                ['-c:v','mpeg4','-q:v','2'])
    comparison_args=(['-c:v','libx264','-crf','20'] if 'libx264' in encoders else
                     ['-c:v','mpeg4','-q:v','3'])
    for r in qualified:
        out=Path(r['path'])
        for ci in range(3):
            inp=out/'render_full'/f'cam_{ci}'/'%04d.png'
            dest=out/'render_full'/f'cam_{ci}.mp4'
            if not dest.exists():
                subprocess.run([ffmpeg,'-y','-hide_banner','-loglevel','error','-framerate','24',
                                '-i',str(inp),*video_args,'-pix_fmt','yuv420p',str(dest)],check=True)
    frames=ROOT/'comparison_frames';frames.mkdir(exist_ok=True)
    index=0
    for r in qualified:
        out=Path(r['path'])/'render_full'
        images=sorted((out/'cam_0').glob('*.png'))
        for p in images:
            panel=Image.new('RGB',(1440,520),'white')
            draw=ImageDraw.Draw(panel)
            draw.text((12,12),r['scene']+' | three fixed cameras | automated PASS',fill=(0,0,0))
            for ci in range(3):
                panel.paste(Image.open(out/f'cam_{ci}'/p.name).convert('RGB'),(ci*480,40))
            panel.save(frames/f'{index:05d}.png');index+=1
    if index:
        subprocess.run([ffmpeg,'-y','-hide_banner','-loglevel','error','-framerate','24',
                        '-i',str(frames/'%05d.png'),*comparison_args,
                        '-pix_fmt','yuv420p',str(ROOT/'comparison.mp4')],check=True)

report=['# Synthetic physics batch quality report','',
        'PASS means automated physics, data, and rendering gates passed. It does not certify fully natural motion by human inspection.','',
        '| Scene | Status | E (Pa) | ν | Density (kg/m³) | Max penetration (mm) | Max relative deformation (mm) | Rebound gap (mm) | Max edge strain | Reason |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
for r in rows:
    p=Path(r['path'])/'physics.json'
    d=json.loads(p.read_text()) if p.exists() else {}
    mm=lambda name: f"{1000*d[name]:.2f}" if d.get(name) is not None else '—'
    strain=f"{d['max_edge_strain']:.3f}" if d.get('max_edge_strain') is not None else '—'
    report.append(f"| {r['scene']} | {r['status']} | {r['E_Pa']} | {r['nu']} | {r['density_kg_m3']} | {mm('max_ground_penetration_m')} | {mm('max_relative_displacement_m')} | {mm('postcontact_bottom_rebound_m')} | {strain} | {r['reason']} |")
report+=['','## Fixed numeric gates','',
         '- All particle positions and velocities finite.','- Maximum speed < 8 m/s and particle displacement < 0.8 m.',
         '- Maximum ground penetration ≤ one grid cell (1/64 m = 15.625 mm).',
         '- Nonrigid relative displacement ≥ 5 mm and maximum tracked mesh edge strain < 1.5.',
         '- Tracked mesh triangle area remains between 0.25× and 2.5× its initial area.',
         '- Precontact center-of-mass ballistic error ≤ 3 mm.',
         '- Postcontact bottom separation ≥ 5 mm and upward center-of-mass speed ≥ 0.1 m/s.',
         '- Three fixed views, each frame 480×480 for formal output; masks cover 1–70% of frame with ≥1.5% border.',
         '- Exact asset-to-MPM surface vertex count, matching timestamps, reverse AD short-path finite gradients and forward parity.',
         '- Complete replay uses the saved initial particles and differs by ≤ 0.01 mm at every saved particle position.',
         '- Geometric visible-surface flow has Blender-mask IoU ≥ 0.90 in every sampled pair and mean valid foreground ≥ 0.80.',
         '', '## Remaining human review','',
         '- Inspect full videos for plausible deformation, texture sliding, shadow behavior, and impact/rebound timing.',
         '- Homogeneous elastic MPM is an approximation for stitched or inflated real-world assets.',
         '- The first football replay re-voxelized the source mesh and produced a different interior particle count. GT/Candidate replays now load the saved initial positions and particle volumes, which is the authoritative initial state; the failed replay remains archived.',
         '- Blender visible-surface geometric flow differs from WonderPlay Gaussian rasterized flow.',
         '- Full-length reverse AD gradient quality and end-to-end PhysFlow training are not established by the short-path gate.',
         '- Poly Haven American Football was rejected at geometry preflight (28 nonmanifold edges), so it was not counted as a Pilot.',
         '- A preliminary rubber-duck high-drop run exceeded the mesh strain gate. The lower-drop candidate passes automated full QA; its small rebound and perceived naturalness still need human review.',
         '']
(ROOT/'quality_report.md').write_text('\n'.join(report))
print(json.dumps(dict(status_counts={k:sum(r['status']==k for r in rows) for k in ['PASS','NEEDS_REVIEW','FAIL']},
                      qualified=[r['scene'] for r in qualified],comparison=(ROOT/'comparison.mp4').exists()),indent=2))
