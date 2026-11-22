"""Conservative physics/data/render QA and diagnostic sheets for a batch scene."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def assess(root,stage):
    cfg=json.loads((root/'config.json').read_text())
    phy=json.loads((root/'physics.json').read_text())
    t=np.load(root/'trajectory.npz')
    render=root/('render_full' if stage=='full' else 'render_pilot')
    cameras=json.loads((render/'cameras.json').read_text())
    P=t['positions'];times=t['times'];n=int(t['n_surface'])
    com=P.mean(axis=1,keepdims=True)
    rel=P-P[0:1]-(com-com[0:1])
    deform_frame=int(np.argmax(np.linalg.norm(rel,axis=-1).max(axis=1)))
    contact_frame=phy.get('contact_frame')
    if contact_frame is not None and contact_frame+1<len(P):
        bottom=P[:,:,2].min(axis=1)
        rebound_frame=contact_frame+1+int(np.argmax(bottom[contact_frame+1:]))
    else:rebound_frame=len(P)//2
    representative=set((0,deform_frame,rebound_frame,len(P)-1))
    check=dict(physics=phy['status']=='PASS_PHYSICS',finite_trajectory=bool(np.isfinite(P).all()),
               frame_count=len(times)==len(P) and (stage!='full' or len(times)==len(cameras['frames'])),
               camera_count=len(cameras['cameras'])==3,
               reverse_ad=(root/'reverse_ad_gate.json').exists() and json.loads((root/'reverse_ad_gate.json').read_text()).get('pass_gate',False))
    samples=[]
    render_info=[]
    expected=480 if stage=='full' else 256
    for ci in range(3):
        first=None
        for rec in cameras['frames']:
            fi=rec['index']
            path=render/f'cam_{ci}'/f'{fi:04d}.png'
            mask=render/'passes'/f'cam_{ci}'/f'mask_{fi:04d}_{fi+1:04d}.png'
            depth=render/'passes'/f'cam_{ci}'/f'depth_{fi:04d}_{fi+1:04d}.exr'
            if not path.exists() or not mask.exists() or not depth.exists():
                render_info.append(dict(camera=ci,frame=fi,error='RGB/mask/depth missing'))
                continue
            raw_image=Image.open(path).convert('RGBA')
            rgba=np.asarray(raw_image)
            im=rgba[:,:,:3]
            m=np.asarray(Image.open(mask).convert('L'))>127
            yy,xx=np.where(m)
            if not len(xx):
                render_info.append(dict(camera=ci,frame=fi,error='empty object mask'))
                continue
            margin=min(xx.min(),yy.min(),expected-1-xx.max(),expected-1-yy.max())
            area=float(m.mean())
            render_info.append(dict(camera=ci,frame=fi,mask_fraction=area,margin_px=int(margin),
                                    rgb_size=list(im.shape[:2]),depth_bytes=depth.stat().st_size,
                                    opaque_fraction=float((rgba[:,:,3]==255).mean()),
                                    time_s=float(rec['time_s'])))
            if ci==0 and (stage=='pilot' or fi in representative):
                samples.append((fi,Image.open(path).convert('RGB')))
            if first is None:first=im
    good=[r for r in render_info if 'error' not in r]
    check['all_render_files']=len(good)==3*len(cameras['frames'])
    check['resolution']=all(r['rgb_size']==[expected,expected] for r in good)
    check['object_visible']=all(.01<=r['mask_fraction']<=.7 for r in good)
    check['no_crop']=all(r['margin_px']>=int(.015*expected) for r in good)
    check['opaque_background']=all(r['opaque_fraction']==1.0 for r in good)
    check['timestamps']=bool(np.all(np.diff(times)>0) and all(abs(float(a['time_s'])-float(b))<1e-7 for a,b in zip(cameras['frames'],times))) if stage=='full' else True
    check['render_surface_tracked']=int(t['n_surface'])==len(np.load(cfg['asset'])['vertices'])
    check['camera_calibration']=all(c.get('projection')=='perspective' and
                                    np.isfinite(np.asarray(c.get('c2w',[]))).all() and
                                    np.isfinite(np.asarray(c.get('intrinsic',[]))).all() and
                                    np.asarray(c.get('intrinsic',[])).shape==(3,3) and
                                    c['intrinsic'][0][0]>0 for c in cameras['cameras']) if stage=='full' else True
    check['repeatability']=((root/'repeatability.json').exists() and
                            json.loads((root/'repeatability.json').read_text()).get('pass_gate',False)) if stage=='full' else True
    if stage=='full':
        flow_report=root/'mesh_visible_flow/report.json'
        flow=json.loads(flow_report.read_text()) if flow_report.exists() else {}
        count=sum(1 for _ in (root/'mesh_visible_flow').glob('cam_*/*.npz'))
        check['visible_mesh_flow']=(count==3*(len(P)-1) and
                                    flow.get('min_mask_iou',0)>=.90 and
                                    flow.get('mean_valid_fraction',0)>=.80)
    else:check['visible_mesh_flow']=True
    hard=('physics','finite_trajectory','frame_count','camera_count','reverse_ad',
          'all_render_files','resolution','object_visible','no_crop',
          'opaque_background','render_surface_tracked')
    status=('FAIL' if not all(check[k] for k in hard) else
            'PASS' if stage=='full' and all(check.values()) else
            'READY_FOR_FULL_RENDER' if stage=='pilot' and all(check.values()) else 'NEEDS_REVIEW')
    report=dict(stage=stage,status=status,checks=check,render_frames=render_info,
                physics=phy,notes=['Automated PASS does not certify human-perceived natural motion.',
                                   'PBR appearance is a visual approximation; material E/nu describe homogeneous elastic MPM surrogate.',
                                   'Depth EXR is camera-space distance; object mask uses Blender IndexOB pass.',
                                   'Blender-visible flow and WonderPlay Gaussian flow are distinct.'])
    (root/f'qa_{stage}.json').write_text(json.dumps(report,indent=2))
    if samples:
        w,h=samples[0][1].size
        sheet=Image.new('RGB',(w*len(samples),h+32),'white')
        draw=ImageDraw.Draw(sheet)
        for j,(fi,im) in enumerate(samples):
            sheet.paste(im,(j*w,32));draw.text((j*w+8,8),f'frame {fi}, t={times[fi]:.3f}s',fill=(0,0,0))
        sheet.save(root/f'contact_{stage}.png')
        if len(samples)>1:
            before=np.asarray(samples[0][1]).astype(np.int16)
            after=np.asarray(samples[-1][1]).astype(np.int16)
            diff=np.clip(np.abs(after-before)*3,0,255).astype(np.uint8)
            Image.fromarray(diff).save(root/f'difference_{stage}.png')
    print(json.dumps(dict(status=status,checks=check)))
    return 0 if status in ('READY_FOR_FULL_RENDER','PASS') else 1

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('scene',type=Path);ap.add_argument('--stage',choices=('pilot','full'),required=True)
    a=ap.parse_args();raise SystemExit(assess(a.scene,a.stage))
