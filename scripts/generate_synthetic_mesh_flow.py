"""Rasterize tracked mesh correspondences and visibility for Blender camera views.

This is geometric visible-surface flow. It is distinct from WonderPlay Gaussian
rasterized flow and from brightness-constancy optical flow.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def project(world,c2w,K,width,height):
    w2c=np.linalg.inv(c2w)
    p=world@w2c[:3,:3].T+w2c[:3,3]
    z=-p[:,2]
    return np.stack((K[0,0]*p[:,0]/z+K[0,2],K[1,2]-K[1,1]*p[:,1]/z,z),axis=1)

def raster(verts,faces,width,height):
    depth=np.full((height,width),np.inf,np.float32)
    face_id=np.full((height,width),-1,np.int32)
    bary=np.zeros((height,width,3),np.float32)
    for fi,tri in enumerate(faces):
        p=verts[tri,:2]
        lo=np.maximum(np.floor(p.min(0)-1).astype(int),0)
        hi=np.minimum(np.ceil(p.max(0)+1).astype(int),[width-1,height-1])
        if (hi<lo).any():continue
        a,b,c=p
        den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-8:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        xx=xx+.5;yy=yy+.5
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        w=1-u-v
        iz=u/verts[tri[0],2]+v/verts[tri[1],2]+w/verts[tri[2],2]
        z=1/np.maximum(iz,1e-12)
        sub=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        ok=(u>=-1e-5)&(v>=-1e-5)&(w>=-1e-5)&(z>0)&(z<sub)
        sub[ok]=z[ok]
        ids=face_id[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        ids[ok]=fi
        weights=bary[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        weights[ok]=np.stack((u[ok]/verts[tri[0],2],v[ok]/verts[tri[1],2],
                              w[ok]/verts[tri[2],2]),axis=1)*z[ok,None]
    return depth,face_id,bary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('scene',type=Path);a=ap.parse_args()
    root=a.scene;traj=np.load(root/'trajectory.npz');P=traj['positions'];F=traj['faces'];ns=int(traj['n_surface'])
    render=root/'render_full';cams=json.loads((render/'cameras.json').read_text())['cameras']
    out=root/'mesh_visible_flow';out.mkdir(exist_ok=True)
    summary=[]
    for ci,cam in enumerate(cams):
        size=Image.open(render/f'cam_{ci}/0000.png').size
        width,height=size
        c2w=np.asarray(cam['c2w'],dtype=np.float64)
        K=np.asarray(cam['intrinsic'],dtype=np.float64)
        prev=None
        view=out/f'cam_{ci}';view.mkdir(exist_ok=True)
        for frame in range(len(P)):
            verts=project(P[frame,:ns],c2w,K,width,height)
            current=raster(verts,F,width,height)
            mask_path=render/'passes'/f'cam_{ci}'/f'mask_{frame:04d}_{frame+1:04d}.png'
            mask=np.asarray(Image.open(mask_path).convert('L'))>127
            raster_mask=current[1]>=0
            union=np.logical_or(mask,raster_mask).sum()
            iou=float(np.logical_and(mask,raster_mask).sum()/max(union,1))
            if prev is not None:
                pverts,pdepth,pids,pbary,pmask,piou=prev
                target=out/f'cam_{ci}'/f'{frame-1:04d}.npz'
                if not target.exists():
                    yy,xx=np.nonzero((pids>=0)&pmask)
                    tris=F[pids[yy,xx]]
                    weights=pbary[yy,xx]
                    next_world=(P[frame,:ns][tris]*weights[:,:,None]).sum(1)
                    next_pos=project(next_world,c2w,K,width,height)
                    tx=np.floor(next_pos[:,0]).astype(int);ty=np.floor(next_pos[:,1]).astype(int)
                    inside=(tx>=0)&(tx<width)&(ty>=0)&(ty<height)&(next_pos[:,2]>0)
                    valid=np.zeros(len(xx),bool)
                    q=np.where(inside)[0]
                    valid[q]=mask[ty[q],tx[q]]&(np.abs(current[0][ty[q],tx[q]]-next_pos[q,2])<=.0015)
                    flow=np.zeros((height,width,2),np.float32)
                    validity=np.zeros((height,width),np.uint8)
                    flow[yy[valid],xx[valid],0]=next_pos[valid,0]-(xx[valid]+.5)
                    flow[yy[valid],xx[valid],1]=next_pos[valid,1]-(yy[valid]+.5)
                    validity[yy[valid],xx[valid]]=1
                    np.savez_compressed(target,flow_px=flow,valid=validity,
                                        t0_s=float(traj['times'][frame-1]),t1_s=float(traj['times'][frame]))
                    valid_frac=float(valid.sum()/max(len(xx),1))
                else:
                    valid_frac=float(np.load(target)['valid'].sum()/max(pmask.sum(),1))
                summary.append(dict(camera=ci,frame=frame-1,mask_iou=piou,valid_fraction=valid_frac))
            prev=(verts,*current,mask,iou)
        print('FLOW_VIEW',ci,'frames',len(P)-1,flush=True)
    report=dict(kind='tracked_surface_geometric_visible_flow',
                not_equal_to='WonderPlay Gaussian rasterized flow or photometric optical flow',
                visibility='source Blender object-index mask AND current mesh z-buffer; destination Blender mask AND 1.5 mm mesh z-buffer tolerance',
                camera_projection='perspective, fixed c2w and K',
                mean_mask_iou=float(np.mean([r['mask_iou'] for r in summary])),
                min_mask_iou=float(min(r['mask_iou'] for r in summary)),
                mean_valid_fraction=float(np.mean([r['valid_fraction'] for r in summary])),
                frames=summary)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('mean_mask_iou','min_mask_iou','mean_valid_fraction')}))

if __name__=='__main__':main()
