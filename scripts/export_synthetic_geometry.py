"""Write world-space tracked mesh and particle point clouds from Warp MPM output."""
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh

ap=argparse.ArgumentParser();ap.add_argument('scene',type=Path);a=ap.parse_args()
root=a.scene
data=np.load(root/'trajectory.npz');P=data['positions'];F=data['faces'];ns=int(data['n_surface'])
meshdir=root/'mesh_frames';partdir=root/'particle_frames'
meshdir.mkdir(exist_ok=True);partdir.mkdir(exist_ok=True)
for fi,p in enumerate(P):
    m=meshdir/f'{fi:04d}.ply'
    q=partdir/f'{fi:04d}.ply'
    if not m.exists():trimesh.Trimesh(vertices=p[:ns],faces=F,process=False).export(m)
    if not q.exists():trimesh.points.PointCloud(p).export(q)
(root/'geometry_manifest.json').write_text(json.dumps(dict(mesh_vertices=ns,particles=P.shape[1],
  faces=len(F),frames=len(P),particle_position_unit='m',mesh_frame_pattern=str(meshdir/'%04d.ply'),
  particle_frame_pattern=str(partdir/'%04d.ply'),
  source_mesh_with_uv='See source .blend in batch assets; PLY frames are geometry only'),indent=2))
print('EXPORTED',root,len(P))
