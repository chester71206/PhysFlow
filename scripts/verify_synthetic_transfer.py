"""Verify the compact Git transfer has every input needed for synthetic training."""
import json
from pathlib import Path

import numpy as np

ROOT=Path('outputs/synthetic_physics_batch_20261121')
SCENES=('baseball_drop','football_deflated_2s','duck_low_drop')
for scene in SCENES:
    folder=ROOT/'scenes'/scene
    cfg=json.loads((folder/'config.json').read_text())
    asset=Path(cfg['asset'])
    assert asset.is_file(),asset
    with np.load(asset) as mesh,np.load(folder/'trajectory.npz') as traj:
        assert len(mesh['vertices'])==int(traj['n_surface']),scene
        assert traj['positions'].shape[0]==len(traj['times']),scene
        assert np.isfinite(traj['positions']).all(),scene
    cameras=json.loads((folder/'render_full/cameras.json').read_text())
    assert len(cameras['cameras'])==3,scene
    for view in range(3):
        assert (folder/'render_full'/f'cam_{view}.mp4').is_file(),(scene,view)
        if scene!='duck_low_drop':
            for frame in (0,2,4,6,8):
                mask=folder/'render_full/passes'/f'cam_{view}'/f'mask_{frame:04d}_{frame+1:04d}.png'
                assert mask.is_file(),mask
            for frame in range(8):
                flow=folder/'mesh_visible_flow'/f'cam_{view}'/f'{frame:04d}.npz'
                assert flow.is_file(),flow
    print(scene,'PASS',flush=True)
print('Synthetic transfer inputs complete.')
