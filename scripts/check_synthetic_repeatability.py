"""Rerun a saved scene from its config and compare the complete tracked trajectory."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]);cfg=json.loads((root/'config.json').read_text())
replay=root/'replay_check'
cmd=[sys.executable,'scripts/synthetic_mpm_pilot.py',
     '--mesh',cfg['asset'],'--out',str(replay),'--E',str(cfg['E_Pa']),
     '--nu',str(cfg['nu']),'--density',str(cfg['density_kg_m3']),
     '--height',str(cfg['height_m']),'--duration',str(cfg['duration_s']),
     '--fps',str(cfg['fps']),'--dt',str(cfg['dt_s']),
     '--spacing',str(cfg['particle_spacing_m']),'--grid',str(cfg['grid_n']),
     '--ground',str(cfg['ground_z_m']),'--velocity',*map(str,cfg['initial_velocity_m_s'])]
cmd+=['--initial-particles',str(root/'trajectory.npz')]
with (root/'replay_check.log').open('w') as f:
    p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy(),timeout=600)
if p.returncode:raise RuntimeError('Replay failed; see replay_check.log')
a=np.load(root/'trajectory.npz');b=np.load(replay/'trajectory.npz')
same=a['positions'].shape==b['positions'].shape and np.array_equal(a['times'],b['times'])
error=float(np.max(np.abs(a['positions']-b['positions']))) if same else None
report=dict(same_shape_and_times=same,max_position_difference_m=error,
            threshold_m=1e-5,pass_gate=same and error<=1e-5)
(root/'repeatability.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
if not report['pass_gate']:raise SystemExit(1)
