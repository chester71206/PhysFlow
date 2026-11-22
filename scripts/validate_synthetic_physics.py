"""Add precontact ballistic and rebound gates to recorded Warp MPM trajectories."""
import json
import sys
from pathlib import Path
import numpy as np

for arg in sys.argv[1:]:
    root=Path(arg)
    cfg=json.loads((root/'config.json').read_text())
    path=root/'physics.json'
    report=json.loads(path.read_text())
    traj=np.load(root/'trajectory.npz')
    P,V,t=traj['positions'],traj['velocities'],traj['times']
    ns=int(traj['n_surface']);faces=traj['faces'];surf=P[:,:ns]
    areas=np.linalg.norm(np.cross(surf[:,faces[:,1]]-surf[:,faces[:,0]],
                                  surf[:,faces[:,2]]-surf[:,faces[:,0]]),axis=-1)/2
    face_ratios=areas/np.maximum(areas[0],1e-12)
    bottom=P[:,:,2].min(1)
    cm=P.mean(1)
    dx=cfg['grid_lim_m']/cfg['grid_n']
    contact=np.where(bottom<=cfg['ground_z_m']+dx/2)[0]
    if len(contact):
        ci=int(contact[0])
        pre=np.arange(ci)
        expected=cm[0,2]+cfg['initial_velocity_m_s'][2]*t[pre]-.5*9.8*t[pre]**2
        ballistic_error=float(np.max(np.abs(cm[pre,2]-expected))) if len(pre) else None
        rebound=float(np.max(bottom[ci+1:]-cfg['ground_z_m'])) if ci+1<len(t) else 0.
        upward=float(np.max(V[ci+1:,:,2].mean(1))) if ci+1<len(t) else 0.
    else:
        ci=None;ballistic_error=None;rebound=0.;upward=0.
    report['contact_frame']=ci
    report['contact_time_s']=float(t[ci]) if ci is not None else None
    report['precontact_ballistic_max_error_m']=ballistic_error
    report['postcontact_bottom_rebound_m']=rebound
    report['postcontact_max_com_upward_speed_m_s']=upward
    report['min_face_area_ratio']=float(face_ratios.min())
    report['max_face_area_ratio']=float(face_ratios.max())
    report['thresholds'].update(precontact_ballistic_error_m=.003,
                                min_rebound_gap_m=.005,min_com_upward_speed_m_s=.10,
                                min_face_area_ratio=.25,max_face_area_ratio=2.5)
    report['gates'].update(precontact_ballistic=ballistic_error is not None and ballistic_error<=.003,
                           rebound=ci is not None and rebound>=.005 and upward>=.10,
                           no_face_collapse=bool(face_ratios.min()>=.25 and face_ratios.max()<=2.5))
    report['status']='PASS_PHYSICS' if all(report['gates'].values()) else 'FAIL_PHYSICS'
    path.write_text(json.dumps(report,indent=2))
    print(root.name,report['status'],{'ballistic_error':ballistic_error,'rebound':rebound,'upward_speed':upward})
