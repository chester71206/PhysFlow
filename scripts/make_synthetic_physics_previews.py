"""Plot saved MPM particles and tracked mesh at initial, impact, and final times."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

for arg in sys.argv[1:]:
    root=Path(arg);d=np.load(root/'trajectory.npz');P=d['positions'];F=d['faces'];ns=int(d['n_surface'])
    cfg=json.loads((root/'config.json').read_text())
    bottom=P[:,:,2].min(1);impact=int(np.argmin(bottom))
    frames=[0,impact,len(P)-1]
    for kind in ('particle','mesh'):
        fig,ax=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
        for col,fi in enumerate(frames):
            p=P[fi];a=ax[col]
            if kind=='particle':
                a.scatter(p[ns:,0],p[ns:,2],s=.4,c='#1b7895',alpha=.45)
                a.scatter(p[:ns,0],p[:ns,2],s=.7,c='#ff6838',alpha=.7)
            else:
                edges=np.unique(np.sort(np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0)
                seg=p[edges][:,:,[0,2]]
                a.add_collection(LineCollection(seg,colors='#2e586b',linewidths=.25,alpha=.7))
            a.axhline(cfg['ground_z_m'],color='black',linewidth=1)
            a.set_title(f'frame {fi}, t={d["times"][fi]:.3f}s')
            a.set_aspect('equal');a.set_xlim(P[:,:,0].min()-.04,P[:,:,0].max()+.04)
            a.set_ylim(cfg['ground_z_m']-.03,max(P[:,:,2].max()+.03,cfg['ground_z_m']+.2))
            a.set_xlabel('x (m)')
        ax[0].set_ylabel('z (m)')
        fig.suptitle(f'{root.name}: saved Warp MPM {kind} trajectory')
        fig.tight_layout();fig.savefig(root/f'{kind}_preview.png',dpi=150);plt.close(fig)
    print('PREVIEW',root)
