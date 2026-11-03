"""Small observation-only AD check; no CUDA, assets, renderer or RAFT needed."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from utils.projected_flow_objective import FrozenVisibilityProjectedFlow

for f,digest in json.loads((ROOT/'release/frozen_mpm.json').read_text()).items():
    assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest, f
torch.manual_seed(0)
camera=SimpleNamespace(full_proj_transform=torch.eye(4,dtype=torch.double),world_view_transform=torch.eye(4,dtype=torch.double))
xy=torch.cartesian_prod(torch.linspace(-.65,.65,5),torch.linspace(-.65,.65,5)).double()
points=torch.cat([xy,torch.ones((len(xy),1),dtype=torch.double)],dim=1)
positions=torch.stack([points,points+torch.tensor([.03,.01,0.]),points+torch.tensor([.05,.04,0.])]).double()
flow=torch.randn(2,2,32,32,dtype=torch.double)*.1
images=torch.zeros(3,3,32,32,dtype=torch.double)
objective=FrozenVisibilityProjectedFlow(positions,[camera]*3,flow,images,torch.ones(len(points)),max_points=25)
assert torch.autograd.gradcheck(objective,(positions.requires_grad_(True),),eps=1e-5,atol=1e-6,rtol=1e-4)
print('PASS: frozen source hashes and observation position AD vs central FD, float64, 25 points/2 pairs')
