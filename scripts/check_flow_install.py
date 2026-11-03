"""Report imports/versions and verify the immutable production physics files."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'utils'), str(ROOT/'gs')]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--cpu-only', action='store_true', help='Check installation without CUDA initialization')
a = p.parse_args()
for name, digest in json.loads((ROOT/'release/frozen_mpm.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
for name in ['torch', 'torchvision', 'warp', 'taichi', 'pytorch3d', 'point_cloud_utils',
             'mcubes', 'cv2', 'omegaconf', 'wandb', 'plyfile', 'h5py',
             'simple_knn._C', 'diff_plane_rasterization',
             'mpm_solver_warp.repaired_temporal_rollout', 'video_distillation.cogv_guidance']:
    module = importlib.import_module(name)
    print(name, getattr(module, '__version__', ''), getattr(module, '__file__', ''), flush=True)
import torch
import diff_plane_rasterization
installed = Path(diff_plane_rasterization.__file__)
expected = ROOT/'submodules/diff-plane-rasterization/diff_plane_rasterization/__init__.py'
assert installed.read_bytes() == expected.read_bytes(), 'Installed rasterizer wrapper differs; rebuild with scripts/install_flow.sh'
for name, digest in json.loads((ROOT/'release/rasterizer_source.json').read_text())['sha256'].items():
    assert hashlib.sha256((ROOT/'submodules/diff-plane-rasterization'/name).read_bytes()).hexdigest() == digest, name
if not a.cpu_only:
    assert torch.cuda.is_available(), 'A compatible NVIDIA GPU/driver is required'
    assert torch.version.cuda == '11.8', torch.version.cuda
    print('CUDA', torch.cuda.get_device_name(0))
print('PASS: imports and all ten frozen MPM hashes')
