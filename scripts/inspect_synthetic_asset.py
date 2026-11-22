"""Inspect a licensed Blender mesh asset for MPM volume sampling suitability."""
import bpy
import bmesh
import json
import sys
from pathlib import Path

args = sys.argv[sys.argv.index('--') + 1:]
source, output = map(Path, args[:2])
bpy.ops.wm.open_mainfile(filepath=str(source.resolve()))
rows = []
for obj in bpy.data.objects:
    if obj.type != 'MESH':
        continue
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    boundary = sum(not e.is_manifold for e in bm.edges)
    rows.append(dict(name=obj.name, vertices=len(obj.data.vertices), faces=len(obj.data.polygons),
                     nonmanifold_edges=boundary, dimensions=list(obj.dimensions),
                     location=list(obj.location), uv_layers=len(obj.data.uv_layers),
                     material_names=[s.material.name if s.material else None for s in obj.material_slots],
                     visible=not obj.hide_render))
    bm.free()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows))
