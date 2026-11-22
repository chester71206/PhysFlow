"""Export evaluated mesh coordinates and topology from a Blender CC0 asset."""
import bpy
import json
import numpy as np
import sys
from pathlib import Path
from mathutils import Vector

args=sys.argv[sys.argv.index('--') + 1:]
source, output = map(Path, args[:2])
bpy.ops.wm.open_mainfile(filepath=str(source.resolve()))
name=args[2] if len(args)>2 else None
obj = bpy.data.objects[name] if name else next(o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_render)
mesh = obj.data
verts = np.asarray([tuple(obj.matrix_world @ v.co) for v in mesh.vertices], dtype=np.float32)
mesh.calc_loop_triangles()
faces = np.asarray([tuple(t.vertices) for t in mesh.loop_triangles], dtype=np.int32)
output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output, vertices=verts, faces=faces)
print(json.dumps(dict(vertices=len(verts), triangles=len(faces), bounds=[verts.min(0).tolist(), verts.max(0).tolist()])))
