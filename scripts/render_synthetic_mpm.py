"""Render only recorded Warp MPM surface trajectories with the source asset PBR mesh."""
import bpy
import json
import math
import os
import numpy as np
import sys
from pathlib import Path
from mathutils import Vector

args=sys.argv[sys.argv.index('--') + 1:]
asset, trajectory, output, mode = args[:4]
object_name=args[4] if len(args)>4 else None
asset, trajectory, output = Path(asset).resolve(), Path(trajectory).resolve(), Path(output).resolve()
output.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(asset))
obj = bpy.data.objects[object_name] if object_name else next(o for o in bpy.data.objects if o.type=='MESH' and not o.hide_render)
obj.hide_render=False
for collection in bpy.data.collections:
    if obj.name in collection.all_objects: collection.hide_render=False
for other in bpy.data.objects:
    if other.type=='MESH' and other!=obj: other.hide_render=True
if obj.data.shape_keys is not None:
    # Poly Haven baseball has a Basis key; leaving it would override tracked mesh coordinates.
    obj.shape_key_clear()
for modifier in obj.modifiers:
    # Surface coordinates in the dataset must be the same tracked mesh as the renderer.
    modifier.show_render=False
data = np.load(trajectory)
P, times = data['positions'], data['times']
ns = int(data['n_surface'])
if len(obj.data.vertices)!=ns:
    raise RuntimeError('Asset surface vertex count does not match MPM tracked particles')
for im in bpy.data.images:
    if im.source=='FILE' and im.filepath:
        path=asset.parent/'textures'/Path(im.filepath).name
        if path.exists(): im.filepath=str(path)

scene=bpy.context.scene
scene.render.engine='CYCLES'
scene.cycles.samples=int(os.environ.get('SYNTH_SAMPLES','12' if mode=='pilot' else '48'))
scene.cycles.use_denoising=True
scene.render.resolution_x=256 if mode=='pilot' else 480
scene.render.resolution_y=256 if mode=='pilot' else 480
scene.render.resolution_percentage=100
scene.render.use_border=False
scene.render.use_crop_to_border=False
scene.render.image_settings.file_format='PNG'
scene.render.image_settings.color_mode='RGBA'
scene.render.film_transparent=False
scene.view_settings.view_transform='AgX'
scene.view_layers[0].use_pass_z=True
scene.view_layers[0].use_pass_object_index=True
obj.pass_index=1
scene.use_nodes=True
nodes=scene.node_tree.nodes
nodes.clear()
layer=nodes.new('CompositorNodeRLayers')
composite=nodes.new('CompositorNodeComposite')
scene.node_tree.links.new(layer.outputs['Image'],composite.inputs[0])
mask_node=nodes.new('CompositorNodeOutputFile')
mask_node.base_path=str(output/'passes')
mask_node.format.file_format='PNG'
mask_node.format.color_mode='BW'
mask_node.format.color_depth='8'
depth_node=nodes.new('CompositorNodeOutputFile')
depth_node.base_path=str(output/'passes')
depth_node.format.file_format='OPEN_EXR'
depth_node.format.color_mode='BW'
depth_node.format.color_depth='32'
scene.node_tree.links.new(layer.outputs['IndexOB'],mask_node.inputs[0])
scene.node_tree.links.new(layer.outputs['Depth'],depth_node.inputs[0])
try:
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.get_devices()
    prefs.compute_device_type='CUDA'
    for d in prefs.devices: d.use=d.type=='CUDA'
    scene.cycles.device='GPU'
    print('CYCLES_DEVICE',[(d.name,d.use) for d in prefs.devices],flush=True)
except Exception as exc:
    print('CYCLES_CPU_FALLBACK',repr(exc),flush=True)
    scene.cycles.device='CPU'

# The rendered ground matches the z plane used by Warp's static collider.
ground_z=float(json.loads((trajectory.parent/'config.json').read_text())['ground_z_m'])
bpy.ops.mesh.primitive_plane_add(size=100, location=(.5,.5,ground_z))
floor=bpy.context.object
floor.name='Physical_ground_visual'
mat=bpy.data.materials.new('matte_studio_floor');mat.diffuse_color=(.16,.20,.24,1)
mat.use_nodes=True
bs=mat.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(.16,.20,.24,1)
bs.inputs['Roughness'].default_value=.72
floor.data.materials.append(mat)
world=bpy.data.worlds.new('synthetic_soft_studio')
scene.world=world;world.use_nodes=True
wn=world.node_tree.nodes;wn.clear()
bg=wn.new('ShaderNodeBackground');wo=wn.new('ShaderNodeOutputWorld')
world.node_tree.links.new(bg.outputs['Background'],wo.inputs['Surface'])
bg.inputs['Color'].default_value=(.35,.42,.50,1)
bg.inputs['Strength'].default_value=.35

def area(name,loc,energy,size):
    ld=bpy.data.lights.new(name,'AREA');ld.energy=energy;ld.shape='DISK';ld.size=size
    o=bpy.data.objects.new(name,ld);scene.collection.objects.link(o);o.location=loc
    o.rotation_euler=(Vector((.5,.5,.25))-o.location).to_track_quat('-Z','Y').to_euler()
area('softbox_key',(.1,-.2,1.1),120,1.0)
area('softbox_fill',(1.2,.4,.8),55,.8)

center=np.mean(P[0,:ns],axis=0)
center[2]=ground_z+.5*(float(P[0,:ns,2].max())-ground_z)
asset_extent=float(np.max(np.ptp(P[0,:ns],axis=0)))
camera_locations=[(.94,-.22,.59),(.07,.12,.53),(.69,1.15,.60)]
cams=[]
for i,loc in enumerate(camera_locations):
    cd=bpy.data.cameras.new(f'cam_{i}');co=bpy.data.objects.new(f'cam_{i}',cd)
    scene.collection.objects.link(co);co.location=loc
    co.rotation_euler=(Vector(center)-co.location).to_track_quat('-Z','Y').to_euler()
    cd.type='PERSP';cd.sensor_fit='HORIZONTAL';cd.sensor_width=36.0
    distance=(co.location-Vector(center)).length
    cd.lens=36.0*distance/max(asset_extent*2.5,.20)
    cams.append(co)

indices=([0] if mode=='debug' else sorted(set([0,len(times)//4,len(times)//2,3*len(times)//4,len(times)-1])) if mode=='pilot' else range(len(times)))
meta=dict(asset=str(asset),trajectory=str(trajectory),fps=int(json.loads((trajectory.parent/'config.json').read_text())['fps']),
          frames=[dict(index=int(i),time_s=float(times[i])) for i in indices],
          cameras=[],render=dict(engine='Cycles',samples=scene.cycles.samples,
                                 resolution=[scene.render.resolution_x,scene.render.resolution_y],
                                 fixed_lighting=True,fixed_background=True,
                                 mesh_motion_source='saved Warp MPM tracked surface particles',
                                 no_additional_deformation=True))
for ci,cam in enumerate(cams):
    scene.camera=cam
    bpy.context.view_layer.update()
    fpx=scene.render.resolution_x*cam.data.lens/cam.data.sensor_width
    K=[[fpx,0,scene.render.resolution_x/2],
       [0,fpx,scene.render.resolution_y/2],[0,0,1]]
    meta['cameras'].append(dict(name=cam.name,fixed=True,c2w=[list(row) for row in cam.matrix_world],
                                projection='perspective',lens_mm=cam.data.lens,
                                sensor_width_mm=cam.data.sensor_width,
                                intrinsic=K))
    frame_dir=output/f'cam_{ci}'
    frame_dir.mkdir(exist_ok=True)
    for fi in indices:
        rgb_path=frame_dir/f'{fi:04d}.png'
        mask_path=output/'passes'/f'cam_{ci}'/f'mask_{fi:04d}_{fi+1:04d}.png'
        depth_path=output/'passes'/f'cam_{ci}'/f'depth_{fi:04d}_{fi+1:04d}.exr'
        if rgb_path.exists() and mask_path.exists() and depth_path.exists():
            print('RESUME_RENDER_SKIP',ci,fi,flush=True)
            continue
        for vi, v in enumerate(obj.data.vertices):
            v.co=obj.matrix_world.inverted() @ Vector(P[fi,vi])
        obj.data.update()
        scene.frame_set(int(fi)+1)
        if fi==0:
            from bpy_extras.object_utils import world_to_camera_view
            direction=cam.matrix_world.to_quaternion() @ Vector((0,0,-1))
            hit=scene.ray_cast(bpy.context.view_layer.depsgraph,cam.location,direction)
            uv=np.asarray([tuple(world_to_camera_view(scene,cam,Vector(q))) for q in P[fi,:ns]])
            print('PROJECTION_DEBUG',ci,tuple(world_to_camera_view(scene,cam,Vector(P[fi,:ns].mean(axis=0)))),
                  'camera',tuple(cam.location),'object',tuple(obj.location),
                  'ray_hit',hit[4].name if hit[0] else None,
                  'uv_bounds',uv.min(0),uv.max(0),flush=True)
        mask_node.file_slots[0].path=f'cam_{ci}/mask_{fi:04d}_'
        depth_node.file_slots[0].path=f'cam_{ci}/depth_{fi:04d}_'
        scene.render.filepath=str(rgb_path)
        bpy.ops.render.render(write_still=True)
        print('RENDERED',ci,fi,flush=True)
(output/'cameras.json').write_text(json.dumps(meta,indent=2))
