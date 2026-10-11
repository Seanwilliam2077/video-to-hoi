"""Build an agent-authored observational asset through a generic Blender backend."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import av
import numpy as np
import trimesh
from v2hoi.clips import load_clips
from v2hoi.native import sha256
from v2hoi.object_refine import face_observation_targets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--episode',type=int,required=True)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--frames',type=int,nargs='+',required=True)
    p.add_argument('--object-instance-id');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blender',type=Path,required=True);a=p.parse_args()
    clip=load_clips('track1',[a.episode],a.dataset_root)[0]
    if not a.frames or any(i<0 or i>=clip.n_frames for i in a.frames): raise ValueError('Invalid original observation frame IDs')
    a.output.mkdir(parents=True,exist_ok=False)
    observations=[]
    with av.open(str(clip.video)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i in a.frames:
                path=a.output/f'observation-{i:06d}.png';frame.to_image().save(path)
                observations.append({'original_frame_id':i,'path':path.name,'sha256':sha256(path)})
    script=Path(__file__).with_name('build_bpy_asset.py')
    mesh=a.output/'object.glb'
    with (a.output/'blender.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(a.blender),'--background','--factory-startup','--python-exit-code','1','--python',str(script.resolve()),'--',
            '--spec',str(a.spec.resolve()),'--output',str(mesh.resolve())],stdout=log,stderr=subprocess.STDOUT,check=True)
    # Blender's glTF can carry both COLOR_0 and materials; the native tracker
    # expects explicit vertex colors. Normalize without changing the surface.
    with np.load(mesh.with_suffix('.npz')) as arrays:
        geometry=trimesh.Trimesh(vertices=arrays['vertices'],faces=arrays['faces'],
            vertex_colors=arrays['colors'],process=False)
    geometry.export(mesh)
    readback=trimesh.load(mesh,force='mesh',process=False)
    if (not np.allclose(readback.vertices,geometry.vertices,atol=1e-6)
            or not np.array_equal(readback.faces,geometry.faces)
            or not np.array_equal(readback.visual.vertex_colors,geometry.visual.vertex_colors)):
        raise ValueError('Blender asset serialization changed geometry or observed colors')
    if not geometry.is_watertight or len(geometry.faces)>4096 or not np.isfinite(geometry.vertices).all():
        raise ValueError('Invalid closed low-budget asset surface')
    spec_copy=a.output/'spec.json';spec_copy.write_bytes(a.spec.read_bytes())
    manifest={'schema':'v2hoi.observed-asset.v1','provenance':'track1','episode':clip.episode,
        'object_instance_id':a.object_instance_id or clip.object_name,'video_sha256':sha256(clip.video),
        'mesh':{'path':mesh.name,'sha256':sha256(mesh)},'entry':'blender.bpy.observed-profile',
        'observations':observations,'spec_sha256':sha256(spec_copy),'builder_sha256':sha256(script),
        'triangles':len(geometry.faces),'vertices':len(geometry.vertices),'watertight':bool(geometry.is_watertight),
        'metric_scale_verified':False,'training':False,'reference_geometry_used':False,
        'scope':'Agent-authored proportions from permitted RGB observations; shared metric scale must be calibrated',
        'limitations':['Supported parametric families only; concealed geometry is an explicit shape prior',
            'Image fit does not establish true physical dimensions or reconstruction accuracy']}
    spec=json.loads(a.spec.read_text())
    if spec.get('face_observations'):
        manifest.update(semantic_normal=spec['semantic_normal'],face_observations=spec['face_observations'],
            semantic_observation_scope='Agent-authored visible face labels; not reference pose or motion')
        face_observation_targets(manifest,clip.n_frames)
    (a.output/'asset.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
