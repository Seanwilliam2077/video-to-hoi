"""Build a compound box asset in Blender from an observation-authored profile."""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    spec = json.loads(args.spec.read_text())
    if args.output.exists():
        raise FileExistsError('Do not overwrite an authored asset')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    objects = []
    for box in spec['boxes']:
        center, size = np.asarray(box['center']), np.asarray(box['size'])
        if center.shape != (3,) or size.shape != (3,) or not np.isfinite([center, size]).all() or (size <= 0).any():
            raise ValueError('Invalid compound-box component')
        bpy.ops.mesh.primitive_cube_add(size=1., location=center)
        obj = bpy.context.object
        obj.name = box['name']
        obj.dimensions = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bevel = obj.modifiers.new('ObservedRoundedEdges', 'BEVEL')
        bevel.width = float(box.get('bevel', .002))
        bevel.segments = 2
        bpy.ops.object.modifier_apply(modifier=bevel.name)
        attribute = obj.data.color_attributes.new(name='Color', type='BYTE_COLOR', domain='CORNER')
        for value in attribute.data:
            value.color = (*box.get('color', [.43, .28, .12]), 1.)
        objects.append(obj)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    obj = objects[0]
    bpy.context.scene.cursor.location = (0, 0, 0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    modifier = obj.modifiers.new('Triangles', 'TRIANGULATE')
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    colors = np.zeros((len(obj.data.vertices), 4), dtype=np.uint8)
    for face in obj.data.polygons:
        for loop in face.loop_indices:
            colors[obj.data.loops[loop].vertex_index] = np.rint(np.array(obj.data.color_attributes['Color'].data[loop].color) * 255).astype('uint8')
    if len(obj.data.polygons) > 4096:
        raise ValueError('Asset exceeds the face budget')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix('.npz'), vertices=np.array([v.co[:] for v in obj.data.vertices]),
             faces=np.array([p.vertices[:] for p in obj.data.polygons]), colors=colors)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.with_suffix('.blend')))


if __name__ == '__main__':
    main()
