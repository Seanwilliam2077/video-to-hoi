"""Blender-only parametric surface construction, independent of video IDs.

Run through Blender --background --factory-startup --python ... -- --spec ...
--output ... . JSON profiles describe observations, not reference geometry.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector


def surface(name, vertices, faces, color):
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata(vertices,[],faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
    attribute=mesh.color_attributes.new(name="Color",type="BYTE_COLOR",domain="CORNER")
    for value in attribute.data: value.color=(*color,1.)
    material=bpy.data.materials.new(name+"Material");material.diffuse_color=(*color,1.)
    material.use_nodes=True
    node=material.node_tree.nodes.get("Principled BSDF")
    vertex=material.node_tree.nodes.new("ShaderNodeVertexColor");vertex.layer_name="Color"
    material.node_tree.links.new(vertex.outputs['Color'],node.inputs['Base Color'])
    node.inputs['Roughness'].default_value=.65
    mesh.materials.append(material)
    for face in mesh.polygons: face.use_smooth=True
    return obj


def revolve(spec):
    n=spec.get('segments',48)
    if not isinstance(n,int) or not 12<=n<=96: raise ValueError('Require 12..96 radial segments')
    profile=spec['profile']
    if len(profile)<4 or any(len(p)!=2 or not all(math.isfinite(x) for x in p) or p[0]<0 for p in profile):
        raise ValueError('Invalid finite radius/height profile')
    vertices=[];rings=[];faces=[]
    for index,(radius,height) in enumerate(profile):
        if index==len(profile)-1 and profile[index]==profile[0]:
            rings.append(rings[0]);continue
        ids=[]
        for j in range(n if radius>0 else 1):
            angle=2*math.pi*j/n;ids.append(len(vertices));vertices.append((radius*math.cos(angle),radius*math.sin(angle),height))
        rings.append(ids)
    for a,b in zip(rings[:-1],rings[1:]):
        for j in range(n):
            q=(j+1)%n
            if len(a)==1: faces.append((a[0],b[j],b[q]))
            elif len(b)==1: faces.append((a[j],b[0],a[q]))
            else: faces.append((a[j],b[j],b[q],a[q]))
    return surface('ObservedRevolvedSurface',vertices,faces,spec.get('color',[.055,.06,.065]))


def extrusion(spec):
    outline=spec['outline'];z=spec['z'];thickness=spec['thickness'];n=len(outline)
    if n<3 or thickness<=0 or not math.isfinite(thickness): raise ValueError('Invalid extrusion')
    vertices=[(x,y,z+dz) for dz in (-thickness/2,thickness/2) for x,y in outline]
    faces=[tuple(reversed(range(n))),tuple(range(n,2*n))]
    faces.extend((j,(j+1)%n,(j+1)%n+n,j+n) for j in range(n))
    obj=surface('ObservedExtrusion',vertices,faces,spec.get('color',[.025,.028,.03]))
    bevel=obj.modifiers.new('ObservedEdgeRoundover','BEVEL');bevel.width=spec.get('bevel',.002);bevel.segments=2
    bpy.context.view_layer.objects.active=obj;obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=bevel.name);obj.select_set(False)
    return obj


def image_box(spec):
    """Closed subdivided box with colors sampled from verified observation quads."""
    extents=np.asarray(spec['extents'],dtype=float)
    n=spec.get('grid',12)
    if extents.shape!=(3,) or not np.isfinite(extents).all() or (extents<=0).any() or not 1<=n<=14:
        raise ValueError('Invalid observed box dimensions or grid')
    x,y,z=extents/2
    planes={'+Z':[(-x,-y,z),(x,-y,z),(x,y,z),(-x,y,z)],'-Z':[(-x,y,-z),(x,y,-z),(x,-y,-z),(-x,-y,-z)],
        '-Y':[(-x,-y,-z),(x,-y,-z),(x,-y,z),(-x,-y,z)],'+X':[(x,-y,-z),(x,y,-z),(x,y,z),(x,-y,z)],
        '+Y':[(x,y,-z),(-x,y,-z),(-x,y,z),(x,y,z)],'-X':[(-x,y,-z),(-x,-y,-z),(-x,-y,z),(-x,y,z)]}
    vertices=[];faces=[];colors=[];counts=[];index={}
    for name,corners in planes.items():
        appearance=spec.get('faces',{}).get(name,{})
        image=None;homography=None
        if 'image' in appearance:
            image=bpy.data.images.load(appearance['image']);image.colorspace_settings.name='Non-Color'
            w,h=image.size;pixels=np.asarray(image.pixels[:],dtype=float).reshape(h,w,4)[::-1]
            quad=appearance['quad'];matrix=[];rhs=[]
            for (u,v),(px,py) in zip([(0,0),(1,0),(1,1),(0,1)],quad):
                matrix.extend([[u,v,1,0,0,0,-px*u,-px*v],[0,0,0,u,v,1,-py*u,-py*v]]);rhs.extend([px,py])
            homography=np.r_[np.linalg.solve(matrix,rhs),1].reshape(3,3)
        grid={};a,b,c,d=map(np.asarray,corners)
        for j in range(n+1):
            for i in range(n+1):
                u,v=i/n,j/n;point=(1-v)*((1-u)*a+u*b)+v*((1-u)*d+u*c)
                key=tuple(np.round(point,9));color=appearance.get('color',spec.get('color',[.5,.33,.15]))
                if image is not None:
                    mapped=homography@np.array([u,v,1]);px,py=mapped[:2]/mapped[2]
                    color=pixels[int(np.clip(round(py),0,h-1)),int(np.clip(round(px),0,w-1)),:3]
                if key not in index:
                    index[key]=len(vertices);vertices.append(point.tolist());colors.append(np.zeros(3));counts.append(0)
                vertex=index[key];grid[(i,j)]=vertex;colors[vertex]+=np.asarray(color);counts[vertex]+=1
        for j in range(n):
            for i in range(n):faces.append((grid[i,j],grid[i+1,j],grid[i+1,j+1],grid[i,j+1]))
    obj=surface('ObservedImageBox',vertices,faces,[.5,.33,.15])
    colors=np.asarray(colors)/np.asarray(counts)[:,None]
    for face in obj.data.polygons:
        face.use_smooth=False
        for loop in face.loop_indices:
            obj.data.color_attributes['Color'].data[loop].color=(*colors[obj.data.loops[loop].vertex_index],1.)
    return obj


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);spec=json.loads(a.spec.read_text())
    if a.output.exists(): raise FileExistsError('Preserve existing assets')
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    objects=[]
    for component in spec['components']:
        if component['kind']=='revolve': objects.append(revolve(component))
        elif component['kind']=='extrusion': objects.append(extrusion(component))
        elif component['kind']=='image_box': objects.append(image_box(component))
        else: raise ValueError('Unsupported parametric surface family')
    for obj in objects: obj.select_set(True)
    bpy.context.view_layer.objects.active=objects[0];bpy.ops.object.join()
    obj=objects[0];obj.name='ObservationAsset'
    # Triangulation is deterministic and part of the asset, before tracking.
    modifier=obj.modifiers.new('ExplicitTriangles','TRIANGULATE');bpy.ops.object.modifier_apply(modifier=modifier.name)
    if len(obj.data.polygons)>4096: raise ValueError('Asset exceeds the 4096 triangle budget')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    colors=np.zeros((len(obj.data.vertices),4),dtype=np.uint8)
    attribute=obj.data.color_attributes['Color']
    for face in obj.data.polygons:
        for loop in face.loop_indices:
            vertex=obj.data.loops[loop].vertex_index
            colors[vertex]=np.round(np.array(attribute.data[loop].color)*255).clip(0,255)
    np.savez(a.output.with_suffix('.npz'),vertices=np.array([v.co[:] for v in obj.data.vertices]),
        faces=np.array([face.vertices[:] for face in obj.data.polygons]),colors=colors)
    bpy.ops.export_scene.gltf(filepath=str(a.output),export_format='GLB',use_selection=True,
        export_yup=False,export_apply=True,export_materials='EXPORT')
    bpy.ops.wm.save_as_mainfile(filepath=str(a.output.with_suffix('.blend')))
    # A derived neutral inspection image; source-video comparisons are separate.
    bpy.ops.object.camera_add(location=(.6,-.65,.48));camera=bpy.context.object
    camera.rotation_euler=(Vector((.06,0,.02))-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.type='ORTHO';camera.data.ortho_scale=.65;bpy.context.scene.camera=camera
    scene=bpy.context.scene;scene.render.engine='BLENDER_WORKBENCH';scene.render.resolution_x=1000;scene.render.resolution_y=800;scene.render.resolution_percentage=100
    scene.display.shading.light='STUDIO';scene.display.shading.color_type='VERTEX'
    scene.display.shading.show_shadows=True;scene.display.shading.show_cavity=True
    scene.display.shading.background_type='WORLD';scene.world.color=(.18,.18,.18)
    scene.render.filepath=str(a.output.with_suffix('.png'));bpy.ops.render.render(write_still=True)
    print('BPY_ASSET',json.dumps({'triangles':len(obj.data.polygons),'vertices':len(obj.data.vertices),'blender':bpy.app.version_string}))


if __name__=='__main__': main()
