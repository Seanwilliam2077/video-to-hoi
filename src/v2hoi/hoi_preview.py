"""Render complete human-object inference in one verified metric scene.

This decodes the existing native MHR inference parameters. It never invokes the
submission converter and never substitutes a cached human mesh for parameters.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from v2hoi.native import sha256
from v2hoi.object_refine import require_timeline


def render_full_hoi(run,clip,candidate_path,output,codec='auto',batch_size=4):
    import av,cv2,torch,trimesh
    from lib_mhr import MHRLayer,MHR_PARAM_DIMS
    from lib_mhr.human_texture import batched_vertex_normals
    from lib_mhr.object_pose_frame import resolve_object_pose_frame,object_poses_to_training_frame
    from prep.mhr_export_utils import camera_calibration,load_edex
    from tools.mhr_mesh_renderer import NvdiffMeshRenderer
    root=run/'native'/f'episode_{clip.episode:06d}'
    export=root/'export'/f'episode_{clip.episode:06d}'
    source=root/'inference/refined.pth'
    bundles=[torch.load(path,map_location='cpu',weights_only=False) for path in (source,candidate_path)]
    mesh_paths=[];poses=[];inputs={str(p.resolve()):sha256(p) for p in (source,candidate_path,clip.video)}
    for label,bundle in zip(('before','after'),bundles):
        require_timeline([int(row['src_frame']) for row in bundle['frame_meta']],clip.n_frames)
        if bundle.get('gt') or bundle['metadata'].get('ground_truth_used') is not False:
            raise ValueError('Full HOI rendering accepts reference-free native inference only')
        path=Path(bundle['metadata']['object_mesh']);inputs[str(path.resolve())]=sha256(path)
        frame=resolve_object_pose_frame(bundle['metadata'])
        geometry=trimesh.load(path,force='mesh',process=False)
        geometry.apply_transform(frame.mesh_to_training)
        materialized=output/f'preview-{label}-mesh.glb';geometry.export(materialized);mesh_paths.append(materialized)
        poses.append(object_poses_to_training_frame(bundle['pr']['pose_abs'],frame.storage_to_training))
    human_keys=[key for key in bundles[0]['pr'] if key.startswith('mhr_')]
    identical=all(np.array_equal(np.asarray(bundles[0]['pr'][key]),np.asarray(bundles[1]['pr'][key])) for key in human_keys)
    if not identical:raise ValueError('Object repair must preserve the native human parameters')
    k,extrinsic=camera_calibration(load_edex(export),0)
    if not np.allclose(extrinsic,np.eye(4),atol=1e-6):raise ValueError('Expected a shared camera-space scene')
    layer=MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'],device='cuda')
    decoder=layer.decoder_identity()
    faces=layer.mesh_faces(device='cpu').numpy().astype(np.int32)
    renderer=NvdiffMeshRenderer(device='cuda')
    width,height=clip.width//2,clip.height//2
    width-=width%2;height-=height%2
    scale_x,scale_y=width/clip.width,height/clip.height
    intrinsics=np.asarray(k,dtype=np.float32).copy();intrinsics[0]*=scale_x;intrinsics[1]*=scale_y
    ffmpeg=shutil.which('ffmpeg') or os.environ.get('IMAGEIO_FFMPEG_EXE')
    if not ffmpeg:raise FileNotFoundError('FFmpeg required for the full-scene preview')
    if codec=='auto':
        probe=subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-f','lavfi','-i','color=size=32x32:rate=1',
            '-frames:v','1','-c:v','h264_nvenc','-f','null','-'],capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        codec='h264_nvenc' if probe.returncode==0 else 'libx264'
    options=['-preset','p4','-cq','18','-b:v','0'] if codec=='h264_nvenc' else ['-preset','fast','-crf','18']
    processes=[];logs=[]
    for name in ('full-hoi.mp4','full-hoi-comparison.mp4'):
        log=(output/(name+'.log')).open('w');logs.append(log)
        processes.append(subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{width*3}x{height+40}','-r',str(clip.fps),'-i','pipe:0','-an','-c:v',codec,*options,
            '-pix_fmt','yuv420p','-movflags','+faststart',str(output/name)],stdin=subprocess.PIPE,stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0))
    def panel(image,title):
        bar=np.full((40,width,3),30,dtype=np.uint8)
        cv2.putText(bar,title,(8,27),cv2.FONT_HERSHEY_SIMPLEX,.62,(245,245,245),1,cv2.LINE_AA)
        return np.concatenate([bar,image])
    def overlay(rgb,rendered,mask):
        result=rgb.copy();result[mask]=np.rint(rgb[mask]*.28+rendered[mask]*.72).clip(0,255).astype('uint8');return result
    count=0
    try:
        with av.open(str(clip.video)) as container:
            frames=iter(container.decode(video=0))
            for start in range(0,clip.n_frames,batch_size):
                stop=min(start+batch_size,clip.n_frames);indices=slice(start,stop)
                renders=[];masks=[]
                for bundle,mesh,trajectory in zip(bundles,mesh_paths,poses):
                    with torch.inference_mode():
                        params={key:torch.as_tensor(np.asarray(bundle['pr'][key][indices]),dtype=torch.float32,device='cuda') for key in MHR_PARAM_DIMS}
                        vertices=layer.mhr_forward(params).vertices.detach().cpu().numpy().astype('float32')
                    normal=batched_vertex_normals(vertices,faces)
                    color,mask=renderer.render_front_batch_constant_human_textured_object(vertices,faces,normal,str(mesh.resolve()),
                        trajectory[indices],np.repeat(intrinsics[None],stop-start,axis=0),(height,width),(.78,.82,.85))
                    renders.append(np.rint(color.clip(0,1)*255).astype('uint8'));masks.append(mask)
                for j,i in enumerate(range(start,stop)):
                    rgb=cv2.resize(next(frames).to_ndarray(format='rgb24'),(width,height))
                    surface=np.full_like(rgb,(32,36,40));surface[masks[1][j]]=renders[1][j][masks[1][j]]
                    complete=np.hstack([panel(rgb,f'Original frame {i}'),panel(surface,'Complete native HOI'),
                        panel(overlay(rgb,renders[1][j],masks[1][j]),'Human + object overlay')])
                    comparison=np.hstack([panel(rgb,f'Original frame {i}'),panel(overlay(rgb,renders[0][j],masks[0][j]),'Previous full HOI'),
                        panel(overlay(rgb,renders[1][j],masks[1][j]),'Updated full HOI')])
                    for process,image in zip(processes,(complete,comparison)):process.stdin.write(image.tobytes())
                    if i%max(1,clip.n_frames//8)==0 or i==clip.n_frames-1:
                        cv2.imwrite(str(output/f'full-hoi-{i:06d}.png'),cv2.cvtColor(complete,cv2.COLOR_RGB2BGR))
                    count+=1
                if stop%60==0 or stop==clip.n_frames:print(f'FULL_HOI_RENDER episode={clip.episode} frame={stop}/{clip.n_frames}',flush=True)
            try:next(frames);raise ValueError('Source has additional unrepresented frames')
            except StopIteration:pass
    finally:
        for process,log in zip(processes,logs):process.stdin.close();process.wait();log.close()
    if count!=clip.n_frames or any(p.returncode for p in processes):raise ValueError('Incomplete full-scene render')
    for name in ('full-hoi.mp4','full-hoi-comparison.mp4'):
        with av.open(str(output/name)) as container:
            if sum(1 for _ in container.decode(video=0))!=clip.n_frames:raise ValueError('HOI video lost original frames')
    report={'schema':'v2hoi.full-hoi-preview.v1','provenance':'track1','episode':clip.episode,'frames':count,
        'original_frame_ids':list(range(count)),'decoder':decoder,'inputs':inputs,'human_arrays_identical':identical,
        'human_and_objects_share_scene':True,'renderer':'Native full-scene renderer with human/object depth compositing',
        'submission_conversion':False,'reference_used':False,'preview_codec':codec,
        'outputs':{name:sha256(output/name) for name in ('full-hoi.mp4','full-hoi-comparison.mp4')},
        'limitations':['Full inference scene is not an official submission payload','Frozen human inference retains its errors; hand contact must be validated']}
    (output/'full-hoi-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
