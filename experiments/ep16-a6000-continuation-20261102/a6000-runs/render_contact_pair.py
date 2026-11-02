"""Full360 native paired scene with fixed object and hand/body collision diagnostics."""
from pathlib import Path
import argparse,os,json,time,hashlib
import cv2,numpy as np,torch,imageio.v2 as imageio
from lib_mhr.human_texture import batched_vertex_normals
from tools.mhr_mesh_renderer import NvdiffMeshRenderer
from tools.render_mhr_wild_inference import HUMAN_ALBEDO
ap=argparse.ArgumentParser();ap.add_argument('--stage',default='v1');args=ap.parse_args();R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');D=R/'a6000-runs/hand-contact-20261102-v1';stage=args.stage;O=D/f'full-scene-{stage}';O.mkdir(exist_ok=False);p0=R/'a6000-runs/direction-correction-v1/source-material-candidate-v6/refined.pth';p1=D/f'native-contact-candidate-{stage}/refined.pth';b0=torch.load(p0,map_location='cpu',weights_only=False);b1=torch.load(p1,map_location='cpu',weights_only=False);g0=np.load(D/'input-geometry.npz');g1=np.load(D/f'native-contact-candidate-{stage}-evaluation/input-geometry.npz');V0=g0['vertices'];V1=g1['vertices'];F=g0['human_faces'].astype(np.int32);hids=g0['hand_vertex_ids'];lo=g0['object_vertices'].min(0);hi=g0['object_vertices'].max(0);T=np.asarray(b0['pr']['pose_abs']);K=np.asarray(b0['observations']['K_full']);changed=[k for k,v in b0['pr'].items() if not np.array_equal(v,b1['pr'][k],equal_nan=True)];assert set(changed)=={'mhr_body_pose_cont','mhr_hand'};assert b0['frames']==b1['frames'];assert np.array_equal(T,b1['pr']['pose_abs']);assert np.array_equal(T,b1['pr']['pose_abs_postopt']);assert np.allclose(V0[:51],V1[:51],atol=1e-6,rtol=0) and np.allclose(V0[290:],V1[290:],atol=1e-6,rtol=0);assert np.isfinite(V1).all();mesh=b0['metadata']['object_mesh'];renderer=NvdiffMeshRenderer(device='cuda');torch.set_num_threads(4);cv2.setNumThreads(2);start=time.time();cap=cv2.VideoCapture(str(R/'runs/full-episode16-20261030/inputs/episode_000016.0.color.mp4'));writer=imageio.get_writer(str(O/'source-old-new-full-scene-360-30fps.mp4'),format='FFMPEG',fps=30,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_params=['-crf','18','-threads','2']);zoom=imageio.get_writer(str(O/'source-old-new-hand-zoom-360-30fps.mp4'),format='FFMPEG',fps=30,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_params=['-crf','18','-threads','2']);rows=[];selected={0,75,96,100,115,128,129,130,139,150,165,173,174,175,194,205,215,220,240,260,275,300,359};samples={}
def exact_box_sdf(points,pose):
 local=(points-pose[:3,3])@pose[:3,:3];q=abs(local-(lo+hi)/2)-(hi-lo)/2;return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
for first in range(0,360,4):
 end=min(first+4,360);outputs=[]
 for VV in [V0,V1]:
  verts=VV[first:end].astype(np.float32);normal=batched_vertex_normals(verts,F)
  with torch.no_grad():rgb,mask=renderer.render_front_batch_constant_human_textured_object(verts,F,normal,str(mesh),T[first:end].astype(np.float32),K[first:end].astype(np.float32),(1152,1536),HUMAN_ALBEDO)
  outputs.append((rgb,mask))
 for j,f in enumerate(range(first,end)):
  ok,im=cap.read();assert ok;source=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);panels=[source]
  for rgb,mask in outputs:
   im=np.full_like(source,(32,36,40));im[mask[j]]=np.rint(np.clip(rgb[j][mask[j]],0,1)*255).astype(np.uint8);panels.append(im)
  full=[];crops=[]
  for im,label in zip(panels,['Original source','Correct turn | old hands','Fixed object | native hand candidate']):
   x=cv2.resize(im,(768,576));cv2.rectangle(x,(0,0),(768,28),(0,0,0),-1);cv2.putText(x,f'{label} | frame{f}',(5,20),0,.45,(255,255,255),1);full.append(x);x=cv2.resize(im[460:810,550:900],(560,560));bar=np.zeros((30,560,3),np.uint8);cv2.putText(bar,f'{label} | frame{f}',(5,21),0,.48,(255,255,255),1);crops.append(np.concatenate([bar,x]))
  writer.append_data(np.concatenate(full,1));zoom.append_data(np.concatenate(crops,1))
  if f in selected:
   cv2.imwrite(str(O/f'crop-{f:03}.jpg'),cv2.cvtColor(np.concatenate(crops,1),cv2.COLOR_RGB2BGR));samples[f]=np.concatenate([cv2.resize(x,(512,384)) for x in panels],1)
  row={'frame':f}
  for name,VV,gg in [('before',V0,g0),('after',V1,g1)]:
   s=exact_box_sdf(VV[f,hids],T[f]);pen=np.maximum(-s,0)*1000;nearest=np.sort(abs(s),axis=1)[:,:32].mean(1)*1000;body_s=exact_box_sdf(VV[f],T[f]);row[name]={'hand_max_penetration_mm':pen.max(1).tolist(),'hand_mean_penetration_mm':pen.mean(1).tolist(),'hand_inside_fraction':(pen>.1).mean(1).tolist(),'nearest32_hand_surface_mean_distance_mm':nearest.tolist(),'wholehuman_vertex_max_penetration_mm':float(np.maximum(-body_s,0).max()*1000)}
  uv=[]
  for gg in [g0,g1]:
   p=gg['keypoints'][f,[7,8,62,41]]@K[f].T;uv.append(p[:,:2]/p[:,2:])
  row['elbow_wrist_projection_shift_from_input_px']=np.linalg.norm(uv[1]-uv[0],axis=1).tolist();rows.append(row)
 if end%40==0:print('Rendered',end,'/360',flush=True)
writer.close();zoom.close();cap.release();check=cv2.VideoCapture(str(O/'source-old-new-full-scene-360-30fps.mp4'));n=0
while check.read()[0]:n+=1
fps=check.get(cv2.CAP_PROP_FPS);check.release();assert n==360 and fps==30
summary={}
for name in ['before','after']:
 selectedrows=[r[name] for r in rows if 70<=r['frame']<=260];summary[name]={'all360_hand_max_penetration_mm':max(max(r[name]['hand_max_penetration_mm']) for r in rows),'all360_hand_mean_penetration_mm':float(np.mean([r[name]['hand_mean_penetration_mm'] for r in rows])),'grasp70_260_nearest32_mean_distance_mm':np.mean([r['nearest32_hand_surface_mean_distance_mm'] for r in selectedrows],0).tolist(),'all360_wholehuman_vertex_max_penetration_mm':max(r[name]['wholehuman_vertex_max_penetration_mm'] for r in rows),'frames_hand_penetration_gt1mm':sum(max(r[name]['hand_max_penetration_mm'])>1 for r in rows)}
report={'status':'full360_paired_scene_and_native_geometry_review_ready','input':str(p0),'candidate':str(p1),'changed_pr_fields':changed,'object_rotation_and_translation_all360_exact':True,'human_root_camera_shape_scale_frozen':True,'frame0_and_outside50_290_geometry_max_difference_below_1micrometer':True,'decoded_frames':n,'fps':fps,'diagnostics':summary,'diagnostic_scope':'Analytic box SDF matches actual authored watertight convex mesh with exactly6axis planes and matching volume. Hand vertices and whole-human vertices measured; not official PEN or reference3D accuracy. Surface-triangle crossing not excluded by vertex-only test.','projection_shift_prior_scope':'Input projected joints, not observed keypoint ground truth.','seconds':time.time()-start,'accepted':False};(O/'report.json').write_text(json.dumps(report,indent=2));(O/'by-frame.json').write_text(json.dumps(rows,indent=2));cv2.imwrite(str(O/'full-scene-contact-sheet.jpg'),cv2.cvtColor(np.concatenate([samples[f] for f in [0,115,130,175,220,300]],0),cv2.COLOR_RGB2BGR));print(json.dumps(report),flush=True)
