"""Full-timeline source/baseline/candidate scene and independent mask/proxy diagnostics."""
from pathlib import Path
import os,time,json,hashlib,traceback,subprocess
import numpy as np,cv2,torch,imageio.v2 as imageio
from lib_mhr import MHRLayer
from lib_mhr.human_texture import batched_vertex_normals
from lib_mhr.object_texture import load_object_mesh_parts,concatenate_object_mesh_parts
from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
from learning.training.mhr_opt_refineout import DEFAULT_HAND_SURFACE_SPEC
from tools.mhr_mesh_renderer import NvdiffMeshRenderer
from tools.render_mhr_wild_inference import _params,HUMAN_ALBEDO
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');Q=R/'a6000-runs/matching-optimization-v2';O=R/'a6000-runs/direction-correction-v1/full-scene-review-v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(O/n).write_text(json.dumps(d,indent=2))
def silhouette(V,F,T,K):
 cam=V@T[:3,:3].T+T[:3,3];assert (cam[:,2]>.001).all();p=cam@K.T;p=np.rint(p[:,:2]/p[:,2:]).astype(np.int32);mask=np.zeros((256,256),np.uint8)
 for f in F:cv2.fillConvexPoly(mask,p[f],1)
 return mask.astype(bool)
def iou(a,b):return float((a&b).sum()/max((a|b).sum(),1))
def box_proxy(hands,T,bounds):
 points=(hands-T[:3,3])@T[:3,:3];lo,hi=bounds;return np.maximum(-(abs(points-(lo+hi)/2)-(hi-lo)/2).max(-1),0)*1000

def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='1';O.mkdir(exist_ok=False);start=time.time();write('state.json',{'status':'loading_complete_native_pair','pid':os.getpid()})
 p=R/'experiments/ep16-joint-direction-20261031/joint-green-v1/refined.pth';c=R/'a6000-runs/direction-correction-v1/source-face-candidate-v3/refined.pth';base=torch.load(p,map_location='cpu',weights_only=False);cand=torch.load(c,map_location='cpu',weights_only=False)
 assert base['frames']==cand['frames'] and list(map(int,base['frames']))==list(range(360));changed=[k for k,v in base['pr'].items() if not np.array_equal(v,cand['pr'][k],equal_nan=True)];assert set(changed)<={'pose_abs','pose_abs_postopt'} and changed
 assert np.array_equal(base['observations']['K_full'],cand['observations']['K_full']);T0=np.asarray(base['pr']['pose_abs']);T1=np.asarray(cand['pr']['pose_abs']);assert np.array_equal(T0[:61],T1[:61]);meshpath=Path(base['metadata']['object_mesh']);assert meshpath==Path(cand['metadata']['object_mesh']);mesh=concatenate_object_mesh_parts(load_object_mesh_parts(meshpath));V=np.asarray(mesh.vertices);F=np.asarray(mesh.faces);K=np.asarray(base['observations']['K_full']);obs=base['observations'];bounds=mesh.bounds
 write('identity-contract.json',{'baseline_sha256':sha(p),'candidate_sha256':sha(c),'mesh_sha256':sha(meshpath),'frames':360,'changed_pr_fields':changed,'human_camera_mesh_scale_frozen':True,'frames_0_60_exactly_unchanged':True,'source_frame_ids':list(range(360))})
 cv2.setNumThreads(2);torch.set_num_threads(4);layer=MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'],device='cuda');faces=layer.mesh_faces(device='cuda').cpu().numpy().astype(np.int32);spec=load_mhr_hand_surface_spec(DEFAULT_HAND_SURFACE_SPEC,faces=faces);renderer=NvdiffMeshRenderer(device='cuda')
 source=R/'runs/full-episode16-20261030/inputs/episode_000016.0.color.mp4';cap=cv2.VideoCapture(str(source));assert int(cap.get(7))==360 and cap.get(5)==30
 writer=imageio.get_writer(str(O/'source-baseline-candidate-360-30fps.mp4'),format='FFMPEG',fps=30,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_params=['-crf','18','-threads','2'])
 frameset={0,90,100,110,120,130,140,150,155,160,165,170,175,180,185,190,195,200,205,210,300,359};sampleimages={};rows=[];proxies=[[],[]]
 for lo in range(0,360,4):
  hi=min(lo+4,360)
  with torch.no_grad():verts=layer.mhr_forward(_params(base['pr'],slice(lo,hi),'cuda')).vertices.cpu().numpy().astype(np.float32)
  normal=batched_vertex_normals(verts,faces);outputs=[]
  for T in [T0,T1]:
   with torch.no_grad():rgb,mask=renderer.render_front_batch_constant_human_textured_object(verts,faces,normal,str(meshpath),T[lo:hi].astype(np.float32),K[lo:hi].astype(np.float32),(1152,1536),HUMAN_ALBEDO)
   outputs.append((rgb,mask))
  for j,t in enumerate(range(lo,hi)):
   ok,im=cap.read();assert ok,t;im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);raw=[im]
   for rgb,mask in outputs:
    surface=np.full_like(im,(32,36,40));surface[mask[j]]=np.rint(np.clip(rgb[j][mask[j]],0,1)*255).astype(np.uint8);raw.append(surface)
   panels=[]
   for img,title in zip(raw,['Source RGB','Baseline intact pair','Source-face direction candidate | UNACCEPTED']):
    panel=cv2.resize(img,(768,576));cv2.rectangle(panel,(0,0),(768,30),(0,0,0),-1);cv2.putText(panel,f'{title} | original frame {t}',(5,21),0,.47,(255,255,255),1);panels.append(panel)
   writer.append_data(np.concatenate(panels,1))
   if t in frameset:
    full=np.concatenate([cv2.resize(x,(512,384)) for x in raw],1);crops=[]
    for x,title in zip(raw,['Source RGB','Baseline','Candidate - unaccepted']):
     crop=cv2.resize(x[490:790,590:890],(450,450));bar=np.zeros((26,450,3),np.uint8);cv2.putText(bar,f'{title}|frame {t}',(5,19),0,.48,(255,255,255),1);crops.append(np.concatenate([bar,crop]))
    cv2.imwrite(str(O/f'crop-{t:06}.jpg'),cv2.cvtColor(np.concatenate(crops,1),cv2.COLOR_RGB2BGR));sampleimages[t]=full
   if 150<=t<=210:
    target=np.asarray(obs['postopt_object_mask'][t])>0;ms=[silhouette(V,F,T[t],obs['postopt_K_rois'][t]) for T in [T0,T1]];row={'frame':t,'temporal_check':bool(t%2),'baseline_unoccluded_mask_iou':iou(ms[0],target),'candidate_unoccluded_mask_iou':iou(ms[1],target)};rows.append(row)
    hands=verts[j][spec.vertex_indices]
    for k,T in enumerate([T0,T1]):proxies[k].append(box_proxy(hands,T[t],bounds))
  write('state.json',{'status':'rendering_full_scene','frames_completed':hi,'seconds':time.time()-start});print('Rendered',hi,'/360',flush=True)
 writer.close();cap.release();check=cv2.VideoCapture(str(O/'source-baseline-candidate-360-30fps.mp4'));n=0
 while check.read()[0]:n+=1
 fps=check.get(5);check.release();assert n==360 and fps==30
 cv2.imwrite(str(O/'full-scene-contact-sheet.jpg'),cv2.cvtColor(np.concatenate([sampleimages[t] for t in [100,120,150,165,175,210]],0),cv2.COLOR_RGB2BGR))
 data={}
 for mode,selected in [('all_window',rows),('temporal_check_odd_frames',[x for x in rows if x['temporal_check']])]:data[mode]={key:float(np.mean([r[key] for r in selected])) for key in ['baseline_unoccluded_mask_iou','candidate_unoccluded_mask_iou']}
 proxy=[]
 for a in proxies:
  a=np.asarray(a);proxy.append({'mean_inside_box_depth_mm':float(a.mean()),'worst_inside_box_depth_mm':float(a.max()),'fraction_positive':float((a>0).mean())})
 report={'status':'full_scene_render_and_diagnostics_completed','decoded_frames':n,'fps':fps,'seconds':time.time()-start,'silhouette_diagnostics':data,'hand_box_proxy_baseline_candidate':proxy,'proxy_scope':'Fixed native hand-surface vertices inside object axis-aligned local bounds. This is a rough proxy, not exact mesh penetration, contact truth or official PEN.','mask_scope':'Unoccluded object geometry projected to original Native crop versus estimated visible object masks; hand occlusion is not subtracted, masks are not ground truth.','candidate_accepted':False,'new_native_optimization_steps':400,'quality_review':'Pending source RGB / baseline / candidate inspection; matching and projection gains are not physical correctness.'};write('mask-diagnostics-by-frame.json',rows);write('report.json',report);write('state.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():write('failure.json',{'status':'failed_preserved','traceback':traceback.format_exc()})
  raise
