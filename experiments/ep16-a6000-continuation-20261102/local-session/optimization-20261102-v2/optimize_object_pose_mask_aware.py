"""One bounded source-short-track object SE(3) optimization; full native bundle retained."""
from pathlib import Path
import json,time,hashlib,os,signal,traceback
import numpy as np,torch
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer import MeshRasterizer,RasterizationSettings,SoftSilhouetteShader,BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
import cv2
from lib_mhr.object_texture import load_object_mesh_parts,concatenate_object_mesh_parts
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');O=R/'a6000-runs/matching-optimization-v2';W=O/'pose-candidate-v2-mask-aware'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,d):(W/n).write_text(json.dumps(d,indent=2,allow_nan=False))
def intersect(points,K,T,vertices,faces):
 tri=vertices[faces];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];origin=-T[:3,:3].T@T[:3,3];dirs=np.c_[points,np.ones(len(points))]@np.linalg.inv(K).T@T[:3,:3];hit=[];ids=[];faceids=[]
 for i,d in enumerate(dirs):
  p=np.cross(d,e2);det=(e1*p).sum(1);ok=np.abs(det)>1e-10;inv=np.zeros_like(det);inv[ok]=1/det[ok];s=origin-tri[:,0];u=(s*p).sum(1)*inv;q=np.cross(s,e1);v=(q*d).sum(1)*inv;distance=(e2*q).sum(1)*inv;ok&=(u>=-1e-6)&(v>=-1e-6)&(u+v<=1+1e-6)&(distance>1e-5)
  if ok.any():
   f=int(np.argmin(np.where(ok,distance,np.inf)));hit.append(origin+d*distance[f]);ids.append(i);faceids.append(f)
 return np.asarray(hit),np.asarray(ids,int),np.asarray(faceids,int)
def deadline(a,b):raise TimeoutError('Fixed optimizer budget reached')
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1';W.mkdir(exist_ok=False);start=time.time();write('state.json',{'status':'binding_short_tracks','optimizer_steps_completed':0,'pid':os.getpid()})
 src=R/'experiments/ep16-joint-direction-20261031/joint-green-v1/refined.pth';assert sha(src)=='61decc2bae1b4c5db67c30c7aee521edf61e5c1f5a352e70eec1f03c57234f32'
 b=torch.load(src,map_location='cpu',weights_only=False);assert list(map(int,b['frames']))==list(range(360));base=np.asarray(b['pr']['pose_abs']).copy();K=np.asarray(b['observations']['K_full']);meshpath=Path(b['metadata']['object_mesh']);mesh=concatenate_object_mesh_parts(load_object_mesh_parts(meshpath));V=np.asarray(mesh.vertices,float);F=np.asarray(mesh.faces,int);groups=[];bindings=[]
 for anchor in [150,165,180,195]:
  path=O/f'group-{anchor}-tracks.npz'
  with np.load(path,allow_pickle=False) as z:frames=z['frames'].copy();xy=z['xy'].copy();valid=z['valid'].copy();score=z['ncc'].copy();fb=z['fb'].copy()
  full=np.flatnonzero(valid.all(0));X,localids,faces=intersect(xy[0,full],K[anchor],base[anchor],V,F);selected=full[localids]
  assert len(X)>=8,(anchor,'too few source-supported ray intersections',len(X))
  camera=X@base[anchor,:3,:3].T+base[anchor,:3,3];uv=camera@K[anchor].T;uv=uv[:,:2]/uv[:,2:];err=float(np.abs(uv-xy[0,selected]).max());assert err<.01,err
  obs=xy[:,selected];groups.append((frames-150,X,obs));bindings.append({'anchor':anchor,'input_track_sha256':sha(path),'complete_short_tracks':len(full),'surface_bound_tracks':len(X),'raw_track_indices':selected.tolist(),'triangle_indices':faces.tolist(),'anchor_projection_max_px':err,'binding_assumption':'First visible surface of the unchanged current object mesh under the current anchor pose; conditional initialization, not true physical feature identity.'})
  np.savez_compressed(W/f'group-{anchor}-surface-bindings.npz',frames=frames,canonical_points=X,observed_xy=obs,raw_track_indices=selected,triangle_ids=faces)
 write('surface-binding-report.json',bindings)
 protocol={'candidate':'One bounded conditional object-pose experiment, not an accepted reconstruction.','optimizer':'Adam','steps':300,'learning_rate':.04,'knot_frames':list(range(150,211,5)),'max_rotation_delta_norm_degrees':10,'max_translation_delta_norm_m':.03,'exactly_preserved_frames':'0..150 and 210..359','fixed':['native human parameters','shared K','object mesh/scale','source frame IDs','other native parameter fields'],'fit_frames':'Even original source frames within 150..210','temporal_development_checks':'Odd source frames; not independent video or ground truth','mask_correction_reason':'First point-only candidate reduced reprojection error but degraded observed silhouette IoU; add the missing source silhouette term, no threshold or parameter-range search.','mask_fit_frames':[150,160,170,180,190,200,210],'mask_raster_resolution':128,'mask_loss_weight':40,'loss':{'reprojection':'Per-group mean SmoothL1 beta=2 original pixels','delta_acceleration':'.25 * mean((100*d2(rot))^2) + .25 * mean((500*d2(trans))^2)','delta_prior':'.01 * mean((rot/rmax)^2) + .01 * mean((trans/tmax)^2)'},'no_hyperparameter_scan':True,'network_downloads':0,'feature_identity_limit':'Robust short-track checks reduce mismatches; repeated pattern aliases and wrong anchor surface binding remain possible. Full human-object rendering is mandatory before interpreting the candidate.'}
 write('optimization-protocol.json',protocol)
 device='cuda';torch.set_num_threads(4);torch.manual_seed(0);rmax=np.deg2rad(10.);tmax=.03;rraw=torch.zeros((11,3),device=device,requires_grad=True);traw=torch.zeros((11,3),device=device,requires_grad=True);opt=torch.optim.Adam([rraw,traw],lr=.04)
 BT=torch.tensor(base[150:211],device=device,dtype=torch.float32);KT=torch.tensor(K[150:211],device=device,dtype=torch.float32);gs=[(torch.as_tensor(fr,device=device,dtype=torch.long),torch.tensor(X,device=device,dtype=torch.float32),torch.tensor(xy,device=device,dtype=torch.float32)) for fr,X,xy in groups]
 mask_frames=torch.tensor([0,10,20,30,40,50,60],device=device,dtype=torch.long);B=len(mask_frames)
 VK=torch.tensor(V,device=device,dtype=torch.float32);FT=torch.tensor(F,device=device,dtype=torch.long)[None].expand(B,-1,-1)
 cropK=np.asarray(b['observations']['postopt_K_rois'][[150,160,170,180,190,200,210]]).copy();cropK[:,:2]*=.5
 mk=torch.tensor(cropK,device=device,dtype=torch.float32);sizes=torch.full((B,2),128.,device=device)
 cameras=cameras_from_opencv_projection(torch.eye(3,device=device)[None].expand(B,-1,-1),torch.zeros((B,3),device=device),mk,sizes)
 blend=BlendParams(sigma=1e-4,gamma=1e-4);raster=MeshRasterizer(cameras=cameras,raster_settings=RasterizationSettings(image_size=128,blur_radius=float(np.log(1/1e-4-1)*1e-4),faces_per_pixel=10,perspective_correct=True,cull_backfaces=False,max_faces_per_bin=len(F)))
 shader=SoftSilhouetteShader(blend_params=blend)
 mask_target=torch.tensor(np.stack([cv2.resize(np.asarray(b['observations']['postopt_object_mask'][t],np.float32),(128,128),interpolation=cv2.INTER_AREA) for t in [150,160,170,180,190,200,210]]),device=device)
 mask_valid=torch.tensor(np.stack([1-cv2.resize(np.asarray(b['observations']['postopt_human_mask'][t],np.float32),(128,128),interpolation=cv2.INTER_NEAREST) for t in [150,160,170,180,190,200,210]]),device=device).clamp(0,1)
 def silhouette_loss(rot,trans):
  verts=torch.einsum('bij,vj->bvi',rot[mask_frames],VK)+trans[mask_frames,None]
  meshes=Meshes(verts=verts,faces=FT);fragments=raster(meshes);alpha=shader(fragments,meshes)[...,3]
  inter=(alpha*mask_target*mask_valid).sum((1,2));union=((alpha+mask_target-alpha*mask_target)*mask_valid).sum((1,2)).clamp(min=1)
  return (1-inter/union).mean()
 q=torch.arange(61,device=device,dtype=torch.float32)/5;lo=torch.floor(q).long().clamp(max=11);hi=lo+1;alpha=(q-lo)[:,None];zero=torch.zeros((1,3),device=device)
 def pose():
  rk=torch.cat([zero,torch.tanh(rraw)*rmax/np.sqrt(3),zero]);tk=torch.cat([zero,torch.tanh(traw)*tmax/np.sqrt(3),zero]);dr=rk[lo]*(1-alpha)+rk[hi]*alpha;dt=tk[lo]*(1-alpha)+tk[hi]*alpha
  rot=axis_angle_to_matrix(dr)@BT[:,:3,:3];trans=BT[:,:3,3]+dt;return rot,trans,dr,dt
 def errors(rot,trans):
  values=[]
  for frames,X,target in gs:
   cam=torch.einsum('bij,nj->bni',rot[frames],X)+trans[frames,None];assert bool((cam[...,2]>.001).all());uv=torch.einsum('bij,bnj->bni',KT[frames],cam);uv=uv[...,:2]/uv[...,2:];values.append((frames,uv-target))
  return values
 def summarize(e):
  fit=[];check=[]
  for fr,res in e:
   v=res.norm(dim=-1).detach().cpu().numpy();f=fr.detach().cpu().numpy()+150;fit.extend(v[f%2==0].reshape(-1).tolist());check.extend(v[f%2==1].reshape(-1).tolist())
  return {'fit_mean_px':float(np.mean(fit)),'fit_median_px':float(np.median(fit)),'check_mean_px':float(np.mean(check)),'check_median_px':float(np.median(check))}
 before=summarize(errors(BT[:,:3,:3],BT[:,:3,3]));write('baseline-reprojection.json',before);print('Baseline',before,flush=True);trace=[];signal.signal(signal.SIGALRM,deadline);signal.alarm(1200)
 for step in range(300):
  opt.zero_grad();rot,trans,dr,dt=pose();ev=errors(rot,trans);data=sum(torch.nn.functional.smooth_l1_loss(res[(frames+150)%2==0],torch.zeros_like(res[(frames+150)%2==0]),beta=2.,reduction='mean') for frames,res in ev)/len(ev)
  d2r=dr[2:]-2*dr[1:-1]+dr[:-2];d2t=dt[2:]-2*dt[1:-1]+dt[:-2];smooth=.25*((100*d2r).square().mean()+(500*d2t).square().mean());prior=.01*((dr/rmax).square().mean()+(dt/tmax).square().mean());mask_loss=silhouette_loss(rot,trans);loss=data+smooth+prior+40*mask_loss;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([rraw,traw],10.);opt.step()
  if step%20==0 or step==299:
   row={'step':step+1,'loss':float(loss.detach()),'data_loss':float(data.detach()),'smooth_loss':float(smooth.detach()),'prior_loss':float(prior.detach()),'silhouette_loss':float(mask_loss.detach())};trace.append(row);write('state.json',{'status':'optimizing_object_pose','optimizer_steps_completed':step+1,'elapsed_seconds':time.time()-start,**row});print(row,flush=True)
 signal.alarm(0)
 with torch.no_grad():rot,trans,dr,dt=pose();after=summarize(errors(rot,trans));candidate=base.copy();candidate[151:210,:3,:3]=rot[1:-1].cpu().numpy();candidate[151:210,:3,3]=trans[1:-1].cpu().numpy()
 assert np.array_equal(candidate[:151],base[:151]) and np.array_equal(candidate[210:],base[210:]);assert np.isfinite(candidate).all();assert np.linalg.det(candidate[:,:3,:3]).min()>.999
 original_pr={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in b['pr'].items()};b['pr']['pose_abs']=candidate
 updated=['pose_abs']
 # Preserve coherent duplicate object-pose aliases only when they were exact copies.
 if 'pose_abs_postopt' in b['pr'] and np.array_equal(original_pr['pose_abs_postopt'],base):b['pr']['pose_abs_postopt']=candidate.copy();updated.append('pose_abs_postopt')
 for k,v in original_pr.items():
  if k not in updated and isinstance(v,np.ndarray):assert np.array_equal(v,b['pr'][k],equal_nan=True),k
 b['experimental_object_pose_refinement']={'method':'source-short-track-and-silhouette-bounded-SE3-v3','input_bundle_sha256':sha(src),'accepted':False,'human_camera_shape_unchanged':True,'updated_pr_fields':updated,'protocol':str(W/'optimization-protocol.json')};target=W/'refined.pth';torch.save(b,target);np.savez_compressed(W/'object-pose-deltas.npz',frame_ids=np.arange(150,211),axis_angle_delta=dr.detach().cpu().numpy(),translation_delta=dt.detach().cpu().numpy())
 report={'status':'optimization_completed_candidate_awaits_full_scene_review','optimizer_steps_completed':300,'physical_gpu':1,'source_silhouette_term_added':True,'mask_fit_frames':[150,160,170,180,190,200,210],'source_track_groups':bindings,'before':before,'after':after,'heldout_mean_improved':after['check_mean_px']<before['check_mean_px'],'max_rotation_delta_degrees':float(dr.detach().norm(dim=-1).max().cpu()*180/np.pi),'max_translation_delta_m':float(dt.detach().norm(dim=-1).max().cpu()),'candidate_sha256':sha(target),'updated_pr_fields':updated,'human_camera_mesh_scale_unchanged':True,'frames_150_210_300_exactly_preserved':True,'full_frame_ids_exact':True,'seconds':time.time()-start,'physical_quality_accepted':False,'limitations':['Surface binding inherits the current anchor geometry and pose; it is an explicit conditional assumption.','Temporal checks use source track proposals from the same clip, not 3D ground truth or unseen video.','Human hand errors remain; object-only fitting is not a full hand-contact solution.','No official PEN/ACC or submission accuracy claim.']};write('optimization-trace.json',trace);write('report.json',report);write('state.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  if W.exists():write('failure.json',{'status':'failed_preserved','traceback':traceback.format_exc()})
  raise
