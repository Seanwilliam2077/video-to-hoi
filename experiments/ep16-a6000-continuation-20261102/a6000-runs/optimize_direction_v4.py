from pathlib import Path
import numpy as np,cv2,json,torch,time,hashlib,os
from scipy.spatial.transform import Rotation,Slerp
from pytorch3d.transforms import axis_angle_to_matrix
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');D=R/'a6000-runs/direction-correction-v1';O=D/'source-face-candidate-v4';O.mkdir(exist_ok=False)
assert os.environ['CUDA_VISIBLE_DEVICES']=='1';torch.set_num_threads(4);start=time.time()
b=torch.load(R/'experiments/ep16-joint-direction-20261031/joint-green-v1/refined.pth',map_location='cpu',weights_only=False);base=np.asarray(b['pr']['pose_abs']).copy();K=np.asarray(b['observations']['K_full']);ann=json.loads((D/'source-geometric-junctions.json').read_text());frames=np.array([int(f) for f in ann['frames'] if 90<=int(f)<=359]);uv=np.array([ann['frames'][str(f)] for f in frames])+ann['roi_offset'];a=.0996603;X=np.array([[-a,-a,a],[-a,a,a],[a,a,a],[a,-a,a],[a,-a,-a],[a,a,-a]],float)
init=[];rows=[]
for f,u in zip(frames,uv):
 ok,rv,t=cv2.solvePnP(X,u.astype(float),K[f].astype(float),None,flags=cv2.SOLVEPNP_EPNP);assert ok
 rv,t=cv2.solvePnPRefineLM(X,u.astype(float),K[f].astype(float),None,rv,t);rot=cv2.Rodrigues(rv)[0];pts=X@rot.T+t[:,0];assert (pts[:,2]>0).all();init.append((rot,t[:,0]));pred=pts@K[f].T;err=np.linalg.norm(pred[:,:2]/pred[:,2:]-u,axis=-1);rows.append({'frame':int(f),'pnp_mean_source_junction_error_px':float(err.mean())})
# A cube has four indistinguishable corner orderings in silhouette. Unwrap one
# continuous material coordinate frame, instead of relabeling corners by screen side.
S=np.array([Rotation.from_euler('z',90*j,degrees=True).as_matrix() for j in range(4)])
C=np.array([[x[0]@z for z in S] for x in init]);cost=np.full((len(frames),4),np.inf);prev=np.zeros((len(frames),4),int)
cost[0]=[Rotation.from_matrix(base[60,:3,:3].T@r).magnitude()**2/30 for r in C[0]]
for i in range(1,len(frames)):
 for j in range(4):
  v=np.array([cost[i-1,k]+Rotation.from_matrix(C[i-1,k].T@C[i,j]).magnitude()**2/(frames[i]-frames[i-1]) for k in range(4)]);prev[i,j]=v.argmin();cost[i,j]=v.min()
phase=np.zeros(len(frames),int);phase[-1]=cost[-1].argmin()
for i in range(len(frames)-1,0,-1):phase[i-1]=prev[i,phase[i]]
init=[(C[i,j],init[i][1]) for i,j in enumerate(phase)];source_X=np.array([X@S[j] for j in phase])
(O/'corner-phase-unwrapping.json').write_text(json.dumps({'frames':frames.tolist(),'quarter_turn_phase':phase.tolist(),'method':'Global minimum angular path across four cube corner symmetries; first gauge connected to preserved frame60. Not per-frame heading truth.'},indent=2))
(D/'source-pnp-report.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(D/'source-pnp-init.npz',frames=frames,rotation=np.array([x[0] for x in init]),translation=np.array([x[1] for x in init]))
# Absolute knot rotations, independently initialized from source PnP; no old-ray binding or angle cap.
knot=np.r_[60,frames];rots=np.array([base[60,:3,:3]]+[x[0] for x in init]);trans=np.array([base[60,:3,3]]+[x[1] for x in init])
rot0=torch.tensor(rots,dtype=torch.float32,device='cuda');t0=torch.tensor(trans,dtype=torch.float32,device='cuda');dr=torch.zeros((len(knot)-1,3),device='cuda',requires_grad=True);dt=torch.zeros_like(dr,requires_grad=True);zero=torch.zeros((1,3),device='cuda');XT=torch.tensor(source_X,dtype=torch.float32,device='cuda');UV=torch.tensor(uv,dtype=torch.float32,device='cuda');KT=torch.tensor(K[frames],dtype=torch.float32,device='cuda');opt=torch.optim.Adam([dr,dt],lr=.01);history=[]
from pytorch3d.renderer import MeshRasterizer,RasterizationSettings,SoftSilhouetteShader,BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from lib_mhr.object_texture import load_object_mesh_parts,concatenate_object_mesh_parts
m=concatenate_object_mesh_parts(load_object_mesh_parts(b['metadata']['object_mesh']));V=torch.tensor(np.asarray(m.vertices),device='cuda',dtype=torch.float32);F=torch.tensor(np.asarray(m.faces),device='cuda',dtype=torch.long)[None].expand(len(frames),-1,-1)
mk=np.asarray(b['observations']['postopt_K_rois'][frames]).copy();mk[:,:2]*=.5
camera=cameras_from_opencv_projection(torch.eye(3,device='cuda')[None].expand(len(frames),-1,-1),torch.zeros((len(frames),3),device='cuda'),torch.tensor(mk,device='cuda',dtype=torch.float32),torch.full((len(frames),2),128.,device='cuda'))
raster=MeshRasterizer(cameras=camera,raster_settings=RasterizationSettings(image_size=128,blur_radius=float(np.log(1/1e-4-1)*1e-4),faces_per_pixel=10,perspective_correct=True,cull_backfaces=False,max_faces_per_bin=len(m.faces)));shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4,gamma=1e-4))
mt=torch.tensor(np.stack([cv2.resize(np.asarray(b['observations']['postopt_object_mask'][f],np.float32),(128,128),interpolation=cv2.INTER_AREA) for f in frames]),device='cuda');valid=torch.tensor(np.stack([1-cv2.resize(np.asarray(b['observations']['postopt_human_mask'][f],np.float32),(128,128),interpolation=cv2.INTER_NEAREST) for f in frames]),device='cuda').clamp(0,1)
def maskloss(rot,t):
 meshes=Meshes(verts=torch.einsum('bij,vj->bvi',rot[1:],V)+t[1:,None],faces=F);alpha=shader(raster(meshes),meshes)[...,3];inter=(alpha*mt*valid).sum((1,2));union=((alpha+mt-alpha*mt)*valid).sum((1,2)).clamp(min=1);return (1-inter/union).mean()
basez=torch.tensor(base[frames,2,3],device='cuda')

# Continue from the independent face/silhouette candidate; use source tracks as
# conditional material constraints, with bindings recorded explicitly.
previous=torch.load(D/'source-face-candidate-v3/refined.pth',map_location='cpu',weights_only=False);previousT=np.asarray(previous['pr']['pose_abs']);rot0=torch.tensor(previousT[knot,:3,:3],device='cuda');t0=torch.tensor(previousT[knot,:3,3],device='cuda')
from optimize_object_pose_v2 import intersect
trackgroups=[]
for anchor in [150,165,180,195]:
 g=np.load(R/f'a6000-runs/matching-optimization-v2/group-{anchor}-tracks.npz');ids=np.flatnonzero(g['valid'].all(0));points,keep,faces=intersect(g['xy'][0,ids],K[anchor],previousT[anchor],np.asarray(m.vertices),np.asarray(m.faces));keep=ids[keep];selected=np.flatnonzero(np.isin(g['frames'],frames));ki=np.array([np.flatnonzero(knot==f)[0] for f in g['frames'][selected]])
 trackgroups.append((torch.tensor(ki,device='cuda'),torch.tensor(points,device='cuda',dtype=torch.float32),torch.tensor(g['xy'][selected][:,keep],device='cuda',dtype=torch.float32),torch.tensor(K[g['frames'][selected]],device='cuda',dtype=torch.float32)))
 np.savez_compressed(O/f'conditional-bindings-{anchor}.npz',canonical_points=points,source_track_ids=keep,frames=g['frames'][selected],anchor_pose=previousT[anchor])
def trackloss(rot,t):
 losses=[]
 for fi,Xg,Yg,Kg in trackgroups:
  cam=torch.einsum('bij,nj->bni',rot[fi],Xg)+t[fi,None];p=torch.einsum('bij,bnj->bni',Kg,cam);p=p[...,:2]/p[...,2:];losses.append(torch.nn.functional.smooth_l1_loss(p,Yg,beta=2.))
 return sum(losses)/len(losses)
for step in range(400):
 opt.zero_grad();rot=axis_angle_to_matrix(torch.cat([zero,dr]))@rot0;t=t0+torch.cat([zero,dt]);cam=torch.einsum('bij,bnj->bni',rot[1:],XT)+t[1:,None];p=torch.einsum('bij,bnj->bni',KT,cam);p=p[...,:2]/p[...,2:];data=torch.nn.functional.smooth_l1_loss(p,UV,beta=3.)
 # Knot velocity changes; weak smoothness must not erase source observed reversals.
 gap=torch.tensor(np.diff(knot),device='cuda',dtype=torch.float32)[:,None,None];velR=(rot[1:]-rot[:-1])/gap;velt=(t[1:]-t[:-1])/gap[:,0];smooth=5*(velR[1:]-velR[:-1]).square().mean()+50*(velt[1:]-velt[:-1]).square().mean();ml=maskloss(rot,t);gauge=.1*((t[1:,2]-basez)/.1).square().mean();tl=trackloss(rot,t);loss=data+smooth+40*ml+gauge+.5*tl;loss.backward();opt.step()
 if step%50==0 or step==399:history.append({'step':step+1,'source_junction_loss':float(data.detach()),'smooth':float(smooth.detach()),'mask_loss':float(ml.detach()),'depth_gauge_prior':float(gauge.detach()),'conditional_track_loss':float(tl.detach())});print(history[-1],flush=True)
rot=rot.detach().cpu().numpy();t=t.detach().cpu().numpy();cand=base.copy();idx=np.arange(61,360);cand[idx,:3,:3]=Slerp(knot,Rotation.from_matrix(rot))(idx).as_matrix();cand[idx,:3,3]=np.stack([np.interp(idx,knot,t[:,c]) for c in range(3)],1);assert np.array_equal(cand[:61],base[:61]);assert np.linalg.det(cand[:,:3,:3]).min()>.999
for key in ['pose_abs','pose_abs_postopt']:assert np.array_equal(b['pr'][key],base);b['pr'][key]=cand.copy()
b['experimental_object_pose_refinement']={'accepted':False,'method':'Source geometric junctions absolute coupled rotation and translation, with temporal interpolation','source_annotations':str(D/'source-geometric-junctions.json'),'angle_cap':None,'conditional_track_weight':.5,'track_binding_source':'source-face-candidate-v3 geometry; not original wrong ray bindings','cube_corner_symmetry_unwrapped':True,'no_return_to_old_final_heading':True,'estimated_visible_silhouette_fit':True,'soft_depth_gauge_prior_m':.1,'prior_surface_bindings':False,'scope':'Object frames61..359; frame0..60, human/camera/mesh/scale preserved','known_limitation':'Approximate manually read corners of a rounded cube; global texture phase not established. Same source fit observations, not ground truth.'};torch.save(b,O/'refined.pth');np.savez_compressed(O/'knots.npz',frames=knot,rotations=rot,translations=t)
report={'optimizer_steps':400,'physical_gpu':1,'seconds':time.time()-start,'candidate_accepted':False,'annotation_frames':frames.tolist(),'frozen_frames':'0..60','changed_fields':['pose_abs','pose_abs_postopt'],'human_camera_mesh_scale_unchanged':True,'initial_pnp':rows,'trace':history,'candidate_sha256':hashlib.sha256((O/'refined.pth').read_bytes()).hexdigest()};(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
