from pathlib import Path
import numpy as np,cv2,json,torch,time,hashlib,os
from scipy.spatial.transform import Rotation,Slerp
from pytorch3d.transforms import axis_angle_to_matrix
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');D=R/'a6000-runs/direction-correction-v1';O=D/'source-face-candidate';O.mkdir(exist_ok=False)
assert os.environ['CUDA_VISIBLE_DEVICES']=='1';torch.set_num_threads(4);start=time.time()
b=torch.load(R/'experiments/ep16-joint-direction-20261031/joint-green-v1/refined.pth',map_location='cpu',weights_only=False);base=np.asarray(b['pr']['pose_abs']).copy();K=np.asarray(b['observations']['K_full']);ann=json.loads((D/'source-geometric-junctions.json').read_text());frames=np.array([int(f) for f in ann['frames'] if 90<=int(f)<=280]);uv=np.array([ann['frames'][str(f)] for f in frames])+ann['roi_offset'];a=.0996603;X=np.array([[-a,-a,a],[-a,a,a],[a,a,a],[a,-a,a],[a,-a,-a],[a,a,-a]],float)
init=[];rows=[]
for f,u in zip(frames,uv):
 ok,rv,t=cv2.solvePnP(X,u.astype(float),K[f].astype(float),None,flags=cv2.SOLVEPNP_EPNP);assert ok
 rv,t=cv2.solvePnPRefineLM(X,u.astype(float),K[f].astype(float),None,rv,t);rot=cv2.Rodrigues(rv)[0];pts=X@rot.T+t[:,0];assert (pts[:,2]>0).all();init.append((rot,t[:,0]));pred=pts@K[f].T;err=np.linalg.norm(pred[:,:2]/pred[:,2:]-u,axis=-1);rows.append({'frame':int(f),'pnp_mean_source_junction_error_px':float(err.mean())})
(D/'source-pnp-report.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(D/'source-pnp-init.npz',frames=frames,rotation=np.array([x[0] for x in init]),translation=np.array([x[1] for x in init]))
# Absolute knot rotations, independently initialized from source PnP; no old-ray binding or angle cap.
knot=np.r_[60,frames,300];rots=np.array([base[60,:3,:3]]+[x[0] for x in init]+[base[300,:3,:3]]);trans=np.array([base[60,:3,3]]+[x[1] for x in init]+[base[300,:3,3]])
rot0=torch.tensor(rots,dtype=torch.float32,device='cuda');t0=torch.tensor(trans,dtype=torch.float32,device='cuda');dr=torch.zeros((len(knot)-2,3),device='cuda',requires_grad=True);dt=torch.zeros_like(dr,requires_grad=True);zero=torch.zeros((1,3),device='cuda');XT=torch.tensor(X,dtype=torch.float32,device='cuda');UV=torch.tensor(uv,dtype=torch.float32,device='cuda');KT=torch.tensor(K[frames],dtype=torch.float32,device='cuda');opt=torch.optim.Adam([dr,dt],lr=.01);history=[]
for step in range(400):
 opt.zero_grad();rot=axis_angle_to_matrix(torch.cat([zero,dr,zero]))@rot0;t=t0+torch.cat([zero,dt,zero]);cam=torch.einsum('bij,nj->bni',rot[1:-1],XT)+t[1:-1,None];p=torch.einsum('bij,bnj->bni',KT,cam);p=p[...,:2]/p[...,2:];data=torch.nn.functional.smooth_l1_loss(p,UV,beta=3.)
 # Knot velocity changes; weak smoothness must not erase source observed reversals.
 gap=torch.tensor(np.diff(knot),device='cuda',dtype=torch.float32)[:,None,None];velR=(rot[1:]-rot[:-1])/gap;velt=(t[1:]-t[:-1])/gap[:,0];smooth=5*(velR[1:]-velR[:-1]).square().mean()+50*(velt[1:]-velt[:-1]).square().mean();loss=data+smooth;loss.backward();opt.step()
 if step%50==0 or step==399:history.append({'step':step+1,'source_junction_loss':float(data.detach()),'smooth':float(smooth.detach())});print(history[-1],flush=True)
rot=rot.detach().cpu().numpy();t=t.detach().cpu().numpy();cand=base.copy();idx=np.arange(61,300);cand[idx,:3,:3]=Slerp(knot,Rotation.from_matrix(rot))(idx).as_matrix();cand[idx,:3,3]=np.stack([np.interp(idx,knot,t[:,c]) for c in range(3)],1);assert np.array_equal(cand[:61],base[:61]) and np.array_equal(cand[300:],base[300:]);assert np.linalg.det(cand[:,:3,:3]).min()>.999
for key in ['pose_abs','pose_abs_postopt']:assert np.array_equal(b['pr'][key],base);b['pr'][key]=cand.copy()
b['experimental_object_pose_refinement']={'accepted':False,'method':'Source geometric junctions absolute coupled rotation and translation, with temporal interpolation','source_annotations':str(D/'source-geometric-junctions.json'),'angle_cap':None,'prior_surface_bindings':False,'scope':'Object active motion61..299; frame0..60 and300..359, human/camera/mesh/scale preserved','known_limitation':'Approximate manually read corners of a rounded cube; global texture phase not established. Same source fit observations, not ground truth.'};torch.save(b,O/'refined.pth');np.savez_compressed(O/'knots.npz',frames=knot,rotations=rot,translations=t)
report={'optimizer_steps':400,'physical_gpu':1,'seconds':time.time()-start,'candidate_accepted':False,'annotation_frames':frames.tolist(),'frozen_frames':'0..60,300..359','changed_fields':['pose_abs','pose_abs_postopt'],'human_camera_mesh_scale_unchanged':True,'initial_pnp':rows,'trace':history,'candidate_sha256':hashlib.sha256((O/'refined.pth').read_bytes()).hexdigest()};(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
