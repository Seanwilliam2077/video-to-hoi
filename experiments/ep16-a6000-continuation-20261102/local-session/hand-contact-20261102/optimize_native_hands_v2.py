"""Fixed accepted object motion, native articulated arm/hand contact continuation."""
from pathlib import Path
import os,json,time,hashlib,copy,signal
import numpy as np,cv2,torch
from scipy.interpolate import CubicSpline
from pytorch3d.transforms import axis_angle_to_matrix
from lib_mhr import MHRLayer
from tools.render_mhr_wild_inference import _params
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');D=R/'a6000-runs/hand-contact-20261102-v1';O=D/'native-contact-candidate-v2';O.mkdir(exist_ok=False);assert os.environ['CUDA_VISIBLE_DEVICES']=='1';torch.set_num_threads(4);torch.manual_seed(7);start=time.time();src=R/'a6000-runs/direction-correction-v1/source-material-candidate-v6/refined.pth';b=torch.load(src,map_location='cpu',weights_only=False);g=np.load(D/'input-geometry.npz');layer=MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'],device='cuda');frames=np.arange(50,291);N=len(frames);knots=np.arange(50,291,2);basis=CubicSpline(knots,np.eye(len(knots)),axis=0,bc_type='natural')(frames);envelope=np.minimum(np.clip((frames-50)/10,0,1),np.clip((290-frames)/10,0,1));envelope=envelope**3*(10+envelope*(-15+6*envelope))
def t(x,dtype=torch.float32):return torch.tensor(x,device='cuda',dtype=dtype)
W=t(basis);E=t(envelope[:,None]);params0=_params(b['pr'],slice(50,291),'cuda');body0=params0['mhr_body_pose_cont'];hand0=params0['mhr_hand'];T=t(b['pr']['pose_abs'][frames]);K=t(b['observations']['K_full'][frames]);KC=t(b['observations']['postopt_K_rois'][frames]);refv=t(g['vertices'][frames]);refk=t(g['keypoints'][frames]);hids=t(g['hand_vertex_ids'],torch.long);faces=t(g['human_faces'][::8],torch.long);normals=t(g['convex_planes'][:,:3]);offset=t(g['convex_planes'][:,3]);nworld=torch.einsum('bij,pj->bpi',T[:,:3,:3],normals);dworld=offset[None]-(nworld*T[:,None,:3,3]).sum(-1)
mask=np.asarray(b['observations']['postopt_object_mask'][frames],np.float32);human=np.asarray(b['observations']['postopt_human_mask'][frames],np.float32);mask=np.stack([cv2.erode((x*(1-y)>.5).astype(np.uint8),np.ones((3,3),np.uint8)).astype(np.float32) for x,y in zip(mask,human)]);OM=t(mask[:,None]);raw=torch.nn.Parameter(torch.zeros((len(knots),122),device='cuda'));oldcurve=np.load(D/'native-contact-candidate-v1/control-knots.npz');raw.data.copy_(t(CubicSpline(oldcurve['knots'],oldcurve['raw'],axis=0,bc_type='natural')(knots)));opt=torch.optim.Adam([raw],lr=.006)
# Physical shoulder increments plus valid sin/cos elbow/wrist increments; native PCA fingers.
shoulder_cap=np.deg2rad(35);pair_cap=t(np.deg2rad([35,50,50,50,35,50,50,50]));protocol={'input':str(src),'object_motion':'All360 rotations AND translations exactly frozen; user accepted turn direction.','native_mutable_fields':['mhr_body_pose_cont selected shoulder/elbow/wrist controls','mhr_hand native PCA108'],'frozen':['object pose aliases','human root','human translation','shape','scale','face','camera','non-arm body controls','frames0..50 and290..359'],'correction_curve':'Natural cubic knots every2frames50..290, quintic zero envelope10frames each boundary.','shoulder_delta_norm_cap_deg':35,'elbow_delta_cap_deg':35,'wrist_delta_component_cap_deg':50,'finger_pca_delta_component_cap':.3,'steps':1200,'batch_frames':12,'collision':'All2318 native surface vertices per hand plus wholehuman vertices and1/8 face-centroids against actual convex mesh6planes,2mm clearance.','occlusion':'Where eroded observed object mask excludes observed human, projected hand vertices must lie beyond exact convex-ray exit surface.','contact':'Closest32 native hand vertices attracted to object surface only70..260; source grasp-window proposal, not measured contact truth.','image_prior':'Input wrist/elbow and hand projection retention is a prior, not source ground truth.','physics_engine':False,'downloads':0,'accepted':False};(O/'protocol.json').write_text(json.dumps(protocol,indent=2))
# Initialize native buffers outside inference_mode, freeze model weights, retain input derivatives.
with torch.no_grad():warm=layer.mhr_forward({k:v[:1] for k,v in params0.items()})
layer.backend.head.requires_grad_(False)
def controls(hand_gain=1.):
 x=W@raw;sh=torch.tanh(x[:,:6]).reshape(N,2,3)*shoulder_cap/np.sqrt(3);pa=torch.tanh(x[:,6:14])*pair_cap;hp=.3*torch.tanh(x[:,14:])*hand_gain;return sh*E[:,None],pa*E,hp*E

def decode(ids,hand_gain=1.):
 sh,pa,hp=controls(hand_gain);p={k:v[ids] for k,v in params0.items()};body=body0[ids].clone()
 for side,col in enumerate([42,54]):
  columns=body0[ids,col:col+6].reshape(-1,2,3).transpose(1,2);new=axis_angle_to_matrix(sh[ids,side])@columns;body[:,col:col+6]=new.transpose(1,2).reshape(-1,6)
 pairs=body0[ids,150:166].reshape(-1,8,2);s,c=pa[ids].sin(),pa[ids].cos();y=pairs[...,0]*c+pairs[...,1]*s;x=pairs[...,1]*c-pairs[...,0]*s;body[:,150:166]=torch.stack([y,x],-1).reshape(-1,16);p['mhr_body_pose_cont']=body;p['mhr_hand']=hand0[ids]+hp[ids];return layer.mhr_forward(p),p

def sdf(points,ids):
 local=(points-T[ids,None,:3,3])@T[ids,:3,:3];return (local@normals.T+offset).amax(-1)
def project(points,kk):
 q=torch.einsum('bij,bnj->bni',kk,points);return q[...,:2]/q[...,2:]

def loss(ids,hand_gain=1.):
 out,_=decode(ids,hand_gain);v=out.vertices;hands=v[:,hids].reshape(len(ids),-1,3);ss=sdf(hands,ids);pen=torch.relu(.002-ss);pc=10*(pen/.01).square().mean()+80*(pen.reshape(len(ids),2,-1).amax(-1)/.005).square().mean();allpts=torch.cat([v,v[:,faces].mean(2)],1);sp=sdf(allpts,ids);ap=torch.relu(.001-sp);pc=pc+20*(ap.topk(64,dim=1).values/.005).square().mean()
 uv=project(hands,KC[ids]);grid=(uv/255*2-1)[:,None];source_obj=torch.nn.functional.grid_sample(OM[ids],grid,align_corners=True,padding_mode='zeros')[:,0,0];rays=hands/hands[...,2:];coef=torch.einsum('bnj,bpj->bnp',rays,nworld[ids]);den=torch.where(coef.abs()>1e-7,coef,torch.ones_like(coef));z=-dworld[ids,None]/den;exit=torch.where(coef>1e-7,z,torch.full_like(z,1e4)).amin(-1);entry=torch.where(coef< -1e-7,z,torch.full_like(z,-1e4)).amax(-1);hit=(exit>=entry)&(exit>0)&(exit<10);occ=20*((torch.relu(exit+.002-hands[...,2])/.03).square()*source_obj*hit).mean()
 # Retain image location while allowing depth and wrist reorientation.
 jointids=[7,8,62,41];up=project(out.keypoints[:,jointids],K[ids]);ur=project(refk[ids][:,jointids],K[ids]);image=.2*((up-ur)/10).square().mean();hu=project(hands,K[ids]);href=project(refv[ids][:,hids].reshape(len(ids),-1,3),K[ids]);image=image+.01*((hu-href)/20).square().mean()
 dist=ss.reshape(len(ids),2,-1).abs().topk(32,dim=-1,largest=False).values.mean(-1);active=((frames[ids.detach().cpu().numpy()]>=70)&(frames[ids.detach().cpu().numpy()]<=260));ct=2.*((dist/.01).square()*t(active)[:,None]).mean() if hand_gain>0 else ss.sum()*0
 sh,pa,hp=controls(hand_gain);prior=.08*((sh/.35).square().mean()+(pa/.5).square().mean())+.1*(hp/.1).square().mean();smooth=.05*((sh[2:]-2*sh[1:-1]+sh[:-2])/.02).square().mean()+.05*((pa[2:]-2*pa[1:-1]+pa[:-2])/.02).square().mean()+.01*((hp[2:]-2*hp[1:-1]+hp[:-2])/.02).square().mean();total=pc+occ+image+ct+prior+smooth
 return total,{'collision':pc,'occlusion':occ,'image_prior':image,'contact':ct,'pose_prior':prior,'smooth':smooth,'actual_hand_max_penetration_mm':torch.relu(-ss).max()*1000,'actual_hand_mean_penetration_mm':torch.relu(-ss).mean()*1000}
# Real input identity and analytic ray/inside sign fixtures before an actual update.
with torch.no_grad():out,p=decode(t([125],torch.long));zeroerr=float((out.vertices-refv[125:126]).abs().max());assert np.isfinite(zeroerr),zeroerr
plane_test=torch.tensor([[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.]],device='cuda');test=torch.tensor([[0,0,0],[.12,0,0]],device='cuda');assert torch.allclose((test@plane_test.T-.1).amax(1),torch.tensor([-.1,.02],device='cuda'),atol=1e-6)
(O/'preflight.json').write_text(json.dumps({'warm_start_decode_delta_from_original_max_abs_m':zeroerr,'inside_sign_fixture_passed':True},indent=2));rng=np.random.default_rng(7);history=[];order=[]
for step in range(1200):
 if not order:order=rng.permutation(np.unique(np.r_[np.arange(0,N-11,8),N-12])).tolist()
 lo=order.pop();ids=torch.arange(lo,min(lo+12,N),device='cuda');opt.zero_grad(set_to_none=True);value,metrics=loss(ids,1.);assert torch.isfinite(value);value.backward();assert torch.isfinite(raw.grad).all();torch.nn.utils.clip_grad_norm_([raw],100.);opt.step();opt.param_groups[0]['lr']=.001+.005*(.5+.5*np.cos(np.pi*step/1200))
 if step%25==0 or step==1199:
  row={'step':step+1,'loss':float(value.detach()),**{k:float(v.detach()) for k,v in metrics.items()},'source_frames':[int(frames[lo]),int(frames[min(lo+11,N-1)])],'seconds':time.time()-start};history.append(row);(O/'state.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
# Native parameters are authoritative. No mesh-only displacement is exported.
outbundle=copy.deepcopy(b)
with torch.no_grad():
 for lo in range(0,N,12):
  ids=torch.arange(lo,min(lo+12,N),device='cuda');_,p=decode(ids);outbundle['pr']['mhr_body_pose_cont'][frames[lo:lo+len(ids)]]=p['mhr_body_pose_cont'].cpu().numpy();outbundle['pr']['mhr_hand'][frames[lo:lo+len(ids)]]=p['mhr_hand'].cpu().numpy()
for key in b['pr']:
 if key not in ['mhr_body_pose_cont','mhr_hand']:assert np.array_equal(b['pr'][key],outbundle['pr'][key],equal_nan=True),key
 else:
  assert np.array_equal(b['pr'][key][:51],outbundle['pr'][key][:51]);assert np.array_equal(b['pr'][key][290:],outbundle['pr'][key][290:])
non=np.setdiff1d(np.arange(260),np.r_[42:48,54:60,150:166]);assert np.array_equal(b['pr']['mhr_body_pose_cont'][:,non],outbundle['pr']['mhr_body_pose_cont'][:,non]);outbundle['experimental_hand_contact']={'protocol':str(O/'protocol.json'),'accepted':False,'native_arm_and_hand_parameter_update':True,'object_poses_exactly_unchanged':True};torch.save(outbundle,O/'refined.pth');np.savez_compressed(O/'control-knots.npz',knots=knots,raw=raw.detach().cpu().numpy());(O/'history.json').write_text(json.dumps(history,indent=2));(O/'report.json').write_text(json.dumps({'status':'native_hand_contact_optimization_completed','steps':1200,'seconds':time.time()-start,'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256((O/'refined.pth').read_bytes()).hexdigest(),'object_rotation_translation_aliases_all_exact':True,'nonarm_root_camera_scale_shape_frozen':True,'full360frame_ids_preserved':True,'physical_gpu':1,'accepted':False,'full_scene_review_required':True},indent=2));print('NATIVE_CONTACT_CANDIDATE_COMPLETED',flush=True)
