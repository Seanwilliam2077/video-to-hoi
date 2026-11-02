"""Finite-mass contact dynamics after native joint optimization; not full human dynamics."""
import json,hashlib,copy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np, torch, mujoco
from scipy.spatial.transform import Rotation,Slerp
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1'); E=R/'experiments/ep16-joint-direction-20261031'
O=E/'mujoco-contact-pilot-v2'; O.mkdir(exist_ok=False)
x=np.load(E/'decoded-hand-physics-intake-v1/green.npz'); hands=x['hands']; ids=x['hand_vertex_ids']; pose=x['pose']; T=len(pose)
mp=R/'weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt'
mh=torch.jit.load(str(mp),map_location='cpu'); b=dict(mh.named_buffers())
si=b['character_torch.linear_blend_skinning.skin_indices_flattened'].numpy();vi=b['character_torch.linear_blend_skinning.vert_indices_flattened'].numpy();sw=b['character_torch.linear_blend_skinning.skin_weights_flattened'].numpy()
owner=np.zeros(int(vi.max())+1,dtype=int);weights=np.zeros(len(owner))
for v,j,w in zip(vi,si,sw):
 if w>weights[v]:owner[v]=j;weights[v]=w
del mh,b
parts=[]; errors=[]; covered=0
for side in range(2):
 for j in np.unique(owner[ids[side]]):
  ix=np.flatnonzero(owner[ids[side]]==j)
  ref=hands[150,side,ix].astype(float); ref-=ref.mean(0)
  if len(ix)<4 or np.linalg.matrix_rank(ref)<3:continue
  positions=[]; rotations=[]
  for t in range(T):
   target=hands[t,side,ix].astype(float);cen=target.mean(0);U,_,Vt=np.linalg.svd(ref.T@(target-cen));Q=U@Vt
   if np.linalg.det(Q)<0:U[:,-1]*=-1;Q=U@Vt
   err=np.linalg.norm(ref@Q+cen-target,axis=-1);errors.extend(err.tolist());positions.append(cen);rotations.append(Q.T)
  parts.append((side,int(j),ref,np.array(positions),np.array(rotations)));covered+=len(ix)
assert parts
fmt=lambda a:' '.join(f'{v:.10g}' for v in np.asarray(a).ravel())
def wxyz(rot):return np.roll(Rotation.from_matrix(rot).as_quat(),1,axis=-1)
root=ET.Element('mujoco',model='current_native_hand_contact');ET.SubElement(root,'compiler',angle='radian')
ET.SubElement(root,'option',timestep=str(1/30/16),gravity='0 0 0',integrator='implicitfast',iterations='100',cone='elliptic')
assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
for k,(side,j,ref,pos,rot) in enumerate(parts):
 ET.SubElement(assets,'mesh',name=f'part{k}',vertex=fmt(ref))
 body=ET.SubElement(world,'body',name=f'hand{k}',mocap='true')
 ET.SubElement(body,'geom',type='mesh',mesh=f'part{k}',contype='1',conaffinity='2',friction='.6 .005 .0001',solref='.02 1',solimp='.9 .95 .001')
obj=ET.SubElement(world,'body',name='object');ET.SubElement(obj,'freejoint')
half=(x['box_hi']-x['box_lo'])/2;center=(x['box_hi']+x['box_lo'])/2
ET.SubElement(obj,'geom',name='object_geom',type='box',size=fmt(half),pos=fmt(center),mass='.2',contype='2',conaffinity='1',friction='.6 .005 .0001',solref='.02 1',solimp='.9 .95 .001')
xml=ET.tostring(root,encoding='unicode');(O/'scene.xml').write_text(xml)
model=mujoco.MjModel.from_xml_string(xml);objid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'object')
qref=np.c_[pose[:,:3,3],wxyz(pose[:,:3,:3])]
part_pos=np.stack([p[3] for p in parts],1); part_quat=np.stack([wxyz(p[4]) for p in parts],1)
# Quaternion sign continuity for normalized linear interpolation.
for arr in [qref[:,3:],part_quat]:
 for t in range(1,T):arr[t]*=np.where(np.sum(arr[t]*arr[t-1],axis=-1,keepdims=True)<0,-1,1)
config={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':hashlib.sha256((E/'joint-green-v1/refined.pth').read_bytes()).hexdigest(),'steps_per_frame':16,'fps':30,'mass_assumption_kg':.2,'friction_assumption':.6,'gravity':[0,0,0],'support_plane':False,'tracking_frequency_hz':5,'hands':'kinematic rigid skin-owner parts; not dynamic human','reference_frame_for_parts':150,'mapping_vertex_error_p95_mm':float(np.percentile(errors,95)*1000),'mapping_vertex_error_max_mm':float(np.max(errors)*1000),'covered_vertices':covered,'total_vertices':int(ids.size),'parts':len(parts),'limits':['Contact-only zero-gravity dynamic regularizer, not complete gravity/support reconstruction','Convex skin-owner proxies can overfill curved surfaces','Image direction is not inferred by physics; wrong target may remain wrong','Physics stage keeps jointly optimized hands fixed; not simultaneous full human dynamics','No exact mesh PEN or GT accuracy claims']}
(O/'config.json').write_text(json.dumps(config,indent=2));print('MAPPING',json.dumps(config),flush=True)
results={};original=torch.load(E/'joint-green-v1/refined.pth',map_location='cpu',weights_only=False)
for collision in [False,True]:
 name='dynamics-contact' if collision else 'dynamics-no-contact';out=O/name;out.mkdir()
 if collision:model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
 else:model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
 data=mujoco.MjData(model);data.qpos[:]=qref[0];data.mocap_pos[:]=part_pos[0];data.mocap_quat[:]=part_quat[0];mujoco.mj_forward(model,data)
 inertia=model.body_inertia[objid];omega=2*np.pi*5;mass=np.r_[np.full(3,.2),inertia];kp=mass*omega**2;kd=2*mass*omega
 output=[pose[0].copy()];counts=[int(data.ncon)];worst=[0.];forces=[0.];steps=0
 for t in range(1,T):
  worst_t=0.;force_t=0.;maxcontacts=0
  vel=np.zeros(6);mujoco.mj_differentiatePos(model,vel,1/30,qref[t-1],qref[t])
  for sub in range(1,17):
   a=sub/16;target=(1-a)*qref[t-1]+a*qref[t];target[3:]/=np.linalg.norm(target[3:])
   data.mocap_pos[:]=(1-a)*part_pos[t-1]+a*part_pos[t]
   quat=(1-a)*part_quat[t-1]+a*part_quat[t];data.mocap_quat[:]=quat/np.linalg.norm(quat,axis=-1,keepdims=True)
   error=np.zeros(6);mujoco.mj_differentiatePos(model,error,1.,data.qpos,target)
   data.qfrc_applied[:]=kp*error+kd*(vel-data.qvel)
   mujoco.mj_step(model,data);steps+=1
   assert np.isfinite(data.qpos).all() and np.linalg.norm(data.qvel)<1e4,'unstable dynamics'
   maxcontacts=max(maxcontacts,int(data.ncon))
   for c in range(data.ncon):
    worst_t=max(worst_t,-float(data.contact[c].dist)*1000);w=np.zeros(6);mujoco.mj_contactForce(model,data,c,w);force_t=max(force_t,float(np.linalg.norm(w[:3])))
  p=np.eye(4);p[:3,:3]=Rotation.from_quat(np.roll(data.qpos[3:],-1)).as_matrix();p[:3,3]=data.qpos[:3];output.append(p);counts.append(maxcontacts);worst.append(worst_t);forces.append(force_t)
  if t%60==0:print(name,t,'contacts',maxcontacts,'penetration',worst_t,flush=True)
 output=np.array(output);result=copy.deepcopy(original);result['pr']['pose_abs']=output.astype(np.asarray(original['pr']['pose_abs']).dtype)
 result['postopt']['physics_engine_used']=True;result['postopt']['physics_pilot']={'engine':'MuJoCo','actual_steps':steps,'contacts_enabled':collision,'gravity_zero':True,'human_fixed_during_physics':True}
 torch.save(result,out/'refined.pth');reload=torch.load(out/'refined.pth',map_location='cpu',weights_only=False)
 assert reload['frames']==original['frames']
 for key in original['pr']:
  if key.startswith('mhr_'):assert np.array_equal(reload['pr'][key],original['pr'][key]),key
 drift=np.linalg.norm(output[:,:3,3]-pose[:,:3,3],axis=-1)*1000
 angle=Rotation.from_matrix(output[:,:3,:3]@pose[:,:3,:3].transpose(0,2,1)).magnitude()*180/np.pi
 np.savez_compressed(out/'simulation.npz',pose=output,frames=x['frames'],contact_count=counts,worst_contact_depth_mm=worst,max_contact_force_N=forces)
 results[name]={'mj_step_count':steps,'simulated_seconds':float(data.time),'frame_count':len(output),'contact_frames':int(np.count_nonzero(counts)),'translation_drift_mean_mm':float(drift.mean()),'translation_drift_max_mm':float(drift.max()),'rotation_drift_mean_deg':float(angle.mean()),'rotation_drift_max_deg':float(angle.max()),'worst_proxy_contact_depth_mm':float(max(worst)),'max_contact_force_N':float(max(forces)),'reload_frames_human_equal':True,'warnings':[int(w.number) for w in data.warning]}
 print(name,json.dumps(results[name]),flush=True)
(O/'report.json').write_text(json.dumps({'config':config,'results':results,'accepted':False},indent=2));print('PHYSICS_PILOT_COMPLETE',flush=True)

