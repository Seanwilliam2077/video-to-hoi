from pathlib import Path
import numpy as np,cv2,torch,json,sys
from scipy.spatial.transform import Rotation,Slerp
from lib_mhr.object_texture import load_object_mesh_parts,concatenate_object_mesh_parts
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');D=R/'a6000-runs/direction-correction-v1';sys.path.insert(0,str(R/'a6000-runs'));from optimize_object_pose_v2 import intersect
b=torch.load(D/'source-face-candidate-v4/refined.pth',map_location='cpu',weights_only=False);T=np.asarray(b['pr']['pose_abs']);K=np.asarray(b['observations']['K_full']);obs=b['observations'];m=concatenate_object_mesh_parts(load_object_mesh_parts(b['metadata']['object_mesh']));ann=json.loads((D/'source-geometric-junctions.json').read_text());fs=np.array([int(f) for f in ann['frames'] if int(f)>=90]);gray={};mask={};cap=cv2.VideoCapture(str(R/'runs/full-episode16-20261030/inputs/episode_000016.0.color.mp4'));cap.set(cv2.CAP_PROP_POS_FRAMES,90);shift=np.array([[1,0,-590],[0,1,-490],[0,0,1.]],float);cv2.setNumThreads(2)
for f in range(90,360):
 ok,im=cap.read();assert ok;gray[f]=cv2.cvtColor(im[490:790,590:890],cv2.COLOR_BGR2GRAY);H=shift@K[f]@np.linalg.inv(obs['postopt_K_rois'][f]);mask[f]=cv2.erode(cv2.warpPerspective(np.uint8(obs['postopt_object_mask'][f]>0)*255,H,(300,300),flags=cv2.INTER_NEAREST),np.ones((9,9),np.uint8))
cap.release();S=np.array([Rotation.from_euler('z',90*i,degrees=True).as_matrix() for i in range(4)]);states=np.tile(T[fs,None],(1,4,1,1));states[:,:,:3,:3]=np.einsum('bij,kjl->bkil',T[fs,:3,:3],S);edges=[];rows=[]
def ncc(a,b):
 a=a.astype(float)-a.mean();b=b.astype(float)-b.mean();return float((a*b).sum()/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9))
for i,(a,z) in enumerate(zip(fs[:-1],fs[1:])):
 p=cv2.goodFeaturesToTrack(gray[a],maxCorners=160,qualityLevel=.01,minDistance=3,mask=mask[a],blockSize=5);p=np.zeros((0,1,2),np.float32) if p is None else p;old=p.copy();valid=np.ones(len(p),bool);template=[cv2.getRectSubPix(gray[a],(11,11),tuple(x)) for x in p[:,0]]
 for f in range(a+1,z+1):
  q,st,_=cv2.calcOpticalFlowPyrLK(gray[f-1],gray[f],p,None,winSize=(15,15),maxLevel=2,criteria=(3,30,.01),minEigThreshold=1e-4);back,bs,_=cv2.calcOpticalFlowPyrLK(gray[f],gray[f-1],q,None,winSize=(15,15),maxLevel=2,criteria=(3,30,.01),minEigThreshold=1e-4);valid&=st[:,0].astype(bool)&bs[:,0].astype(bool)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<=.75)
  for j,x in enumerate(q[:,0]):
   if not valid[j]:continue
   xx,yy=np.rint(x).astype(int)
   if not(6<=xx<294 and 6<=yy<294) or not mask[f][yy,xx] or ncc(template[j],cv2.getRectSubPix(gray[f],(11,11),tuple(x)))<.8:valid[j]=False
  p=q
 ids=np.flatnonzero(valid);u=old[ids,0]+[590,490];v=p[ids,0]+[590,490];cost=np.zeros((4,4));counts=[]
 for j in range(4):
  X,keep,_=intersect(u,K[a],states[i,j],np.asarray(m.vertices),np.asarray(m.faces));counts.append(len(X))
  for k in range(4):
   rel=Rotation.from_matrix(states[i,j,:3,:3].T@states[i+1,k,:3,:3]).magnitude()
   if len(X)>=8:
    cam=X@states[i+1,k,:3,:3].T+states[i+1,k,:3,3];pr=cam@K[z].T;pr=pr[:,:2]/pr[:,2:];err=np.linalg.norm(pr-v[keep],axis=-1);cost[j,k]=min(float(np.median(err)),40)+.02*rel**2
   else:cost[j,k]=10*rel**2/(z-a)
 np.savez_compressed(D/f'phase-source-tracks-{a}-{z}.npz',xy_first=u,xy_last=v,raw_ids=ids,pair_cost=cost)
 rows.append({'pair':[int(a),int(z)],'source_full_track_count':len(ids),'surface_bound_counts':counts,'material_supported':min(counts)>=8,'pair_cost_px':cost.tolist()});edges.append(cost)
 print(rows[-1]['pair'],len(ids),np.round(cost[0],2).tolist(),flush=True)
cost=np.full((len(fs),4),np.inf);cost[0,0]=0;prev=np.zeros((len(fs),4),int)
for i in range(1,len(fs)):
 for k in range(4):v=cost[i-1]+edges[i-1][:,k];prev[i,k]=v.argmin();cost[i,k]=v.min()
phase=np.zeros(len(fs),int);phase[-1]=cost[-1].argmin()
for i in range(len(fs)-1,0,-1):phase[i-1]=prev[i,phase[i]]
report={'scope':'Source-image material-motion development check across every annotated edge. No new model calls. One-frame LK FB<=0.75px, fixed11x11 anchor NCC>=0.8, eroded visible Native mask, no reactivation. Conditional mesh ray binding, not ground-truth material identity.','frames':fs.tolist(),'quarter_turn_offsets_from_v4':phase.tolist(),'supported_edges':sum(x['material_supported'] for x in rows),'total_edges':len(rows),'edges':rows};(D/'global-material-phase-audit.json').write_text(json.dumps(report,indent=2));np.savez_compressed(D/'global-material-phase-audit.npz',frames=fs,offsets=phase,rotations=states[np.arange(len(fs)),phase,:3,:3]);print(json.dumps({'offsets':phase.tolist(),'supported_edges':report['supported_edges'],'total_edges':len(rows)}),flush=True)
