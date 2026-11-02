from pathlib import Path
import numpy as np,cv2,json,torch
from scipy.spatial.transform import Rotation
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');P=R/'mast3r-pilot';Q=R/'a6000-runs/matching-optimization-v2';O=R/'a6000-runs/direction-correction-v1'
b=torch.load(R/'experiments/ep16-joint-direction-20261031/joint-green-v1/refined.pth',map_location='cpu',weights_only=False);K=np.asarray(b['observations']['K_full']);base=np.asarray(b['pr']['pose_abs']);polys=json.loads((P/'source-interior-correction-v1/source-box-interiors.json').read_text())['polygons']
def sample(x,xy):
 net=(xy-[590,490]+.5)*512/300-.5-[0,64]
 return cv2.remap(x.astype(np.float32),net[:,0].astype(np.float32).reshape(-1,1),net[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR)[:,0]
def camera(xyz,f):
 yy,xx=np.mgrid[2:384:4,2:512:4];net=np.c_[xx.ravel(),yy.ravel()];orig=(net+[0,64]+.5)*300/512-.5+[590,490];sel=np.array([cv2.pointPolygonTest(np.asarray(polys[str(f)],np.float32),tuple(map(float,p)),False)>=0 for p in orig-[590,490]])
 X=xyz[net[sel,1],net[sel,0]].astype(float);uv=orig[sel].astype(float)
 ok,rv,t,ids=cv2.solvePnPRansac(X,uv,K[f].astype(float),None,iterationsCount=200,reprojectionError=3.,flags=cv2.SOLVEPNP_EPNP);assert ok
 rv,t=cv2.solvePnPRefineLM(X[ids[:,0]],uv[ids[:,0]],K[f].astype(float),None,rv,t)
 return cv2.Rodrigues(rv)[0],t[:,0]
def sim3(X,Y):
 w=np.ones(len(X))
 for _ in range(8):
  w=w/w.sum();a=(X*w[:,None]).sum(0);b=(Y*w[:,None]).sum(0);U,S,VT=np.linalg.svd(((Y-b)*w[:,None]).T@(X-a));D=np.eye(3);D[2,2]=np.linalg.det(U@VT);rot=U@D@VT;scale=(S*np.diag(D)).sum()/((X-a)**2*w[:,None]).sum();t=b-scale*rot@a;err=np.linalg.norm(scale*X@rot.T+t-Y,axis=1);cut=max(float(np.median(err)*2.5),1e-5);w=1/np.maximum(1,err/cut)
 return rot,t,scale,float(np.median(err))
rows=[];rots=[];translations=[]
for i,a in enumerate(range(150,210,5)):
 anchor=150+15*((a-150)//15);g=np.load(Q/f'group-{anchor}-tracks.npz');ids=np.flatnonzero(g['valid'].all(0));xy1=g['xy'][a-anchor,ids];xy2=g['xy'][a+5-anchor,ids];rr=[];tt=[];stats=[]
 for reverse in [False,True]:
  f1,f2=(a+5,a) if reverse else (a,a+5);j=2*i+int(reverse);z=np.load(P/f'object-cue-13frames-v1/raw/{j:02}_{f1}_{f2}_predictions.npz');X=z['pred1_pts3d'][0];Y=z['pred2_pts3d_in_other_view'][0];r1,t1=camera(X,f1);r2,t2=camera(Y,f2);u,v=(xy2,xy1) if reverse else (xy1,xy2);rg,tg,s,e=sim3(sample(X,u),sample(Y,v));rot=r2@rg@r1.T;trans=(r2@tg+t2)/s-rot@t1
  depth=np.median((sample(X,u)@r1.T+t1)[:,2]);metric=base[f1,2,3]/depth;trans*=metric
  if reverse:trans=-rot.T@trans;rot=rot.T
  rr.append(rot);tt.append(trans);stats.append({'scale':float(s),'median_cloud_residual':e,'rvec_degrees':Rotation.from_matrix(rot).as_rotvec(degrees=True).tolist()})
 discrepancy=float(Rotation.from_matrix(rr[0].T@rr[1]).magnitude()*180/np.pi);rot=rr[0]@Rotation.from_rotvec(.5*Rotation.from_matrix(rr[0].T@rr[1]).as_rotvec()).as_matrix();trans=np.mean(tt,axis=0);row={'pair':[a,a+5],'tracks':len(ids),'forward_reverse':stats,'disagreement_degrees':discrepancy,'rvec_degrees':Rotation.from_matrix(rot).as_rotvec(degrees=True).tolist(),'baseline_rvec_degrees':Rotation.from_matrix(base[a+5,:3,:3]@base[a,:3,:3].T).as_rotvec(degrees=True).tolist()};rows.append(row);rots.append(rot);translations.append(trans);print(json.dumps(row),flush=True)
(O/'coupled-relative-motion.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(O/'coupled-relative-motion.npz',frames=np.arange(150,211,5),relative_R=rots,relative_t=translations)
