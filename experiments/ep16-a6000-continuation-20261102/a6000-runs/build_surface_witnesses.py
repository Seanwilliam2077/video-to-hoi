from pathlib import Path
import argparse,time,json,numpy as np,torch
from scipy.optimize import linprog
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');D=R/'a6000-runs/hand-contact-20261102-v1';ap=argparse.ArgumentParser();ap.add_argument('--stage',default='v8');args=ap.parse_args();g=np.load(D/f'native-contact-candidate-{args.stage}-evaluation/input-geometry.npz');b=torch.load(D/f'native-contact-candidate-{args.stage}/refined.pth',map_location='cpu',weights_only=False);planes=g['convex_planes'];normal=planes[:,:3];offset=planes[:,3];V=g['vertices'];F=g['human_faces'];lo=g['object_vertices'].min(0);hi=g['object_vertices'].max(0);center=(lo+hi)/2;half=(hi-lo)/2;start=time.time()
def candidates(tri,h):
 ids=np.flatnonzero((tri.min(1)<=h).all(-1)&(tri.max(1)>=-h).all(-1));x=tri[ids]
 if not len(x):return ids
 e=np.roll(x,-1,axis=1)-x;axes=np.cross(e[:,:,None,:],np.eye(3)[None,None,:,:]).reshape(len(x),9,3);p=np.einsum('mnj,maj->man',x,axes);r=np.einsum('maj,j->ma',abs(axes),h);keep=~((p.min(-1)>r+1e-10)|(p.max(-1)<-r-1e-10)).any(1);x=x[keep];ids=ids[keep]
 if not len(x):return ids
 n=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]);p=np.einsum('mj,mj->m',x[:,0],n);r=abs(n)@h;return ids[abs(p)<=r+1e-10]
def deepest(tri):
 A=tri[0];edges=(tri[1:]-A).T;lhs=np.c_[normal@edges,np.ones(len(normal))];rhs=-normal@A-offset;res=linprog([0,0,-1],A_ub=np.r_[lhs,[[1,1,0]]],b_ub=np.r_[rhs,1],bounds=[(0,1),(0,1),(None,None)],method='highs');assert res.success,res.message;u,v,d=res.x;return np.array([1-u-v,u,v]),d
assert deepest(np.array([[-.2,-.2,0],[.2,-.2,0],[0,.2,0]]))[1]>.09
rows=[];barys=[];idslist=[]
for f in range(50,291):
 T=b['pr']['pose_abs'][f];local=(V[f].astype(float)-T[:3,3])@T[:3,:3];ids=candidates(local[F]-center,half-.0001);ww=[];valid=[];depth=[]
 for idx in ids:
  w,d=deepest(local[F[idx]])
  if d>.00005:ww.append(w);valid.append(idx);depth.append(d)
 rows.append({'frame':f,'witness_count':len(valid),'max_interior_depth_mm':float(max(depth,default=0)*1000)});barys.append(np.asarray(ww).reshape(-1,3));idslist.append(np.asarray(valid,int))
M=max(map(len,idslist));indices=np.zeros((241,M,3),int);weights=np.zeros((241,M,3));mask=np.zeros((241,M),bool)
for i,(ids,w) in enumerate(zip(idslist,barys)):
 indices[i,:len(ids)]=F[ids];weights[i,:len(ids)]=w;mask[i,:len(ids)]=True
np.savez_compressed(D/f'surface-witnesses-{args.stage}.npz',frames=np.arange(50,291),vertex_indices=indices,barycentric_weights=weights,mask=mask);report={'stage':args.stage,'synthetic_triangle_interior_fixture_passed':True,'method':'Exact deepest triangle-interior point from linear programming over barycentric coordinates and6mesh planes. Frozen barycentric attachment to real native human triangle, not a displaced output mesh.','total_witnesses':int(mask.sum()),'frames_with_witnesses':int(mask.any(1).sum()),'rows':rows,'seconds':time.time()-start};(D/f'surface-witnesses-{args.stage}.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
