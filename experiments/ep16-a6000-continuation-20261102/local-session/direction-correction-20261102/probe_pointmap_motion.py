"""Foreground learned-geometry relative camera/object-motion probe, no pose update."""
from pathlib import Path
import json,cv2,numpy as np,torch
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');P=R/'mast3r-pilot';O=R/'a6000-runs/direction-correction-v1';O.mkdir(exist_ok=True)
b=torch.load(R/'experiments/ep16-joint-direction-20261031/joint-green-v1/refined.pth',map_location='cpu',weights_only=False);K=b['observations']['K_full'];T=b['pr']['pose_abs'];regions=json.loads((P/'source-interior-correction-v1/source-box-interiors.json').read_text())['polygons'];rows=[];relative=[]
def inside(x,poly):return np.array([cv2.pointPolygonTest(np.asarray(poly,np.float32),tuple(map(float,p)),False)>=0 for p in x],bool)
def solve(points,pixels,k):
 ok,r,t,inliers=cv2.solvePnPRansac(points.astype(np.float64),pixels.astype(np.float64),np.asarray(k,np.float64),None,iterationsCount=100,reprojectionError=3.,confidence=.999,flags=cv2.SOLVEPNP_EPNP);assert ok
 ids=inliers[:,0];r,t=cv2.solvePnPRefineLM(points[ids].astype(np.float64),pixels[ids].astype(np.float64),np.asarray(k,np.float64),None,r,t)
 rot=cv2.Rodrigues(r)[0];q=points@rot.T+t.reshape(3);uv=q@k.T;uv=uv[:,:2]/uv[:,2:];err=np.linalg.norm(uv-pixels,axis=1)
 return rot,t[:,0],{'count':len(points),'inliers':len(ids),'mean_all_px':float(err.mean()),'median_all_px':float(np.median(err)),'all_z_positive':bool((q[:,2]>0).all())}
for i,a in enumerate(range(150,210,5)):
 z=np.load(P/f'object-cue-13frames-v1/raw/{2*i:02}_{a}_{a+5}_predictions.npz');print(a,z.files,flush=True)
 cam=[]
 for side,f in [(1,a),(2,a+5)]:
  key=f'pred{side}_pts3d' if side==1 else 'pred2_pts3d_in_other_view';xyz=z[key][0];h,w=xyz.shape[:2];yy,xx=np.mgrid[2:h:4,2:w:4];net=np.stack([xx.ravel(),yy.ravel()],1);orig=(net+[0,64]+.5)*(300/512)-.5+[590,490];sel=inside(orig-[590,490],regions[str(f)]);net=net[sel];orig=orig[sel];points=xyz[net[:,1],net[:,0]];good=np.isfinite(points).all(1);points=points[good];orig=orig[good];rot,t,stats=solve(points,orig,K[f]);cam.append((rot,t,stats))
 r1,t1,s1=cam[0];r2,t2,s2=cam[1];RR=r2@r1.T;tt=t2-RR@t1;rb=cv2.Rodrigues(T[a+5,:3,:3]@T[a,:3,:3].T)[0][:,0];rv=cv2.Rodrigues(RR)[0][:,0]
 row={'pair':[a,a+5],'view1':s1,'view2':s2,'learned_relative_rvec_degrees':np.rad2deg(rv).tolist(),'baseline_relative_rvec_degrees':np.rad2deg(rb).tolist(),'rotation_vector_dot':float(rv@rb),'learned_relative_translation':tt.tolist(),'relative_R':RR.tolist()};rows.append(row);relative.append(RR);print(json.dumps(row),flush=True)
(O/'pointmap-relative-motion.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(O/'pointmap-relative-rotations.npz',frames=np.arange(150,211,5),relative_R=np.asarray(relative))
cap=cv2.VideoCapture(str(R/'runs/full-episode16-20261030/inputs/episode_000016.0.color.mp4'));tiles=[]
for f in [0,90,100,110,120,130,140,150,155,160,165,170,175,180,190,200,210,220,240,300,359]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok;im=cv2.resize(im[490:790,590:890],(450,450));bar=np.zeros((26,450,3),np.uint8);cv2.putText(bar,f'Source original frame {f}',(6,19),0,.5,(255,255,255),1);tiles.append(np.concatenate([bar,im]))
cap.release();cv2.imwrite(str(O/'source-direction-events.jpg'),np.concatenate([np.concatenate(tiles[k:k+3],1) for k in range(0,len(tiles),3)],0))
