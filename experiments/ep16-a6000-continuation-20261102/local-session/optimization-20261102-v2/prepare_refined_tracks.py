"""Source-anchored, frame-by-frame short tracks with fixed template and mask checks."""
from pathlib import Path
import json,hashlib,time,cv2,numpy as np,torch
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');O=R/'a6000-runs/matching-optimization-v2';P=R/'mast3r-pilot';ROI=np.array([590,490]);FRAMES=np.arange(150,211)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,d):(O/n).write_text(json.dumps(d,indent=2))
def ncc(a,b):
 a=a.astype(float);b=b.astype(float);a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
 return float((a*b).sum()/den) if den>1e-8 else -1.
def main():
 O.mkdir(exist_ok=False);start=time.time();cv2.setNumThreads(2)
 protocol={'authorization':'Current user explicitly requested fixing unreliable matching and running optimization on A6000, not stopping at analysis.','source':'Track 1 EP16 only','window_original_frames':[150,210],'source_roi_xyxy':[590,490,890,790],'anchor_frames':[150,165,180,195],'tracking':'One source frame per step; points seeded by cached MASt3R object candidates plus image corners within independently source-reviewed visible-box interiors.','mask':'Current Native full-resolution object mask mapping via K_full @ inverse(K_crop), not rejected published table masks.','checks':{'LK_window':15,'LK_levels':2,'LK_FB_max_original_pixels':.75,'anchor_template_size':11,'anchor_template_ZNCC_min':.7,'minimum_source_mask_erosion':2,'no_reactivation':True},'difference_from_old_rejected_LK':'New short 15-frame tracks, source-reviewed anchor interiors, correct moving-object mask and direct anchor-template test at every source frame. No old 120-210 track IDs are reused.','important_limit':'These are source-image observation proposals, not true physical material-point labels. The subsequent pose candidate remains conditional and unaccepted.','fit_validation_split':'Even original frames fit; odd frames are temporal development checks, not unseen-video or 3D ground truth.','pose_budget':{'one_candidate':True,'steps':300,'rotation_delta_norm_cap_degrees':10,'translation_delta_norm_cap_m':.03,'human_camera_mesh_scale_frozen':True,'frame_150_210_and_outside_window_exactly_preserved':True},'downloads':'None; all existing checkpoint/source bytes reused. Any later downloads must use proxy or mirror.'}
 write('protocol.json',protocol)
 src=R/'runs/full-episode16-20261030/inputs/episode_000016.0.color.mp4';assert sha(src)=='a37a978ef0f3d0eeaad74f2ed75fb32502d8a1a047587291c14433d052f9eb74'
 inp=R/'experiments/ep16-joint-direction-20261031/joint-green-v1/refined.pth';assert sha(inp)=='61decc2bae1b4c5db67c30c7aee521edf61e5c1f5a352e70eec1f03c57234f32'
 b=torch.load(inp,map_location='cpu',weights_only=False);obs=b['observations'];cap=cv2.VideoCapture(str(src));cap.set(cv2.CAP_PROP_POS_FRAMES,150);imgs=[];gray=[];masks=[]
 shift=np.array([[1,0,-590],[0,1,-490],[0,0,1.]],float)
 for t in FRAMES:
  ok,im=cap.read();assert ok;crop=im[490:790,590:890].copy();imgs.append(crop);gray.append(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY))
  H=shift@obs['K_full'][t]@np.linalg.inv(obs['postopt_K_rois'][t]);mask=cv2.warpPerspective(np.uint8(obs['postopt_object_mask'][t]>0)*255,H,(300,300),flags=cv2.INTER_NEAREST);masks.append(cv2.erode(mask,np.ones((5,5),np.uint8)))
 cap.release();regions=json.loads((P/'source-interior-correction-v1/source-box-interiors.json').read_text())['polygons'];groups=[];summary=[]
 for group,anchor in enumerate([150,165,180,195]):
  lo=anchor-150;hi=min(lo+15,60);core=np.zeros((300,300),np.uint8);cv2.fillPoly(core,[np.asarray(regions[str(anchor)],np.int32)],255);core=cv2.bitwise_and(core,masks[lo]);core=cv2.erode(core,np.ones((3,3),np.uint8))
  points=[]
  idx=2*((anchor-150)//5)
  with np.load(P/f'object-cue-13frames-v1/raw/{idx:02}_{anchor}_{anchor+5}_matches.npz') as z:raw=z['original_xy1']-ROI
  for q in raw:
   x,y=np.rint(q).astype(int)
   if 6<=x<294 and 6<=y<294 and core[y,x] and all(np.linalg.norm(q-p)>2.5 for p in points):points.append(q)
  corners=cv2.goodFeaturesToTrack(gray[lo],maxCorners=120,qualityLevel=.01,minDistance=3,mask=core,blockSize=5)
  for q in ([] if corners is None else corners[:,0]):
   if all(np.linalg.norm(q-p)>2.5 for p in points):points.append(q)
  points=np.asarray(points,np.float32);assert len(points)>=8,(anchor,len(points));N=len(points);L=hi-lo+1;xy=np.full((L,N,2),np.nan,np.float32);valid=np.zeros((L,N),bool);fb=np.full((L,N),np.nan);scores=np.full((L,N),np.nan);reason=np.zeros((L,N),np.int32);xy[0]=points;valid[0]=True;fb[0]=0;scores[0]=1
  templates=[cv2.getRectSubPix(gray[lo],(11,11),tuple(p)) for p in points]
  for j in range(1,L):
   old=xy[j-1].copy();old[~np.isfinite(old).all(1)]=points[~np.isfinite(old).all(1)];p1,st,err=cv2.calcOpticalFlowPyrLK(gray[lo+j-1],gray[lo+j],old[:,None],None,winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
   p0,back,err2=cv2.calcOpticalFlowPyrLK(gray[lo+j],gray[lo+j-1],p1,None,winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=1e-4)
   q=p1[:,0];xy[j]=q;fb[j]=np.linalg.norm(p0[:,0]-old,axis=1);inside=(q[:,0]>=6)&(q[:,0]<294)&(q[:,1]>=6)&(q[:,1]<294)
   maskgood=np.zeros(N,bool)
   for k,p in enumerate(q):
    if inside[k]:
     x,y=np.rint(p).astype(int);maskgood[k]=masks[lo+j][y,x]>0;scores[j,k]=ncc(templates[k],cv2.getRectSubPix(gray[lo+j],(11,11),tuple(p)))
   flowgood=st[:,0].astype(bool)&back[:,0].astype(bool)&(fb[j]<=.75)&np.isfinite(q).all(1);photo=scores[j]>=.7
   valid[j]=valid[j-1]&flowgood&inside&maskgood&photo;reason[j]=(~flowgood).astype(int)+2*(~inside)+4*(~maskgood)+8*(~photo)+16*(~valid[j-1])
  groups.append({'anchor':anchor,'frames':np.arange(anchor,anchor+L),'xy':xy+ROI,'valid':valid,'fb':fb,'ncc':scores,'reason':reason})
  np.savez_compressed(O/f'group-{anchor}-tracks.npz',frames=np.arange(anchor,anchor+L),xy=xy+ROI,valid=valid,fb=fb,ncc=scores,rejection_bits=reason)
  full=valid.all(0);summary.append({'anchor':anchor,'seed_points':N,'full_short_tracks':int(full.sum()),'per_frame_survivors':valid.sum(1).tolist(),'track_observations':int(valid.sum())})
  tiles=[]
  for j in [0,5,10,15]:
   if j>=L:continue
   im=cv2.resize(imgs[lo+j],(600,600));ids=np.flatnonzero(full)
   for kk,k in enumerate(ids):
    p=np.rint(xy[j,k]*2).astype(int);color=(int(40+kk*61%215),int(40+kk*79%215),int(40+kk*103%215));cv2.circle(im,tuple(p),3,color,1)
   bar=np.zeros((28,600,3),np.uint8);cv2.putText(bar,f'Anchor {anchor} | source {anchor+j} | {len(ids)} short tracks',(8,20),0,.5,(255,255,255),1);tiles.append(np.concatenate([bar,im]))
  cv2.imwrite(str(O/f'group-{anchor}-source-review.jpg'),np.concatenate(tiles,1));print(summary[-1],flush=True)
 write('matching-report.json',{'status':'framewise_template_checked_tracks_ready','groups':summary,'seconds':time.time()-start,'new_neural_inference_calls':0,'native_parameters_changed':False,'source_sha256':sha(src),'input_bundle_sha256':sha(inp),'limits':'Short-track survival is not physical identity truth. Pose fitting uses explicit surface-binding assumptions and full-scene review.'})
if __name__=='__main__':main()
