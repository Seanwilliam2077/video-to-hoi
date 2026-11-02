"""Fixed tolerance temporal association audit of cached object-supported matches."""
from pathlib import Path
import json,cv2,numpy as np,hashlib
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1/mast3r-pilot');O=R/'source-interior-correction-v1';P=R/'object-cue-13frames-v1';tol=3*300/512
edges=[];counts=[]
for k,a in enumerate(range(150,210,5)):
 b=a+5
 with np.load(O/f'{2*k:02}_{a}_{b}_object_support.npz') as z:x=z['original_xy1'].copy();y=z['original_xy2'].copy()
 with np.load(O/f'{2*k+1:02}_{b}_{a}_object_support.npz') as z:rx=z['original_xy1'].copy();ry=z['original_xy2'].copy()
 if len(x) and len(rx):
  dist=np.linalg.norm(y[:,None]-rx[None],axis=-1);j=dist.argmin(1);err1=dist[np.arange(len(x)),j];err2=np.linalg.norm(x-ry[j],axis=1);good=(err1<=tol)&(err2<=tol)
 else:good=np.zeros(len(x),bool)
 counts.append({'pair':[a,b],'source_supported':len(x),'roundtrip_supported':int(good.sum())});edges.append((x[good],y[good]))
tracks=[]
for start in range(len(edges[0][0])):
 xy=[edges[0][0][start],edges[0][1][start]];gaps=[]
 for x,y in edges[1:]:
  if not len(x):break
  d=np.linalg.norm(x-xy[-1],axis=1);j=int(d.argmin())
  if d[j]>tol:break
  gaps.append(float(d[j]));xy.append(y[j])
 tracks.append({'id':start,'frames':list(range(150,150+5*len(xy),5)),'xy':np.asarray(xy).tolist(),'association_gaps_px':gaps,'material_identity_certified':False})
full=[t for t in tracks if len(t['xy'])==13]
report={'status':'temporal_association_checked_not_physical_identity','coordinate_tolerance_original_pixels':tol,'tolerance_source':'Three network pixels under the frozen 300/512 inverse mapping, fixed once; no tolerance scan.','edge_counts':counts,'seed_tracks':len(tracks),'full_13_frame_tracks':len(full),'track_lengths':[len(t['xy']) for t in tracks],'new_model_forward_calls':0,'native_pose_changes':False,'interpretation':'Roundtrip and adjacent-pair association can reject broken tracks; repeated-pattern aliases can also be self-consistent, so this does not certify physical feature identity or rotation.'}
(O/'temporal-association-report.json').write_text(json.dumps(report,indent=2));(O/'temporal-tracks.json').write_text(json.dumps(tracks,indent=2))
if full:
 points=np.asarray([t['xy'][0] for t in full]);center=points.mean(0);order=[int(np.argmax(np.linalg.norm(points-center,axis=1)))]
 while len(order)<min(6,len(full)):
  d=np.linalg.norm(points[:,None]-points[order][None],axis=-1).min(1);d[order]=-1;order.append(int(d.argmax()))
 selected=[full[i] for i in order];tiles=[]
 for j,f in enumerate(range(150,211,5)):
  im=cv2.imread(str(P/'inputs'/f'{f:06}.png'));im=cv2.resize(im,(600,600))
  for k,t in enumerate(selected):
   q=np.rint((np.asarray(t['xy'][j])-[590,490])*2).astype(int);color=tuple(map(int,cv2.cvtColor(np.uint8([[[k*28,230,255]]]),cv2.COLOR_HSV2BGR)[0,0]));cv2.circle(im,tuple(q),4,color,1);cv2.putText(im,str(k),tuple(q+[5,-5]),0,.5,color,1)
  bar=np.zeros((28,600,3),np.uint8);cv2.putText(bar,f'Source {f} | temporal candidates, IDs NOT certified',(7,21),0,.45,(255,255,255),1);tiles.append(np.concatenate([bar,im]))
 while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
 sheet=np.concatenate([np.concatenate(tiles[k:k+3],1) for k in range(0,len(tiles),3)],0);cv2.imwrite(str(O/'temporal-source-review.jpg'),sheet)
 (O/'temporal-review-selection.json').write_text(json.dumps({'selected_track_ids':[t['id'] for t in selected],'selection':'Farthest spatial samples, fixed deterministic rule','physical_identity_certified':False},indent=2))
print(json.dumps(report))
