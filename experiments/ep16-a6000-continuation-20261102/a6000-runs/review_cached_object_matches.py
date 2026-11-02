"""Correct a source mask semantic error without rerunning the matcher or fitting poses."""
from pathlib import Path
import json,hashlib,cv2,numpy as np
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1/mast3r-pilot');P=R/'object-cue-13frames-v1';O=R/'source-interior-correction-v1';ROI=np.array([590.,490.])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(O/n).write_text(json.dumps(d,indent=2,allow_nan=False))
def inside(x,poly):return np.array([cv2.pointPolygonTest(poly.astype(np.float32),tuple(map(float,p)),False)>=0 for p in x],bool)
def cover(x,poly):
 if len(x)<3:return {'count':len(x),'grid_cells':0,'hull_fraction':0.,'rank':0}
 lo=poly.min(0);hi=poly.max(0);grid=np.clip(((x-lo)/(hi-lo)*3).astype(int),0,2)
 return {'count':len(x),'grid_cells':len(np.unique(grid,axis=0)),'hull_fraction':float(cv2.contourArea(cv2.convexHull(x.astype(np.float32)))/cv2.contourArea(poly.astype(np.float32))),'rank':int(np.linalg.matrix_rank(x-x.mean(0)))}
regions=json.loads((O/'source-box-interiors.json').read_text());polys={int(k):np.array(v,dtype=float)+ROI for k,v in regions['polygons'].items()};rows=[]
write('source-region-review-decision.json',{'source_only_regions_reviewed':13,'decision':'All specified regions lie within the visibly held box and avoid the stationary table; use as positive interior support only.','complete_silhouette':False,'exact_material_point_truth':False,'raw_object_match_sets_inspected_before_region_freeze':False,'region_contract_sha256':sha(O/'source-box-interiors.json'),'new_model_forwards':0})
for k,(a,b) in enumerate([(a,b) for a,b in zip(range(150,210,5),range(155,211,5)) for a,b in [(a,b),(b,a)]]):
 path=P/'raw'/f'{k:02}_{a}_{b}_matches.npz'
 with np.load(path,allow_pickle=False) as z:
  x=z['original_xy1'].copy();y=z['original_xy2'].copy();border=z['passes_border'].copy();old=z['inside_both_eroded_source_masks'].copy()
 good=border&inside(x,polys[a])&inside(y,polys[b]);xv=x[good];yv=y[good]
 c1=cover(xv,polys[a]);c2=cover(yv,polys[b]);spatial=all(c['count']>=12 and c['grid_cells']>=4 and c['hull_fraction']>=.1 and c['rank']==2 for c in [c1,c2])
 np.savez_compressed(O/f'{k:02}_{a}_{b}_object_support.npz',raw_match_indices=np.flatnonzero(good),original_xy1=xv,original_xy2=yv,source_region_contract_sha256=np.array(sha(O/'source-box-interiors.json')))
 selected=[];ids=np.flatnonzero(good)
 for gy in [1/6,.5,5/6]:
  for gx in [1/6,.5,5/6]:
   if not len(ids):continue
   target=polys[a].min(0)+np.array([gx,gy])*(polys[a].max(0)-polys[a].min(0));order=ids[np.argsort(((x[ids]-target)**2).sum(1))];q=next((int(j) for j in order if int(j) not in selected),None)
   if q is not None:selected.append(q)
 im1=cv2.imread(str(P/'inputs'/f'{a:06}.png'));im2=cv2.imread(str(P/'inputs'/f'{b:06}.png'));canvas=np.concatenate([cv2.resize(im1,(600,600)),cv2.resize(im2,(600,600))],1)
 for j,i in enumerate(selected):
  left=np.rint((x[i]-ROI)*2).astype(int);right=np.rint((y[i]-ROI)*2).astype(int)+[600,0];color=tuple(map(int,cv2.cvtColor(np.uint8([[[j*17%180,230,255]]]),cv2.COLOR_HSV2BGR)[0,0]));cv2.line(canvas,tuple(left),tuple(right),color,1);cv2.circle(canvas,tuple(left),4,color,1);cv2.circle(canvas,tuple(right),4,color,1);cv2.putText(canvas,str(j),tuple(left+[4,-4]),0,.4,color,1);cv2.putText(canvas,str(j),tuple(right+[4,-4]),0,.4,color,1)
 bar=np.zeros((32,1200,3),np.uint8);cv2.putText(bar,f'Source {a}->{b} | positive BOX interiors | material identity requires review',(8,23),0,.55,(255,255,255),1);cv2.imwrite(str(O/f'{k:02}_{a}_{b}_review.jpg'),np.concatenate([bar,canvas]))
 def motion(f):return np.median(y[f]-x[f],axis=0).tolist() if f.any() else None
 rows.append({'index':k,'pair':[a,b],'raw_match_file_sha256':sha(path),'old_table_supported_count':int(old.sum()),'corrected_box_interior_count':int(good.sum()),'old_new_overlap':int((old&good).sum()),'coverage1':c1,'coverage2':c2,'fixed_spatial_screen_passed':spatial,'old_table_median_motion_xy':motion(old),'new_box_median_motion_xy':motion(good),'selected_raw_indices':selected,'source_identity_qualified':False})
write('corrected-pair-results.json',rows)
report={'status':'cached_predictions_requalified_for_correct_semantic_region_pending_identity_review','new_model_forwards':0,'original_pilot_forward_calls':24,'original_raw_files_modified':False,'source_region_contract_sha256':sha(O/'source-box-interiors.json'),'old_published_masks':'REJECTED: stationary table, not held box','spatial_screen_passed_pairs':sum(x['fixed_spatial_screen_passed'] for x in rows),'corrected_match_count_range':[min(x['corrected_box_interior_count'] for x in rows),max(x['corrected_box_interior_count'] for x in rows)],'native_parameters_changed':False,'geometry_or_contact_accepted':False};write('report.json',report);print(json.dumps(report))
