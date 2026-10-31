"""Audit completed cache02 inference, exact cache-hit arrays and fresh-view coverage."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
T=Path(__file__).resolve().parent
R=Path('J:/video-to-hoi/runs/egohand-refined-pilots-20261031')
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=int,choices=(16,11),required=True);a=p.parse_args();n={16:360,11:877}[a.episode]
 root=R/f'episode_{a.episode:06d}'/'inference-attempt02';rp=root/'prediction-receipt.json';r=read(rp)
 assert r['status']=='completed' and r['frames']==r['source_frames']==n and r['complete_source_coverage']
 assert r['selected_original_frame_ids']==list(range(n)) and r['raw_cache_enabled'] and not r['all_original_hand_views_neurally_recomputed']
 assert r['official_smoothing_recomputed_over_full_merged_raw_clip'] and not r['raw_cache_lineage']['smoothed_cache_reused']
 assert r['runner_sha256']=='91729499f3454489e9611e4a51498e8c4886d1421acf6df2f367ac53533c4ec4'
 assert r['protocol_sha256']=='eeabb07a21a77a9a96c71e9a7adb4e753c96df24d66819e57340df8ca677a34d'
 assert r['input_manifest_sha256']=='449311c5c03fa8aacd0b7c14e54b4a9c812edf0e7bf169894eb046fa4e53162f'
 old=read(r['raw_cache_lineage']['receipt']);assert sha(r['raw_cache_lineage']['receipt'])==r['raw_cache_lineage']['receipt_sha256']
 ib=r['input_bindings'];boxes=np.load(ib['crop_boxes'],allow_pickle=False);assert sha(ib['crop_boxes'])==ib['crop_boxes_sha256']
 with np.load(old['input_bindings']['original_observations'],allow_pickle=False) as o:oldboxes=o['crop_boxes']
 mask=np.all(boxes==oldboxes,axis=-1)
 K=np.asarray(read(ib['camera'])['camera_matrix']['matrix_3x3'],dtype=np.float64);checks=[];cache_checks=[]
 chunkids={'left':[],'right':[]}
 for chunk in r['chunks']:
  for item in chunk['outputs']:
   assert sha(item['path'])==item['sha256']
   with np.load(item['path'],allow_pickle=False) as z:ids=[int(x) for x in z['frame_names'].tolist()]
   assert len(ids)==item['frames'] and all(i in r['neural_recomputed_original_frame_ids_per_side'][item['side']] for i in ids)
   chunkids[item['side']]+=ids
 for s,side in enumerate(['left','right']):
  hits=np.flatnonzero(mask[:,s]).tolist();fresh=np.flatnonzero(~mask[:,s]).tolist()
  assert hits==r['cache_hit_original_frame_ids_per_side'][side] and fresh==r['neural_recomputed_original_frame_ids_per_side'][side]
  assert len(chunkids[side])==len(set(chunkids[side])) and sorted(chunkids[side])==fresh
  assert len(hits)==r['cache_hits_per_side'][side] and len(fresh)==r['neural_recomputed_views_per_side'][side]
  item=next(x for x in r['raw_outputs'] if x['side']==side);original=next(x for x in old['raw_outputs'] if x['side']==side)
  assert sha(original['path'])==original['sha256']
  with np.load(item['path'],allow_pickle=False) as z,np.load(original['path'],allow_pickle=False) as o:
   assert set(z.files)==set(o.files)
   compared=[]
   for key in z.files:
    if key in ['num_frames','frame_names']:continue
    assert np.array_equal(z[key][hits],o[key][hits],equal_nan=True),key;compared.append(key)
  cache_checks.append({'side':side,'exact_raw_cache_hit_count':len(hits),'newly_inferred_view_count':len(fresh),'all_cached_parameter_geometry_projection_arrays_exact':True,'compared_fields':compared,'fresh_chunk_coverage_exact':True})
 for item in r['raw_outputs']+r['smoothed_outputs']:
  assert sha(item['path'])==item['sha256'] and item['frames']==n and item['complete_source_coverage']
  with np.load(item['path'],allow_pickle=False) as z:
   assert z['frame_names'].tolist()==[f'{i:06d}' for i in range(n)] and int(z['num_frames'])==n
   for key in ['hand_poses','identity_coeffs','global_scale','vertices','joints','cam_t','scale_params']:assert np.isfinite(z[key]).all(),key
   pts=z['joints'].astype(np.float64)+z['cam_t'].astype(np.float64)[:,None,:];pix=pts@K.T;pix=pix[...,:2]/pix[...,2:]
   valid=z['pred_projection_valid'].astype(bool);expected=np.isfinite(pts).all(-1)&(pts[...,2]>1e-4);assert np.array_equal(valid,expected)
   side=['left','right'].index(item['side']);b=boxes[:,side];S=b[:,2]-b[:,0];c=(b[:,:2]+b[:,2:])/2
   saved=z['pred_keypoints_2d'].astype(np.float64)*S[:,None,None]+c[:,None,:]
   err=float(np.linalg.norm(pix[valid]-saved[valid],axis=-1).max()) if valid.any() else None
   assert err is None or err<1e-3
   checks.append({'variant':item['variant'],'side':item['side'],'frames':n,'sha256':item['sha256'],'finite_3d_geometry':True,'invalid_2d_projection_count':int((~valid).sum()),'valid_projection_roundtrip_max_pixels':err})
 report={'recorded_at':datetime.now(timezone.utc).isoformat(),'track':1,'episode':a.episode,'passed':True,'source_video_sha256':r['source_video_sha256'],'prediction_receipt_sha256':sha(rp),'raw_cache_checks':cache_checks,'output_checks':checks,'cache_hits':sum(r['cache_hits_per_side'].values()),'newly_inferred_hand_views':sum(r['neural_recomputed_views_per_side'].values()),'smoothed_cache_reused':False,'official_smoothing_recomputed_full_clip':True,'quality_or_contact_accepted':False,'official_scores_computed':False,'all30_execution_started':False}
 out=T/f'ep{a.episode}-inference-integrity02.json'
 with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
