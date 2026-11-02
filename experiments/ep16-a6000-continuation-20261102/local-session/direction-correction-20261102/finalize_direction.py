from pathlib import Path
import json,torch,numpy as np,cv2,imageio.v2 as imageio,hashlib,datetime,shutil
from scipy.spatial.transform import Rotation
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');D=R/'a6000-runs/direction-correction-v1';O=D/'full-scene-review-v6';candidate=D/'source-material-candidate-v6/refined.pth';src=R/'experiments/ep16-joint-direction-20261031/joint-green-v1/refined.pth';a=torch.load(src,map_location='cpu',weights_only=False);b=torch.load(candidate,map_location='cpu',weights_only=False);changed=[k for k,v in a['pr'].items() if not np.array_equal(v,b['pr'][k],equal_nan=True)];assert set(changed)=={'pose_abs','pose_abs_postopt'};assert a['frames']==b['frames'];assert list(map(int,b['frames']))==list(range(360));T=np.asarray(b['pr']['pose_abs']);assert np.array_equal(T[:61],a['pr']['pose_abs'][:61]);assert np.array_equal(T,b['pr']['pose_abs_postopt']);assert np.isfinite(T).all();RR=T[:,:3,:3];orth=float(np.max(np.abs(RR@RR.transpose(0,2,1)-np.eye(3))));assert orth<1e-5
for k,v in a['observations'].items():
 if isinstance(v,np.ndarray):assert np.array_equal(v,b['observations'][k],equal_nan=v.dtype.kind in 'fc'),k
assert a['metadata']['object_mesh']==b['metadata']['object_mesh'];assert hashlib.sha256(src.read_bytes()).hexdigest()=='61decc2bae1b4c5db67c30c7aee521edf61e5c1f5a352e70eec1f03c57234f32'
# Source / preserved old track / new candidate, zoomed to the object, all360frames.
cap=cv2.VideoCapture(str(O/'source-baseline-candidate-360-30fps.mp4'));writer=imageio.get_writer(str(O/'source-old-new-object-zoom-360-30fps.mp4'),format='FFMPEG',fps=30,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_params=['-crf','18','-threads','2']);n=0
while True:
 ok,im=cap.read()
 if not ok:break
 tiles=[]
 for j,title in enumerate(['Original source','Old trajectory','New direction candidate']):
  tile=cv2.resize(im[245:395,295+j*768:445+j*768],(450,450));bar=np.zeros((30,450,3),np.uint8);cv2.putText(bar,f'{title} | frame{n}',(5,21),0,.5,(255,255,255),1);tiles.append(np.concatenate([bar,tile]))
 out=np.concatenate(tiles,1);writer.append_data(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
 if n in [120,165,175,220,240,300]:cv2.imwrite(str(O/f'zoom-{n:03}.jpg'),out)
 n+=1
cap.release();writer.close();assert n==360
now=datetime.datetime.now(datetime.timezone.utc).isoformat();render=json.loads((O/'report.json').read_text());verification={'changed_pr_fields':changed,'all_human_fields_unchanged':True,'observation_arrays_unchanged':True,'shared_camera_mesh_scale_unchanged':True,'frames0_60_exact':True,'original_input_sha_verified':True,'original_source_snapshot_preserved':True,'rotation_orthogonality_max_abs':orth,'rotation_det_min':float(np.linalg.det(RR).min()),'full_scene_decoded_frames':render['decoded_frames'],'zoom_frames':n,'fps':30,'downloads':0};(D/'final-verification.json').write_text(json.dumps(verification,indent=2))
report={'updated_at_utc':now,'phase':'SOURCE_MATERIAL_DIRECTION_REOPTIMIZED_REVIEW_CANDIDATE','accepted_as_final':False,'previous_candidate_user_rejected':'Rotation direction incorrect; never use previous reprojection improvement as acceptance.','candidate':str(candidate),'full_scene_video':str(O/'source-baseline-candidate-360-30fps.mp4'),'zoom_video':str(O/'source-old-new-object-zoom-360-30fps.mp4'),'final_continuous_optimization_steps':500,'physical_gpu':1,'method':'Source face geometry with unordered cube corner observations, estimated visible silhouette, coupled unrestricted object rotation/translation, source material-motion intervals and global4-state discrete phase optimization. No old pose ray bindings,10degree cap or forced final heading return.','source_material_intervals':41,'evidence_scope':'Same-source development matching and approximate manual geometric junctions, not true3D or independent benchmark validation.','verification':verification,'silhouette_diagnostics':render['silhouette_diagnostics'],'limitations':['235..240 is a weaker material branch decision; the selected path remains a proposal and full direction is not certified.','Several asset side faces have uniform brown material. Newly exposed missing texture is not observed video appearance.','Fixed native hands still have contact/projection/penetration errors. Box proxy is not official PEN.','Original baseline and failed candidates retained; no result promoted as accepted or submitted.'],'original_reference':str(src)};(D/'FINAL-DIRECTION-RESULT.json').write_text(json.dumps(report,indent=2))
# Preserve old handoffs before recording this actual continuation state.
for p in [R/'LATEST-OPTIMIZATION.json',R/'LATEST-A6000-HANDOFF.json',R.parent/'ACTIVE_CONTINUATION.json',R/'experiments/ep16-joint-direction-20261031/LATEST-RESEARCH-HANDOFF-20261102.json']:
 if not p.exists():continue
 backup=D/f'previous-{p.name}'
 if not backup.exists():shutil.copy2(p,backup)
 old=json.loads(p.read_text());old.update({'updated_at_utc':now,'latest_direction_result':str(D/'FINAL-DIRECTION-RESULT.json'),'latest_direction_candidate':str(candidate),'direction_candidate_accepted':False,'previous_local_delta_candidate_direction_rejected_by_user':True})
 if p.name=='LATEST-OPTIMIZATION.json':old.update({'result':str(D/'FINAL-DIRECTION-RESULT.json'),'candidate':str(candidate),'accepted_as_final':False})
 if p.name=='LATEST-A6000-HANDOFF.json':old['phase']=report['phase'];old['direction_correction']=report
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(old,indent=2));tmp.replace(p)
print(json.dumps(report),flush=True)
