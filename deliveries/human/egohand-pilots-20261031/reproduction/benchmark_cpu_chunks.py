from pathlib import Path
import json,time,datetime
import run_controlled_egohand as runner
import torch,cv2,numpy as np
from inference.pipeline.predictor import EgoHandPredictor
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(20261031);cv2.setNumThreads(1)
ep=next(x for x in json.loads((ROOT/'inputs/manifest.json').read_text(encoding='utf-8'))['episodes'] if x['episode']==16)
boxes={b['side']:json.loads(Path(b['path']).read_text(encoding='utf-8')) for b in ep['saved_boxes']}
K=np.asarray(json.loads(Path(ep['camera']).read_text(encoding='utf-8'))['camera_matrix']['matrix_3x3'],dtype=np.float32)
ids=[f'{i:06d}' for i in [0,108,109,110,111,112,285,286]]
cap=cv2.VideoCapture(ep['video']);frames={};i=0
while True:
 ok,f=cap.read()
 if not ok:break
 if f'{i:06d}' in ids:frames[f'{i:06d}']=f
 i+=1
cap.release();assert i==360
references={}
for s in ['left','right']:
 with np.load(Path('J:/video-to-hoi/runs/egohand-pilots-20261031/episode_000016/smoke-cpu-attempt01/raw')/f'{s}_results.npz',allow_pickle=False) as f:references[s]={k:f[k] for k in f.files}
t=time.monotonic();model,model_report=runner.load_model('cpu');load_seconds=time.monotonic()-t
out=[]
for threads,batch in [(12,4),(12,8),(12,16)]:
 torch.set_num_threads(threads);p=EgoHandPredictor(model,batch,device='cpu',dtype=torch.float32,smooth=False,use_perspective_cropping=False)
 t=time.monotonic();r=p.run([frames[f] for f in ids],ids,{f:{s:boxes[s][f] for s in boxes} for f in ids},['left','right'],intrinsics_per_image={f:K for f in ids});seconds=time.monotonic()-t
 deviations={k:0.0 for k in ['pred_keypoints_3d','transl','pred_keypoints_2d']};finite=True
 for side in ['left','right']:
  reference=references[side]
  for fid,result in r[side].items():
   row=reference['frame_names'].tolist().index(fid)
   for key,refkey in [('pred_keypoints_3d','joints'),('transl','cam_t'),('pred_keypoints_2d','pred_keypoints_2d')]:
    value=np.asarray(result[key]);finite &= bool(np.isfinite(value).all());deviations[key]=max(deviations[key],float(np.nanmax(np.abs(value-reference[refkey][row]))))
 item={'threads':threads,'batch_size':batch,'source_chunk_frames':len(ids),'source_frames':len(ids),'hand_crops':len(ids)*2,'seconds':seconds,'seconds_per_source_frame':seconds/len(ids),'finite':finite,'max_absolute_difference_vs_frozen_8threads_batch1_smoke':deviations};out.append(item);print(json.dumps(item),flush=True)
receipt={'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_video_sha256':ep['video_sha256'],'frame_ids':ids,'model_load_seconds':load_seconds,'configurations':out,'selection_rule':'Fastest finite configuration with max3D/translationdifference <0.0001m and normalizedcrop2Ddifference <0.0001. No quality-driven selection.'}
eligible=[x for x in out if x['finite'] and max(x['max_absolute_difference_vs_frozen_8threads_batch1_smoke'].values())<0.0001]
receipt['selected']=min(eligible,key=lambda x:x['seconds'])
(ROOT/'cpu-chunk-benchmark-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({'selected':receipt['selected']}),flush=True)
