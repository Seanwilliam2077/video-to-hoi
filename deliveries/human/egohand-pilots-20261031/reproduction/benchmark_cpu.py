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
cap=cv2.VideoCapture(ep['video']);frames={};i=0
while True:
 ok,f=cap.read()
 if not ok:break
 if i in (0,110,285):frames[f'{i:06d}']=f
 i+=1
cap.release();assert i==360
t=time.monotonic();model,model_report=runner.load_model('cpu');load_seconds=time.monotonic()-t
baseline=None;out=[]
for threads,batch in [(8,1),(8,2),(12,1),(12,2)]:
 torch.set_num_threads(threads);p=EgoHandPredictor(model,batch,device='cpu',dtype=torch.float32,smooth=False,use_perspective_cropping=False)
 outputs={};t=time.monotonic()
 for fid,f in frames.items():
  r=p.run([f],[fid],{fid:{s:boxes[s][fid] for s in ['left','right']}},['left','right'],intrinsics_per_image={fid:K})
  for side in ['left','right']:
   v=r[side][fid];outputs[(side,fid)]={k:np.asarray(v[k]).copy() for k in ['pred_keypoints_3d','transl','pred_keypoints_2d']}
 seconds=time.monotonic()-t
 if baseline is None:baseline=outputs
 finite=all(np.isfinite(a).all() for v in outputs.values() for a in v.values())
 deviations={k:max(float(np.nanmax(np.abs(v[k]-baseline[key][k]))) for key,v in outputs.items()) for k in ['pred_keypoints_3d','transl','pred_keypoints_2d']}
 item={'threads':threads,'batch_size':batch,'source_frames':3,'hand_crops':6,'seconds':seconds,'seconds_per_source_frame':seconds/3,'finite':finite,'max_absolute_difference_vs_8threads_batch1':deviations}
 out.append(item);print(json.dumps(item),flush=True)
receipt={'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_video_sha256':ep['video_sha256'],'frame_ids':[0,110,285],'model_load_seconds':load_seconds,'model_report':model_report,'configurations':out,'selection_rule':'Fastest finite configuration with maximum3D/translation difference <0.0001metres and normalizedcrop2Ddifference <0.0001; no quality-driven selection'}
eligible=[x for x in out if x['finite'] and max(x['max_absolute_difference_vs_8threads_batch1'].values())<0.0001]
receipt['selected']=min(eligible,key=lambda x:x['seconds'])
(ROOT/'cpu-benchmark-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps({'selected':receipt['selected']}),flush=True)
