"""Geometry-only CPU parity on previously inferred Track1 EP16 hand parameters."""
from pathlib import Path
import os,json,time,datetime,traceback,hashlib
ROOT=Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE']='1'
import torch,numpy as np
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from soma.hand import SOMAHandLayer
from soma.units import Unit
reference=Path('J:/video-to-hoi/runs/egohand-pilots-20261031/episode_000016/smoke-cpu-attempt01/raw')
receipt={'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CPU-only geometry backend parity on actual previously inferred hand parameters; no image/backbone or new sequence inference','frame_ids':[0,110,285],'device':'cpu','threads':1,'cases':[],'passed':False}
try:
 for side in ['left','right']:
  path=reference/f'{side}_results.npz'
  with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
  rows=[a['frame_names'].tolist().index(f'{i:06d}') for i in receipt['frame_ids']]
  results={};times={}
  for mode in ['torch','warp']:
   print('Construct',side,mode,flush=True);start=time.monotonic()
   hand=SOMAHandLayer(data_root=ROOT/'weights/soma',hand_type=side,device='cpu',mode=mode,identity_model_type='mhr',output_unit=Unit.METERS,load_correctives_model=False)
   with torch.no_grad():
    r=hand(torch.as_tensor(a['hand_poses'][rows],dtype=torch.float32),torch.as_tensor(a['identity_coeffs'][rows],dtype=torch.float32),pose2rot=False,global_scale=torch.as_tensor(a['global_scale'][rows],dtype=torch.float32),scale_params=torch.as_tensor(a['scale_params'][rows],dtype=torch.float32))
   results[mode]={k:r[k].cpu().numpy().copy() for k in ['joints','vertices']};times[mode]=time.monotonic()-start
   del hand
  metrics={k:{'max_abs_metres':float(np.max(np.abs(results['torch'][k]-results['warp'][k]))),'max_point_norm_metres':float(np.max(np.linalg.norm(results['torch'][k]-results['warp'][k],axis=-1))),'mean_point_norm_metres':float(np.mean(np.linalg.norm(results['torch'][k]-results['warp'][k],axis=-1)))} for k in ['joints','vertices']}
  with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
  case={'side':side,'input_path':str(path),'input_sha256':sha,'metrics':metrics,'construction_and_forward_seconds':times,'finite':all(np.isfinite(x).all() for mode in results.values() for x in mode.values())};receipt['cases'].append(case);print(json.dumps(case),flush=True)
 receipt['passed']=all(c['finite'] and max(m['max_point_norm_metres'] for m in c['metrics'].values())<1e-5 for c in receipt['cases'])
except Exception as error:receipt.update(error=repr(error),traceback=traceback.format_exc());print(receipt['traceback'],flush=True)
(ROOT/'skinning-backend-parity.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({'passed':receipt['passed']}),flush=True)
raise SystemExit(not receipt['passed'])
