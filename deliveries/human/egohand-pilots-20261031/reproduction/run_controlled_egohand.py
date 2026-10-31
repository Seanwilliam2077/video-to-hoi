"""EP16/EP11-only source-pinned EgoHand runner with immutable controlled inputs."""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['PYOPENGL_PLATFORM']='win32'
os.environ['PYTHONUNBUFFERED']='1'
def digest(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_model(device):
 import gc,torch
 from omegaconf import OmegaConf
 import egohand.models.egohand as module
 checkpoint=ROOT/'weights/egohand/egohand_s060000.ckpt'
 raw=torch.load(str(checkpoint),map_location='cpu',weights_only=False,mmap=True)
 cfg=OmegaConf.create(raw['hyper_parameters']['cfg'])
 cfg.backbone_weights=None;cfg.soma_hand.data_root=str(ROOT/'weights/soma')
 factory=module.SomaHandWrapper
 # Upstream constructor hardcodes CUDA even for CPU inference. Device/backend
 # selection changes no learned parameters, crop or model prediction equations.
 if device=='cpu':
  def cpu_factory(*args,**kwargs):
   kwargs['device']='cpu';kwargs['mode']='torch'
   return factory(*args,**kwargs)
  module.SomaHandWrapper=cpu_factory
 try:model=module.EgoHand(cfg)
 finally:module.SomaHandWrapper=factory
 model.validate_camera_depth_checkpoint(raw)
 incompatible=model.load_state_dict(raw['state_dict'],strict=False)
 missing=list(incompatible.missing_keys);unexpected=list(incompatible.unexpected_keys)
 missing_learned=[k for k in missing if not k.startswith('soma_hand.')]
 assert not missing_learned and not unexpected,(missing_learned,unexpected)
 report={'state_keys':len(raw['state_dict']),'missing_geometry_keys':missing,'missing_learned_keys':missing_learned,'unexpected_keys':unexpected,'camera_mode':getattr(model,'camera_mode',None),'soma_hand_identity':getattr(model,'soma_hand_identity',None),'checkpoint_cfg':OmegaConf.to_container(cfg,resolve=True),'construction_device':device,'CPU_compatibility_factory_override':device=='cpu','torch_load_mmap':True}
 del raw;gc.collect();return model.eval(),report
def export(results,path):
 import numpy as np
 path.mkdir(parents=True,exist_ok=False)
 outputs=[]
 for side in ['left','right']:
  hand=results[side];ids=sorted(hand,key=int);values=[hand[i] for i in ids]
  arrays={'num_frames':np.asarray(len(ids)),'frame_names':np.asarray(ids)}
  mapping={'hand_poses':'poses','identity_coeffs':'identity_coeffs','global_scale':'global_scale','vertices':'pred_vertices','joints':'pred_keypoints_3d','cam_t':'transl','scale_params':'scale_params','pred_kp2d_crop_presence':'pred_kp2d_crop_presence','pred_kp2d_sigma':'pred_kp2d_sigma','pred_keypoints_2d':'pred_keypoints_2d','pred_projection_valid':'pred_projection_valid','source_bbox':'bbox'}
  for outkey,inkey in mapping.items():
   present=[v.get(inkey) for v in values]
   if present and all(v is not None for v in present):arrays[outkey]=np.stack(present)
  out=path/f'{side}_results.npz';np.savez(out,**arrays)
  outputs.append({'side':side,'path':str(out),'sha256':digest(out),'frames':len(ids),'fields':{k:list(v.shape) for k,v in arrays.items()},'finite_camera_geometry':bool(np.isfinite(arrays['joints']).all() and np.isfinite(arrays['cam_t']).all())})
 return outputs
def main():
 import cv2,numpy as np,torch
 parser=argparse.ArgumentParser();parser.add_argument('--episode',type=int,choices=[16,11],required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--dtype',choices=['float32','bfloat16','float16'],default='float32');parser.add_argument('--threads',type=int,default=8);parser.add_argument('--batch-size',type=int,default=1);parser.add_argument('--frame-chunk-size',type=int,default=8);parser.add_argument('--frame-ids',type=str);parser.add_argument('--with-smoothing',action='store_true')
 args=parser.parse_args();assert not args.output.exists();args.output.mkdir(parents=True)
 torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.manual_seed(20261031);cv2.setNumThreads(1)
 assert 1<=args.frame_chunk_size<=64
 manifest=json.loads((ROOT/'inputs/manifest.json').read_text(encoding='utf-8'));ep=next(e for e in manifest['episodes'] if e['episode']==args.episode)
 for p,key in [('video','video_sha256'),('camera','camera_sha256'),('original_observations','original_observations_sha256')]:assert digest(ep[p])==ep[key]
 for box in ep['saved_boxes']:assert digest(box['path'])==box['sha256']
 boxes={b['side']:json.loads(Path(b['path']).read_text(encoding='utf-8')) for b in ep['saved_boxes']}
 K=np.asarray(json.loads(Path(ep['camera']).read_text(encoding='utf-8'))['camera_matrix']['matrix_3x3'],dtype=np.float32)
 selected=set(map(int,args.frame_ids.split(','))) if args.frame_ids else set(range(ep['frames']))
 assert selected and min(selected)>=0 and max(selected)<ep['frames']
 receipt={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'episode':args.episode,'source_video_sha256':ep['video_sha256'],'source_frames':ep['frames'],'frames':len(selected),'complete_source_coverage':len(selected)==ep['frames'],'selected_original_frame_ids':sorted(selected),'same_original_camera':True,'camera_estimated_not_calibrated':True,'same_crop_bounds':True,'perspective_cropping':False,'pixel_aspect_rectification':False,'box_expansion':False,'upstream_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'upstream-git',text=True).strip(),'runner_sha256':digest(__file__),'checkpoint_sha256':digest(ROOT/'weights/egohand/egohand_s060000.ckpt'),'device':args.device,'dtype':args.dtype,'threads':args.threads,'batch_size':args.batch_size,'frame_chunk_size':args.frame_chunk_size,'all30_execution_started':False,'status':'running','input_bindings':ep}
 if args.device=='cuda':
  torch.cuda.reset_peak_memory_stats();receipt['cuda_free_total_before']=list(torch.cuda.mem_get_info())
 (args.output/'started.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
 begin=time.monotonic();model,load_report=load_model(args.device);receipt['model_load']=load_report;receipt['model_load_seconds']=time.monotonic()-begin
 print(json.dumps({'model_loaded':True,'seconds':receipt['model_load_seconds'],'camera_mode':load_report['camera_mode'],'identity':load_report['soma_hand_identity']}),flush=True)
 from inference.pipeline.predictor import EgoHandPredictor
 predictor=EgoHandPredictor(model,args.batch_size,device=args.device,dtype=getattr(torch,args.dtype),smooth=False,use_perspective_cropping=False)
 cap=cv2.VideoCapture(ep['video']);assert cap.isOpened();results={'left':{},'right':{}};row=0;inference=0.0;chunk_frames=[];chunk_ids=[];chunks=[]
 def flush_chunk():
  nonlocal inference
  if not chunk_ids:return
  bboxes={i:{s:boxes[s][i] for s in ['left','right']} for i in chunk_ids}
  start=time.monotonic();r=predictor.run(chunk_frames,chunk_ids,bboxes,['left','right'],intrinsics_per_image={i:K for i in chunk_ids});elapsed=time.monotonic()-start;inference+=elapsed
  for side in results:results[side].update(r[side])
  saved=export(r,args.output/'chunks'/f'chunk-{len(chunks):04d}')
  chunks.append({'original_frame_ids':list(chunk_ids),'inference_seconds':elapsed,'outputs':saved})
  progress={'last_original_frame_id':int(chunk_ids[-1]),'prediction_frames_per_side':{s:len(v) for s,v in results.items()},'inference_seconds':inference,'wall_seconds':time.monotonic()-begin,'last_chunk_seconds':elapsed,'last_chunk_frames':len(chunk_ids),'chunks':chunks}
  (args.output/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
  print(json.dumps({k:v for k,v in progress.items() if k!='chunks'}),flush=True)
  chunk_frames.clear();chunk_ids.clear()
 while True:
  ok,frame=cap.read()
  if not ok:break
  if row in selected:
   chunk_frames.append(frame);chunk_ids.append(f'{row:06d}')
   if len(chunk_ids)>=args.frame_chunk_size:flush_chunk()
  row+=1
 cap.release();flush_chunk();assert row==ep['frames'],(row,ep['frames'])
 assert all(set(v)=={f'{i:06d}' for i in selected} for v in results.values())
 receipt['raw_outputs']=export(results,args.output/'raw');receipt['inference_seconds']=inference;receipt['decoded_source_frames']=row;receipt['chunks']=chunks
 for out in receipt['raw_outputs']:out['variant']='raw';out['complete_source_coverage']=len(selected)==ep['frames']
 receipt['wrist_relative_joint0_max_norm_metres']={s:max(float(np.linalg.norm(r['pred_keypoints_3d'][0])) for r in hand.values()) for s,hand in results.items()}
 assert max(receipt['wrist_relative_joint0_max_norm_metres'].values())<1e-4,'Unexpected nonlocal output joints; do not double-add camera translation'
 if args.with_smoothing:
  # Rebuild geometry from smoothed parameters, never average mesh vertices.
  smoothed=copy.deepcopy(results);ids=[f'{i:06d}' for i in sorted(selected)];bb={i:{s:boxes[s][i] for s in boxes} for i in ids};ft={i:int(i)/ep['fps'] for i in ids}
  predictor.smooth=True;t=time.monotonic();predictor.postprocess_results(smoothed,ids,bb,frame_times=ft,intrinsics_per_image={i:K for i in ids});receipt['smoothing_seconds']=time.monotonic()-t
  receipt['smoothed_outputs']=export(smoothed,args.output/'smoothed')
  for out in receipt['smoothed_outputs']:out['variant']='upstream_adaptive_smoothed';out['complete_source_coverage']=len(selected)==ep['frames']
 if args.device=='cuda':receipt['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated();receipt['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
 import psutil
 info=psutil.Process().memory_info();receipt['process_memory']={k:getattr(info,k) for k in ['rss','vms','peak_wset'] if hasattr(info,k)}
 receipt.update(status='completed',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-begin)
 (args.output/'prediction-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({'completed':True,'episode':args.episode,'wall_seconds':receipt['wall_seconds']}),flush=True)
if __name__=='__main__':main()
