"""Fixed 13-frame object-observation pilot; retains raw predictions and all misses."""
from pathlib import Path
import os,sys,json,time,hashlib,traceback,subprocess,signal
import cv2,numpy as np,torch
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');T=R/'mast3r-pilot';O=T/'object-cue-13frames-v1'
sys.path.insert(0,str(T/'code'))
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images
FRAMES=list(range(150,211,5));ROI=(590,490,890,790);EXPECTED='0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,d):(O/n).write_text(json.dumps(d,indent=2,allow_nan=False))
def coverage(x,mask):
 ys,xs=np.nonzero(mask)
 if len(x)<3 or not len(xs):return {'count':len(x),'grid_cells':0,'hull_fraction':0.,'rank':0}
 box=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],float);cells=np.clip(((x-box[:2])/(box[2:]-box[:2])*3).astype(int),0,2);area=float(cv2.contourArea(cv2.convexHull(x.astype(np.float32))))
 return {'count':len(x),'grid_cells':len(np.unique(cells,axis=0)),'hull_fraction':area/float(mask.sum()),'rank':int(np.linalg.matrix_rank(x-x.mean(0))),'bbox_xyxy':box.tolist()}
def draw_review(a,b,x,y,good,frame1,frame2):
 indices=np.flatnonzero(good);chosen=[]
 if len(indices):
  for gy in [1/6,.5,5/6]:
   for gx in [1/6,.5,5/6]:
    z=x[indices];lo=z.min(0);hi=z.max(0);target=lo+np.array([gx,gy])*(hi-lo)
    order=indices[np.argsort(((z-target)**2).sum(1))]
    pick=next((int(q) for q in order if int(q) not in chosen),None)
    if pick is not None:chosen.append(pick)
 a=a.copy();b=b.copy();a=cv2.resize(a,(600,600));b=cv2.resize(b,(600,600));im=np.concatenate([a,b],1)
 for k,i in enumerate(chosen):
  p=np.rint((x[i]-ROI[:2])*2).astype(int);q=np.rint((y[i]-ROI[:2])*2).astype(int)+[600,0];col=tuple(int(v) for v in cv2.cvtColor(np.uint8([[[k*17%180,230,255]]]),cv2.COLOR_HSV2RGB)[0,0]);cv2.line(im,tuple(p),tuple(q),col,1);cv2.circle(im,tuple(p),4,col,1);cv2.circle(im,tuple(q),4,col,1)
 bar=np.zeros((32,1200,3),np.uint8);cv2.putText(bar,f'Source {frame1} -> {frame2} | spatial samples; correspondence identity NOT certified',(8,23),0,.55,(255,255,255),1)
 return np.concatenate([bar,im]),chosen
def gpu_deadline(signum,frame):
 raise TimeoutError('Fixed 30-minute GPU wall-time budget reached')
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
 O.mkdir(exist_ok=False);(O/'raw').mkdir();(O/'inputs').mkdir();(O/'reviews').mkdir()
 protocol={'host':'a6000dove','physical_gpu':2,'user_authorization':'User approved migration and continuation on A6000 after the concrete MASt3R proposal; protocol host changed from 1080 Ti to A6000.','frames':FRAMES,'ordered_pairs':[(a,b) for a,b in zip(FRAMES[:-1],FRAMES[1:]) for a,b in [(a,b),(b,a)]],'max_forward_calls':24,'max_gpu_wall_seconds':1800,'first_forward_included':True,'batch_size':1,'precision':'FP32, no AMP or precision fallback','ROI_xyxy':ROI,'author_loader':'load_images(size=512, square_ok=False), 300x300 -> 512x512 -> central 512x384','inverse_pixel_center_mapping':'original_xy=(network_xy+[0,64]+0.5)*300/512-0.5+[590,490]','matcher':{'subsample_or_initxy1':8,'dist':'dot','block_size':8192,'border':3},'fixed_spatial_screen':{'minimum_mask_supported_matches':12,'minimum_3x3_bbox_grid_cells':4,'minimum_hull_fraction_of_mask':.1,'required_rank':2},'qualification':'Spatial screen is necessary only. Source patch identity and repeat-pattern/occlusion ambiguity require visual review. Masks, network confidence and cycle consistency are not ground truth.','excluded_actions':['training','other model downloads','native trajectory update','camera/scale changes','threshold/crop/budget search','precision fallback']}
 write('protocol.json',protocol);write('state.json',{'status':'verifying_inputs'})
 source=R/'runs/full-episode16-20261030/inputs/episode_000016.0.color.mp4';assert sha(source)=='a37a978ef0f3d0eeaad74f2ed75fb32502d8a1a047587291c14433d052f9eb74'
 model_path=T/'model/model.safetensors';assert sha(model_path)==EXPECTED
 mask_root=T/'source-masks';manifest=json.loads((mask_root/'masks-manifest.json').read_text());assert manifest['video_sha256']==sha(source) and manifest['frame_count']==360;maskrows={v['path']:v for v in manifest['files']}
 torch.set_num_threads(4);cv2.setNumThreads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False
 cap=cv2.VideoCapture(str(source));assert (int(cap.get(7)),int(cap.get(3)),int(cap.get(4)))==(360,1536,1152);assert cap.get(5)==30
 crops={};masks={};bindings=[]
 for t in FRAMES:
  cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read();assert ok and int(cap.get(cv2.CAP_PROP_POS_FRAMES))==t+1
  name=f'object_masks/{t:06}.png';p=mask_root/name;assert sha(p)==maskrows[name]['sha256'];mask=cv2.imread(str(p),0);assert mask.shape==(1152,1536)
  masks[t]=cv2.erode((mask>0).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
  crop=cv2.cvtColor(im[490:790,590:890],cv2.COLOR_BGR2RGB);crops[t]=crop;dest=O/'inputs'/f'{t:06}.png';assert cv2.imwrite(str(dest),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR));bindings.append({'frame':t,'original_mask_sha256':sha(p),'crop_sha256':sha(dest),'source_sha256':sha(source),'mask_pixels':int((mask>0).sum()),'mask_not_ground_truth':True})
 cap.release();write('input-bindings.json',bindings)
 images=load_images([str(O/'inputs'/f'{t:06}.png') for t in FRAMES],size=512,square_ok=False,verbose=False);assert all(tuple(x['true_shape'][0])==(384,512) for x in images)
 # The raw checkpoint is unchanged; author model loader uses only the local directory.
 model=AsymmetricMASt3R.from_pretrained(str(T/'model'),local_files_only=True).eval();assert all(p.dtype==torch.float32 for p in model.parameters())
 used=int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());assert used<500,used
 apps=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();assert not apps,apps
 gpu_start=time.monotonic();signal.signal(signal.SIGALRM,gpu_deadline);signal.alarm(1800);model=model.to('cuda');rows=[]
 for k,(f1,f2) in enumerate(protocol['ordered_pairs']):
  assert k<24 and time.monotonic()-gpu_start<1800,'Fixed inference budget reached'
  i,j=FRAMES.index(f1),FRAMES.index(f2);t0=time.monotonic();write('state.json',{'status':'running_fixed_pairs','completed_forward_calls':k,'attempted_forward_calls':k+1,'pair':[f1,f2]})
  with torch.no_grad():out=inference([(images[i],images[j])],model,'cuda',batch_size=1,verbose=False)
  torch.cuda.synchronize();raw={}
  for side in ['pred1','pred2']:
   for name,value in out[side].items():
    if isinstance(value,torch.Tensor):raw[side+'_'+name]=value.detach().cpu().numpy()
  # Persist each forward before matching; failed/missing predictions are not dropped.
  np.savez_compressed(O/'raw'/f'{k:02}_{f1}_{f2}_predictions.npz',**raw)
  assert all(np.isfinite(value).all() for value in raw.values()),'Nonfinite raw model output; preserved before stopping'
  d1=out['pred1']['desc'].squeeze(0).detach();d2=out['pred2']['desc'].squeeze(0).detach()
  xnet,ynet=fast_reciprocal_NNs(d1,d2,subsample_or_initxy1=8,device='cuda',dist='dot',block_size=8192);torch.cuda.synchronize()
  x=(xnet.astype(float)+[0,64]+.5)*(300/512)-.5+ROI[:2];y=(ynet.astype(float)+[0,64]+.5)*(300/512)-.5+ROI[:2]
  border=((xnet[:,0]>=3)&(xnet[:,0]<509)&(xnet[:,1]>=3)&(xnet[:,1]<381)&(ynet[:,0]>=3)&(ynet[:,0]<509)&(ynet[:,1]>=3)&(ynet[:,1]<381))
  xi=np.rint(x).astype(int);yi=np.rint(y).astype(int);supported=border&masks[f1][xi[:,1],xi[:,0]]&masks[f2][yi[:,1],yi[:,0]]
  np.savez_compressed(O/'raw'/f'{k:02}_{f1}_{f2}_matches.npz',network_xy1=xnet,network_xy2=ynet,original_xy1=x,original_xy2=y,passes_border=border,inside_both_eroded_source_masks=supported)
  cov1=coverage(x[supported],masks[f1]);cov2=coverage(y[supported],masks[f2]);spatial=all(c['count']>=12 and c['grid_cells']>=4 and c['hull_fraction']>=.1 and c['rank']==2 for c in [cov1,cov2])
  review,selected=draw_review(crops[f1],crops[f2],x,y,supported,f1,f2);cv2.imwrite(str(O/'reviews'/f'{k:02}_{f1}_{f2}.jpg'),cv2.cvtColor(review,cv2.COLOR_RGB2BGR))
  rows.append({'index':k,'pair':[f1,f2],'raw_reciprocal_matches':len(x),'mask_supported_matches':int(supported.sum()),'coverage1':cov1,'coverage2':cov2,'spatial_screen_passed':spatial,'source_identity_qualified':False,'review_match_indices':selected,'seconds':time.monotonic()-t0,'not_a_native_pose_candidate':True});write('pair-results.json',rows)
  print(json.dumps(rows[-1]),flush=True);del out,d1,d2,raw
  write('state.json',{'status':'running_fixed_pairs','completed_forward_calls':k+1,'gpu_wall_seconds':time.monotonic()-gpu_start})
 signal.alarm(0);elapsed=time.monotonic()-gpu_start
 report={'status':'completed_requires_source_correspondence_review','forward_calls':len(rows),'max_forward_calls':24,'gpu_wall_seconds':elapsed,'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),'model_sha256':EXPECTED,'all_raw_predictions_and_misses_retained':True,'spatial_screen_passed_pairs':sum(v['spatial_screen_passed'] for v in rows),'native_parameters_changed':False,'training_performed':False,'other_models_downloaded':False,'full_rotation_recovered':False,'next_gate':'Source-patch review for repetitive texture, hand/grass confusion and background dominance; no automatic native fitting.'};write('report.json',report);write('state.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():
   prior=json.loads((O/'state.json').read_text()) if (O/'state.json').exists() else {}
   prior.update(status='stopped_on_failure_no_fallback',traceback=traceback.format_exc());write('state.json',prior)
  raise
