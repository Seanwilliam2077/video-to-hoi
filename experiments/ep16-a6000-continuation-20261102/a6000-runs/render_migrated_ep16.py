"""Verify and render the intact migrated 360-frame human-object paired state."""
from pathlib import Path
import os,time,json,hashlib,subprocess,traceback
import cv2,imageio.v2 as imageio,numpy as np,torch
from lib_mhr import MHRLayer
from lib_mhr.human_texture import batched_vertex_normals
from tools.mhr_mesh_renderer import NvdiffMeshRenderer
from tools.render_mhr_wild_inference import _params,HUMAN_ALBEDO
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');O=R/'a6000-runs/ep16-migrated-reference-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,d):(O/name).write_text(json.dumps(d,indent=2))
def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='2'
 m,u=map(int,subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','));print('GPU preflight',m,u,flush=True);assert m<500
 O.mkdir(exist_ok=False);start=time.time();write('state.json',{'status':'validating_inputs','pid':os.getpid()})
 p=R/'experiments/ep16-joint-direction-20261031/joint-green-v1/refined.pth';source=R/'runs/full-episode16-20261030/inputs/episode_000016.0.color.mp4'
 assert sha(p)=='61decc2bae1b4c5db67c30c7aee521edf61e5c1f5a352e70eec1f03c57234f32';assert sha(source)=='a37a978ef0f3d0eeaad74f2ed75fb32502d8a1a047587291c14433d052f9eb74'
 b=torch.load(p,map_location='cpu',weights_only=False);assert list(map(int,b['frames']))==list(range(360));assert [int(x['src_frame']) for x in b['frame_meta']]==list(range(360));assert b['metadata']['ground_truth_used'] is False
 mesh=Path(b['metadata']['object_mesh']);K=np.asarray(b['observations']['K_full'],np.float32);T=np.asarray(b['pr']['pose_abs'],np.float32)
 assert mesh.exists() and np.isfinite(K).all() and np.isfinite(T).all()
 write('input-contract.json',{'paired_bundle_sha256':sha(p),'source_sha256':sha(source),'object_mesh_sha256':sha(mesh),'frames':360,'original_frame_ids':list(range(360)),'same_human_object_pair':True,'parameters_modified':False,'path_translation_only_in_memory':True,'quality_accepted':False})
 torch.set_num_threads(4);cv2.setNumThreads(2);layer=MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'],device='cuda');faces=layer.mesh_faces(device='cuda').cpu().numpy().astype(np.int32);renderer=NvdiffMeshRenderer(device='cuda')
 cap=cv2.VideoCapture(str(source));assert cap.get(7)==360 and cap.get(5)==30
 writer=imageio.get_writer(str(O/'ep16-source-and-native-360-30fps.mp4'),format='FFMPEG',fps=30,codec='libx264',pixelformat='yuv420p',macro_block_size=None,ffmpeg_params=['-crf','18','-threads','2'])
 selected={0,120,150,173,174,190,210,300,359};sheets=[];crops=[]
 with torch.no_grad():
  for lo in range(0,360,4):
   hi=min(lo+4,360);v=layer.mhr_forward(_params(b['pr'],slice(lo,hi),'cuda')).vertices.cpu().numpy().astype(np.float32);assert np.isfinite(v).all();normal=batched_vertex_normals(v,faces)
   rgb,mask=renderer.render_front_batch_constant_human_textured_object(v,faces,normal,str(mesh),T[lo:hi],K[lo:hi],(1152,1536),HUMAN_ALBEDO)
   for j,t in enumerate(range(lo,hi)):
    ok,img=cap.read();assert ok,t;img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);surface=np.full_like(img,(32,36,40));surface[mask[j]]=np.rint(np.clip(rgb[j][mask[j]],0,1)*255).astype(np.uint8)
    panels=[]
    for im,title in [(img,'Original Track 1 source'),(surface,'Intact migrated human + object | UNACCEPTED')]:
     panel=cv2.resize(im,(1024,768));cv2.rectangle(panel,(0,0),(1024,31),(0,0,0),-1);cv2.putText(panel,f'{title} | original frame {t}',(6,22),0,.52,(255,255,255),1);panels.append(panel)
    frame=np.concatenate(panels,1);writer.append_data(frame)
    if t in selected:
     sheets.append(cv2.resize(frame,(1024,384)));crops.append(np.concatenate([cv2.resize(im[490:790,590:890],(450,450)) for im in (img,surface)],1))
   write('state.json',{'status':'rendering','frames_completed':hi,'seconds':time.time()-start,'physical_gpu':2});print('Rendered',hi,'/360',flush=True)
 writer.close();cap.release();assert not cap.read()[0]
 cv2.imwrite(str(O/'full-scene-review.jpg'),cv2.cvtColor(np.concatenate(sheets,0),cv2.COLOR_RGB2BGR));cv2.imwrite(str(O/'hand-object-review.jpg'),cv2.cvtColor(np.concatenate(crops,0),cv2.COLOR_RGB2BGR))
 c=cv2.VideoCapture(str(O/'ep16-source-and-native-360-30fps.mp4'));n=0
 while c.read()[0]:n+=1
 c.release();assert n==360
 report={'status':'completed','decoded_output_frames':n,'fps':30,'seconds':time.time()-start,'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),'parameters_modified':False,'full_human_object_scene':True,'quality_accepted':False,'limitations':['Migration replay only; existing turn and hand projection failures remain.','No official score, true contact, or physical scale validation.']};write('report.json',report);write('state.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  if O.exists():write('state.json',{'status':'failed','traceback':traceback.format_exc()})
  raise
