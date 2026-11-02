from pathlib import Path
import json,os,time,numpy as np,torch
from scipy.spatial import cKDTree
from lib_mhr import MHRLayer
from tools.render_mhr_wild_inference import _params
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');D=R/'a6000-runs/hand-contact-20261102-v1';O=R/'a6000-runs/hand-review-20261102';O.mkdir(exist_ok=True);p=D/'native-contact-candidate-v11/refined.pth';b=torch.load(p,map_location='cpu',weights_only=False);g=np.load(D/'native-contact-candidate-v11-evaluation/input-geometry.npz');faces=g['human_faces'];hids=g['hand_vertex_ids'];HF=[]
for side in range(2):
 sel=np.zeros(len(g['vertices'][0]),bool);sel[hids[side]]=True;HF.append(faces[sel[faces].all(1)])
layer=MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'],device='cuda');torch.set_num_threads(4);rows=[];start=time.time()
def pierce(A,B):
 e1=B[:,1]-B[:,0];e2=B[:,2]-B[:,0];hit=np.zeros(len(A),bool)
 for i in range(3):
  o=A[:,i];d=A[:,(i+1)%3]-o;pv=np.cross(d,e2);det=(e1*pv).sum(1);ok=abs(det)>1e-15;inv=np.divide(1.,det,out=np.zeros_like(det),where=ok);tv=o-B[:,0];u=(tv*pv).sum(1)*inv;q=np.cross(tv,e1);v=(d*q).sum(1)*inv;t=(e2*q).sum(1)*inv;hit|=ok&(u>1e-6)&(v>1e-6)&(u+v<1-1e-6)&(t>1e-6)&(t<1-1e-6)
 return hit
for f in [96,115,130,175,220,240]:
 with torch.no_grad():fresh=layer.mhr_forward(_params(b['pr'],slice(f,f+1),'cuda')).vertices.cpu().numpy()[0]
 row={'frame':f,'fresh_vs_render_cache_max_mm':float(abs(fresh-g['vertices'][f]).max()*1000),'self_triangle_pairs':[]}
 for side in range(2):
  ff=HF[side];tri=fresh[ff].astype(float);c=tri.mean(1);radius=np.linalg.norm(tri-c[:,None],axis=-1).max(1);pairs=cKDTree(c).query_pairs(2*radius.max(),output_type='ndarray');a,z=pairs.T;good=~(ff[a,:,None]==ff[z,None,:]).any((1,2));a=a[good];z=z[good];good=(tri[a].min(1)<=tri[z].max(1)).all(1)&(tri[z].min(1)<=tri[a].max(1)).all(1);a=a[good];z=z[good];hit=pierce(tri[a],tri[z])|pierce(tri[z],tri[a]);row['self_triangle_pairs'].append(int(hit.sum()))
 rows.append(row);print(json.dumps(row),flush=True)
try:
 import mujoco
 version=mujoco.__version__
except ImportError:version=None
report={'candidate':str(p),'mujoco_installed_version':version,'physics_used_for_this_candidate':False,'same_native_parameter_fresh_decode_check':rows,'scope':'Proper noncoplanar intersections between nonadjacent triangles within each hand,6review frames; not an exhaustive hand-body collision audit. Positive counts establish self-intersection; zeros do not certify absence.','seconds':time.time()-start};(O/'audit.json').write_text(json.dumps(report,indent=2));print('REVIEW_COMPLETE',flush=True)
