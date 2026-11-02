from pathlib import Path
import numpy as np,torch,json
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');Q=R/'a6000-runs/matching-optimization-v2';D=R/'a6000-runs/direction-correction-v1'
# Reuse the pure geometric ray intersection, not its old pose bindings or optimizer.
import sys;sys.path.insert(0,str(R/'a6000-runs'));from optimize_object_pose_v2 import intersect
from lib_mhr.object_texture import load_object_mesh_parts,concatenate_object_mesh_parts
base=torch.load(R/'experiments/ep16-joint-direction-20261031/joint-green-v1/refined.pth',map_location='cpu',weights_only=False);new=torch.load(D/'source-face-candidate-v3/refined.pth',map_location='cpu',weights_only=False);K=np.asarray(base['observations']['K_full']);m=concatenate_object_mesh_parts(load_object_mesh_parts(base['metadata']['object_mesh']));rows=[]
for anchor in [150,165,180,195]:
 g=np.load(Q/f'group-{anchor}-tracks.npz');ids=np.flatnonzero(g['valid'].all(0));f=g['frames'];target=g['xy'][:,ids];out={'anchor':anchor,'source_track_count':len(ids)}
 for name,b in [('old',base),('new',new)]:
  T=np.asarray(b['pr']['pose_abs']);X,keep,faces=intersect(target[0],K[anchor],T[anchor],np.asarray(m.vertices),np.asarray(m.faces));obs=target[:,keep];cam=np.einsum('bij,nj->bni',T[f,:3,:3],X)+T[f,None,:3,3];p=np.einsum('bij,bnj->bni',K[f],cam);uv=p[...,:2]/p[...,2:];err=np.linalg.norm(uv-obs,axis=-1)
  # Relative motion centered across tracks removes common image translation.
  obs_d=obs[-1]-obs[0];pred_d=uv[-1]-uv[0];obs_d-=obs_d.mean(0);pred_d-=pred_d.mean(0);cos=float((obs_d*pred_d).sum()/max(np.linalg.norm(obs_d)*np.linalg.norm(pred_d),1e-9));out[name]={'bound_count':len(X),'mean_px':float(err.mean()),'median_px':float(np.median(err)),'end_centered_motion_cosine':cos}
 rows.append(out)
(D/'short-track-direction-check-v3.json').write_text(json.dumps({'scope':'Same-source development diagnostic; conditional first-surface ray binding at each group anchor. Tracks are proposals, not physical identity truth. Centered motion agreement is not full 3D rotation validation.','groups':rows},indent=2));print(json.dumps(rows),flush=True)
