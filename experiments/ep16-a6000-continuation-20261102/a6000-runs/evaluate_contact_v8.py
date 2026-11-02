from pathlib import Path
import os,json,time,numpy as np,torch
from lib_mhr import MHRLayer
from lib_mhr.object_texture import load_object_mesh_parts,concatenate_object_mesh_parts
from lib_mhr.hand_surface_contact import load_mhr_hand_surface_spec
from learning.training.mhr_opt_refineout import DEFAULT_HAND_SURFACE_SPEC
from tools.render_mhr_wild_inference import _params
from scipy.spatial import ConvexHull
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');O=R/'a6000-runs/hand-contact-20261102-v1/native-contact-candidate-v8-evaluation';O.mkdir(exist_ok=True);source=R/'a6000-runs/hand-contact-20261102-v1/native-contact-candidate-v8/refined.pth';b=torch.load(source,map_location='cpu',weights_only=False);m=concatenate_object_mesh_parts(load_object_mesh_parts(b['metadata']['object_mesh']));hull=ConvexHull(m.vertices);planes=np.unique(np.round(hull.equations,7),axis=0);layer=MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'],device='cuda');faces=layer.mesh_faces(device='cuda').cpu().numpy();spec=load_mhr_hand_surface_spec(DEFAULT_HAND_SURFACE_SPEC,faces=faces);rows=[];vv=[];kk=[];start=time.time();torch.set_num_threads(4)
for lo in range(0,360,8):
 with torch.no_grad():out=layer.mhr_forward(_params(b['pr'],slice(lo,min(lo+8,360)),'cuda'))
 v=out.vertices.cpu().numpy();k=out.keypoints.cpu().numpy();vv.append(v);kk.append(k)
 for j,t in enumerate(range(lo,min(lo+8,360))):
  T=b['pr']['pose_abs'][t];hands=(v[j,spec.vertex_indices]-T[:3,3])@T[:3,:3];s=(hands@planes[:,:3].T+planes[:,3]).max(-1);pen=np.maximum(-s,0)*1000;rows.append({'frame':t,'hand_mean_penetration_mm':pen.mean(1).tolist(),'hand_max_penetration_mm':pen.max(1).tolist(),'hand_inside_fraction':(pen>0).mean(1).tolist()})
np.savez_compressed(O/'input-geometry.npz',vertices=np.concatenate(vv),keypoints=np.concatenate(kk),human_faces=faces,hand_vertex_ids=spec.vertex_indices,object_vertices=np.asarray(m.vertices),object_faces=np.asarray(m.faces),convex_planes=planes)
report={'input':str(source),'human_vertex_count':int(len(v[0])),'object_vertex_count':len(m.vertices),'object_faces':len(m.faces),'object_watertight':bool(m.is_watertight),'object_is_convex':bool(m.is_convex),'mesh_volume':float(m.volume),'hull_volume':float(hull.volume),'hull_planes':planes.tolist(),'frames':rows,'seconds':time.time()-start,'scope':'Signed convex hull face-plane depth for inside hand vertices; exact interior face clearance if actual mesh is convex, not an official penetration metric. Native hand spec all2318 vertices per hand.'};(O/'input-collision-report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['frames','hull_planes']}),flush=True);print('WORST',sorted(rows,key=lambda x:max(x['hand_max_penetration_mm']),reverse=True)[:6],flush=True)
