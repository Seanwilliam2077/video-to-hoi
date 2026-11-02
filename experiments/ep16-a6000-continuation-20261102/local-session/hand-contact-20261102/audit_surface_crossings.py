"""Exact triangle versus actual cuboid SAT crossing checks, not official PEN."""
from pathlib import Path
import argparse,json,time,numpy as np,torch
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');D=R/'a6000-runs/hand-contact-20261102-v1';ap=argparse.ArgumentParser();ap.add_argument('--stage',default='v2');args=ap.parse_args();start=time.time();b=torch.load(D/f'native-contact-candidate-{args.stage}/refined.pth',map_location='cpu',weights_only=False);base=np.load(D/'input-geometry.npz');after=np.load(D/f'native-contact-candidate-{args.stage}-evaluation/input-geometry.npz');F=base['human_faces'];lo=base['object_vertices'].min(0);hi=base['object_vertices'].max(0);center=(lo+hi)/2;half=(hi-lo)/2;plane=base['convex_planes'];assert plane.shape==(6,4);assert np.allclose(np.sort(abs(plane[:,:3]),axis=1),[0,0,1],atol=1e-6)
def intersects(tri,h):
 good=(tri.min(1)<=h).all(-1)&(tri.max(1)>=-h).all(-1);x=tri[good]
 if not len(x):return 0
 e=np.roll(x,-1,axis=1)-x;axes=np.cross(e[:,:,None,:],np.eye(3)[None,None,:,:]).reshape(len(x),9,3);p=np.einsum('mnj,maj->man',x,axes);r=np.einsum('maj,j->ma',abs(axes),h);good=~((p.min(-1)>r+1e-10)|(p.max(-1)<-r-1e-10)).any(1);x=x[good]
 if not len(x):return 0
 normal=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0]);p=np.einsum('mj,mj->m',x[:,0],normal);r=abs(normal)@h;return int((abs(p)<=r+1e-10).sum())
fixture=np.array([[[-2.,-2,0],[2,-2,0],[0,2,0]],[[-2.,-2,2],[2,-2,2],[0,2,2]],[[-2.,-2,1],[2,-2,1],[0,2,1]]]);assert intersects(fixture[:1],np.ones(3))==1;assert intersects(fixture[1:2],np.ones(3))==0;assert intersects(fixture[2:],np.ones(3)-.001)==0
rows=[];Ts=b['pr']['pose_abs']
for tag,g in [('before',base),('after',after)]:
 verts=g['vertices'];counts=[]
 for frame in range(360):
  T=Ts[frame];local=(verts[frame].astype(float)-T[:3,3])@T[:3,:3]-center;tri=local[F];counts.append(intersects(tri,half-.0001))
 rows.append({'variant':tag,'frames_surface_crossing_deeper_than_point1mm':int((np.array(counts)>0).sum()),'maximum_crossing_triangles_per_frame':int(max(counts)),'per_frame_crossing_triangles':counts})
report={'algorithm':'Triangle-cuboid separating axis theorem,3box axes,9cross axes and triangle normal; every canonical human triangle, every360frame.','actual_mesh_match':'Watertight convex866vertex1728face mesh with6axis planes and matching box volume.','tolerance':'Object half extents reduced0.1mm along each axis to exclude boundary contact and numerical noise.','fixture_passed':True,'results':rows,'seconds':time.time()-start,'scope':'Surface intersection geometry, not hidden-reference human accuracy, grasp truth or official PEN.'};(D/f'surface-crossings-{args.stage}.json').write_text(json.dumps(report,indent=2));print(json.dumps({**report,'results':[{k:v for k,v in x.items() if k!='per_frame_crossing_triangles'} for x in rows]}),flush=True)
