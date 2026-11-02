from pathlib import Path
s=(Path(__file__).parent/'optimize_direction.py').read_text().replace("O=D/'source-face-candidate'","O=D/'source-face-candidate-v2'")
s=s.replace('init=[];rows=[]','init=[];rows=[]')
mark="(D/'source-pnp-report.json').write_text"
pos=s.index(mark)
s=s[:pos]+'''# A cube has four indistinguishable corner orderings in silhouette. Unwrap one
# continuous material coordinate frame, instead of relabeling corners by screen side.
S=np.array([Rotation.from_euler('z',90*j,degrees=True).as_matrix() for j in range(4)])
C=np.array([[x[0]@z for z in S] for x in init]);cost=np.full((len(frames),4),np.inf);prev=np.zeros((len(frames),4),int)
cost[0]=[Rotation.from_matrix(base[60,:3,:3].T@r).magnitude()**2/30 for r in C[0]]
for i in range(1,len(frames)):
 for j in range(4):
  v=np.array([cost[i-1,k]+Rotation.from_matrix(C[i-1,k].T@C[i,j]).magnitude()**2/(frames[i]-frames[i-1]) for k in range(4)]);prev[i,j]=v.argmin();cost[i,j]=v.min()
phase=np.zeros(len(frames),int);phase[-1]=cost[-1].argmin()
for i in range(len(frames)-1,0,-1):phase[i-1]=prev[i,phase[i]]
init=[(C[i,j],init[i][1]) for i,j in enumerate(phase)];source_X=np.array([X@S[j] for j in phase])
(O/'corner-phase-unwrapping.json').write_text(json.dumps({'frames':frames.tolist(),'quarter_turn_phase':phase.tolist(),'method':'Global minimum angular path across four cube corner symmetries; first gauge connected to preserved frame60. Not per-frame heading truth.'},indent=2))
''' +s[pos:]
s=s.replace("XT=torch.tensor(X,dtype=torch.float32,device='cuda')","XT=torch.tensor(source_X,dtype=torch.float32,device='cuda')")
s=s.replace("torch.einsum('bij,nj->bni',rot[1:-1],XT)","torch.einsum('bij,bnj->bni',rot[1:-1],XT)")
s=s.replace("opt=torch.optim.Adam([dr,dt],lr=.01);history=[]",'''opt=torch.optim.Adam([dr,dt],lr=.01);history=[]
from pytorch3d.renderer import MeshRasterizer,RasterizationSettings,SoftSilhouetteShader,BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from lib_mhr.object_texture import load_object_mesh_parts,concatenate_object_mesh_parts
m=concatenate_object_mesh_parts(load_object_mesh_parts(b['metadata']['object_mesh']));V=torch.tensor(np.asarray(m.vertices),device='cuda',dtype=torch.float32);F=torch.tensor(np.asarray(m.faces),device='cuda',dtype=torch.long)[None].expand(len(frames),-1,-1)
mk=np.asarray(b['observations']['postopt_K_rois'][frames]).copy();mk[:,:2]*=.5
camera=cameras_from_opencv_projection(torch.eye(3,device='cuda')[None].expand(len(frames),-1,-1),torch.zeros((len(frames),3),device='cuda'),torch.tensor(mk,device='cuda',dtype=torch.float32),torch.full((len(frames),2),128.,device='cuda'))
raster=MeshRasterizer(cameras=camera,raster_settings=RasterizationSettings(image_size=128,blur_radius=float(np.log(1/1e-4-1)*1e-4),faces_per_pixel=10,perspective_correct=True,cull_backfaces=False,max_faces_per_bin=len(m.faces)));shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4,gamma=1e-4))
mt=torch.tensor(np.stack([cv2.resize(np.asarray(b['observations']['postopt_object_mask'][f],np.float32),(128,128),interpolation=cv2.INTER_AREA) for f in frames]),device='cuda');valid=torch.tensor(np.stack([1-cv2.resize(np.asarray(b['observations']['postopt_human_mask'][f],np.float32),(128,128),interpolation=cv2.INTER_NEAREST) for f in frames]),device='cuda').clamp(0,1)
def maskloss(rot,t):
 meshes=Meshes(verts=torch.einsum('bij,vj->bvi',rot[1:-1],V)+t[1:-1,None],faces=F);alpha=shader(raster(meshes),meshes)[...,3];inter=(alpha*mt*valid).sum((1,2));union=((alpha+mt-alpha*mt)*valid).sum((1,2)).clamp(min=1);return (1-inter/union).mean()
basez=torch.tensor(base[frames,2,3],device='cuda')
''')
s=s.replace('loss=data+smooth;loss.backward()',"ml=maskloss(rot,t);gauge=.1*((t[1:-1,2]-basez)/.1).square().mean();loss=data+smooth+40*ml+gauge;loss.backward()")
s=s.replace("'smooth':float(smooth.detach())","'smooth':float(smooth.detach()),'mask_loss':float(ml.detach()),'depth_gauge_prior':float(gauge.detach())")
s=s.replace("'angle_cap':None","'angle_cap':None,'cube_corner_symmetry_unwrapped':True,'estimated_visible_silhouette_fit':True,'soft_depth_gauge_prior_m':.1")
(Path(__file__).parent/'optimize_direction_v2.py').write_text(s)
