from pathlib import Path
r=Path('/data1/defu/zly/nvda/continuation-20261102-v1');p=r/'a6000-runs/optimize_object_pose_v2.py';s=p.read_text()
s=s.replace("W=O/'pose-candidate-v1'","W=O/'pose-candidate-v2-mask-aware'")
s=s.replace('from pytorch3d.transforms import axis_angle_to_matrix','from pytorch3d.transforms import axis_angle_to_matrix\nfrom pytorch3d.renderer import MeshRasterizer,RasterizationSettings,SoftSilhouetteShader,BlendParams\nfrom pytorch3d.structures import Meshes\nfrom pytorch3d.utils import cameras_from_opencv_projection\nimport cv2')
s=s.replace("'loss':{'reprojection'", "'mask_correction_reason':'First point-only candidate reduced reprojection error but degraded observed silhouette IoU; add the missing source silhouette term, no threshold or parameter-range search.','mask_fit_frames':[150,160,170,180,190,200,210],'mask_raster_resolution':128,'mask_loss_weight':40,'loss':{'reprojection'")
needle="q=torch.arange(61,device=device,dtype=torch.float32)/5;"
insert="""mask_frames=torch.tensor([0,10,20,30,40,50,60],device=device,dtype=torch.long);B=len(mask_frames)
 VK=torch.tensor(V,device=device,dtype=torch.float32);FT=torch.tensor(F,device=device,dtype=torch.long)[None].expand(B,-1,-1)
 cropK=np.asarray(b['observations']['postopt_K_rois'][[150,160,170,180,190,200,210]]).copy();cropK[:,:2]*=.5
 mk=torch.tensor(cropK,device=device,dtype=torch.float32);sizes=torch.full((B,2),128.,device=device)
 cameras=cameras_from_opencv_projection(torch.eye(3,device=device)[None].expand(B,-1,-1),torch.zeros((B,3),device=device),mk,sizes)
 blend=BlendParams(sigma=1e-4,gamma=1e-4);raster=MeshRasterizer(cameras=cameras,raster_settings=RasterizationSettings(image_size=128,blur_radius=float(np.log(1/1e-4-1)*1e-4),faces_per_pixel=10,perspective_correct=True,cull_backfaces=False,max_faces_per_bin=len(F)))
 shader=SoftSilhouetteShader(blend_params=blend)
 mask_target=torch.tensor(np.stack([cv2.resize(np.asarray(b['observations']['postopt_object_mask'][t],np.float32),(128,128),interpolation=cv2.INTER_AREA) for t in [150,160,170,180,190,200,210]]),device=device)
 mask_valid=torch.tensor(np.stack([1-cv2.resize(np.asarray(b['observations']['postopt_human_mask'][t],np.float32),(128,128),interpolation=cv2.INTER_NEAREST) for t in [150,160,170,180,190,200,210]]),device=device).clamp(0,1)
 def silhouette_loss(rot,trans):
  verts=torch.einsum('bij,vj->bvi',rot[mask_frames],VK)+trans[mask_frames,None]
  meshes=Meshes(verts=verts,faces=FT);fragments=raster(meshes);alpha=shader(fragments,meshes)[...,3]
  inter=(alpha*mask_target*mask_valid).sum((1,2));union=((alpha+mask_target-alpha*mask_target)*mask_valid).sum((1,2)).clamp(min=1)
  return (1-inter/union).mean()
 """
assert needle in s;s=s.replace(needle,insert+needle)
s=s.replace("loss=data+smooth+prior;assert torch.isfinite(loss)","mask_loss=silhouette_loss(rot,trans);loss=data+smooth+prior+40*mask_loss;assert torch.isfinite(loss)")
s=s.replace("'prior_loss':float(prior.detach())","'prior_loss':float(prior.detach()),'silhouette_loss':float(mask_loss.detach())")
s=s.replace("'source-anchored-short-track-bounded-SE3-v2'","'source-short-track-and-silhouette-bounded-SE3-v3'")
s=s.replace("'optimizer_steps_completed':300,'source_track_groups'","'optimizer_steps_completed':300,'source_silhouette_term_added':True,'mask_fit_frames':[150,160,170,180,190,200,210],'source_track_groups'")
out=r/'a6000-runs/optimize_object_pose_v3_silhouette.py';compile(s,str(out),'exec');out.write_text(s);print('Mask-aware optimization script installed')
