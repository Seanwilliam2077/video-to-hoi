"""Independent synthetic multi-shape gradient/coordinate checks; no challenge data."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import trimesh
import Utils
from scipy.spatial.transform import Rotation
from v2hoi.object_refine_runtime import ObservationRenderer


def main():
    results=[]
    for kind in ("box", "ring", "pan"):
        if kind=="box":
            mesh=trimesh.creation.box(extents=[.25,.2,.3])
        elif kind=="ring":
            mesh=trimesh.creation.annulus(r_min=.12,r_max=.2,height=.03)
        else:
            handle=trimesh.creation.box(extents=[.2,.025,.02])
            handle.apply_translation([.23,0,0])
            mesh=trimesh.util.concatenate([trimesh.creation.cylinder(radius=.15,height=.025),handle])
        mesh.apply_translation([.3,-.2,.1])
        color=np.array([90,160,60,255],dtype=np.uint8)
        mesh.visual.vertex_colors=np.tile(color,(len(mesh.vertices),1))
        renderer=ObservationRenderer(mesh)
        center=torch.tensor(mesh.bounds.mean(axis=0),dtype=torch.float32,device="cuda")
        k=np.array([[220.,0,96.],[0,220.,96.],[0,0,1.]],dtype=np.float32)
        r=torch.tensor(Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix()[None],dtype=torch.float32,device="cuda")
        expected_center=torch.tensor([[0.,0.,1.2]],device="cuda")
        target,alpha,_=renderer.render(r,expected_center,1.,center,k[None],(192,192))
        target,alpha=target.detach(),alpha.detach()
        branch_scales=torch.tensor([.9,1.1],dtype=torch.float32,device='cuda',requires_grad=True)
        branch_images,branch_alpha,_=renderer.render(r.repeat(2,1,1),expected_center.repeat(2,1),
            branch_scales[:,None,None],center,np.repeat(k[None],2,axis=0),(192,192))
        branch_loss=(branch_alpha-alpha).square().mean()+(branch_images-target).square().mean()
        branch_loss.backward()
        assert branch_alpha.shape==(2,192,192,1)
        assert torch.isfinite(branch_scales.grad).all() and (branch_scales.grad.abs()>0).all()
        pose=torch.eye(4,device="cuda")[None]
        pose[:,:3,:3]=r
        pose[:,:3,3]=expected_center-(r@center[:,None])[...,0]
        ordinary,z,_=Utils.nvdiffrast_render(K=k,H=192,W=192,ob_in_cams=pose,
            glctx=renderer.context,mesh=mesh,use_light=False,extra={})
        # AA intentionally changes edge pixels; compare only the shared interior.
        erosion=cv2.erode((alpha[0,...,0].cpu().numpy()>.999).astype(np.uint8),np.ones((5,5),np.uint8))
        interior=torch.tensor(erosion[None].astype(bool),device="cuda") & (z>.001)
        difference=float((target-ordinary).abs()[interior].max())
        assert difference < 1e-5, (kind,difference)
        c=torch.tensor([[.015,-.01,1.25]],device="cuda",requires_grad=True)
        scale=torch.tensor(1.03,device="cuda",requires_grad=True)
        image,silhouette,_=renderer.render(r,c,scale,center,k[None],(192,192))
        intersection=(silhouette*alpha).sum()
        union=(silhouette+alpha-silhouette*alpha).sum()
        loss=1-intersection/union+(image-target).square().mean()
        loss.backward()
        assert torch.isfinite(c.grad).all() and c.grad.abs().sum()>0
        assert torch.isfinite(scale.grad) and scale.grad.abs()>0
        results.append({"shape":kind,"render_coordinate_max_color_difference":difference,
            "center_gradient":c.grad.detach().cpu().tolist(),"scale_gradient":float(scale.grad)})
    path=Path("artifacts/object-refine-synthetic-renderer.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"provenance":"independent synthetic geometry","checks":results},indent=2))
    print(path)


if __name__=="__main__":
    main()
