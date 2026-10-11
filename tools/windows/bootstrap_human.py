"""Estimate one permitted source-frame human mesh for object depth conditioning."""
import argparse
import json
import os
from pathlib import Path
import sys
import h5py
import joblib
import numpy as np
import torch
from v2hoi.native import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb",type=Path,required=True)
    p.add_argument("--masks",type=Path,required=True)
    p.add_argument("--sequence",required=True)
    p.add_argument("--intrinsics",type=Path,required=True)
    p.add_argument("--video-sha256",required=True)
    p.add_argument("--frame",type=int,default=0)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,os.environ["SAM3D_BODY_ROOT"])
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    with h5py.File(a.masks) as f:
        if f.attrs["source_video_sha256"]!=a.video_sha256 or f.attrs["fill_hole_area"]!=8:
            raise ValueError("Require the exact official Track 1 segmentation")
        if a.frame<0: raise ValueError("Original frame ID must be nonnegative")
        human=f[f"{a.sequence}/{a.frame:06d}-k0.person_mask.png"][()].astype(bool)
    y,x=np.where(human)
    if len(y)<100: raise ValueError("No supported human in bootstrap frame")
    box=np.array([[x.min(),y.min(),x.max(),y.max()]],dtype=np.float32)
    intrinsics=joblib.load(a.intrinsics)
    k=np.array([[intrinsics['fx'],0,intrinsics['cx']],[0,intrinsics['fy'],intrinsics['cy']],[0,0,1]],dtype=np.float32)
    assets=Path(os.environ["MHR_ASSETS_ROOT"])/"checkpoints/sam-3d-body-dinov3"
    model,cfg=load_sam_3d_body(checkpoint_path=str(assets/"model.ckpt"),device="cuda",mhr_path=str(assets/"assets/mhr_model.pt"))
    estimator=SAM3DBodyEstimator(sam_3d_body_model=model,model_cfg=cfg)
    with torch.inference_mode():
        result=estimator.process_one_image(str(a.rgb),bboxes=box,masks=human[None],
            cam_int=torch.tensor(k,device="cuda")[None],inference_type="body")
    if len(result)!=1: raise ValueError("Ambiguous bootstrap human")
    value=result[0]
    vertices=(value["pred_vertices"]+value["pred_cam_t"][None]).detach().cpu()
    if not torch.isfinite(vertices).all() or vertices[:,2].median()<=0:
        raise ValueError("Invalid camera-space human geometry")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists(): raise FileExistsError("Preserve completed bootstrap meshes")
    torch.save({"pred_vertices":vertices,"faces":torch.as_tensor(estimator.faces).cpu()},a.output)
    a.output.with_suffix(".json").write_text(json.dumps({"provenance":"track1","source_video_sha256":a.video_sha256,
        "bootstrap_original_frame_id":a.frame,"rgb_sha256":sha256(a.rgb),"mask_sha256":sha256(a.masks),
        "checkpoint_sha256":sha256(assets/"model.ckpt"),"mesh_sha256":sha256(a.output),
        "scope":"Selected source-frame human scale prior; not a complete human trajectory or reference"},indent=2))


if __name__=="__main__": main()
