"""Track observed object texture on a fitted shape hypothesis, with mask gates."""
import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from fit_object_hypotheses import FrameEvidence, VIDEO, MASKS, K, digest


def detect_object(frame, mask, sift):
    y,x=np.nonzero(mask)
    x0,y0=max(0,x.min()-8),max(0,y.min()-8)
    x1,y1=min(frame.shape[1],x.max()+9),min(frame.shape[0],y.max()+9)
    inner=cv2.erode(mask.astype(np.uint8)*255,np.ones((3,3),np.uint8))
    kp,desc=sift.detectAndCompute(frame[y0:y1,x0:x1],inner[y0:y1,x0:x1])
    xy=np.array([[p.pt[0]+x0,p.pt[1]+y0] for p in kp],dtype=float)
    return xy,desc


def intersect_reference(xy,T,half,nxz,ny):
    rotation=T[:3,:3]; center=T[:3,3]
    invK=np.linalg.inv(K)
    locations=[]; usable=[]
    for index,point in enumerate(xy):
        ray=invK@np.r_[point,1.]
        def implicit(depth):
            local=(ray*depth-center)@rotation/half
            horizontal=(abs(local[0])**nxz+abs(local[2])**nxz)**(ny/nxz)
            return horizontal+abs(local[1])**ny-1
        depths=np.linspace(center[2]-half*2,center[2]+half*2,256)
        values=np.array([implicit(d) for d in depths])
        crossing=np.flatnonzero((values[:-1]>=0)&(values[1:]<0))
        if len(crossing):
            j=crossing[0]
            depth=brentq(implicit,depths[j],depths[j+1])
            local=(ray*depth-center)@rotation
            locations.append(local); usable.append(index)
    return np.array(locations),np.array(usable)


def visible_iou(T,vertices,mask,human):
    points=vertices@T[:3,:3].T+T[:3,3]
    hom=points@K.T
    xy=hom[:,:2]/hom[:,2:]
    hull=cv2.convexHull(xy.astype(np.float32)).reshape(-1,2)
    predicted=np.zeros(mask.shape,np.uint8)
    cv2.fillConvexPoly(predicted,np.rint(hull).astype(np.int32),1)
    visible=~cv2.dilate(human.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    a,b=mask&visible,predicted.astype(bool)&visible
    return float((a&b).sum()/max((a|b).sum(),1))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rotation-limit",type=float,default=.5)
    parser.add_argument("--mask-loss-limit",type=float,default=.025)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"object_candidate.npz").exists(): raise FileExistsError("Use fresh output")
    start=time.perf_counter()
    p=np.load(args.source/"object_candidate.npz",allow_pickle=False)
    base={name:p[name] for name in p.files}
    report=json.loads((args.source/"report.json").read_text())
    shape=next(item for item in report["hypotheses"] if item["name"]==report["selected_hypothesis"])
    transforms=p["object_to_camera"].copy()
    cap=cv2.VideoCapture(str(VIDEO)); ok,first=cap.read()
    if not ok: raise RuntimeError("Source video frame0 unreadable")
    firstmask=cv2.imread(str(MASKS/"object/000000.png"),0)>0
    sift=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.01)
    reference_xy,reference_desc=detect_object(first,firstmask,sift)
    reference_xyz,usable=intersect_reference(reference_xy,transforms[0],report["metric_half_extent_m"],shape["nxz"],shape["ny"])
    reference_xy=reference_xy[usable]; reference_desc=reference_desc[usable]
    bf=cv2.BFMatcher()
    stats=[]; accepted=[]
    cv2.setRNGSeed(20261015)
    for index in range(360):
        if index==0: frame=first
        else:
            ok,frame=cap.read()
            if not ok: raise RuntimeError(f"Decode failure {index}")
        mask=cv2.imread(str(MASKS/"object"/f"{index:06d}.png"),0)>0
        human=cv2.imread(str(MASKS/"human"/f"{index:06d}.png"),0)>0
        xy,desc=detect_object(frame,mask,sift)
        matches=[]
        if desc is not None and len(desc)>=2:
            matches=[a for a,b in bf.knnMatch(reference_desc,desc,k=2) if a.distance<.75*b.distance]
        good=False; inlier_count=0; reproj=None; proposed_iou=None; rotation_delta=None
        base_iou=visible_iou(transforms[index],p["vertices"],mask,human)
        if len(matches)>=8:
            xyz=np.array([reference_xyz[m.queryIdx] for m in matches])
            observed=np.array([xy[m.trainIdx] for m in matches])
            rv=cv2.Rodrigues(transforms[index,:3,:3])[0]
            tv=transforms[index,:3,3].reshape(3,1).copy()
            solved,rv,tv,inliers=cv2.solvePnPRansac(xyz,observed,K,np.zeros(5),rv,tv,useExtrinsicGuess=True,iterationsCount=150,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_ITERATIVE)
            if solved and inliers is not None and len(inliers)>=8:
                selected=inliers.ravel(); inlier_count=len(selected)
                rv,tv=cv2.solvePnPRefineLM(xyz[selected],observed[selected],K,np.zeros(5),rv,tv)
                proposed=transforms[index].copy(); proposed[:3,:3]=cv2.Rodrigues(rv)[0]; proposed[:3,3]=tv.ravel()
                rendered=cv2.projectPoints(xyz[selected],rv,tv,K,np.zeros(5))[0].reshape(-1,2)
                reproj=float(np.linalg.norm(rendered-observed[selected],axis=1).mean())
                proposed_iou=visible_iou(proposed,p["vertices"],mask,human)
                rotation_delta=float(Rotation.from_matrix(proposed[:3,:3]@transforms[index,:3,:3].T).magnitude())
                good=bool(reproj<=1.5 and proposed_iou>=base_iou-args.mask_loss_limit and rotation_delta<=args.rotation_limit and .8<=proposed[2,3]/transforms[index,2,3]<=1.2)
                if good: transforms[index]=proposed
        accepted.append(good)
        stats.append({"frame":index,"matches":len(matches),"pnp_inliers":inlier_count,"accepted":good,"mean_inlier_reprojection_px":reproj,"baseline_visible_mask_iou":base_iou,"proposed_visible_mask_iou":proposed_iou,"rotation_delta_rad":rotation_delta})
        if (index+1)%60==0: print(f"Object texture PnP {index+1}/360, accepted {sum(accepted)}",flush=True)
    cap.release()
    base["object_to_camera"]=transforms
    base["rotation"]=transforms[:,:3,:3]
    base["translation"]=transforms[:,:3,3]
    base["texture_tracking_accepted"]=np.array(accepted,dtype=bool)
    np.savez_compressed(args.output/"object_candidate.npz",**base)
    for name in ("object.obj","object.mtl"):
        shutil.copy2(args.source/name,args.output/name)
    np.savez_compressed(args.output/"reference_features.npz",reference_xy=reference_xy,reference_xyz=reference_xyz)
    report.update({"pose_refinement":"Image texture correspondences: frame0 observed SIFT points ray-intersected with the fitted candidate surface; per-frame RANSAC PnP + LM. Reject if insufficient inliers, high reprojection, mask regression, implausible change. Fall back to original silhouette pose where rejected.","pose_refinement_limit":"Reference feature 3D locations assume the selected shape, so this is not independent SfM geometry or truth. Pose/shape/depth remain underconstrained and reference texture can repeat.","source_candidate":str(args.source),"texture_tracking_rotation_limit_rad":args.rotation_limit,"texture_tracking_mask_loss_limit":args.mask_loss_limit,"texture_tracking_frames_accepted":int(sum(accepted)),"texture_tracking_reference_features":len(reference_xyz),"texture_tracking_stats":stats,"texture_tracking_seconds":time.perf_counter()-start,"texture_tracking_script_sha256":digest(Path(__file__))})
    (args.output/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"accepted":int(sum(accepted)),"reference_features":len(reference_xyz),"seconds":time.perf_counter()-start},indent=2),flush=True)


if __name__=="__main__": main()
