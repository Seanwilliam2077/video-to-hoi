"""Joint image-constrained temporal object-pose candidate with held-out pixels."""
import argparse
import json
from pathlib import Path
import shutil
import time
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from fit_object_hypotheses import VIDEO, MASKS, K, digest, sample_polygon, point_polygon_distance
from track_object_texture import detect_object, intersect_reference, visible_iou


def acquire(data, report):
    cap=cv2.VideoCapture(str(VIDEO)); ok,first=cap.read()
    if not ok: raise RuntimeError("Cannot decode frame0")
    shape=next(x for x in report["hypotheses"] if x["name"]==report["selected_hypothesis"])
    sift=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.01)
    reference_mask=cv2.imread(str(MASKS/"object/000000.png"),0)>0
    xy,desc=detect_object(first,reference_mask,sift)
    xyz,indices=intersect_reference(xy,data["object_to_camera"][0],report["metric_half_extent_m"],shape["nxz"],shape["ny"])
    xy,desc=xy[indices],desc[indices]
    bf=cv2.BFMatcher(); evidence=[]
    cv2.setRNGSeed(20261015)
    for index in range(360):
        if index==0: frame=first
        else:
            ok,frame=cap.read()
            if not ok: raise RuntimeError(f"Decode failure {index}")
        mask=cv2.imread(str(MASKS/"object"/f"{index:06d}.png"),0)>0
        person=cv2.imread(str(MASKS/"human"/f"{index:06d}.png"),0)>0
        current_xy,current_desc=detect_object(frame,mask,sift)
        matches=[]
        if current_desc is not None and len(current_desc)>=2:
            raw=[a for a,b in bf.knnMatch(desc,current_desc,k=2) if a.distance<.75*b.distance]
            seen=set()
            for m in sorted(raw,key=lambda a:a.distance):
                if m.trainIdx not in seen: matches.append(m);seen.add(m.trainIdx)
        q=np.array([m.queryIdx for m in matches],dtype=int)
        target=np.array([current_xy[m.trainIdx] for m in matches],dtype=float).reshape(-1,2)
        training=(q%5)!=0
        geometric=np.zeros(len(q),dtype=bool)
        if training.sum()>=6:
            homography,inliers=cv2.findHomography(xy[q[training]],target[training],cv2.RANSAC,2.,maxIters=1000,confidence=.999)
            if homography is not None:
                predicted=cv2.perspectiveTransform(xy[q].reshape(-1,1,2),homography).reshape(-1,2)
                geometric=np.linalg.norm(predicted-target,axis=1)<=3.
        train=training&geometric; test=(~training)&geometric
        proposal=None
        if train.sum()>=8:
            reference_T=data['object_to_camera'][index]
            proposals=[]
            for method in [cv2.SOLVEPNP_EPNP,cv2.SOLVEPNP_ITERATIVE]:
                rv=cv2.Rodrigues(reference_T[:3,:3])[0];tv=reference_T[:3,3].reshape(3,1).copy()
                solved,rv,tv,inliers=cv2.solvePnPRansac(xyz[q[train]],target[train],K,np.zeros(5),rv,tv,useExtrinsicGuess=True,iterationsCount=150,reprojectionError=2.,confidence=.999,flags=method)
                if solved and inliers is not None and len(inliers)>=8:
                    selected=inliers.ravel()
                    rv,tv=cv2.solvePnPRefineLM(xyz[q[train]][selected],target[train][selected],K,np.zeros(5),rv,tv)
                    R=cv2.Rodrigues(rv)[0]
                    angle=Rotation.from_matrix(R@data['object_to_camera'][0,:3,:3].T).magnitude()
                    if report['depth_anchor_m']*.65<=tv[2,0]<=report['depth_anchor_m']*1.35 and angle<=1.5:
                        projected=cv2.projectPoints(xyz[q[train]][selected],rv,tv,K,np.zeros(5))[0].reshape(-1,2)
                        error=np.linalg.norm(projected-target[train][selected],axis=1).mean()
                        proposals.append((len(selected),float(error),R,tv.ravel()))
            if proposals:
                selected=max(proposals,key=lambda x:(x[0],-x[1]))
                if selected[1]<=1.5:
                    proposal=np.eye(4);proposal[:3,:3]=selected[2];proposal[:3,3]=selected[3]
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(contours,key=cv2.contourArea)[:,0].astype(float)
        x,y,w,h=cv2.boundingRect(contour.astype(np.float32))
        x0,y0=max(0,x-40),max(0,y-40);x1,y1=min(mask.shape[1],x+w+40),min(mask.shape[0],y+h+40)
        cropped=mask[y0:y1,x0:x1]
        ignore=cv2.dilate(person.astype(np.uint8),np.ones((5,5),np.uint8))[y0:y1,x0:x1]>0
        distance=cv2.distanceTransform((~cropped).astype(np.uint8),cv2.DIST_L2,5)-cv2.distanceTransform(cropped.astype(np.uint8),cv2.DIST_L2,5)
        observed=sample_polygon(contour,32)
        observed_visible=map_coordinates(ignore.astype(float),(observed-np.array([x0,y0]))[:,::-1].T,order=0,mode="constant",cval=0)<.5
        evidence.append({"xyz":xyz[q[train]],"xy":target[train],"test_xyz":xyz[q[test]],"test_xy":target[test],"distance":distance,"ignore":ignore,"offset":np.array([x0,y0]),"observed":observed,"observed_visible":observed_visible,"proposal":proposal})
    cap.release()
    return evidence


def project(vertices,rotation,translation):
    points=vertices@rotation.T+translation
    homogeneous=points@K.T
    return homogeneous[:,:2]/homogeneous[:,2:]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--nfev",type=int,default=35)
    parser.add_argument("--initial-poses",type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"report.json").exists(): raise FileExistsError("Use fresh output")
    start=time.perf_counter()
    data=np.load(args.source/"object_candidate.npz",allow_pickle=False)
    base={k:data[k] for k in data.files}
    report=json.loads((args.source/"report.json").read_text())
    evidence=acquire(data,report)
    np.savez_compressed(args.output/"feature_evidence.npz",**{f"{name}_{i:03d}":item[name] for i,item in enumerate(evidence) for name in ["xyz","xy","test_xyz","test_xy"]})
    original=data["object_to_camera"].copy()
    R0=original[0,:3,:3].copy()
    proposed=original.copy()
    valid=[]
    for i,item in enumerate(evidence):
        if item['proposal'] is not None:
            proposed[i]=item['proposal'];valid.append(i)
    relative=Rotation.from_matrix(proposed[:,:3,:3]@R0.T).as_rotvec()
    initial=np.concatenate([relative,proposed[:,:3,3]],axis=1)
    # Interpolation is an initializer only; every final frame retains its own
    # image/contour objective. No interpolated target observations are created.
    if len(valid)>=2:
        for axis in range(6):
            initial[:,axis]=np.interp(np.arange(360),valid,initial[valid,axis])
    if args.initial_poses:
        warm=np.load(args.initial_poses,allow_pickle=False)
        if not np.array_equal(warm['camera_intrinsics'],data['camera_intrinsics']):raise ValueError('Warm-start camera mismatch')
        warmT=warm['object_to_camera']
        initial[:,:3]=Rotation.from_matrix(warmT[:,:3,:3]@R0.T).as_rotvec()
        initial[:,3:]=warmT[:,:3,3]
    initial[:,:3]=np.clip(initial[:,:3],-1.4,1.4)
    depth=report["depth_anchor_m"]
    vertices=data["vertices"]
    # Downsample mesh vertices only for the fitting outline; the exported mesh
    # retains all original triangles. Include extrema to keep the same hull.
    fit_vertices=vertices
    frame_sizes=[len(item["xy"])*2+80 for item in evidence]
    block_starts=np.r_[0,np.cumsum(frame_sizes)]
    image_rows=int(block_starts[-1]);n=360
    total_rows=image_rows+(n-2)*6+(n-1)*6+n+6
    sparsity=lil_matrix((total_rows,n*6),dtype=np.int8)
    for i in range(n): sparsity[block_starts[i]:block_starts[i+1],i*6:i*6+6]=1
    row=image_rows
    for i in range(n-2):
        sparsity[row:row+6,i*6:(i+3)*6]=1;row+=6
    for i in range(n-1):
        sparsity[row:row+6,i*6:(i+2)*6]=1;row+=6
    for i in range(n): sparsity[row+i,i*6+5]=1
    row+=n;sparsity[row:row+6,:6]=1
    iteration=[0]
    def image_robust(value):
        magnitude=np.abs(value)
        return np.where(magnitude<=3,value,np.sign(value)*np.sqrt(np.maximum(6*magnitude-9,0)))
    def residual(flat):
        params=flat.reshape(n,6)
        rotations=Rotation.from_rotvec(params[:,:3]).as_matrix()@R0
        chunks=[]
        for i,item in enumerate(evidence):
            if len(item["xy"]):
                diff=project(item["xyz"],rotations[i],params[i,3:])-item["xy"]
                chunks.append(image_robust(diff.ravel())*np.sqrt(15/max(len(diff),1)))
            xy=project(fit_vertices,rotations[i],params[i,3:])
            polygon=cv2.convexHull(xy.astype(np.float32)).reshape(-1,2).astype(float)
            boundary=sample_polygon(polygon,48)
            local=boundary-item["offset"]
            forward=map_coordinates(item["distance"],local[:,::-1].T,order=1,mode="constant",cval=50.)
            occluded=map_coordinates(item["ignore"].astype(float),local[:,::-1].T,order=0,mode="constant",cval=0)>.5
            forward[occluded]*=.1
            backward=point_polygon_distance(item["observed"],polygon)
            backward[~item["observed_visible"]]*=.1
            chunks.append(image_robust(np.r_[forward,backward])*.7)
        d2=params[2:]-2*params[1:-1]+params[:-2]
        d1=np.diff(params,axis=0)
        chunks.append((d2*np.array([1000,1000,1000,1200,1200,1200])).ravel())
        chunks.append((d1*np.array([.5,.5,.5,5,5,5])).ravel())
        chunks.append((params[:,5]-depth)*.5)
        chunks.append((params[0]-initial[0])*np.array([1,1,1,10,10,10]))
        return np.concatenate(chunks)
    lower=np.tile([-1.5,-1.5,-1.5,-1.,-1.,depth*.8],(n,1)).ravel()
    upper=np.tile([1.5,1.5,1.5,1.,1.,depth*1.2],(n,1)).ravel()
    init=np.clip(initial.ravel(),lower+1e-6,upper-1e-6)
    initial_res=residual(init)
    print(f"Temporal solve: {len(init)} variables, {len(initial_res)} residuals, train features {sum(len(x['xy']) for x in evidence)}, heldout {sum(len(x['test_xy']) for x in evidence)}, PnP initializer frames {len(valid)}",flush=True)
    solve_start=time.perf_counter()
    result=least_squares(residual,init,bounds=(lower,upper),jac_sparsity=sparsity.tocsr(),x_scale=np.tile([.2,.2,.2,.1,.1,.1],n),diff_step=1e-3,loss="linear",max_nfev=args.nfev,ftol=1e-4,xtol=1e-4,gtol=1e-4,verbose=1)
    params=result.x.reshape(n,6)
    transforms=original.copy();transforms[:,:3,:3]=Rotation.from_rotvec(params[:,:3]).as_matrix()@R0;transforms[:,:3,3]=params[:,3:]
    stats=[]
    for i,item in enumerate(evidence):
        mask=cv2.imread(str(MASKS/"object"/f"{i:06d}.png"),0)>0
        human=cv2.imread(str(MASKS/"human"/f"{i:06d}.png"),0)>0
        row={"frame":i,"train_points":len(item["xy"]),"heldout_points":len(item["test_xy"]),"initial_mask_iou":visible_iou(original[i],vertices,mask,human),"candidate_mask_iou":visible_iou(transforms[i],vertices,mask,human)}
        for key,T in [("initial",original[i]),("candidate",transforms[i])]:
            for partition,xyname,xyzname in [("train","xy","xyz"),("heldout","test_xy","test_xyz")]:
                errors=np.linalg.norm(project(item[xyzname],T[:3,:3],T[:3,3])-item[xyname],axis=1) if len(item[xyname]) else np.array([])
                row[f"{key}_{partition}_mean_px"]=float(errors.mean()) if len(errors) else None
        stats.append(row)
    def motion(T):
        steps=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
        d2=np.linalg.norm(T[2:,:3,3]-2*T[1:-1,:3,3]+T[:-2,:3,3],axis=1)
        angles=Rotation.from_matrix(T[1:,:3,:3]@T[:-1,:3,:3].transpose(0,2,1)).magnitude()
        return {"translation_step_max_m":float(steps.max()),"translation_step_p95_m":float(np.percentile(steps,95)),"translation_second_difference_max_m":float(d2.max()),"rotation_step_max_deg":float(np.rad2deg(angles.max())),"depth_range_m":[float(T[:,2,3].min()),float(T[:,2,3].max())]}
    base["object_to_camera"]=transforms;base["rotation"]=transforms[:,:3,:3];base["translation"]=transforms[:,:3,3]
    np.savez_compressed(args.output/"object_candidate.npz",**base)
    for file in ["object.obj","object.mtl"]: shutil.copy2(args.source/file,args.output/file)
    summary={"source_candidate":str(args.source),"optimization":"Joint 360-frame Lie-rotation/metric-translation least squares with original RGB SIFT training pixels, mask contours, cross-frame rotation/translation priors and weak assumed-depth anchor. No direct pose filtering.","heldout":"Reference SIFT query IDs divisible by5 excluded from pose objective; homography gate estimated from training matches only. These validate pixel tracking conditional on the hypothesized mesh, not true 3D or scale.","solver_success":bool(result.success),"solver_message":str(result.message),"nfev":result.nfev,"solve_seconds":time.perf_counter()-solve_start,"initial_cost":float(np.mean(initial_res**2)),"final_residual_mse":float(np.mean(result.fun**2)),"initial_motion":motion(original),"candidate_motion":motion(transforms),"mean_initial_mask_iou":float(np.mean([x['initial_mask_iou'] for x in stats])),"mean_candidate_mask_iou":float(np.mean([x['candidate_mask_iou'] for x in stats])),"mean_initial_heldout_px":float(np.mean([x['initial_heldout_mean_px'] for x in stats if x['initial_heldout_mean_px'] is not None])),"mean_candidate_heldout_px":float(np.mean([x['candidate_heldout_mean_px'] for x in stats if x['candidate_heldout_mean_px'] is not None])),"diagnostics":stats,"script_sha256":digest(Path(__file__)),"wall_seconds":time.perf_counter()-start}
    report.update({"temporal_refinement":summary})
    (args.output/"report.json").write_text(json.dumps(report,indent=2))
    (args.output/"temporal-summary.json").write_text(json.dumps({k:v for k,v in summary.items() if k!='diagnostics'},indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='diagnostics'},indent=2),flush=True)


if __name__=="__main__": main()
