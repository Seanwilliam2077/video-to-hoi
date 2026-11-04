"""Observed ellipse hypotheses for thin bpy rings, followed by the same bundle.

No gate is relaxed. Weak-perspective circle poses are ambiguous hypotheses, and
the original perspective silhouette/depth costs, source tracks and FP candidates
still decide which anchors are admissible. Circle in-plane yaw is unobservable.
"""
from pathlib import Path
import json
import sys

import numpy as np


def ellipse_poses(mask, camera, vertices):
    import cv2
    points = np.column_stack(np.where(mask)[::-1]).astype('float32')
    if len(points) < 20:
        return np.empty((0,4,4))
    rays = cv2.undistortPoints(points[:,None], np.asarray(camera,dtype=float), None)[:,0]
    center, diameters, angle = cv2.fitEllipse(rays.reshape(-1,1,2))
    diameters = np.asarray(diameters)
    if not np.isfinite([*center,*diameters,angle]).all() or diameters.min() <= 1e-6:
        return np.empty((0,4,4))
    theta = np.deg2rad(angle)
    axis = np.array([np.cos(theta),np.sin(theta)])
    if diameters[1] > diameters[0]:
        axis = np.array([-axis[1],axis[0]])
    major, minor = np.sort(diameters)[::-1]
    ratio = np.clip(minor/major,.03,1.)
    radius = (np.ptp(vertices[:,0])+np.ptp(vertices[:,1]))/4
    z = 2*radius/major
    canonical_center = np.mean([vertices.min(0),vertices.max(0)],axis=0)
    poses = []
    for sign in (-1,1):
        u = np.r_[axis,0.]
        v = np.array([-axis[1]*ratio,axis[0]*ratio,sign*np.sqrt(1-ratio**2)])
        rotation = np.column_stack([u,v,np.cross(u,v)])
        for factor in (.8,1.,1.2):
            distance = z*factor
            pose=np.eye(4)
            pose[:3,:3]=rotation
            pose[:3,3]=np.r_[center,1.]*distance-rotation@canonical_center
            poses.append(pose)
    return np.asarray(poses)


def main():
    from v2hoi.native import sha256
    from v2hoi import temporal_object_runtime as runtime
    from prep import run_foundationpose_mhr_export as fp
    import torch
    from v2hoi import ring_geometry
    # Refine the weak-perspective seeds against the full perspective ellipse.
    global ellipse_poses
    ellipse_poses=ring_geometry.ellipse_poses
    start_hash=sha256(Path(__file__))
    geometry_source=Path(ring_geometry.__file__).resolve()
    geometry_hash=sha256(geometry_source)
    factory=fp._pose_estimator
    added=[]
    def augmented(mesh, debug):
        estimator=factory(mesh,debug)
        register=estimator.register
        def call(*values,**keywords):
            best=register(*values,**keywords)
            hypotheses=ellipse_poses(keywords['ob_mask'],keywords['K'],np.asarray(mesh.vertices))
            if len(hypotheses):
                transform=estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
                centered=hypotheses@np.linalg.inv(transform)
                estimator.poses=torch.cat([torch.as_tensor(centered,dtype=estimator.poses.dtype,device=estimator.poses.device),estimator.poses])
            added.append(dict(registered_frame_call=len(added),ellipse_hypotheses=len(hypotheses)))
            return best
        estimator.register=call
        return estimator
    fp._pose_estimator=augmented
    runtime.main()
    if '--worker' in sys.argv:
        output=Path(sys.argv[sys.argv.index('--output')+1])
        if sha256(Path(__file__)) != start_hash or sha256(geometry_source)!=geometry_hash:
            raise ValueError('Ellipse initializer changed during inference')
        path=output/'result.json'
        report=json.loads(path.read_text())
        report['inputs'][str(Path(__file__).resolve())]=start_hash
        report['inputs'][str(geometry_source)]=geometry_hash
        report['thin_ring_initialization']=dict(enabled=True,calls=added,
            evidence='Automatically fitted source ellipse, weak-perspective seeds refined by perspective circle projection',
            silhouette_admission_threshold_unchanged=True,authored_poses=False)
        path.write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
