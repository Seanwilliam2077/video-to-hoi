"""Ambiguous perspective circle poses from a visible source ellipse."""
import numpy as np


def ellipse_poses(mask, camera, vertices):
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    points=np.column_stack(np.where(mask)[::-1]).astype('float32')
    if len(points)<20:
        return np.empty((0,4,4))
    if np.linalg.matrix_rank(points-points.mean(0),tol=1e-5)<2:
        return np.empty((0,4,4))
    rays=cv2.undistortPoints(points[:,None],np.asarray(camera,dtype=float),None)[:,0]
    center,diameters,angle=cv2.fitEllipse(rays.reshape(-1,1,2))
    center=np.asarray(center); diameters=np.asarray(diameters)
    if not np.isfinite([*center,*diameters,angle]).all() or diameters.min()<=1e-6:
        return np.empty((0,4,4))
    theta=np.deg2rad(angle)
    axes=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    axis=axes[:,np.argmax(diameters)]
    major,minor=np.sort(diameters)[::-1]
    ratio=np.clip(minor/major,.03,1.)
    canonical_center=(vertices.min(0)+vertices.max(0))/2
    radius=np.median(np.linalg.norm(vertices[:,:2]-canonical_center[:2],axis=1))
    angles=np.linspace(0,2*np.pi,96,endpoint=False)
    circle=np.column_stack([radius*np.cos(angles),radius*np.sin(angles),np.zeros(len(angles))])
    z=2*radius/major
    if z<=.05:
        return np.empty((0,4,4))
    poses=[]
    def residual(parameters):
        r=Rotation.from_rotvec(parameters[:3]).as_matrix()
        xyz=circle@r.T+parameters[3:]
        projected=xyz[:,:2]/np.maximum(xyz[:,2:],.01)
        ellipse=((projected-center)@axes)/(diameters/2)
        return np.linalg.norm(ellipse,axis=1)-1
    for sign in (-1,1):
        u=np.r_[axis,0.]
        v=np.array([-axis[1]*ratio,axis[0]*ratio,sign*np.sqrt(1-ratio**2)])
        rotation=np.column_stack([u,v,np.cross(u,v)])
        initial=np.r_[Rotation.from_matrix(rotation).as_rotvec(),np.r_[center,1.]*z]
        fit=least_squares(residual,initial,max_nfev=150,bounds=([-np.inf]*5+[.05],[np.inf]*6))
        if not fit.success or not np.isfinite(fit.x).all():
            continue
        rotation=Rotation.from_rotvec(fit.x[:3]).as_matrix()
        for factor in (.95,1.,1.05):
            pose=np.eye(4)
            pose[:3,:3]=rotation
            pose[:3,3]=fit.x[3:]*factor-rotation@canonical_center
            poses.append(pose)
    return np.asarray(poses) if poses else np.empty((0,4,4))
