"""Independent synthetic camera/ray checks; no challenge assets or model inference."""
import importlib.util
from pathlib import Path
import numpy as np
import trimesh
import torch
from pytorch3d.transforms import axis_angle_to_matrix
spec=importlib.util.spec_from_file_location('pose_trial',Path(__file__).parents[1]/'optimize_object_pose_v2.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def test_nearest_surface_and_original_pixel_units():
 m=trimesh.creation.box(extents=(1,1,1));K=np.diag([100.,100.,1.]);T=np.eye(4);T[2,3]=3
 X,ids,faces=module.intersect(np.array([[0.,0.],[10.,0.],[200.,0.]]),K,T,np.asarray(m.vertices),np.asarray(m.faces))
 assert ids.tolist()==[0,1]
 np.testing.assert_allclose(X,[[0,0,-.5],[.25,0,-.5]],atol=1e-9)
def test_rotated_object_binding_matches_camera_frame():
 m=trimesh.creation.box(extents=(1,1,1));K=np.diag([100.,100.,1.]);T=np.eye(4);T[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]];T[2,3]=3
 X,ids,faces=module.intersect(np.array([[0.,0.],[10.,0.]]),K,T,np.asarray(m.vertices),np.asarray(m.faces))
 np.testing.assert_allclose(X,[[.5,0,0],[.5,0,.25]],atol=1e-9)
def test_behind_camera_is_rejected():
 m=trimesh.creation.box();T=np.eye(4);T[2,3]=-3
 X,ids,faces=module.intersect(np.array([[0.,0.]]),np.eye(3),T,np.asarray(m.vertices),np.asarray(m.faces))
 assert len(ids)==0
def test_zero_tangent_has_finite_useful_rotation_gradient():
 a=torch.zeros((1,3),requires_grad=True);R=axis_angle_to_matrix(a);(R[0,0,1]+R[0,0,2]).backward()
 assert torch.isfinite(a.grad).all() and float(a.grad.abs().sum())>0
 np.testing.assert_allclose(R.detach().numpy()[0],np.eye(3),atol=1e-7)
