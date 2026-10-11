"""Rigid orientation hypotheses for observation-only review, without rescaling."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def one_frame_vertices(value):
    """Normalize a single mesh to [1,V,3], never interpreting vertices as frames."""
    vertices=np.asarray(value,dtype=np.float32)
    if vertices.ndim==2:
        vertices=vertices[None]
    if vertices.ndim!=3 or vertices.shape[0]!=1 or vertices.shape[2]!=3 or vertices.shape[1]<3 or not np.isfinite(vertices).all():
        raise ValueError("Expected one finite mesh with shape [V,3] or [1,V,3]")
    return vertices


def rigid_pose(value):
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite 4 x 4 rigid pose")
    r = pose[:3, :3]
    if (not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-4, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-4, rtol=0)):
        raise ValueError("Pose must contain a proper rotation, without scale or shear")
    return pose


def rotate_about_mesh_center(pose, center, delta):
    """Apply an object-local rotation while retaining the camera-space center.

    The translation of the canonical origin generally changes. Retaining it
    instead would move a mesh whose canonical origin is not its center.
    """
    pose = rigid_pose(pose)
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Expected a finite mesh center")
    update = np.eye(4)
    update[:3, :3] = np.asarray(delta, dtype=np.float64)
    rigid_pose(update)
    result = pose.copy()
    result[:3, :3] = pose[:3, :3] @ update[:3, :3]
    result[:3, 3] = pose[:3, :3] @ center + pose[:3, 3] - result[:3, :3] @ center
    return result


def orientation_hypotheses(pose, center):
    """24 quarter-turn hypotheses; these are candidates, not assumed symmetries."""
    rotations = Rotation.create_group("O").as_matrix()
    identity = np.argmin(np.linalg.norm(rotations - np.eye(3), axis=(1, 2)))
    order = [int(identity)] + [i for i in range(len(rotations)) if i != identity]
    return np.stack([rotate_about_mesh_center(pose, center, rotations[i]) for i in order])
