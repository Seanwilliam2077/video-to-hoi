"""Metric primitives. Inputs in metres and frames; callers convert units."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def chamfer(a: np.ndarray, b: np.ndarray, workers: int = -1) -> float:
    """Symmetric Chamfer: mean of the two directed mean nearest-neighbour distances."""
    d_ab, _ = cKDTree(b).query(a, workers=workers)
    d_ba, _ = cKDTree(a).query(b, workers=workers)
    return 0.5 * (float(d_ab.mean()) + float(d_ba.mean()))


def icp_residual(src: np.ndarray, dst: np.ndarray, iterations: int = 30, workers: int = -1) -> float:
    """Chamfer between src and dst after centring and rigid point-to-point ICP of src onto dst.

    Starts from the given orientation, so the rotation error must be small
    enough for ICP to converge (true once poses are roughly right).
    """
    from v2hoi.geometry import umeyama

    tree = cKDTree(dst)
    cur = np.asarray(src, dtype=np.float64)
    cur = cur - cur.mean(0) + dst.mean(0)
    prev = np.inf
    for _ in range(iterations):
        dist, idx = tree.query(cur, workers=workers)
        _, R, t = umeyama(cur, dst[idx])
        cur = cur @ R.T + t
        if prev - dist.mean() < 1e-7:
            break
        prev = dist.mean()
    return chamfer(cur, dst, workers)


def _triplets(valid: np.ndarray) -> np.ndarray:
    """Frames t (1..T-2) where t-1, t, t+1 are all valid."""
    return valid[:-2] & valid[1:-1] & valid[2:]


def accel_error(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Mean |a_pred - a_gt| with a_t = x_{t+1} - 2 x_t + x_{t-1}. x: (T, ..., 3)."""
    if valid is None:
        valid = np.ones(len(gt), dtype=bool)
    ok = _triplets(valid)
    if not ok.any():
        return float("nan")

    def acc(x):
        return x[2:] - 2 * x[1:-1] + x[:-2]

    diff = np.linalg.norm(acc(pred)[ok] - acc(gt)[ok], axis=-1)
    return float(diff.mean())


def angular_accel_error(R_pred: np.ndarray, R_gt: np.ndarray, valid: np.ndarray) -> float:
    """Mean |alpha_pred - alpha_gt| in rad/frame^2, using world-frame angular velocity.

    omega_t = log(R_{t+1} R_t^T), so a constant offset in the object's canonical
    frame (R -> R C) cancels; only the trajectory matters.
    """
    ok = _triplets(valid)
    if not ok.any():
        return float("nan")

    def omega(R):
        R = np.where(valid[:, None, None], R, np.eye(3))
        rel = R[1:] @ np.swapaxes(R[:-1], 1, 2)
        return Rotation.from_matrix(rel).as_rotvec()

    a_pred = np.diff(omega(R_pred), axis=0)
    a_gt = np.diff(omega(R_gt), axis=0)
    return float(np.linalg.norm(a_pred[ok] - a_gt[ok], axis=-1).mean())


def penetration_depth(points_obj: np.ndarray, sdf, workers: int = -1) -> float:
    """Deepest point inside the object (metres, >= 0). points_obj in the object's frame."""
    d = sdf(points_obj, workers)
    return float(max(0.0, -d.min())) if len(d) else 0.0


def orient_plane(plane: np.ndarray, free_points: np.ndarray) -> np.ndarray:
    """Flip plane [a, b, c, d] so that most of ``free_points`` have positive distance."""
    plane = np.asarray(plane, dtype=np.float64)
    d = free_points.reshape(-1, 3) @ plane[:3] + plane[3]
    return plane if np.median(d) >= 0 else -plane


def plane_penetration(points: np.ndarray, plane: np.ndarray) -> float:
    """Deepest point below an oriented plane (metres, >= 0)."""
    d = points @ plane[:3] + plane[3]
    return float(max(0.0, -d.min()))
