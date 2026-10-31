"""Explicit monocular object-depth hypotheses, never reference alignment.

A common positive multiplier on object vertices and camera translations leaves
the object's perspective projection unchanged.  A contact-supported estimate
therefore adds a geometric assumption; image agreement alone cannot identify it.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def apply_projective_gauge(candidate: Mapping[str, np.ndarray], factor: float) -> dict[str, np.ndarray]:
    """Scale one rigid object's canonical mesh and every camera translation once."""
    if isinstance(factor, bool) or not np.isfinite(factor) or factor <= 0:
        raise ValueError("projective gauge factor must be finite and positive")
    result = {name: np.array(value, copy=True) for name, value in candidate.items()}
    vertices = np.asarray(candidate["vertices"])
    transforms = np.asarray(candidate["object_to_camera"])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("object vertices must be finite [V,3]")
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or not np.isfinite(transforms).all():
        raise ValueError("object transforms must be finite [T,4,4]")
    if not np.allclose(transforms[:, 3], [0, 0, 0, 1]):
        raise ValueError("invalid object homogeneous transform")
    result["vertices"] = vertices * factor
    result["object_to_camera"][:, :3, 3] *= factor
    if "translation" in result:
        if not np.allclose(result["translation"], transforms[:, :3, 3]):
            raise ValueError("duplicate object translations disagree")
        result["translation"] *= factor
    if "rotation" in result and not np.allclose(result["rotation"], transforms[:, :3, :3]):
        raise ValueError("duplicate object rotations disagree")
    return result


def projection(vertices: np.ndarray, transform: np.ndarray, camera: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices) @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]
    if np.any(points[:, 2] <= 0):
        raise ValueError("projected object must lie in front of the camera")
    homogeneous = points @ np.asarray(camera).T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def fit_projective_gauge(points_camera: np.ndarray, surface_camera: np.ndarray, frames: np.ndarray,
                         *, lower=0.94, upper=1.06, grid_step=0.00025,
                         data_sigma_m=0.01, prior_sigma=0.02) -> dict[str, Any]:
    """Frame-balanced robust scan under an explicitly fixed contact hypothesis.

The prior is a dimensionless squared distance from gauge one; the data term
is pseudo-Huber distance between fixed skin points and scaled fixed surface
points.  The two components and several prior sensitivities are returned.
"""
    p, q = np.asarray(points_camera, float), np.asarray(surface_camera, float)
    ids = np.asarray(frames)
    if p.ndim != 2 or p.shape[1] != 3 or q.shape != p.shape or not len(p):
        raise ValueError("nonempty paired skin/surface points must be [N,3]")
    if ids.shape != (len(p),) or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("point frame identities must be nonnegative integers")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(p[:, 2] <= 0) or np.any(q[:, 2] <= 0):
        raise ValueError("contact-supported points must be finite and in front of the camera")
    if not 0 < lower < 1 < upper or not np.isfinite([lower, upper, grid_step, data_sigma_m, prior_sigma]).all() or min(grid_step, data_sigma_m, prior_sigma) <= 0:
        raise ValueError("invalid gauge design bounds or scales")
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weight = 1.0 / counts[inverse]
    weight /= weight.sum()
    grid = np.linspace(lower, upper, int(round((upper - lower) / grid_step)) + 1)
    distance = np.linalg.norm(p[None] - grid[:, None, None] * q[None], axis=-1)
    data = ((np.sqrt(1 + (distance / data_sigma_m) ** 2) - 1) * weight[None]).sum(1)
    prior = ((grid - 1) / prior_sigma) ** 2
    total = data + prior
    best = int(total.argmin())
    sensitivity = []
    for sigma in (prior_sigma / 2, prior_sigma, prior_sigma * 2):
        component = ((grid - 1) / sigma) ** 2
        index = int((data + component).argmin())
        sensitivity.append({"prior_sigma": sigma, "factor": float(grid[index]),
                            "data_objective": float(data[index]), "prior_objective": float(component[index]),
                            "at_design_bound": index in (0, len(grid) - 1)})
    ratio = (p * q).sum(1) / (q * q).sum(1)
    data_best = int(data.argmin())
    return {
        "factor": float(grid[best]), "data_only_factor": float(grid[data_best]),
        "median_unconstrained_projection_factor": float(np.median(ratio)),
        "factor_ratio_p10_p90": np.percentile(ratio, [10, 90]).tolist(),
        "samples": len(p), "frames": len(counts),
        "selection_is_frozen_during_scan": True,
        "data_sigma_m": data_sigma_m, "prior_sigma": prior_sigma,
        "bounds": [lower, upper], "grid_step": float(grid[1] - grid[0]),
        "at_design_bound": best in (0, len(grid) - 1),
        "best_objective": {"data": float(data[best]), "prior": float(prior[best]), "total": float(total[best])},
        "prior_sensitivity": sensitivity,
        "curve": [{"factor": float(g), "data": float(d), "prior": float(r), "total": float(t)}
                  for g, d, r, t in zip(grid, data, prior, total)],
        "interpretation": "Conditional contact-supported monocular gauge, not calibrated depth, true contact or R1 reference alignment",
    }


def forward_ray_interval(points: np.ndarray, normals_camera: np.ndarray, offsets_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Near/far z on camera rays through points for a closed convex object."""
    p = np.asarray(points, float)
    normals, offsets = np.asarray(normals_camera, float), np.asarray(offsets_camera, float)
    if p.ndim != 2 or p.shape[1] != 3 or np.any(p[:, 2] <= 0) or not np.isfinite(p).all():
        raise ValueError("ray points must be finite [N,3] with positive z")
    if normals.ndim != 2 or normals.shape[1] != 3 or offsets.shape != (len(normals),) or not np.isfinite(normals).all() or not np.isfinite(offsets).all():
        raise ValueError("convex camera planes must be finite aligned arrays")
    direction = p / p[:, 2:]
    denominator = direction @ normals.T
    parallel = np.abs(denominator) < 1e-12
    fraction = np.divide(offsets[None], denominator, out=np.zeros_like(denominator), where=~parallel)
    near = np.maximum(np.where(denominator < -1e-12, fraction, -np.inf).max(1), 0)
    far = np.where(denominator > 1e-12, fraction, np.inf).min(1)
    valid = (far >= near) & (far > 0) & ~np.any(parallel & (offsets[None] < 0), axis=1)
    return near, far, valid


def far_side_rigid_shift(points: np.ndarray, wrist: np.ndarray, normals_camera: np.ndarray,
                         offsets_camera: np.ndarray, margin_m=0.003) -> dict[str, float | int]:
    """Conditional rigid translation needed to put overlapping skin past the box.

The translation direction preserves the wrist's image projection.  This is a
fixed-articulation feasibility diagnostic, not a native inverse-kinematics solve
or evidence that all hidden skin truly belongs behind the object.
"""
    p, w = np.asarray(points, float), np.asarray(wrist, float)
    normals, offsets = np.asarray(normals_camera, float), np.asarray(offsets_camera, float)
    if w.shape != (3,) or not np.isfinite(w).all() or w[2] <= 0 or not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("invalid wrist ray or far-side slack")
    _, far, overlap = forward_ray_interval(p, normals, offsets)
    behind = overlap & (p[:, 2] >= far + margin_m)
    direction = w / w[2]
    denominator = normals @ direction
    upper = denominator > 1e-12
    if not upper.any():
        raise ValueError("convex object has no exit in wrist-forward direction")
    exits = ((offsets[upper][None] - p @ normals[upper].T) / denominator[upper][None]).min(1)
    required = np.maximum(exits[overlap] + margin_m, 0)
    delta_z = float(required.max()) if len(required) else 0.0
    shifted = p + delta_z * direction
    old_uv, new_uv = p[:, :2] / p[:, 2:], shifted[:, :2] / shifted[:, 2:]
    return {"projected_skin_overlap_count": int(overlap.sum()), "already_far_side_count": int(behind.sum()),
            "rigid_delta_z_m": delta_z, "rigid_wrist_displacement_m": float(delta_z * np.linalg.norm(direction)),
            "required_delta_z_p95_m": float(np.percentile(required, 95)) if len(required) else 0.0,
            "normalized_projection_max_change": float(np.linalg.norm(old_uv - new_uv, axis=1).max()),
            "wrist_projection_change": 0.0}
