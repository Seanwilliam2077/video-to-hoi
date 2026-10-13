"""Bounded numerical checks for the proposed v2 representation.

These checks neither decode MHR nor establish rig identity, native parameter
units, scene provenance, mesh topology, or reconstruction quality. Those require
the pinned artifacts and real decoder/round-trip evidence. In particular there
is intentionally no conversion from the decomposed v1 human representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .artifacts import ContractError


_ROTATION_ATOL = 1e-6


def _real_array(value, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{name} must be a real numeric array") from exc
    if array.dtype.kind not in "fiu":
        raise ContractError(f"{name} must be a real numeric array, not {array.dtype}")
    if not np.isfinite(array).all():
        raise ContractError(f"{name} must contain only finite values")
    # Arithmetic must not truncate fractions or overflow small integer dtypes.
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ContractError(f"{name} cannot be represented as finite float64 values")
    return array


def _frame_ids(value, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{name} must be a one-dimensional integer frame map") from exc
    if array.ndim != 1 or array.dtype.kind not in "iu" or len(array) == 0:
        raise ContractError(f"{name} must be a nonempty one-dimensional integer frame map")
    return array


def _full_frame_count(frame_ids, expected_frame_ids) -> int:
    expected = _frame_ids(expected_frame_ids, "expected_frame_ids")
    actual = _frame_ids(frame_ids, "frame_ids")
    if not np.array_equal(expected, np.arange(len(expected), dtype=np.int64)):
        raise ContractError("expected_frame_ids must cover the untrimmed input as 0..T-1")
    if not np.array_equal(actual, expected):
        raise ContractError("frame_ids must exactly match all original frames in presentation order")
    return len(expected)


def _shape(array: np.ndarray, expected: tuple, name: str) -> None:
    if array.shape != expected:
        raise ContractError(f"{name} has shape {array.shape}, expected {expected}")


def _proper_rotations(rotations: np.ndarray, name: str) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        orthogonal = rotations @ np.swapaxes(rotations, -1, -2)
        determinant = np.linalg.det(rotations)
    if not np.allclose(orthogonal, np.eye(3), atol=_ROTATION_ATOL, rtol=0) or not np.allclose(
        determinant, 1.0, atol=_ROTATION_ATOL, rtol=0
    ):
        raise ContractError(f"{name} must contain proper SO(3) rotations")


def validate_native_human(pose, shape, scales, frame_ids, expected_frame_ids) -> None:
    """Check finite native-array shapes and full source-frame coverage only.

    Pose has 136 native parameters per frame; shape45 and scales68 are static.
    No per-parameter range or metre/radian assumption is made: units and valid
    parameter domains belong to the pinned native decoder. Passing this helper
    does not prove a compatible rig, a valid decode, or submission compliance.
    """
    count = _full_frame_count(frame_ids, expected_frame_ids)
    for value, dimensions, name in (
        (pose, (count, 136), "pose"), (shape, (45,), "shape"), (scales, (68,), "scales")
    ):
        _shape(_real_array(value, name), dimensions, name)


def _object_arrays(vertices, rotations, translations, scale_mode, scale):
    vertices = _real_array(vertices, "vertices")
    rotations = _real_array(rotations, "rotations")
    translations = _real_array(translations, "translations")
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        raise ContractError("vertices must have shape (V, 3), with at least one vertex")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or len(rotations) == 0:
        raise ContractError("rotations must have shape (T, 3, 3), with at least one frame")
    _shape(translations, (len(rotations), 3), "translations")
    _proper_rotations(rotations, "rotations")
    if not isinstance(scale_mode, str) or scale_mode not in ("baked_metric", "explicit_uniform"):
        raise ContractError("scale_mode must be baked_metric or explicit_uniform")
    scale_array = _real_array(scale, "scale")
    if scale_array.ndim != 0 or scale_array.item() <= 0:
        raise ContractError("scale must be one positive finite constant, never a per-frame array")
    scale_value = float(scale_array)
    if scale_mode == "baked_metric" and scale_value != 1.0:
        raise ContractError("baked_metric requires scale exactly 1; provenance scale must not be reapplied")
    return vertices, rotations, translations, scale_value


def validate_object_trajectory(
    vertices, rotations, translations, frame_ids, expected_frame_ids, scale_mode, scale
) -> None:
    """Check rigid-object numerics and coverage; asset/pose hashes are checked elsewhere."""
    count = _full_frame_count(frame_ids, expected_frame_ids)
    _, rotations, _, _ = _object_arrays(vertices, rotations, translations, scale_mode, scale)
    _shape(rotations, (count, 3, 3), "rotations")


def world_object_vertices(vertices, rotations, translations, *, scale_mode, scale) -> np.ndarray:
    """Compute P[t] = s R[t] V + t[t], returning (T, V, 3) in metres.

    Array inputs use row-major point storage. For baked_metric, provenance
    scale is already in V and the only accepted runtime scale is exactly one.
    This array helper cannot verify artifact bindings or original-frame IDs.
    """
    vertices, rotations, translations, scale = _object_arrays(
        vertices, rotations, translations, scale_mode, scale
    )
    with np.errstate(over="ignore", invalid="ignore"):
        world = scale * np.einsum("tij,vj->tvi", rotations, vertices) + translations[:, None, :]
    if not np.isfinite(world).all():
        raise ContractError("world vertices overflowed the finite scene representation")
    return world


@dataclass(frozen=True)
class RebasedObject:
    """A geometry/pose pair that must be rebound and revalidated before reuse.

    This numerical result is not a SceneState or an authorization to reuse prior
    trajectory/contact/export reports, even when world vertices are preserved.
    """

    vertices: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    scale_mode: str
    scale: float
    requires_revalidation: bool = field(default=True, init=False)


def rebase_object_canonical(
    vertices, rotations, translations, T_new_from_old, *, scale_mode, scale
) -> RebasedObject:
    """Apply a rigid canonical-frame change to both mesh and every pose.

    If V_new = A V_old + b, then R_new = R_old A.T and
    t_new = t_old - s R_new b. Translation b uses the same canonical units as V.
    The function rejects scale/shear/reflection in T_new_from_old and never
    edits the supplied arrays. It is not an acceleration-metric optimization.
    """
    vertices, rotations, translations, scale = _object_arrays(
        vertices, rotations, translations, scale_mode, scale
    )
    transform = _real_array(T_new_from_old, "T_new_from_old")
    _shape(transform, (4, 4), "T_new_from_old")
    if not np.array_equal(transform[3], [0, 0, 0, 1]):
        raise ContractError("T_new_from_old must have rigid homogeneous last row [0, 0, 0, 1]")
    rotation, offset = transform[:3, :3], transform[:3, 3]
    _proper_rotations(rotation, "T_new_from_old rotation")
    with np.errstate(over="ignore", invalid="ignore"):
        new_vertices = vertices @ rotation.T + offset
        new_rotations = rotations @ rotation.T
        new_translations = translations - scale * np.einsum("tij,j->ti", new_rotations, offset)
    _object_arrays(new_vertices, new_rotations, new_translations, scale_mode, scale)
    return RebasedObject(new_vertices, new_rotations, new_translations, scale_mode, scale)


def _image_transform(transform) -> np.ndarray:
    transform = _real_array(transform, "T_processed_from_source")
    _shape(transform, (3, 3), "T_processed_from_source")
    if not np.array_equal(transform[2], [0, 0, 1]):
        raise ContractError("image transform must be affine with last row [0, 0, 1]; nonlinear warps are separate")
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError as exc:
        raise ContractError("image transform must be nonsingular") from exc
    if not np.isfinite(inverse).all():
        raise ContractError("image transform must have a finite inverse")
    return transform


def validate_image_transform(transform, source_size, processed_size) -> None:
    """Check an affine pixel-coordinate map and explicit (width, height) sizes.

    Pixel-centre convention, valid regions, distortion and interpolation remain
    explicit observation metadata; this function cannot infer any of them.
    """
    _image_transform(transform)
    for value, name in ((source_size, "source_size"), (processed_size, "processed_size")):
        try:
            size = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContractError(f"{name} must contain positive integer width and height") from exc
        if size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0):
            raise ContractError(f"{name} must contain positive integer width and height")


def transform_image_points(points, transform) -> np.ndarray:
    """Map (N, 2) pixel-centre coordinates with an explicit affine transform."""
    points = _real_array(points, "image points")
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ContractError("image points must have shape (N, 2)")
    transform = _image_transform(transform)
    with np.errstate(over="ignore", invalid="ignore"):
        result = points @ transform[:2, :2].T + transform[:2, 2]
    if not np.isfinite(result).all():
        raise ContractError("transformed image points must remain finite")
    return result


def transform_intrinsics(intrinsics, transform) -> np.ndarray:
    """Return K_processed = T_processed_from_source K_source.

    This transforms a pinhole projection matrix in pixel coordinates. It does
    not estimate focal length, remove distortion or infer a crop convention.
    """
    intrinsics = _real_array(intrinsics, "intrinsics")
    _shape(intrinsics, (3, 3), "intrinsics")
    if not np.array_equal(intrinsics[2], [0, 0, 1]) or intrinsics[1, 0] != 0:
        raise ContractError("source intrinsics must be an upper triangular pinhole matrix")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ContractError("source intrinsics must have positive focal lengths")
    with np.errstate(over="ignore", invalid="ignore"):
        result = _image_transform(transform) @ intrinsics
    if not np.isfinite(result).all():
        raise ContractError("transformed intrinsics must remain finite")
    return result
