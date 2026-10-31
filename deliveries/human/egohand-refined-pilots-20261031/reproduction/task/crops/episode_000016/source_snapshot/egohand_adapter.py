"""Project sparse EgoHand geometry into original-frame MHR fitting proposals.

These are model-generated geometric targets, not independent image landmarks,
visibility labels, calibrated confidence, or reconstruction ground truth.
No model package or challenge asset is imported by this module.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


# Verified against IsaacCapture 9dfc9f25fe0b8c69a2f4f0bc01eb669097f7d1b9:
# inference/postprocessing/soma_constants.py and
# keypoint_2_soma_converter/src/soma_sequence_fit/geometry.py DEFAULT_JOINT_MAP.
# Non-thumb SOMA chains start with an extra metacarpal (5,10,15,20).
# This is an approximate anatomical correspondence, not decoder equivalence.
SOMA25_TO_LANDMARKS21 = np.array(
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24],
    dtype=np.int64,
)
SOMA25_TO_LANDMARKS21.setflags(write=False)
SIDE_ORDER = ("left", "right")


def validate_frame_indices(frame_indices: Any) -> np.ndarray:
    """Require a complete original-video timeline, never a reindexed subset."""
    ids = np.asarray(frame_indices)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids):
        raise ValueError("frame_indices must be a nonempty integer vector")
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError("frame_indices must cover consecutive original frames starting at zero")
    return ids.astype(np.int64, copy=True)


def validate_intrinsics(intrinsics: Any) -> np.ndarray:
    """Accept an explicitly supplied, finite pinhole camera in source pixels."""
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsics must be a finite 3x3 pinhole matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.array_equal(matrix[2], [0, 0, 1]):
        raise ValueError("intrinsics must have positive focal lengths and bottom row [0,0,1]")
    if matrix[1, 0] != 0:
        raise ValueError("intrinsics must use the supported upper-triangular pinhole convention")
    return matrix.copy()


def project_camera_points(points: Any, intrinsics: Any) -> tuple[np.ndarray, np.ndarray]:
    """Project OpenCV camera geometry; invalid/behind-camera points stay NaN."""
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim < 2 or xyz.shape[-1] != 3:
        raise ValueError("camera points must end in [landmarks,3]")
    matrix = validate_intrinsics(intrinsics)
    valid = np.isfinite(xyz).all(-1) & (xyz[..., 2] > 0)
    pixels = np.full(xyz.shape[:-1] + (2,), np.nan, dtype=np.float64)
    homogeneous = xyz[valid] @ matrix.T
    pixels[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
    valid &= np.isfinite(pixels).all(-1)
    pixels[~valid] = np.nan
    return pixels, valid


def parse_prediction_frame_names(frame_names: Any, expected_count: int) -> np.ndarray:
    """Resolve numeric string IDs without assuming row order or filling gaps."""
    names = np.asarray(frame_names)
    if names.ndim != 1 or names.dtype.kind not in "US":
        raise ValueError("frame_names must be a string vector of original frame IDs")
    decoded = []
    for name in names.tolist():
        if isinstance(name, bytes):
            name = name.decode("ascii", errors="strict")
        if not name or not name.isascii() or not name.isdecimal():
            raise ValueError("frame_names must contain only decimal original frame IDs")
        decoded.append(int(name))
    if len(set(decoded)) != len(decoded):
        raise ValueError("duplicate original frame ID in EgoHand output")
    if any(index < 0 or index >= expected_count for index in decoded):
        raise ValueError("EgoHand frame ID outside the original video")
    return np.asarray(decoded, dtype=np.int64)


def _prediction_geometry(prediction: Mapping[str, Any], expected_count: int) -> tuple[np.ndarray, np.ndarray]:
    names = parse_prediction_frame_names(prediction["frame_names"], expected_count)
    count = np.asarray(prediction["num_frames"])
    if count.shape != () or count.dtype.kind not in "iu" or int(count) != len(names):
        raise ValueError("num_frames must equal the number of saved prediction rows")
    if not len(names):
        # The upstream empty-hand file deliberately omits all geometry arrays.
        return names, np.empty((0, 25, 3), dtype=np.float64)
    local = np.asarray(prediction["joints"], dtype=np.float64)
    translation = np.asarray(prediction["cam_t"], dtype=np.float64)
    if local.shape != (len(names), 25, 3) or translation.shape != (len(names), 3):
        raise ValueError("expected wrist-relative joints[N,25,3] and cam_t[N,3]")
    return names, local + translation[:, None, :]


def adapt_egohand_proposals(
    predictions: Mapping[str, Mapping[str, Any] | None],
    *,
    frame_indices: Any,
    intrinsics: Any,
    image_size: Any,
    crop_boxes: Any,
    variant: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Keep all original rows, admitting targets only by fixed geometric rules.

    ``image_size`` is [width,height]; boxes are original-pixel xyxy with an
    exclusive right/bottom boundary. Binary weights only mean finite positive
    depth and a projection inside both the image and the frozen crop. They do
    not mean the finger is visible. Optional model presence/sigma values are
    preserved as diagnostics and never converted to invented visibility labels.
    EgoHand has already unmirrored outputs into anatomical source-camera space;
    this adapter must not mirror either side again.
    """
    if variant not in {"raw", "smoothed"}:
        raise ValueError("variant must explicitly be raw or smoothed")
    if set(predictions) - set(SIDE_ORDER):
        raise ValueError("predictions must use anatomical left/right side keys")
    ids = validate_frame_indices(frame_indices)
    matrix = validate_intrinsics(intrinsics)
    size = np.asarray(image_size)
    if size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0):
        raise ValueError("image_size must be integer [width,height]")
    boxes = np.asarray(crop_boxes)
    if boxes.shape != (len(ids), 2, 4) or not np.isfinite(boxes).all():
        raise ValueError("crop_boxes must be finite [T,left/right,4] original-pixel xyxy")
    if np.any(boxes[..., 2:] <= boxes[..., :2]):
        raise ValueError("crop_boxes must have positive width and height")

    n = len(ids)
    points = np.zeros((n, 2, 21, 2), dtype=np.float32)
    confidence = np.zeros((n, 2, 21), dtype=np.float32)
    geometry = np.zeros((n, 2, 21, 3), dtype=np.float64)
    camera_valid = np.zeros((n, 2, 21), dtype=bool)
    present = np.zeros((n, 2), dtype=bool)
    statuses = np.full((n, 2), "missing_prediction", dtype="<U32")
    crop_presence = np.full((n, 2, 25), np.nan, dtype=np.float32)
    crop_sigma = np.full((n, 2, 25, 2), np.nan, dtype=np.float32)
    summary = {}

    for side_index, side in enumerate(SIDE_ORDER):
        prediction = predictions.get(side)
        if prediction is None:
            summary[side] = {"saved_prediction_rows": 0, "missing_prediction_frames": ids.tolist(),
                             "admitted_landmarks": 0, "admitted_wrist_frames": 0,
                             "presence_available": False, "sigma_available": False}
            continue
        rows, all_camera = _prediction_geometry(prediction, n)
        camera = all_camera[:, SOMA25_TO_LANDMARKS21]
        pixels, valid = project_camera_points(camera, matrix)
        image_valid = valid & (pixels[..., 0] >= 0) & (pixels[..., 0] < size[0])
        image_valid &= (pixels[..., 1] >= 0) & (pixels[..., 1] < size[1])
        row_boxes = boxes[rows, side_index]
        admitted = image_valid & (pixels[..., 0] >= row_boxes[:, None, 0])
        admitted &= pixels[..., 0] < row_boxes[:, None, 2]
        admitted &= pixels[..., 1] >= row_boxes[:, None, 1]
        admitted &= pixels[..., 1] < row_boxes[:, None, 3]
        present[rows, side_index] = True
        camera_valid[rows, side_index] = valid
        statuses[rows, side_index] = "model_projection"
        statuses[rows[~admitted.any(-1)], side_index] = "projection_not_admitted"
        points[rows, side_index] = np.where(admitted[..., None], pixels, 0).astype(np.float32)
        confidence[rows, side_index] = admitted.astype(np.float32)
        # Preserve finite proposal geometry for analysis, including points whose
        # image/crop test failed. Its validity is separate from fitting weights.
        geometry[rows, side_index] = np.where(np.isfinite(camera), camera, 0)
        presence_available = "pred_kp2d_crop_presence" in prediction
        sigma_available = "pred_kp2d_sigma" in prediction
        if presence_available:
            value = np.asarray(prediction["pred_kp2d_crop_presence"], dtype=np.float32)
            if value.shape != (len(rows), 25) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
                raise ValueError("crop presence must be finite [N,25] values in [0,1]")
            crop_presence[rows, side_index] = value
        if sigma_available:
            value = np.asarray(prediction["pred_kp2d_sigma"], dtype=np.float32)
            if value.shape != (len(rows), 25, 2) or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError("crop sigma must be finite nonnegative [N,25,2]")
            crop_sigma[rows, side_index] = value
        summary[side] = {
            "saved_prediction_rows": len(rows),
            "missing_prediction_frames": ids[~present[:, side_index]].tolist(),
            "admitted_landmarks": int(admitted.sum()),
            "admitted_wrist_frames": int(admitted[:, 0].sum()) if len(rows) else 0,
            "nonfinite_or_nonpositive_depth_landmarks": int((~valid).sum()),
            "outside_image_landmarks": int((valid & ~image_valid).sum()),
            "outside_crop_landmarks": int((image_valid & ~admitted).sum()),
            "presence_available": presence_available,
            "sigma_available": sigma_available,
        }

    arrays = {
        "frame_indices": ids, "side_order": np.array(SIDE_ORDER), "joints2d": points,
        "confidence": confidence, "crop_boxes": boxes.copy(), "image_size": size.copy(),
        "camera_intrinsics": matrix.astype(np.float32), "statuses": statuses,
        "prediction_row_present": present, "proposal_joints3d_camera": geometry,
        "proposal_camera_geometry_valid": camera_valid,
        "pred_kp2d_crop_presence_soma25": crop_presence, "pred_kp2d_sigma_soma25": crop_sigma,
        "soma25_to_landmarks21": SOMA25_TO_LANDMARKS21.copy(),
        "proposal_source": np.array("EgoHand"), "proposal_variant": np.array(variant),
        "independent_image_observations": np.array(False),
        "confidence_semantics": np.array("binary_geometric_admission_not_visibility_or_probability"),
    }
    report = {
        "schema_version": "egohand_geometric_proposals.v1", "frames": n,
        "variant": variant, "side_order": list(SIDE_ORDER), "hands": summary,
        "joint_mapping": SOMA25_TO_LANDMARKS21.tolist(),
        "mapping_limit": "Approximate semantic correspondence; no equivalence of SOMA and full-body native MHR geometry is asserted.",
        "coordinate_system": "OpenCV camera: x right, y down, z forward; metres",
        "targets": "3D model predictions projected into original-video pixels; not independent image detections",
        "confidence_policy": "Binary finite positive-depth source-image-and-frozen-crop admission; not visibility, calibrated confidence or reconstruction truth.",
        "optional_model_signals": "Crop presence and normalized-crop sigma are preserved without thresholding or visibility interpretation.",
        "mirroring": "No additional mirror; EgoHand anatomical outputs are already in source-camera coordinates.",
        "missing_policy": "Zero fitting weights on absent predictions; no interpolation, reindexing or invented targets.",
        "holdout_policy": "No fitting/validation frames are removed here; the same downstream original-frame holdout mask must be used for each method. Model image inference itself sees supplied frames.",
        "official_scores_computed": False, "quality_accepted": False,
    }
    return arrays, report
