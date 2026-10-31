"""Training-only crop localization and fixed-depth EgoHand retargeting.

Detector support is an image-evidence heuristic, never true visibility. Missing
support creates no new image targets. This module changes neither native MHR
identity nor the source observations and has no model/data dependencies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .egohand_adapter import project_camera_points, validate_frame_indices, validate_intrinsics

SIDES = ("left", "right")


@dataclass(frozen=True)
class RefinedHandPolicy:
    holdout_stride: int = 5
    minimum_accepted_fingers: int = 8
    minimum_confidence: float = .05
    crop_padding: float = 1.5
    minimum_crop_side: int = 64
    maximum_crop_side: int = 208
    center_median_window: int = 5
    side_median_window: int = 15
    maximum_crop_missing_frames: int = 8
    accepted_point_margin_pixels: int = 2
    minimum_scale_anchor_frames: int = 5

    def __post_init__(self):
        for key in ("holdout_stride", "minimum_accepted_fingers", "minimum_crop_side",
                    "maximum_crop_side", "center_median_window", "side_median_window",
                    "minimum_scale_anchor_frames"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.holdout_stride < 2 or self.minimum_accepted_fingers > 20:
            raise ValueError("Invalid holdout stride or finger count")
        if self.minimum_crop_side > self.maximum_crop_side:
            raise ValueError("Crop side bounds are reversed")
        if self.center_median_window % 2 != 1 or self.side_median_window % 2 != 1:
            raise ValueError("Median windows must be odd")
        if self.maximum_crop_missing_frames < 0 or self.accepted_point_margin_pixels < 0:
            raise ValueError("Crop gap and margin must be nonnegative")
        if not 0 < self.minimum_confidence <= 1 or not np.isfinite(self.crop_padding) or self.crop_padding < 1:
            raise ValueError("Invalid confidence threshold or crop padding")


def _observations(observations: Mapping[str, Any], policy: RefinedHandPolicy):
    ids = validate_frame_indices(observations["frame_indices"])
    if np.asarray(observations["side_order"]).tolist() != list(SIDES):
        raise ValueError("Require anatomical left/right side order")
    points = np.asarray(observations["joints2d"], dtype=np.float64)
    confidence = np.asarray(observations["confidence"], dtype=np.float64)
    if points.shape != (len(ids), 2, 21, 2) or confidence.shape != (len(ids), 2, 21):
        raise ValueError("Require complete joints2d[T,2,21,2] and confidence[T,2,21]")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Invalid detector weights")
    # Mask before computing any coordinate-based anchor, crop, scale or gate.
    train = ids % policy.holdout_stride != 0
    points = points.copy()
    points[~train] = np.nan
    weights = np.where(train[:, None, None], confidence, 0.)
    accepted = (weights >= policy.minimum_confidence) & np.isfinite(points).all(-1)
    hand = accepted[:, :, 0] & (accepted[:, :, 1:].sum(-1) >= policy.minimum_accepted_fingers)
    return ids, points, weights, accepted, hand


def _image_size(value):
    size = np.asarray(value)
    if size.shape != (2,) or size.dtype.kind not in "iu" or np.any(size <= 0):
        raise ValueError("image_size must be integer [width,height]")
    return size.astype(int)


def _boxes(value, count, size):
    boxes = np.asarray(value)
    if boxes.shape != (count, 2, 4) or boxes.dtype.kind not in "iu":
        raise ValueError("Require integer crop_boxes[T,2,4]")
    spans = boxes[..., 2:] - boxes[..., :2]
    if np.any(spans <= 0) or not np.array_equal(spans[..., 0], spans[..., 1]):
        raise ValueError("Crop boxes must be positive squares")
    if np.any(boxes[..., :2] < 0) or np.any(boxes[..., 2:] > size):
        raise ValueError("Crop boxes must remain inside the source image")
    return boxes.astype(np.int32, copy=True)


def _contiguous(mask):
    indices = np.flatnonzero(mask)
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1) if len(indices) else []


def _median_segment(values, radius):
    return np.asarray([np.median(values[max(0, i-radius):min(len(values), i+radius+1)], axis=0)
                       for i in range(len(values))])


def _integer_box(center, side, size, accepted_points=None, margin=0):
    """Move only the crop, never its landmarks, to preserve accepted pixels."""
    side = int(side)
    if side <= 0 or side > min(size):
        return None
    proposed = np.rint(np.asarray(center) - side / 2).astype(int)
    low = np.zeros(2, dtype=int)
    high = size - side
    if accepted_points is not None and len(accepted_points):
        # Inclusive accepted-point margin; right/bottom box bounds are exclusive.
        low = np.maximum(low, np.ceil(accepted_points.max(0) + margin - side).astype(int))
        high = np.minimum(high, np.floor(accepted_points.min(0) - margin).astype(int))
    if np.any(low > high):
        return None
    lower = np.maximum(low, np.minimum(high, proposed))
    return np.r_[lower, lower + side].astype(np.int32)


def build_refined_crops(observations: Mapping[str, Any], *, policy: RefinedHandPolicy | None = None):
    """Freeze common observer crops without reading heldout detector positions.

    Full finite detector landmarks may localize a crop only after a training
    wrist and eight accepted fingers corroborate that hand. The unaccepted
    coordinates used for localization never become image fitting targets.
    """
    policy = policy or RefinedHandPolicy()
    ids, points, _, accepted, supported = _observations(observations, policy)
    size = _image_size(observations["image_size"])
    if min(size) < policy.maximum_crop_side:
        raise ValueError("Source image is smaller than the declared crop maximum")
    fallback = _boxes(observations["crop_boxes"], len(ids), size)
    boxes = fallback.copy()
    anchors = supported & np.isfinite(points).all(axis=(-1, -2))
    centers = np.full((len(ids), 2, 2), np.nan)
    sides = np.full((len(ids), 2), np.nan)
    origins = np.full((len(ids), 2), "sam_fallback_unknown", dtype="<U40")
    for side in range(2):
        for frame in np.flatnonzero(anchors[:, side]):
            all_points = points[frame, side]
            good = all_points[accepted[frame, side]]
            center = (all_points.min(0) + all_points.max(0)) / 2
            width = int(np.clip(np.ceil(np.ptp(all_points, axis=0).max() * policy.crop_padding),
                                policy.minimum_crop_side, policy.maximum_crop_side))
            required = int(np.ceil(np.ptp(good, axis=0).max() + 2 * policy.accepted_point_margin_pixels))
            if required > policy.maximum_crop_side:
                anchors[frame, side] = False
                origins[frame, side] = "sam_fallback_unusable_anchor"
                continue
            width = max(width, required)
            if _integer_box(center, width, size, good, policy.accepted_point_margin_pixels) is None:
                anchors[frame, side] = False
                origins[frame, side] = "sam_fallback_unusable_anchor"
                continue
            centers[frame, side], sides[frame, side] = center, width
        usable = anchors[:, side].copy()
        actual = np.flatnonzero(usable)
        for a, b in zip(actual[:-1], actual[1:]):
            if b-a-1 <= policy.maximum_crop_missing_frames:
                for frame in range(a+1, b):
                    alpha = (frame-a) / (b-a)
                    centers[frame, side] = (1-alpha)*centers[a, side] + alpha*centers[b, side]
                    sides[frame, side] = (1-alpha)*sides[a, side] + alpha*sides[b, side]
                    usable[frame] = True
        for segment in _contiguous(usable):
            c = _median_segment(centers[segment, side], policy.center_median_window//2)
            w = _median_segment(sides[segment, side], policy.side_median_window//2)
            for offset, frame in enumerate(segment):
                width = int(np.clip(np.ceil(w[offset]), policy.minimum_crop_side, policy.maximum_crop_side))
                good = points[frame, side][accepted[frame, side]] if anchors[frame, side] else None
                if good is not None:
                    width = max(width, int(np.ceil(np.ptp(good, axis=0).max() + 2*policy.accepted_point_margin_pixels)))
                box = _integer_box(c[offset], width, size, good, policy.accepted_point_margin_pixels)
                if box is None:
                    raise AssertionError("A previously feasible crop anchor became infeasible")
                boxes[frame, side] = box
                origins[frame, side] = ("training_image_anchor" if anchors[frame, side]
                                       else "training_neighbor_crop_interpolation")
    arrays = {"frame_indices": ids, "side_order": np.asarray(SIDES),
              "training_anchor_weights": anchors.astype(np.float32),
              "crop_source_class": origins, "raw_anchor_centers": centers,
              "raw_anchor_sides": sides, "crop_boxes": boxes.copy()}
    report = {"schema_version": "refined_hand_crops.v1", "policy": asdict(policy), "frames": len(ids),
              "training_anchor_weights": anchors.astype(int).tolist(),
              "training_anchor_original_frame_ids": {name: ids[anchors[:, s]].tolist() for s, name in enumerate(SIDES)},
              "heldout_original_frame_ids": ids[ids % policy.holdout_stride == 0].tolist(),
              "crop_source_counts": {name: {str(k): int(n) for k, n in zip(*np.unique(origins[:, s], return_counts=True))}
                                     for s, name in enumerate(SIDES)},
              "changed_hand_frames": (np.any(boxes != fallback, axis=-1)).sum(0).tolist(),
              "source_side_median": np.median(fallback[..., 2]-fallback[..., 0], axis=0).tolist(),
              "revised_side_median": np.median(boxes[..., 2]-boxes[..., 0], axis=0).tolist(),
              "image_evidence_limit": "Training-only detector corroboration is not a true visibility or occlusion label. Full finite predictions localize crops only; no missing image targets are filled.",
              "holdout_policy": "Stride-5 detector positions and weights are excluded before localization. Crop-only short-gap interpolation uses training neighbors. Original SAM fallback remains transductive.",
              "quality_accepted": False, "official_scores_computed": False}
    return boxes, arrays, report


def retarget_egohand_proposals(proposals: Mapping[str, Any], observations: Mapping[str, Any],
                              native_hands: Any, *, intrinsics: Any,
                              scale_reference: Mapping[str, Any] | None = None,
                              policy: RefinedHandPolicy | None = None):
    """Keep Ego articulation, use one scale/hand and the native body's depth.

    The fresh detector supplies only the training wrist's lateral position and
    shared support weights. Finger coordinates are projected Ego geometry, not
    copied detector coordinates. Unknown/heldout rows retain prior-only geometry
    with zero new image weights. Raw reference geometry fixes both variants'
    per-hand clip scales without using any heldout detector input.
    """
    policy = policy or RefinedHandPolicy()
    ids, mp, _, accepted, image_support = _observations(observations, policy)
    matrix = validate_intrinsics(intrinsics)
    size = _image_size(observations["image_size"])
    boxes = _boxes(observations["crop_boxes"], len(ids), size)
    for candidate in (proposals, scale_reference or proposals):
        if not np.array_equal(candidate["frame_indices"], ids):
            raise ValueError("Proposal original-frame IDs differ")
        if np.asarray(candidate["side_order"]).tolist() != list(SIDES):
            raise ValueError("Proposal side order differs")
        if not np.array_equal(candidate["crop_boxes"], boxes):
            raise ValueError("Proposal and detector crop bounds differ")
    native = np.asarray(native_hands, dtype=np.float64)
    if native.shape != (len(ids), 2, 21, 3) or not np.isfinite(native).all() or np.any(native[..., 2] <= 0):
        raise ValueError("Require complete finite positive-depth native hands[T,2,21,3]")
    geometry = np.asarray(proposals["proposal_joints3d_camera"], dtype=np.float64)
    geometry_valid = np.asarray(proposals["proposal_camera_geometry_valid"], dtype=bool)
    reference = scale_reference or proposals
    ref = np.asarray(reference["proposal_joints3d_camera"], dtype=np.float64)
    ref_valid = np.asarray(reference["proposal_camera_geometry_valid"], dtype=bool)
    if geometry.shape != native.shape or ref.shape != native.shape or geometry_valid.shape != native.shape[:-1] or ref_valid.shape != native.shape[:-1]:
        raise ValueError("Invalid proposal camera geometry shape")
    native_width = np.linalg.norm(native[:, :, 5]-native[:, :, 17], axis=-1)
    reference_width = np.linalg.norm(ref[:, :, 5]-ref[:, :, 17], axis=-1)
    scale_frames = image_support & ref_valid[:, :, [0, 5, 17]].all(-1)
    scale_frames &= np.isfinite(reference_width) & (reference_width > 0) & (native_width > 0)
    scalars = np.ones(2, dtype=np.float64)
    calibrated = np.zeros(2, bool)
    counts = scale_frames.sum(0)
    for side in range(2):
        if counts[side] >= policy.minimum_scale_anchor_frames:
            values = native_width[scale_frames[:, side], side] / reference_width[scale_frames[:, side], side]
            scalars[side] = np.median(values)
            calibrated[side] = np.isfinite(scalars[side]) and scalars[side] > 0
    anchor_weights = image_support.astype(np.float32)
    support_weights = image_support.astype(np.float32)
    wrist = native[:, :, 0].copy()
    hom = np.concatenate((np.nan_to_num(mp[:, :, 0]), np.ones((len(ids), 2, 1))), axis=-1)
    rays = hom @ np.linalg.inv(matrix).T
    located = rays / rays[..., 2:] * native[:, :, :1, 2]
    wrist[image_support] = located[image_support]
    local = geometry - geometry[:, :, :1]
    retargeted = local * scalars[None, :, None, None] + wrist[:, :, None]
    projected, valid = project_camera_points(retargeted, matrix)
    valid &= geometry_valid & geometry_valid[:, :, :1] & calibrated[None, :, None]
    inside = valid & (projected[..., 0] >= 0) & (projected[..., 0] < size[0])
    inside &= (projected[..., 1] >= 0) & (projected[..., 1] < size[1])
    inside &= (projected[..., 0] >= boxes[..., None, 0]) & (projected[..., 0] < boxes[..., None, 2])
    inside &= (projected[..., 1] >= boxes[..., None, 1]) & (projected[..., 1] < boxes[..., None, 3])
    admitted = inside & accepted & image_support[..., None]
    # Cheap declared ablation only; never selected by a fitting/evaluation score.
    original_pixels, original_valid = project_camera_points(geometry, matrix)
    anchor_pixels, _ = project_camera_points(wrist, matrix)
    translated = original_pixels - original_pixels[:, :, :1] + anchor_pixels[:, :, None]
    translated[~(original_valid & original_valid[:, :, :1])] = np.nan
    arrays = {key: np.array(value, copy=True) for key, value in proposals.items()}
    arrays.update(joints2d=np.where(admitted[..., None], projected, 0).astype(np.float32),
                  confidence=admitted.astype(np.float32),
                  original_proposal_joints3d_camera=geometry.copy(),
                  proposal_joints3d_camera=np.where(np.isfinite(retargeted), retargeted, 0),
                  proposal_camera_geometry_valid=valid,
                  shared_wrist_anchor_weights=anchor_weights,
                  shared_visibility_gate_weights=support_weights,
                  shared_image_joint_gate_weights=accepted.astype(np.float32),
                  fixed_native_to_ego_palm_scalar=scalars,
                  fixed_scale_anchor_weights=scale_frames.astype(np.float32),
                  fixed_scale_calibrated=calibrated,
                  diagnostic_2d_wrist_translation=translated,
                  observation_kind=np.asarray("egohand_geometric_proposals.v1"),
                  refinement_kind=np.asarray("native_depth_training_wrist_image_gate.v1"),
                  independent_image_observations=np.asarray(False),
                  confidence_semantics=np.asarray("corroborated_detector_support_for_generated_geometry_not_true_visibility"))
    arrays["statuses"] = np.where(image_support & calibrated[None], "image_supported_generated_target", "prior_only_no_new_image_weight")
    report = {"schema_version": "refined_egohand_proposals.v1", "frames": len(ids), "policy": asdict(policy),
              "fixed_native_to_ego_palm_scalar": scalars.tolist(), "fixed_scale_calibrated": calibrated.tolist(),
              "fixed_scale_anchor_frames": counts.tolist(),
              "scale_anchor_original_frame_ids": {name: ids[scale_frames[:, s]].tolist() for s, name in enumerate(SIDES)},
              "shared_image_support_frames": image_support.sum(0).tolist(),
              "new_image_target_frames": admitted.any(-1).sum(0).tolist(),
              "new_image_joint_weights": admitted.sum(axis=(0, 2)).tolist(),
              "insufficient_scale_anchor_policy": "Only that hand receives zero new image targets; its continuous prior geometry and the other hand remain available. Scalar 1 is explicitly uncalibrated.",
              "retargeting": "One fixed raw-reference native/Ego 3D palm scalar per hand/clip, unchanged baseline native wrist depth, training-only fresh detector wrist lateral anchor. No per-frame scale, clamp, rotation fit or copied detector finger coordinate.",
              "support_semantics": "Shared detector wrist plus at least eight accepted finger points. Per-joint image support remains required. This heuristic does not establish true visibility, occlusion or confidence probability.",
              "holdout_policy": "Stride-5 detector coordinates/weights are excluded before crop/anchor/scale/gate construction. Unknown/heldout new image weights are zero; prior geometry remains. Observer inference, baseline and optional upstream smoothing remain transductive development inputs.",
              "diagnostic_ablation": "Unscaled 2D wrist translation is retained only for analysis; main retarget method was declared before neural rerun and is not chosen from evaluation scores.",
              "mapping_limit": "SOMA/full-body MHR anatomical landmarks and wrist definitions are approximate, not identical geometry.",
              "quality_accepted": False, "official_scores_computed": False}
    return arrays, report
