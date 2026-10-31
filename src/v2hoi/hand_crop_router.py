"""Conservative training-only routing between two immutable hand crop trials.

This selects crop pixels, not reconstruction results. MediaPipe support is a
shared, fallible observer dependency; it is neither true visibility nor quality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .refined_egohand import RefinedHandPolicy, _boxes, _image_size, _observations

SIDES = ("left", "right")


@dataclass(frozen=True)
class CropRouterPolicy:
    holdout_stride: int = 5
    minimum_accepted_fingers: int = 8
    minimum_confidence: float = .05
    consecutive_training_successes: int = 3
    mixed_pair_duplicate_distance_pixels: float = 12.

    def __post_init__(self):
        for key in ("holdout_stride", "minimum_accepted_fingers", "consecutive_training_successes"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.holdout_stride < 2 or self.minimum_accepted_fingers > 20:
            raise ValueError("Invalid router holdout stride or finger count")
        if not 0 < self.minimum_confidence <= 1:
            raise ValueError("Invalid support threshold")
        if not np.isfinite(self.mixed_pair_duplicate_distance_pixels) or self.mixed_pair_duplicate_distance_pixels <= 0:
            raise ValueError("Invalid duplicate distance")


def route_hand_crops(old_observations: Mapping[str, Any], trial_observations: Mapping[str, Any],
                     *, policy: CropRouterPolicy | None = None):
    """Copy exactly one old/trial box per hand without heldout detector input.

    Regressed/unsupported training slots widen immediately. Three consecutive
    successful training slots are needed to tighten; a heldout slot is skipped
    by that state machine. Mixed source pairs receive the original observer's
    raw-finite/12px duplicate check before accepting a training route. Heldout
    slots use only the nearest routed training slots on both sides.
    """
    policy = policy or CropRouterPolicy()
    evidence_policy = RefinedHandPolicy(holdout_stride=policy.holdout_stride,
                                       minimum_accepted_fingers=policy.minimum_accepted_fingers,
                                       minimum_confidence=policy.minimum_confidence)
    ids, old_points, _, old_accepted, old_supported = _observations(old_observations, evidence_policy)
    trial_ids, trial_points, _, trial_accepted, trial_supported = _observations(trial_observations, evidence_policy)
    if not np.array_equal(ids, trial_ids):
        raise ValueError("Both observers must retain identical original frame IDs")
    size = _image_size(old_observations["image_size"])
    if not np.array_equal(size, _image_size(trial_observations["image_size"])):
        raise ValueError("Observer image sizes differ")
    wide = _boxes(old_observations["crop_boxes"], len(ids), size)
    tight = _boxes(trial_observations["crop_boxes"], len(ids), size)
    training = ids % policy.holdout_stride != 0
    old_count = old_accepted[:, :, 1:].sum(-1)
    trial_count = trial_accepted[:, :, 1:].sum(-1)
    eligible = trial_supported & (trial_count >= old_count)
    selected_tight = np.zeros((len(ids), 2), bool)
    reasons = np.full((len(ids), 2), "heldout_wide_without_two_tight_neighbors", dtype="<U56")
    successes = np.zeros((len(ids), 2), dtype=np.int32)
    duplicate = np.zeros(len(ids), bool)
    candidate_mixed = np.zeros(len(ids), bool)
    selected_wrist_distance = np.full(len(ids), np.nan)
    streak = np.zeros(2, dtype=int)
    training_ids = ids[training]
    for frame in training_ids:
        for side in range(2):
            if eligible[frame, side]:
                streak[side] += 1
                selected_tight[frame, side] = streak[side] >= policy.consecutive_training_successes
                reasons[frame, side] = ("tight_after_consecutive_training_support" if selected_tight[frame, side]
                                        else "wide_waiting_for_training_support")
            else:
                streak[side] = 0
                reasons[frame, side] = ("wide_trial_finger_count_regressed" if trial_supported[frame, side]
                                        else "wide_trial_support_missing")
        if selected_tight[frame, 0] != selected_tight[frame, 1]:
            candidate_mixed[frame] = True
            chosen = np.stack([trial_points[frame, s] if selected_tight[frame, s]
                               else old_points[frame, s] for s in range(2)])
            # Exact raw-coordinate condition of the existing global observer:
            # even a zero-weight hand can cause its opposite to be rejected.
            if np.isfinite(chosen).all():
                distance = float(np.linalg.norm(chosen[0, 0]-chosen[1, 0]))
                selected_wrist_distance[frame] = distance
                if distance < policy.mixed_pair_duplicate_distance_pixels:
                    selected_tight[frame] = False
                    duplicate[frame] = True
                    streak[:] = 0
                    reasons[frame] = "wide_both_new_mixed_pair_duplicate"
        successes[frame] = streak
    for frame in ids[~training]:
        insertion = np.searchsorted(training_ids, frame)
        if 0 < insertion < len(training_ids):
            before, after = training_ids[insertion-1], training_ids[insertion]
            selected_tight[frame] = selected_tight[before] & selected_tight[after]
            reasons[frame, selected_tight[frame]] = "heldout_tight_between_two_tight_training_routes"
    boxes = np.where(selected_tight[..., None], tight, wide).astype(np.int32)
    simulated_count = np.where(selected_tight, trial_count, old_count)
    if np.any(simulated_count[training] < old_count[training]):
        raise AssertionError("Router lost known per-hand training support")
    anchor_weights = (old_supported | trial_supported).astype(np.float32)
    anchor_weights[~training] = 0
    arrays = {"frame_indices": ids, "side_order": np.asarray(SIDES), "crop_boxes": boxes.copy(),
              "route_tight": selected_tight, "routing_reason": reasons,
              "training_anchor_weights": anchor_weights,
              "old_support_weights": old_supported.astype(np.float32),
              "trial_support_weights": trial_supported.astype(np.float32),
              "old_accepted_finger_count": old_count.astype(np.int16),
              "trial_accepted_finger_count": trial_count.astype(np.int16),
              "eligible_training_slot": eligible, "consecutive_training_successes": successes,
              "mixed_source_candidate": candidate_mixed, "new_mixed_pair_duplicate": duplicate,
              "selected_raw_wrist_distance_pixels": selected_wrist_distance,
              "crop_source_class": np.where(selected_tight, "router_tight_trial01", "router_original_wide")}
    report = {"schema_version": "refined_hand_crop_router.v1", "frames": len(ids),
              "policy": asdict(policy),
              "training_anchor_weights": anchor_weights.astype(int).tolist(),
              "training_anchor_original_frame_ids": {name: ids[anchor_weights[:, s] > 0].tolist()
                                                       for s, name in enumerate(SIDES)},
              "heldout_original_frame_ids": ids[~training].tolist(),
              "route_tight_frames": selected_tight.sum(0).tolist(),
              "route_wide_frames": (~selected_tight).sum(0).tolist(),
              "eligible_training_frames": eligible.sum(0).tolist(),
              "new_mixed_pair_duplicate_original_frame_ids": ids[duplicate].tolist(),
              "routing_reason_counts": {name: {str(k): int(n) for k, n in zip(*np.unique(reasons[:, s], return_counts=True))}
                                        for s, name in enumerate(SIDES)},
              "simulated_training_finger_count_nonregression": True,
              "selection_dependency": "Shared MediaPipe trial support selects crops on training slots only. This is method-dependent development selection, not independent validation, true visibility or reconstruction ranking.",
              "heldout_policy": "Stride-5 coordinates and weights are masked before counts or duplicate checks. Heldout slots use only adjacent routed training slots; no heldout detection result is read for routing.",
              "duplicate_policy": "A tentative mixed-source training pair with both full raw21 arrays finite and wrist distance<12px routes both wide and resets both counters, even if one source hand has zero weights. Heldout points are never inspected.",
              "geometry_policy": "Every output box is copied unchanged from the immutable original or tight trial; no target/crop interpolation or retarget-math change.",
              "coverage_limit": "Simulated per-hand training support retention does not guarantee actual MP02 rerun support, cross-hand association or heldout coverage. Run and review the observer again on the common selected boxes.",
              "quality_accepted": False, "official_scores_computed": False}
    return boxes, arrays, report
