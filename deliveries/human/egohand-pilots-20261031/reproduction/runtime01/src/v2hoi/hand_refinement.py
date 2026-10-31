"""Evidence-weighted, finger-enabled native MHR sequence refinement.

Observations are independent image detections, not hidden challenge answers.
Camera, identity and skeleton scales stay fixed; selected arm/wrist controls
may receive explicitly bounded updates. All frames share one objective.
Diagnostics are input agreement and motion measurements, never challenge scores.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .human_result import validate_human_result

# Verified against SAM's public MHR308 keypoint_mapping and the native rig's
# joint_names. Every selected hand landmark is a one-hot skeleton joint, not a
# mesh regression. Order is anatomical left/right, MediaPipe wrist + 5x4 digits.
HAND_JOINTS = np.asarray([
    [78, 97, 98, 99, 100, 92, 93, 94, 95, 88, 89, 90, 91, 84, 85, 86, 87, 80, 81, 82, 83],
    [42, 61, 62, 63, 64, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, 44, 45, 46, 47],
], dtype=np.int64)
HAND_SLICES = ((95, 122), (68, 95))
ARM_CONTROLS = np.asarray([[43, 44, 45, 46, 47, 48, 49], [33, 34, 35, 36, 37, 38, 39]], dtype=np.int64)


@dataclass(frozen=True)
class HandFitConfig:
    iterations: int = 180
    learning_rate: float = 0.025
    max_delta_radians: float = 0.65
    min_confidence: float = 0.05
    min_landmarks: int = 8
    max_gap_frames: int = 8
    pixel_scale: float = 10.0
    robust_delta_pixels: float = 5.0
    relative_weight: float = 1.0
    velocity_weight: float = 0.10
    acceleration_weight: float = 0.025
    pose_prior_weight: float = 0.03
    motion_gate_pixels: float = 3.0
    holdout_stride: int = 5
    unobserved_policy: str = "preserve"
    optimize_wrist_chain: bool = False
    arm_delta_limit_radians: float = 0.10
    arm_prior_weight: float = 0.03
    wrist_observation_weight: float = 0.5
    wrist_anchor_weight: float = 0.025
    wrist_max_displacement_metres: float = 0.04
    arm_update_temporal_weight: float = 0.08
    finger_update_temporal_weight: float = 0.08
    spatial_acceleration_weight: float = 0.0
    support_transition_frames: int = 0
    correction_velocity_budget_weight: float = 0.0
    correction_budget_safety_fraction: float = 1.0

    def validate(self) -> None:
        for name in ("iterations", "min_landmarks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_landmarks > 20:
            raise ValueError("min_landmarks must not exceed 20 finger landmarks")
        if isinstance(self.max_gap_frames, bool) or not isinstance(self.max_gap_frames, int) or self.max_gap_frames < 0:
            raise ValueError("max_gap_frames must be a nonnegative integer")
        if isinstance(self.support_transition_frames, bool) or not isinstance(self.support_transition_frames, int) or self.support_transition_frames < 0:
            raise ValueError("support_transition_frames must be a nonnegative integer")
        if isinstance(self.holdout_stride, bool) or not isinstance(self.holdout_stride, int) or self.holdout_stride < 0 or self.holdout_stride == 1:
            raise ValueError("holdout_stride must be zero (disabled) or an integer >=2")
        if self.unobserved_policy not in ("preserve", "bounded_temporal_prior"):
            raise ValueError("unobserved_policy must be preserve or bounded_temporal_prior")
        if not isinstance(self.optimize_wrist_chain, bool):
            raise ValueError("optimize_wrist_chain must be boolean")
        for name in ("learning_rate", "max_delta_radians", "pixel_scale", "robust_delta_pixels", "motion_gate_pixels",
                     "arm_delta_limit_radians", "wrist_max_displacement_metres"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("velocity_weight", "acceleration_weight", "pose_prior_weight", "arm_prior_weight", "wrist_observation_weight", "wrist_anchor_weight", "arm_update_temporal_weight", "finger_update_temporal_weight", "spatial_acceleration_weight", "correction_velocity_budget_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.min_confidence <= 1 or not 0 <= self.relative_weight <= 1:
            raise ValueError("confidence and relative weight must lie in [0,1]")
        if not np.isfinite(self.correction_budget_safety_fraction) or not 0 < self.correction_budget_safety_fraction <= 1:
            raise ValueError("correction_budget_safety_fraction must be finite and in (0,1]")


def validate_observations(observations: Mapping[str, Any], frame_indices: np.ndarray) -> dict[str, np.ndarray]:
    """Reject mismatched sequence, side conventions, and invalid observations.

    Missing coordinates may be NaN only where confidence is exactly zero. The
    confidence is a reliability weight, not a calibrated per-joint probability.
    """
    indices = np.asarray(observations["frame_indices"])
    if indices.dtype.kind not in "iu" or not np.array_equal(indices, frame_indices):
        raise ValueError("observation frame_indices must exactly match the Human result")
    sides = np.asarray(observations["side_order"]).tolist()
    if sides != ["left", "right"]:
        raise ValueError("side_order must be anatomical ['left', 'right']")
    points = np.asarray(observations["joints2d"], dtype=np.float32)
    confidence = np.asarray(observations["confidence"], dtype=np.float32)
    count = len(indices)
    if points.shape != (count, 2, 21, 2) or confidence.shape != (count, 2, 21):
        raise ValueError("expected joints2d[T,2,21,2] and confidence[T,2,21]")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be finite and in [0,1]")
    if np.any(~np.isfinite(points).all(-1) & (confidence > 0)):
        raise ValueError("positive-confidence landmarks must have finite coordinates")
    # Own the returned arrays: callers' memmaps and detection evidence are immutable.
    return {"frame_indices": indices.copy(), "joints2d": np.where(np.isfinite(points), points, 0).copy(),
            "confidence": confidence.copy()}


def supported_frames(confidence: np.ndarray, config: HandFitConfig) -> np.ndarray:
    """Activate detected hands and bounded *internal* gaps, without extrapolation."""
    config.validate()
    if np.asarray(confidence).ndim != 3 or confidence.shape[1:] != (2, 21):
        raise ValueError("confidence must have shape [T,2,21]")
    valid = (confidence[:, :, 1:] >= config.min_confidence).sum(-1) >= config.min_landmarks
    # Wrist-relative evidence needs an observed wrist as well.
    valid &= confidence[:, :, 0] >= config.min_confidence
    active = valid.copy()
    for side in range(2):
        detected = np.flatnonzero(valid[:, side])
        for first, last in zip(detected[:-1], detected[1:]):
            if last - first - 1 <= config.max_gap_frames:
                active[first:last + 1, side] = True
    return active


def support_transition_envelope(support: np.ndarray, transition_frames: int) -> np.ndarray:
    """Bounded cosine collar around admitted support, with no new observations.

    Values are one on supported frames, taper outside them, and are exactly zero
    beyond the collar. Overlapping collars use their maximum, so a short missing
    interval can connect corrections. A long occlusion returns to the initializer
    and does not claim a recovered hidden hand trajectory.
    """
    support = np.asarray(support, dtype=bool)
    if support.ndim != 2 or support.shape[1] != 2:
        raise ValueError("support must have shape [T,2]")
    if isinstance(transition_frames, bool) or not isinstance(transition_frames, int) or transition_frames < 0:
        raise ValueError("transition_frames must be a nonnegative integer")
    envelope = support.astype(np.float32)
    for distance in range(1, min(transition_frames, len(support) - 1) + 1):
        weight = np.float32(0.5 * (1 + np.cos(np.pi * distance / (transition_frames + 1))))
        envelope[distance:] = np.maximum(envelope[distance:], support[:-distance] * weight)
        envelope[:-distance] = np.maximum(envelope[:-distance], support[distance:] * weight)
    return envelope


def observation_edge_motion_gate(points: np.ndarray, confidence: np.ndarray, config: HandFitConfig) -> np.ndarray:
    """Relax correction continuity only on independently observed fast edges.

    Wrist-relative displacement avoids treating body/camera translation as finger
    articulation. Only adjacent high-confidence observations contribute; long
    unobserved gaps never get an asserted motion target.
    """
    relative = points[:, :, 1:] - points[:, :, :1]
    speed = np.linalg.norm(np.diff(relative, axis=0), axis=-1)
    weight = np.minimum(confidence[1:, :, 1:], confidence[:-1, :, 1:])
    weight *= np.minimum(confidence[1:, :, :1], confidence[:-1, :, :1])
    weight = np.where(weight >= config.min_confidence, weight, 0)
    mean_speed = (speed * weight).sum(-1) / np.maximum(weight.sum(-1), 1e-8)
    return (1 / (1 + (mean_speed / config.motion_gate_pixels) ** 2)).astype(np.float32)


def observation_motion_gate(points: np.ndarray, confidence: np.ndarray, config: HandFitConfig) -> np.ndarray:
    """Reduce acceleration regularization during independently observed motion."""
    edges = observation_edge_motion_gate(points, confidence, config)
    return np.minimum(edges[1:], edges[:-1])


def hand_diagnostics(projected: np.ndarray, observations: Mapping[str, Any], active: np.ndarray) -> dict[str, Any]:
    """Describe agreement without calling it accuracy against real geometry."""
    confidence = np.asarray(observations["confidence"])
    target = np.asarray(observations["joints2d"])
    finite = np.isfinite(projected).all(-1)
    weights = np.where(finite, confidence, 0)
    distance = np.linalg.norm(np.nan_to_num(projected) - target, axis=-1)
    relative = projected[:, :, 1:] - projected[:, :, :1]
    target_relative = target[:, :, 1:] - target[:, :, :1]
    relative_distance = np.linalg.norm(np.nan_to_num(relative) - target_relative, axis=-1)
    relative_weights = np.minimum(weights[:, :, 1:], weights[:, :, :1])
    step = np.linalg.norm(np.diff(relative, axis=0), axis=-1)
    second = np.linalg.norm(np.diff(relative, n=2, axis=0), axis=-1)
    output: dict[str, Any] = {"interpretation": "Agreement with independent image detections; not hidden-reference accuracy or challenge scores"}
    for side, name in enumerate(("left", "right")):
        weight = weights[:, side]
        visible = weight > 0
        output[name] = {
            "active_frames": int(active[:, side].sum()),
            "observed_frames": int((confidence[:, side, 1:].sum(-1) > 0).sum()),
            "weighted_reprojection_pixels": float((distance[:, side] * weight).sum() / max(float(weight.sum()), 1e-8)),
            "median_observed_reprojection_pixels": float(np.median(distance[:, side][visible])) if visible.any() else None,
            "weighted_wrist_relative_reprojection_pixels": float((relative_distance[:, side] * relative_weights[:, side]).sum() / max(float(relative_weights[:, side].sum()), 1e-8)),
            "relative_step_median_pixels": float(np.median(step[:, side])) if len(step) else None,
            "relative_step_p95_pixels": float(np.percentile(step[:, side], 95)) if len(step) else None,
            "relative_second_difference_p95_pixels_per_frame2": float(np.percentile(second[:, side], 95)) if len(second) else None,
        }
    return output


def correction_boundary_diagnostics(baseline: np.ndarray, candidate: np.ndarray, active: np.ndarray,
                                    frame_indices: np.ndarray, intrinsics: np.ndarray,
                                    observations: Mapping[str, Any], min_confidence: float) -> dict[str, Any]:
    """Measure every support transition without an unavailable reference target."""
    def relative_pixels(joints):
        projected = joints @ intrinsics.T
        projected = projected[..., :2] / np.maximum(projected[..., 2:], 1e-5)
        return projected[:, :, 1:] - projected[:, :, :1]

    base_px, candidate_px = relative_pixels(baseline), relative_pixels(candidate)
    base_m = baseline[:, :, 1:] - baseline[:, :, :1]
    candidate_m = candidate[:, :, 1:] - candidate[:, :, :1]
    observed = observations["joints2d"][:, :, 1:] - observations["joints2d"][:, :, :1]
    confidence = observations["confidence"]
    rows = []
    for before, side in np.argwhere(active[1:] != active[:-1]):
        after = before + 1
        row = {"side": ("left", "right")[side], "from_original_frame_id": int(frame_indices[before]),
               "to_original_frame_id": int(frame_indices[after]),
               "transition": "enter_support" if active[after, side] else "leave_support"}
        for unit, original, fitted in (("px", base_px, candidate_px), ("metres", base_m, candidate_m)):
            correction = fitted - original
            for name, value in (("baseline", original), ("candidate", fitted), ("correction_only", correction)):
                row[f"{name}_mean_wrist_relative_step_{unit}"] = float(np.linalg.norm(value[after, side] - value[before, side], axis=-1).mean())
        wrist_supported = bool(min(confidence[before, side, 0], confidence[after, side, 0]) >= min_confidence)
        common = (np.minimum(confidence[before, side, 1:], confidence[after, side, 1:]) >= min_confidence) & wrist_supported
        row.update(common_accepted_finger_count=int(common.sum()), adjacent_wrists_accepted=wrist_supported,
                   observed_mean_wrist_relative_step_px=None, baseline_mean_velocity_vector_discrepancy_px=None,
                   candidate_mean_velocity_vector_discrepancy_px=None)
        if common.any():
            velocity = observed[after, side, common] - observed[before, side, common]
            row["observed_mean_wrist_relative_step_px"] = float(np.linalg.norm(velocity, axis=-1).mean())
            for name, value in (("baseline", base_px), ("candidate", candidate_px)):
                predicted_velocity = value[after, side, common] - value[before, side, common]
                row[f"{name}_mean_velocity_vector_discrepancy_px"] = float(np.linalg.norm(predicted_velocity - velocity, axis=-1).mean())
        rows.append(row)
    return {"interpretation": "Prediction-only support-boundary steps relative to each wrist, not reference-relative ACC or verified physical motion",
            "independent_motion_evidence": "Full independent observations are used only after fitting for this report; accepted adjacent fingers require both accepted wrists and both accepted finger observations",
            "boundary_count": len(rows), "boundaries": rows}


class NativeMHRHands:
    """Differentiable official skeleton path; equivalent to full-model joints."""
    def __init__(self, model_path: str, *, device: str = "cpu"):
        import torch

        self.device = torch.device(device)
        self.model = torch.jit.load(str(model_path), map_location=self.device).float().eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        names = list(self.model.character_torch.skeleton.joint_names)
        expected = ["l_wrist", "l_thumb1", "l_thumb2", "l_thumb3", "l_thumb_null",
                    "l_index1", "l_index2", "l_index3", "l_index_null", "l_middle1", "l_middle2", "l_middle3", "l_middle_null",
                    "l_ring1", "l_ring2", "l_ring3", "l_ring_null", "l_pinky1", "l_pinky2", "l_pinky3", "l_pinky_null"]
        for side, prefix in enumerate(("l_", "r_")):
            actual = [names[index] for index in HAND_JOINTS[side]]
            if actual != [prefix + name[2:] for name in expected]:
                raise ValueError("MHR rig joint names differ from the verified hand mapping")
        self.indices = torch.as_tensor(HAND_JOINTS.reshape(-1).copy(), device=self.device)
        controls = list(self.model.character_torch.parameter_transform.parameter_names)
        if not all(name.startswith("r_") for name in controls[68:95]) or not all(name.startswith("l_") for name in controls[95:122]):
            raise ValueError("MHR finger control layout differs from the verified parameter mapping")
        expected = ["uparm_twist", "uparm_ry", "uparm_rz", "elbow_bend", "lowarm_twist", "wrist_ry", "wrist_rz"]
        for side, prefix in enumerate(("l_", "r_")):
            if [controls[index] for index in ARM_CONTROLS[side]] != [prefix + name for name in expected]:
                raise ValueError("MHR arm/wrist control names differ from the verified mapping")
        self.parameter_names = controls

    def __call__(self, pose, scales):
        import torch

        count = len(pose)
        params = torch.cat([pose, scales[None].expand(count, -1), pose.new_zeros(count, 45)], dim=1)
        character = self.model.character_torch
        joints = character.joint_parameters_to_skeleton_state(character.model_parameters_to_joint_parameters(params))
        # Native centimetres/y-up/z-back -> Track 1 metres/y-down/z-forward.
        selected = joints[:, self.indices, :3].reshape(count, 2, 21, 3)
        return selected * selected.new_tensor([0.01, -0.01, -0.01])

    def verify_full_forward(self, result: Mapping[str, Any], positions=(0,)) -> dict[str, Any]:
        import torch

        with torch.no_grad():
            pose = torch.as_tensor(np.array(result["pose"])[list(positions)], device=self.device)
            scales = torch.as_tensor(np.array(result["scales"]), device=self.device)
            shape = torch.as_tensor(np.array(result["shape"]), device=self.device)
            params = torch.cat([pose, scales[None].expand(len(pose), -1)], dim=1)
            _, joints = self.model(shape[None].expand(len(pose), -1).contiguous(), params, pose.new_zeros(len(pose), 72), True)
            expected = joints[:, self.indices, :3].reshape(len(pose), 2, 21, 3) * pose.new_tensor([0.01, -0.01, -0.01])
            actual = self(pose, scales)
            maximum = float((expected - actual).abs().max().cpu())
        if maximum > 1e-6:
            raise ValueError(f"skeleton-only hand forward differs from full MHR by {maximum} metres")
        return {"positions": list(positions), "max_absolute_difference_metres": maximum,
                "hand_mapping": HAND_JOINTS.tolist(), "full_native_forward_checked": True}


def refine_hands(
    result: Mapping[str, Any], observations: Mapping[str, Any], intrinsics: np.ndarray,
    hand_forward: Callable, *, config: HandFitConfig | None = None, device: str = "cpu",
    log: Callable[[str], None] = print, contact_constraint: Any | None = None,
    motion_prior_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Optimize true finger controls with independent 2D and gated motion terms.

    By default body/root/wrist controls are fixed. Optional minimal arm/wrist
    updates keep native skeletal coupling, bounded angles and wrist displacement.
    The best total objective is retained; this does not prove that real error fell.
    """
    import time
    import torch

    started = time.monotonic()
    config = config or HandFitConfig()
    config.validate()
    validate_human_result(result)
    indices = np.asarray(result.get("frame_indices", np.arange(len(result["pose"]))))
    if len(indices) > 1 and np.any(np.diff(indices) != 1):
        raise ValueError("temporal hand refinement requires consecutive original frames")
    observation = validate_observations(observations, indices)
    training = {key: value.copy() for key, value in observation.items()}
    heldout = {key: value.copy() for key, value in observation.items()}
    holdout_frames = indices % config.holdout_stride == 0 if config.holdout_stride else np.zeros(len(indices), dtype=bool)
    training["confidence"][holdout_frames] = 0
    heldout["confidence"][~holdout_frames] = 0
    observed_support = supported_frames(training["confidence"], config)
    if not observed_support.any():
        raise ValueError("no sufficiently supported independent hand observations")
    original_active = np.ones_like(observed_support) if config.unobserved_policy == "bounded_temporal_prior" else observed_support
    support_envelope = support_transition_envelope(original_active, config.support_transition_frames)
    active = support_envelope > 0
    if config.spatial_acceleration_weight > 0 and motion_prior_weights is None:
        raise ValueError("spatial hand acceleration requires explicit admitted motion-prior phases")
    prior_phase = np.zeros((len(indices), 2), dtype=np.float32) if motion_prior_weights is None else np.asarray(motion_prior_weights, dtype=np.float32)
    if prior_phase.shape != (len(indices), 2) or not np.isfinite(prior_phase).all() or np.any((prior_phase < 0) | (prior_phase > 1)):
        raise ValueError("motion-prior phase weights must have shape [T,2] in [0,1]")
    k_array = np.asarray(intrinsics, dtype=np.float32)
    if k_array.shape != (3, 3) or not np.isfinite(k_array).all() or min(k_array[0, 0], k_array[1, 1]) <= 0:
        raise ValueError("intrinsics must be finite 3x3 with positive focal lengths")
    base = torch.as_tensor(np.array(result["pose"], dtype=np.float32, copy=True), device=device)
    scales = torch.as_tensor(np.array(result["scales"], dtype=np.float32, copy=True), device=device)
    k = torch.as_tensor(k_array, device=device)
    points = torch.as_tensor(observation["joints2d"], device=device)
    weights = torch.as_tensor(training["confidence"], device=device)
    weights = torch.where(weights >= config.min_confidence, weights, 0)
    # The collar extends correction priors, never the admitted image evidence.
    # Partial observations outside the original count/wrist support stay unused.
    weights = weights * torch.as_tensor(original_active, device=device)[:, :, None]
    parameter_active = np.zeros((len(base), 54), dtype=np.float32)
    parameter_active[:, :27] = active[:, 1:2]
    parameter_active[:, 27:] = active[:, :1]
    mask = torch.as_tensor(parameter_active, device=device)
    envelope = torch.as_tensor(np.concatenate([np.repeat(support_envelope[:, 1:2], 27, axis=1),
                                               np.repeat(support_envelope[:, :1], 27, axis=1)], axis=1), device=device)
    original_mask = torch.as_tensor(np.concatenate([np.repeat(original_active[:, 1:2], 27, axis=1),
                                                    np.repeat(original_active[:, :1], 27, axis=1)], axis=1).astype(np.float32), device=device)
    delta = torch.nn.Parameter(torch.zeros_like(base[:, 68:122]))
    arm_delta = torch.nn.Parameter(base.new_zeros(len(base), 14))
    arm_active = np.repeat(observed_support[:, :, None], 7, axis=2)
    # The existing unobserved policy only extends wrist rotation priors. The
    # opt-in collar below can also carry a bounded neighboring arm correction
    # into its transition interval, with no independent image target there.
    if config.unobserved_policy == "bounded_temporal_prior":
        arm_active[:, :, 4:] = original_active[:, :, None]
    original_arm_mask = torch.as_tensor(arm_active.reshape(len(base), 14).astype(np.float32), device=device)
    arm_envelope_array = np.stack([support_transition_envelope(arm_active[:, :, control], config.support_transition_frames)
                                   for control in range(7)], axis=-1)
    arm_active = arm_envelope_array > 0
    arm_mask = torch.as_tensor(arm_active.reshape(len(base), 14).astype(np.float32), device=device)
    arm_envelope = torch.as_tensor(arm_envelope_array.reshape(len(base), 14), device=device)
    arm_indices = torch.as_tensor(ARM_CONTROLS.reshape(-1), device=device)
    arm_selection = base.new_zeros(14, 136)
    arm_selection[torch.arange(14, device=device), arm_indices] = 1
    optimizer = torch.optim.Adam([delta, arm_delta] if config.optimize_wrist_chain else [delta], lr=config.learning_rate)
    motion_gate = torch.as_tensor(observation_motion_gate(training["joints2d"], training["confidence"], config), device=device)
    edge_motion_gate = torch.as_tensor(observation_edge_motion_gate(training["joints2d"], training["confidence"], config), device=device)
    update_edge_gate = torch.cat([edge_motion_gate[:, 1:2].expand(-1, 27), edge_motion_gate[:, :1].expand(-1, 27)], dim=1)
    update_edge_support = torch.maximum(mask[:-1], mask[1:])
    update_second_support = torch.maximum(torch.maximum(mask[:-2], mask[1:-1]), mask[2:])
    spatial_phase = torch.as_tensor(np.minimum(np.minimum(prior_phase[:-2], prior_phase[1:-1]), prior_phase[2:]), device=device)
    gate_controls = torch.cat([motion_gate[:, 1:2].expand(-1, 27), motion_gate[:, :1].expand(-1, 27)], dim=1)
    # A transition collar regularizes added corrections only. It must not turn
    # absent image evidence into a request to flatten the initializer's motion.
    triple = original_mask[:-2] * original_mask[1:-1] * original_mask[2:]
    relative_weights = torch.minimum(weights[:, :, 1:], weights[:, :, :1])
    velocity_weights = torch.minimum(relative_weights[1:], relative_weights[:-1])
    observed_relative = points[:, :, 1:] - points[:, :, :1]
    with torch.no_grad():
        baseline_hands = hand_forward(base, scales).detach()
        baseline_wrists = baseline_hands[:, :, 0]
        baseline_projected_h = baseline_hands @ k.T
        baseline_projected = baseline_projected_h[..., :2] / baseline_projected_h[..., 2:].clamp_min(1e-5)
        baseline_relative = baseline_projected[:, :, 1:] - baseline_projected[:, :, :1]
        baseline_relative_3d = baseline_hands[:, :, 1:] - baseline_hands[:, :, :1]
    budget_edge_support = torch.as_tensor(active[1:] | active[:-1], device=device)
    budget_common = velocity_weights >= config.min_confidence
    budget_observed_edges = budget_common.sum(-1) >= config.min_landmarks

    def assembled_pose():
        hand = base[:, 68:122] + delta * envelope
        pose = torch.cat([base[:, :68], hand, base[:, 122:]], dim=1)
        if config.optimize_wrist_chain:
            pose = pose + (arm_delta * arm_envelope) @ arm_selection
        return pose, hand

    def weighted_huber(residual, weight):
        # Vector norm Huber gives each landmark one robust residual.
        distance = torch.linalg.vector_norm(residual, dim=-1) / config.pixel_scale
        threshold = config.robust_delta_pixels / config.pixel_scale
        loss = torch.where(distance <= threshold, 0.5 * distance.square(), threshold * (distance - 0.5 * threshold))
        return (loss * weight).sum() / weight.sum().clamp_min(1)

    def evaluate():
        pose, hand = assembled_pose()
        joints = hand_forward(pose, scales)
        projected_h = joints @ k.T
        projected = projected_h[..., :2] / projected_h[..., 2:].clamp_min(1e-5)
        if not torch.isfinite(projected).all():
            raise ValueError("hand forward produced non-finite projected landmarks")
        if torch.any((joints[..., 2] <= 1e-4) & (weights > 0)):
            raise ValueError("observed hand landmarks must remain in front of the camera")
        valid_depth = (joints[..., 2] > 1e-4).to(weights.dtype)
        absolute = weighted_huber(projected - points, weights * valid_depth)
        relative = projected[:, :, 1:] - projected[:, :, :1]
        relative_loss = weighted_huber(relative - observed_relative, relative_weights * valid_depth[:, :, 1:])
        velocity = weighted_huber(torch.diff(relative, dim=0) - torch.diff(observed_relative, dim=0), velocity_weights)
        angular_acceleration = torch.diff(hand, n=2, dim=0)
        # All finger controls are angular. Wrapping avoids a 2*pi seam becoming
        # an apparent high acceleration. True visible fast motion gates this term.
        angular_acceleration = torch.atan2(torch.sin(angular_acceleration), torch.cos(angular_acceleration))
        acceleration = (angular_acceleration.square() * gate_controls * triple).sum() / triple.sum().clamp_min(1) / 0.2**2
        prior = ((delta * envelope).square()).sum() / mask.sum().clamp_min(1) / 0.35**2
        # The old full-active triple mask ignores an observation onset/offset.
        # Regularize the ADDED correction across that boundary, where inactive
        # corrections stay exactly zero. The baseline motion is never smoothed
        # by this term. Supported fast image articulation relaxes the penalty;
        # missing observations supply no asserted motion or target coordinates.
        finger_update = delta * envelope
        def per_hand_supported_mean(values, support):
            denominator = support.reshape(-1, 2, 27).sum(dim=(0, 2))
            numerator = values.reshape(-1, 2, 27).sum(dim=(0, 2))
            present = (denominator > 0).to(values.dtype)
            # A sparsely observed hand must not be diluted by the other hand's
            # much longer support interval. Unsupported sides add no penalty.
            return ((numerator / denominator.clamp_min(1)) * present).sum() / present.sum().clamp_min(1)
        finger_update_temporal = (
            per_hand_supported_mean(torch.diff(finger_update, n=2, dim=0).square() * gate_controls * update_second_support,
                                    update_second_support)
            + .25 * per_hand_supported_mean(torch.diff(finger_update, dim=0).square() * update_edge_gate * update_edge_support,
                                           update_edge_support)
        ) / 0.2**2
        data = (1 - config.relative_weight) * absolute + config.relative_weight * relative_loss
        total = data + config.velocity_weight * velocity + config.acceleration_weight * acceleration + config.pose_prior_weight * prior + config.finger_update_temporal_weight * finger_update_temporal
        parts = {"data": data, "absolute": absolute, "relative": relative_loss, "velocity": velocity,
                 "acceleration": acceleration, "prior": prior, "finger_update_temporal": finger_update_temporal, "total": total}
        if config.correction_velocity_budget_weight > 0:
            correction_px = relative - baseline_relative
            relative_3d = joints[:, :, 1:] - joints[:, :, :1]
            correction_3d = relative_3d - baseline_relative_3d
            correction_step_px = torch.linalg.vector_norm(torch.diff(correction_px, dim=0), dim=-1).mean(-1)
            correction_step_m = torch.linalg.vector_norm(torch.diff(correction_3d, dim=0), dim=-1).mean(-1)
            candidate_velocity_error = torch.linalg.vector_norm(torch.diff(relative, dim=0) - torch.diff(observed_relative, dim=0), dim=-1)
            baseline_velocity_error = torch.linalg.vector_norm(torch.diff(baseline_relative, dim=0) - torch.diff(observed_relative, dim=0), dim=-1)
            common_count = budget_common.sum(-1).clamp_min(1)
            candidate_error = (candidate_velocity_error * budget_common).sum(-1) / common_count
            baseline_error = (baseline_velocity_error * budget_common).sum(-1) / common_count
            # Match the existing engineering guard's units and budgets. These
            # losses are not guarantees or a substitute for independent checks.
            safety = config.correction_budget_safety_fraction
            # Tighten only the training margins, preserving penalty curvature
            # and the independent check's fixed thresholds. This reserves room
            # for unobserved/held-out transitions without fitting their targets.
            unknown_penalty = (torch.relu(correction_step_px - safety).square()
                               + torch.relu(correction_step_m / .005 - safety).square())
            observed_penalty = torch.relu(candidate_error - (baseline_error * 1.05 + .25 * safety)).square()
            penalty = torch.where(budget_observed_edges, observed_penalty, unknown_penalty)
            if config.optimize_wrist_chain:
                wrist_correction_px = projected[:, :, 0] - baseline_projected[:, :, 0]
                wrist_correction_m = joints[:, :, 0] - baseline_wrists
                wrist_unknown = (torch.relu(torch.linalg.vector_norm(torch.diff(wrist_correction_px, dim=0), dim=-1) - safety).square()
                                 + torch.relu(torch.linalg.vector_norm(torch.diff(wrist_correction_m, dim=0), dim=-1) / .005 - safety).square())
                observed_wrist_velocity = torch.diff(points[:, :, 0], dim=0)
                candidate_wrist_error = torch.linalg.vector_norm(torch.diff(projected[:, :, 0], dim=0) - observed_wrist_velocity, dim=-1)
                baseline_wrist_error = torch.linalg.vector_norm(torch.diff(baseline_projected[:, :, 0], dim=0) - observed_wrist_velocity, dim=-1)
                wrist_observed = torch.minimum(weights[1:, :, 0], weights[:-1, :, 0]) >= config.min_confidence
                wrist_penalty = torch.where(wrist_observed,
                                           torch.relu(candidate_wrist_error - (baseline_wrist_error * 1.05 + .25 * safety)).square(),
                                           wrist_unknown)
                penalty = penalty + wrist_penalty
            supported_penalty = penalty * budget_edge_support
            denominator = budget_edge_support.sum(0).clamp_min(1)
            per_hand_mean = supported_penalty.sum(0) / denominator
            # Include the worst edge so a long clip cannot dilute a single
            # observation onset. Both are differentiable scalar objectives.
            worst_edge = supported_penalty.amax(0) if len(penalty) else penalty.new_zeros(2)
            present = budget_edge_support.any(0).to(penalty.dtype)
            correction_budget = ((per_hand_mean + worst_edge) * present).sum() / present.sum().clamp_min(1)
            total = total + config.correction_velocity_budget_weight * correction_budget
            parts.update(correction_velocity_budget=correction_budget, total=total)
        if config.optimize_wrist_chain:
            wrist_observation = weighted_huber(projected[:, :, 0] - points[:, :, 0], weights[:, :, 0])
            wrist_anchor = (joints[:, :, 0] - baseline_wrists).square().sum(-1).mean() / 0.02**2
            arm_prior = ((arm_delta * arm_envelope).square()).sum() / arm_mask.sum().clamp_min(1) / 0.08**2
            wrist_pose = torch.cat([pose[:, 47:50], pose[:, 37:40]], dim=1)
            wrist_mask = torch.cat([original_arm_mask[:, 4:7], original_arm_mask[:, 11:14]], dim=1)
            wrist_triple = wrist_mask[:-2] * wrist_mask[1:-1] * wrist_mask[2:]
            wrist_gate = torch.cat([motion_gate[:, :1].expand(-1, 3), motion_gate[:, 1:2].expand(-1, 3)], dim=1)
            wrist_second = torch.diff(wrist_pose, n=2, dim=0)
            wrist_second = torch.atan2(torch.sin(wrist_second), torch.cos(wrist_second))
            wrist_acceleration = (wrist_second.square() * wrist_gate * wrist_triple).sum() / wrist_triple.sum().clamp_min(1) / 0.2**2
            # Smooth the correction, not the original observed arm trajectory.
            # This lets held-out detections inherit a neighboring correction
            # instead of reverting every Nth frame to the raw initializer.
            arm_update = arm_delta * arm_envelope
            arm_update_temporal = (torch.diff(arm_update, n=2, dim=0).square().sum() / max(1, (len(base) - 2) * 14)
                                   + 0.25 * torch.diff(arm_update, dim=0).square().sum() / max(1, (len(base) - 1) * 14)) / 0.08**2
            total = total + config.wrist_observation_weight * wrist_observation + config.wrist_anchor_weight * wrist_anchor + config.arm_prior_weight * arm_prior + config.acceleration_weight * wrist_acceleration + config.arm_update_temporal_weight * arm_update_temporal
            parts.update(wrist_observation=wrist_observation, wrist_anchor=wrist_anchor, arm_prior=arm_prior,
                         wrist_acceleration=wrist_acceleration, arm_update_temporal=arm_update_temporal, total=total)
        if contact_constraint is not None:
            contact_terms = contact_constraint.loss(joints)
            total = total + contact_terms["contact_total"]
            parts.update(contact_terms)
            parts["total"] = total
        if config.spatial_acceleration_weight > 0:
            # This sees every articulation that moves the palm/fingers, including
            # upper-arm/forearm twist; smoothing finger Euler controls alone
            # cannot address a rotating wrist's inherited orientation spike.
            relative_3d = joints[:, :, 1:] - joints[:, :, :1]
            acceleration_3d = torch.linalg.vector_norm(torch.diff(relative_3d, n=2, dim=0), dim=-1) / 0.01
            robust_3d = torch.sqrt(1 + acceleration_3d.square()) - 1
            phase_gate = spatial_phase * motion_gate
            spatial_loss = (robust_3d * phase_gate[:, :, None]).sum() / (spatial_phase.sum() * 20).clamp_min(1)
            total = total + config.spatial_acceleration_weight * spatial_loss
            parts.update(spatial_hand_acceleration=spatial_loss, total=total)
        return total, parts, pose, projected

    with torch.no_grad():
        initial, initial_parts, _, initial_projected = evaluate()
        before = hand_diagnostics(initial_projected.cpu().numpy(), observation, active)
        before_training = hand_diagnostics(initial_projected.cpu().numpy(), training, active)
        before_heldout = hand_diagnostics(initial_projected.cpu().numpy(), heldout, active)
    best_loss = float(initial)
    best_delta = delta.detach().clone()
    best_arm_delta = arm_delta.detach().clone()
    trace = []
    for iteration in range(config.iterations):
        optimizer.zero_grad(set_to_none=True)
        total, parts, _, _ = evaluate()
        if not torch.isfinite(total):
            raise ValueError(f"non-finite hand objective at iteration {iteration}")
        value = float(total.detach())
        if value < best_loss:
            best_loss, best_delta = value, delta.detach().clone()
            best_arm_delta = arm_delta.detach().clone()
        total.backward()
        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise ValueError("hand control gradient is missing or non-finite")
        if config.optimize_wrist_chain and (arm_delta.grad is None or not torch.isfinite(arm_delta.grad).all()):
            raise ValueError("arm/wrist control gradient is missing or non-finite")
        torch.nn.utils.clip_grad_norm_([delta, arm_delta] if config.optimize_wrist_chain else [delta], 10.0)
        optimizer.step()
        with torch.no_grad():
            delta.clamp_(-config.max_delta_radians, config.max_delta_radians)
            delta.mul_(mask)
            if config.optimize_wrist_chain:
                arm_delta.clamp_(-config.arm_delta_limit_radians, config.arm_delta_limit_radians)
                arm_delta.mul_(arm_mask)
                # Project bounded arm updates back into a true 3D wrist trust
                # region. The native kinematic forward enforces the coupling.
                for _ in range(4):
                    candidate, _ = assembled_pose()
                    distance = torch.linalg.vector_norm(hand_forward(candidate, scales)[:, :, 0] - baseline_wrists, dim=-1)
                    if float(distance.max()) <= config.wrist_max_displacement_metres + 1e-6:
                        break
                    factor = torch.where(distance > config.wrist_max_displacement_metres,
                                         config.wrist_max_displacement_metres / distance.clamp_min(1e-8) * 0.999,
                                         torch.ones_like(distance))
                    arm_delta.mul_(factor[:, :, None].expand(-1, -1, 7).reshape(len(base), 14))
        if iteration == 0 or (iteration + 1) % 20 == 0 or iteration + 1 == config.iterations:
            record = {"iteration": iteration + 1, **{key: float(value.detach()) for key, value in parts.items()}}
            trace.append(record)
            log(f"Hand iteration {iteration + 1}/{config.iterations}: total={record['total']:.6f}, data={record['data']:.6f}")
    with torch.no_grad():
        last, _, _, _ = evaluate()
        if float(last) < best_loss:
            best_loss, best_delta = float(last), delta.detach().clone()
            best_arm_delta = arm_delta.detach().clone()
        delta.copy_(best_delta)
        arm_delta.copy_(best_arm_delta)
        _, final_parts, pose, projected = evaluate()
        after = hand_diagnostics(projected.cpu().numpy(), observation, active)
        after_training = hand_diagnostics(projected.cpu().numpy(), training, active)
        after_heldout = hand_diagnostics(projected.cpu().numpy(), heldout, active)
        out_pose = pose.cpu().numpy().astype(np.float32)
        final_hands = hand_forward(pose, scales)
        final_wrists = final_hands[:, :, 0]
        wrist_displacements = torch.linalg.vector_norm(final_wrists - baseline_wrists, dim=-1).cpu().numpy()
        if config.optimize_wrist_chain and wrist_displacements.max() > config.wrist_max_displacement_metres + 1e-5:
            raise ValueError("optimized wrists left the configured 3D displacement trust region")
    untouched = np.r_[np.arange(68), np.arange(122, 136)]
    if config.optimize_wrist_chain:
        untouched = np.setdiff1d(untouched, ARM_CONTROLS.reshape(-1))
    if not np.array_equal(out_pose[:, untouched], np.asarray(result["pose"])[:, untouched]):
        raise AssertionError("body, root or wrist controls changed during hand-only fitting")
    for side, (start, stop) in enumerate(HAND_SLICES):
        if not np.array_equal(out_pose[~active[:, side], start:stop], np.asarray(result["pose"])[~active[:, side], start:stop]):
            raise AssertionError("inactive finger controls changed from the baseline")
    boundaries = correction_boundary_diagnostics(baseline_hands.cpu().numpy(), final_hands.cpu().numpy(), active, indices, k_array, observation, config.min_confidence)
    original_boundaries = correction_boundary_diagnostics(baseline_hands.cpu().numpy(), final_hands.cpu().numpy(), original_active, indices, k_array, observation, config.min_confidence)
    improvement = float(initial) - best_loss
    retained_zero_update = bool(np.array_equal(out_pose, np.asarray(result["pose"])))
    return {
        "pose": out_pose, "scales": np.array(result["scales"], dtype=np.float32, copy=True),
        "shape": np.array(result["shape"], dtype=np.float32, copy=True),
        "frame_indices": indices.copy(), "fps": np.asarray(result.get("fps", 30), dtype=np.float32),
        "hand_projected": projected.cpu().numpy(), "active_hands": active,
        "original_active_hands": original_active, "support_transition_envelope": support_envelope,
        "hand_joints_m": final_hands.cpu().numpy(),
        "optimization_weights": weights.cpu().numpy(),
        "report": {
            "method": "independent_2d_finger_enabled_native_mhr_sequence_fit", "config": asdict(config),
            "seconds": round(time.monotonic() - started, 3), "device": device,
            "initial_objective": float(initial), "best_objective": best_loss,
            "optimization_progress": {
                "objective_improvement": improvement,
                "relative_objective_improvement": improvement / max(abs(float(initial)), 1e-12),
                "retained_zero_update": retained_zero_update,
                "interpretation": "A zero retained update is an optimization stall or abstention, not successful hand refinement. Objective improvement alone does not establish reconstruction quality.",
            },
            "initial_terms": {key: float(value) for key, value in initial_parts.items()},
            "final_terms": {key: float(value) for key, value in final_parts.items()},
            "trace": trace, "before": before, "after": after,
            "before_training": before_training, "after_training": after_training,
            "before_heldout": before_heldout, "after_heldout": after_heldout,
            "heldout_frame_indices": indices[holdout_frames].tolist(),
            "heldout_policy": "All image position, velocity and motion gate targets on these frames excluded from fitting; temporal priors may update them",
            "finger_update_temporal_semantics": "Per-hand normalized first/second differences of the added masked finger correction include active/inactive boundaries; inactive values stay at baseline. Independently observed wrist-relative fast motion relaxes the penalty; no missing image targets or baseline trajectory smoothing are introduced.",
            "support_transition_semantics": "Opt-in bounded cosine correction collar outside original admitted support; no independent image coordinates are added. Both original and expanded support boundaries remain reported. Zero collar reproduces the previous support policy.",
            "correction_velocity_budget_semantics": "Opt-in loss on every modified edge: unknown wrist-relative finger correction steps above 1 pixel or 0.005 native estimated metres, or training-observed velocity discrepancy above baseline*1.05+0.25 pixels. Optional wrist-chain correction receives corresponding wrist bounds. Mean plus worst-edge loss discourages dilution on long clips; independent post-fit checks remain required.",
            "correction_budget_safety_semantics": "An optional fraction in (0,1] tightens only the internal training budgets: fraction pixels, fraction*0.005 native estimated metres, and baseline*1.05+fraction*0.25 observed-velocity pixels. The loss curvature and independent diagnostic thresholds remain unchanged; no held-out or missing image targets enter training.",
            "support_boundary_motion": boundaries, "inactive_finger_controls_exactly_preserved": True,
            "original_support_boundary_motion": original_boundaries,
            "prior_only_frame_indices": {name: indices[active[:, side] & ~observed_support[:, side]].tolist()
                                         for side, name in enumerate(("left", "right"))},
            "admitted_motion_prior_frame_indices": {name: indices[prior_phase[:, side] > 0].tolist()
                                                     for side, name in enumerate(("left", "right"))},
            "finger_controls": {"right": [68, 95], "left": [95, 122], "interval_end_exclusive": True},
            "body_root_wrist_controls_unchanged": not config.optimize_wrist_chain, "identity_and_scales_unchanged": True,
            "root_and_all_nonselected_controls_unchanged": True,
            "arm_wrist_free_control_indices": ARM_CONTROLS.tolist() if config.optimize_wrist_chain else [],
            "arm_wrist_max_absolute_control_delta_radians": float(np.abs(out_pose[:, ARM_CONTROLS.reshape(-1)] - np.asarray(result["pose"])[:, ARM_CONTROLS.reshape(-1)]).max()),
            "max_wrist_displacement_metres": float(wrist_displacements.max()),
            "p95_wrist_displacement_metres": float(np.percentile(wrist_displacements, 95)),
            "max_absolute_control_delta_radians": float(np.abs(out_pose[:, 68:122] - np.asarray(result["pose"])[:, 68:122]).max()),
            "object_contact_constraints": contact_constraint is not None, "official_scores_available": False,
            "contact": ({"before": contact_constraint.before, "after": contact_constraint.diagnostics(final_hands.cpu().numpy()),
                          "anchor_weight": contact_constraint.anchor_weight, "penetration_weight": contact_constraint.penetration_weight,
                          "status": "experimental_uncertain_object_hypothesis_not_reference_contact"} if contact_constraint is not None else None),
            "limitations": [("Camera fixed; only bounded selected arm/wrist controls change" if config.optimize_wrist_chain else
                             "Camera and body/wrist initializer remain fixed"), "Monocular depth ambiguity remains",
                            "Reliability weights are not calibrated visibility probabilities",
                            ("Unsupported long gaps are unchanged" if config.unobserved_policy == "preserve" else
                             "Unobserved hands are regularized by a bounded motion prior, not recovered independent observations"),
                            "Image landmark agreement and smoother motion do not establish hidden-reference accuracy"],
        },
    }
