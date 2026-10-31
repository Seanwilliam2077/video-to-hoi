"""Independently review one source-bound EP16 occluded-hand contact experiment.

Manual parameter eligibility is not an image observation. Depth-gauge changes
are inspected with every human under both object hypotheses. All measurements
remain conditional reconstruction diagnostics, never hidden-reference scores.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ROLES = ("baseline", "old_a", "old_b", "a", "b")
PAIRS = (("a", "b"), ("old_a", "a"), ("old_b", "b"), ("baseline", "a"), ("baseline", "b"))
FINGERS = ((95, 122), (68, 95))
ARM = np.asarray([list(range(43, 50)), list(range(33, 40))])


def validate_frozen_criteria(criteria):
    expected = {
        "image_reference_policy": {"relative_tolerance": 1.05, "absolute_tolerance_pixels": .25},
        "motion_policy": {"observed_velocity_relative_tolerance": 1.05,
            "observed_velocity_absolute_tolerance_pixels": .25,
            "unknown_motion_max_correction_step_pixels": 1.,
            "unknown_motion_max_correction_step_native_estimated_metres": .005},
        "skin_b_versus_a": {"maximum_mean_depth_increase_mm": .5,
            "maximum_worst_vertex_depth_increase_mm": 2.},
    }
    for name, fields in expected.items():
        for key, value in fields.items():
            if criteria.get(name, {}).get(key) != value:
                raise ValueError("Historical frozen engineering tolerance changed: " + name + "/" + key)
    return True


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers(tool_root):
    old = load_module(tool_root / "tools/evaluate_contact_ab.py", "occluded_contact_old_evaluator")
    policy, refined = old.load_helpers(tool_root)
    return old, policy, refined


def manual_control_envelope(annotation, count, video_sha256):
    """Independently reconstruct the declared prior-only cosine envelope."""
    if (annotation.get("schema_version") != "native_hand_control_prior.v1"
            or annotation.get("track") != 1 or annotation.get("episode") != 16
            or annotation.get("frame_count") != count
            or annotation.get("video_sha256") != video_sha256
            or annotation.get("role") != "parameter_eligibility_only_no_image_targets"
            or not str(annotation.get("annotation_method", "")).strip()):
        raise ValueError("Manual eligibility must identify EP16 and its non-observation role")
    envelope = np.zeros((count, 2), np.float32)
    occupied = np.zeros((count, 2), bool)
    for interval in annotation["eligible_intervals_inclusive"]:
        first, last = interval["first"], interval["last"]
        ramp = interval.get("transition_frames", 8)
        weight = interval.get("control_weight", 1.0)
        sides = interval.get("eligible_sides", [])
        if (type(first) is not int or type(last) is not int or not 0 <= first <= last < count
                or type(ramp) is not int or ramp < 0 or not sides
                or len(set(sides)) != len(sides) or any(side != "left" for side in sides)
                or isinstance(weight, bool) or not np.isfinite(weight) or not 0 < weight <= 1):
            raise ValueError("Invalid or non-left prior-only eligibility interval")
        position = np.arange(last - first + 1, dtype=np.float64)
        distance = np.minimum(position + 1, len(position) - position)
        values = weight * .5 * (1 - np.cos(np.pi * np.minimum(distance / (ramp + 1), 1)))
        for name in sides:
            side = ("left", "right").index(name)
            if occupied[first:last + 1, side].any():
                raise ValueError("Manual eligibility intervals overlap")
            occupied[first:last + 1, side] = True
            envelope[first:last + 1, side] = values.astype(np.float32)
    if not np.any(envelope[:, 0] > 0):
        raise ValueError("Manual left-hand eligibility is empty")
    return envelope


def image_arm_envelope(confidence, heldout, config):
    """Reconstruct the old per-control arm collar without importing fitter maths."""
    admitted = np.asarray(confidence) >= config["min_confidence"]
    admitted[np.asarray(heldout, bool)] = False
    observed = admitted[:, :, 0] & (admitted[:, :, 1:].sum(-1) >= config["min_landmarks"])
    for side in range(2):
        detected = np.flatnonzero(observed[:, side])
        for first, last in zip(detected[:-1], detected[1:]):
            if last - first - 1 <= config["max_gap_frames"]:
                observed[first:last + 1, side] = True
    support = np.repeat(observed[:, :, None], 7, axis=2)
    if config["unobserved_policy"] == "bounded_temporal_prior":
        support[:, :, 4:] = True
    result = support.astype(np.float32)
    collar = config["support_transition_frames"]
    for distance in range(1, min(collar, len(support) - 1) + 1):
        weight = np.float32(.5 * (1 + np.cos(np.pi * distance / (collar + 1))))
        result[distance:] = np.maximum(result[distance:], support[:-distance] * weight)
        result[:-distance] = np.maximum(result[:-distance], support[distance:] * weight)
    return result


def declared_control_limits(manual, config):
    """The new left prior can change solver bounds, never the evaluation guards."""
    arm_base, wrist_base = config["arm_delta_limit_radians"], config["wrist_max_displacement_metres"]
    arm_prior = config.get("control_prior_arm_delta_limit_radians")
    wrist_prior = config.get("control_prior_wrist_max_displacement_metres")
    arm_prior = arm_base if arm_prior is None else arm_prior
    wrist_prior = wrist_base if wrist_prior is None else wrist_prior
    if not all(np.isfinite(x) and x > 0 for x in (arm_base, wrist_base, arm_prior, wrist_prior)):
        raise ValueError("Declared angular/wrist solver limits must be finite and positive")
    arm = arm_base + (arm_prior - arm_base) * np.asarray(manual)
    wrist = wrist_base + (wrist_prior - wrist_base) * np.asarray(manual)
    return np.repeat(arm[:, :, None], 7, axis=2), wrist


def validate_prior_arrays(arrays, manual, old_arrays, confidence, heldout, threshold, old, expected_arm=None):
    """A/B share eligibility; inherited image evidence must remain byte-identical."""
    support = {role: arrays[role]["support"] for role in ("a", "b")}
    weights = {role: arrays[role]["weights"] for role in ("a", "b")}
    result = old.validate_controlled_arrays(support, weights, confidence, heldout, threshold)
    names = ("original_support", "control_prior_array", "arm_envelope", "image_envelope", "effective_envelope")
    for name in names:
        if not np.array_equal(arrays["a"][name], arrays["b"][name]):
            raise ValueError("A/B differ in exact " + name)
    count = len(manual)
    for role in ("a", "b"):
        current = arrays[role]
        if not np.array_equal(current["original_support"], old_arrays["original_support"]):
            raise ValueError("Original image support changed")
        if not np.array_equal(current["weights"], old_arrays["weights"]):
            raise ValueError("Image target weights changed from old A")
        prior = np.asarray(current["control_prior_array"])
        image = np.asarray(current["image_envelope"])
        effective = np.asarray(current["effective_envelope"])
        arm = np.asarray(current["arm_envelope"])
        if (prior.shape != (count, 2) or image.shape != prior.shape or effective.shape != prior.shape
                or arm.shape != (count, 2, 7) or not all(np.isfinite(x).all() for x in (prior, image, effective, arm))
                or any(np.any((x < 0) | (x > 1)) for x in (prior, image, effective, arm))):
            raise ValueError("Invalid manual/image/finger/arm control envelopes")
        if not np.allclose(prior, manual, atol=1e-7, rtol=0):
            raise ValueError("Saved manual envelope differs from the source-bound annotation")
        if not np.array_equal(image, old_arrays["effective_envelope"]):
            raise ValueError("Original image-only correction collar changed")
        if not np.allclose(effective, np.maximum(image, manual), atol=1e-7, rtol=0):
            raise ValueError("Effective finger mask is not max(image collar, manual prior)")
        if not np.array_equal(current["support"], effective > 0):
            raise ValueError("Mutable finger support differs from its effective envelope")
        if np.any(arm + 1e-7 < manual[:, :, None]):
            raise ValueError("Manual eligibility did not reach each selected arm control")
        if expected_arm is not None and not np.allclose(arm, np.maximum(expected_arm, manual[:, :, None]), atol=1e-7, rtol=0):
            raise ValueError("Arm envelope expanded outside original image collar or manual eligibility")
    result.update({"manual_prior_recomputed_from_annotation": True,
        "image_support_and_exact_weights_equal_old_a": True,
        "manual_prior_positive_frames": (manual > 0).sum(0).tolist(),
        "manual_eligibility_role": "Parameter availability only; no hidden image observations",
        "exact_arm_and_finger_envelopes_identical": True})
    return result


def update_diagnostics(base_pose, pose, base_hands, hands, summary, windows, *, envelope=None, arm_envelope=None,
                       config=None, arm_limits=None, wrist_limits=None, camera=None, manual=None):
    """Expose cumulative updates, including modifications made before paired fitting."""
    delta = np.asarray(pose, float) - np.asarray(base_pose, float)
    if delta.shape != (len(hands), 136) or np.asarray(base_hands).shape != np.asarray(hands).shape:
        raise ValueError("Update diagnostics require common full-frame native trajectories")
    output = {"sides": {}, "interpretation": "Cumulative prediction changes, not true hidden motion or accuracy"}
    for side, name in enumerate(("left", "right")):
        first, last = FINGERS[side]
        finger = np.abs(delta[:, first:last])
        arm = np.abs(delta[:, ARM[side]])
        wrist = np.linalg.norm(hands[:, side, 0] - base_hands[:, side, 0], axis=-1)
        projected_wrist = None
        if camera is not None:
            old = base_hands[:, side, 0] @ np.asarray(camera).T
            new = hands[:, side, 0] @ np.asarray(camera).T
            projected_wrist = np.linalg.norm(new[:, :2] / new[:, 2:] - old[:, :2] / old[:, 2:], axis=-1)
        rows = []
        for window in [{"label": "whole_clip", "first": 0, "last": len(hands) - 1}, *windows]:
            mask = np.zeros(len(hands), bool)
            mask[window["first"]:window["last"] + 1] = True
            rows.append({"annotation": window,
                "finger_absolute_delta_radians": summary(finger[mask]),
                "selected_arm_absolute_delta_radians": summary(arm[mask]),
                "wrist_displacement_native_estimated_mm": summary(wrist[mask] * 1000),
                "wrist_projected_displacement_pixels": summary(projected_wrist[mask]) if projected_wrist is not None else None,
                "changed_finger_original_frame_ids": np.flatnonzero(mask & (finger.max(-1) > 1e-7)).tolist(),
                "changed_arm_original_frame_ids": np.flatnonzero(mask & (arm.max(-1) > 1e-7)).tolist()})
        row = {"phase_summaries": rows}
        if envelope is not None:
            outside = np.asarray(envelope)[:, side] <= 0
            if np.any(finger[outside] > 1e-7):
                raise ValueError("Finger controls changed outside declared availability")
            row["inactive_finger_controls_unchanged"] = True
            if config is not None and np.any(finger > config["max_delta_radians"] * np.asarray(envelope)[:, side, None] + 2e-6):
                raise ValueError("Finger updates exceed the actual ramped envelope")
        if arm_envelope is not None and config is not None:
            limit = np.asarray(arm_limits)[:, side] if arm_limits is not None else config["arm_delta_limit_radians"]
            wrist_limit = np.asarray(wrist_limits)[:, side] if wrist_limits is not None else config["wrist_max_displacement_metres"]
            if np.any(arm > limit * np.asarray(arm_envelope)[:, side] + 2e-6):
                raise ValueError("Arm updates exceed the actual ramped envelope")
            if np.any(wrist > wrist_limit + 2e-6):
                raise ValueError("Native wrist displacement exceeds the frozen budget")
            row["ramped_finger_arm_and_wrist_budgets_passed"] = True
        projection_limit = config.get("control_prior_wrist_max_projection_pixels") if config is not None else None
        if projection_limit is not None:
            if projected_wrist is None or manual is None:
                raise ValueError("Projective wrist trust requires the shared camera and manual prior mask")
            eligible = np.asarray(manual)[:, side] > 0
            if np.any(projected_wrist[eligible] > projection_limit + 1e-4):
                raise ValueError("Manual-prior wrist update exceeds the frozen projective trust limit")
            row["manual_prior_projective_wrist_trust_passed"] = True
            row["manual_prior_projected_wrist_displacement_pixels"] = summary(projected_wrist[eligible])
        output["sides"][name] = row
    return output


def boundary_windows(manual, radius=8):
    """Source-ID windows at every manual eligibility entry/exit, including collars."""
    current = np.asarray(manual) > 0
    rows = []
    for side, name in enumerate(("left", "right")):
        transitions = np.flatnonzero(np.diff(np.r_[False, current[:, side], False].astype(int)))
        for boundary in transitions:
            point = int(min(boundary, len(current) - 1))
            rows.append({"label": name + "_manual_eligibility_boundary_" + str(boundary),
                "side": name, "first": max(0, point - radius), "last": min(len(current) - 1, point + radius),
                "boundary_kind": "eligibility_entry_or_exit",
                "eligibility_boundary_original_frame_id": int(boundary)})
        for first, last in zip(transitions[::2], transitions[1::2]):
            values = np.asarray(manual)[first:last, side]
            plateau = np.flatnonzero(np.isclose(values, values.max(), atol=1e-7, rtol=0)) + first
            for point in np.unique(plateau[[0, -1]]):
                if point in (first, last - 1):
                    continue
                rows.append({"label": name + "_manual_ramp_plateau_" + str(point), "side": name,
                    "first": max(0, int(point) - radius), "last": min(len(current) - 1, int(point) + radius),
                    "boundary_kind": "ramp_plateau_endpoint", "eligibility_boundary_original_frame_id": int(point)})
    return rows


def anchor_consistency_by_side(hands, candidate, targets, weights, to_object):
    """Keep same-target fit diagnostics separate for left and right hands."""
    if targets is None:
        return {"available": False, "interpretation": "This intervention did not use frozen fingertip anchors"}
    local = to_object(np.asarray(hands)[:, :, [4, 8, 12, 16, 20]], candidate["object_to_camera"])
    target, weight = np.asarray(targets), np.asarray(weights)
    if (target.shape != local.shape or weight.shape != local.shape[:-1]
            or not np.isfinite(target).all() or not np.isfinite(weight).all() or np.any(weight < 0)):
        raise ValueError("Frozen same-anchor diagnostics require finite aligned targets/weights")
    distance = np.linalg.norm(local - target, axis=-1) * 1000
    return {"available": True, "sides": {name: {
        "positive_anchor_samples": int((weight[:, side] > 0).sum()),
        "weighted_distance_mm": float((distance[:, side] * weight[:, side]).sum() / weight[:, side].sum()) if weight[:, side].sum() else None}
        for side, name in enumerate(("left", "right"))},
        "interpretation": "Fit to the same intervention-selected heuristic anchors; not independent true contact accuracy"}


def convex_ray_bounds(points, transform, normals, offsets):
    """Ray/convex-body depths in camera z; invalid or nonintersecting rays abstain."""
    points = np.asarray(points, float)
    transform, normals, offsets = np.asarray(transform, float), np.asarray(normals, float), np.asarray(offsets, float)
    if (points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4)
            or normals.ndim != 2 or normals.shape[1] != 3 or offsets.shape != (len(normals),)
            or not all(np.isfinite(x).all() for x in (points, transform, normals, offsets))
            or np.any(points[:, 2] <= 0)):
        raise ValueError("Ordering rays require finite positive-depth camera points and convex planes")
    rotation, translation = transform[:3, :3], transform[:3, 3]
    origin = -translation @ rotation
    direction = (points / points[:, 2:]) @ rotation
    numerator = offsets - normals @ origin
    denominator = direction @ normals.T
    parallel = np.abs(denominator) < 1e-12
    valid_parallel = ~np.any(parallel & (numerator[None] < -1e-10), axis=1)
    quotient = np.divide(numerator[None], denominator, out=np.zeros_like(denominator), where=~parallel)
    near = np.maximum(0, np.max(np.where(denominator < -1e-12, quotient, -np.inf), axis=1))
    far = np.min(np.where(denominator > 1e-12, quotient, np.inf), axis=1)
    hit = valid_parallel & np.isfinite(near) & np.isfinite(far) & (far >= near) & (far > 0)
    return np.where(hit, near, np.nan), np.where(hit, far, np.nan), hit


def ordering_diagnostics(hands, candidate, summary, first=90, last=230):
    """Report coarse near/far evidence without inventing exact hidden fingertip labels."""
    if not 0 <= first <= last < len(hands):
        raise ValueError("Ordering interval lies outside original frame IDs")
    rows = []
    for frame in range(first, last + 1):
        points = hands[frame].reshape(-1, 3)
        near, far, hit = convex_ray_bounds(points, candidate["object_to_camera"][frame], candidate["normals"], candidate["offsets"])
        z = points[:, 2]
        for side, name in enumerate(("left", "right")):
            select = slice(side * 21, (side + 1) * 21)
            admitted = hit[select]
            front = (z[select] - near[select]) * 1000
            back = (z[select] - far[select]) * 1000
            rows.append({"original_frame_id": frame, "side": name,
                "object_pose_usable": bool(candidate["pose_usable"][frame]),
                "ray_intersecting_joint_indices": np.flatnonzero(admitted).tolist(),
                "depth_minus_near_surface_mm": [float(value) if valid else None for value, valid in zip(front, admitted)],
                "depth_minus_far_surface_mm": [float(value) if valid else None for value, valid in zip(back, admitted)],
                "coarse_order_conflicts": int(np.sum((front < 0 if side == 0 else front > 0) & admitted)),
                "intersecting_joint_count": int(admitted.sum())})
    reduced = {}
    for name in ("left", "right"):
        reduced[name] = {}
        for scope in ("all_original_frames", "same_fixed_pose_usable_gate", "training", "stride5_holdout", "usable_training", "usable_stride5_holdout"):
            chosen = [r for r in rows if r["side"] == name
                and (scope not in ("same_fixed_pose_usable_gate", "usable_training", "usable_stride5_holdout") or r["object_pose_usable"])
                and (scope not in ("training", "usable_training") or r["original_frame_id"] % 5 != 0)
                and (scope not in ("stride5_holdout", "usable_stride5_holdout") or r["original_frame_id"] % 5 == 0)]
            front = [x for r in chosen for x in r["depth_minus_near_surface_mm"] if x is not None]
            back = [x for r in chosen for x in r["depth_minus_far_surface_mm"] if x is not None]
            reduced[name][scope] = {"frames": len(chosen),
                "intersecting_joint_samples": sum(r["intersecting_joint_count"] for r in chosen),
                "coarse_order_conflict_samples": sum(r["coarse_order_conflicts"] for r in chosen),
                "depth_minus_near_surface_mm": summary(front), "depth_minus_far_surface_mm": summary(back)}
    return {"original_frame_interval_inclusive": [first, last], "sides": reduced, "rows": rows,
        "interpretation": "Coarse RGB evidence expects visible right hand near/in front and hidden left hand behind the visible front surface. Native joint-centre ray diagnostics are not skin visibility, exact fingertip contacts, a far-surface contact label or an accuracy score."}


def distributed_native_ids(rest, candidates, count=128):
    """Independent deterministic farthest-point selection, including stable ties."""
    candidates = np.sort(np.asarray(candidates, np.int64))
    positions = np.asarray(rest)[candidates].astype(np.float64)
    if len(np.unique(candidates)) != len(candidates) or not 1 <= count <= len(candidates):
        raise ValueError("Farthest-point selection needs distinct candidates and bounded count")
    chosen = [int(np.argmax(np.square(positions - positions.mean(0)).sum(-1)))]
    distance = np.square(positions - positions[chosen[0]]).sum(-1)
    distance[chosen] = -1
    while len(chosen) < count:
        next_index = int(np.argmax(distance))
        chosen.append(next_index)
        distance = np.minimum(distance, np.square(positions - positions[next_index]).sum(-1))
        distance[chosen] = -1
    return np.sort(candidates[chosen])


def selected_surface_signed(vertices, candidate, vertex_indices):
    world = np.asarray(vertices[:, vertex_indices], float)
    local = np.einsum("tsni,tij->tsnj", world - candidate["object_to_camera"][:, None, None, :3, 3],
        candidate["object_to_camera"][:, :3, :3])
    signed = np.empty(local.shape[:-1], np.float64)
    for frame in range(len(vertices)):
        signed[frame] = (local[frame] @ candidate["normals"].T - candidate["offsets"]).max(-1)
    return local, signed


def select_patch_positions(signed, weights, original_ids):
    positions = []
    for side in range(2):
        training = (np.asarray(original_ids) % 5 != 0) & (weights[:, side] > 0)
        if not training.any():
            raise ValueError("Native surface patches require training-only admitted phase samples")
        median = np.median(np.abs(signed[training, side]), axis=0)
        positions.append(np.argsort(median, kind="stable")[:16])
    return np.asarray(positions, np.int64)


def surface_surrogate_terms(signed, hands, candidate, weights, patches, configuration, reference_hands=None):
    """Reconstruct the current surface intervention independently in NumPy."""
    penetration_parts, attraction_parts = [], []
    for side in range(2):
        phase = weights[:, side]
        penalty = np.maximum(-signed[:, side] - .003, 0) ** 2 / .01 ** 2
        mean = float((penalty.mean(-1) * phase).sum() / max(1., phase.sum()))
        worst = float((penalty.max(-1) * phase).max())
        penetration_parts.append(mean + .25 * worst)
        distance = np.abs(signed[:, side, patches[side]]) / .02
        robust = np.sqrt(1 + distance ** 2) - 1
        attraction_parts.append(float((robust.mean(-1) * phase).sum() / max(1., phase.sum())))
    active_ids = np.flatnonzero((weights > 0).any(-1))
    if not len(active_ids):
        raise ValueError("Surface intervention diagnostics require a nonempty phase")
    mode = configuration.get("order_ray_mode", "current_query")
    if mode not in ("current_query", "fixed_shared_seed") or (mode == "fixed_shared_seed" and reference_hands is None):
        raise ValueError("Far-order ray mode must explicitly bind its shared-seed reference")
    gaps, valid_rows, current_valid_rows, reference_valid_rows = [], [], [], []
    for frame in active_ids:
        query = np.stack([hands[frame, 0, 0], hands[frame, 0, [5, 9, 13, 17]].mean(0)])
        _, far, valid = convex_ray_bounds(query, candidate["object_to_camera"][frame], candidate["normals"], candidate["offsets"])
        current_valid_rows.append(valid.copy())
        reference_far, reference_valid = far, valid
        if reference_hands is not None:
            reference = np.stack([reference_hands[frame, 0, 0], reference_hands[frame, 0, [5, 9, 13, 17]].mean(0)])
            _, reference_far, reference_valid = convex_ray_bounds(reference, candidate["object_to_camera"][frame], candidate["normals"], candidate["offsets"])
        reference_valid_rows.append(reference_valid.copy())
        if mode == "fixed_shared_seed":
            far, valid = reference_far, reference_valid
        gaps.append(np.where(valid, np.maximum(far + .003 - query[:, 2], 0), 0))
        valid_rows.append(valid)
    valid = np.asarray(valid_rows, bool)
    gaps = np.asarray(gaps)
    order_weights = weights[active_ids, 0, None] * valid
    order = float(((gaps / .02) ** 2 * order_weights).sum() / max(1., order_weights.sum()))
    penetration, attraction = float(np.mean(penetration_parts)), float(np.mean(attraction_parts))
    result = {"surface_skin_penetration": penetration, "surface_sliding_attraction": attraction,
        "surface_left_wrist_palm_order": order, "surface_order_ray_intersection_fraction": float(valid.mean()),
        "surface_contact_total": configuration["penetration_weight"] * penetration
            + configuration["surface_weight"] * attraction + configuration["far_order_weight"] * order}
    result["surface_order_reference_ray_intersection_fraction"] = float(np.asarray(reference_valid_rows).mean())
    result["surface_order_current_ray_intersection_fraction"] = float(np.asarray(current_valid_rows).mean())
    return result


def surface_patch_diagnostics(vertices, hands, candidate, vertex_indices, patch_positions,
                              weights, closest_points, summary, sample_frames, configuration, reference_hands=None):
    """Independent native patch geometry, without assigning hidden-contact truth.

    Signed maximum supporting-plane values are exact interior clearances, but
    are only a surrogate for Euclidean distance outside a convex object. Exact
    triangle queries are separately retained on predefined source-review frames.
    """
    indices = np.asarray(vertex_indices)
    patches = np.asarray(patch_positions)
    weights = np.asarray(weights, float)
    if (indices.shape != (2, 128) or indices.dtype.kind not in "iu" or np.any(indices < 0)
            or np.any(indices >= 18439) or any(len(np.unique(row)) != 128 for row in indices)
            or patches.shape != (2, 16) or patches.dtype.kind not in "iu"
            or np.any((patches < 0) | (patches >= 128)) or any(len(np.unique(row)) != 16 for row in patches)
            or weights.shape != (len(hands), 2) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1))):
        raise ValueError("Native surface diagnostics require aligned fixed128/16 patch selections and phase weights")
    local, signed = selected_surface_signed(vertices, candidate, indices)
    summaries = {}
    samples = []
    for side, name in enumerate(("left", "right")):
        surface = signed[:, side, patches[side]]
        active = weights[:, side] > 0
        summaries[name] = {"selected_native_vertex_count": 128, "attraction_patch_count": 16,
            "active_phase_frames": int(active.sum()),
            "all128_signed_max_plane_mm": summary(signed[:, side] * 1000),
            "active128_signed_max_plane_mm": summary(signed[active, side] * 1000),
            "all16_absolute_signed_max_plane_mm": summary(np.abs(surface) * 1000),
            "active16_absolute_signed_max_plane_mm": summary(np.abs(surface[active]) * 1000),
            "phase_weighted_mean16_absolute_signed_max_plane_mm": float((np.abs(surface) * weights[:, side, None]).sum() / (16 * weights[:, side].sum()) * 1000) if weights[:, side].sum() else None}
        for frame in sample_frames:
            if type(frame) is not int or not 0 <= frame < len(hands):
                raise ValueError("Exact triangle audit samples must be original in-range frame IDs")
            points = local[frame, side, patches[side]]
            _, distances = closest_points(points, candidate["vertices"], candidate["faces"])
            samples.append({"original_frame_id": frame, "side": name,
                "object_pose_usable": bool(candidate["pose_usable"][frame]),
                "phase_weight": float(weights[frame, side]),
                "native_vertex_ids": indices[side, patches[side]].tolist(),
                "exact_triangle_distances_mm": (distances * 1000).tolist(),
                "absolute_max_plane_surrogate_mm": (np.abs(surface[frame]) * 1000).tolist()})
    return {"sides": summaries, "independently_reconstructed_surrogate_terms": surface_surrogate_terms(signed, hands, candidate, weights, patches, configuration, reference_hands),
        "predefined_exact_triangle_sample_rows": samples,
        "all_original_frame_rows": [{"original_frame_id": frame,
            "signed_max_plane_mm": (signed[frame] * 1000).tolist(),
            "phase_weights": weights[frame].tolist()} for frame in range(len(hands))],
        "interpretation": "The same generated native skin patches are measured for each human. Max-plane attraction is a surrogate outside; exact triangle samples are separate. Patch fit is not observed contact accuracy."}, signed


def summarize_pair(reference_role, candidate_role, hands, projected, points, confidence, heldout, threshold,
                   original_edges, windows, policy, refined):
    rows = policy.motion_rows(hands[reference_role], {"candidate": hands[candidate_role], "old_candidate": hands[reference_role]},
        points, confidence, projected["camera"], threshold, heldout, original_edges)
    return {"reference_role": reference_role, "candidate_role": candidate_role,
        "image_agreement": policy.image_agreement({"baseline": projected[reference_role], "candidate": projected[candidate_role],
            "old_candidate": projected[reference_role]}, points, confidence, heldout),
        "all_adjacent_original_frames": refined.image_reference_vocabulary(policy.summarize_motion(rows)),
        "manual_and_phase_windows": [{"annotation": window, "motion": refined.image_reference_vocabulary(policy.summarize_motion(
            [r for r in rows if window["first"] <= r["from_original_frame_id"] and r["to_original_frame_id"] <= window["last"]
             and ("side" not in window or window["side"] == r["side"])]))} for window in windows],
        "all_adjacent_original_frame_rows": refined.image_reference_vocabulary(rows),
        "interpretation": "Unchanged relative engineering guards; detector references are method-dependent, not true hidden motion"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model-batch", type=int, default=8)
    parser.add_argument("--hand-skin-output", type=Path)
    args = parser.parse_args(argv)
    skin_output = args.hand_skin_output or args.output.parent / "native-hand-skin-indices01.npz"
    if args.output.exists() or skin_output.exists() or args.threads < 1 or args.model_batch < 1:
        parser.error("Use a new output and positive CPU threads/model batch")
    old, policy, refined = helpers(args.tool_root)
    task = old.read(args.input_manifest)
    if task.get("track") != 1 or task.get("episode") != 16 or task.get("all30_execution_authorized") is not False:
        raise ValueError("Only the explicitly authorized Track 1 EP16 experiment is supported")
    required = (*ROLES, "shared_initialization", "source_video", "manifest", "object", "old_object", "object_report",
        "old_object_report", "phases", "criteria", "protocol", "legacy_reference", "fresh_reference", "original_active_hands",
        "control_prior_support", "fit_configuration_reference")
    paths = {name: old.bound(task[name], name) for name in required}
    manifest = old.read(paths["manifest"])
    if manifest.get("track") != 1 or manifest.get("episode") != 16 or manifest.get("video_frames") != 360:
        raise ValueError("Wrong source episode or complete original-frame coverage")
    if manifest["video_sha256"] != task["source_video"]["sha256"]:
        raise ValueError("Source video identity differs")
    count, expected = manifest["video_frames"], np.arange(manifest["video_frames"])
    intrinsics = np.asarray(manifest["camera_intrinsics"], np.float32)
    config = old.read(paths["fit_configuration_reference"])["config"]
    if config["holdout_stride"] != 5:
        raise ValueError("Original stride-five partition changed")
    heldout, threshold = expected % 5 == 0, config["min_confidence"]
    phases, criteria = old.read(paths["phases"]), old.read(paths["criteria"])
    validate_frozen_criteria(criteria)
    if (phases.get("track") != 1 or phases.get("episode") != 16 or phases.get("video_sha256") != manifest["video_sha256"]
            or phases.get("frame_count") != count or criteria.get("manual_phase_sha256") != task["phases"]["sha256"]):
        raise ValueError("Phases/criteria do not identify the same original source")
    windows = old.phase_windows(phases, count)
    annotation = old.read(paths["control_prior_support"])
    manual = manual_control_envelope(annotation, count, manifest["video_sha256"])
    source_review_binding = {"path": annotation["source_review_receipt"], "sha256": annotation["source_review_receipt_sha256"]}
    old.bound(source_review_binding, "manual source-only review receipt")
    transitions = boundary_windows(manual, criteria.get("support_boundary_radius_frames", 8))
    from v2hoi.hand_contact import closest_mesh_points, native_hand_vertex_indices, native_mesh_penetration_proxy, validate_object_candidate
    objects = {}
    for label, key in (("new", "object"), ("old", "old_object")):
        raw = old.load_npz(paths[key])
        report = old.read(paths[key + "_report"])
        if report.get("source_video_sha256", report.get("video_sha256")) != manifest["video_sha256"]:
            raise ValueError("Object hypothesis belongs to another source")
        if not np.array_equal(np.asarray(raw["camera_intrinsics"], np.float32), intrinsics):
            raise ValueError("Object and human cameras differ")
        objects[label] = validate_object_candidate(raw, expected)
    if not np.array_equal(objects["new"]["pose_usable"], objects["old"]["pose_usable"]):
        raise ValueError("Depth-gauge comparison changed the fixed reliability subset")
    points, confidence = {}, {}
    for role in ("legacy_reference", "fresh_reference"):
        evidence = old.load_npz(paths[role])
        if str(evidence["source_video_sha256"].item()) != manifest["video_sha256"]:
            raise ValueError("Image reference belongs to another source")
        points[role], confidence[role] = policy.validate_evidence(evidence, expected)
    arrays, reports = {}, {}
    array_names = ("support", "weights", "original_support", "control_prior_array", "arm_envelope", "image_envelope", "effective_envelope", "arm_limits", "wrist_limits")
    for role in ("a", "b"):
        reports[role] = old.read(old.bound(task[role]["report"], role + " report"))
        report = reports[role]
        if (report.get("source_video_sha256") != manifest["video_sha256"] or report.get("mhr_model_sha256") != manifest["mhr_model_sha256"]
                or report["config"] != config or report["baseline_sha256"] != task["shared_initialization"]["sha256"]
                or report["observations_sha256"] != task["fresh_reference"]["sha256"]
                or report["heldout_frame_indices"] != expected[heldout].tolist()):
            raise ValueError("Controlled configuration/initialization/evidence differs")
        provenance = report.get("control_prior_provenance", {})
        if provenance.get("sha256") != task["control_prior_support"]["sha256"]:
            raise ValueError("Fitter did not bind the same prior-only eligibility annotation")
        arrays[role] = {name: np.load(old.bound(task[role][name], role + " " + name), allow_pickle=False) for name in array_names}
    old_arrays = {name: np.load(paths["old_a"].parent / filename, allow_pickle=False) for name, filename in (
        ("original_support", "original_active_hands.npy"), ("weights", "optimization_weights.npy"),
        ("effective_envelope", "support_transition_envelope.npy"))}
    historical_files = {name: paths["old_a"].parent / filename for name, filename in (
        ("original_support", "original_active_hands.npy"), ("weights", "optimization_weights.npy"),
        ("effective_envelope", "support_transition_envelope.npy"))}
    historical_hashes = {name: old.digest(path) for name, path in historical_files.items()}
    matched = validate_prior_arrays(arrays, manual, old_arrays, confidence["fresh_reference"], heldout, threshold, old,
        expected_arm=image_arm_envelope(confidence["fresh_reference"], heldout, config))
    expected_arm_limits, expected_wrist_limits = declared_control_limits(manual, config)
    for role in ("a", "b"):
        if (not np.allclose(arrays[role]["arm_limits"], expected_arm_limits, atol=1e-7, rtol=0)
                or not np.allclose(arrays[role]["wrist_limits"], expected_wrist_limits, atol=1e-7, rtol=0)):
            raise ValueError("Saved per-control bounds differ from the frozen manual/config interpolation")
    if not np.array_equal(arrays["a"]["arm_limits"], arrays["b"]["arm_limits"]) or not np.array_equal(arrays["a"]["wrist_limits"], arrays["b"]["wrist_limits"]):
        raise ValueError("A/B angular and wrist bounds differ")
    if reports["a"].get("contact_provenance") is not None:
        raise ValueError("A must not contain contact losses")
    contact = reports["b"].get("contact_provenance")
    surface = reports["b"].get("surface_contact")
    if reports["a"].get("surface_contact") is not None or reports["a"].get("surface_contact_provenance") is not None:
        raise ValueError("A must not contain native surface contact losses")
    surface_configuration = None
    if surface is not None:
        if contact is not None or surface.get("method") != "native_hand_surface_contact.v1":
            raise ValueError("B must explicitly identify exactly one supported contact method")
        surface_binding = task["surface_contact_configuration"]
        surface_path = old.bound(surface_binding, "frozen native surface-contact configuration")
        surface_configuration = old.read(surface_path)
        provenance = reports["b"].get("surface_contact_provenance", {})
        if (provenance.get("sha256") != surface_binding["sha256"]
                or provenance.get("object_sha256") != task["object"]["sha256"]
                or provenance.get("phases_sha256") != task["phases"]["sha256"]
                or provenance.get("baseline_sha256") != task["shared_initialization"]["sha256"]
                or surface.get("configuration") != surface_configuration):
            raise ValueError("Native surface contact configuration/provenance differs from the frozen pair")
        anchor_reference = {"used": False, "role": "Native skin surface intervention has no fixed fingertip anchors"}
    else:
        if not contact or contact.get("object_sha256") != task["object"]["sha256"] or contact.get("phases_sha256") != task["phases"]["sha256"] or contact.get("scale_factor") != 1:
            raise ValueError("B contact differs from the frozen object/phases/baked-scale contract")
        anchor_reference = old.validate_anchor_reference(task, contact, old.load_npz(paths["shared_initialization"]), manifest)
    model_path = args.model or Path(manifest["mhr_model"])
    if old.digest(model_path) != manifest["mhr_model_sha256"]:
        raise ValueError("Native decoder identity differs")
    import torch
    torch.set_num_threads(args.threads)
    model = torch.jit.load(str(model_path), map_location="cpu").float().eval()
    parameters, hands, parity, hashes = {}, {}, {}, {}
    review_names = ("vertices_m", "joints_3d", "faces", "frame_indices", "camera_intrinsics", "fps")
    decode_roles = list(ROLES)
    if paths["shared_initialization"] not in [paths[role] for role in ROLES]:
        decode_roles.append("shared_initialization")
    for role in decode_roles:
        print("Fresh complete native decode: " + role, flush=True)
        parameters[role], hands[role], parity[role] = policy.decode_variant(paths[role].parent, manifest, model, manifest["mhr_model_sha256"], device="cpu", batch=args.model_batch)
        hashes[role] = {name: old.digest(paths[role].parent / "review_arrays" / (name + ".npy")) for name in review_names}
    if "shared_initialization" not in parameters:
        source = next(role for role in ROLES if paths[role] == paths["shared_initialization"])
        parameters["shared_initialization"], hands["shared_initialization"] = parameters[source], hands[source]
    topology = hashes["baseline"]["faces"]
    if any(value["faces"] != topology for value in hashes.values()):
        raise ValueError("Native review topology differs across variants")
    preservation, updates = {}, {}
    for role in ("old_a", "old_b", "a", "b", "shared_initialization"):
        if not np.array_equal(parameters[role]["fps"], parameters["baseline"]["fps"]):
            raise ValueError("Native trajectory frame rate changed")
        preservation[role] = policy.parameter_preservation(parameters["baseline"], parameters[role], config["optimize_wrist_chain"])
        for field in ("fixed_identity_unchanged", "fixed_skeleton_scales_unchanged", "root_controls_unchanged", "all_unselected_controls_unchanged", "finite_parameters"):
            if not preservation[role][field]:
                raise ValueError("Native fixed-parameter contract failed: " + role + "/" + field)
        updates[role] = {}
        for reference in ("baseline", "old_a", "shared_initialization"):
            updates[role]["relative_to_" + reference] = update_diagnostics(parameters[reference]["pose"], parameters[role]["pose"], hands[reference], hands[role], policy.summary, windows, camera=intrinsics)
    for role in ("a", "b"):
        updates[role]["ramped_update_contract"] = update_diagnostics(parameters["shared_initialization"]["pose"], parameters[role]["pose"], hands["shared_initialization"], hands[role], policy.summary, windows,
            envelope=arrays[role]["effective_envelope"], arm_envelope=arrays[role]["arm_envelope"], config=config,
            arm_limits=arrays[role]["arm_limits"], wrist_limits=arrays[role]["wrist_limits"], camera=intrinsics, manual=manual)
    projected = {role: policy.project_hands(hands[role], intrinsics) for role in ROLES}
    projected["camera"] = intrinsics
    original_edges = set(policy.support_edges(np.load(paths["original_active_hands"], allow_pickle=False)))
    image = {reference: refined.image_distances({role: projected[role] for role in ROLES}, points[reference], confidence[reference], heldout) for reference in points}
    direct = {}
    for reference in points:
        direct[reference] = {before + "_to_" + after: summarize_pair(before, after, hands, projected, points[reference], confidence[reference], heldout, threshold,
            original_edges, [*windows, *transitions], policy, refined) for before, after in PAIRS}
    selection = native_hand_vertex_indices(model)
    anchor_directory = paths["b"].parent
    targets = anchor_weights = None
    contact_files = {}
    if surface is not None:
        selection_file = old.bound(task["b"]["surface_selection"], "native surface selection")
        phase_file = old.bound(task["b"]["surface_weights"], "native surface phase weights")
        contact_files = {"surface_selection": selection_file, "surface_weights": phase_file}
        phase_weights = np.load(phase_file, allow_pickle=False)
        if not np.array_equal(phase_weights, np.asarray(surface["phase_weights"], np.float32)):
            raise ValueError("Saved surface phase weights differ from the actual intervention report")
        from v2hoi.hand_contact import phase_weights as declared_phase_weights
        expected_phase = declared_phase_weights(phases, count) * objects["new"]["pose_usable"][:, None]
        if not np.array_equal(phase_weights, expected_phase):
            raise ValueError("Surface phase weights differ from manual phases and frozen object reliability")
        sampled_ids = np.asarray(surface["native_hand_vertex_ids"], np.int64)
        patch_ids = np.asarray(surface["native_patch_vertex_ids"], np.int64)
        with torch.inference_mode():
            native_shape = torch.as_tensor(np.array(parameters["shared_initialization"]["shape"], np.float32, copy=True))
            rest = model.character_torch.blend_shape(native_shape[None])[0].cpu().numpy()
        independently_selected = np.asarray([distributed_native_ids(rest, group) for group in selection])
        if not np.array_equal(sampled_ids, independently_selected):
            raise ValueError("Surface intervention did not use deterministic fixed-identity distributed native skin")
        patch_positions = np.asarray([[int(np.flatnonzero(sampled_ids[side] == vertex)[0]) for vertex in patch_ids[side]] for side in range(2)])
        saved_selection = old.load_npz(selection_file)
        if (not np.array_equal(saved_selection["hand_vertex_ids"], sampled_ids)
                or not np.array_equal(saved_selection["patch_vertex_ids"], patch_ids)):
            raise ValueError("Saved native surface vertex/patch selections differ from the fitted intervention")
        seed_vertices = np.load(paths["shared_initialization"].parent / "review_arrays/vertices_m.npy", mmap_mode="r", allow_pickle=False)
        _, seed_signed = selected_surface_signed(seed_vertices, objects["new"], sampled_ids)
        independent_patch_positions = select_patch_positions(seed_signed, phase_weights, expected)
        for side in range(2):
            training = (expected % 5 != 0) & (phase_weights[:, side] > 0)
            median = np.median(np.abs(seed_signed[training, side]), axis=0)
            if np.any(median[patch_positions[side]] > median[independent_patch_positions[side]].max() + 1e-6):
                raise ValueError("Surface patches are not lowest shared-seed training-only median-clearance vertices")
        surface_selection_contract = {"distributed128_exactly_reconstructed": True,
            "median16_selection_within_float_roundoff": True,
            "median16_exact_index_match": bool(np.array_equal(independent_patch_positions, patch_positions)),
            "median_tie_roundoff_tolerance_metres": 1e-6,
            "native_hand_vertex_ids": sampled_ids.tolist(), "native_patch_vertex_ids": patch_ids.tolist()}
    else:
        targets = np.load(anchor_directory / "contact_anchor_targets_m.npy", allow_pickle=False)
        anchor_weights = np.load(anchor_directory / "contact_anchor_weights.npy", allow_pickle=False)
        phase_weights = np.load(anchor_directory / "contact_phase_weights.npy", allow_pickle=False)
    allowed = np.zeros((count, 2), bool)
    for window in windows:
        if window["anchor_weight"] > 0:
            for name in window["eligible_sides"]:
                allowed[window["first"]:window["last"] + 1, ("left", "right").index(name)] = True
    if (phase_weights.shape != allowed.shape or not np.isfinite(phase_weights).all() or np.any((phase_weights < 0) | (phase_weights > 1))
            or np.any(phase_weights[~allowed] != 0) or (anchor_weights is not None and np.any(anchor_weights[~allowed] != 0))):
        raise ValueError("Contact attraction exists outside admitted phases")
    geometry, ordering, surface_geometry = {}, {}, {}
    source_review = old.read(old.bound(criteria["independent_source_review"], "pre-fit independent source review"))
    sample_frames = source_review["actual_source_frame_ids_represented"]
    for object_role, candidate in objects.items():
        geometry[object_role], ordering[object_role] = {}, {}
        for role in ROLES:
            print("Full skin / ray ordering: " + object_role + "/" + role, flush=True)
            vertices = np.load(paths[role].parent / "review_arrays/vertices_m.npy", mmap_mode="r", allow_pickle=False)
            skin = native_mesh_penetration_proxy(vertices, selection, candidate, np.ones((count, 2)))
            geometry[object_role][role] = old.geometry_diagnostics(hands[role], candidate, windows, skin,
                targets if object_role == "new" else None, anchor_weights if object_role == "new" else None, closest_mesh_points, policy.summary)
            geometry[object_role][role]["same_frozen_anchor_distance_by_side"] = anchor_consistency_by_side(hands[role], candidate,
                targets if object_role == "new" else None, anchor_weights if object_role == "new" else None, old.to_object)
            ordering[object_role][role] = ordering_diagnostics(hands[role], candidate, policy.summary)
            if surface is not None and object_role == "new":
                surface_geometry[role], _ = surface_patch_diagnostics(vertices, hands[role], candidate, sampled_ids, patch_positions,
                    phase_weights, closest_mesh_points, policy.summary, sample_frames, surface_configuration, hands["shared_initialization"])
    skin_guards = {object_role: {before + "_to_" + after: old.skin_nonregression(values[before], values[after], criteria)
        for before, after in PAIRS} for object_role, values in geometry.items()}
    result = {"schema_version": "ep16_occluded_depth_contact_ab_evaluation.v1", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "track": 1, "episode": 16, "frames": count, "original_frame_coverage_complete": True,
        "source_video_sha256": manifest["video_sha256"], "input_manifest_sha256": old.digest(args.input_manifest),
        "input_bindings": task, "evaluation_source_sha256": old.digest(Path(__file__)),
        "helper_source_sha256": {name: old.digest(args.tool_root / "tools" / (name + ".py")) for name in ("evaluate_contact_ab", "evaluate_human_hand_pilots", "evaluate_refined_egohand_pilots")},
        "mhr_model_sha256": manifest["mhr_model_sha256"], "frozen_engineering_criteria": criteria,
        "controlled_pair": ["a", "b"], "native_decoder": parity, "parameter_preservation": preservation,
        "native_review_array_sha256": hashes, "retained_native_updates": updates, "matched_controls": matched,
        "anchor_reference_contract": anchor_reference, "same_original_camera_and_fixed_object_for_both": True,
        "manual_phase_windows": windows, "manual_eligibility_boundary_windows": transitions,
        "contact_attraction_outside_admitted_phase_zero": True, "hand_skin_vertex_counts": [len(x) for x in selection],
        "image_reference_distances": image, "direct_pair_engineering_checks": direct,
        "native_opening_proxy_curves": {role: {"projected_2d": refined.opening_curves(projected[role]), "native_3d": refined.opening_curves(hands[role], units="native_estimated_metres")} for role in ROLES},
        "object_hypothesis_cross_geometry": geometry, "coarse_ray_ordering_diagnostics": ordering,
        "direct_pair_hand_skin_nonregression": skin_guards,
        "object_contact_diagnostics": geometry["new"], "direct_b_versus_a_hand_skin_nonregression": skin_guards["new"]["a_to_b"],
        "geometry_interpretation": "Every human is measured under both frozen object hypotheses. Gauge changes preserve monocular ambiguity; joint-ray order, contacts and skin depths are conditional diagnostics, not physical truth or official scores.",
        "quality_accepted": False, "contact_quality_accepted": False, "scale_accepted": False,
        "official_score_available": False, "automatic_promotion": False, "all30_execution_authorized": False,
        "contact_model_training_started": False, "episode11_execution_started": False}
    if surface is not None:
        reconstructed = surface_geometry["b"]["independently_reconstructed_surrogate_terms"]
        declared = surface["terms"]
        matched_terms = {name: {"independent": value, "fitter": declared[name],
            "absolute_difference": abs(value - declared[name]),
            "passed": bool(np.isclose(value, declared[name], atol=2e-4, rtol=2e-4))} for name, value in reconstructed.items()}
        if not all(row["passed"] for row in matched_terms.values()):
            raise ValueError("Independent native surface intervention terms do not match the actual fitter: " + str(matched_terms))
        result["surface_contact_intervention"] = {"method": surface["method"], "selection_contract": surface_selection_contract,
            "independent_term_parity": matched_terms, "native_patch_geometry": surface_geometry,
            "interpretation": "Actual native surface intervention reconstructed independently; agreement proves implementation consistency, not real grasp accuracy"}
    for name, path in historical_files.items():
        if old.digest(path) != historical_hashes[name]:
            raise ValueError("Historical image controls changed during evaluation")
    result["historical_image_control_sha256"] = historical_hashes
    for role, names in hashes.items():
        for name, sha in names.items():
            if old.digest(paths[role].parent / "review_arrays" / (name + ".npy")) != sha:
                raise ValueError("Review arrays changed during evaluation")
    for name in required:
        old.bound(task[name], "post-evaluation " + name)
    old.bound(source_review_binding, "post-evaluation manual source-only review receipt")
    for role in ("a", "b"):
        for name in ("report", *array_names):
            old.bound(task[role][name], "post-evaluation " + role + "/" + name)
    for name in contact_files:
        old.bound(task["b"][name], "post-evaluation " + name)
    if surface is not None:
        old.bound(task["surface_contact_configuration"], "post-evaluation native surface configuration")
    skin_output.parent.mkdir(parents=True, exist_ok=True)
    with skin_output.open("xb") as stream:
        np.savez(stream, left=np.asarray(selection[0], np.int64), right=np.asarray(selection[1], np.int64))
    result["native_hand_skin_selection"] = {"path": str(skin_output), "sha256": old.digest(skin_output),
        "mhr_model_sha256": manifest["mhr_model_sha256"], "minimum_descendant_blendweight": .5,
        "counts": [len(x) for x in selection],
        "interpretation": "Native rig-derived hand-skin vertex indices for diagnostics/display, not observed skin or hidden visibility truth"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "frames": count, "quality_accepted": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
