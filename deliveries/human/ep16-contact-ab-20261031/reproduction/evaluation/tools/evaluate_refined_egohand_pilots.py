"""Compare explicitly bound Track 1 native variants with separate evidence roles.

Only EP11/EP16 development pilots are supported. Historical comparisons retain
their own crop policy. The fresh controlled pair must share its new source crop
bounds, full fit configuration, original stride-five split and native decoder.
No generated target agreement, low admission rate or zero update accepts quality.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MIN_NORMALIZED_PALM_PIXELS = 2.0
MIN_NORMALIZED_PALM_METRES = 0.001


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_policy(tool_root):
    source = tool_root / "tools/evaluate_human_hand_pilots.py"
    spec = importlib.util.spec_from_file_location("refined_pilot_frozen_policy", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bound(binding, policy, label):
    path = Path(binding["path"])
    if policy.digest(path) != binding["sha256"]:
        raise ValueError(f"Changed {label} binding: {path}")
    return path


def role_name(role):
    if not isinstance(role, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", role):
        raise ValueError("Roles must be explicit lowercase identifiers")
    return role


def image_reference_vocabulary(value):
    """The numerical policy's inherited wording must not imply independence."""
    if isinstance(value, dict):
        return {key.replace("independently_observed", "accepted_image_reference")
                   .replace("independent_observed", "accepted_image_reference"):
                image_reference_vocabulary(item) for key, item in value.items()}
    if isinstance(value, list):
        return [image_reference_vocabulary(item) for item in value]
    return value


def summarize_multi_motion(rows, labels, policy):
    """Reuse the original three-role reductions without changing their maths."""
    result = {"edge_count": len(rows), "variants": {},
              "absolute_wrist_motion": {"edge_count": len(rows), "variants": {}}}
    for label in labels:
        proxy = [{**row,
                  "variants": {"baseline": row["variants"]["baseline"],
                               "old_candidate": row["variants"][label],
                               "candidate": row["variants"][label]},
                  "wrist_variants": {"baseline": row["wrist_variants"]["baseline"],
                                     "old_candidate": row["wrist_variants"][label],
                                     "candidate": row["wrist_variants"][label]}}
                 for row in rows]
        reduced = policy.summarize_motion(proxy)
        result["variants"][label] = reduced["variants"]["candidate"]
        result["absolute_wrist_motion"]["variants"][label] = reduced["absolute_wrist_motion"]["variants"]["candidate"]
    result["absolute_wrist_motion"]["interpretation"] = (
        "Unchanged separate wrist engineering budgets; not official wrist scoring or verified physical error")
    return result


def image_distances(projected, points, confidence, heldout, generated=False):
    """Report distances only: an own-target fit never creates an acceptance flag."""
    output = {}
    for split, mask in (("all_original_frames", np.ones(len(points), bool)),
                        ("original_development_holdout", heldout),
                        ("original_development_training", ~heldout)):
        output[split] = {}
        for side, name in enumerate(("left", "right")):
            weights = confidence[:, side] * mask[:, None]
            relative_weights = np.minimum(weights[:, 1:], weights[:, :1])
            row = {"admitted_frames": int((weights[:, 1:].sum(-1) > 0).sum()),
                   "admitted_landmarks": int((weights > 0).sum()), "variants": {}}
            for role, prediction in projected.items():
                absolute = np.linalg.norm(prediction[:, side] - points[:, side], axis=-1)
                relative = np.linalg.norm((prediction[:, side, 1:] - prediction[:, side, :1])
                                          - (points[:, side, 1:] - points[:, side, :1]), axis=-1)
                row["variants"][role] = {
                    "weighted_reprojection_pixels": float((absolute * weights).sum() / weights.sum()) if weights.sum() else None,
                    "weighted_wrist_relative_reprojection_pixels": float((relative * relative_weights).sum() / relative_weights.sum()) if relative_weights.sum() else None}
            row["interpretation"] = ("Generated-target fit consistency, not independent accuracy or acceptance" if generated
                                     else "Method-dependent image-reference distance, not ground truth or acceptance")
            output[split][name] = row
    return output


def opening_curves(points, confidence=None, threshold=0.05, units="pixels"):
    """Opening proxies retain raw distances and explicitly suppress tiny denominators."""
    limit = MIN_NORMALIZED_PALM_PIXELS if units == "pixels" else MIN_NORMALIZED_PALM_METRES
    output = {"units": units, "minimum_normalization_palm_width": limit, "sides": {},
              "interpretation": "Thumb-tip distances and projected/native palm-width ratios are proxies, not joint angles, grasp labels or contact truth"}
    for side, name in enumerate(("left", "right")):
        palm = np.linalg.norm(points[:, side, 5] - points[:, side, 17], axis=-1)
        index = np.linalg.norm(points[:, side, 4] - points[:, side, 8], axis=-1)
        middle = np.linalg.norm(points[:, side, 4] - points[:, side, 12], axis=-1)
        valid = np.isfinite(points[:, side]).all((-1, -2))
        if confidence is not None:
            valid &= (confidence[:, side, [4, 5, 8, 12, 17]] >= threshold).all(-1)
        normalized_valid = valid & (palm >= limit)
        rows = []
        for frame in range(len(points)):
            rows.append({"original_frame_id": frame, "admitted": bool(valid[frame]),
                         "palm_width": float(palm[frame]) if valid[frame] else None,
                         "thumb_index_distance": float(index[frame]) if valid[frame] else None,
                         "thumb_middle_distance": float(middle[frame]) if valid[frame] else None,
                         "normalized_ratio_valid": bool(normalized_valid[frame]),
                         "thumb_index_over_palm": float(index[frame] / palm[frame]) if normalized_valid[frame] else None,
                         "thumb_middle_over_palm": float(middle[frame] / palm[frame]) if normalized_valid[frame] else None})
        output["sides"][name] = {"admitted_frames": int(valid.sum()),
                                 "normalization_valid_frames": int(normalized_valid.sum()),
                                 "tiny_palm_denominator_frames": int((valid & ~normalized_valid).sum()),
                                 "rows": rows}
    return output


def retained_update(parameters, baseline, policy):
    delta = np.asarray(parameters["pose"], float) - np.asarray(baseline["pose"], float)
    fingers = delta[:, 68:122]
    arms = delta[:, policy.ARM_CONTROLS]
    frame_maximum = np.abs(np.concatenate((fingers, arms), axis=1)).max(-1)
    changed = frame_maximum > 1e-7
    return {"finger_absolute_delta_radians": policy.summary(np.abs(fingers)),
            "arm_absolute_delta_radians": policy.summary(np.abs(arms)),
            "per_frame_maximum_absolute_delta_radians": policy.summary(frame_maximum),
            "changed_original_frame_count": int(changed.sum()),
            "changed_original_frame_ids": np.flatnonzero(changed).tolist(),
            "retained_zero_update": bool(not changed.any()),
            "interpretation": "A zero update is abstention or an optimization stall; fewer admitted targets or lower self-motion alone do not establish improvement"}


def admission_counts(confidence, heldout, threshold, support=None, weights=None):
    output = {}
    for side, name in enumerate(("left", "right")):
        output[name] = {}
        for label, mask in (("all_original_frames", np.ones(len(confidence), bool)),
                            ("training", ~heldout), ("holdout", heldout)):
            accepted = confidence[mask, side] >= threshold
            row = {"admitted_landmarks": int(accepted.sum()),
                   "admitted_wrist_frames": int(accepted[:, 0].sum()),
                   "frames_with_any_admitted_finger": int(accepted[:, 1:].any(-1).sum()),
                   "frames_with_eight_admitted_fingers": int((accepted[:, 1:].sum(-1) >= 8).sum())}
            if support is not None:
                row["active_hand_frames"] = int(support[mask, side].sum())
            if weights is not None:
                row["positive_optimization_landmarks"] = int((weights[mask, side] > 0).sum())
            output[name][label] = row
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=(11, 16), required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--review-windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model-batch", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Preserve existing evaluation; output must be new")
    if min(args.threads, args.model_batch) < 1:
        parser.error("Positive thread and model batch counts required")
    policy = load_policy(args.tool_root)
    task = read(args.input_manifest)
    if task.get("track") != 1 or task.get("all30_execution_authorized") is not False:
        raise ValueError("Wrong task scope")
    authorized = task.get("authorized_episodes", [entry["episode"] for entry in task["episodes"]])
    if args.episode not in authorized or not set(authorized) <= {11, 16}:
        raise ValueError("Only the two authorized pilots are supported")
    matches = [entry for entry in task["episodes"] if entry["episode"] == args.episode]
    if len(matches) != 1:
        raise ValueError("Each source episode must appear exactly once")
    item = matches[0]
    manifest_path = bound({"path": item["manifest"], "sha256": item["manifest_sha256"]}, policy, "original manifest")
    manifest = read(manifest_path)
    if manifest.get("track") != 1 or manifest.get("episode") != args.episode:
        raise ValueError("Wrong original episode manifest")
    expected = np.arange(manifest["video_frames"])
    if item.get("frames", len(expected)) != len(expected) or item["video_sha256"] != manifest["video_sha256"]:
        raise ValueError("Source count or identity differs")
    bound({"path": item["video"], "sha256": item["video_sha256"]}, policy, "source video")
    intrinsics = np.asarray(manifest["camera_intrinsics"], dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("Invalid original camera")
    baseline_path = bound(item["baseline"], policy, "original SAM native parameters")
    baseline_hash = item["baseline"]["sha256"]
    model_path = args.model or Path(manifest["mhr_model"])
    model_hash = policy.digest(model_path)
    if model_hash != manifest["mhr_model_sha256"]:
        raise ValueError("Original native decoder identity changed")
    reference_path = bound(item["fit_configuration_reference"], policy, "fit configuration reference")
    config = read(reference_path)["config"]
    if config["holdout_stride"] != 5:
        raise ValueError("Original stride-five development split changed")
    heldout_ids = expected[::5].tolist()
    heldout = np.isin(expected, heldout_ids)
    threshold = config["min_confidence"]
    controlled = item["controlled_pair"]
    if len(controlled) != 2 or len(set(controlled)) != 2 or "baseline" in controlled:
        raise ValueError("A distinct fresh image-detector / generated-proposal pair is required")
    variants = {"baseline": {"role": "baseline", "label": "Original fixed-identity SAM MHR",
                              "path": str(baseline_path), "sha256": baseline_hash}}
    for variant in item["variants"]:
        role = role_name(variant["role"])
        if role in variants:
            raise ValueError("Duplicate native variant role")
        variants[role] = variant
    if not set(controlled) <= set(variants):
        raise ValueError("Controlled native roles are missing")
    evidence, points, confidence, evidence_paths = {}, {}, {}, {}
    for name, binding in item["evidence"].items():
        role_name(name)
        if binding["kind"] not in ("image_detector", "generated_proposal"):
            raise ValueError("Evidence must distinguish image detector from generated proposal")
        path = bound(binding, policy, f"{name} evidence")
        evidence_paths[name] = path
        evidence[name] = load_npz(path)
        if str(evidence[name]["source_video_sha256"].item()) != manifest["video_sha256"]:
            raise ValueError("Evidence source-video identity changed")
        points[name], confidence[name] = policy.validate_evidence(evidence[name], expected)
        boxes = np.asarray(evidence[name]["crop_boxes"])
        if boxes.shape != (len(expected), 2, 4) or not np.isfinite(boxes).all():
            raise ValueError("Invalid source crop bounds")
        if "report" in binding:
            bound(binding["report"], policy, f"{name} evidence report")
    fresh_evidence = [variants[role]["evidence_role"] for role in controlled]
    if len(set(fresh_evidence)) != 2 or not set(fresh_evidence) <= set(evidence):
        raise ValueError("Controlled pair needs two explicit evidence roles")
    if [item["evidence"][name]["kind"] for name in fresh_evidence] != ["image_detector", "generated_proposal"]:
        raise ValueError("Controlled roles must order fresh MP then revised generated proposal")
    if not np.array_equal(evidence[fresh_evidence[0]]["crop_boxes"], evidence[fresh_evidence[1]]["crop_boxes"]):
        raise ValueError("Controlled fresh source crop bounds differ")
    if not item["evidence"][fresh_evidence[0]].get("shared_inputs") or not item["evidence"][fresh_evidence[1]].get("shared_inputs"):
        raise ValueError("Shared MP wrist anchoring/gating inputs must be disclosed")
    reports, support, weights = {}, {}, {}
    for role, variant in variants.items():
        path = bound(variant, policy, f"{role} native parameters")
        if role == "baseline":
            continue
        report_path = bound({"path": str(path.parent / "report.json"), "sha256": variant["report_sha256"]}, policy, f"{role} native report")
        reports[role] = read(report_path)
        report = reports[role]
        if report.get("source_video_sha256") != manifest["video_sha256"] or report.get("baseline_sha256") != baseline_hash:
            raise ValueError("Candidate source or baseline provenance differs")
        if report.get("mhr_model_sha256") != model_hash:
            raise ValueError("Candidate native decoder differs")
        if role in controlled:
            evidence_role = variant["evidence_role"]
            if report.get("observations_sha256") != item["evidence"][evidence_role]["sha256"]:
                raise ValueError("Controlled observation binding differs")
            if report.get("config") != config:
                raise ValueError("Controlled fit configuration differs from preserved reference")
            if report.get("heldout_frame_indices") != heldout_ids:
                raise ValueError("Original stride-five heldout frame IDs changed")
            support[role] = np.load(bound(variant["support"], policy, f"{role} support"), allow_pickle=False)
            if support[role].shape != (len(expected), 2):
                raise ValueError("Invalid controlled support dimensions")
            policy.support_edges(support[role])
            weights[role] = np.load(bound(variant["weights"], policy, f"{role} optimization weights"), allow_pickle=False)
            value = weights[role]
            if value.shape != confidence[evidence_role].shape or not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError("Invalid controlled optimization weights")
            if np.any(value[heldout] > 0) or np.any((value > 0) & (confidence[evidence_role] < threshold)):
                raise ValueError("Unsupported or heldout image target used")
    original_support = np.load(bound(item["original_bindings"]["old_active_hands"], policy, "original legacy support"), allow_pickle=False)
    if original_support.shape != (len(expected), 2):
        raise ValueError("Invalid original legacy support dimensions")
    original_edges = set(policy.support_edges(original_support))
    edges = {role: set(policy.support_edges(value)) for role, value in support.items()}
    shared_exclusion = {}
    proposal = evidence[fresh_evidence[1]]
    for name in ("shared_wrist_anchor_weights", "shared_visibility_gate_weights"):
        if name not in proposal:
            shared_exclusion[name] = {"verified": False, "reason": "No numeric provenance array supplied"}
            continue
        value = np.asarray(proposal[name])
        if value.shape != (len(expected), 2) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("Invalid shared-input provenance weights")
        if np.any(value[heldout] > 0):
            raise ValueError("Heldout fresh MP used for shared wrist anchor or gate")
        shared_exclusion[name] = {"verified": True, "positive_training_hand_frames": int((value > 0).sum()),
                                  "positive_holdout_hand_frames": 0}
    windows = read(args.review_windows)
    if windows.get("track") != 1 or windows.get("episode") != args.episode or windows["source_video_sha256"] != manifest["video_sha256"]:
        raise ValueError("Source inspection windows identify a different episode")
    # Validate windows before loading any expensive model or native trajectory.
    for window in windows["windows"]:
        first, last = window["first_original_frame_id"], window["last_original_frame_id"]
        if type(first) is not int or type(last) is not int or not 0 <= first < last < len(expected) or window["side"] not in ("left", "right", "both"):
            raise ValueError("Source inspection window must retain in-range original IDs")
    sys.path.insert(0, str(args.tool_root / "src"))
    import torch
    torch.set_num_threads(args.threads)
    model = torch.jit.load(str(model_path), map_location=args.device).float().eval()
    parameters, hands, parity = {}, {}, {}
    for role, variant in variants.items():
        print(f"Fresh all-original-frame native decode: {role}", flush=True)
        parameters[role], hands[role], parity[role] = policy.decode_variant(
            Path(variant["path"]).parent, manifest, model, model_hash, device=args.device, batch=args.model_batch)
    projected = {role: policy.project_hands(value, intrinsics) for role, value in hands.items()}
    labels = list(variants)
    motion, image, generated_consistency = {}, {}, {}
    for evidence_role, binding in item["evidence"].items():
        if binding["kind"] == "generated_proposal":
            generated_consistency[evidence_role] = {"interpretation": "Own generated targets; fit consistency only; no motion or acceptance guards",
                "distances": image_distances(projected, points[evidence_role], confidence[evidence_role], heldout, generated=True)}
            continue
        rows = policy.motion_rows(hands["baseline"], {role: value for role, value in hands.items() if role != "baseline"},
                                  points[evidence_role], confidence[evidence_role], intrinsics, threshold, heldout, original_edges)
        summaries = {"all_adjacent_original_frames": summarize_multi_motion(rows, labels, policy),
                     "original_legacy_support_boundaries": summarize_multi_motion([row for row in rows if row["original_support_boundary"]], labels, policy),
                     "support_boundaries_by_variant": {role: summarize_multi_motion([row for row in rows if
                         (row["from_original_frame_id"], ("left", "right").index(row["side"])) in boundaries], labels, policy)
                         for role, boundaries in edges.items()},
                     "holdout_touching_reference_edges": summarize_multi_motion([row for row in rows if row["touches_original_holdout"] and row["independent_observed_motion_sufficient"]], labels, policy),
                     "fast_reference_motion": summarize_multi_motion([row for row in rows if row["fast_observed_motion"]], labels, policy),
                     "manual_source_inspection_windows": [], "all_adjacent_frame_rows": rows}
        for window in windows["windows"]:
            selected = [row for row in rows if window["first_original_frame_id"] <= row["from_original_frame_id"]
                        and row["to_original_frame_id"] <= window["last_original_frame_id"] and window["side"] in ("both", row["side"])]
            summaries["manual_source_inspection_windows"].append({"annotation": window, "all_edges": summarize_multi_motion(selected, labels, policy)})
        motion[evidence_role] = {"interpretation": "Method-dependent image reference, not ground truth or independent verification of shared MP anchoring/gating",
                                "shared_inputs": binding.get("shared_inputs", []), **image_reference_vocabulary(summaries)}
        image[evidence_role] = image_distances(projected, points[evidence_role], confidence[evidence_role], heldout)
    self_motion, opening, preservation, updates, native_bindings = {}, {}, {}, {}, {}
    for role, joints in hands.items():
        directory = Path(variants[role]["path"]).parent
        native_files = [Path(variants[role]["path"]), directory / "report.json"]
        native_files += sorted((directory / "review_arrays").glob("*.npy"))
        native_bindings[role] = {"directory": str(directory), "files": {
            path.relative_to(directory).as_posix(): {"sha256": policy.digest(path), "bytes": path.stat().st_size}
            for path in native_files if path.exists()}}
        self_motion[role] = {}
        for side, name in enumerate(("left", "right")):
            relative_m = policy.relative(joints)[:, side]
            relative_px = policy.relative(projected[role])[:, side]
            self_motion[role][name] = {
                "wrist_relative_step_pixels": policy.summary(np.linalg.norm(np.diff(relative_px, axis=0), axis=-1)),
                "wrist_relative_second_difference_pixels_per_frame_squared": policy.summary(np.linalg.norm(np.diff(relative_px, n=2, axis=0), axis=-1)),
                "wrist_relative_step_native_estimated_metres": policy.summary(np.linalg.norm(np.diff(relative_m, axis=0), axis=-1)),
                "wrist_relative_second_difference_native_estimated_metres_per_frame_squared": policy.summary(np.linalg.norm(np.diff(relative_m, n=2, axis=0), axis=-1)),
                "interpretation": "Prediction-only motion, not reference-relative acceleration error; freezing can lower it"}
        opening[role] = {"projected_2d": opening_curves(projected[role]),
                         "native_3d": opening_curves(joints, units="native_estimated_metres")}
        if role != "baseline":
            preservation[role] = policy.parameter_preservation(parameters["baseline"], parameters[role],
                bool(reports[role]["config"]["optimize_wrist_chain"]))
            invariant_keys = ("fixed_identity_unchanged", "fixed_skeleton_scales_unchanged", "root_controls_unchanged",
                              "all_unselected_controls_unchanged", "finite_parameters")
            if not all(preservation[role][key] for key in invariant_keys):
                raise ValueError("Native preserved identity/root/nonselected parameter contract failed")
            updates[role] = retained_update(parameters[role], parameters["baseline"], policy)
            displacement = np.linalg.norm(joints[:, :, 0] - hands["baseline"][:, :, 0], axis=-1)
            updates[role]["absolute_native_estimated_wrist_displacement_metres"] = policy.summary(displacement)
            if role in controlled:
                checks = {
                    "finger_control_limit": updates[role]["finger_absolute_delta_radians"]["max"] <= config["max_delta_radians"] + 1e-6,
                    "arm_control_limit": updates[role]["arm_absolute_delta_radians"]["max"] <= config["arm_delta_limit_radians"] + 1e-6,
                    "wrist_displacement_limit": float(displacement.max()) <= config["wrist_max_displacement_metres"] + 1e-6}
                updates[role]["unchanged_native_budget_checks"] = checks
                if not all(checks.values()):
                    raise ValueError("Controlled native update exceeds frozen parameter or wrist budgets")
    admission = {role: admission_counts(confidence[variant["evidence_role"]], heldout, threshold, support[role], weights[role])
                 for role, variant in variants.items() if role in controlled}
    crop_comparison = {name: {"identical_to_fresh_pair": bool(np.array_equal(value["crop_boxes"], evidence[fresh_evidence[0]]["crop_boxes"]))}
                       for name, value in evidence.items()}
    result = {"schema_version": "refined_egohand_pilot_evaluation.v1", "track": 1, "episode": args.episode,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "frames": len(expected),
        "original_frame_coverage_complete": True, "source_video_sha256": manifest["video_sha256"],
        "mhr_model_sha256": model_hash, "evaluation_helper_sha256": policy.digest(Path(__file__)),
        "reused_unchanged_policy_source": {"path": str(args.tool_root / "tools/evaluate_human_hand_pilots.py"),
                                           "sha256": policy.digest(args.tool_root / "tools/evaluate_human_hand_pilots.py")},
        "unchanged_original_numerical_guards": policy.POLICY,
        "role_labels": {role: variant.get("label", role) for role, variant in variants.items()},
        "controlled_pair": controlled, "historical_context_roles": [role for role in variants if role not in controlled and role != "baseline"],
        "frozen_conditions": {"same_original_camera_and_decoder": True, "same_fresh_source_crop_bounds": True,
            "same_complete_fit_configuration_as_preserved_reference": True, "same_original_stride5_frame_ids": heldout_ids,
            "native_holdout_target_weights_zero": True, "positive_target_masks_identical": bool(np.array_equal(weights[controlled[0]] > 0, weights[controlled[1]] > 0)),
            "crop_comparison_by_evidence_role": crop_comparison,
            "model_specific_resize_normalization_and_raster_sampling_can_differ": True,
            "shared_input_stride5_exclusion": shared_exclusion,
            "full_clip_model_inference_and_smoothing_are_transductive": True,
            "development_validation_not_untouched_test": True},
        "input_bindings": {"task_manifest": {"path": str(args.input_manifest), "sha256": policy.digest(args.input_manifest)},
                           "review_windows": {"path": str(args.review_windows), "sha256": policy.digest(args.review_windows)},
                           "original_manifest": {"path": str(manifest_path), "sha256": policy.digest(manifest_path)},
                           "native_variants": variants, "evidence": item["evidence"], "fit_configuration_reference": item["fit_configuration_reference"]},
        "native_decoder": parity, "native_artifact_bindings": native_bindings,
        "parameter_preservation": preservation, "retained_native_updates": updates,
        "admission_and_optimization_counts": admission,
        "support_boundaries": {"original_legacy": [{"from_original_frame_id": frame, "to_original_frame_id": frame + 1, "side": ("left", "right")[side]} for frame, side in sorted(original_edges)],
                               **{role: [{"from_original_frame_id": frame, "to_original_frame_id": frame + 1, "side": ("left", "right")[side]} for frame, side in sorted(value)] for role, value in edges.items()}},
        "image_reference_distances": image, "image_reference_motion_screens": motion,
        "generated_target_fit_consistency": generated_consistency, "prediction_self_motion": self_motion,
        "native_opening_proxy_curves": opening,
        "image_reference_opening_proxy_curves": {name: opening_curves(points[name], confidence[name], threshold) for name, binding in item["evidence"].items() if binding["kind"] == "image_detector"},
        "automatic_candidate_promotion": False, "all30_execution_authorized": False,
        "quality_accepted": False, "contact_quality_accepted": False, "official_score_available": False,
        "limitations": ["All image references are method-dependent; shared wrist/gate evidence is not independent validation.",
            "Historical methods can have different input crop bounds; only the explicitly controlled fresh pair shares the new bounds.",
            "Generated targets are not observed recovery during occlusion; their agreement has no acceptance guards.",
            "Reduced admission, a zero retained update or lower prediction self-motion can conceal missing motion.",
            "Opening distances and palm ratios are proxies; tiny denominators are explicitly invalidated and no ratio-based ranking is made.",
            "Full-clip inference/filtering sees downstream-development-holdout frames and future frames.",
            "Native parity, complete coverage and engineering budgets do not establish hidden articulation, physical contact, scale or official scores.",
            "Source-visible alignment, grasp timing and amplitude require separately recorded inspection; full occlusion has no available truth."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "frames": len(expected), "quality_accepted": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
