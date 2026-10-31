"""Independently inspect source-bound EP16 no-contact/manual-contact native fits.

Only the contact loss differs between A and B. Image references, uncertain
object geometry and manual phases remain separate evidence roles. Contact
self-consistency, smoothness and decoder parity never accept reconstruction.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TIPS = np.asarray([4, 8, 12, 16, 20])


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_helpers(tool_root):
    out = []
    for name in ("evaluate_human_hand_pilots", "evaluate_refined_egohand_pilots"):
        spec = importlib.util.spec_from_file_location("contact_ab_" + name, tool_root / "tools" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out.append(module)
    return out


def bound(entry, label):
    path = Path(entry["path"])
    if digest(path) != entry["sha256"]:
        raise ValueError(f"Changed {label} bytes: {path}")
    return path


def phase_windows(phases, frames):
    """Retain zero-attraction release windows instead of silently dropping them."""
    covered = np.zeros(frames, dtype=np.uint8)
    out = []
    for row in phases["phase_intervals_inclusive"]:
        first, last = row["first"], row["last"]
        if type(first) is not int or type(last) is not int or not 0 <= first <= last < frames:
            raise ValueError("Manual phase must use original in-range integer IDs")
        if covered[first:last + 1].any():
            raise ValueError("Manual phase intervals overlap")
        weight = float(row.get("anchor_weight", 0))
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("Manual phase weight is invalid")
        covered[first:last + 1] += 1
        out.append({"label": row["phase"], "first": first, "last": last,
                    "anchor_weight": weight, "eligible_sides": row.get("eligible_sides", [])})
    if not np.all(covered == 1):
        raise ValueError("Manual phase annotations must partition every original frame")
    return out


def matched_numeric_arrays(a, b, names=("pose", "shape", "scales", "fps", "frame_indices")):
    return {key: bool(np.array_equal(a[key], b[key])) for key in names}


def validate_controlled_arrays(support, weights, confidence, heldout, threshold):
    for role in ("a", "b"):
        active, value = np.asarray(support[role]), np.asarray(weights[role])
        if active.dtype.kind != "b" or active.shape != confidence.shape[:2]:
            raise ValueError("Controlled support must be boolean [T,2]")
        if value.shape != confidence.shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("Controlled image weights must be finite nonnegative [T,2,21]")
        if np.any(value[heldout] > 0) or np.any((value > 0) & (confidence < threshold)):
            raise ValueError("Unsupported or heldout image targets were used")
    if not np.array_equal(support["a"], support["b"]) or not np.array_equal(weights["a"], weights["b"]):
        raise ValueError("A/B must retain identical support and exact image weights")
    return {"exact_active_hand_masks_identical": True, "exact_image_weights_identical": True,
            "heldout_image_weights_zero": True,
            "active_hand_frames": support["a"].sum(0).tolist(),
            "inactive_hand_frames": (~support["a"]).sum(0).tolist(),
            "interpretation": "No B-only hidden-hand degrees of freedom. Detector support is not true visibility."}


def inactive_finger_abstention(baseline, candidate, active):
    out = {}
    for side, (name, first, last) in enumerate((("left", 95, 122), ("right", 68, 95))):
        inactive = ~np.asarray(active)[:, side]
        unchanged = np.array_equal(np.asarray(baseline)[inactive, first:last], np.asarray(candidate)[inactive, first:last])
        if not unchanged:
            raise ValueError("Unsupported finger controls changed outside the frozen active mask")
        out[name] = {"inactive_frames": int(inactive.sum()), "inactive_finger_controls_exact_baseline": True}
    return out


def to_object(points, transform):
    points, transform = np.asarray(points), np.asarray(transform)
    if points.ndim < 3 or points.shape[0] != len(transform) or points.shape[-1] != 3 or transform.shape[1:] != (4, 4):
        raise ValueError("Object-local coordinates require aligned [T,...,3] points and [T,4,4] poses")
    if not np.isfinite(points).all() or not np.isfinite(transform).all():
        raise ValueError("Object-local coordinates must be finite")
    expand = (len(transform),) + (1,) * (points.ndim - 2) + (3,)
    return np.einsum("t...i,tij->t...j", points - transform[:, :3, 3].reshape(expand), transform[:, :3, :3])


def selected_summary(values, mask, summary):
    return summary(np.asarray(values)[np.asarray(mask, bool)])


def geometry_diagnostics(hands, object_candidate, windows, skin, anchor_targets, anchor_weights,
                         closest_points, summary):
    """Tip/skin diagnostics against one fixed uncertain object, including release."""
    count = len(hands)
    local = to_object(hands[:, :, TIPS], object_candidate["object_to_camera"])
    _, distances = closest_points(local.reshape(-1, 3), object_candidate["vertices"], object_candidate["faces"])
    distances = distances.reshape(count, 2, 5)
    relative_step = np.linalg.norm(np.diff(local, axis=0), axis=-1)
    phases = []
    usable = np.asarray(object_candidate["pose_usable"], bool)
    for window in [{"label": "whole_clip", "first": 0, "last": count - 1, "anchor_weight": None}, *windows]:
        mask = np.zeros(count, bool)
        mask[window["first"]:window["last"] + 1] = True
        edge = mask[:-1] & mask[1:]
        row = {"annotation": window, "sides": {}}
        for side, name in enumerate(("left", "right")):
            sampled = distances[:, side]
            row["sides"][name] = {
                "frames": int(mask.sum()), "object_pose_usable_frames": int((mask & usable).sum()),
                "tip_to_triangle_distance_native_estimated_mm": selected_summary(sampled * 1000, mask, summary),
                "nearest_of_five_tips_to_surface_native_estimated_mm": selected_summary(sampled.min(-1) * 1000, mask, summary),
                "object_local_tip_step_native_estimated_mm_per_frame": selected_summary(relative_step[:, side] * 1000, edge, summary),
                "skin_mean_depth_over_selected_hand_vertices_mm": selected_summary(skin["per_frame_mean_depth_m"][:, side] * 1000, mask, summary),
                "skin_maximum_vertex_depth_mm": selected_summary(skin["per_frame_max_depth_m"][:, side] * 1000, mask, summary),
                "penetrating_skin_vertex_count": selected_summary(skin["per_frame_penetrating_vertices"][:, side], mask, summary),
                "usable_pose_skin_mean_depth_over_selected_hand_vertices_mm": selected_summary(skin["per_frame_mean_depth_m"][:, side] * 1000, mask & usable, summary),
                "usable_pose_skin_maximum_vertex_depth_mm": selected_summary(skin["per_frame_max_depth_m"][:, side] * 1000, mask & usable, summary)}
        phases.append(row)
    anchor = None
    if anchor_targets is not None:
        target, weight = np.asarray(anchor_targets), np.asarray(anchor_weights)
        if target.shape != local.shape or weight.shape != local.shape[:-1] or not np.isfinite(target).all() or not np.isfinite(weight).all() or np.any(weight < 0):
            raise ValueError("Frozen anchor arrays differ from native tip geometry")
        difference = np.linalg.norm(local - target, axis=-1) * 1000
        anchor = {"weighted_distance_mm": float((difference * weight).sum() / weight.sum()) if weight.sum() else None,
                  "positive_anchor_samples": int((weight > 0).sum()),
                  "interpretation": "Agreement with B's frozen heuristic anchors, not independent contact accuracy"}
    return {"phase_summaries": phases, "frozen_anchor_fit_consistency": anchor,
            "all_original_frame_rows": [{"original_frame_id": frame, "object_pose_usable": bool(usable[frame]),
                "tip_surface_distances_mm": (distances[frame] * 1000).tolist(),
                "skin_mean_depth_mm": (skin["per_frame_mean_depth_m"][frame] * 1000).tolist(),
                "skin_maximum_depth_mm": (skin["per_frame_max_depth_m"][frame] * 1000).tolist(),
                "skin_penetrating_vertex_count": skin["per_frame_penetrating_vertices"][frame].tolist()} for frame in range(count)],
            "interpretation": "Distances/sliding/penetration conditional on a reconstructed object and estimated scale. Not true contact, no-slip, detachment truth or official PEN."}


def skin_nonregression(a, b, criteria):
    """Frozen relative engineering checks retain both unfiltered and usable poses."""
    before, after = a["phase_summaries"], b["phase_summaries"]
    if [row["annotation"] for row in before] != [row["annotation"] for row in after]:
        raise ValueError("A/B geometry phase windows differ")
    limits = criteria["skin_b_versus_a"]
    out = []
    for old, new in zip(before, after):
        for side in ("left", "right"):
            for label, prefix in (("all_original_frames", ""), ("same_fixed_pose_usable_gate", "usable_pose_")):
                old_mean = old["sides"][side][prefix + "skin_mean_depth_over_selected_hand_vertices_mm"]["mean"]
                new_mean = new["sides"][side][prefix + "skin_mean_depth_over_selected_hand_vertices_mm"]["mean"]
                old_max = old["sides"][side][prefix + "skin_maximum_vertex_depth_mm"]["max"]
                new_max = new["sides"][side][prefix + "skin_maximum_vertex_depth_mm"]["max"]
                enough = all(x is not None for x in (old_mean, new_mean, old_max, new_max))
                out.append({"phase": old["annotation"], "side": side, "object_pose_scope": label,
                    "a_mean_depth_mm": old_mean, "b_mean_depth_mm": new_mean,
                    "a_maximum_vertex_depth_mm": old_max, "b_maximum_vertex_depth_mm": new_max,
                    "mean_depth_change_mm": new_mean - old_mean if enough else None,
                    "maximum_depth_change_mm": new_max - old_max if enough else None,
                    "sufficient": enough,
                    "mean_depth_nonregression": bool(new_mean <= old_mean + limits["maximum_mean_depth_increase_mm"]) if enough else None,
                    "maximum_depth_nonregression": bool(new_max <= old_max + limits["maximum_worst_vertex_depth_increase_mm"]) if enough else None})
    return {"rows": out, "limits": limits, "interpretation": "Relative engineering checks against one uncertain object; not physical accuracy, official PEN or quality acceptance"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, default=ROOT)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model-batch", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists() or args.threads < 1 or args.model_batch < 1:
        parser.error("Use a new output and positive CPU threads/model batch")
    task = read(args.input_manifest)
    if task.get("track") != 1 or task.get("episode") != 16 or task.get("all30_execution_authorized") is not False:
        raise ValueError("Only the explicitly authorized Track 1 EP16 comparison is supported")
    policy, refined = load_helpers(args.tool_root)
    manifest_path = bound(task["manifest"], "original manifest")
    manifest = read(manifest_path)
    if manifest.get("track") != 1 or manifest.get("episode") != 16 or manifest["video_frames"] != 360:
        raise ValueError("Wrong source episode or original-frame coverage")
    video = bound(task["source_video"], "source video")
    if task["source_video"]["sha256"] != manifest["video_sha256"]:
        raise ValueError("Source video identity differs")
    count, expected = manifest["video_frames"], np.arange(manifest["video_frames"])
    intrinsics = np.asarray(manifest["camera_intrinsics"], np.float32)
    config = read(bound(task["fit_configuration_reference"], "preserved MP configuration"))["config"]
    if config["holdout_stride"] != 5:
        raise ValueError("Original stride-five image partition changed")
    threshold = config["min_confidence"]
    heldout = expected % 5 == 0
    protocol_path = bound(task["protocol"], "frozen comparison protocol")
    criteria_path = bound(task["criteria"], "frozen engineering criteria")
    criteria = read(criteria_path)
    if criteria.get("track") != 1 or criteria.get("episode") != 16 or criteria.get("manual_phase_sha256") != task["phases"]["sha256"]:
        raise ValueError("Frozen criteria identify different episode or manual phases")
    phases_path = bound(task["phases"], "manual source phases")
    phases = read(phases_path)
    if phases.get("track") != 1 or phases.get("episode") != 16 or phases.get("video_sha256") != manifest["video_sha256"] or phases.get("frame_count") != count:
        raise ValueError("Manual phases do not identify the same original video")
    windows = phase_windows(phases, count)
    object_path = bound(task["object"], "fixed object geometry and trajectory")
    object_report = read(bound(task["object_report"], "fixed object preparation report"))
    if object_report.get("source_video_sha256", object_report.get("video_sha256")) != manifest["video_sha256"]:
        raise ValueError("Object preparation belongs to another source video")
    obj = load_npz(object_path)
    if "camera_intrinsics" not in obj or not np.array_equal(np.asarray(obj["camera_intrinsics"], np.float32), intrinsics):
        raise ValueError("Object and human must share the exact original camera")
    sys.path.insert(0, str(args.tool_root / "src"))
    from v2hoi.hand_contact import (closest_mesh_points, native_hand_vertex_indices,
                                    native_mesh_penetration_proxy, validate_object_candidate)
    candidate = validate_object_candidate(obj, expected)
    evidence, points, confidence = {}, {}, {}
    for role in ("legacy_reference", "fresh_reference"):
        evidence[role] = load_npz(bound(task[role], role))
        if str(evidence[role]["source_video_sha256"].item()) != manifest["video_sha256"]:
            raise ValueError("Image reference source identity differs")
        points[role], confidence[role] = policy.validate_evidence(evidence[role], expected)
    baseline_path = bound(task["baseline"], "original SAM baseline")
    historical_path = bound(task["historical_no_contact"], "preserved fresh MP control")
    original_active = np.load(bound(task["original_active_hands"], "original legacy support"), allow_pickle=False)
    original_edges = set(policy.support_edges(original_active))
    reports, support, weights = {}, {}, {}
    for role in ("a", "b"):
        path = bound(task[role], role + " native parameters")
        reports[role] = read(bound(task[role]["report"], role + " report"))
        report = reports[role]
        if report.get("source_video_sha256") != manifest["video_sha256"] or report.get("mhr_model_sha256") != manifest["mhr_model_sha256"]:
            raise ValueError("Controlled report source video or decoder identity differs")
        if report["config"] != config or report["baseline_sha256"] != task["baseline"]["sha256"] or report["observations_sha256"] != task["fresh_reference"]["sha256"] or report["heldout_frame_indices"] != expected[heldout].tolist():
            raise ValueError("Controlled fit initialization/config/evidence/heldout partition differs")
        support[role] = np.load(bound(task[role]["support"], role + " support"), allow_pickle=False)
        weights[role] = np.load(bound(task[role]["weights"], role + " image weights"), allow_pickle=False)
    matched = validate_controlled_arrays(support, weights, confidence["fresh_reference"], heldout, threshold)
    if reports["a"].get("contact_provenance") is not None:
        raise ValueError("A must have no contact constraint")
    contact = reports["b"].get("contact_provenance")
    if not contact or contact["object_sha256"] != task["object"]["sha256"] or contact["phases_sha256"] != task["phases"]["sha256"] or contact["scale_factor"] != 1:
        raise ValueError("B contact input/phase identity or static baked object scale differs")
    model_path = args.model or Path(manifest["mhr_model"])
    model_hash = digest(model_path)
    if model_hash != manifest["mhr_model_sha256"]:
        raise ValueError("Native decoder identity changed")
    import torch
    torch.set_num_threads(args.threads)
    model = torch.jit.load(str(model_path), map_location="cpu").float().eval()
    paths = {"baseline": baseline_path, "a": Path(task["a"]["path"]), "b": Path(task["b"]["path"])}
    parameters, hands, parity = {}, {}, {}
    for role, path in paths.items():
        print("Fresh all-original-frame native decode: " + role, flush=True)
        parameters[role], hands[role], parity[role] = policy.decode_variant(path.parent, manifest, model, model_hash, device="cpu", batch=args.model_batch)
    review_names = ("vertices_m", "joints_3d", "faces", "frame_indices", "camera_intrinsics", "fps")
    review_hashes = {role: {name: digest(path.parent / "review_arrays" / (name + ".npy")) for name in review_names} for role, path in paths.items()}
    baseline_faces = np.load(baseline_path.parent / "review_arrays/faces.npy", allow_pickle=False)
    for role in ("a", "b"):
        if not np.array_equal(np.load(paths[role].parent / "review_arrays/faces.npy", allow_pickle=False), baseline_faces):
            raise ValueError("Controlled review mesh faces differ from the original native topology")
    historical = load_npz(historical_path)
    historical_numeric = matched_numeric_arrays(parameters["a"], historical)
    preservation, updates = {}, {}
    for role in ("a", "b"):
        preservation[role] = policy.parameter_preservation(parameters["baseline"], parameters[role], config["optimize_wrist_chain"])
        for key in ("fixed_identity_unchanged", "fixed_skeleton_scales_unchanged", "root_controls_unchanged", "all_unselected_controls_unchanged", "finite_parameters"):
            if not preservation[role][key]:
                raise ValueError("Native parameter preservation failed: " + role + "/" + key)
        updates[role] = refined.retained_update(parameters[role], parameters["baseline"], policy)
        updates[role]["inactive_finger_abstention"] = inactive_finger_abstention(parameters["baseline"]["pose"], parameters[role]["pose"], support[role])
        wrist = np.linalg.norm(hands[role][:, :, 0] - hands["baseline"][:, :, 0], axis=-1)
        updates[role]["absolute_native_estimated_wrist_displacement_metres"] = policy.summary(wrist)
        if updates[role]["finger_absolute_delta_radians"]["max"] > config["max_delta_radians"] + 1e-6 or updates[role]["arm_absolute_delta_radians"]["max"] > config["arm_delta_limit_radians"] + 1e-6 or wrist.max() > config["wrist_max_displacement_metres"] + 1e-6:
            raise ValueError("A/B native update exceeds unchanged parameter budgets")
    projected = {role: policy.project_hands(value, intrinsics) for role, value in hands.items()}
    image, motion, direct = {}, {}, {}
    for reference in evidence:
        rows = policy.motion_rows(hands["baseline"], {role: hands[role] for role in ("a", "b")}, points[reference], confidence[reference], intrinsics, threshold, heldout, original_edges)
        image[reference] = refined.image_distances(projected, points[reference], confidence[reference], heldout)
        motion[reference] = {"all_adjacent_original_frames": refined.summarize_multi_motion(rows, list(paths), policy),
            "original_legacy_support_boundaries": refined.summarize_multi_motion([row for row in rows if row["original_support_boundary"]], list(paths), policy),
            "current_support_boundaries": refined.summarize_multi_motion([row for row in rows if (row["from_original_frame_id"], ("left", "right").index(row["side"])) in set(policy.support_edges(support["a"]))], list(paths), policy),
            "holdout_touching_reference_edges": refined.summarize_multi_motion([row for row in rows if row["touches_original_holdout"] and row["independent_observed_motion_sufficient"]], list(paths), policy),
            "fast_reference_motion": refined.summarize_multi_motion([row for row in rows if row["fast_observed_motion"]], list(paths), policy),
            "manual_phase_windows": [{"annotation": window, "all_edges": refined.summarize_multi_motion([row for row in rows if window["first"] <= row["from_original_frame_id"] and row["to_original_frame_id"] <= window["last"]], list(paths), policy)} for window in windows],
            "all_adjacent_frame_rows": refined.image_reference_vocabulary(rows)}
        ab_rows = policy.motion_rows(hands["a"], {"candidate": hands["b"], "old_candidate": hands["a"]}, points[reference], confidence[reference], intrinsics, threshold, heldout, original_edges)
        direct[reference] = {"image_agreement": policy.image_agreement({"baseline": projected["a"], "candidate": projected["b"], "old_candidate": projected["a"]}, points[reference], confidence[reference], heldout),
            "motion": policy.summarize_motion(ab_rows), "original_support_boundaries": policy.summarize_motion([row for row in ab_rows if row["original_support_boundary"]]),
            "interpretation": "Unchanged local engineering guards comparing additional B motion directly with A; image references are method-dependent, not ground truth"}
    selection = native_hand_vertex_indices(model)
    anchors = Path(task["b"]["path"]).parent
    anchor_targets = np.load(anchors / "contact_anchor_targets_m.npy", allow_pickle=False)
    anchor_weights = np.load(anchors / "contact_anchor_weights.npy", allow_pickle=False)
    phase_weights = np.load(anchors / "contact_phase_weights.npy", allow_pickle=False)
    allowed = np.zeros((count, 2), bool)
    for window in windows:
        if window["anchor_weight"] > 0:
            for side in window["eligible_sides"]:
                allowed[window["first"]:window["last"] + 1, ("left", "right").index(side)] = True
    if phase_weights.shape != allowed.shape or not np.isfinite(phase_weights).all() or np.any(phase_weights < 0) or np.any(phase_weights > 1) or np.any(phase_weights[~allowed] != 0) or np.any(anchor_weights[~allowed] != 0):
        raise ValueError("Attraction remains active outside manually admitted phases")
    geometry, self_motion, opening = {}, {}, {}
    for role, path in paths.items():
        print("Whole-clip native hand-skin and object-local diagnostics: " + role, flush=True)
        vertices = np.load(path.parent / "review_arrays/vertices_m.npy", mmap_mode="r", allow_pickle=False)
        skin = native_mesh_penetration_proxy(vertices, selection, candidate, np.ones((count, 2)))
        geometry[role] = geometry_diagnostics(hands[role], candidate, windows, skin, anchor_targets, anchor_weights, closest_mesh_points, policy.summary)
        opening[role] = {"projected_2d": refined.opening_curves(projected[role]), "native_3d": refined.opening_curves(hands[role], units="native_estimated_metres")}
        self_motion[role] = {name: {"wrist_relative_D2_native_estimated_metres_per_frame_squared": policy.summary(np.linalg.norm(np.diff(policy.relative(hands[role])[:, side], n=2, axis=0), axis=-1)),
                                  "wrist_relative_D2_pixels_per_frame_squared": policy.summary(np.linalg.norm(np.diff(policy.relative(projected[role])[:, side], n=2, axis=0), axis=-1))} for side, name in enumerate(("left", "right"))}
    result = {"schema_version": "ep16_contact_ab_evaluation.v1", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "track": 1, "episode": 16, "frames": count, "original_frame_coverage_complete": True,
        "source_video_sha256": manifest["video_sha256"], "input_manifest_sha256": digest(args.input_manifest),
        "input_bindings": task, "evaluation_source_sha256": digest(Path(__file__)),
        "helper_source_sha256": {name: digest(args.tool_root / "tools" / (name + ".py")) for name in ("evaluate_human_hand_pilots", "evaluate_refined_egohand_pilots")},
        "protocol_sha256": digest(protocol_path), "mhr_model_sha256": model_hash,
        "frozen_engineering_criteria": criteria, "criteria_sha256": digest(criteria_path),
        "controlled_pair": ["a", "b"], "role_labels": {"baseline": "Original SAM MHR", "a": "A: matched MP without contact", "b": "B: matched MP with manually admitted contact"},
        "native_decoder": parity, "parameter_preservation": preservation, "retained_native_updates": updates,
        "native_review_array_sha256": review_hashes,
        "matched_controls": matched, "historical_no_contact_numeric_parity": historical_numeric,
        "same_original_camera_and_fixed_object_for_both": True, "manual_phase_windows": windows,
        "contact_attraction_outside_admitted_phase_zero": True, "hand_skin_vertex_counts": [len(x) for x in selection],
        "image_reference_distances": image, "image_reference_motion_screens": motion, "direct_b_versus_a_engineering_checks": direct,
        "prediction_self_motion": self_motion, "native_opening_proxy_curves": opening, "object_contact_diagnostics": geometry,
        "direct_b_versus_a_hand_skin_nonregression": skin_nonregression(geometry["a"], geometry["b"], criteria),
        "geometry_interpretation": "All hand-skin depths and surface distances are conditional on the same uncertain reconstructed convex object and estimated camera/relative scale; exact hidden contact and metric scale are unavailable.",
        "stock_hidden_hand_limitation": "A/B use identical MP support plus transition collar. B does not enable extra hidden-hand controls; unchanged unsupported hands are abstention, not recovery.",
        "quality_accepted": False, "contact_quality_accepted": False, "scale_accepted": False,
        "official_score_available": False, "automatic_promotion": False, "all30_execution_authorized": False,
        "contact_model_training_started": False, "episode11_execution_started": False}
    for role, path in paths.items():
        for name in review_names:
            if digest(path.parent / "review_arrays" / (name + ".npy")) != review_hashes[role][name]:
                raise ValueError("Native review geometry changed during evaluation")
    for name in ("baseline", "a", "b", "historical_no_contact", "object", "object_report", "phases", "protocol", "criteria", "legacy_reference", "fresh_reference"):
        bound(task[name], "post-evaluation " + name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "frames": count, "quality_accepted": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
