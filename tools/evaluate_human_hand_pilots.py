"""Independently compare Track 1 hand pilots without accepting reconstruction quality.

The old experiment fixes the image evidence, held-out frames and original support
edges. Extending optimizer support cannot erase a previously failing edge. All
saved native geometry is freshly decoded, on every original frame. No reference
trajectory, object contact target or official competition score is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Independent copies of the verified native skeleton and allowed control contract.
# Do not import the optimizer: evaluation must not inherit its support decisions.
HAND_JOINTS = np.asarray([
    [78, 97, 98, 99, 100, 92, 93, 94, 95, 88, 89, 90, 91, 84, 85, 86, 87, 80, 81, 82, 83],
    [42, 61, 62, 63, 64, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, 44, 45, 46, 47],
], dtype=np.int64)
ARM_CONTROLS = [33, 34, 35, 36, 37, 38, 39, 43, 44, 45, 46, 47, 48, 49]
POLICY = {
    "minimum_common_accepted_fingers": 8,
    "observed_velocity_relative_tolerance": 1.05,
    "observed_velocity_absolute_tolerance_pixels": .25,
    "unknown_motion_max_correction_step_pixels": 1.,
    "unknown_motion_max_correction_step_native_estimated_metres": .005,
    "fast_observed_motion_min_pixels_per_frame": 3.,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summary(values) -> dict:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite diagnostic value")
    return {"count": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(array.max())}


def support_edges(active: np.ndarray) -> list[tuple[int, int]]:
    active = np.asarray(active)
    if active.ndim != 2 or active.shape[1] != 2 or active.dtype.kind != "b":
        raise ValueError("Support must be a boolean [T,2] array")
    return [(int(frame), int(side)) for frame, side in np.argwhere(active[1:] != active[:-1])]


def project_hands(joints: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    projected = np.asarray(joints) @ np.asarray(intrinsics).T
    if not np.isfinite(projected).all() or np.any(projected[..., 2] <= 0):
        raise ValueError("Native hand projection must have finite positive depth")
    return projected[..., :2] / projected[..., 2:]


def relative(points: np.ndarray) -> np.ndarray:
    return points[:, :, 1:] - points[:, :, :1]


def validate_evidence(observations: dict, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.array_equal(observations["frame_indices"], expected):
        raise ValueError("Observations must retain every original source frame ID")
    if list(observations["side_order"].astype(str)) != ["left", "right"]:
        raise ValueError("Observation side order must be anatomical left/right")
    points = observations["joints2d"].astype(np.float64)
    weights = observations["confidence"].astype(np.float64)
    if points.shape != (len(expected), 2, 21, 2) or weights.shape != points.shape[:-1]:
        raise ValueError("Observation dimensions do not match the source video")
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("Invalid observation reliability weights")
    if np.any((weights > 0) & ~np.isfinite(points).all(-1)):
        raise ValueError("Accepted observations must be finite")
    return np.where(np.isfinite(points), points, 0), weights


def motion_rows(base_m: np.ndarray, variants_m: dict[str, np.ndarray], observations: np.ndarray,
                confidence: np.ndarray, intrinsics: np.ndarray, threshold: float,
                heldout: np.ndarray, original_edges: set[tuple[int, int]]) -> list[dict]:
    """Evaluate ALL adjacent frames, never differences across missing source IDs."""
    base_projected = project_hands(base_m, intrinsics)
    base_px = relative(base_projected)
    observed = relative(observations)
    vectors_px = {"baseline": np.diff(base_px, axis=0)}
    vectors_m = {"baseline": np.diff(relative(base_m), axis=0)}
    wrist_vectors_px = {"baseline": np.diff(base_projected[:, :, 0], axis=0)}
    wrist_vectors_m = {"baseline": np.diff(base_m[:, :, 0], axis=0)}
    for name, joints in variants_m.items():
        projected = project_hands(joints, intrinsics)
        vectors_px[name] = np.diff(relative(projected), axis=0)
        vectors_m[name] = np.diff(relative(joints), axis=0)
        wrist_vectors_px[name] = np.diff(projected[:, :, 0], axis=0)
        wrist_vectors_m[name] = np.diff(joints[:, :, 0], axis=0)
    observed_velocity = np.diff(observed, axis=0)
    observed_wrist_velocity = np.diff(observations[:, :, 0], axis=0)
    rows = []
    for frame in range(len(base_m) - 1):
        for side, name in enumerate(("left", "right")):
            wrist = bool(min(confidence[frame, side, 0], confidence[frame + 1, side, 0]) >= threshold)
            common = (np.minimum(confidence[frame, side, 1:], confidence[frame + 1, side, 1:]) >= threshold) & wrist
            enough = int(common.sum()) >= POLICY["minimum_common_accepted_fingers"]
            obs_velocity = observed_velocity[frame, side, common]
            speed = float(np.linalg.norm(obs_velocity, axis=-1).mean()) if common.any() else None
            row = {"side": name, "from_original_frame_id": frame, "to_original_frame_id": frame + 1,
                   "original_support_boundary": (frame, side) in original_edges,
                   "touches_original_holdout": bool(heldout[frame] or heldout[frame + 1]),
                   "common_accepted_finger_count": int(common.sum()), "adjacent_wrists_accepted": wrist,
                   "independent_observed_motion_sufficient": enough,
                   "observed_mean_wrist_relative_step_px": speed,
                   "observed_absolute_wrist_step_px": float(np.linalg.norm(observed_wrist_velocity[frame, side])) if wrist else None,
                   "fast_observed_motion": bool(enough and speed >= POLICY["fast_observed_motion_min_pixels_per_frame"]),
                   "variants": {}, "wrist_variants": {}}
            baseline_error = None
            baseline_wrist_error = None
            for label in vectors_px:
                velocity = vectors_px[label][frame, side]
                correction_px = velocity - vectors_px["baseline"][frame, side]
                correction_m = vectors_m[label][frame, side] - vectors_m["baseline"][frame, side]
                error = float(np.linalg.norm(velocity[common] - obs_velocity, axis=-1).mean()) if common.any() else None
                if label == "baseline":
                    baseline_error = error
                # Projection onto observed vectors detects frozen or reversed fast motion.
                energy = float(np.square(obs_velocity).sum())
                gain = float((velocity[common] * obs_velocity).sum() / energy) if energy > 1e-12 else None
                entry = {"mean_wrist_relative_step_px": float(np.linalg.norm(velocity, axis=-1).mean()),
                         "mean_wrist_relative_step_native_estimated_metres": float(np.linalg.norm(vectors_m[label][frame, side], axis=-1).mean()),
                         "correction_only_mean_wrist_relative_step_px": float(np.linalg.norm(correction_px, axis=-1).mean()),
                         "correction_only_mean_wrist_relative_step_native_estimated_metres": float(np.linalg.norm(correction_m, axis=-1).mean()),
                         "mean_velocity_vector_discrepancy_px": error,
                         "observed_vector_projection_gain": gain}
                if enough:
                    entry["guard"] = "observed_velocity_nonregression"
                    entry["passed"] = bool(error <= baseline_error * POLICY["observed_velocity_relative_tolerance"]
                                             + POLICY["observed_velocity_absolute_tolerance_pixels"])
                else:
                    entry["guard"] = "unknown_motion_correction_budget"
                    entry["passed"] = bool(entry["correction_only_mean_wrist_relative_step_px"] <= POLICY["unknown_motion_max_correction_step_pixels"]
                                             and entry["correction_only_mean_wrist_relative_step_native_estimated_metres"] <= POLICY["unknown_motion_max_correction_step_native_estimated_metres"])
                row["variants"][label] = entry
                wrist_velocity = wrist_vectors_px[label][frame, side]
                observed_wrist = observed_wrist_velocity[frame, side]
                wrist_error = float(np.linalg.norm(wrist_velocity - observed_wrist)) if wrist else None
                if label == "baseline":
                    baseline_wrist_error = wrist_error
                wrist_energy = float(np.square(observed_wrist).sum())
                wrist_gain = float((wrist_velocity * observed_wrist).sum() / wrist_energy) if wrist and wrist_energy > 1e-12 else None
                wrist_entry = {
                    "absolute_wrist_step_px": float(np.linalg.norm(wrist_velocity)),
                    "absolute_wrist_step_native_estimated_metres": float(np.linalg.norm(wrist_vectors_m[label][frame, side])),
                    "correction_only_absolute_wrist_step_px": float(np.linalg.norm(wrist_velocity - wrist_vectors_px["baseline"][frame, side])),
                    "correction_only_absolute_wrist_step_native_estimated_metres": float(np.linalg.norm(wrist_vectors_m[label][frame, side] - wrist_vectors_m["baseline"][frame, side])),
                    "absolute_wrist_velocity_vector_discrepancy_px": wrist_error,
                    "observed_wrist_vector_projection_gain": wrist_gain,
                }
                if wrist:
                    wrist_entry["guard"] = "observed_absolute_wrist_velocity_nonregression"
                    wrist_entry["passed"] = bool(wrist_error <= baseline_wrist_error * POLICY["observed_velocity_relative_tolerance"]
                                                   + POLICY["observed_velocity_absolute_tolerance_pixels"])
                else:
                    wrist_entry["guard"] = "unknown_motion_absolute_wrist_correction_budget"
                    wrist_entry["passed"] = bool(wrist_entry["correction_only_absolute_wrist_step_px"] <= POLICY["unknown_motion_max_correction_step_pixels"]
                                                   and wrist_entry["correction_only_absolute_wrist_step_native_estimated_metres"] <= POLICY["unknown_motion_max_correction_step_native_estimated_metres"])
                row["wrist_variants"][label] = wrist_entry
            rows.append(row)
    return rows


def summarize_motion(rows: list[dict]) -> dict:
    output = {"edge_count": len(rows), "variants": {}, "absolute_wrist_motion": summarize_wrist_motion(rows)}
    for label in ("baseline", "old_candidate", "candidate"):
        samples = [row["variants"][label] for row in rows]
        observed = [entry for entry in samples if entry["guard"] == "observed_velocity_nonregression"]
        unknown = [entry for entry in samples if entry["guard"] == "unknown_motion_correction_budget"]
        output["variants"][label] = {
            "all_engineering_guards_passed": bool(all(entry["passed"] for entry in samples)) if samples else None,
            "observed_motion_edges": len(observed), "observed_motion_failed": sum(not entry["passed"] for entry in observed),
            "unknown_motion_edges": len(unknown), "unknown_motion_budget_failed": sum(not entry["passed"] for entry in unknown),
            "observed_velocity_discrepancy_px": summary([entry["mean_velocity_vector_discrepancy_px"] for entry in observed]),
            "observed_vector_projection_gain": summary([entry["observed_vector_projection_gain"] for entry in observed if entry["observed_vector_projection_gain"] is not None]),
            "correction_step_px": summary([entry["correction_only_mean_wrist_relative_step_px"] for entry in samples]),
            "correction_step_native_estimated_metres": summary([entry["correction_only_mean_wrist_relative_step_native_estimated_metres"] for entry in samples]),
        }
    return output


def summarize_wrist_motion(rows: list[dict]) -> dict:
    """Independent absolute wrist tests cannot be hidden by subtracting the wrist."""
    output = {"edge_count": len(rows), "variants": {},
              "interpretation": "Separate local engineering diagnostics using the same numerical tolerances as finger support checks; not official wrist scoring or verified physical error"}
    for label in ("baseline", "old_candidate", "candidate"):
        samples = [row["wrist_variants"][label] for row in rows]
        observed = [entry for entry in samples if entry["guard"] == "observed_absolute_wrist_velocity_nonregression"]
        unknown = [entry for entry in samples if entry["guard"] == "unknown_motion_absolute_wrist_correction_budget"]
        output["variants"][label] = {
            "all_engineering_guards_passed": bool(all(entry["passed"] for entry in samples)) if samples else None,
            "observed_wrist_edges": len(observed), "observed_wrist_failed": sum(not entry["passed"] for entry in observed),
            "unknown_wrist_edges": len(unknown), "unknown_wrist_budget_failed": sum(not entry["passed"] for entry in unknown),
            "observed_absolute_wrist_velocity_discrepancy_px": summary([entry["absolute_wrist_velocity_vector_discrepancy_px"] for entry in observed]),
            "observed_wrist_vector_projection_gain": summary([entry["observed_wrist_vector_projection_gain"] for entry in observed if entry["observed_wrist_vector_projection_gain"] is not None]),
            "correction_only_absolute_wrist_step_px": summary([entry["correction_only_absolute_wrist_step_px"] for entry in samples]),
            "correction_only_absolute_wrist_step_native_estimated_metres": summary([entry["correction_only_absolute_wrist_step_native_estimated_metres"] for entry in samples]),
        }
    return output


def image_agreement(projected: dict[str, np.ndarray], points: np.ndarray, weights: np.ndarray,
                    heldout: np.ndarray) -> dict:
    output = {}
    for split, mask in (("all_original_observations", np.ones(len(points), bool)),
                        ("original_holdout", heldout), ("original_training", ~heldout)):
        output[split] = {}
        for side, name in enumerate(("left", "right")):
            w = weights[:, side] * mask[:, None]
            rw = np.minimum(w[:, 1:], w[:, :1])
            data = {"observed_frames": int((w[:, 1:].sum(-1) > 0).sum()),
                    "accepted_landmarks": int((w > 0).sum()), "variants": {}}
            for label, prediction in projected.items():
                distance = np.linalg.norm(prediction[:, side] - points[:, side], axis=-1)
                rd = np.linalg.norm(relative(prediction)[:, side] - relative(points)[:, side], axis=-1)
                data["variants"][label] = {
                    "weighted_reprojection_pixels": float((distance * w).sum() / w.sum()) if w.sum() else None,
                    "weighted_wrist_relative_reprojection_pixels": float((rd * rw).sum() / rw.sum()) if rw.sum() else None}
            sufficient = data["observed_frames"] >= 5
            before = data["variants"]["baseline"]["weighted_reprojection_pixels"]
            after = data["variants"]["candidate"]["weighted_reprojection_pixels"]
            data["original_image_guard_sufficient"] = sufficient
            data["candidate_image_nonregression"] = bool(after <= before * 1.05 + .25) if sufficient else None
            output[split][name] = data
    return output


def review_window_metrics(windows: list[dict], rows: list[dict], frames: int) -> list[dict]:
    """Keep manually reviewed action intervals separate from detector support."""
    output = []
    for window in windows:
        first, last = window["first_original_frame_id"], window["last_original_frame_id"]
        side = window["side"]
        if type(first) is not int or type(last) is not int or not 0 <= first < last < frames:
            raise ValueError("Review window must use an in-range original-frame interval")
        if side not in ("left", "right", "both") or not str(window.get("label", "")).strip():
            raise ValueError("Review window needs side and a descriptive source-review label")
        selected = [row for row in rows if first <= row["from_original_frame_id"]
                    and row["to_original_frame_id"] <= last and side in ("both", row["side"])]
        observed = [row for row in selected if row["independent_observed_motion_sufficient"]]
        output.append({"annotation": window, "all_edges": summarize_motion(selected),
                       "independently_observed_edges": summarize_motion(observed),
                       "fast_observed_edges": summarize_motion([row for row in observed if row["fast_observed_motion"]]),
                       "heldout_observed_edges": summarize_motion([row for row in observed if row["touches_original_holdout"]]),
                       "original_support_boundaries": summarize_motion([row for row in selected if row["original_support_boundary"]]),
                       "interpretation": "Manual source-review interval, not a contact label inferred from detector support; insufficient observations cannot verify hidden grasp changes"})
    return output


def parameter_preservation(baseline: dict, candidate: dict, optimize_wrist: bool) -> dict:
    allowed = list(range(68, 122)) + (ARM_CONTROLS if optimize_wrist else [])
    fixed = [index for index in range(136) if index not in allowed]
    return {"fixed_identity_unchanged": bool(np.array_equal(baseline["shape"], candidate["shape"])),
            "fixed_skeleton_scales_unchanged": bool(np.array_equal(baseline["scales"], candidate["scales"])),
            "root_controls_unchanged": bool(np.array_equal(baseline["pose"][:, :6], candidate["pose"][:, :6])),
            "all_unselected_controls_unchanged": bool(np.array_equal(baseline["pose"][:, fixed], candidate["pose"][:, fixed])),
            "allowed_control_indices": sorted(allowed),
            "maximum_allowed_control_change_radians": float(np.max(np.abs(candidate["pose"][:, allowed] - baseline["pose"][:, allowed]))),
            "finite_parameters": bool(all(np.isfinite(candidate[key]).all() for key in ("pose", "shape", "scales")))}


def decode_variant(directory: Path, manifest: dict, model, model_hash: str, *, device: str, batch: int) -> tuple[dict, np.ndarray, dict]:
    """One model load; all-frame joint/mesh parity with bounded temporary memory."""
    import torch
    from v2hoi.human_result import load_human_result, native_vertices_to_track1, validate_human_result

    path = directory / f"episode_{manifest['episode']:06d}_human.npz"
    result = load_human_result(path)
    count = manifest["video_frames"]
    structure = validate_human_result(result, count)
    expected = np.arange(count)
    if not np.array_equal(result["frame_indices"], expected):
        raise ValueError(f"Changed original frame coverage: {directory}")
    report = result.get("report", {})
    if report.get("source_video_sha256", report.get("video_sha256")) != manifest["video_sha256"] or report.get("mhr_model_sha256") != model_hash:
        raise ValueError(f"Native parameter provenance mismatch: {directory}")
    review = directory / "review_arrays"
    if not np.array_equal(np.load(review / "frame_indices.npy", allow_pickle=False), expected):
        raise ValueError("Review frame IDs differ from source")
    if not np.array_equal(np.load(review / "camera_intrinsics.npy", allow_pickle=False), np.asarray(manifest["camera_intrinsics"], dtype=np.float32)):
        raise ValueError("Review camera differs from original manifest")
    if not np.isclose(float(np.load(review / "fps.npy", allow_pickle=False)), manifest["video_fps"]):
        raise ValueError("Review frame rate differs from source")
    vertices = np.load(review / "vertices_m.npy", mmap_mode="r", allow_pickle=False)
    joints = np.load(review / "joints_3d.npy", mmap_mode="r", allow_pickle=False)
    if vertices.shape != (count, 18439, 3) or joints.shape != (count, 127, 3):
        raise ValueError("Wrong native review topology or frame count")
    shape = torch.as_tensor(np.array(result["shape"], copy=True), device=device)
    scales = torch.as_tensor(np.array(result["scales"], copy=True), device=device)
    maximum_mesh = maximum_joint = 0.
    all_hands = []
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for start in range(0, count, batch):
            pose = torch.as_tensor(np.array(result["pose"][start:start + batch], copy=True), device=device)
            params = torch.cat([pose, scales[None].expand(len(pose), -1)], dim=1)
            v, j = model(shape[None].expand(len(pose), -1).contiguous(), params, pose.new_zeros(len(pose), 72), True)
            v = native_vertices_to_track1(v.cpu().numpy())
            j = native_vertices_to_track1(j[..., :3].cpu().numpy())
            if not np.isfinite(v).all() or not np.isfinite(j).all():
                raise ValueError("Nonfinite freshly decoded geometry")
            saved_v, saved_j = vertices[start:start + len(pose)], joints[start:start + len(pose)]
            if not np.isfinite(saved_v).all() or not np.isfinite(saved_j).all():
                raise ValueError("Nonfinite saved geometry")
            maximum_mesh = max(maximum_mesh, float(np.max(np.abs(v - saved_v))))
            maximum_joint = max(maximum_joint, float(np.max(np.abs(j - saved_j))))
            all_hands.append(j[:, HAND_JOINTS])
    hand_joints = np.concatenate(all_hands).astype(np.float64)
    saved_hands = directory / "hand_joints_m.npy"
    hand_parity = float(np.max(np.abs(np.load(saved_hands, allow_pickle=False) - hand_joints))) if saved_hands.exists() else None
    parity = {"frames_freshly_decoded": count, "native_mesh_max_abs_metres": maximum_mesh,
              "native_joint_max_abs_metres": maximum_joint, "saved_hand_max_abs_metres": hand_parity,
              "passed": bool(maximum_mesh <= 1e-5 and maximum_joint <= 1e-5 and (hand_parity is None or hand_parity <= 1e-5)),
              "interpretation": "Decoder agreement, not accuracy against real geometry", **structure,
              "parameters_sha256": digest(path)}
    if not parity["passed"]:
        raise ValueError(f"Native decoder parity failed: {directory}: {parity}")
    return result, hand_joints, parity


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("baseline", "old-candidate", "candidate", "observations", "manifest", "model", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model-batch", type=int, default=8)
    parser.add_argument("--review-windows", type=Path,
                        help="Track 1 source-bound JSON with manually reviewed action/occlusion windows")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; preserve previous evidence and choose a new file")
    if args.threads < 1 or args.model_batch < 1:
        parser.error("threads and model-batch must be positive")
    manifest = read_json(args.manifest)
    if manifest.get("track") != 1 or manifest.get("episode") not in (11, 16):
        parser.error("This evaluator is restricted to the two authorized Track 1 pilots")
    expected = np.arange(manifest["video_frames"])
    with np.load(args.observations, allow_pickle=False) as archive:
        observations = {name: archive[name] for name in archive.files}
    if str(observations["source_video_sha256"].item()) != manifest["video_sha256"]:
        raise ValueError("Independent observations identify a different video")
    points, weights = validate_evidence(observations, expected)
    old_report = read_json(args.old_candidate / "report.json")
    new_report = read_json(args.candidate / "report.json")
    obs_hash, model_hash = digest(args.observations), digest(args.model)
    baseline_hash = digest(args.baseline / f"episode_{manifest['episode']:06d}_human.npz")
    if model_hash != manifest["mhr_model_sha256"]:
        raise ValueError("Native decoder differs from original experiment")
    for report in (old_report, new_report):
        if report.get("observations_sha256") != obs_hash or report.get("source_video_sha256") != manifest["video_sha256"]:
            raise ValueError("Candidate evidence changed or does not match this source")
        if report.get("baseline_sha256") != baseline_hash:
            raise ValueError("Candidate baseline changed from the frozen original experiment")
    holdout_ids = old_report["heldout_frame_indices"]
    if new_report["heldout_frame_indices"] != holdout_ids:
        raise ValueError("New candidate changed the frozen held-out image split")
    heldout = np.isin(expected, holdout_ids)
    threshold = old_report["config"]["min_confidence"]
    old_active = np.load(args.old_candidate / "active_hands.npy", allow_pickle=False)
    new_active = np.load(args.candidate / "active_hands.npy", allow_pickle=False)
    if old_active.shape != (len(expected), 2) or new_active.shape != old_active.shape:
        raise ValueError("Support coverage differs from source frames")
    original_edges, new_edges = set(support_edges(old_active)), set(support_edges(new_active))
    declared = {(row["from_original_frame_id"], ("left", "right").index(row["side"]))
                for row in old_report["support_boundary_motion"]["boundaries"]}
    if declared != original_edges:
        raise ValueError("Old recorded boundaries do not match its immutable support array")
    old_weights = np.load(args.old_candidate / "optimization_weights.npy", allow_pickle=False)
    new_weights = np.load(args.candidate / "optimization_weights.npy", allow_pickle=False)
    if (old_weights.shape != weights.shape or new_weights.shape != weights.shape
            or not np.isfinite(old_weights).all() or not np.isfinite(new_weights).all()
            or np.any(old_weights < 0) or np.any(new_weights < 0)):
        raise ValueError("Invalid saved optimization weights")
    masks_unchanged = bool(old_weights.shape == new_weights.shape and np.array_equal(old_weights > 0, new_weights > 0))
    holdout_excluded = bool(not np.any(new_weights[heldout] > 0))
    if not masks_unchanged or not holdout_excluded:
        raise ValueError("New optimization changed the original image target mask or used held-out targets")
    if np.any((new_weights > 0) & (weights < threshold)):
        raise ValueError("Optimizer introduced unsupported image targets")
    import torch
    torch.set_num_threads(args.threads)
    model = torch.jit.load(str(args.model), map_location=args.device).float().eval()
    results, hand_joints, parities = {}, {}, {}
    for name, path in (("baseline", args.baseline), ("old_candidate", args.old_candidate), ("candidate", args.candidate)):
        print(f"Fresh all-frame native decode: {name}", flush=True)
        results[name], hand_joints[name], parities[name] = decode_variant(path, manifest, model, model_hash, device=args.device, batch=args.model_batch)
    intrinsics = np.asarray(manifest["camera_intrinsics"], dtype=np.float64)
    rows = motion_rows(hand_joints["baseline"], {name: hand_joints[name] for name in ("old_candidate", "candidate")},
                       points, weights, intrinsics, threshold, heldout, original_edges)
    original = [row for row in rows if row["original_support_boundary"]]
    extended = [row for row in rows if (row["from_original_frame_id"], ("left", "right").index(row["side"])) in new_edges - original_edges]
    observed = [row for row in rows if row["independent_observed_motion_sufficient"]]
    preservation = parameter_preservation(results["baseline"], results["candidate"], bool(new_report["config"]["optimize_wrist_chain"]))
    projected = {name: project_hands(value, intrinsics) for name, value in hand_joints.items()}
    manual_windows = []
    if args.review_windows:
        annotation = read_json(args.review_windows)
        if (annotation.get("track") != 1 or annotation.get("episode") != manifest["episode"]
                or annotation.get("source_video_sha256") != manifest["video_sha256"]):
            raise ValueError("Review windows must identify the exact authorized Track 1 video")
        manual_windows = review_window_metrics(annotation["windows"], rows, len(expected))
    result = {"schema_version": "human_hand_pilot_evaluation.v1", "track": 1, "episode": manifest["episode"],
              "frames": len(expected), "original_frame_coverage_complete": True,
              "source_video_sha256": manifest["video_sha256"], "mhr_model_sha256": model_hash,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "inputs": {name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in {
                  "manifest": args.manifest, "observations": args.observations,
                  "old_report": args.old_candidate / "report.json", "new_report": args.candidate / "report.json",
                  "old_active_hands": args.old_candidate / "active_hands.npy", "new_active_hands": args.candidate / "active_hands.npy"}.items()},
              "evaluation_source_sha256": digest(Path(__file__)), "unchanged_original_guard_policy": POLICY,
              "frozen_evidence": {"original_image_target_positive_masks_unchanged": masks_unchanged,
                                  "original_holdout_targets_excluded_from_fit": holdout_excluded,
                                  "heldout_original_frame_ids": holdout_ids,
                                  "original_support_boundary_count": len(original_edges),
                                  "candidate_support_boundary_count": len(new_edges),
                                  "new_boundary_count_in_addition_to_original": len(extended)},
              "native_decoder": parities, "parameter_preservation": preservation,
              "image_agreement": image_agreement(projected, points, weights, heldout),
              "original_support_boundaries": {**summarize_motion(original), "rows": original},
              "additional_candidate_support_boundaries": {**summarize_motion(extended), "rows": extended},
              "all_adjacent_original_frames": summarize_motion(rows),
              "independently_observed_interior_edges": summarize_motion([row for row in observed if not row["original_support_boundary"]]),
              "independently_observed_edges_touching_holdout": summarize_motion([row for row in observed if row["touches_original_holdout"]]),
              "fast_independently_observed_motion": {**summarize_motion([row for row in observed if row["fast_observed_motion"]]),
                                                    "rows": [row for row in observed if row["fast_observed_motion"]]},
              "manual_source_review_windows": manual_windows,
              "manual_source_review_windows_sha256": digest(args.review_windows) if args.review_windows else None,
              "all_adjacent_frame_rows": rows,
              "experimental_only": True, "automatic_candidate_promotion": False,
              "reconstruction_quality_accepted": False, "contact_quality_accepted": False, "official_score_available": False,
              "limitations": ["Independent image agreement is not true 3D accuracy; hidden fingers lack verified motion targets.",
                              "Unknown-motion budgets are conservative engineering diagnostics, not verified errors or official scores.",
                              "Native estimated metres are not calibrated physical scale. Decoder parity is a numerical contract check.",
                              "Fast observed motion is not a grasp/release annotation. Original video review must verify action timing and contact.",
                              "A support transition is an observation/optimization dropout, not proof that the hand is visually occluded.",
                              "Lower self-acceleration alone does not establish preserved action or correct occluded motion.",
                              "Only the owner may authorize a subsequent all-30 batch; this report cannot do so."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "original_support_boundaries": result["original_support_boundaries"]["variants"],
                      "experimental_only": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
