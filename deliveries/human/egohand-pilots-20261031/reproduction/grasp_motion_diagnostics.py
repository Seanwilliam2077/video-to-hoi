"""Independent opening/motion curves for the two authorized hand comparisons.

No fitting, interpolation, phase inference, thresholds, official scores or result
selection. The ratios are geometric proxies, not true joint angles or contact.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

TASK = Path(__file__).resolve().parent
RESULTS = Path("J:/video-to-hoi/runs/egohand-pilots-20261031").resolve()
HAND_JOINTS = np.array([
    [78,97,98,99,100,92,93,94,95,88,89,90,91,84,85,86,87,80,81,82,83],
    [42,61,62,63,64,56,57,58,59,52,53,54,55,48,49,50,51,44,45,46,47],
])
SIDES = ("left", "right")


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def distribution(values, ids=None):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    good = values[valid]
    if not good.size:
        return {"count": 0, "mean": None, "p05": None, "median": None,
                "p95": None, "maximum": None, "minimum": None, "p95_minus_p05": None}
    q = np.quantile(good, [.05, .5, .95])
    out = {"count": int(good.size), "mean": float(good.mean()), "p05": float(q[0]),
           "median": float(q[1]), "p95": float(q[2]), "maximum": float(good.max()),
           "minimum": float(good.min()), "p95_minus_p05": float(q[2] - q[0])}
    if ids is not None:
        if values.ndim != 1 or np.shape(ids) != values.shape:
            raise ValueError("Extremum source-frame IDs require a matching scalar curve")
        valid_ids = np.asarray(ids)[valid]
        out["minimum_original_frame_id"] = int(valid_ids[np.argmin(good)])
        out["maximum_original_frame_id"] = int(valid_ids[np.argmax(good)])
    return out


def masked_mean(values, valid):
    count = valid.sum(axis=-1)
    total = np.where(valid, values, 0).sum(axis=-1)
    mean = np.full(count.shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    return mean, count


def opening_curve(points, valid):
    """Ratio of tip separation to index-MCP/little-MCP width, same frame."""
    palm = np.linalg.norm(points[:, :, 5] - points[:, :, 17], axis=-1)
    palm_valid = valid[:, :, 5] & valid[:, :, 17] & np.isfinite(palm) & (palm > 0)
    curves = {}
    for name, tip in (("thumb_index", 8), ("thumb_middle", 12)):
        separation = np.linalg.norm(points[:, :, 4] - points[:, :, tip], axis=-1)
        ok = palm_valid & valid[:, :, 4] & valid[:, :, tip] & np.isfinite(separation)
        ratio = np.full(separation.shape, np.nan, dtype=np.float64)
        np.divide(separation, palm, out=ratio, where=ok)
        curves[name + "_opening_ratio"] = ratio
        curves[name + "_tip_separation"] = np.where(ok, separation, np.nan)
    curves["palm_width"] = np.where(palm_valid, palm, np.nan)
    return curves


def motion_curves(points, valid, frame_ids):
    """Per-edge/triplet finger motion, never differentiated across missing IDs."""
    relative = points[:, :, 1:] - points[:, :, :1]
    joint_valid = valid[:, :, 1:] & valid[:, :, :1]
    step = np.linalg.norm(relative[1:] - relative[:-1], axis=-1)
    step_valid = joint_valid[1:] & joint_valid[:-1]
    step_valid &= (np.diff(frame_ids) == 1)[:, None, None]
    step_valid &= np.isfinite(step)
    step_mean, step_count = masked_mean(step, step_valid)
    second = np.linalg.norm(relative[2:] - 2 * relative[1:-1] + relative[:-2], axis=-1)
    second_valid = joint_valid[2:] & joint_valid[1:-1] & joint_valid[:-2]
    second_valid &= ((np.diff(frame_ids)[1:] == 1) & (np.diff(frame_ids)[:-1] == 1))[:, None, None]
    second_valid &= np.isfinite(second)
    second_mean, second_count = masked_mean(second, second_valid)
    return {"wrist_relative_step": step_mean, "step_joint_count": step_count,
            "step_from_frame": frame_ids[:-1].copy(), "step_to_frame": frame_ids[1:].copy(),
            "wrist_relative_second_difference": second_mean, "second_joint_count": second_count,
            "second_center_frame": frame_ids[1:-1].copy(),
            "_step_landmark_values": np.where(step_valid, step, np.nan),
            "_second_landmark_values": np.where(second_valid, second, np.nan)}


def describe_curves(curves, ids, start, end):
    frame_mask = (ids >= start) & (ids <= end)
    edge_mask = (curves["step_from_frame"] >= start) & (curves["step_to_frame"] <= end)
    triplet_mask = (curves["second_center_frame"] - 1 >= start) & (curves["second_center_frame"] + 1 <= end)
    by_side = {}
    for side, label in enumerate(SIDES):
        result = {}
        for name in ("thumb_index_opening_ratio", "thumb_middle_opening_ratio", "palm_width"):
            result[name] = distribution(curves[name][frame_mask, side], ids[frame_mask])
        result["wrist_relative_edge_mean_motion"] = distribution(curves["wrist_relative_step"][edge_mask, side],
                                                                 curves["step_to_frame"][edge_mask])
        result["wrist_relative_all_landmark_steps"] = distribution(curves["_step_landmark_values"][edge_mask, side])
        result["wrist_relative_triplet_mean_second_difference"] = distribution(
            curves["wrist_relative_second_difference"][triplet_mask, side], curves["second_center_frame"][triplet_mask])
        result["wrist_relative_all_landmark_second_differences"] = distribution(
            curves["_second_landmark_values"][triplet_mask, side])
        by_side[label] = result
    return {"first_original_frame_id": start, "last_original_frame_id": end, "hands": by_side}


def native_variant(directory, expected, K, source_sha):
    directory = Path(directory)
    arrays = directory / "review_arrays"
    files = {name: arrays / name for name in ("joints_3d.npy", "frame_indices.npy", "camera_intrinsics.npy")}
    ids = np.load(files["frame_indices.npy"], allow_pickle=False)
    camera = np.load(files["camera_intrinsics.npy"], allow_pickle=False)
    if not np.array_equal(ids, expected) or not np.array_equal(camera, K):
        raise ValueError("Every variant must retain the identical original frames and camera")
    joints = np.load(files["joints_3d.npy"], allow_pickle=False)
    if joints.shape != (len(expected), 127, 3) or not np.isfinite(joints).all():
        raise ValueError("Require complete finite native-MHR joint export")
    hand = joints[:, HAND_JOINTS].astype(np.float64)
    h = hand @ K.astype(np.float64).T
    positive = hand[..., 2] > 0
    uv = np.full(hand.shape[:-1] + (2,), np.nan)
    np.divide(h[..., :2], h[..., 2:3], out=uv, where=positive[..., None])
    valid2d = positive & np.isfinite(uv).all(-1)
    # No image or crop clipping here: curves describe the exported prediction,
    # not only the easiest projectable subset. Negative depth remains invalid.
    valid3d = np.isfinite(hand).all(-1)
    report_path = directory / "report.json"
    if report_path.is_file():
        report = read(report_path)
        if report.get("source_video_sha256", report.get("video_sha256")) != source_sha:
            raise ValueError("Native output report identifies a different source")
        files["report.json"] = report_path
    return {"2d": (uv, valid2d), "3d": (hand, valid3d)}, {
        "directory": str(directory), "input_sha256": {k: digest(v) for k, v in files.items()},
        "full_native_decoder_parity": "Not established by this helper; see separate fresh native evaluation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=(11, 16), required=True)
    parser.add_argument("--raw-candidate", type=Path, required=True)
    parser.add_argument("--smoothed-candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(RESULTS / f"episode_{args.episode:06d}") or destination.exists():
        parser.error("Use a fresh output under the authorized episode's isolated result directory")
    begin = time.monotonic()
    manifest_path = TASK / "inputs/manifest.json"
    manifest = read(manifest_path)
    if manifest.get("track") != 1 or manifest.get("all30_execution_authorized") is not False:
        raise ValueError("Wrong task scope")
    item = next(e for e in manifest["episodes"] if e["episode"] == args.episode)
    expected = np.arange(item["frames"])
    original_manifest = read(item["manifest"])
    K = np.asarray(original_manifest["camera_intrinsics"], dtype=np.float32)
    observation_path = Path(item["original_observations"])
    if digest(observation_path) != item["original_observations_sha256"]:
        raise ValueError("Original MediaPipe evidence changed")
    observations = load_npz(observation_path)
    if not np.array_equal(observations["frame_indices"], expected):
        raise ValueError("Independent observation source-frame mapping differs")
    if str(observations["source_video_sha256"].item()) != item["video_sha256"]:
        raise ValueError("Independent observation video differs")
    if observations["side_order"].tolist() != list(SIDES):
        raise ValueError("Independent observation anatomical side order differs")
    windows_path = TASK / f"ep{args.episode}-review-windows.json"
    window_report = read(windows_path)
    if window_report["source_video_sha256"] != item["video_sha256"]:
        raise ValueError("Review intervals identify a different video")
    windows = [(0, len(expected) - 1)] + [(x["first_original_frame_id"], x["last_original_frame_id"])
                                         for x in window_report["windows"]]
    paths = {"baseline": Path(item["baseline"]["path"]).parent,
             "mediapipe_mhr": Path(item["mediapipe_candidate"]).parent,
             "egohand_raw_mhr": args.raw_candidate}
    if args.smoothed_candidate:
        paths["egohand_smoothed_mhr"] = args.smoothed_candidate
    all_curves, bindings, summaries, saved = {}, {}, {}, {}
    for label, path in paths.items():
        data, bindings[label] = native_variant(path, expected, K, item["video_sha256"])
        all_curves[label] = {}
        summaries[label] = {}
        for dimensions, (points, valid) in data.items():
            curves = {**opening_curve(points, valid), **motion_curves(points, valid, expected)}
            all_curves[label][dimensions] = curves
            summaries[label][dimensions] = [describe_curves(curves, expected, a, b) for a, b in windows]
            for name, values in curves.items():
                if not name.startswith("_"):
                    saved[label + "__" + dimensions + "__" + name] = values
    observed_points = observations["joints2d"].astype(np.float64)
    observed_valid = (observations["confidence"] > 0) & np.isfinite(observed_points).all(-1)
    observed_curves = {**opening_curve(observed_points, observed_valid),
                       **motion_curves(observed_points, observed_valid, expected)}
    summaries["accepted_mediapipe_2d_evidence"] = [describe_curves(observed_curves, expected, a, b) for a, b in windows]
    for name, values in observed_curves.items():
        if not name.startswith("_"):
            saved["accepted_mediapipe_2d_evidence__" + name] = values
    comparisons = {}
    for label, dimensions in all_curves.items():
        comparisons[label] = {}
        for side, hand_name in enumerate(SIDES):
            comparisons[label][hand_name] = {}
            for name in ("thumb_index_opening_ratio", "thumb_middle_opening_ratio"):
                model_curve = dimensions["2d"][name][:, side]
                evidence_curve = observed_curves[name][:, side]
                good = np.isfinite(model_curve) & np.isfinite(evidence_curve)
                comparisons[label][hand_name][name] = {
                    "same_original_frame_ids": expected[good].tolist(),
                    "absolute_ratio_discrepancy": distribution(np.abs(model_curve[good] - evidence_curve[good])),
                    "model_ratio_on_common_frames": distribution(model_curve[good], expected[good]),
                    "accepted_mediapipe_ratio_on_common_frames": distribution(evidence_curve[good], expected[good]),
                    "interpretation": "Agreement with a fallible detector on accepted landmarks, not true grasp-angle accuracy.",
                }
    destination.mkdir(parents=True)
    curves_path = destination / "curves.npz"
    np.savez_compressed(curves_path, original_frame_ids=expected, fps=np.array(item["fps"]), **saved)
    report = {"schema_version": "hand_opening_motion_diagnostics.v1",
              "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "episode": args.episode, "track": 1, "frames": len(expected), "fps": item["fps"],
              "source_video_sha256": item["video_sha256"], "helper_sha256": digest(__file__),
              "inputs_manifest_sha256": digest(manifest_path), "review_windows_sha256": digest(windows_path),
              "mediapipe_evidence_sha256": digest(observation_path), "native_variant_bindings": bindings,
              "curves_file": str(curves_path), "curves_sha256": digest(curves_path),
              "curve_npz_nan_policy": "NaN denotes unsupported/invalid scalar; never filled or interpolated.",
              "geometry_roles": "All 21 per-hand native-MHR joints in the verified MediaPipe semantic order.",
              "geometry_mapping": HAND_JOINTS.tolist(),
              "opening_definition": "Distance(thumb_tip4,index_tip8) or Distance(thumb_tip4,middle_tip12), divided by Distance(index_MCP5,little_MCP17). A dimensionless geometric opening proxy, not angle/contact/force.",
              "motion_definition": "Euclidean first and second differences of finger joints1:21 relative to wrist0. 2D in source px/frame and px/frame^2; 3D in native estimated m/frame and m/frame^2. No FPS^2 multiplier.",
              "independent_evidence_policy": "Opening uses only frames where each of the two tips and both palm-width endpoints has accepted MediaPipe weights. Relative motion also requires accepted wrist and finger landmarks on every consecutive edge/triplet. No differentiation across gaps.",
              "review_window_policy": "Frozen broad manual review windows, not ground-truth grasp phases. Extremum IDs are descriptive only.",
              "lag_limit": "No post-hoc best-lag alignment, new phase labels or shifted comparison. Inspect original-ID curves against synchronized RGB to assess event timing.",
              "plotting": "No installed matplotlib was found in the existing Human, development, EgoHand or bundled Python runtimes. Environment unchanged; complete curves are provided in NPZ with strict JSON summary.",
              "summaries": summaries, "same_frame_detector_comparisons": comparisons,
              "official_scores_computed": False, "quality_accepted": False,
              "contact_quality_accepted": False, "all30_execution_started": False,
              "wall_seconds": time.monotonic() - begin}
    with (destination / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"episode": args.episode, "frames": len(expected), "variants": list(paths),
                      "output": str(destination), "summary_sha256": digest(destination / "summary.json")}))


if __name__ == "__main__":
    main()
