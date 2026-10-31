"""Render new isolated pilot evidence with one common stabilized display ROI.

Requires a completed fresh native evaluation. Display ROIs are not model input
crops: they include all selected native methods and admitted proposal points.
No fitting, model inference, candidate selection or quality acceptance occurs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


SAMPLES = {16: [0, 40, 60, 110, 120, 240, 284, 285, 286, 287, 308, 359],
           11: [0, 280, 340, 400, 480, 511, 512, 559, 560, 620, 680, 800, 803, 850, 876]}
HAND_JOINTS = np.asarray([
    [78, 97, 98, 99, 100, 92, 93, 94, 95, 88, 89, 90, 91, 84, 85, 86, 87, 80, 81, 82, 83],
    [42, 61, 62, 63, 64, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, 44, 45, 46, 47]], dtype=np.int64)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def bound(binding):
    path = Path(binding["path"])
    if digest(path) != binding["sha256"]:
        raise ValueError(f"Changed frozen input: {path}")
    return path


def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def common_display_rois(source_boxes, point_sets, radius=4, margin=1.08):
    """Median centers and local maximum sizes, enlarged to contain each union.

    Every method uses the exact same square on a given source frame and side.
    The noncausal nine-frame display filter never changes input or fit targets.
    """
    boxes = np.asarray(source_boxes, dtype=np.float64)
    if boxes.ndim != 3 or boxes.shape[1:] != (2, 4) or not np.isfinite(boxes).all():
        raise ValueError("Display ROI needs finite [T,2,4] source bounds")
    if np.any(boxes[..., 2:] <= boxes[..., :2]):
        raise ValueError("Source ROI bounds must have positive area")
    low, high = boxes[..., :2].copy(), boxes[..., 2:].copy()
    for points, valid in point_sets:
        points = np.asarray(points, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool) & np.isfinite(points).all(-1)
        if points.shape[:2] != boxes.shape[:2] or points.shape[-1] != 2 or valid.shape != points.shape[:-1]:
            raise ValueError("Display point geometry differs from source frame/side mapping")
        for frame in range(len(boxes)):
            for side in range(2):
                accepted = points[frame, side, valid[frame, side]]
                if len(accepted):
                    low[frame, side] = np.minimum(low[frame, side], accepted.min(0))
                    high[frame, side] = np.maximum(high[frame, side], accepted.max(0))
    centers = (low + high) / 2
    widths = np.max(high - low, axis=-1)
    result = np.empty_like(boxes, dtype=np.int64)
    for frame in range(len(boxes)):
        first, last = max(0, frame - radius), min(len(boxes), frame + radius + 1)
        center = np.median(centers[first:last], axis=0)
        required = 2 * np.maximum(np.abs(low[frame] - center), np.abs(high[frame] - center)).max(-1)
        size = np.maximum(required, widths[first:last].max(0)) * margin + 2
        size = np.maximum(40, np.ceil(size)).astype(int)
        if np.any(size > 4096):
            raise ValueError("A method projects outside the supported display ROI; preserve failure instead of hiding it")
        minimum = np.floor(center - size[:, None] / 2).astype(int)
        result[frame, :, :2] = minimum
        result[frame, :, 2:] = minimum + size[:, None]
    if np.any(result[..., :2] > low) or np.any(result[..., 2:] < high):
        raise ValueError("Common display ROI omitted a selected method or source crop")
    return result


def verify_videos(root, expected):
    rows = []
    for name, (frames, fps) in expected.items():
        path = root / name
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Undecodable video: {path}")
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if list(frame.shape[1::-1]) != size:
                raise ValueError("Video dimensions changed")
            count += 1
        capture.release()
        if count != frames or not np.isclose(actual_fps, fps, rtol=0, atol=1e-6):
            raise ValueError(f"Wrong video coverage/fps: {name}")
        rows.append({"file": name, "decoded_frames": count, "fps": actual_fps,
                     "size": size, "bytes": path.stat().st_size, "sha256": digest(path)})
    return rows


def generated_panel(image, points, valid, admitted, side, hand_edges):
    """Show prior-only generated geometry instead of hiding gated-out guesses."""
    marked = image.copy()
    valid = valid[side] & np.isfinite(points[side]).all(-1)
    admitted = admitted[side] > 0
    for first, last in hand_edges:
        if valid[first] and valid[last]:
            color = (80, 240, 120) if admitted[first] and admitted[last] else (40, 150, 255)
            cv2.line(marked, tuple(np.rint(points[side, first]).astype(int)),
                     tuple(np.rint(points[side, last]).astype(int)), color, 1, cv2.LINE_AA)
    for joint in np.flatnonzero(valid):
        color = (80, 240, 120) if admitted[joint] else (40, 150, 255)
        cv2.circle(marked, tuple(np.rint(points[side, joint]).astype(int)), 2, color, -1, cv2.LINE_AA)
    return marked, int((valid & admitted).sum()), int(valid.sum())


def roi_self_test():
    boxes = np.broadcast_to([10., 20., 80., 90.], (9, 2, 4)).copy()
    points = np.zeros((9, 2, 2, 2))
    points[..., 0, :] = [5, 15]
    points[..., 1, :] = [120, 100]
    points[4, 0, 1] = [180, 100]
    rois = common_display_rois(boxes, [(points, np.ones((9, 2, 2), bool))])
    assert np.all(rois[..., 2] - rois[..., 0] == rois[..., 3] - rois[..., 1])
    assert np.all(rois[..., :2] <= points.min(2)) and np.all(rois[..., 2:] >= points.max(2))
    assert np.all(rois[..., :2] <= boxes[..., :2]) and np.all(rois[..., 2:] >= boxes[..., 2:])
    invalid = np.zeros((9, 2, 1, 2))
    unchanged = common_display_rois(boxes, [(invalid, np.zeros((9, 2, 1), bool))])
    assert np.all(unchanged[..., 0] > -20)
    generated = np.broadcast_to(np.stack([np.arange(21) + 10, np.arange(21) + 20], -1), (2, 21, 2)).copy()
    pixels, admitted, count = generated_panel(np.zeros((100, 100, 3), np.uint8), generated,
        np.ones((2, 21), bool), np.zeros((2, 21)), 0, [(0, 1), (1, 2)])
    assert admitted == 0 and count == 21 and np.any(pixels[..., 2] > 0)
    print(json.dumps({"independent_synthetic_roi_union_and_invalid_mask_test": "passed",
                      "zero_admission_generated_geometry_remains_visible": "passed", "frames": 9}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=(11, 16), required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--review-windows", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-roles", nargs="+", default=["legacy_mediapipe", "fresh_mediapipe", "refined_egohand"])
    parser.add_argument("--ablation-roles", nargs="+", default=["legacy_egohand", "refined_egohand"])
    parser.add_argument("--proposal-roles", nargs="+", default=["fresh_mediapipe", "refined_egohand"])
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior attempts; output must not exist")
    if args.size % 2 or not 128 <= args.size <= 1024:
        parser.error("Use an even panel size from128 to1024")
    started = time.monotonic()
    task, evaluation = read(args.input_manifest), read(args.evaluation)
    if task.get("track") != 1 or task.get("all30_execution_authorized") is not False:
        raise ValueError("Wrong task scope")
    item = next(entry for entry in task["episodes"] if entry["episode"] == args.episode)
    count, fps = item["frames"], item["fps"]
    if evaluation["schema_version"] != "refined_egohand_pilot_evaluation.v1" or evaluation["episode"] != args.episode:
        raise ValueError("Wrong completed native evaluation")
    if evaluation["input_bindings"]["task_manifest"]["sha256"] != digest(args.input_manifest):
        raise ValueError("Evaluation and rendering task inputs differ")
    if evaluation["input_bindings"]["review_windows"]["sha256"] != digest(args.review_windows):
        raise ValueError("Frozen inspection windows differ")
    video = bound({"path": item["video"], "sha256": item["video_sha256"]})
    manifest = read(bound({"path": item["manifest"], "sha256": item["manifest_sha256"]}))
    sys.path.insert(0, str(args.tool_root / "tools"))
    from compare_human_hands import MeshOnly, crop_panel, fitted_text, observations_for_side, panel_header, PANEL_HEADER, HAND_EDGES
    from review_human import Predictions, VideoWriter, draw_prediction
    cv2.setNumThreads(1)
    labels = evaluation["role_labels"]
    selected = list(dict.fromkeys(args.primary_roles + args.ablation_roles))
    expected = np.arange(count)
    K = np.asarray(manifest["camera_intrinsics"], dtype=np.float32)
    predictions, point_sets = {}, []
    for role in selected:
        if not evaluation["native_decoder"][role]["passed"] or evaluation["native_decoder"][role]["frames_freshly_decoded"] != count:
            raise ValueError("All displayed native variants need completed all-frame parity")
        binding = evaluation["native_artifact_bindings"][role]
        for relative, metadata in binding["files"].items():
            bound({"path": str(Path(binding["directory"]) / relative), "sha256": metadata["sha256"]})
        prediction = Predictions(Path(binding["directory"]) / "review_arrays")
        if not np.array_equal(prediction.arrays["frame_indices"], expected) or not np.array_equal(prediction.arrays["camera_intrinsics"], K):
            raise ValueError("Native source frame or camera convention differs")
        joints = np.asarray(prediction.arrays["joints_3d"])[:, HAND_JOINTS]
        projected = joints @ K.T
        if not np.isfinite(projected).all() or np.any(projected[..., 2] <= 0):
            raise ValueError("Invalid native projected geometry")
        projected = projected[..., :2] / projected[..., 2:]
        point_sets.append((projected, np.ones(projected.shape[:-1], bool)))
        predictions[role] = prediction
    observations, generated_geometry = {}, {}
    fresh_role = item["variants"][next(i for i, v in enumerate(item["variants"]) if v["role"] == item["controlled_pair"][0])]["evidence_role"]
    for role in dict.fromkeys([fresh_role] + args.proposal_roles):
        observations[role] = npz(bound(item["evidence"][role]))
        data = observations[role]
        if not np.array_equal(data["frame_indices"], expected) or list(data["side_order"].astype(str)) != ["left", "right"] or str(data["source_video_sha256"].item()) != item["video_sha256"]:
            raise ValueError("Proposal source frame/side mapping differs")
        if role in args.proposal_roles:
            if item["evidence"][role]["kind"] == "generated_proposal" and "proposal_joints3d_camera" in data:
                geometry = np.asarray(data["proposal_joints3d_camera"], dtype=np.float64)
                valid = np.isfinite(geometry).all(-1) & (geometry[..., 2] > 0)
                valid &= data["proposal_camera_geometry_valid"]
                projected = geometry @ K.T
                projected = np.divide(projected[..., :2], projected[..., 2:],
                                      out=np.zeros_like(projected[..., :2]), where=valid[..., None])
                generated_geometry[role] = (projected, valid)
                point_sets.append((projected, valid))
            else:
                point_sets.append((data["joints2d"], data["confidence"] > 0))
    rois = common_display_rois(observations[fresh_role]["crop_boxes"], point_sets)
    args.output.mkdir(parents=True)
    (args.output / "frames").mkdir()
    (args.output / "critical").mkdir()
    np.save(args.output / "display-rois.npy", rois, allow_pickle=False)
    windows = read(args.review_windows)
    intervals = sorted({(row["first_original_frame_id"], row["last_original_frame_id"]) for row in windows["windows"]})
    samples = set(SAMPLES[args.episode])
    for edges in evaluation["support_boundaries"].values():
        for edge in edges:
            for value in (edge["from_original_frame_id"], edge["to_original_frame_id"]):
                samples.update(i for i in (value - 1, value, value + 1) if 0 <= i < count)
    height = 44 + 2 * (args.size + PANEL_HEADER)
    groups = {"hands-comparison": args.primary_roles, "old-new-egohand-comparison": args.ablation_roles,
              "proposal-comparison": args.proposal_roles}
    writers, slow, expected_videos = {}, {}, {}
    for group, roles in groups.items():
        width = args.size * (1 + len(roles))
        writers[group] = VideoWriter(args.output / (group + ".mp4"), (width, height), fps, cv2)
        expected_videos[group + ".mp4"] = (count, fps)
        if group != "proposal-comparison":
            for first, last in intervals:
                name = f"critical/{group}-slow-{first:06d}-{last:06d}.mp4"
                slow[group, first, last] = VideoWriter(args.output / name, (width, height), fps / 3, cv2)
                expected_videos[name] = (last - first + 1, fps / 3)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened() or not np.isclose(capture.get(cv2.CAP_PROP_FPS), fps):
        raise ValueError("Source video unavailable or fps differs")
    frame_id, snapshots = 0, []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if frame_id >= count:
                raise ValueError("Extra source frames")
            native_images = {role: draw_prediction(image, MeshOnly(pred), frame_id, 0., 50000, cv2)[0]
                             for role, pred in predictions.items()}
            for group, roles in groups.items():
                canvas = np.full((height, args.size * (1 + len(roles)), 3), (24, 26, 30), np.uint8)
                cv2.putText(canvas, f"EP{args.episode:02d} | Original frame {frame_id:06d} | {frame_id/fps:.3f}s | One common display ROI",
                            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .44, (240, 240, 240), 1, cv2.LINE_AA)
                for side, side_name in enumerate(("Left", "Right")):
                    y = 44 + side * (args.size + PANEL_HEADER)
                    for column, role in enumerate(["source"] + roles):
                        x = column * args.size
                        admitted = None
                        if role == "source":
                            pixels, label = image, "Source RGB"
                        elif group == "proposal-comparison":
                            data = observations[role]
                            if role in generated_geometry:
                                points, valid = generated_geometry[role]
                                pixels, admitted, generated_count = generated_panel(image, points[frame_id], valid[frame_id],
                                    data["confidence"][frame_id], side, HAND_EDGES)
                            else:
                                pixels, admitted = observations_for_side(image, data["joints2d"][frame_id], data["confidence"][frame_id], side)
                            label = item["evidence"][role].get("label", role)
                        else:
                            pixels, label = native_images[role], labels[role]
                        panel_header(canvas, x, y, args.size, side_name, label)
                        panel = crop_panel(pixels, rois[frame_id, side], args.size)
                        if admitted is not None:
                            fitted_text(panel, f"{admitted}/21 admitted; not visibility truth", 20, scale=.36)
                            if role in generated_geometry:
                                fitted_text(panel, f"{generated_count} generated; orange = prior only", 40, scale=.36)
                        canvas[y + PANEL_HEADER:y + PANEL_HEADER + args.size, x:x + args.size] = panel
                writers[group].write(canvas)
                for (name, first, last), writer in slow.items():
                    if name == group and first <= frame_id <= last:
                        writer.write(canvas)
                if frame_id in samples:
                    name = f"frames/{group}-{frame_id:06d}.jpg"
                    if not cv2.imwrite(str(args.output / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError("Snapshot writing failed")
                    snapshots.append({"original_frame_id": frame_id, "group": group, "file": name,
                                      "sha256": digest(args.output / name)})
            frame_id += 1
            if frame_id % 60 == 0:
                print(f"Compared {frame_id}/{count} original source frames", flush=True)
    finally:
        capture.release()
        for writer in list(writers.values()) + list(slow.values()):
            writer.close()
        for prediction in predictions.values():
            prediction.close()
    if frame_id != count:
        raise ValueError("Incomplete source video")
    videos = verify_videos(args.output, expected_videos)
    report = {"episode": args.episode, "track": 1, "frames": count, "fps": fps,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "wall_seconds": time.monotonic() - started,
        "source_video_sha256": item["video_sha256"], "helper_sha256": digest(__file__),
        "renderer_source_sha256": {name: digest(args.tool_root / "tools" / name) for name in ("compare_human_hands.py", "review_human.py")},
        "input_manifest_sha256": digest(args.input_manifest), "completed_evaluation": {"path": str(args.evaluation), "sha256": digest(args.evaluation)},
        "native_variant_bindings": evaluation["native_artifact_bindings"],
        "proposal_bindings": {role: item["evidence"][role] for role in observations},
        "generated_proposal_display": {role: "Full retargeted camera geometry projected through frozen K; green admitted, orange prior-only, no visibility truth" for role in generated_geometry},
        "original_specialist_geometry_preserved_in_npz_not_replaced_by_admission_mask": True,
        "columns": {group: ["Source RGB"] + [item["evidence"][role].get("label", role) if group == "proposal-comparison" else labels[role] for role in roles] for group, roles in groups.items()},
        "display_roi": {"file": "display-rois.npy", "sha256": digest(args.output / "display-rois.npy"),
                        "same_for_every_method_and_comparison_video": True, "source_input_crops_unchanged": True,
                        "center_policy": "Nine-frame median union centers", "size_policy": "Local maximum size enlarged to contain every current-frame selected native hand/displayed generated proposal and fresh input crop, with8% margin",
                        "noncausal_display_only_filter": True, "per_method_magnification": False,
                        "minimum_square_pixels": int((rois[..., 2] - rois[..., 0]).min()),
                        "maximum_square_pixels": int((rois[..., 2] - rois[..., 0]).max())},
        "source_inspection_windows": windows, "videos": videos, "snapshots": snapshots,
        "browser_playback_checked": False, "static_frames_manually_reviewed": False,
        "quality_accepted": False, "contact_quality_accepted": False, "official_score_available": False,
        "all30_execution_authorized": False,
        "limitations": ["Full video decode checks integrity, not visual quality.",
            "The fresh MP reference is shared with Ego wrist anchoring/gating and is not independent validation.",
            "Admitted generated points do not establish hand visibility or hidden recovery.",
            "Mesh overlay has no object-depth compositing and cannot prove contact or penetration.",
            "Display ROIs differ from model input crops; all methods share each exact saved display square.",
            "Slow motion repeats original frames at10fps; it adds no inferred frames or timing truth."]}
    write(args.output / "summary.json", report)
    sections = []
    for name in expected_videos:
        if name.startswith("critical/"):
            sections.append(f'<p><a href="{html.escape(name)}">{html.escape(name)}</a></p>')
        else:
            sections.append(f'<h2>{html.escape(name)}</h2><video controls preload="metadata" src="{html.escape(name)}"></video>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Refined EgoHand pilot review</title><style>body{font:16px system-ui;background:#10151d;color:#e4edf6;margin:24px}main{max-width:1350px;margin:auto}video{max-width:100%;display:block}a{color:#8fc7ff}p{line-height:1.5}</style><main>'
    page += f'<h1>EP{args.episode:02d}: controlled crop, wrist and visibility pilot</h1><p>Source RGB and all methods share one saved display ROI. Fresh MediaPipe is a shared wrist/gating input, not ground truth. Generated proposals do not establish hidden hand recovery. Quality and contact remain unaccepted.</p><p><a href="summary.json">Video and source receipts</a></p>'
    page += ''.join(sections) + '</main></html>'
    with (args.output / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(page)
    print(json.dumps({"output": str(args.output), "frames": count, "videos_fully_decoded": len(videos),
                      "wall_seconds": report["wall_seconds"], "quality_accepted": False}), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--roi-self-test"]:
        roi_self_test()
    else:
        main()
