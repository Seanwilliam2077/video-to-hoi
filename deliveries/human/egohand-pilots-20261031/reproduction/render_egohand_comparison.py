"""Render frozen-crop EP11/EP16 comparisons with explicit evidence semantics.

Writes only a new task-local review directory, never fits or selects a result.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


TASK = Path(__file__).resolve().parent
WORKTREE = Path("C:/Users/arthasxu/.codex/worktrees/human-all30/video-to-hoi")
RESULTS = Path("J:/video-to-hoi/runs/egohand-pilots-20261031").resolve()
WINDOWS = {16: [(40, 110), (240, 308)], 11: [(280, 340), (480, 620), (680, 850)]}
SAMPLES = {16: [0, 40, 60, 110, 120, 240, 284, 285, 286, 287, 308, 359],
           11: [0, 280, 340, 400, 480, 511, 512, 559, 560, 620, 680, 800, 803, 850, 876]}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def verify_videos(root, expected):
    reports = []
    for name, (frames, fps) in expected.items():
        path = root / name
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"Undecodable video: {path}")
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (h, w):
                raise ValueError("Video frame dimensions changed")
            count += 1
        capture.release()
        if count != frames or not np.isclose(actual_fps, fps, atol=1e-6, rtol=0):
            raise ValueError(f"Wrong video count/fps: {name}, {count}, {actual_fps}")
        reports.append({"file": name, "decoded_frames": count, "fps": actual_fps,
                        "size": [w, h], "bytes": path.stat().st_size, "sha256": digest(path)})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=(11, 16), required=True)
    parser.add_argument("--candidate", type=Path, required=True, help="EgoHand-driven native MHR result directory")
    parser.add_argument("--smoothed-candidate", type=Path,
                        help="Optional native MHR fit to smoothed proposals; the authorized candidate-raw01 layout also resolves its completed candidate-smoothed01 sibling")
    parser.add_argument("--egohand-observations", type=Path, required=True)
    parser.add_argument("--smoothed-egohand-observations", type=Path,
                        help="Optional separately labelled upstream-smoothed EgoHand proposals")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, default=WORKTREE)
    parser.add_argument("--input-manifest", type=Path, default=TASK / "inputs/manifest.json")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == RESULTS or not output.is_relative_to(RESULTS):
        parser.error("Output must be a child of this isolated pilot results root")
    if output.exists():
        parser.error("Preserve prior attempts; output must not exist")
    if args.size < 128 or args.size > 1024 or args.size % 2:
        parser.error("Use an even panel size from 128 to 1024")
    started = time.monotonic()
    inputs = read(args.input_manifest)
    if inputs.get("all30_execution_authorized") is not False:
        raise ValueError("Wrong task scope")
    item = next(x for x in inputs["episodes"] if x["episode"] == args.episode)
    manifest = read(item["manifest"])
    if manifest["track"] != 1 or manifest["episode"] != args.episode:
        raise ValueError("Wrong source identity")
    if digest(Path(item["manifest"])) != item["manifest_sha256"]:
        raise ValueError("Original manifest changed")
    if digest(Path(item["video"])) != item["video_sha256"]:
        raise ValueError("Original RGB video changed")
    sys.path.insert(0, str(args.tool_root / "tools"))
    from compare_human_hands import MeshOnly, crop_panel, fitted_text, observations_for_side, panel_header, PANEL_HEADER
    from review_human import Predictions, VideoWriter, draw_prediction
    cv2.setNumThreads(1)
    observations = {"MediaPipe accepted image evidence": npz(Path(item["original_observations"])),
                    "EgoHand raw generated proposal": npz(args.egohand_observations)}
    if args.smoothed_egohand_observations:
        observations["EgoHand upstream-smoothed proposal"] = npz(args.smoothed_egohand_observations)
    expected = np.arange(item["frames"])
    boxes = observations["MediaPipe accepted image evidence"]["crop_boxes"]
    for name, obs in observations.items():
        if not np.array_equal(obs["frame_indices"], expected) or list(obs["side_order"].astype(str)) != ["left", "right"]:
            raise ValueError(f"Wrong frame/side convention: {name}")
        if str(obs["source_video_sha256"].item()) != item["video_sha256"]:
            raise ValueError("Observation source identity changed")
        if not np.array_equal(obs["crop_boxes"], boxes):
            raise ValueError("Source crop bounds differ")
    paths = [("SAM baseline", Path(item["baseline"]["path"]).parent / "review_arrays"),
             ("Bounded MediaPipe MHR", Path(item["mediapipe_candidate"]).parent / "review_arrays"),
             ("EgoHand raw-driven MHR", args.candidate / "review_arrays")]
    smoothed_candidate = args.smoothed_candidate
    smoothed_resolution = "explicit_argument" if smoothed_candidate else None
    if smoothed_candidate is None and args.candidate.name == "candidate-raw01":
        sibling = args.candidate.parent / "candidate-smoothed01"
        if (sibling / "report.json").is_file() and (sibling / "review_arrays/joints_3d.npy").is_file():
            smoothed_candidate = sibling
            smoothed_resolution = "completed_authorized_sibling_candidate_smoothed01"
    if smoothed_candidate is not None:
        paths.append(("EgoHand smoothed-driven MHR", smoothed_candidate / "review_arrays"))
    predictions = [(label, Predictions(path)) for label, path in paths]
    K = np.asarray(manifest["camera_intrinsics"], dtype=np.float32)
    for label, pred in predictions:
        if not np.array_equal(pred.arrays["frame_indices"], expected) or not np.array_equal(pred.arrays["camera_intrinsics"], K):
            raise ValueError(f"Variant frame/camera differs: {label}")
    native_bindings = {}
    for label, path in paths:
        files = sorted(path.glob("*.npy"))
        files += [path.parent / "report.json", path.parent / f"episode_{args.episode:06d}_human.npz"]
        native_bindings[label] = {"directory": str(path.parent), "files": {
            p.relative_to(path.parent).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in files}}
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    (output / "critical").mkdir()
    height = 44 + 2 * (args.size + PANEL_HEADER)
    primary_width = args.size * 4
    diagnostic_width = args.size * (1 + len(observations))
    primary = VideoWriter(output / "hands-comparison.mp4", (primary_width, height), item["fps"], cv2)
    smoothed_writer = (VideoWriter(output / "raw-smoothed-mhr-comparison.mp4", (primary_width, height), item["fps"], cv2)
                       if smoothed_candidate is not None else None)
    diagnostic = VideoWriter(output / "proposal-comparison.mp4", (diagnostic_width, height), item["fps"], cv2)
    slow = [(a, b, VideoWriter(output / "critical" / f"slow-{a:06d}-{b:06d}.mp4",
                             (primary_width, height), item["fps"] / 3, cv2)) for a, b in WINDOWS[args.episode]]
    smoothed_slow = ([(a, b, VideoWriter(output / "critical" / f"raw-smoothed-slow-{a:06d}-{b:06d}.mp4",
                                       (primary_width, height), item["fps"] / 3, cv2))
                     for a, b in WINDOWS[args.episode]] if smoothed_candidate is not None else [])
    samples = set(SAMPLES[args.episode])
    for directory in [Path(item["mediapipe_candidate"]).parent, args.candidate] + ([smoothed_candidate] if smoothed_candidate else []):
        for row in read(directory / "report.json").get("support_boundary_motion", {}).get("boundaries", []):
            for key in ("from_original_frame_id", "to_original_frame_id"):
                value = int(row[key])
                samples.update(i for i in (value - 1, value, value + 1) if 0 <= i < item["frames"])
    capture = cv2.VideoCapture(item["video"])
    if not capture.isOpened() or not np.isclose(capture.get(cv2.CAP_PROP_FPS), item["fps"]):
        raise ValueError("Source video unavailable or fps changed")
    count, saved = 0, []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if count >= item["frames"]:
                raise ValueError("Source video has extra frames")
            canvas = np.full((height, primary_width, 3), (24, 26, 30), np.uint8)
            cv2.putText(canvas, f"EP{args.episode:02d} | Original frame {count:06d} | {count/item['fps']:.3f} s | Shared source crops",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1, cv2.LINE_AA)
            mesh_images = [image] + [draw_prediction(image, MeshOnly(pred), count, 0., 50000, cv2)[0] for _, pred in predictions]
            for side, side_name in enumerate(("Left", "Right")):
                y = 44 + side * (args.size + PANEL_HEADER)
                for column, (label, pixels) in enumerate(zip(["Source RGB"] + [name for name, _ in predictions[:3]], mesh_images[:4])):
                    x = column * args.size
                    panel_header(canvas, x, y, args.size, side_name, label)
                    canvas[y + PANEL_HEADER:y + PANEL_HEADER + args.size, x:x + args.size] = crop_panel(pixels, boxes[count, side], args.size)
            primary.write(canvas)
            for first, last, writer in slow:
                if first <= count <= last:
                    writer.write(canvas)
            smoothed_canvas = None
            if smoothed_writer is not None:
                smoothed_canvas = np.full_like(canvas, (24, 26, 30))
                cv2.putText(smoothed_canvas, f"EP{args.episode:02d} | Original frame {count:06d} | {count/item['fps']:.3f} s | Raw vs upstream-smoothed MHR fits",
                            (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .44, (240, 240, 240), 1, cv2.LINE_AA)
                comparison_labels = ["Source RGB", predictions[1][0], predictions[2][0], predictions[3][0]]
                comparison_pixels = [mesh_images[i] for i in (0, 2, 3, 4)]
                for side, side_name in enumerate(("Left", "Right")):
                    y = 44 + side * (args.size + PANEL_HEADER)
                    for column, (label, pixels) in enumerate(zip(comparison_labels, comparison_pixels)):
                        x = column * args.size
                        panel_header(smoothed_canvas, x, y, args.size, side_name, label)
                        smoothed_canvas[y + PANEL_HEADER:y + PANEL_HEADER + args.size, x:x + args.size] = crop_panel(pixels, boxes[count, side], args.size)
                smoothed_writer.write(smoothed_canvas)
                for first, last, writer in smoothed_slow:
                    if first <= count <= last:
                        writer.write(smoothed_canvas)
            diag = np.full((height, diagnostic_width, 3), (24, 26, 30), np.uint8)
            cv2.putText(diag, f"Original frame {count:06d} | MP: image detector; EgoHand: generated 3D projection, not truth",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1, cv2.LINE_AA)
            for side, side_name in enumerate(("Left", "Right")):
                y = 44 + side * (args.size + PANEL_HEADER)
                panel_header(diag, 0, y, args.size, side_name, "Source RGB")
                diag[y + PANEL_HEADER:y + PANEL_HEADER + args.size, :args.size] = crop_panel(image, boxes[count, side], args.size)
                for column, (label, obs) in enumerate(observations.items(), 1):
                    marked, accepted = observations_for_side(image, obs["joints2d"][count], obs["confidence"][count], side)
                    x = column * args.size
                    panel_header(diag, x, y, args.size, side_name, label)
                    panel = crop_panel(marked, boxes[count, side], args.size)
                    fitted_text(panel, f"{accepted}/21 admitted points", 20, scale=.41)
                    if not accepted:
                        fitted_text(panel, "NO ADMITTED TARGET", 40, scale=.41, color=(80, 100, 255))
                    diag[y + PANEL_HEADER:y + PANEL_HEADER + args.size, x:x + args.size] = panel
            diagnostic.write(diag)
            if count in samples:
                path = output / "frames" / f"{count:06d}.jpg"
                if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError("Failed primary comparison image")
                proposal = output / "frames" / f"proposal-{count:06d}.jpg"
                if not cv2.imwrite(str(proposal), diag, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError("Failed diagnostic comparison image")
                saved.append({"original_frame_id": count, "primary": path.relative_to(output).as_posix(),
                              "primary_sha256": digest(path), "proposal": proposal.relative_to(output).as_posix(),
                              "proposal_sha256": digest(proposal)})
                if smoothed_canvas is not None:
                    smooth_path = output / "frames" / f"raw-smoothed-{count:06d}.jpg"
                    if not cv2.imwrite(str(smooth_path), smoothed_canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError("Failed raw/smoothed native comparison image")
                    saved[-1].update(smoothed_native=smooth_path.relative_to(output).as_posix(),
                                     smoothed_native_sha256=digest(smooth_path))
            count += 1
            if count % 60 == 0:
                print(f"Compared {count}/{item['frames']} source frames", flush=True)
    finally:
        capture.release()
        primary.close()
        if smoothed_writer is not None:
            smoothed_writer.close()
        diagnostic.close()
        for _, _, writer in slow + smoothed_slow:
            writer.close()
        for _, prediction in predictions:
            prediction.close()
    if count != item["frames"]:
        raise ValueError("Incomplete source video")
    expected_videos = {"hands-comparison.mp4": (count, item["fps"]), "proposal-comparison.mp4": (count, item["fps"])}
    expected_videos.update({f"critical/slow-{a:06d}-{b:06d}.mp4": (b - a + 1, item["fps"] / 3) for a, b in WINDOWS[args.episode]})
    if smoothed_writer is not None:
        expected_videos["raw-smoothed-mhr-comparison.mp4"] = (count, item["fps"])
        expected_videos.update({f"critical/raw-smoothed-slow-{a:06d}-{b:06d}.mp4": (b - a + 1, item["fps"] / 3) for a, b in WINDOWS[args.episode]})
    qa = verify_videos(output, expected_videos)
    report = {"episode": args.episode, "track": 1, "frames": count, "fps": item["fps"],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_video_sha256": item["video_sha256"], "helper_sha256": digest(Path(__file__)),
        "renderer_source_sha256": {n: digest(args.tool_root / "tools" / n) for n in ("compare_human_hands.py", "review_human.py")},
        "input_manifest_sha256": digest(args.input_manifest),
        "native_variant_bindings": native_bindings,
        "proposal_bindings": {"raw": {"path": str(args.egohand_observations), "sha256": digest(args.egohand_observations)},
                              "smoothed": ({"path": str(args.smoothed_egohand_observations), "sha256": digest(args.smoothed_egohand_observations)} if args.smoothed_egohand_observations else None)},
        "primary_columns": ["Source RGB"] + [label for label, _ in paths[:3]],
        "raw_smoothed_native_columns": (["Source RGB", predictions[1][0], predictions[2][0], predictions[3][0]] if smoothed_writer else None),
        "smoothed_native_candidate": str(smoothed_candidate) if smoothed_candidate else None,
        "smoothed_candidate_resolution": smoothed_resolution,
        "diagnostic_columns": ["Source RGB"] + list(observations),
        "all_frames_and_shared_crops": True, "camera_estimated_not_calibrated": True,
        "videos": qa, "snapshots": saved,
        "slow_windows": [{"first_original_frame_id": a, "last_original_frame_id": b, "speed_fraction": 1/3,
                          "file": f"critical/slow-{a:06d}-{b:06d}.mp4"} for a, b in WINDOWS[args.episode]],
        "browser_playback_checked": False, "static_frames_manually_reviewed": False,
        "quality_accepted": False, "contact_quality_accepted": False, "official_score_available": False,
        "all30_execution_authorized": False,
        "limitations": ["Full decode is video integrity, not visual quality acceptance.",
                        "EgoHand panels show generated geometry projections; admission weights are not calibrated visibility.",
                        "Mesh overlay does not composite object depth and cannot verify physical contact or penetration.",
                        "Source windows are inspection aids, not ground-truth grasp-phase labels."]}
    write(output / "summary.json", report)
    slow_html = "".join(f'<p>Original frames {a}-{b}, one-third speed</p><video controls preload="metadata" src="critical/slow-{a:06d}-{b:06d}.mp4"></video>' for a, b in WINDOWS[args.episode])
    smoothed_html = ""
    if smoothed_writer is not None:
        smoothed_html = '<h2>Raw versus upstream-smoothed proposals, both fitted into fixed MHR</h2><p>Source RGB, current bounded MediaPipe MHR, raw EgoHand-driven MHR, upstream-smoothed EgoHand-driven MHR. Both are retained; no winner is automatically selected.</p><video controls preload="metadata" src="raw-smoothed-mhr-comparison.mp4"></video>'
        smoothed_html += "".join(f'<p>Original frames {a}-{b}, one-third speed</p><video controls preload="metadata" src="critical/raw-smoothed-slow-{a:06d}-{b:06d}.mp4"></video>' for a, b in WINDOWS[args.episode])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EP{args.episode:02d} EgoHand comparison</title><style>body{{font:16px/1.6 system-ui;max-width:1500px;margin:30px auto;padding:0 20px;background:#11151c;color:#e9eef5}}a{{color:#90c4ff}}video,img{{width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style><h1>EP{args.episode:02d}: EgoHand versus bounded MediaPipe</h1><p>All {count} original frames, 30 fps, identical source crops and estimated camera. Primary panels: Source RGB, fixed-identity SAM baseline, current bounded MediaPipe MHR, EgoHand-driven MHR.</p><p>These are experimental comparisons. Smoothness and decoder agreement do not prove hidden grasp, contact, scale or official scores. No all-30 batch is authorized.</p><video controls preload="metadata" src="hands-comparison.mp4"></video><h2>Original-frame transition views</h2>{slow_html}<h2>Detector evidence and generated proposals</h2><p>MediaPipe is an image detector; EgoHand is a model-generated 3D proposal projected through the same estimated camera. Neither is ground truth.</p><video controls preload="metadata" src="proposal-comparison.mp4"></video><p><a href="summary.json">Video integrity and input bindings</a></p><details><summary>Report</summary><pre>{html.escape(json.dumps(report, indent=2))}</pre></details></html>'''
    page = page.replace('<h2>Detector evidence and generated proposals</h2>', smoothed_html + '<h2>Detector evidence and generated proposals</h2>')
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(output), "frames": count, "videos_fully_decoded": len(qa),
                      "wall_seconds": time.monotonic() - started, "quality_accepted": False}), flush=True)


if __name__ == "__main__":
    main()
