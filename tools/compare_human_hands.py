"""Render synchronized hand close-ups from a permitted Track 1 experiment.

Every variant uses the same source frame, camera and crop. The detector panel
shows accepted observations only; an empty panel is labelled explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import time

import cv2
import numpy as np

from review_human import Predictions, VideoWriter, draw_prediction


HAND_EDGES = [(0, start) for start in (1, 5, 9, 13, 17)] + [
    (start + j, start + j + 1) for start in (1, 5, 9, 13, 17) for j in range(3)
]


class MeshOnly:
    """Hide large diagnostic dots while preserving exactly the same geometry."""
    def __init__(self, prediction: Predictions):
        self.prediction = prediction

    def at(self, key, row):
        if key in ("joints_2d", "joints_3d", "skeleton_edges", "confidence"):
            return None
        return self.prediction.at(key, row)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def crop_panel(image: np.ndarray, box: np.ndarray, size: int) -> np.ndarray:
    """Use constant padding so an image-border crop keeps its coordinate frame."""
    left, top, right, bottom = np.rint(box).astype(int)
    if right <= left or bottom <= top or max(right-left, bottom-top) > 4096:
        raise ValueError("Invalid or unexpectedly large observation crop")
    cropped = np.zeros((bottom-top, right-left, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    x0, y0, x1, y1 = max(0, left), max(0, top), min(w, right), min(h, bottom)
    if x1 > x0 and y1 > y0:
        cropped[y0-top:y1-top, x0-left:x1-left] = image[y0:y1, x0:x1]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True,
                        help="Display label=review-array-directory; repeat in display order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, choices=range(30), default=16)
    parser.add_argument("--expected-frames", type=int, default=360)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--mesh-only", action="store_true", help="Hide diagnostic joint dots in reconstruction panels")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output must be empty; existing experiments are immutable")
    if args.size < 128 or args.size > 1024 or args.size % 2:
        raise ValueError("Panel size must be even and between 128 and 1024")
    cv2.setNumThreads(1)
    started = time.perf_counter()
    with np.load(args.observations, allow_pickle=False) as archive:
        obs = {name: archive[name] for name in archive.files}
    frame_ids = obs["frame_indices"]
    if not np.array_equal(frame_ids, np.arange(args.expected_frames)):
        raise ValueError("Observation frame mapping must cover the exact source video")
    points, weights = obs["joints2d"], obs["confidence"]
    boxes = obs["crop_boxes"]
    if points.shape != (len(frame_ids), 2, 21, 2) or weights.shape != points.shape[:-1]:
        raise ValueError("Expected original-pixel observations [T,left/right,21,2]")
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("Observation reliability weights must be finite in [0,1]")
    if np.any(~np.isfinite(points).all(-1) & (weights > 0)):
        raise ValueError("Accepted observations must have finite coordinates")
    if boxes.shape != (len(frame_ids), 2, 4) or not np.isfinite(boxes).all():
        raise ValueError("Expected finite shared crops [T,2,4]")
    if "side_order" in obs and list(obs["side_order"].astype(str)) != ["left", "right"]:
        raise ValueError("Hand rows must be anatomical left then right")
    video_sha = digest(args.video)
    if "source_video_sha256" not in obs or str(obs["source_video_sha256"].item()) != video_sha:
        raise ValueError("Observations must identify this exact source video")
    variants = []
    for specification in args.variant:
        label, path = specification.split("=", 1)
        prediction = Predictions(Path(path))
        if not np.array_equal(prediction.arrays["frame_indices"], frame_ids):
            raise ValueError(f"Variant frame mapping differs: {label}")
        if variants and not np.array_equal(prediction.arrays["camera_intrinsics"], variants[0][1].arrays["camera_intrinsics"]):
            raise ValueError(f"Variant camera differs: {label}")
        variants.append((label, prediction))
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError("Cannot open source video")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source fps")
    if "fps" not in obs or not np.isclose(float(obs["fps"]), fps, atol=1e-6, rtol=0):
        raise ValueError("Observation fps differs from the source video")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frames").mkdir()
    headers = ["Source RGB", "Accepted 2D observations"] + [label for label, _ in variants]
    width, height = args.size * len(headers), 2 * (args.size + 32) + 44
    writer = VideoWriter(args.output / "hands-comparison.mp4", (width, height), fps, cv2)
    count = 0
    snapshots = {0, 40, 60, 110, 120, 210, 240, 284, 285, 286, 287, 308, 359}
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if count >= len(frame_ids):
                raise ValueError("Source video has more frames than the experiment")
            observed = image.copy()
            accepted_by_side = []
            for side in range(2):
                valid = (weights[count, side] > 0) & np.isfinite(points[count, side]).all(axis=1)
                accepted_by_side.append(int(valid.sum()))
                for a, b in HAND_EDGES:
                    if valid[a] and valid[b]:
                        cv2.line(observed, tuple(np.rint(points[count, side, a]).astype(int)),
                                 tuple(np.rint(points[count, side, b]).astype(int)),
                                 (80, 240, 120), 1, cv2.LINE_AA)
                for point in points[count, side, valid]:
                    cv2.circle(observed, tuple(np.rint(point).astype(int)), 2,
                               (80, 240, 120), -1, cv2.LINE_AA)
            images = [image, observed] + [
                draw_prediction(image, MeshOnly(prediction) if args.mesh_only else prediction, count, 0.0, 50000, cv2)[0]
                for _, prediction in variants
            ]
            canvas = np.full((height, width, 3), (24, 26, 30), dtype=np.uint8)
            cv2.putText(canvas, f"Source frame {count:03d} / {len(frame_ids)-1} | {count/fps:.3f} s | Same crops for every variant",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            for side, name in enumerate(("Left hand", "Right hand")):
                y = 44 + side * (args.size + 32)
                for column, (label, panel) in enumerate(zip(headers, images)):
                    x = column * args.size
                    suffix = f" ({accepted_by_side[side]}/21)" if column == 1 else ""
                    cv2.putText(canvas, f"{name}: {label}{suffix}", (x+6, y+21),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.39, (235, 235, 235), 1, cv2.LINE_AA)
                    canvas[y+32:y+32+args.size, x:x+args.size] = crop_panel(panel, boxes[count, side], args.size)
                    if column == 1 and accepted_by_side[side] == 0:
                        cv2.putText(canvas, "NO RELIABLE DETECTION", (x+8, y+54),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 100, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
            if count in snapshots:
                if not cv2.imwrite(str(args.output / "frames" / f"{count:06d}.jpg"), canvas):
                    raise RuntimeError("Cannot save comparison image")
            count += 1
            if count % 60 == 0:
                print(f"Compared {count}/{len(frame_ids)} frames", flush=True)
    finally:
        capture.release()
        writer.close()
        for _, prediction in variants:
            prediction.close()
    if count != args.expected_frames:
        raise ValueError(f"Decoded {count}, expected {args.expected_frames}")
    report = {
        "track": 1, "episode": args.episode, "frames": count, "fps": fps,
        "video_sha256": video_sha, "observations_sha256": digest(args.observations),
        "renderer_sha256": digest(Path(__file__)),
        "variants": [{"label": label, "path": str(prediction.path.resolve())} for label, prediction in variants],
        "crop_contract": "Same observation crop, source frame, original time and camera across all panels",
        "diagnostic_joint_dots_displayed": not args.mesh_only,
        "elapsed_seconds": round(time.perf_counter()-started, 3), "codec": writer.codec,
        "scope": "Image-consistency comparison, not hidden-reference accuracy or official scores",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Episode 16 hand reconstruction comparison</title><style>body{background:#11151b;color:#e6e9ef;font:16px/1.6 system-ui;max-width:1500px;margin:32px auto;padding:0 20px}video,img{width:100%;height:auto}a{color:#8ac8ff}pre{white-space:pre-wrap;overflow-wrap:anywhere}figure{margin:24px 0}</style>
<h1>Episode 16 hand reconstruction comparison</h1><p>All panels show the same source frame and hand crop. Green points are accepted independent 2D observations; cyan surfaces are projected MHR predictions. Missing observations are labelled. These are diagnostics, not official reconstruction scores.</p>
<video controls preload="metadata" src="hands-comparison.mp4"></video><p><a href="summary.json">Experiment summary</a></p>"""
    page = page.replace("Episode 16", f"Episode {args.episode:02d}")
    for frame in sorted(snapshots & set(range(count))):
        page += f'<figure><figcaption>Frame {frame}, {frame/fps:.3f} s</figcaption><img loading="lazy" src="frames/{frame:06d}.jpg" alt="Synchronized hands at frame {frame}"></figure>'
    page += "<details><summary>Provenance</summary><pre>" + html.escape(json.dumps(report, indent=2)) + "</pre></details></html>"
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
