"""Build a frame-exact, CPU-only human reconstruction review for a Track 1 clip.

Install numpy and opencv-python-headless; imageio-ffmpeg is optional for H.264.
Example:
    python tools/review_human.py --video episode_000016.mp4 --predictions human.npz \
        --episode 16 --output runs/human-16/review

Predictions are an NPZ or a directory of NPY arrays (the latter are memory mapped):
    frame_indices [T]                 decoded, zero-based source frame indices
    joints_2d [T,J,2]                  original video pixels, optional
    joints_3d [T,J,3]                  OpenCV camera coordinates, metres, optional
    camera_intrinsics [3,3] or [T,3,3]  required when projecting 3D arrays
    vertices [T,V,3]                   OpenCV camera coordinates, metres, optional
    faces [F,3]                       triangle indices, optional
    skeleton_edges [E,2]              joint indices, optional
    confidence [T,J]                  optional, thresholds affect display only

At least one of joints_2d, joints_3d or vertices is required. No implicit root
translation, coordinate flip, image crop transform or unit conversion is applied.
Use the fitted MHR forward output to review fitting results with the same tool.
This tool does not load model weights or compute hidden-reference challenge scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

ALIASES = {
    "joints_2d": ("joints_2d", "pred_keypoints_2d"),
    "joints_3d": ("joints_3d", "pred_keypoints_3d"),
    "vertices": ("vertices", "vertices_m", "pred_vertices"),
    "camera_intrinsics": ("camera_intrinsics", "intrinsics", "K"),
    "confidence": ("confidence", "keypoint_confidence"),
}


class Predictions:
    """Validated named arrays with an explicit source-frame mapping."""

    def __init__(self, path: Path):
        self.path = path
        self.archive = None
        if path.is_dir():
            self.arrays = {p.stem: np.load(p, mmap_mode="r", allow_pickle=False)
                           for p in path.glob("*.npy")}
        else:
            self.archive = np.load(path, allow_pickle=False)
            self.arrays = {key: self.archive[key] for key in self.archive.files}
        for key, aliases in ALIASES.items():
            for alias in aliases:
                if alias in self.arrays:
                    self.arrays[key] = self.arrays[alias]
                    break
        if "frame_indices" not in self.arrays:
            raise ValueError("Predictions must include explicit frame_indices.")
        indices = self.arrays["frame_indices"]
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("frame_indices must be a one-dimensional integer array.")
        if len(indices) == 0 or np.any(indices < 0) or len(np.unique(indices)) != len(indices):
            raise ValueError("frame_indices must be nonempty, unique, and nonnegative.")
        self.lookup = {int(frame): row for row, frame in enumerate(indices)}
        self.count = len(indices)
        if not any(key in self.arrays for key in ("joints_2d", "joints_3d", "vertices")):
            raise ValueError("Supply joints_2d, joints_3d, or vertices.")
        for key, dimensions in (("joints_2d", 2), ("joints_3d", 3), ("vertices", 3)):
            if key in self.arrays:
                a = self.arrays[key]
                if a.ndim != 3 or a.shape[0] != self.count or a.shape[-1] != dimensions:
                    raise ValueError(f"{key} must have shape [T,N,{dimensions}].")
        if "joints_2d" in self.arrays and "joints_3d" in self.arrays:
            if self.arrays["joints_2d"].shape[:2] != self.arrays["joints_3d"].shape[:2]:
                raise ValueError("2D and 3D joint counts must match.")
        joints = self.arrays.get("joints_2d", self.arrays.get("joints_3d"))
        if "confidence" in self.arrays:
            if joints is None or self.arrays["confidence"].shape != joints.shape[:2]:
                raise ValueError("confidence must match [T,J] of the joint arrays.")
        if "joints_3d" in self.arrays or "vertices" in self.arrays:
            if "camera_intrinsics" not in self.arrays:
                raise ValueError("3D predictions require camera_intrinsics.")
        if "camera_intrinsics" in self.arrays:
            k = self.arrays["camera_intrinsics"]
            if k.shape not in ((3, 3), (self.count, 3, 3)) or not np.isfinite(k).all():
                raise ValueError("camera_intrinsics must be finite [3,3] or [T,3,3].")
            if np.any(k[..., 0, 0] <= 0) or np.any(k[..., 1, 1] <= 0):
                raise ValueError("Focal lengths must be positive.")
            if not np.allclose(k[..., 2, :], [0, 0, 1]):
                raise ValueError("camera_intrinsics must have homogeneous row [0,0,1].")
        self._validate_indices("skeleton_edges", 2, None if joints is None else joints.shape[1])
        vertices = self.arrays.get("vertices")
        self._validate_indices("faces", 3, None if vertices is None else vertices.shape[1])

    def _validate_indices(self, key: str, columns: int, count: int | None) -> None:
        if key not in self.arrays:
            return
        a = self.arrays[key]
        if count is None or a.ndim != 2 or a.shape[1] != columns:
            raise ValueError(f"{key} must be [N,{columns}] and have matching geometry.")
        if not np.issubdtype(a.dtype, np.integer) or np.any(a < 0) or np.any(a >= count):
            raise ValueError(f"{key} contains invalid geometry indices.")

    def at(self, key: str, row: int):
        array = self.arrays.get(key)
        if array is None:
            return None
        if key in ("faces", "skeleton_edges") or (key == "camera_intrinsics" and array.ndim == 2):
            return array
        return array[row]

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()


def project(points: np.ndarray, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project positive-depth OpenCV points without altering their frame or units."""
    points = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    pixels = np.full((len(points), 2), np.nan)
    h = points[valid] @ camera.T
    pixels[valid] = h[:, :2] / h[:, 2, None]
    valid &= np.isfinite(pixels).all(axis=1)
    return pixels, valid


def average_length(vectors: np.ndarray | None) -> float | None:
    if vectors is None:
        return None
    valid = np.isfinite(vectors).all(axis=-1)
    return float(np.linalg.norm(vectors[valid], axis=-1).mean()) if np.any(valid) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contact_sheet_indices(total: int, predicted, budget: int = 9) -> np.ndarray:
    """Show real reconstructions in sparse smoke runs, rather than empty time samples."""
    available = np.asarray(sorted(frame for frame in predicted if 0 <= frame < total), dtype=int)
    if len(available):
        selected = np.linspace(0, len(available) - 1, min(budget, len(available))).round().astype(int)
        return available[selected]
    return np.unique(np.linspace(0, total - 1, min(budget, total)).round().astype(int))


def ffmpeg_binary() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


class VideoWriter:
    def __init__(self, path: Path, size: tuple[int, int], fps: float, cv2):
        self.process = None
        self.writer = None
        self.log = None
        binary = ffmpeg_binary()
        if binary:
            self.log = path.with_suffix(".encoding.log").open("wb")
            self.process = subprocess.Popen(
                [binary, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                 "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps),
                 "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log,
            )
            self.codec = "H.264"
        else:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not self.writer.isOpened():
                raise RuntimeError("Could not open video writer. Install imageio-ffmpeg for H.264.")
            self.codec = "MPEG-4 Part 2; exact-frame JPEG viewer works independently"

    def write(self, frame: np.ndarray) -> None:
        if self.process is not None:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.writer.write(frame)

    def close(self) -> None:
        if self.process is not None:
            self.process.stdin.close()
            result = self.process.wait()
            self.log.close()
            if result:
                raise RuntimeError(f"ffmpeg encoding failed (exit {result}); see encoding log.")
        if self.writer is not None:
            self.writer.release()


def draw_prediction(image: np.ndarray, predictions: Predictions, row: int, confidence: float,
                    max_faces: int, cv2) -> tuple[np.ndarray, dict, np.ndarray | None, np.ndarray | None]:
    overlay = image.copy()
    height, width = image.shape[:2]
    k = predictions.at("camera_intrinsics", row)
    joints3d = predictions.at("joints_3d", row)
    if joints3d is not None:
        joints3d = np.asarray(joints3d, dtype=np.float64)
    joints2d = predictions.at("joints_2d", row)
    if joints2d is None and joints3d is not None:
        joints2d, _ = project(joints3d, k)
    vertices = predictions.at("vertices", row)
    geometry_valid = None
    projected = None
    if vertices is not None:
        projected, geometry_valid = project(vertices, k)
        faces = predictions.at("faces", row)
        if faces is not None and len(faces):
            selected = faces[::max(1, math.ceil(len(faces) / max_faces))]
            selected = selected[geometry_valid[selected].all(axis=1)]
            if len(selected):
                screen = projected[selected]
                # Exclude huge projections close to the camera plane; this is a review renderer.
                bounded = (np.abs(screen) < max(width, height) * 8).all(axis=(1, 2))
                selected, screen = selected[bounded], screen[bounded]
                order = np.argsort(vertices[selected, 2].mean(axis=1))[::-1]
                surface = overlay.copy()
                triangles = vertices[selected]
                normals = np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0])
                norms = np.linalg.norm(normals, axis=1).astype(np.float64)
                shades = 0.4 + 0.6 * np.abs(normals[:, 2].astype(np.float64)) / np.maximum(norms, 1e-9)
                colors_float = shades[:, None] * np.array([225, 205, 65], dtype=np.float64)
                colors = colors_float.astype(np.int32)
                # Batched norms can differ slightly from the scalar dot product.
                # Preserve truncation at color boundaries with the original expression.
                near_boundary = (np.abs(colors_float - np.rint(colors_float)) < 1e-3).any(axis=1)
                for index in np.flatnonzero(near_boundary):
                    normal = normals[index]
                    shade = 0.4 + 0.6 * abs(normal[2]) / max(float(np.linalg.norm(normal)), 1e-9)
                    colors[index] = tuple(int(v * shade) for v in (225, 205, 65))
                pixels = np.rint(screen).astype(np.int32)
                for index in order:
                    color = tuple(int(v) for v in colors[index])
                    cv2.fillConvexPoly(surface, pixels[index], color, lineType=cv2.LINE_AA)
                overlay = cv2.addWeighted(surface, 0.60, overlay, 0.40, 0)
        else:
            valid_ids = np.flatnonzero(geometry_valid)
            for point in projected[valid_ids[::max(1, len(valid_ids) // 3000)]]:
                if 0 <= point[0] < width and 0 <= point[1] < height:
                    cv2.circle(overlay, tuple(np.rint(point).astype(int)), 1, (225, 205, 65), -1)
    valid = np.zeros(0, dtype=bool)
    inside = valid
    if joints2d is not None:
        joints2d = np.asarray(joints2d, dtype=np.float64)
        valid = np.isfinite(joints2d).all(axis=1)
        inside = valid & (joints2d[:, 0] >= 0) & (joints2d[:, 0] < width)
        inside &= (joints2d[:, 1] >= 0) & (joints2d[:, 1] < height)
        visible = inside.copy()
        scores = predictions.at("confidence", row)
        if scores is not None:
            visible &= np.isfinite(scores) & (scores >= confidence)
        edges = predictions.at("skeleton_edges", row)
        if edges is not None:
            for a, b in edges:
                if visible[a] and visible[b]:
                    cv2.line(overlay, tuple(np.rint(joints2d[a]).astype(int)),
                             tuple(np.rint(joints2d[b]).astype(int)), (80, 210, 255), 2, cv2.LINE_AA)
        radius = max(2, round(width / 500))
        for index in np.flatnonzero(visible):
            cv2.circle(overlay, tuple(np.rint(joints2d[index]).astype(int)), radius,
                       (45, 140, 255), -1, cv2.LINE_AA)
    depth = joints3d if joints3d is not None else vertices
    finite_depth = None if depth is None else depth[np.isfinite(depth).all(axis=1), 2]
    diagnostic = {
        "joint_count": len(valid), "finite_joints_2d": int(valid.sum()),
        "joints_in_image": int(inside.sum()),
        "in_image_fraction": float(inside.sum() / len(inside)) if len(inside) else None,
        "median_depth_m": float(np.median(finite_depth)) if finite_depth is not None and len(finite_depth) else None,
        "positive_depth_fraction": float((finite_depth > 0).mean()) if finite_depth is not None and len(finite_depth) else None,
        "vertex_count": 0 if vertices is None else len(vertices),
        "projectable_vertices": None if geometry_valid is None else int(geometry_valid.sum()),
    }
    return overlay, diagnostic, joints2d, joints3d


def render_review(args) -> dict:
    import cv2

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty; use a new run directory to preserve reviews.")
    predictions = Predictions(args.predictions)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        predictions.close()
        raise ValueError(f"Cannot decode video: {args.video}")
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "frames"
    image_dir.mkdir()
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        predictions.close()
        raise ValueError("Video has no valid frame rate; provide a correctly timestamped input video.")
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    diagnostics, previous = [], []
    writer = None
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            row = predictions.lookup.get(index)
            diagnostic = {"frame_index": index, "time_seconds": index / fps, "prediction_present": row is not None}
            if row is not None:
                overlay, values, points2d, points3d = draw_prediction(
                    frame, predictions, row, args.confidence, args.max_faces, cv2)
                diagnostic.update(values)
                for key in ("mean_joint_step_px", "mean_joint_second_difference_px", "mean_joint_second_difference_m"):
                    diagnostic[key] = None
                if previous and previous[-1][0] == index - 1:
                    if points2d is not None and previous[-1][1] is not None:
                        diagnostic["mean_joint_step_px"] = average_length(points2d - previous[-1][1])
                    if len(previous) == 2 and previous[0][0] == index - 2:
                        for position, current, name in ((1, points2d, "mean_joint_second_difference_px"),
                                                        (2, points3d, "mean_joint_second_difference_m")):
                            if current is not None and all(p[position] is not None for p in previous):
                                diagnostic[name] = average_length(current - 2 * previous[-1][position] + previous[0][position])
                previous.append((index, points2d, points3d))
                previous = previous[-2:]
            else:
                overlay = frame.copy()
                previous = []
                cv2.rectangle(overlay, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (60, 60, 240), 10)
                cv2.putText(overlay, "NO PREDICTION FOR THIS FRAME", (25, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.5, frame.shape[1] / 1300), (60, 60, 240), 2, cv2.LINE_AA)
            diagnostics.append(diagnostic)
            scale = min(1.0, args.panel_width / frame.shape[1])
            panel_width = round(frame.shape[1] * scale) // 2 * 2
            panel_height = round(frame.shape[0] * scale) // 2 * 2
            panels = [cv2.resize(a, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
                      for a in (frame, overlay)]
            canvas = np.zeros((panel_height + 48, panel_width * 2, 3), np.uint8)
            canvas[48:] = np.concatenate(panels, axis=1)
            cv2.putText(canvas, f"Input | Frame {index:06d} | {index / fps:.3f} s", (12, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 233, 238), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Reconstruction overlay" if row is not None else "Missing reconstruction",
                        (panel_width + 12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                        (230, 233, 238) if row is not None else (100, 100, 255), 1, cv2.LINE_AA)
            if writer is None:
                writer = VideoWriter(args.output / "comparison.mp4", (canvas.shape[1], canvas.shape[0]), fps, cv2)
            writer.write(canvas)
            if not cv2.imwrite(str(image_dir / f"{index:06d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError(f"Failed to write frame {index}.")
            index += 1
            if index % 60 == 0:
                print(f"Reviewed {index} decoded frames", flush=True)
    finally:
        capture.release()
        predictions.close()
        if writer is not None:
            writer.close()
    if index == 0:
        raise ValueError("Video decoded no frames.")
    missing = [r["frame_index"] for r in diagnostics if not r["prediction_present"]]
    outside = sorted(frame for frame in predictions.lookup if frame >= index)
    selection = contact_sheet_indices(index, predictions.lookup)
    fields = list(dict.fromkeys(key for row in diagnostics for key in row))
    with (args.output / "diagnostics.csv").open("w", encoding="utf-8", newline="") as file:
        table = csv.DictWriter(file, fieldnames=fields)
        table.writeheader()
        table.writerows(diagnostics)
    summary = {
        "title": args.title or ("Synthetic human review" if args.synthetic else f"Track 1 episode {args.episode:06d}"),
        "dataset": "independent_synthetic" if args.synthetic else "track_1",
        "episode": None if args.synthetic else args.episode,
        "source_video": str(args.video.resolve()), "source_video_sha256": sha256(args.video),
        "prediction_source": str(args.predictions.resolve()),
        "prediction_sha256": ({p.name: sha256(p) for p in sorted(args.predictions.glob("*.npy"))}
                              if args.predictions.is_dir() else sha256(args.predictions)),
        "coordinate_convention": "OpenCV camera: x right, y down, z forward; 3D metres; 2D original video pixels",
        "decoded_frames": index, "video_metadata_frames": expected_frames, "fps": fps,
        "prediction_frames": predictions.count, "missing_frames": missing,
        "prediction_frames_outside_video": outside,
        "coverage_fraction": (index - len(missing)) / index,
        "decode_count_matches_metadata": expected_frames <= 0 or expected_frames == index,
        "codec": writer.codec,
        "contact_sheet_frames": selection.tolist(),
        "notice": "Input-consistency diagnostics only. No hidden-reference challenge scores are computed."
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    payload = {"summary": summary, "frames": diagnostics}
    template = (Path(__file__).parent / "human_review" / "template.html").read_text(encoding="utf-8")
    html = template.replace("__REVIEW_DATA__", json.dumps(payload, allow_nan=False).replace("</", "<\\/"))
    (args.output / "index.html").write_text(html, encoding="utf-8")
    tiles = []
    for selected in selection:
        image = cv2.imread(str(image_dir / f"{selected:06d}.jpg"))
        tile_width = 720
        tiles.append(cv2.resize(image, (tile_width, round(image.shape[0] * tile_width / image.shape[1]))))
    columns = min(3, len(tiles))
    sheet = np.full((math.ceil(len(tiles) / columns) * tiles[0].shape[0] + 48, columns * 720, 3), 20, np.uint8)
    cv2.putText(sheet, f"Prediction coverage: {index - len(missing)} / {index} decoded frames | Input / overlay",
                (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)
    for i, tile in enumerate(tiles):
        y, x = (i // columns) * tile.shape[0] + 48, (i % columns) * 720
        sheet[y:y + tile.shape[0], x:x + 720] = tile
    cv2.imwrite(str(args.output / "contact-sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(json.dumps({
        "review": str((args.output / "index.html").resolve()),
        "decoded_frames": index,
        "prediction_frames": predictions.count,
        "missing_frame_count": len(missing),
        "prediction_frames_outside_video": outside,
        "decode_count_matches_metadata": summary["decode_count_matches_metadata"],
        "contact_sheet_frames": selection.tolist(),
        "codec": writer.codec,
    }, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    provenance = parser.add_mutually_exclusive_group(required=True)
    provenance.add_argument("--episode", type=int, choices=range(30), help="Caller-confirmed Track 1 source episode")
    provenance.add_argument("--synthetic", action="store_true", help="Independently generated synthetic fixture")
    parser.add_argument("--title")
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.0, help="Display threshold; does not remove diagnostic points")
    parser.add_argument("--max-faces", type=int, default=50000, help="Maximum CPU-rendered mesh faces per frame")
    args = parser.parse_args()
    if args.panel_width < 160 or args.max_faces < 1 or not np.isfinite(args.confidence):
        parser.error("Use panel-width >=160, max-faces >=1, and a finite confidence threshold.")
    try:
        summary = render_review(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Review failed: {exc}\n")
    if summary["missing_frames"] or summary["prediction_frames_outside_video"] or not summary["decode_count_matches_metadata"]:
        parser.exit(2, "Review written with coverage/decoding issues; inspect summary.json.\n")


if __name__ == "__main__":
    main()
