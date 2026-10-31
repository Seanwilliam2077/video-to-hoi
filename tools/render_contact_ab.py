"""Render evaluated EP16 A/B geometry with one shared display and Z-buffer."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v2hoi.contact_review import fixed_union_roi, project, rasterize_meshes
from v2hoi.hand_contact import phase_weights

SAMPLES = [0, 40, 60, 70, 75, 90, 110, 120, 180, 210, 240, 250, 251, 260, 280, 285, 287, 290, 308, 359]
WINDOWS = [(40, 110), (230, 300)]
HAND_JOINTS = np.asarray([
    [78, 97, 98, 99, 100, 92, 93, 94, 95, 88, 89, 90, 91, 84, 85, 86, 87, 80, 81, 82, 83],
    [42, 61, 62, 63, 64, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, 44, 45, 46, 47]], dtype=int)
_WORKER = None


def initialize_renderer(paths, object_path, camera, roi, size):
    """Workers map immutable arrays; no model inference or geometry edits."""
    global _WORKER
    import cv2
    cv2.setNumThreads(1)
    arrays = [{name: np.load(Path(path) / "review_arrays" / f"{name}.npy", mmap_mode="r", allow_pickle=False)
               for name in ("vertices_m", "faces")} for path in paths]
    with np.load(object_path, allow_pickle=False) as archive:
        obj = {name: archive[name] for name in ("vertices", "faces", "object_to_camera")}
    posed = np.einsum("vi,tji->tvj", obj["vertices"], obj["object_to_camera"][:, :3, :3]) + obj["object_to_camera"][:, None, :3, 3]
    _WORKER = (arrays, posed, obj["faces"], camera, roi, size)


def render_layers(frame_and_source):
    frame, source = frame_and_source
    arrays, posed, object_faces, camera, roi, size = _WORKER
    tick = time.monotonic()
    overlays = []
    for data in arrays:
        colored, _, identities = rasterize_meshes([
            (data["vertices_m"][frame], data["faces"], [230, 195, 60], 1),
            (posed[frame], object_faces, [70, 170, 245], 2)], camera, roi, size)
        marked = source.copy()
        coverage = identities > 0
        marked[coverage] = np.clip(.70*colored[coverage] + .30*source[coverage], 0, 255).astype(np.uint8)
        overlays.append(marked)
    return overlays, time.monotonic()-tick, os.getpid()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bound(row):
    path = Path(row["path"])
    if digest(path) != row["sha256"]:
        raise ValueError(f"Frozen input changed: {path}")
    return path


def write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_videos(root, expected, cv2):
    rows = []
    for name, (count, fps) in expected.items():
        path = root / name
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot decode comparison: {name}")
        decoded = 0
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if list(frame.shape[1::-1]) != size:
                raise ValueError("Comparison dimensions changed")
            decoded += 1
        cap.release()
        if decoded != count or not np.isclose(actual_fps, fps, rtol=0, atol=1e-6):
            raise ValueError(f"Wrong video frame count or rate: {name}")
        rows.append({"file": name, "frames_fully_decoded": decoded, "fps": actual_fps,
                     "size": size, "bytes": path.stat().st_size, "sha256": digest(path)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous attempts; output must be absent")
    if args.size % 2 or not 128 <= args.size <= 768:
        parser.error("Use an even panel size from128 to768")
    if not 1 <= args.workers <= 8:
        parser.error("Use1 to8 bounded CPU renderer processes")
    import cv2
    from review_human import VideoWriter
    cv2.setNumThreads(1)
    started = time.monotonic()
    spec, evaluated = read(args.input_manifest), read(args.evaluation)
    if spec.get("track") != 1 or spec.get("episode") != 16 or spec.get("all30_execution_authorized") is not False:
        raise ValueError("Only the authorized Track 1 EP16 A/B is supported")
    if evaluated["input_manifest_sha256"] != digest(args.input_manifest) or evaluated["frames"] != 360:
        raise ValueError("Completed evaluation binds different inputs")
    for role in ("baseline", "a", "b"):
        parity = evaluated["native_decoder"][role]
        if not parity["passed"] or parity["frames_freshly_decoded"] != 360:
            raise ValueError("Every displayed variant requires fresh all-frame native parity")
    video = bound(spec["source_video"])
    manifest = read(bound(spec["manifest"]))
    if manifest["video_sha256"] != evaluated["source_video_sha256"] or manifest["video_frames"] != 360:
        raise ValueError("Original source identity changed")
    K = np.asarray(manifest["camera_intrinsics"], dtype=np.float32)
    arrays, all_points, native_bindings = {}, [], {}
    for role in ("a", "b"):
        root = bound(spec[role]).parent
        arrays[role] = {key: np.load(root / "review_arrays" / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                        for key in ("vertices_m", "joints_3d", "faces", "frame_indices", "camera_intrinsics")}
        data = arrays[role]
        if not np.array_equal(data["frame_indices"], np.arange(360)) or not np.array_equal(data["camera_intrinsics"], K):
            raise ValueError("Native display frame/camera mismatch")
        if data["vertices_m"].shape != (360, 18439, 3) or data["joints_3d"].shape != (360, 127, 3):
            raise ValueError("Native display topology differs")
        native_bindings[role] = {name: digest(root / "review_arrays" / f"{name}.npy") for name in data}
        if any(evaluated["native_review_array_sha256"][role].get(name) != value
               for name, value in native_bindings[role].items()):
            raise ValueError("Displayed native arrays differ from the completed evaluation")
        all_points.append(project(np.asarray(data["joints_3d"])[:, HAND_JOINTS], K))
    with np.load(bound(spec["object"]), allow_pickle=False) as archive:
        obj = {key: archive[key] for key in archive.files}
    if not np.array_equal(obj["frame_indices"], np.arange(360)) or not np.array_equal(obj["camera_intrinsics"], K):
        raise ValueError("Prepared object must share the native camera and original IDs")
    transformed = np.einsum("vi,tji->tvj", obj["vertices"], obj["object_to_camera"][:, :3, :3]) + obj["object_to_camera"][:, None, :3, 3]
    all_points.append(project(transformed, K))
    with np.load(bound(spec["fresh_reference"]), allow_pickle=False) as archive:
        boxes = archive["crop_boxes"]
    all_points += [boxes[..., :2], boxes[..., 2:]]
    roi = fixed_union_roi(all_points)
    phases_path = bound(spec["phases"])
    phases = read(phases_path)
    weights = phase_weights(phases, 360) * obj["pose_usable"][:, None]
    before = {"manifest": digest(args.input_manifest), "evaluation": digest(args.evaluation),
              "renderer": digest(__file__), "rasterizer": digest(ROOT / "src/v2hoi/contact_review.py")}
    args.output.mkdir()
    (args.output / "frames").mkdir()
    np.save(args.output / "display-roi.npy", roi, allow_pickle=False)
    expected = {"contact-comparison.mp4": (360, 30.)}
    expected.update({f"contact-slow-{first:06d}-{last:06d}.mp4": (last-first+1, 10.) for first, last in WINDOWS})
    writers = {name: VideoWriter(args.output / name, (args.size*3, args.size+78), fps, cv2)
               for name, (_, fps) in expected.items()}
    cap = cv2.VideoCapture(str(video))
    scale = args.size / float(roi[2]-roi[0])
    warp = np.asarray([[scale, 0, -roi[0]*scale], [0, scale, -roi[1]*scale]], dtype=float)
    samples, draw_seconds, worker_pids = [], [], set()
    # Spawned numerical libraries inherit one thread per task-owned worker.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    init = ([str(Path(spec[role]["path"]).parent) for role in ("a", "b")],
            str(Path(spec["object"]["path"])), K, roi, args.size)
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_renderer, initargs=init) if args.workers > 1 else None
    if pool is None:
        initialize_renderer(*init)

    def prepared_frames():
        pending = deque()
        for index in range(360):
            ok, rgb = cap.read()
            if not ok:
                raise ValueError(f"Original source decode failed at{index}")
            source = cv2.warpAffine(rgb, warp, (args.size, args.size), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
            job = pool.submit(render_layers, (index, source)) if pool else None
            pending.append((index, source, job))
            if len(pending) >= args.workers*2:
                yield pending.popleft()
        while pending:
            yield pending.popleft()

    try:
        for frame, source, job in prepared_frames():
            overlays, seconds, pid = job.result() if job else render_layers((frame, source))
            panels = [source, *overlays]
            draw_seconds.append(seconds)
            worker_pids.add(pid)
            canvas = np.full((args.size+78, args.size*3, 3), 20, np.uint8)
            canvas[48:48+args.size] = np.concatenate(panels, axis=1)
            labels = ["Source RGB", "A: same fit, no contact", "B: manual-phase contact"]
            for column, label in enumerate(labels):
                cv2.putText(canvas, label, (column*args.size+8, 39), cv2.FONT_HERSHEY_SIMPLEX, .46, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"EP16 original {frame:06d} | {frame/30:.3f}s | one fixed source ROI", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 235, 235), 1, cv2.LINE_AA)
            text = f"Manual attraction {'ON' if weights[frame].any() else 'OFF'} | estimated geometry/depth; no real-scene occlusion truth"
            cv2.putText(canvas, text, (8, args.size+68), cv2.FONT_HERSHEY_SIMPLEX, .41, (200, 210, 225), 1, cv2.LINE_AA)
            writers["contact-comparison.mp4"].write(canvas)
            for first, last in WINDOWS:
                if first <= frame <= last:
                    writers[f"contact-slow-{first:06d}-{last:06d}.mp4"].write(canvas)
            if frame in SAMPLES:
                path = args.output / "frames" / f"contact-{frame:06d}.jpg"
                if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 93]):
                    raise ValueError("Failed to save a source review image")
                samples.append({"original_frame_id": frame, "file": str(path.relative_to(args.output)), "sha256": digest(path)})
            if (frame+1) % 30 == 0:
                print(f"Compared {frame+1}/360 original frames", flush=True)
        if cap.read()[0]:
            raise ValueError("Source contains more than360 decoded frames")
    finally:
        cap.release()
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        for writer in writers.values():
            writer.close()
    videos = validate_videos(args.output, expected, cv2)
    after = {"manifest": digest(args.input_manifest), "evaluation": digest(args.evaluation),
             "renderer": digest(__file__), "rasterizer": digest(ROOT / "src/v2hoi/contact_review.py")}
    if before != after:
        raise ValueError("Rendering sources or input receipts changed")
    for role in ("a", "b"):
        root = Path(spec[role]["path"]).parent
        if native_bindings[role] != {name: digest(root / "review_arrays" / f"{name}.npy") for name in arrays[role]}:
            raise ValueError("Native review arrays changed during rendering")
    summary = {"schema_version": "ep16_contact_ab_video_review.v1", "recorded_at": datetime.now(timezone.utc).isoformat(),
               "track": 1, "episode": 16, "frames": 360, "fps": 30, "input_manifest_sha256": before["manifest"],
               "evaluation_sha256": before["evaluation"], "source_video_sha256": manifest["video_sha256"],
               "sources_before_and_after": before, "native_review_array_sha256": native_bindings,
               "display_roi_original_pixels": roi.tolist(), "videos": videos, "actually_saved_static_samples": samples,
               "requested_cpu_workers": args.workers, "actual_worker_pids": sorted(worker_pids),
               "render_seconds_mean": float(np.mean(draw_seconds)), "total_seconds": time.monotonic()-started,
               "rendering": "All native and object triangles, perspective-correct shared Z-buffer, constant albedo. One fixed display ROI for both methods and every frame.",
               "limitations": "Visibility is between estimated meshes only. No table/background geometry, calibrated scale, true contact, hidden hand accuracy or official score is established.",
               "quality_accepted": False, "all30_execution_authorized": False}
    write(args.output / "summary.json", summary)
    chunks = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>EP16 contact A/B review</title>',
              '<style>body{font:16px system-ui;background:#10151d;color:#edf3fa;margin:24px}main{max-width:1200px;margin:auto}video{max-width:100%;display:block}a{color:#85c6ff}p{line-height:1.5}</style><main>',
              '<h1>EP16: same-camera manual-contact A/B</h1><p>Left: source RGB. Center: A without contact. Right: B with manual-phase contact. All360 original frames share the same geometry preparation and display region.</p>',
              '<p>Blue: estimated native human. Orange: uncertain reconstructed object. Mesh-to-mesh depth testing does not establish real-scene occlusion, physical contact or scale accuracy.</p><p><a href="summary.json">Exact video/source receipts</a></p>']
    for row in videos:
        name = html.escape(row["file"])
        chunks.append(f'<h2>{name}</h2><video controls preload="metadata" src="{name}"></video>')
    chunks.append('</main></html>')
    (args.output / "index.html").write_text(''.join(chunks), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "frames": 360, "videos_fully_decoded": len(videos), "seconds": summary["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
