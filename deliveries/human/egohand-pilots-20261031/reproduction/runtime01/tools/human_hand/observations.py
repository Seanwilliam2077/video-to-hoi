"""Acquire independent 2D hand observations from an explicit Track 1 video.

SAM predictions locate stable crops only. MediaPipe provides all target pixels.
The exported ``confidence`` is a conservative binary fitting weight, not a
calibrated detection probability or per-landmark visibility prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.ndimage import median_filter


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
SIDE_ORDER = ("left", "right")
SAM_WRISTS = (62, 41)
SAM_FINGERS = (list(range(42, 62)), list(range(21, 41)))
SAM_BASES = ([45, 49, 53, 57, 61], [24, 28, 32, 36, 40])
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
    (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
    (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_crops(seeds: np.ndarray) -> np.ndarray:
    """Offline robust crop centers/sizes; no predicted target interpolation."""
    boxes = np.empty((len(seeds), 2, 4), dtype=np.int32)
    for side, (wrist, bases, fingers) in enumerate(zip(SAM_WRISTS, SAM_BASES, SAM_FINGERS)):
        palm = np.median(seeds[:, [wrist, *bases]], axis=1)
        centers = median_filter(palm, size=(5, 1), mode="nearest")
        hand = seeds[:, [wrist, *fingers]]
        span = np.max(np.ptp(hand, axis=1), axis=1)
        widths = median_filter(np.clip(span * 2.6, 112, 208), size=15, mode="nearest")
        lower = np.rint(centers - widths[:, None] / 2).astype(np.int32)
        widths = np.rint(widths).astype(np.int32)
        boxes[:, side, :2] = lower
        boxes[:, side, 2:] = lower + widths[:, None]
    return boxes


def crop_image(frame: np.ndarray, box: np.ndarray, size: int = 384) -> np.ndarray:
    x0, y0, x1, y1 = map(int, box)
    height, width = frame.shape[:2]
    padded = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if sx1 > sx0 and sy1 > sy0:
        padded[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    return cv2.resize(padded, (size, size), interpolation=cv2.INTER_CUBIC)


def normalized_to_pixels(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    return points * (box[2:] - box[:2]) + box[:2]


def local_skin_fraction(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Weak color support only; neither a visibility model nor a skin classifier."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # A deliberately broad color gate: unsupported pixels get no fitting weight.
    mask = (
        (ycrcb[:, :, 0] > 35) & (ycrcb[:, :, 1] >= 132)
        & (ycrcb[:, :, 1] <= 185) & (ycrcb[:, :, 2] >= 75)
        & (ycrcb[:, :, 2] <= 140)
    ).astype(np.float32)
    fractions = cv2.blur(mask, (9, 9))
    pixels = np.rint(points).astype(np.int32)
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < frame.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < frame.shape[0])
    result = np.zeros(len(points), dtype=np.float32)
    result[inside] = fractions[pixels[inside, 1], pixels[inside, 0]]
    return result


def draw_crop(image: np.ndarray, points: np.ndarray, weights: np.ndarray, box: np.ndarray, title: str) -> np.ndarray:
    display = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    if np.isfinite(points).all():
        pixels = np.rint((points - box[:2]) / (box[2:] - box[:2]) * 512).astype(int)
        for a, b in CONNECTIONS:
            cv2.line(display, tuple(pixels[a]), tuple(pixels[b]), (200, 200, 200), 1, cv2.LINE_AA)
        for point, weight in zip(pixels, weights):
            cv2.circle(display, tuple(point), 3, (50, 230, 50) if weight else (40, 40, 230), -1, cv2.LINE_AA)
    canvas = np.zeros((552, 512, 3), dtype=np.uint8)
    canvas[40:] = display
    cv2.putText(canvas, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sam-keypoints", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, choices=range(30), help="Caller-confirmed original Track 1 episode ID")
    parser.add_argument("--frames", nargs="*", type=int, help="Debug exact source frames; default all frames")
    parser.add_argument("--require-handedness", action="store_true", help="Reject classifier/seed-side disagreement; valid for the manually checked nonmirrored source")
    parser.add_argument("--mirror-check", action="store_true", help="Require independent flipped-crop localization consistency, without smoothing targets")
    args = parser.parse_args()
    if "track_1" not in str(args.video).replace("\\", "/").split("/"):
        raise ValueError("Explicit Track 1 source path required")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frames").mkdir(exist_ok=True)
    if (args.output / "observations.npz").exists():
        raise FileExistsError("Preserve prior observations; choose a fresh output directory")
    import mediapipe as mp

    seeds = np.load(args.sam_keypoints, allow_pickle=False)
    if seeds.ndim != 3 or seeds.shape[1:] != (70, 2) or not np.isfinite(seeds).all():
        raise ValueError("Expected finite SAM MHR70 source-pixel seeds")
    boxes = stable_crops(seeds)
    chosen = list(range(len(seeds))) if not args.frames else args.frames
    if len(set(chosen)) != len(chosen) or any(i < 0 or i >= len(seeds) for i in chosen):
        raise ValueError("Frame indices must be unique and within the source")
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    source_size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    n = len(chosen)
    joints = np.full((n, 2, 21, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((n, 2, 21), dtype=np.float32)
    skin_support = np.zeros_like(confidence)
    handedness_score = np.full((n, 2), np.nan, dtype=np.float32)
    handedness_label = np.full((n, 2), "", dtype="U5")
    wrist_distance = np.full((n, 2), np.nan, dtype=np.float32)
    mirror_joints = np.full_like(joints, np.nan)
    mirror_disagreement = np.full((n, 2, 21), np.nan, dtype=np.float32)
    statuses = np.full((n, 2), "missing", dtype="U32")
    all_candidates = []
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model.resolve())),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=.5,
        min_hand_presence_confidence=.5,
        min_tracking_confidence=.5,
    )
    start = time.perf_counter()
    previous_index = -1
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for row, index in enumerate(chosen):
            if index != previous_index + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, frame = cap.read()
            if not success:
                raise RuntimeError(f"Could not decode original frame {index}")
            previous_index = index
            crops = []
            frame_candidates = []
            for side in range(2):
                box = boxes[index, side]
                crop = crop_image(frame, box)
                crops.append(crop)
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                candidates = []
                for landmarks, categories in zip(result.hand_landmarks, result.handedness):
                    xy = normalized_to_pixels(np.array([[p.x, p.y] for p in landmarks]), box)
                    category = categories[0]
                    target = np.linalg.norm(xy[0] - seeds[index, SAM_WRISTS[side]])
                    opposite = np.linalg.norm(xy[0] - seeds[index, SAM_WRISTS[1 - side]])
                    candidates.append((float(target), float(opposite), xy, category))
                frame_candidates.append([
                    {"wrist_distance": a, "opposite_wrist_distance": b, "joints2d": xy.tolist(),
                     "handedness_label": cat.category_name, "handedness_score": cat.score}
                    for a, b, xy, cat in candidates
                ])
                if candidates:
                    distance, opposite, xy, category = min(candidates, key=lambda entry: entry[0])
                    joints[row, side] = xy
                    wrist_distance[row, side] = distance
                    handedness_score[row, side] = category.score
                    handedness_label[row, side] = category.category_name
                    status = "accepted"
                    if distance > max(28, (box[2] - box[0]) * .26):
                        status = "wrist_mismatch"
                    elif opposite < distance + 8:
                        status = "side_ambiguous"
                    elif np.ptp(xy, axis=0).max() < 12:
                        status = "degenerate"
                    statuses[row, side] = status
                    skin_support[row, side] = local_skin_fraction(frame, xy)
                    if status == "accepted":
                        confidence[row, side] = (skin_support[row, side] >= .35).astype(np.float32)
                    if confidence[row, side].sum() < 5 and status == "accepted":
                        confidence[row, side] = 0
                        statuses[row, side] = "low_color_support"
                    if args.require_handedness and category.category_name.lower() != SIDE_ORDER[side]:
                        confidence[row, side] = 0
                        statuses[row, side] = "handedness_disagree"
                    if args.mirror_check:
                        flipped = cv2.cvtColor(crop[:, ::-1].copy(), cv2.COLOR_BGR2RGB)
                        mirror_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=flipped))
                        mirror_candidates = []
                        for flip_landmarks in mirror_result.hand_landmarks:
                            normalized = np.array([[p.x, p.y] for p in flip_landmarks])
                            normalized[:, 0] = 1 - normalized[:, 0] - 1 / crop.shape[1]
                            flip_xy = normalized_to_pixels(normalized, box)
                            mirror_candidates.append(flip_xy)
                        if mirror_candidates:
                            flip_xy = min(mirror_candidates, key=lambda candidate: np.linalg.norm(candidate[0] - xy[0]))
                            mirror_joints[row, side] = flip_xy
                            errors = np.linalg.norm(xy - flip_xy, axis=1)
                            mirror_disagreement[row, side] = errors
                            confidence[row, side] *= (errors <= 6).astype(np.float32)
                            if np.median(errors) > 4 or confidence[row, side].sum() < 5:
                                confidence[row, side] = 0
                                if statuses[row, side] == "accepted":
                                    statuses[row, side] = "mirror_disagree"
                        else:
                            confidence[row, side] = 0
                            if statuses[row, side] == "accepted":
                                statuses[row, side] = "mirror_missing"
            # Reject a duplicate observation assigned to both anatomical hands.
            if np.isfinite(joints[row]).all() and np.linalg.norm(joints[row, 0, 0] - joints[row, 1, 0]) < 12:
                loser = int(np.argmax(wrist_distance[row]))
                confidence[row, loser] = 0
                statuses[row, loser] = "duplicate_hand"
            panels = [draw_crop(crops[side], joints[row, side], confidence[row, side], boxes[index, side],
                                f"{SIDE_ORDER[side]} frame {index:03d}: {statuses[row, side]}") for side in range(2)]
            cv2.imwrite(str(args.output / "frames" / f"{index:06d}.jpg"), np.concatenate(panels, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 92])
            all_candidates.append({"frame": index, "crops": frame_candidates})
            if (row + 1) % 30 == 0 or row + 1 == n:
                print(f"Independent hand observations: {row + 1}/{n}", flush=True)
    cap.release()
    elapsed = time.perf_counter() - start
    np.savez_compressed(
        args.output / "observations.npz", frame_indices=np.array(chosen, dtype=np.int64),
        joints2d=joints, confidence=confidence, skin_support=skin_support,
        handedness_score=handedness_score, handedness_label=handedness_label,
        wrist_distance=wrist_distance, statuses=statuses, crop_boxes=boxes[chosen],
        side_order=np.array(SIDE_ORDER), fps=np.float32(fps), image_size=np.array(source_size),
        source_video_sha256=np.array(sha256(args.video)),
        mirror_joints2d=mirror_joints, mirror_disagreement_px=mirror_disagreement,
    )
    report = {
        "observer": "MediaPipe HandLandmarker independent 2D observations",
        "mediapipe_version": importlib.metadata.version("mediapipe"), "episode": args.episode,
        "model_url": MODEL_URL, "model_sha256": sha256(args.model),
        "source_video": str(args.video.resolve()), "source_video_sha256": sha256(args.video),
        "crop_seed_source": str(args.sam_keypoints.resolve()), "crop_seed_sha256": sha256(args.sam_keypoints),
        "script_sha256": sha256(Path(__file__)), "side_order": list(SIDE_ORDER),
        "frames": n, "fps": fps, "image_size_wh": source_size,
        "wall_seconds": elapsed, "accepted_hand_frames": (confidence.sum(axis=2) > 0).sum(axis=0).tolist(),
        "require_handedness": args.require_handedness, "mirror_check": args.mirror_check,
        "mirror_gate": "When enabled, preserve original observed pixels only where flip-then-unflip point discrepancy <=6 original pixels; reject hand median discrepancy >4 pixels or fewer than5 supported points. No target smoothing/interpolation.",
        "positive_joint_weights": (confidence > 0).sum(axis=(0, 2)).tolist(),
        "status_counts": {side: {str(key): int(value) for key, value in zip(*np.unique(statuses[:, i], return_counts=True))} for i, side in enumerate(SIDE_ORDER)},
        "confidence_semantics": "Binary fitting weights, NOT probabilities. Requires model detection, SAM-wrist association, nonduplicate side assignment, local color support and any enabled handedness/mirror gates. Missing/ambiguous/unsupported joints weight 0.",
        "visibility_limit": "MediaPipe returns no per-joint visibility/confidence. Broad YCrCb color support is a heuristic, may reject visible skin or accept skin-colored objects. Occluded predictions are not verified evidence.",
        "handedness_semantics": "Anatomical side assigned by nearest original SAM wrist, not selfie handedness class. Classification scores stored separately, not used as observation confidence.",
        "crop_semantics": "SAM seeds locate median-filtered crops only; target coordinates are from independent image model. IMAGE mode forces a new image observation each frame, no target interpolation/tracking carryover.",
        "haMeR_handflow_status": "Not run: licensed MANO_RIGHT.pkl absent in verified local inventory; no license-bypassing model-buffer reconstruction attempted.",
        "data_policy": "Only the explicit Track 1 source video and its own SAM crop seeds. Generic public model, no additional challenge data.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "candidates.json").write_text(json.dumps(all_candidates), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
