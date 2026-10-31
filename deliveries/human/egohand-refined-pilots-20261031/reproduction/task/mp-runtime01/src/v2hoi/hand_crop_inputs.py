"""Verify externally frozen source-pixel crops for the two controlled pilots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np


def load_verified_crop_boxes(
    boxes_path: Path,
    receipt_path: Path,
    *,
    source_video_sha256: str,
    episode: int,
    frames: int,
    image_size_wh: tuple[int, int] | list[int],
) -> tuple[np.ndarray, dict]:
    """Reject unbound, ambiguous or out-of-source crop inputs before inference."""
    if isinstance(episode, bool) or episode not in (11, 16):
        raise ValueError("External crops are restricted to the two authorized pilots")
    if not re.fullmatch(r"[0-9a-f]{64}", source_video_sha256):
        raise ValueError("Expected a source-video SHA-256")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError("Expected a positive complete source-frame count")
    size = np.asarray(image_size_wh)
    if size.shape != (2,) or not np.issubdtype(size.dtype, np.integer) or np.any(size <= 0):
        raise ValueError("Expected positive integer source width and height")
    boxes_path, receipt_path = Path(boxes_path), Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("track") != 1 or receipt.get("episode") != episode
            or receipt.get("frames") != frames
            or receipt.get("source_video_sha256") != source_video_sha256
            or receipt.get("all30_execution_authorized") is not False):
        raise ValueError("Crop receipt does not bind the authorized source and coverage")
    boxes_hash = hashlib.sha256(boxes_path.read_bytes()).hexdigest()
    if receipt.get("crop_boxes_sha256") != boxes_hash:
        raise ValueError("Crop-array identity differs from its frozen receipt")
    boxes = np.load(boxes_path, allow_pickle=False)
    if boxes.shape != (frames, 2, 4) or not np.issubdtype(boxes.dtype, np.integer):
        raise ValueError("Expected integer original-frame [T,2,4] crop boxes")
    lengths = boxes[..., 2:].astype(np.int64) - boxes[..., :2].astype(np.int64)
    if np.any(lengths <= 0) or np.any(lengths[..., 0] != lengths[..., 1]):
        raise ValueError("Crops must be nonempty squares")
    if np.any(boxes[..., :2] < 0) or np.any(boxes[..., 2:] > size):
        raise ValueError("Crops must remain inside the original source image")
    if receipt.get("side_order", ["left", "right"]) != ["left", "right"]:
        raise ValueError("Crop side order must be anatomical left/right")
    return boxes, receipt
