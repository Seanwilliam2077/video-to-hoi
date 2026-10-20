"""Verified Track 1 mask/camera inputs for SAM 3D Body, without model imports.

The receiver gate verifies original-video timelines and visual review. This
consumer binds that receipt to the bytes actually consumed, then checks every
mask again while inference runs. No detector, guessed camera or empty-mask
fallback is permitted in this profile.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Mapping

import numpy as np
from PIL import Image

PROFILE = "sam3d_body_masks_camera.v1"
SOURCE_REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
ARTIFACTS = {"timeline.json", "observations.json", "intrinsics.json", "camera-estimation.json", "masks-manifest.json"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Mapping) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=reject)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _regular(path: Path, root: Path) -> None:
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), f"Unsafe/missing input file: {path}")
    for parent in path.parents:
        require(not parent.is_symlink(), f"Input symlink is forbidden: {parent}")
        if parent == root:
            break


def camera_matrix(fields: Mapping, width: int, height: int) -> np.ndarray:
    """Preserve supplied integer-pixel-center coordinates, with no half-pixel shift."""
    require(set(fields) == {"fx", "fy", "cx", "cy", "width", "height"}, "Camera must have exactly six native fields")
    require(type(fields["width"]) is int and type(fields["height"]) is int
            and (fields["width"], fields["height"]) == (width, height), "Camera dimensions differ from video")
    for key in ("fx", "fy", "cx", "cy"):
        require(type(fields[key]) in (int, float) and math.isfinite(fields[key]), f"Invalid camera {key}")
    require(fields["fx"] > 0 and fields["fy"] > 0, "Camera focal lengths must be positive")
    require(0 <= fields["cx"] < width and 0 <= fields["cy"] < height, "Camera principal point outside original pixels")
    return np.asarray([[fields["fx"], 0, fields["cx"]], [0, fields["fy"], fields["cy"]], [0, 0, 1]], dtype=np.float32)


def decoded_frame_sha256(bgr: np.ndarray) -> str:
    require(bgr.dtype == np.uint8 and bgr.ndim == 3 and bgr.shape[2] == 3, "Decoded frame must be uint8 HxWx3 BGR")
    digest = hashlib.sha256()
    digest.update(f"uint8:BGR:{bgr.shape[0]}:{bgr.shape[1]}:3:".encode("ascii"))
    digest.update(np.ascontiguousarray(bgr).tobytes())
    return digest.hexdigest()


def validate_resume_frame(record: Mapping, frame_index: int, configuration_sha256: str,
                          pixel_sha256: str, mask_sha256: str) -> None:
    """Resume only exact predictions, rather than trusting an existing filename."""
    for name, expected in (("frame_index", frame_index), ("input_configuration_sha256", configuration_sha256),
                           ("decoded_bgr_sha256", pixel_sha256), ("input_mask_sha256", mask_sha256)):
        require(name in record and np.asarray(record[name]).shape == (), f"Missing/scalar resume field: {name}")
        require(np.asarray(record[name]).item() == expected, f"Resume {name} differs at frame {frame_index}")
    for name, shape in (("pose", (136,)), ("scales", (68,)), ("shape", (45,)), ("pred_vertices", (18439, 3)),
                        ("pred_keypoints_2d", (70, 2)), ("pred_keypoints_3d", (70, 3)), ("pred_cam_t", (3,))):
        require(name in record and np.asarray(record[name]).dtype.kind in "fiu" and np.isfinite(record[name]).all(),
                f"Invalid saved prediction {name} at frame {frame_index}")
        require(np.asarray(record[name]).shape == shape, f"Invalid saved prediction shape {name} at frame {frame_index}")


@dataclass(frozen=True)
class BodyInputs:
    root: Path
    episode: int
    frame_count: int
    width: int
    height: int
    intrinsics: np.ndarray
    inventory: dict
    mask_files: list
    observations: list
    provenance: dict

    def frame(self, index: int) -> tuple[np.ndarray, np.ndarray, str]:
        require(type(index) is int and 0 <= index < self.frame_count, "Invalid original frame index")
        entry = self.mask_files[index]
        path = self.root / entry["path"]
        _regular(path, self.root)
        expected_sha = entry["sha256"]
        require(path.stat().st_size == entry["bytes"] and sha256(path) == expected_sha, f"Mask bytes changed at frame {index}")
        with Image.open(path) as image:
            require(image.format == "PNG" and image.mode in ("1", "L") and image.size == (self.width, self.height)
                    and getattr(image, "n_frames", 1) == 1, f"Invalid mask format/dimensions at frame {index}")
            pixels = np.asarray(image)
        require(np.all((pixels == 0) | (pixels == 1)) or np.all((pixels == 0) | (pixels == 255)), f"Nonbinary mask at frame {index}")
        occupied = pixels != 0
        area = int(occupied.sum())
        require(0 < area < self.width * self.height, f"Empty/full mask at frame {index}")
        ys, xs = np.nonzero(occupied)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        require(bbox[2] - bbox[0] > 1 and bbox[3] - bbox[1] > 1, f"Mask would trigger full-image fallback at frame {index}")
        observation = self.observations[index]
        require(observation.get("frame_id") == index and observation.get("valid") is True
                and observation.get("area_pixels") == area and observation.get("bbox_xyxy") == bbox,
                f"Mask observation disagrees with pixels at frame {index}")
        # Upstream ToTensor divides uint8 by 255. Supplying 0/1 uint8 would
        # silently attenuate the prompt by 255, so normalize explicitly here.
        masks = (occupied.astype(np.uint8) * 255)[None, :, :, None]
        return masks, np.asarray(bbox, dtype=np.float32).reshape(1, 4), expected_sha


def load_body_inputs(root: Path, receiver_checks: Path, *, video: Path, episode: int,
                     width: int, height: int, frame_count: int, fps: float,
                     video_sha256: str | None = None) -> BodyInputs:
    root, receiver_checks = root.resolve(strict=True), receiver_checks.resolve(strict=True)
    require(type(episode) is int and 0 <= episode < 30, "Only Track 1 episodes 0..29 are supported")
    report = read_json(receiver_checks)
    require(report.get("schema_version") == "body_handoff_checks.v1" and report.get("profile") == PROFILE
            and report.get("passed") is True and report.get("all_30_body_inputs_verified") is True
            and report.get("requested_episodes") == list(range(30)), "A successful all-30 receiver gate is required")
    rows = report.get("episodes", [])
    require(isinstance(rows, list) and len(rows) == 30
            and all(isinstance(row, dict) and type(row.get("episode_index")) is int for row in rows)
            and sorted(row["episode_index"] for row in rows) == list(range(30)), "Receiver episode records are incomplete/duplicate")
    require(all(row.get("body_inputs_verified") is True and row.get("inputs_structurally_verified") is True
                and row.get("visual_review_verified") is True and row.get("errors") == [] for row in rows), "Receiver contains a failed episode")
    received = next(row for row in rows if row["episode_index"] == episode)
    _regular(root / "SHA256SUMS", root)
    inventory = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        pieces = line.split("  ", 1)
        require(len(pieces) == 2 and SHA256.fullmatch(pieces[0]) is not None, "Malformed checksum entry")
        expected, name = pieces
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and "\\" not in name and ":" not in name and all(x not in ("", ".", "..") for x in name.split("/"))
                and name not in inventory and name not in {"SHA256SUMS", "checks.json"}, "Unsafe/duplicate checksum path")
        target = root / name
        _regular(target, root)
        require(sha256(target) == expected, f"Input checksum differs: {name}")
        inventory[name] = expected
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
              and path.relative_to(root).as_posix() not in {"SHA256SUMS", "checks.json"}}
    require(actual == set(inventory), "Input checksum inventory does not cover the exact package")
    require(ARTIFACTS | {"input.json", "review.json"} <= set(inventory), "Missing input artifacts")
    identity = read_json(root / "input.json")
    require(identity.get("schema_version") == "body_input.v1" and identity.get("profile") == PROFILE
            and identity.get("episode_index") == episode, "Wrong input episode/schema")
    source = identity.get("source", {})
    video_sha = video_sha256 if video_sha256 is not None else sha256(video)
    expected_video = f"videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4"
    require(source.get("repository") == "nvidia/video_to_data_challenge" and source.get("revision") == SOURCE_REVISION
            and source.get("video_path") == expected_video and source.get("video_sha256") == video_sha,
            "Input does not bind this pinned Track 1 video")
    require(received.get("input_sha256") == inventory["input.json"] and received.get("review_sha256") == inventory["review.json"]
            and received.get("video_sha256") == video_sha and received.get("frame_count") == frame_count,
            "Receiver receipt differs from consumed episode bytes")
    require(identity.get("artifacts") == {name: inventory[name] for name in ARTIFACTS}
            and source.get("timeline_sha256") == inventory["timeline.json"], "Input artifact binding is stale")
    review = read_json(root / "review.json")
    require(review.get("status") == "approved" and review.get("unresolved_issues") == []
            and review.get("reviewed_artifacts") == {name: inventory[name] for name in ARTIFACTS | {"input.json"}},
            "Visual review does not bind these inputs")
    timeline = read_json(root / "timeline.json")
    require(timeline.get("schema_version") == "track1_original_timeline.v1" and timeline.get("episode_index") == episode
            and timeline.get("video_sha256") == video_sha and timeline.get("first_source_frame_id") == 0
            and (timeline.get("width"), timeline.get("height"), timeline.get("frame_count")) == (width, height, frame_count)
            and timeline.get("frame_ids") == list(range(frame_count)), "Original timeline, frame IDs or dimensions differ")
    timestamps, pts = timeline.get("timestamps_seconds"), timeline.get("pts")
    require(isinstance(timestamps, list) and len(timestamps) == frame_count and isinstance(pts, list) and len(pts) == frame_count
            and all(type(t) in (int, float) and math.isfinite(t) for t in timestamps)
            and all(type(p) is int for p in pts) and all(b > a for a, b in zip(timestamps, timestamps[1:])), "Invalid original timeline")
    require(math.isfinite(fps) and fps > 0 and (frame_count < 2 or math.isclose((frame_count - 1) / (timestamps[-1] - timestamps[0]), fps, rel_tol=1e-4)),
            "Decoded video FPS differs from original timeline")
    intrinsics_fields = read_json(root / "intrinsics.json")
    intrinsics = camera_matrix(intrinsics_fields, width, height)
    masks = read_json(root / "masks-manifest.json")
    files = masks.get("files")
    expected_names = [f"human_masks/{i:06d}.png" for i in range(frame_count)]
    require(masks.get("schema_version") == "body_masks.v1" and masks.get("episode_index") == episode
            and masks.get("video_sha256") == video_sha and isinstance(files, list) and len(files) == frame_count
            and all(isinstance(row, dict) for row in files) and [row.get("path") for row in files] == expected_names,
            "Mask manifest must cover original frames in order")
    require({name for name in inventory if name.startswith("human_masks/")} == set(expected_names)
            and all(row.get("sha256") == inventory[row["path"]] and type(row.get("bytes")) is int and row["bytes"] > 0 for row in files),
            "Mask manifest identity/coverage differs from inventory")
    observations = read_json(root / "observations.json")
    require(observations.get("schema_version") == "body_observations.v1" and observations.get("episode_index") == episode
            and observations.get("target_id") == identity.get("target_id") and isinstance(observations.get("frames"), list)
            and len(observations["frames"]) == frame_count, "Invalid target observations")
    camera = read_json(root / "camera-estimation.json")
    require(camera.get("schema_version") == "body_camera_estimation.v1" and camera.get("episode_index") == episode
            and camera.get("video_sha256") == video_sha and camera.get("status") == "camera_estimate_consistent", "Wrong camera provenance")
    provenance = {"profile": PROFILE, "body_input_dir": str(root), "receiver_checks": str(receiver_checks),
                  "receiver_checks_sha256": sha256(receiver_checks), "input_inventory_sha256": sha256(root / "SHA256SUMS"),
                  "input_artifact_sha256": {name: inventory[name] for name in sorted(ARTIFACTS | {"input.json", "review.json"})},
                  "target_id": identity["target_id"], "input_producer": identity.get("producer"),
                  "camera_producer": camera, "camera_fields": intrinsics_fields,
                  "bbox_method": "per-frame binary-mask tight bbox, exclusive maximum coordinates",
                  "mask_encoding": "uint8[1,H,W,1], zero background, 255 foreground",
                  "intrinsics_status": "received MoGe2 pinhole estimate in original integer pixel centers, not measured calibration"}
    result = BodyInputs(root, episode, frame_count, width, height, intrinsics, inventory, files, observations["frames"], provenance)
    # Check all pixels before loading the large model. frame() repeats the check
    # at consumption time, so later modification cannot silently alter prompts.
    for index in range(frame_count):
        result.frame(index)
    return result
