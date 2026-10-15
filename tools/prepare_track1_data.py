"""Verify pinned Track 1 bytes and decode original frame timelines on CPU.

This gate prepares raw inputs for the Body workstream. It neither runs models nor
creates masks, crops, calibration or predicted body parameters. Requires PyAV,
NumPy and pandas with a parquet engine. A fresh output directory is mandatory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from numbers import Integral
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import stat
import sys

REPOSITORY = "nvidia/video_to_data_challenge"
REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
VIDEO_KEY = "observation.images.exo_camera"
EPISODE_IDS = range(30)
META_FILES = {"README.md", "meta/info.json", "meta/episodes.jsonl",
              "meta/episodes_metadata.jsonl", "meta/episodes_stats.jsonl", "meta/tasks.jsonl"}
OPTIONAL_FILES = {".gitignore"}
TIMESTAMP_ATOL_SECONDS = 1e-6
TIMESTAMP_RTOL = 2e-7  # Allows float32 timestamp rounding, well below one frame.


class PreparationError(ValueError):
    """The raw-data gate failed; no ready handoff should be consumed."""


def episode_paths(index: int) -> tuple[str, str]:
    stem = f"episode_{index:06d}"
    return (f"videos/chunk-000/{VIDEO_KEY}/{stem}.mp4",
            f"data/chunk-000/{stem}.parquet")


def expected_paths() -> set[str]:
    return META_FILES | {path for index in EPISODE_IDS for path in episode_paths(index)}


def reject_links(path: Path) -> None:
    """Reject symbolic links and Windows reparse points, including ancestors."""
    for candidate in [path.absolute(), *path.absolute().parents]:
        if candidate.name.lower() in {"track_2", "track2"}:
            raise PreparationError("prohibited data path")
        if candidate.exists() or candidate.is_symlink():
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise PreparationError(f"symlink/reparse path is not allowed: {candidate}")


def safe_relative(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).anchor
            or any(p in {"", ".", ".."} for p in value.split("/"))
            or any(p.lower() in {"track_2", "track2"} for p in value.split("/"))):
        raise PreparationError(f"unsafe or prohibited source path: {value!r}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inventory(path: Path) -> dict[str, dict]:
    reject_links(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("repository") != REPOSITORY or doc.get("revision") != REVISION:
        raise PreparationError("source inventory must identify the pinned Track 1 repository revision")
    if not isinstance(doc.get("files"), list):
        raise PreparationError("source inventory files must be a list")
    entries = {}
    for entry in doc["files"]:
        full = safe_relative(entry.get("path"))
        if not full.startswith("track_1/") or entry.get("type") != "file":
            raise PreparationError(f"unexpected inventory entry: {full}")
        relative = full[len("track_1/"):]
        if relative not in expected_paths() | OPTIONAL_FILES or relative in entries:
            raise PreparationError(f"unexpected or duplicate inventory path: {full}")
        size = entry.get("size")
        if type(size) is not int or size <= 0:
            raise PreparationError(f"invalid inventory size: {full}")
        lfs = entry.get("lfs")
        if lfs is not None:
            if not isinstance(lfs, dict) or lfs.get("size") != size or not re.fullmatch(r"[0-9a-f]{64}", lfs.get("oid", "")):
                raise PreparationError(f"invalid LFS identity: {full}")
        elif not re.fullmatch(r"[0-9a-f]{40}", entry.get("oid", "")):
            raise PreparationError(f"invalid Git blob identity: {full}")
        entries[relative] = entry
    if expected_paths() - entries.keys():
        raise PreparationError(f"inventory is missing required Track 1 paths: {sorted(expected_paths() - entries.keys())}")
    return entries


def validate_tree(root: Path, entries: dict[str, dict]) -> list[str]:
    reject_links(root)
    if not root.is_dir():
        raise PreparationError("dataset root is not a directory")
    allowed = set(entries)
    allowed_dirs = {str(parent) for name in allowed for parent in PurePosixPath(name).parents if str(parent) != "."}
    found = set()
    for directory, names, files in os.walk(root, followlinks=False):
        for name in [*names, *files]:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            safe_relative(relative)
            reject_links(path)
            if path.is_dir():
                if relative not in allowed_dirs:
                    raise PreparationError(f"unexpected dataset directory: {relative}")
            elif path.is_file() and relative in allowed:
                found.add(relative)
            else:
                raise PreparationError(f"unexpected dataset file: {relative}")
    missing = expected_paths() - found
    if missing:
        raise PreparationError(f"dataset is missing required files: {sorted(missing)}")
    return sorted(found)


def verify_file(path: Path, entry: dict) -> dict:
    reject_links(path)
    size = path.stat().st_size
    if size != entry["size"]:
        raise PreparationError(f"source byte-size mismatch: {entry['path']}")
    digest = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
            blob.update(block)
    lfs = entry.get("lfs")
    actual = digest.hexdigest() if lfs else blob.hexdigest()
    expected = lfs["oid"] if lfs else entry["oid"]
    if actual != expected:
        raise PreparationError(f"source checksum mismatch: {entry['path']}")
    return {"size_bytes": size, "sha256": digest.hexdigest(),
            "verified_identity": {"algorithm": "sha256" if lfs else "git_blob_sha1", "value": expected}}


def decode_timeline(path: Path, expected_size: tuple[int, int]) -> dict:
    import av

    pts_values = []
    time_base = None
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise PreparationError(f"expected exactly one video stream: {path.name}")
        stream = container.streams.video[0]
        # Decode errors must surface instead of silently yielding concealed frames.
        stream.codec_context.options = {"err_detect": "explode"}
        codec = stream.codec_context.name
        declared_frames = stream.frames
        for frame in container.decode(stream):
            if frame.is_corrupt:
                raise PreparationError(f"corrupt frame {len(pts_values)}: {path.name}")
            if (frame.width, frame.height) != expected_size:
                raise PreparationError(f"decoded dimensions changed or disagree with metadata: {path.name}")
            if not isinstance(frame.pts, Integral) or isinstance(frame.pts, bool):
                raise PreparationError(f"missing/noninteger PTS: {path.name}")
            if frame.time_base is None:
                raise PreparationError(f"missing rational time base: {path.name}")
            current_base = Fraction(frame.time_base)
            if current_base <= 0 or (time_base is not None and current_base != time_base):
                raise PreparationError(f"invalid/changing rational time base: {path.name}")
            time_base = current_base
            if pts_values and frame.pts <= pts_values[-1]:
                raise PreparationError(f"non-increasing PTS at frame {len(pts_values)}: {path.name}")
            pts_values.append(int(frame.pts))
    if not pts_values or (declared_frames and declared_frames != len(pts_values)):
        raise PreparationError(f"empty or truncated video: {path.name}")
    average_rate = (Fraction(len(pts_values) - 1, pts_values[-1] - pts_values[0]) / time_base
                    if len(pts_values) > 1 else None)
    return {"frame_count": len(pts_values), "width": expected_size[0], "height": expected_size[1],
            "codec": codec, "frame_ids": list(range(len(pts_values))), "pts": pts_values,
            "time_base": {"numerator": time_base.numerator, "denominator": time_base.denominator},
            "decoded_average_fps": float(average_rate) if average_rate is not None else None,
            "timestamps_seconds": [float(value * time_base) for value in pts_values]}


def validate_table(table, episode: int, timeline: dict, expected_length: int) -> dict:
    import numpy as np
    from pandas.api.types import is_integer_dtype, is_numeric_dtype

    required = {"frame_index", "episode_index", "timestamp"}
    if not required.issubset(table.columns) or len(table) != expected_length or len(table) != timeline["frame_count"]:
        raise PreparationError(f"parquet/decode/metadata frame-count mismatch: episode {episode}")
    for name in ["frame_index", "episode_index"]:
        if not is_integer_dtype(table[name].dtype) or table[name].isna().any():
            raise PreparationError(f"noninteger parquet {name}: episode {episode}")
    if not np.array_equal(table["frame_index"].to_numpy(), np.arange(expected_length)):
        raise PreparationError(f"parquet frame_index must be exactly 0..T-1: episode {episode}")
    if not np.all(table["episode_index"].to_numpy() == episode):
        raise PreparationError(f"parquet episode_index mismatch: episode {episode}")
    if not is_numeric_dtype(table["timestamp"].dtype):
        raise PreparationError(f"nonnumeric parquet timestamp: episode {episode}")
    timestamps = table["timestamp"].to_numpy(dtype=float)
    actual = np.asarray(timeline["timestamps_seconds"])
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise PreparationError(f"nonfinite/non-increasing parquet timestamp: episode {episode}")
    if not np.allclose(timestamps, actual, atol=TIMESTAMP_ATOL_SECONDS, rtol=TIMESTAMP_RTOL):
        raise PreparationError(f"parquet timestamp disagrees with original PTS timeline: episode {episode}")
    return {"parquet_timestamps_seconds": timestamps.tolist(),
            "timestamp_max_abs_error_seconds": float(np.max(np.abs(timestamps - actual)))}


def indexed_metadata(path: Path) -> dict[int, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {}
    for row in rows:
        index = row.get("episode_index")
        if type(index) is not int or index not in EPISODE_IDS or index in result:
            raise PreparationError(f"invalid or duplicate episode index: {path.name}")
        result[index] = row
    if set(result) != set(EPISODE_IDS):
        raise PreparationError(f"metadata must cover exactly 30 episodes: {path.name}")
    return result


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare(dataset_root: Path, inventory_path: Path, output: Path) -> dict:
    import av
    import pandas as pd

    reject_links(output)
    if output.exists():
        raise PreparationError("output already exists; choose a fresh preparation directory")
    reject_links(dataset_root)
    root = dataset_root.resolve()
    output = output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise PreparationError("dataset and preparation output directories must not overlap")
    reject_links(inventory_path)
    inventory_sha256 = sha256(inventory_path)
    entries = load_inventory(inventory_path)
    found = validate_tree(root, entries)
    # Hash every source before loading any dataset metadata or parquet file.
    sources = {relative: verify_file(root / relative, entries[relative]) for relative in found}
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    episodes = indexed_metadata(root / "meta/episodes.jsonl")
    metadata = indexed_metadata(root / "meta/episodes_metadata.jsonl")
    if info.get("total_episodes") != 30 or info.get("total_videos") != 30:
        raise PreparationError("metadata must declare exactly 30 episodes and videos")
    fps = info.get("fps")
    if type(fps) not in {int, float} or not math.isfinite(fps) or fps <= 0:
        raise PreparationError("metadata must declare a positive finite fps")
    shape = info["features"][VIDEO_KEY]["shape"]
    if len(shape) != 3 or shape[2] != 3 or any(type(v) is not int or v <= 0 for v in shape):
        raise PreparationError("invalid video shape in metadata")
    for index in EPISODE_IDS:
        if type(episodes[index].get("length")) is not int or episodes[index]["length"] <= 0:
            raise PreparationError(f"invalid episode length: {index}")
        if metadata[index].get("video_key") != VIDEO_KEY:
            raise PreparationError(f"unexpected video key: {index}")
    expected_frames = sum(row["length"] for row in episodes.values())
    if info.get("total_frames") != expected_frames:
        raise PreparationError("metadata total_frames disagrees with episode lengths")
    # Reserve the destination exclusively. If decoding fails, checks.json records
    # failure and no ready manifest is written; reruns require a new destination.
    output.mkdir(parents=True, exist_ok=False)
    (output / "timelines").mkdir()
    checks = {"schema_version": "track1_raw_checks.v1", "passed": False,
              "raw_data_ready": False, "body_inference_ready": False, "models_executed": False}
    try:
        results = []
        for index in EPISODE_IDS:
            video, parquet = episode_paths(index)
            timeline = decode_timeline(root / video, (shape[1], shape[0]))
            average_fps = timeline["decoded_average_fps"]
            if average_fps is not None and not math.isclose(average_fps, fps, rel_tol=1e-7, abs_tol=1e-7):
                raise PreparationError(f"decoded average frame rate disagrees with declared fps: episode {index}")
            timeline.update(validate_table(pd.read_parquet(root / parquet), index, timeline, episodes[index]["length"]))
            timeline.update({"schema_version": "track1_original_timeline.v1", "episode_index": index,
                             "first_source_frame_id": 0, "video_path": video, "parquet_path": parquet,
                             "video_sha256": sources[video]["sha256"], "parquet_sha256": sources[parquet]["sha256"]})
            relative = f"timelines/episode_{index:06d}.json"
            write_json(output / relative, timeline)
            results.append({"episode_index": index, "sequence_id": metadata[index].get("sequence_id"),
                            "object": metadata[index].get("object"), "camera": metadata[index].get("camera"),
                            "object_prompt": metadata[index].get("object_prompt"),
                            "tasks": episodes[index].get("tasks", []), "video_path": video, "parquet_path": parquet,
                            "first_source_frame_id": 0, "frame_count": timeline["frame_count"],
                            "declared_fps": fps, "decoded_average_fps": average_fps,
                            "width": shape[1], "height": shape[0], "timeline_path": relative,
                            "timeline_sha256": sha256(output / relative),
                            "timestamp_max_abs_error_seconds": timeline["timestamp_max_abs_error_seconds"]})
            print(f"[{index + 1:02d}/30] episode_{index:06d}: {timeline['frame_count']} frames verified", flush=True)
        # Downloads or other writers must not replace inputs during a long decode.
        # Recheck the exact tree and every byte before publishing readiness.
        if validate_tree(root, entries) != found:
            raise PreparationError("source file set changed during preparation")
        for relative in found:
            if verify_file(root / relative, entries[relative]) != sources[relative]:
                raise PreparationError(f"source changed during preparation: {relative}")
        reject_links(inventory_path)
        if sha256(inventory_path) != inventory_sha256:
            raise PreparationError("source inventory changed during preparation")
        manifest = {"schema_version": "track1_raw_preparation.v1",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "repository": REPOSITORY, "revision": REVISION, "track": "track_1",
                    "dataset_root_relative": Path(os.path.relpath(root, output)).as_posix(),
                    "source_inventory_sha256": inventory_sha256,
                    "source_files": sources, "episodes": results, "total_frames": expected_frames,
                    "raw_data_ready": True, "body_inference_ready": False, "models_executed": False,
                    "body_pending": ["human masks with quality checks", "camera intrinsics with provenance and validation"],
                    "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                    "packages": {name: importlib.metadata.version(name) for name in ["av", "pandas", "numpy"]},
                                    "ffmpeg_libraries": {name: list(version) for name, version in av.library_versions.items()}},
                    "timestamp_comparison": {"absolute_tolerance_seconds": TIMESTAMP_ATOL_SECONDS,
                                             "relative_tolerance": TIMESTAMP_RTOL,
                                             "reference": "original integer PTS multiplied by rational time_base; no reindexing or resampling"}}
        write_json(output / "manifest.json", manifest)
        checks.update({"passed": True, "raw_data_ready": True, "source_files_verified": len(sources),
                       "videos_fully_decoded": len(results), "frames_verified": expected_frames,
                       "parquet_timelines_matched": len(results), "manifest_sha256": sha256(output / "manifest.json"),
                       "source_bytes_reverified_before_publish": True,
                       "checks": ["exact Track 1 file allowlist", "pinned source sizes and hashes", "full sequential CPU decode",
                                  "constant dimensions and positive rational time base", "integer increasing PTS",
                                  "contiguous original frame IDs", "parquet frame IDs, episode IDs and timestamps"],
                       "hash_coverage": "SHA256SUMS covers every generated JSON; the checksum listing excludes itself"})
        write_json(output / "checks.json", checks)
        files = sorted(output.rglob("*.json"))
        (output / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in files), encoding="utf-8")
        return manifest
    except Exception as error:
        (output / "manifest.json").unlink(missing_ok=True)
        checks.update({"passed": False, "raw_data_ready": False, "error": str(error)})
        write_json(output / "checks.json", checks)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, default=Path("configs/track1-source.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = prepare(args.dataset_root, args.source_inventory, args.output)
    except Exception as error:
        print(f"Raw-data gate failed: {error}", file=sys.stderr)
        return 1
    print(f"Raw data ready: 30 videos / {manifest['total_frames']} original frames. Body inference pending masks and intrinsics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
