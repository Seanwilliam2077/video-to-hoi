"""Verify portable, complete Track 1 inputs for the pinned standalone Body adapter.

This is an input gate, not a Body inference or scientific-accuracy certificate.
It rechecks original source bytes and decoded timelines, reads every PNG, and
requires a current recorded visual review. Missing episodes fail the default
all-30 gate; --episodes is only a diagnostic subset and never reports all-30 ready.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

_SPEC = importlib.util.spec_from_file_location("body_raw_gate", Path(__file__).with_name("prepare_track1_data.py"))
raw = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(raw)

PROFILE = "sam3d_body_masks_camera.v1"
BODY_SOURCE_REVISION = "33129dd0f2d2dcfd1164d43fd076542660756ed2"
ARTIFACTS = {"timeline.json", "observations.json", "intrinsics.json", "camera-estimation.json", "masks-manifest.json"}
REVIEW_ARTIFACTS = ARTIFACTS | {"input.json"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class HandoffError(ValueError):
    """An input is missing, changed, invalid, unreviewed, or not portable."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HandoffError(message)


def _reject_constant(value):
    raise HandoffError(f"nonfinite JSON constant: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    raw.reject_links(path)
    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant,
                        object_pairs_hook=_unique_object)
    require(isinstance(result, dict), f"JSON object required: {path.name}")
    return result


def text(value, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"missing {label}")
    return value


def digest(value, label: str) -> str:
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None, f"invalid {label} SHA256")
    return value


def finite(value, label: str) -> float:
    require(type(value) in {int, float} and math.isfinite(value), f"{label} must be finite")
    return float(value)


def frame_ids(value, count: int, label: str, *, nonempty: bool = True) -> list[int]:
    require(isinstance(value, list) and (bool(value) or not nonempty), f"{label} must be a frame list")
    require(all(type(i) is int and 0 <= i < count for i in value), f"{label} contains invalid source frame IDs")
    require(len(set(value)) == len(value), f"{label} contains duplicate source frame IDs")
    return value


def validate_intrinsics(doc: dict, width: int, height: int) -> None:
    require(set(doc) == {"fx", "fy", "cx", "cy", "width", "height"}, "intrinsics must have exactly the six native fields")
    require(type(doc["width"]) is int and type(doc["height"]) is int
            and (doc["width"], doc["height"]) == (width, height), "intrinsics dimensions mismatch")
    for name in ("fx", "fy"):
        require(finite(doc[name], name) > 0, f"{name} must be positive")
    require(0 <= finite(doc["cx"], "cx") < width and 0 <= finite(doc["cy"], "cy") < height,
            "principal point outside original pixel grid")


def verify_checksum_inventory(root: Path) -> dict[str, str]:
    raw.reject_links(root)
    checksum_path = root / "SHA256SUMS"
    raw.reject_links(checksum_path)
    inventory = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        require(bool(line), "blank SHA256SUMS entry")
        parts = line.split("  ", 1)
        require(len(parts) == 2, "malformed SHA256SUMS entry")
        expected, relative = parts
        digest(expected, relative)
        raw.safe_relative(relative)
        require(relative not in inventory and relative not in {"SHA256SUMS", "checks.json"},
                f"duplicate or reserved checksum entry: {relative}")
        inventory[relative] = expected
    actual_files = set()
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            item = Path(directory) / name
            raw.reject_links(item)
            raw.safe_relative(item.relative_to(root).as_posix())
        for name in files:
            item = Path(directory) / name
            relative = item.relative_to(root).as_posix()
            if relative in {"SHA256SUMS", "checks.json"}:
                continue
            require(item.is_file(), f"not a regular file: {relative}")
            actual_files.add(relative)
    require(actual_files == set(inventory), "SHA256SUMS must cover the exact package file set")
    require(REVIEW_ARTIFACTS | {"review.json"} <= actual_files, "missing required package artifact")
    for relative, expected in inventory.items():
        require(raw.sha256(root / relative) == expected, f"checksum mismatch: {relative}")
    return inventory


def validate_producer(producer: dict) -> None:
    require(isinstance(producer, dict), "missing producer provenance")
    text(producer.get("method"), "producer method")
    text(producer.get("code_revision"), "producer code_revision")
    digest(producer.get("checkpoint_sha256"), "producer checkpoint")
    require(isinstance(producer.get("environment"), dict) and bool(producer["environment"]),
            "missing producer environment")


def validate_camera(doc: dict, episode: int, video_sha256: str, count: int, width: int, height: int,
                    selected_intrinsics: dict | None = None) -> None:
    require(doc.get("schema_version") == "body_camera_estimation.v1", "unknown camera-estimation schema")
    require(type(doc.get("episode_index")) is int and doc["episode_index"] == episode, "camera episode mismatch")
    require(doc.get("video_sha256") == video_sha256, "camera source video mismatch")
    require(doc.get("status") == "camera_estimate_consistent", "camera estimate is not marked consistent")
    for field in ("method", "code_revision", "aggregation", "coordinate_transform", "uncertainty_review"):
        text(doc.get(field), f"camera {field}")
    digest(doc.get("checkpoint_sha256"), "camera checkpoint")
    ids = frame_ids(doc.get("source_frame_ids"), count, "camera source_frame_ids")
    require(len(ids) >= min(9, count) and ids == sorted(ids) and {0, count - 1} <= set(ids),
            "camera needs endpoint-inclusive ordered samples across the original clip")
    estimates = doc.get("raw_estimates")
    require(isinstance(estimates, list) and len(estimates) == len(ids), "camera raw estimate coverage mismatch")
    seen = []
    for estimate in estimates:
        require(isinstance(estimate, dict), "camera raw estimate must be an object")
        seen.append(estimate.get("frame_id"))
        validate_intrinsics({name: estimate.get(name) for name in ("fx", "fy", "cx", "cy", "width", "height")}, width, height)
    require(seen == ids and all(type(i) is int for i in seen), "camera estimates do not follow source frame IDs")
    # Independently recompute aggregation: a status string is not a quality gate.
    medians = {key: statistics.median(row[key] for row in estimates) for key in ("fx", "fy", "cx", "cy")}
    if selected_intrinsics is not None:
        require(all(math.isclose(selected_intrinsics[key], medians[key], rel_tol=1e-9, abs_tol=1e-9)
                    for key in medians), "selected camera differs from the median of all raw estimates")
    for key in ("fx", "fy"):
        values = sorted(float(row[key]) for row in estimates)
        def percentile(fraction):
            position = (len(values) - 1) * fraction
            lower, upper = math.floor(position), math.ceil(position)
            return values[lower] + (values[upper] - values[lower]) * (position - lower)
        spread = (percentile(.9) - percentile(.1)) / medians[key]
        deviation = max(abs(v - medians[key]) for v in values) / medians[key]
        require(spread <= .25 and deviation <= .40, f"camera {key} estimates are inconsistent")


def validate_masks(root: Path, inventory: dict, episode: int, video_sha256: str,
                   count: int, width: int, height: int, target_id: str) -> dict:
    import numpy as np
    from PIL import Image

    manifest = read_json(root / "masks-manifest.json")
    require(manifest.get("schema_version") == "body_masks.v1", "unknown masks manifest schema")
    require(type(manifest.get("episode_index")) is int and manifest["episode_index"] == episode, "mask manifest episode mismatch")
    require(manifest.get("video_sha256") == video_sha256, "mask manifest source mismatch")
    files = manifest.get("files")
    require(isinstance(files, list) and len(files) == count, "mask manifest frame count mismatch")
    expected_names = [f"human_masks/{i:06d}.png" for i in range(count)]
    require([f.get("path") if isinstance(f, dict) else None for f in files] == expected_names,
            "mask manifest names must be exactly 000000.png through T-1 in order")
    actual_names = {p.relative_to(root).as_posix() for p in (root / "human_masks").iterdir()}
    require(actual_names == set(expected_names), "missing or extra human mask files")
    observations = read_json(root / "observations.json")
    require(observations.get("schema_version") == "body_observations.v1", "unknown observations schema")
    require(type(observations.get("episode_index")) is int and observations["episode_index"] == episode,
            "observations episode mismatch")
    require(observations.get("target_id") == target_id, "observations target identity mismatch")
    rows = observations.get("frames")
    require(isinstance(rows, list) and len(rows) == count, "observations frame coverage mismatch")
    areas = []
    for frame, (relative, entry, observation) in enumerate(zip(expected_names, files, rows)):
        path = root / relative
        require(type(entry.get("bytes")) is int and path.stat().st_size == entry["bytes"], f"mask byte size mismatch: {relative}")
        require(entry.get("sha256") == inventory.get(relative), f"mask identity mismatch: {relative}")
        require(isinstance(observation, dict) and type(observation.get("frame_id")) is int
                and observation["frame_id"] == frame, "observations must follow exact original frame IDs")
        require(observation.get("valid") is True, f"invalid/missing human observation at frame {frame}")
        if "target_id" in observation:
            require(observation["target_id"] == target_id, f"target switch at frame {frame}")
        confidence = observation.get("confidence")
        if confidence is not None:
            require(0 <= finite(confidence, "confidence") <= 1, "confidence outside [0,1]")
        with Image.open(path) as image:
            require(image.format == "PNG" and image.mode in {"L", "1"}, f"mask must be single-channel binary PNG: {relative}")
            require(image.size == (width, height), f"mask dimensions mismatch: {relative}")
            require(getattr(image, "n_frames", 1) == 1, f"animated mask is not allowed: {relative}")
            pixels = np.asarray(image)
            require(np.all((pixels == 0) | (pixels == 255)) or np.all((pixels == 0) | (pixels == 1)),
                    f"nonbinary mask pixels: {relative}")
            occupied = pixels != 0
            area = int(np.count_nonzero(occupied))
            require(0 < area < width * height, f"empty/full-image mask is not localization: {relative}")
            ys, xs = np.nonzero(occupied)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            require(bbox[2] - bbox[0] > 1 and bbox[3] - bbox[1] > 1,
                    f"mask bbox would trigger native Body full-image fallback: {relative}")
        require(type(observation.get("area_pixels")) is int and observation["area_pixels"] == area,
                f"observation area disagrees with mask at frame {frame}")
        declared_box = observation.get("bbox_xyxy")
        require(isinstance(declared_box, list) and all(type(x) is int for x in declared_box)
                and declared_box == bbox, f"observation bbox disagrees with mask at frame {frame}")
        areas.append(area)
    return {"masks_verified": count, "minimum_mask_area_pixels": min(areas), "maximum_mask_area_pixels": max(areas)}


def validate_review(doc: dict, episode: int, count: int, inventory: dict) -> None:
    require(doc.get("schema_version") == "body_input_review.v1", "unknown review schema")
    require(type(doc.get("episode_index")) is int and doc["episode_index"] == episode, "review episode mismatch")
    require(doc.get("status") == "approved", "visual review is not approved")
    text(doc.get("reviewer"), "actual reviewer")
    require(doc.get("review_method") in {"human_visual", "machine_assisted_visual"}, "missing actual visual review method")
    when = text(doc.get("reviewed_at"), "review timestamp")
    try:
        timestamp = datetime.fromisoformat(when.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HandoffError("review timestamp must be ISO8601") from exc
    require(timestamp.tzinfo is not None, "review timestamp must include timezone")
    ids = frame_ids(doc.get("sample_frame_ids"), count, "review sample_frame_ids")
    require({0, count // 2, count - 1} <= set(ids), "visual review must include beginning, middle and end")
    text(doc.get("findings"), "actual visual review findings")
    require(doc.get("unresolved_issues") == [], "visual review has unresolved issues")
    bindings = doc.get("reviewed_artifacts")
    require(isinstance(bindings, dict) and set(bindings) == REVIEW_ARTIFACTS, "review must bind every model input artifact")
    require(all(bindings[name] == inventory[name] for name in REVIEW_ARTIFACTS), "visual review is stale after input changes")


def check_episode(root: Path, dataset_root: Path, raw_root: Path, entry: dict,
                  source_entries: dict, source_files: dict) -> dict:
    """Fully check one episode; callers supply a verified raw-manifest context."""
    episode = entry["episode_index"]
    result = {"episode_index": episode, "inputs_structurally_verified": False,
              "visual_review_verified": False, "body_inputs_verified": False, "errors": []}
    try:
        inventory = verify_checksum_inventory(root)
        identity = read_json(root / "input.json")
        require(identity.get("schema_version") == "body_input.v1" and identity.get("profile") == PROFILE,
                "unsupported Body input schema/profile")
        require(type(identity.get("episode_index")) is int and identity["episode_index"] == episode, "input episode mismatch")
        target_id = text(identity.get("target_id"), "target identity")
        validate_producer(identity.get("producer"))
        source = identity.get("source")
        require(isinstance(source, dict), "missing source identity")
        video_path, parquet_path = raw.episode_paths(episode)
        require(source.get("repository") == raw.REPOSITORY and source.get("revision") == raw.REVISION,
                "input must reference pinned Track 1 source")
        require(source.get("video_path") == video_path == entry["video_path"], "wrong episode video path")
        for relative in (video_path, parquet_path):
            verified = raw.verify_file(dataset_root / relative, source_entries[relative])
            require(verified == source_files[relative], f"raw preparation source identity changed: {relative}")
        video_sha = source_files[video_path]["sha256"]
        require(source.get("video_sha256") == video_sha, "input source video hash mismatch")
        require(source.get("timeline_sha256") == entry["timeline_sha256"] == inventory["timeline.json"],
                "input timeline identity mismatch")
        timeline_path = raw.safe_relative(entry["timeline_path"])
        require(raw.sha256(raw_root / timeline_path) == entry["timeline_sha256"], "verified raw timeline has changed")
        bindings = identity.get("artifacts")
        require(isinstance(bindings, dict) and set(bindings) == ARTIFACTS, "input must bind all model input artifacts")
        require(all(bindings[name] == inventory[name] for name in ARTIFACTS), "input artifact identity mismatch")
        timeline = read_json(root / "timeline.json")
        count, width, height = (entry[name] for name in ("frame_count", "width", "height"))
        require(all(type(i) is int and i > 0 for i in (count, width, height)), "invalid original dimensions/frame count")
        require(timeline.get("schema_version") == "track1_original_timeline.v1"
                and type(timeline.get("episode_index")) is int and timeline["episode_index"] == episode,
                "wrong episode timeline")
        require(timeline.get("video_sha256") == video_sha and timeline.get("parquet_sha256") == source_files[parquet_path]["sha256"],
                "timeline source identity mismatch")
        decoded = raw.decode_timeline(dataset_root / video_path, (width, height))
        for field in ("frame_count", "width", "height", "frame_ids", "pts", "time_base", "timestamps_seconds"):
            require(timeline.get(field) == decoded[field], f"timeline differs from original decode: {field}")
        require(count == decoded["frame_count"], "raw manifest frame count mismatch")
        require(timeline.get("first_source_frame_id") == 0 and entry.get("first_source_frame_id") == 0,
                "original reference frame must be preserved")
        import pandas as pd
        raw.validate_table(pd.read_parquet(dataset_root / parquet_path), episode, decoded, count)
        intrinsics = read_json(root / "intrinsics.json")
        validate_intrinsics(intrinsics, width, height)
        validate_camera(read_json(root / "camera-estimation.json"), episode, video_sha, count, width, height, intrinsics)
        result.update(validate_masks(root, inventory, episode, video_sha, count, width, height, target_id))
        result["inputs_structurally_verified"] = True
        validate_review(read_json(root / "review.json"), episode, count, inventory)
        result["visual_review_verified"] = True
        # Recheck generated files after potentially lengthy image reads.
        require(verify_checksum_inventory(root) == inventory, "package changed during verification")
        for relative in (video_path, parquet_path):
            require(raw.verify_file(dataset_root / relative, source_entries[relative]) == source_files[relative],
                    f"source changed during verification: {relative}")
        result.update({"body_inputs_verified": True, "frame_count": count, "video_sha256": video_sha,
                       "input_sha256": inventory["input.json"], "review_sha256": inventory["review.json"]})
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        result["errors"].append(str(exc))
    return result


def check_package(package_root: Path, dataset_root: Path, raw_root: Path, source_inventory: Path,
                  episodes: list[int] | None = None) -> dict:
    selected = list(range(30)) if episodes is None else episodes
    frame_ids(selected, 30, "requested episodes")
    for path in (package_root, dataset_root, raw_root, source_inventory):
        raw.reject_links(path)
    require(package_root.is_dir(), "package root does not exist")
    source_entries = raw.load_inventory(source_inventory)
    manifest = read_json(raw_root / "manifest.json")
    checks = read_json(raw_root / "checks.json")
    require(manifest.get("schema_version") == "track1_raw_preparation.v1" and manifest.get("raw_data_ready") is True,
            "raw preparation is not ready")
    require(checks.get("schema_version") == "track1_raw_checks.v1" and checks.get("passed") is True
            and checks.get("manifest_sha256") == raw.sha256(raw_root / "manifest.json"), "raw preparation checks/manifest mismatch")
    require(manifest.get("repository") == raw.REPOSITORY and manifest.get("revision") == raw.REVISION
            and manifest.get("track") == "track_1", "raw preparation source pin mismatch")
    require(manifest.get("source_inventory_sha256") == raw.sha256(source_inventory), "raw source inventory changed")
    records = manifest.get("episodes")
    require(isinstance(records, list) and len(records) == 30, "raw manifest must contain all 30 episodes")
    by_id = {}
    for row in records:
        require(isinstance(row, dict) and type(row.get("episode_index")) is int and 0 <= row["episode_index"] < 30
                and row["episode_index"] not in by_id, "raw manifest episode coverage is invalid")
        by_id[row["episode_index"]] = row
    sources = manifest.get("source_files")
    require(isinstance(sources, dict), "missing raw source identities")
    episode_results = []
    for episode in selected:
        result = check_episode(package_root / f"episode_{episode:06d}", dataset_root, raw_root,
                               by_id[episode], source_entries, sources)
        episode_results.append(result)
        print(f"episode_{episode:06d}: {'READY' if result['body_inputs_verified'] else 'BLOCKED'}"
              + (": " + "; ".join(result["errors"]) if result["errors"] else ""), flush=True)
    passed = all(result["body_inputs_verified"] for result in episode_results)
    return {"schema_version": "body_handoff_checks.v1", "checked_at": datetime.now(timezone.utc).isoformat(),
            "profile": PROFILE, "body_source_revision": BODY_SOURCE_REVISION, "requested_episodes": selected,
            "passed": passed, "all_30_body_inputs_verified": passed and set(selected) == set(range(30)),
            "body_execution_completed": False, "scientific_accuracy_certified": False,
            "videos_verified": sum(r["body_inputs_verified"] for r in episode_results),
            "frames_verified": sum(r.get("frame_count", 0) for r in episode_results if r["body_inputs_verified"]),
            "source_inventory_sha256": raw.sha256(source_inventory),
            "raw_manifest_sha256": raw.sha256(raw_root / "manifest.json"), "episodes": episode_results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--raw-preparation-root", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, default=Path(__file__).resolve().parents[1] / "configs/track1-source.json")
    parser.add_argument("--episodes", type=int, nargs="+", help="Diagnostic subset; omit to require all 30 episodes")
    parser.add_argument("--output", type=Path, help="Optional fresh check report path; never overwrites evidence")
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            raw.reject_links(args.output)
            require(not args.output.exists(), "output report already exists; choose a fresh path")
        report = check_package(args.package_root, args.dataset_root, args.raw_preparation_root,
                               args.source_inventory, args.episodes)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps({key: report[key] for key in ("passed", "all_30_body_inputs_verified", "videos_verified", "frames_verified")}))
        return 0 if report["passed"] else 1
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Body handoff gate failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
