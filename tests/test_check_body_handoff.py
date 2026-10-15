"""Independent synthetic tests for model-input handoff; no challenge pixels."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location("check_body_handoff", Path(__file__).parents[1] / "tools/check_body_handoff.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def rehash(root):
    """Update storage identity without silently refreshing any review binding."""
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "checks.json"})
    (root / "SHA256SUMS").write_text("".join(f"{gate.raw.sha256(p)}  {p.relative_to(root).as_posix()}\n" for p in files))


def bind_inputs(root):
    identity = json.loads((root / "input.json").read_text())
    identity["artifacts"] = {name: gate.raw.sha256(root / name) for name in gate.ARTIFACTS}
    write(root / "input.json", identity)
    review = json.loads((root / "review.json").read_text())
    review["reviewed_artifacts"] = {name: gate.raw.sha256(root / name) for name in gate.REVIEW_ARTIFACTS}
    write(root / "review.json", review)
    rehash(root)


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "episode_000000"
    root.mkdir()
    (root / "human_masks").mkdir()
    video_sha = "a" * 64
    files, rows = [], []
    for index in range(3):
        pixels = np.zeros((12, 16), dtype=np.uint8)
        pixels[2:10, 4:12] = 255
        relative = f"human_masks/{index:06d}.png"
        Image.fromarray(pixels).save(root / relative)
        files.append({"path": relative, "bytes": (root / relative).stat().st_size,
                      "sha256": gate.raw.sha256(root / relative)})
        rows.append({"frame_id": index, "valid": True, "area_pixels": 64,
                     "bbox_xyxy": [4, 2, 12, 10], "confidence": None})
    write(root / "masks-manifest.json", {"schema_version": "body_masks.v1", "episode_index": 0,
                                         "video_sha256": video_sha, "files": files})
    write(root / "observations.json", {"schema_version": "body_observations.v1", "episode_index": 0,
                                       "target_id": "human_0", "frames": rows})
    intrinsics = {"fx": 14.0, "fy": 14.0, "cx": 7.5, "cy": 5.5, "width": 16, "height": 12}
    write(root / "intrinsics.json", intrinsics)
    camera = {"schema_version": "body_camera_estimation.v1", "episode_index": 0,
              "video_sha256": video_sha, "status": "camera_estimate_consistent", "method": "synthetic camera",
              "code_revision": "independent synthetic fixture", "checkpoint_sha256": "c" * 64,
              "source_frame_ids": [0, 1, 2], "raw_estimates": [{"frame_id": i, **intrinsics} for i in range(3)],
              "aggregation": "median", "coordinate_transform": "original pixels", "uncertainty_review": "synthetic only"}
    write(root / "camera-estimation.json", camera)
    write(root / "timeline.json", {"synthetic_fixture": True})
    identity = {"schema_version": "body_input.v1", "profile": gate.PROFILE, "episode_index": 0,
                "target_id": "human_0", "source": {"repository": gate.raw.REPOSITORY, "revision": gate.raw.REVISION,
                "video_path": gate.raw.episode_paths(0)[0], "video_sha256": video_sha,
                "timeline_sha256": gate.raw.sha256(root / "timeline.json")},
                "producer": {"method": "independent synthetic fixture", "code_revision": "synthetic-v1",
                             "checkpoint_sha256": "b" * 64, "environment": {"fixture": True}}}
    write(root / "input.json", identity)
    write(root / "review.json", {"schema_version": "body_input_review.v1", "episode_index": 0,
          "status": "approved", "reviewer": "synthetic test", "review_method": "machine_assisted_visual",
          "reviewed_at": "2026-10-15T00:00:00+00:00", "sample_frame_ids": [0, 1, 2],
          "findings": "Independent synthetic input, not an actual review", "unresolved_issues": []})
    bind_inputs(root)
    return root


def masks(root):
    return gate.validate_masks(root, gate.verify_checksum_inventory(root), 0, "a" * 64, 3, 16, 12, "human_0")


def test_complete_binary_masks_and_null_confidence_are_valid(package):
    assert masks(package) == {"masks_verified": 3, "minimum_mask_area_pixels": 64, "maximum_mask_area_pixels": 64}


@pytest.mark.parametrize("width,height,usable", [(1, 8, False), (8, 1, False), (1, 1, False),
    (2, 2, True), (3, 3, True), (2, 8, True), (8, 2, True)])
def test_bbox_matches_exact_native_exclusive_max_fallback_rule(package, width, height, usable):
    pixels = np.zeros((12, 16), dtype=np.uint8)
    pixels[2:2 + height, 4:4 + width] = 255
    path = package / "human_masks/000001.png"
    Image.fromarray(pixels).save(path)
    manifest = gate.read_json(package / "masks-manifest.json")
    manifest["files"][1].update(bytes=path.stat().st_size, sha256=gate.raw.sha256(path))
    write(package / "masks-manifest.json", manifest)
    observations = gate.read_json(package / "observations.json")
    observations["frames"][1].update(area_pixels=width * height, bbox_xyxy=[4, 2, 4 + width, 2 + height])
    write(package / "observations.json", observations)
    bind_inputs(package)
    spec = importlib.util.spec_from_file_location("native_reader_bbox_test", Path(__file__).parents[1] / "tools/check_body_native_readers.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    if usable:
        assert masks(package)["masks_verified"] == 3
        native.require_usable_mask_bbox(pixels)
    else:
        with pytest.raises(gate.HandoffError, match="native Body full-image fallback"):
            masks(package)
        with pytest.raises(ValueError, match="full-image fallback"):
            native.require_usable_mask_bbox(pixels)


@pytest.mark.parametrize("kind", ["missing", "extra", "wrong_dimensions", "nonbinary", "rgb", "empty", "full", "corrupt"])
def test_actual_mask_content_and_exact_coverage_enforced(package, kind):
    path = package / "human_masks/000002.png"
    if kind == "missing":
        path.unlink()
    elif kind == "extra":
        (package / "human_masks/000003.png").write_bytes(path.read_bytes())
    elif kind == "corrupt":
        path.write_bytes(b"not a PNG")
    else:
        pixels = np.zeros((12, 16), dtype=np.uint8)
        pixels[2:10, 4:12] = 255
        if kind == "wrong_dimensions":
            pixels = pixels[:6]
        elif kind == "nonbinary":
            pixels[3, 6] = 125
        elif kind == "rgb":
            pixels = np.stack([pixels] * 3, axis=-1)
        elif kind == "empty":
            pixels[:] = 0
        elif kind == "full":
            pixels[:] = 255
        Image.fromarray(pixels).save(path)
    manifest = json.loads((package / "masks-manifest.json").read_text())
    if path.exists():
        manifest["files"][2].update({"bytes": path.stat().st_size, "sha256": gate.raw.sha256(path)})
        write(package / "masks-manifest.json", manifest)
    bind_inputs(package)
    with pytest.raises((gate.HandoffError, OSError)):
        masks(package)


@pytest.mark.parametrize("field,value", [("frame_id", 4), ("valid", False), ("area_pixels", 1),
    ("bbox_xyxy", [4, 2, 11, 9]), ("confidence", 1.5), ("target_id", "other_person")])
def test_observations_are_verified_against_mask_pixels(package, field, value):
    observations = json.loads((package / "observations.json").read_text())
    observations["frames"][1][field] = value
    write(package / "observations.json", observations)
    bind_inputs(package)
    with pytest.raises(gate.HandoffError):
        masks(package)


@pytest.mark.parametrize("mutation", ["hash", "omitted", "extra", "traversal", "duplicate"])
def test_checksum_inventory_fails_closed(package, mutation):
    path = package / "SHA256SUMS"
    lines = path.read_text().splitlines()
    if mutation == "hash":
        lines[0] = "0" * 64 + lines[0][64:]
    elif mutation == "omitted":
        lines.pop()
    elif mutation == "extra":
        lines.append("0" * 64 + "  missing.json")
    elif mutation == "traversal":
        lines[0] = "0" * 64 + "  ../outside.json"
    else:
        lines.append(lines[0])
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        gate.verify_checksum_inventory(package)


@pytest.mark.parametrize("field,value", [("fx", 0), ("fy", float("nan")), ("cx", 16),
    ("cy", -1), ("width", 8), ("height", True)])
def test_intrinsics_are_finite_positive_in_original_grid(field, value):
    camera = {"fx": 14, "fy": 14, "cx": 7.5, "cy": 5.5, "width": 16, "height": 12}
    camera[field] = value
    with pytest.raises(gate.HandoffError):
        gate.validate_intrinsics(camera, 16, 12)


@pytest.mark.parametrize("field,value", [("status", "pending"), ("video_sha256", "f" * 64),
    ("source_frame_ids", [0, 0, 1]), ("checkpoint_sha256", "missing"), ("raw_estimates", [])])
def test_camera_requires_actual_source_bound_estimates(package, field, value):
    camera = json.loads((package / "camera-estimation.json").read_text())
    camera[field] = value
    with pytest.raises(gate.HandoffError):
        gate.validate_camera(camera, 0, "a" * 64, 3, 16, 12)


def test_machine_assisted_review_is_named_and_binds_current_artifacts(package):
    inventory = gate.verify_checksum_inventory(package)
    review = gate.read_json(package / "review.json")
    gate.validate_review(review, 0, 3, inventory)
    changed = dict(inventory)
    changed["masks-manifest.json"] = "f" * 64
    with pytest.raises(gate.HandoffError, match="stale"):
        gate.validate_review(review, 0, 3, changed)


@pytest.mark.parametrize("field,value", [("status", "pending"), ("reviewer", ""),
    ("sample_frame_ids", [0]), ("unresolved_issues", ["target switched"]), ("reviewed_at", "2026-10-15")])
def test_review_has_real_identity_coverage_and_no_unresolved_findings(package, field, value):
    review = gate.read_json(package / "review.json")
    review[field] = value
    with pytest.raises(gate.HandoffError):
        gate.validate_review(review, 0, 3, gate.verify_checksum_inventory(package))


@pytest.mark.parametrize("contents", ['{"x":NaN}', '{"x":1,"x":2}', '[]'])
def test_json_disallows_nonfinite_ambiguous_or_nonobject_input(tmp_path, contents):
    path = tmp_path / "input.json"
    path.write_text(contents)
    with pytest.raises(gate.HandoffError):
        gate.read_json(path)


def test_missing_episode_returns_explicit_blocker(tmp_path):
    result = gate.check_episode(tmp_path / "missing", tmp_path, tmp_path, {"episode_index": 17}, {}, {})
    assert result["body_inputs_verified"] is False
    assert result["inputs_structurally_verified"] is False
    assert result["episode_index"] == 17
    assert result["errors"]


def test_review_cannot_be_refreshed_by_rehashing_storage_only(package):
    observations = json.loads((package / "observations.json").read_text())
    observations["producer_note"] = "changed observation metadata"
    write(package / "observations.json", observations)
    rehash(package)
    with pytest.raises(gate.HandoffError, match="stale"):
        gate.validate_review(gate.read_json(package / "review.json"), 0, 3, gate.verify_checksum_inventory(package))


def test_cli_never_overwrites_existing_report(tmp_path):
    report = tmp_path / "keep.json"
    report.write_text("keep")
    assert gate.main(["--package-root", str(tmp_path), "--dataset-root", str(tmp_path),
        "--raw-preparation-root", str(tmp_path), "--output", str(report)]) == 1
    assert report.read_text() == "keep"


def test_camera_consistency_and_aggregation_are_recomputed(package):
    camera = gate.read_json(package / "camera-estimation.json")
    intrinsics = gate.read_json(package / "intrinsics.json")
    intrinsics["fx"] += 1
    with pytest.raises(gate.HandoffError, match="median"):
        gate.validate_camera(camera, 0, "a" * 64, 3, 16, 12, intrinsics)
    camera["raw_estimates"][1]["fx"] *= 2
    with pytest.raises(gate.HandoffError, match="inconsistent"):
        gate.validate_camera(camera, 0, "a" * 64, 3, 16, 12)


def test_portable_all30_gate_and_subset_reporting(package, tmp_path):
    """Encode tiny independent clips, relocate packages, then verify real bytes."""
    av = pytest.importorskip("av")
    import pandas as pd

    dataset = tmp_path / "raw"
    dataset.mkdir()
    source_video = tmp_path / "synthetic.mp4"
    with av.open(str(source_video), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height = 16, 12
        stream.pix_fmt = "yuv420p"
        for frame in range(3):
            rgb = av.VideoFrame.from_ndarray(np.full((12, 16, 3), frame * 40, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(rgb):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    for name in gate.raw.expected_paths():
        path = dataset / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic metadata")
    for episode in range(30):
        video_name, table_name = gate.raw.episode_paths(episode)
        shutil.copyfile(source_video, dataset / video_name)
        pd.DataFrame({"frame_index": [0, 1, 2], "episode_index": [episode] * 3,
                      "timestamp": [0.0, 1 / 30, 2 / 30]}).to_parquet(dataset / table_name)
    write(dataset / "meta/info.json", {"total_episodes": 30, "total_videos": 30, "total_frames": 90,
          "fps": 30, "features": {gate.raw.VIDEO_KEY: {"shape": [12, 16, 3]}}})
    (dataset / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": i, "length": 3}) for i in range(30)))
    (dataset / "meta/episodes_metadata.jsonl").write_text("\n".join(json.dumps({"episode_index": i, "video_key": gate.raw.VIDEO_KEY}) for i in range(30)))
    entries = []
    for name in sorted(gate.raw.expected_paths()):
        content = (dataset / name).read_bytes()
        entries.append({"type": "file", "path": "track_1/" + name, "size": len(content),
                        "oid": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()})
    lock = tmp_path / "synthetic-source.json"
    write(lock, {"repository": gate.raw.REPOSITORY, "revision": gate.raw.REVISION, "files": entries})
    prepared = tmp_path / "prepared"
    manifest = gate.raw.prepare(dataset, lock, prepared)
    packages = tmp_path / "portable-packages"
    for episode in range(30):
        root = packages / f"episode_{episode:06d}"
        shutil.copytree(package, root)
        row = manifest["episodes"][episode]
        shutil.copyfile(prepared / row["timeline_path"], root / "timeline.json")
        video_sha = gate.raw.sha256(dataset / row["video_path"])
        for name in ("observations.json", "masks-manifest.json", "camera-estimation.json", "review.json", "input.json"):
            doc = gate.read_json(root / name)
            doc["episode_index"] = episode
            if name in {"masks-manifest.json", "camera-estimation.json"}:
                doc["video_sha256"] = video_sha
            if name == "input.json":
                doc["source"].update(video_path=row["video_path"], video_sha256=video_sha,
                                      timeline_sha256=row["timeline_sha256"])
            write(root / name, doc)
        bind_inputs(root)
    report = gate.check_package(packages, dataset, prepared, lock)
    assert report["all_30_body_inputs_verified"] is True
    assert report["videos_verified"] == 30 and report["frames_verified"] == 90
    assert report["body_execution_completed"] is False
    assert report["scientific_accuracy_certified"] is False
    subset = gate.check_package(packages, dataset, prepared, lock, [16, 12])
    assert subset["passed"] is True and subset["all_30_body_inputs_verified"] is False
    # Mutating actual source bytes must fail even when the input package is intact.
    source_path = dataset / gate.raw.episode_paths(29)[0]
    original = source_path.read_bytes()
    source_path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    blocked = gate.check_package(packages, dataset, prepared, lock, [29])
    assert blocked["passed"] is False
    assert "checksum" in blocked["episodes"][0]["errors"][0]
    source_path.write_bytes(original)
    # Missing the last mask blocks the complete handoff, not only that frame.
    (packages / "episode_000029/human_masks/000002.png").unlink()
    blocked = gate.check_package(packages, dataset, prepared, lock)
    assert blocked["all_30_body_inputs_verified"] is False
    assert blocked["videos_verified"] == 29
