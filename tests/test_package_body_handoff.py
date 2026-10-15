"""Portable packaging tests using only independently generated synthetic inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest

SPEC = importlib.util.spec_from_file_location("package_body_handoff", Path(__file__).parents[1] / "tools/package_body_handoff.py")
pack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pack)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def synthetic_source(tmp_path):
    av = pytest.importorskip("av")
    dataset = tmp_path / "data/v2d/track_1"
    dataset.mkdir(parents=True)
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
    for name in pack.raw.expected_paths():
        path = dataset / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic metadata")
    for episode in range(30):
        video_name, table_name = pack.raw.episode_paths(episode)
        shutil.copyfile(source_video, dataset / video_name)
        pd.DataFrame({"frame_index": [0, 1, 2], "episode_index": [episode] * 3,
                      "timestamp": [0.0, 1 / 30, 2 / 30]}).to_parquet(dataset / table_name)
    write(dataset / "meta/info.json", {"total_episodes": 30, "total_videos": 30, "total_frames": 90,
          "fps": 30, "features": {pack.raw.VIDEO_KEY: {"shape": [12, 16, 3]}}})
    (dataset / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": i, "length": 3}) for i in range(30)))
    (dataset / "meta/episodes_metadata.jsonl").write_text("\n".join(json.dumps({"episode_index": i, "video_key": pack.raw.VIDEO_KEY}) for i in range(30)))
    entries = []
    for name in sorted(pack.raw.expected_paths()):
        content = (dataset / name).read_bytes()
        entries.append({"type": "file", "path": "track_1/" + name, "size": len(content),
                        "oid": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()})
    lock = tmp_path / "synthetic-source.json"
    write(lock, {"repository": pack.raw.REPOSITORY, "revision": pack.raw.REVISION, "files": entries})
    raw_root = tmp_path / "data/prepared/original"
    manifest = pack.raw.prepare(dataset, lock, raw_root)
    return dataset, lock, raw_root, manifest


@pytest.fixture
def producer(tmp_path):
    root = tmp_path / "producer"
    root.mkdir()
    (root / "human_masks").mkdir()
    (root / "mask-qa").mkdir()
    files, rows = [], []
    for frame in range(3):
        pixels = np.zeros((12, 16), dtype=np.uint8)
        pixels[2:10, 4:12] = 255
        relative = f"human_masks/{frame:06d}.png"
        Image.fromarray(pixels).save(root / relative)
        files.append({"path": relative, "bytes": (root / relative).stat().st_size, "sha256": pack.raw.sha256(root / relative)})
        rows.append({"frame_id": frame, "valid": True, "area_pixels": 64, "bbox_xyxy": [4, 2, 12, 10], "confidence": None})
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(root / "mask-qa/review-001.jpg")
    write(root / "observations.json", {"schema_version": "body_observations.v1", "episode_index": 0,
          "target_id": "human_0", "frames": rows})
    write(root / "masks-manifest.json", {"schema_version": "body_masks.v1", "episode_index": 0,
          "video_sha256": "a" * 64, "files": files})
    intrinsics = {"fx": 14, "fy": 14, "cx": 7.5, "cy": 5.5, "width": 16, "height": 12}
    write(root / "intrinsics.json", intrinsics)
    write(root / "camera-estimation.json", {"schema_version": "body_camera_estimation.v1", "episode_index": 0,
          "video_sha256": "a" * 64, "status": "camera_estimate_consistent", "method": "synthetic camera",
          "code_revision": "synthetic-v1", "checkpoint_sha256": "c" * 64, "aggregation": "median",
          "coordinate_transform": "original pixels", "uncertainty_review": "Independent fixture only",
          "source_frame_ids": [0, 1, 2], "raw_estimates": [{"frame_id": i, **intrinsics} for i in range(3)]})
    write(root / "masks-provenance.json", {"schema_version": "body_masks_provenance.v1", "episode_index": 0,
          "method": "independent synthetic fixture", "code_revision": "synthetic-v1", "checkpoint_sha256": "b" * 64,
          "environment": {"fixture": True}, "source_video_sha256": "a" * 64, "timeline_sha256": "d" * 64,
          "visual_review_frame_ids": [0, 1, 2], "invalid_frame_ids": []})
    receipt = {"schema_version": "body_visual_review_receipt.v1", "episode_index": 0, "video_sha256": "a" * 64,
          "target_id": "human_0", "status": "approved", "reviewer": "independent synthetic fixture",
          "review_method": "machine_assisted_visual", "reviewed_at": "2026-10-16T00:00:00+08:00",
          "sample_frame_ids": [0, 1, 2], "findings": "Synthetic input; this is no actual visual approval",
          "unresolved_issues": [], "flag_resolutions": [{"frame_id": 1, "status": "resolved", "findings": "fixture"}]}
    receipt["reviewed_artifacts"] = {name: pack.raw.sha256(root / name)
          for name in pack.RECEIPT_ARTIFACTS | {"mask-qa/review-001.jpg"}}
    return root, receipt


@pytest.mark.parametrize("field,value", [("status", "pending"), ("sample_frame_ids", [0, 2]),
    ("reviewer", ""), ("unresolved_issues", ["target switches"]), ("video_sha256", "f" * 64)])
def test_assembly_requires_supplied_complete_actual_review(producer, field, value):
    root, receipt = producer
    receipt[field] = value
    with pytest.raises(pack.gate.HandoffError):
        pack.validate_receipt(receipt, root, {"episode_index": 0, "frame_count": 3, "timeline_sha256": "d" * 64}, "a" * 64)


def test_receipt_cannot_approve_changed_or_unseen_evidence(producer):
    root, receipt = producer
    row = {"episode_index": 0, "frame_count": 3, "timeline_sha256": "d" * 64}
    pack.validate_receipt(receipt, root, row, "a" * 64)
    (root / "mask-qa/targets-001.jpg").write_bytes((root / "mask-qa/review-001.jpg").read_bytes())
    with pytest.raises(pack.gate.HandoffError, match="every QA"):
        pack.validate_receipt(receipt, root, row, "a" * 64)
    receipt["reviewed_artifacts"]["mask-qa/targets-001.jpg"] = pack.raw.sha256(root / "mask-qa/targets-001.jpg")
    (root / "observations.json").write_text("changed")
    with pytest.raises(pack.gate.HandoffError, match="stale"):
        pack.validate_receipt(receipt, root, row, "a" * 64)


def simple_archive(tmp_path, payload=b"synthetic"):
    path = tmp_path / "fixture.zip"
    manifest = {"schema_version": "body_handoff_bundle.v1", "all_30_body_inputs_verified": True,
                "files": [{"path": "example.txt", "bytes": len(payload), "sha256": pack.byte_sha256(payload)}]}
    metadata = pack.json_bytes(manifest)
    sums = f"{pack.byte_sha256(metadata)}  bundle-manifest.json\n{pack.byte_sha256(payload)}  example.txt\n".encode()
    with zipfile.ZipFile(path, "x") as archive:
        archive.writestr("example.txt", payload)
        archive.writestr("bundle-manifest.json", metadata)
        archive.writestr("SHA256SUMS", sums)
    return path


@pytest.mark.parametrize("name", ["../escape", "C:/escape", "track_2/pixels", "EXAMPLE.TXT"])
def test_archive_rejects_unsafe_or_colliding_names_before_extraction(tmp_path, name):
    path = simple_archive(tmp_path)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(name, b"bad")
    output = tmp_path / "extract"
    with pytest.raises(ValueError):
        pack.extract_verified(path, output)
    assert not output.exists()


def test_archive_rejects_symlink_and_wrong_bytes(tmp_path):
    path = simple_archive(tmp_path)
    with zipfile.ZipFile(path, "a") as archive:
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "outside")
    with pytest.raises(pack.gate.HandoffError, match="nonregular"):
        pack.verify_archive(path)
    altered = tmp_path / "altered.zip"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(altered, "w") as output:
        for item in source.infolist():
            if item.filename != "link":
                output.writestr(item.filename, b"altered!!" if item.filename == "example.txt" else source.read(item))
    with pytest.raises(pack.gate.HandoffError, match="hash/size"):
        pack.verify_archive(altered)


def test_extraction_never_overwrites_existing_directory(tmp_path):
    archive = simple_archive(tmp_path)
    output = tmp_path / "extract"
    output.mkdir()
    (output / "keep").write_text("keep")
    with pytest.raises(pack.gate.HandoffError, match="already exists"):
        pack.extract_verified(archive, output)
    assert (output / "keep").read_text() == "keep"


def test_all30_archive_roundtrip_executes_bundled_gate_and_preserves_reviews(tmp_path, producer):
    dataset, lock, raw_root, manifest = synthetic_source(tmp_path)
    template, template_receipt = producer
    inputs, reviews = tmp_path / "inputs", tmp_path / "reviews"
    inputs.mkdir()
    reviews.mkdir()
    for episode, row in enumerate(manifest["episodes"]):
        target = inputs / f"episode_{episode:06d}"
        shutil.copytree(template, target)
        video_sha = manifest["source_files"][row["video_path"]]["sha256"]
        for name in ("observations.json", "masks-manifest.json", "masks-provenance.json", "camera-estimation.json"):
            doc = pack.gate.read_json(target / name)
            doc["episode_index"] = episode
            if "video_sha256" in doc:
                doc["video_sha256"] = video_sha
            if name == "masks-provenance.json":
                doc.update(source_video_sha256=video_sha, timeline_sha256=row["timeline_sha256"])
            write(target / name, doc)
        receipt = dict(template_receipt, episode_index=episode, video_sha256=video_sha)
        receipt["reviewed_artifacts"] = {name: pack.raw.sha256(target / name) for name in template_receipt["reviewed_artifacts"]}
        write(reviews / f"episode_{episode:06d}.json", receipt)
    last_review = reviews / "episode_000029.json"
    original_last_review = last_review.read_bytes()
    pending = pack.gate.read_json(last_review)
    pending["status"] = "pending"
    write(last_review, pending)
    unapproved = tmp_path / "must-not-exist"
    with pytest.raises(pack.gate.HandoffError, match="not approved"):
        pack.assemble(inputs, dataset, raw_root, reviews, lock, unapproved)
    assert not unapproved.exists()
    last_review.write_bytes(original_last_review)
    assembled = tmp_path / "assembled"
    report = pack.assemble(inputs, dataset, raw_root, reviews, lock, assembled)
    assert report["all_30_body_inputs_verified"] is True
    original_receipt = (reviews / "episode_000016.json").read_bytes()
    assert (assembled / "episode_000016/visual-review-receipt.json").read_bytes() == original_receipt
    review = pack.gate.read_json(assembled / "episode_000016/review.json")
    assert review["reviewed_at"] == template_receipt["reviewed_at"]
    assert review["sample_frame_ids"] == template_receipt["sample_frame_ids"]
    archive, extracted = tmp_path / "delivery.zip", tmp_path / "received"
    verified = pack.package(assembled, dataset, raw_root, lock, archive, extracted)
    assert verified["extracted_receiver_passed"] is True
    assert verified["all_30_body_inputs_verified"] is True
    assert verified["body_execution_completed"] is False
    assert pack.raw.sha256(archive) == verified["sha256"]
    assert not archive.with_name("delivery.zip.incomplete").exists()
    restored_raw = pack.gate.read_json(extracted / pack.PREPARED_PREFIX / "manifest.json")
    assert (extracted / pack.PREPARED_PREFIX / restored_raw["dataset_root_relative"]).resolve() == (extracted / pack.RAW_PREFIX).resolve()
    assert not list(extracted.rglob("__pycache__"))
    # Every byte, including review receipts and the actual checker source, is inventoried.
    shipped = pack.verify_archive(archive)
    names = {row["path"] for row in shipped["files"]}
    assert "runtime/check_body_handoff.py" in names
    assert "receiving-guide.md" in names
    assert "body-inputs/episode_000029/human_masks/000002.png" in names
    assert "body-inputs/episode_000016/visual-review-receipt.json" in names
    assert (extracted / "body-inputs/episode_000016/visual-review-receipt.json").read_bytes() == original_receipt


def test_failed_extracted_receiver_is_not_a_success(tmp_path):
    extracted = tmp_path / "received"
    (extracted / "runtime").mkdir(parents=True)
    (extracted / "runtime/check_body_handoff.py").write_text("raise SystemExit(1)\n")
    with pytest.raises(pack.gate.HandoffError, match="receiver gate failed"):
        pack.run_receiver(extracted, tmp_path / "receiver-checks.json")


@pytest.mark.parametrize("mutation", ["stale_input", "wrong_frame_count", "duplicate_episode", "missing_episode", "wrong_pin", "failed_episode", "wrong_total"])
def test_optional_native_evidence_must_match_current_all30_inputs(mutation):
    checks = {"frames_verified": 90, "episodes": [{"episode_index": index, "frame_count": 3,
              "input_sha256": hashlib.sha256(f"synthetic-{index}".encode()).hexdigest()} for index in range(30)]}
    report = {"schema_version": "body_native_reader_checks.v1", "all_30_native_readers_passed": True,
              "upstream_common_source": {"repository": "nvidia-isaac/video_to_data", "revision": pack.gate.BODY_SOURCE_REVISION},
              "frames_verified": 90, "episodes": [{"episode_index": row["episode_index"], "frames_verified": 3,
              "passed": True, "input_sha256": row["input_sha256"]} for row in checks["episodes"]]}
    pack.validate_native_report(report, checks)
    if mutation == "stale_input":
        report["episodes"][16]["input_sha256"] = "0" * 64
    elif mutation == "wrong_frame_count":
        report["episodes"][16]["frames_verified"] = 2
    elif mutation == "duplicate_episode":
        report["episodes"][16]["episode_index"] = 15
    elif mutation == "missing_episode":
        report["episodes"].pop()
    elif mutation == "wrong_pin":
        report["upstream_common_source"]["revision"] = "0" * 40
    elif mutation == "failed_episode":
        report["episodes"][16]["passed"] = False
    else:
        report["frames_verified"] = 91
    with pytest.raises(pack.gate.HandoffError):
        pack.validate_native_report(report, checks)
