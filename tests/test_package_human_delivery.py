"""Portable transfer and declaration checks using independent synthetic arrays."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v2hoi.human_result import write_human_result

SPEC = importlib.util.spec_from_file_location("package_human_delivery", ROOT / "tools/package_human_delivery.py")
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_batch(tmp_path):
    """No challenge asset is read: geometry, bytes and receipts are invented here."""
    batch = tmp_path / "batch"
    episode = batch / "episode_000016"
    outputs = {name: episode / name / "attempt-0001" for name in package.STAGES}
    for directory in outputs.values():
        directory.mkdir(parents=True)
    source_sha = hashlib.sha256(b"independent synthetic source identity").hexdigest()
    model_sha = hashlib.sha256(b"independent generic model identity").hexdigest()
    camera_sha = hashlib.sha256(b"independent estimated camera identity").hexdigest()
    manifest = {"track": 1, "episode": 16, "video_frames": 3, "video_fps": 30, "video_width": 32, "video_height": 24,
                "video_url": "https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/5f68335f3acc802033d1e80728c1633197521de8/track_1/videos/chunk-000/observation.images.exo_camera/episode_000016.mp4",
                "profile": "sam3d_body_masks_camera.v1", "video_sha256": source_sha, "mhr_model_sha256": model_sha,
                "camera_intrinsics": [[20, 0, 16], [0, 20, 12], [0, 0, 1]], "intrinsics_status": "independent synthetic estimate fixture",
                "input_artifact_sha256": {"intrinsics.json": camera_sha}}
    save(outputs["inference"] / "manifest.json", manifest)
    # Large intermediates are intentionally excluded even when they exist.
    (outputs["inference"] / "raw").mkdir()
    (outputs["inference"] / "raw/vertices_m.npy").write_bytes(b"synthetic omitted intermediate")
    native = {"pose": np.zeros((3, 136), dtype=np.float32), "shape": np.zeros(45, dtype=np.float32),
              "scales": np.zeros(68, dtype=np.float32), "frame_indices": np.arange(3, dtype=np.int64), "fps": np.asarray(30.)}
    hashes = {}
    for label in ("baseline", "refined"):
        report = {"video_sha256": source_sha, "mhr_model_sha256": model_sha,
                  "fixture_provenance": "independent synthetic, no challenge data"}
        path = outputs[label] / "episode_000016_human.npz"
        metadata = write_human_result(path, native, report)
        save(outputs[label] / "report.json", metadata)
        hashes[label] = {"parameters_sha256": package.digest(path)}
        (outputs[label] / "review_arrays").mkdir()
        (outputs[label] / "review_arrays/vertices_m.npy").write_bytes(b"synthetic omitted review mesh")
    observations = outputs["observations"]
    np.savez(observations / "observations.npz", frame_indices=np.arange(3), confidence=np.zeros((3, 2, 21)))
    save(observations / "report.json", {"fixture_provenance": "independent synthetic observer"})
    for stage, video in (("baseline_review", "comparison.mp4"), ("refined_review", "comparison.mp4"), ("hand_comparison", "hands-comparison.mp4")):
        directory = outputs[stage]
        (directory / "frames").mkdir()
        for frame in range(3):
            (directory / "frames" / f"{frame:06d}.jpg").write_bytes(b"synthetic image bytes")
        (directory / video).write_bytes(b"synthetic media transport bytes, not real model video")
        (directory / "index.html").write_text(f'<video src="{video}"></video><img src="frames/000000.jpg"><a href="summary.json">details</a>')
        if stage == "hand_comparison":
            summary = {"track": 1, "episode": 16, "video_sha256": source_sha, "frames": 3}
        else:
            (directory / "contact-sheet.jpg").write_bytes(b"independent synthetic rendered contact sheet")
            summary = {"dataset": "track_1", "episode": 16, "source_video_sha256": source_sha,
                       "decoded_frames": 3, "prediction_frames": 3, "missing_frames": [],
                       "prediction_frames_outside_video": [], "decode_count_matches_metadata": True,
                       "contact_sheet_frames": [0, 2]}
        summary["fixture_provenance"] = "independent synthetic transport fixture"
        save(directory / "summary.json", summary)
    checks = {"schema_version": "human_batch_episode_checks.v1", "track": 1, "episode": 16, "frames": 3,
              "structural_checks_passed": True, "source_video_sha256": source_sha, "mhr_model_sha256": model_sha,
              "native_results": hashes, "selected_candidate": "baseline", "selection_reason": "Independent synthetic fixture tests preserve the baseline",
              "reconstruction_quality_accepted": False}
    save(outputs["checks"] / "checks.json", checks)
    for stage, directory in outputs.items():
        files = {path.relative_to(directory).as_posix(): package.file_identity(path) for path in directory.rglob("*") if path.is_file()}
        save(directory.parent / "attempt-0001.started.json", {"stage": stage, "fixture": "independent synthetic"})
        log = directory.parent / "attempt-0001.log"
        log.write_text("Independent synthetic process log\n")
        save(directory.parent / "attempt-0001.finished.json", {"stage": stage, "returncode": 0, "status": "completed",
            "output": str(directory), "outputs": files, "log": str(log), "log_sha256": package.digest(log)})
    save(episode / "current-result.json", {"episode": 16, "status": "completed_candidate", "frames": 3,
        "selected_candidate": "baseline", "outputs": {key: str(value) for key, value in outputs.items()}})
    save(batch / "runtime-synthetic.json", {"recorded_at": "2026-10-21T00:00:00Z", "environments": {}, "asset_identities": {}})
    upstream = tmp_path / "upstream"
    save(upstream / "body-inputs-20261016-ep16.json", {"id": "body-inputs-20261016-ep16", "episode_id": "episode_000016",
         "stage": "inputs", "kind": "real_track1", "artifacts": [{"requirement_id": "camera", "sha256": camera_sha}]})
    return batch, outputs, upstream


def test_package_is_portable_and_selected_native_bytes_are_exact(tmp_path):
    batch, outputs, _ = fixture_batch(tmp_path)
    destination = tmp_path / "delivery"
    destination.mkdir()
    result = package.package_episode(batch, 16, destination, "synthetic-transport")
    archive = Path(result["archive"]["path"])
    assert result["zip_verification"]["all_member_sha256_verified"]
    relocated = tmp_path / "different-host-layout"
    with zipfile.ZipFile(archive) as source:
        source.extractall(relocated)
        assert not any("review_arrays/" in name or "/raw/" in name for name in source.namelist())
    package.verify_local_links(relocated)
    assert (relocated / "parameters/selectedhuman.npz").read_bytes() == (outputs["baseline"] / "episode_000016_human.npz").read_bytes()
    selection = package.read_json(relocated / "selection.json")
    assert selection["selected_is_exact_candidate_copy"] is True
    depth = package.read_json(relocated / "depth-scale.json")
    assert depth["status"] == "unverified_monocular_scale"
    assert depth["shared_human_object_scale_verified"] is False
    assert "scale_factor" not in depth
    assert (relocated / "attempts/inference/attempt-0001.log").is_file()
    previews = {item["role"]: item for item in result["previews"]}
    assert set(previews) == {"baseline_full_body", "refined_full_body", "hands_comparison", "baseline_contact_sheet", "refined_contact_sheet", "hand_qa_frame"}
    for role, stage, name in (("baseline_full_body", "baseline_review", "comparison.mp4"),
                              ("refined_full_body", "refined_review", "comparison.mp4"),
                              ("hands_comparison", "hand_comparison", "hands-comparison.mp4")):
        preview = previews[role]
        assert Path(preview["path"]).read_bytes() == (outputs[stage] / name).read_bytes()
        assert preview["content_type"] == "video/mp4"
        assert preview["coverage"] == {"kind": "full_sequence", "original_frame_count": 3}
        assert preview["source_video_sha256"] == result["source_video_sha256"]
        assert preview["mhr_model_sha256"] == result["mhr_model_sha256"]
    assert previews["baseline_contact_sheet"]["coverage"] == {"kind": "sampled", "original_frame_ids": [0, 2]}
    assert previews["hand_qa_frame"]["coverage"] == {"kind": "sampled", "original_frame_ids": [1]}
    assert previews["hand_qa_frame"]["content_type"] == "image/jpeg"


def test_changed_model_output_cannot_be_packaged_from_old_check_receipt(tmp_path):
    batch, outputs, _ = fixture_batch(tmp_path)
    path = outputs["baseline"] / "episode_000016_human.npz"
    with path.open("ab") as stream:
        stream.write(b"changed after checks")
    destination = tmp_path / "delivery"
    destination.mkdir()
    with pytest.raises(ValueError, match="changed since decode check"):
        package.package_episode(batch, 16, destination, "synthetic-tamper")


def test_broken_relative_review_link_blocks_portable_handoff(tmp_path):
    batch, outputs, _ = fixture_batch(tmp_path)
    page = outputs["baseline_review"] / "index.html"
    page.write_text('<a href="../../../../private-host-only/result.mp4">broken</a>')
    receipt_path = outputs["baseline_review"].parent / "attempt-0001.finished.json"
    receipt = package.read_json(receipt_path)
    receipt["outputs"]["index.html"] = package.file_identity(page)
    receipt_path.write_text(json.dumps(receipt))
    destination = tmp_path / "delivery"
    destination.mkdir()
    with pytest.raises(ValueError, match="Broken portable page link"):
        package.package_episode(batch, 16, destination, "synthetic-relocation")


def test_submitted_records_bind_actual_upstream_and_keep_quality_pending(tmp_path):
    batch, _, upstream = fixture_batch(tmp_path)
    destination = tmp_path / "delivery"
    destination.mkdir()
    result = package.package_episode(batch, 16, destination, "synthetic-board")
    # This URL is fixture metadata only; no request, upload or real declaration.
    envelope = package.published_records([result], "https://fixtures.example.test/assets/synthetic-board", "c" * 40,
                                         "synthetic-board", upstream, "Independent synthetic test executor")
    record = envelope["records"][0]
    assert record["state"] == "submitted"
    assert record["upstream"] == [{"stage": "inputs", "record_id": "body-inputs-20261016-ep16"}]
    assert record["checks"]["structural"]["status"] == "pass"
    assert record["checks"]["quality"]["status"] == "not_run"
    assert record["checks"]["export"]["status"] == "not_run"
    assert record["coverage"]["objects_complete"] is None
    depth = next(asset for asset in record["artifacts"] if asset["requirement_id"] == "depth_scale")
    assert depth["sha256"] == next(asset["sha256"] for asset in result["assets"] if asset["name"].endswith("-depth-scale.json"))


def test_local_packaging_emits_no_fake_remote_records_or_urls(tmp_path):
    batch, _, _ = fixture_batch(tmp_path)
    destination = tmp_path / "local-delivery"
    assert package.main(["--batch-root", str(batch), "--episodes", "16", "--output", str(destination),
                         "--delivery-id", "synthetic-local"]) == 0
    assert not (destination / "dashboard-submitted-records.json").exists()
    uploads = package.read_json(destination / "upload-manifest.json")
    assert uploads["upload_performed"] is False
    assert all(asset["intended_url"] is None for asset in uploads["assets"])
    assert len([item for item in uploads["assets"] if item["content_type"] == "video/mp4"]) == 3
    assert len([item for item in uploads["assets"] if item["content_type"] == "image/jpeg"]) == 3
    assert package.main(["--batch-root", str(batch), "--episodes", "16", "--output", str(destination),
                         "--delivery-id", "synthetic-local"]) == 1


def test_upstream_camera_mismatch_cannot_be_attached_to_board(tmp_path):
    batch, _, upstream = fixture_batch(tmp_path)
    destination = tmp_path / "delivery"
    destination.mkdir()
    result = package.package_episode(batch, 16, destination, "synthetic-wrong-camera")
    path = upstream / "body-inputs-20261016-ep16.json"
    data = package.read_json(path)
    data["artifacts"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not bind the consumed intrinsics"):
        package.published_records([result], "https://fixtures.example.test/assets/synthetic", "c" * 40,
                                  "synthetic-wrong-camera", upstream, "Independent synthetic test executor")


def test_preview_coverage_cannot_be_invented_from_other_episode_summary(tmp_path):
    batch, outputs, _ = fixture_batch(tmp_path)
    summary_path = outputs["baseline_review"] / "summary.json"
    summary = package.read_json(summary_path)
    summary["source_video_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary))
    receipt_path = outputs["baseline_review"].parent / "attempt-0001.finished.json"
    receipt = package.read_json(receipt_path)
    receipt["outputs"]["summary.json"] = package.file_identity(summary_path)
    receipt_path.write_text(json.dumps(receipt))
    destination = tmp_path / "delivery"
    destination.mkdir()
    with pytest.raises(ValueError, match="does not bind this Track 1 episode"):
        package.package_episode(batch, 16, destination, "synthetic-wrong-media-source")


def test_modified_video_bytes_fail_receipt_verification_before_public_preview(tmp_path):
    batch, outputs, _ = fixture_batch(tmp_path)
    (outputs["hand_comparison"] / "hands-comparison.mp4").write_bytes(b"changed video bytes")
    destination = tmp_path / "delivery"
    destination.mkdir()
    with pytest.raises(ValueError, match="Source result changed after stage completion"):
        package.package_episode(batch, 16, destination, "synthetic-modified-media")
