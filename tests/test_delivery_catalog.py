"""Offline integrity checks for the published Track 1 delivery evidence.

No inference, video decoding, network requests, or Track 2 assets are used.
QA JPEG bytes are hashed only; these tests do not inspect image pixels.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("delivery_catalog", ROOT / "tools/build_delivery_catalog.py")
catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog)


@pytest.fixture
def repository(tmp_path):
    """A standalone tracked-files-only checkout, with no ignored data directory."""
    for relative in (catalog.DELIVERY, catalog.RAW_DELIVERY, catalog.PREVIEWS, Path("tracking")):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    return tmp_path


def change_json(path, mutate):
    value = json.loads(path.read_bytes())
    mutate(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_catalog_is_offline_and_preserves_input_only_boundaries(repository):
    result = catalog.build_catalog(repository)
    assert not (repository / "data").exists()
    assert len(result["entries"]) == 30
    assert sum(entry["frame_count"] for entry in result["entries"]) == 16563
    assert {entry["record"]["state"] for entry in result["entries"]} == {"submitted"}
    assert all(entry["raw_verified"] and entry["body_inputs_verified"]
               and entry["native_readers_passed"] and not entry["body_execution_completed"]
               for entry in result["entries"])
    assert all(entry["record"]["coverage"]["objects_complete"] is False
               for entry in result["entries"])
    assert all(entry["record"]["checks"]["structural"]["status"] == "not_run"
               for entry in result["entries"])
    assert "416" in result["entries"][10]["review"]["sampling_limitation"]
    assert "unmeasured" in result["entries"][10]["review"]["sampling_limitation"]


def test_check_accepts_committed_catalog_without_local_raw_data(repository):
    before = (repository / catalog.CATALOG).read_bytes()
    assert catalog.main(["--root", str(repository), "--check"]) == 0
    assert (repository / catalog.CATALOG).read_bytes() == before
    assert not (repository / "data").exists()


@pytest.mark.parametrize("filename", [
    "input.json", "review.json", "intrinsics.json", "visual-review-receipt.json",
    "camera-estimation.json", "masks-provenance.json",
])
def test_tampered_episode_evidence_never_yields_verified_catalog(repository, filename):
    path = repository / catalog.DELIVERY / "episodes/episode_000014" / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(catalog.CatalogError, match="SHA256 mismatch"):
        catalog.build_catalog(repository)


@pytest.mark.parametrize("filename", ["inventory.json", "checks.json", "native-reader-checks.json"])
def test_modified_root_receipt_is_rejected_before_claiming_ready(repository, filename):
    path = repository / catalog.DELIVERY / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(catalog.CatalogError, match="SHA256 mismatch"):
        catalog.build_catalog(repository)


def test_missing_registered_episode_cannot_be_replaced_by_other_record(repository):
    (repository / "tracking/records/body-inputs-20261016-ep12.json").unlink()
    with pytest.raises(catalog.CatalogError, match="Missing or unreadable"):
        catalog.build_catalog(repository)


def test_changed_record_cannot_inherit_original_body_verification(repository):
    path = repository / "tracking/records/body-inputs-20261016-ep03.json"
    change_json(path, lambda value: value.update(iteration="another-run"))
    with pytest.raises(catalog.CatalogError, match="Registered immutable"):
        catalog.build_catalog(repository)


def test_board_with_missing_or_incompatible_record_is_rejected(repository):
    path = repository / "tracking/board.json"
    change_json(path, lambda value: value["records"][0].update(iteration="new-run"))
    with pytest.raises(catalog.CatalogError, match="Board record missing, duplicated, or mismatched"):
        catalog.build_catalog(repository)


def test_unrelated_newer_record_does_not_borrow_old_delivery_proof(repository):
    baseline = catalog.build_catalog(repository)
    path = repository / "tracking/board.json"
    change_json(path, lambda value: value["records"].append({
        "id": "unrelated-new-run", "episode_id": "episode_000000",
        "stage": "inputs", "iteration": "next-iteration", "state": "running",
    }))
    assert catalog.build_catalog(repository) == baseline


@pytest.mark.parametrize("missing", [False, True])
def test_missing_or_tampered_preview_fails_hash_binding(repository, missing):
    path = repository / catalog.PREVIEWS / "episode_000019-targets.jpg"
    if missing:
        path.unlink()
    else:
        path.write_bytes(b"independent synthetic tamper fixture")
    with pytest.raises(catalog.CatalogError):
        catalog.build_catalog(repository)


def test_stale_catalog_fails_check_without_silently_rebuilding(repository):
    path = repository / catalog.CATALOG
    change_json(path, lambda value: value["entries"][0].update(body_execution_completed=True))
    before = path.read_bytes()
    assert catalog.main(["--root", str(repository), "--check"]) == 1
    assert path.read_bytes() == before


def test_public_links_are_pinned_and_archive_paths_are_episode_specific(repository):
    result = catalog.build_catalog(repository)
    for entry in result["entries"]:
        assert f"/resolve/{catalog.DATASET_REVISION}/track_1/videos/" in entry["source_video_url"]
        for evidence in entry["evidence"]:
            assert f"/{catalog.DELIVERY_REVISION}/deliveries/" in evidence["url"]
            assert len(evidence["sha256"]) == 64
        for file in entry["files"]:
            assert file.get("url") or file.get("archive_path")
            if file["kind"] != "video":
                assert file["archive_path"].startswith(f"body-inputs/{entry['episode_id']}/")
        assert len(entry["previews"]) == 2
        assert all(preview["url"].startswith(f"previews/{entry['episode_id']}-")
                   for preview in entry["previews"])


def test_raw_receipt_git_line_endings_do_not_change_public_evidence(repository):
    expected = catalog.build_catalog(repository)
    for filename in ("inventory.json", "checks.json"):
        path = repository / catalog.RAW_DELIVERY / filename
        data = path.read_bytes().replace(b"\r\n", b"\n")
        path.write_bytes(data)
    assert catalog.build_catalog(repository) == expected


def test_check_cannot_copy_or_mutate_preview_files(repository):
    assert catalog.main(["--root", str(repository), "--check",
                         "--copy-previews", str(repository / "absent")]) == 1
