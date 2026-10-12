"""Offline tests for restoring split model ZIPs; all payloads are tiny fixtures."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import zipfile
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "restore_model_release", Path(__file__).parents[1] / "tools/restore_model_release.py"
)
restorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restorer)


def _fixture(tmp_path: Path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("synthetic-model.bin", b"small fixture; no real model")
    payload = buffer.getvalue()
    chunks = [payload[:20], payload[20:45], payload[45:]]
    parts_dir = tmp_path / "parts"
    parts_dir.mkdir()
    parts = []
    for index, chunk in enumerate(chunks, 1):
        name = f"body.zip.part{index:03d}"
        (parts_dir / name).write_bytes(chunk)
        parts.append({
            "name": name, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest(),
            "url": restorer.RELEASE_PAGE.replace("/tag/", "/download/") + f"/{name}",
        })
    manifest = {
        "schema_version": 1,
        "release_url": restorer.RELEASE_PAGE,
        "assets": [{"id": "body", "archive": {
            "name": "body.zip", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
        }, "parts": parts}],
    }
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest, parts_dir, payload, chunks


def test_restore_uses_manifest_order_and_skips_matching_output(tmp_path: Path):
    manifest_path, manifest, parts_dir, payload, _ = _fixture(tmp_path)
    output_dir = tmp_path / "output"
    first = restorer.restore(manifest_path, parts_dir, output_dir)
    assert first[0]["status"] == "created"
    assert (output_dir / "body.zip").read_bytes() == payload
    with zipfile.ZipFile(output_dir / "body.zip") as archive:
        assert archive.namelist() == ["synthetic-model.bin"]
    assert restorer.restore(manifest_path, parts_dir, output_dir)[0]["status"] == "existing"

    manifest["assets"][0]["parts"].reverse()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(restorer.RestoreError, match="sequential"):
        restorer.restore(manifest_path, parts_dir, tmp_path / "wrong-order")


def test_existing_part_hash_and_conflicting_output_are_rejected(tmp_path: Path):
    manifest_path, _, parts_dir, _, chunks = _fixture(tmp_path)
    (parts_dir / "body.zip.part002").write_bytes(b"x" * len(chunks[1]))
    with pytest.raises(restorer.RestoreError, match="SHA-256 differs"):
        restorer.restore(manifest_path, parts_dir, tmp_path / "output")
    assert not (tmp_path / "output/body.zip").exists()

    (parts_dir / "body.zip.part002").write_bytes(chunks[1])
    output = tmp_path / "output/body.zip"
    output.write_bytes(b"conflicting archive")
    with pytest.raises(restorer.RestoreError, match="byte count differs"):
        restorer.restore(manifest_path, parts_dir, output.parent)
    assert output.read_bytes() == b"conflicting archive"


def test_archive_hash_failure_never_creates_final_output(tmp_path: Path):
    manifest_path, manifest, parts_dir, _, _ = _fixture(tmp_path)
    manifest["assets"][0]["archive"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_dir = tmp_path / "output"
    with pytest.raises(restorer.RestoreError, match="restored size or SHA-256"):
        restorer.restore(manifest_path, parts_dir, output_dir)
    assert list(output_dir.iterdir()) == []


def test_symlink_part_is_rejected(tmp_path: Path):
    manifest_path, _, parts_dir, _, _ = _fixture(tmp_path)
    part = parts_dir / "body.zip.part001"
    actual = parts_dir / "source.part"
    part.rename(actual)
    try:
        os.symlink(actual, part)
    except OSError:
        pytest.skip("creating symlinks is unavailable on this host")
    with pytest.raises(restorer.RestoreError, match="linked file"):
        restorer.restore(manifest_path, parts_dir, tmp_path / "output")


def test_symlink_parent_directory_is_rejected(tmp_path: Path):
    manifest_path, _, _, _, _ = _fixture(tmp_path)
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    try:
        os.symlink(actual, linked, target_is_directory=True)
    except OSError:
        pytest.skip("creating directory symlinks is unavailable on this host")
    with pytest.raises(restorer.RestoreError, match="contains a symlink"):
        restorer.restore(manifest_path, linked / "parts", tmp_path / "output", download=True)
    assert not (actual / "parts").exists()


def test_explicit_download_fetches_only_missing_part_with_mock_network(tmp_path: Path, monkeypatch):
    manifest_path, manifest, parts_dir, payload, chunks = _fixture(tmp_path)
    missing = parts_dir / "body.zip.part002"
    missing.unlink()
    with pytest.raises(restorer.RestoreError, match="missing part"):
        restorer.restore(manifest_path, parts_dir, tmp_path / "offline")

    opened = []

    def fake_open(url, part_name):
        opened.append((url, part_name))
        return io.BytesIO(chunks[1])

    monkeypatch.setattr(restorer, "open_release_url", fake_open)
    result = restorer.restore(manifest_path, parts_dir, tmp_path / "downloaded", download=True)
    assert result[0]["status"] == "created"
    assert (tmp_path / "downloaded/body.zip").read_bytes() == payload
    assert missing.read_bytes() == chunks[1]
    assert opened == [(manifest["assets"][0]["parts"][1]["url"], "body.zip.part002")]


def test_download_creates_missing_parts_directory_with_mock_network(tmp_path: Path, monkeypatch):
    manifest_path, manifest, _, payload, chunks = _fixture(tmp_path)
    parts_dir = tmp_path / "fresh" / "parts"
    with pytest.raises(restorer.RestoreError, match="parts directory is missing"):
        restorer.restore(manifest_path, parts_dir, tmp_path / "offline")
    assert not parts_dir.exists()

    payloads = {part["name"]: chunk for part, chunk in zip(manifest["assets"][0]["parts"], chunks)}
    opened = []

    def fake_open(url, part_name):
        opened.append(part_name)
        return io.BytesIO(payloads[part_name])

    monkeypatch.setattr(restorer, "open_release_url", fake_open)
    result = restorer.restore(manifest_path, parts_dir, tmp_path / "downloaded", download=True)
    assert result[0]["status"] == "created"
    assert (tmp_path / "downloaded/body.zip").read_bytes() == payload
    assert sorted(opened) == sorted(payloads)
    assert sorted(path.name for path in parts_dir.iterdir()) == sorted(payloads)


@pytest.mark.parametrize("field,value", [
    ("archive", "../escape.zip"),
    ("part", "../escape.part001"),
    ("url", "https://example.com/body.zip.part001"),
])
def test_unsafe_manifest_paths_and_urls_are_rejected(tmp_path: Path, field: str, value: str):
    manifest_path, manifest, parts_dir, _, _ = _fixture(tmp_path)
    asset = manifest["assets"][0]
    if field == "archive":
        asset["archive"]["name"] = value
    elif field == "part":
        asset["parts"][0]["name"] = value
    else:
        asset["parts"][0]["url"] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(restorer.RestoreError):
        restorer.restore(manifest_path, parts_dir, tmp_path / "output", download=True)


def test_duplicate_names_and_unapproved_redirects_are_rejected(tmp_path: Path):
    manifest_path, manifest, _, _, _ = _fixture(tmp_path)
    duplicate = json.loads(json.dumps(manifest["assets"][0]))
    duplicate["id"] = "other"
    manifest["assets"].append(duplicate)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(restorer.RestoreError, match="duplicate"):
        restorer.read_manifest(manifest_path)
    with pytest.raises(restorer.RestoreError, match="approved GitHub asset origins"):
        restorer._validate_url("https://example.com/file", "body.zip.part001", redirect=True)
    assert restorer._validate_url(
        "https://release-assets.githubusercontent.com/github-production-release-asset/example",
        "body.zip.part001", redirect=True,
    )
