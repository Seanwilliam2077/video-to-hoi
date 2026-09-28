"""Synthetic transfer fixtures; no challenge, model or upstream example data."""
import hashlib
import io
import json
import zipfile

import pytest

from tools import restore_handoff as handoff


PAGE = "https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/assets-completion-20260928"


def fixture_manifest(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("synthetic.txt", "independently created transfer fixture")
    data = buffer.getvalue()
    parts = tmp_path / "parts"
    parts.mkdir()
    name = "fixture.zip.part001"
    (parts / name).write_bytes(data)
    record = {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
              "url": PAGE.replace("/tag/", "/download/") + "/" + name}
    doc = {"schema_version": 1, "release_url": PAGE, "assets": [{"id": "fixture",
           "archive": {"name": "fixture.zip", "bytes": len(data), "sha256": record["sha256"]},
           "parts": [record]}]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    return manifest, parts, data, doc


def test_offline_restore_and_repeat_are_verified(tmp_path):
    manifest, parts, data, _ = fixture_manifest(tmp_path)
    output = tmp_path / "restored"
    original = handoff.core.RELEASE_PAGE
    assert handoff.restore(manifest, parts, output)[0]["status"] == "created"
    assert (output / "fixture.zip").read_bytes() == data
    assert handoff.restore(manifest, parts, output)[0]["status"] == "existing"
    assert handoff.core.RELEASE_PAGE == original


@pytest.mark.parametrize("page", [PAGE + "/", PAGE.replace("Seanwilliam2077", "another-owner"),
                                PAGE.replace("github.com", "github.com.evil.example"), "https://drive.google.com/x"])
def test_unapproved_release_rejected_before_download(tmp_path, monkeypatch, page):
    manifest, parts, _, doc = fixture_manifest(tmp_path)
    doc["release_url"] = page
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(handoff.core, "open_release_url", lambda *a: pytest.fail("network attempted"))
    with pytest.raises(handoff.core.RestoreError, match="approved"):
        handoff.restore(manifest, parts, tmp_path / "out", download=True)


def test_cross_release_part_url_rejected(tmp_path):
    manifest, parts, _, doc = fixture_manifest(tmp_path)
    doc["assets"][0]["parts"][0]["url"] = doc["assets"][0]["parts"][0]["url"].replace(
        "assets-completion-20260928", "hf-models-20260927")
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(handoff.core.RestoreError, match="pinned"):
        handoff.restore(manifest, parts, tmp_path / "out")


def test_corrupt_part_does_not_create_archive(tmp_path):
    manifest, parts, data, _ = fixture_manifest(tmp_path)
    (parts / "fixture.zip.part001").write_bytes(b"x" + data[1:])
    output = tmp_path / "out"
    original = handoff.core.RELEASE_PAGE
    with pytest.raises(handoff.core.RestoreError):
        handoff.restore(manifest, parts, output)
    assert not (output / "fixture.zip").exists()
    assert handoff.core.RELEASE_PAGE == original
