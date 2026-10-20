"""Asset restoration guards using independent, tiny synthetic files only."""
import hashlib
import importlib.util
import io
from pathlib import Path
import zipfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "prepare_human_assets", Path(__file__).resolve().parents[1] / "tools/human_runtime/prepare_assets.py",
)
ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSETS)


def record(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "C:/outside", "safe\\..\\escape"])
def test_asset_destinations_cannot_escape_root(tmp_path, relative):
    with pytest.raises(ValueError, match="unsafe"):
        ASSETS._under(tmp_path, relative)


def test_hash_failure_publishes_no_file_and_preserves_other_files(tmp_path):
    existing = tmp_path / "existing"
    existing.write_bytes(b"keep")
    with pytest.raises(ValueError, match="SHA-256"):
        ASSETS._write_stream(io.BytesIO(b"wrong"), tmp_path / "model", record(b"right"))
    assert not (tmp_path / "model").exists()
    assert existing.read_bytes() == b"keep"
    assert list(tmp_path.glob("*.partial")) == []


def test_selected_zip_member_only_and_no_changed_file_overwrite(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("allowed/model", b"model")
        archive.writestr("../escape", b"must not extract")
    stream.seek(0)
    destination = tmp_path / "model"
    with zipfile.ZipFile(stream) as archive:
        ASSETS._extract_member(archive, "allowed/model", destination, record(b"model"))
        ASSETS._extract_member(archive, "allowed/model", destination, record(b"model"))
        assert destination.read_bytes() == b"model"
        destination.write_bytes(b"owner edits")
        with pytest.raises(ValueError, match="byte count"):
            ASSETS._extract_member(archive, "allowed/model", destination, record(b"model"))
    assert destination.read_bytes() == b"owner edits"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["model"]


def test_unapproved_download_is_rejected_before_network(tmp_path):
    with pytest.raises(ValueError, match="explicitly pinned"):
        ASSETS._ensure_download(tmp_path / "missing", record(b"fixture"),
                                "https://huggingface.co/unapproved", True, "track1")
    assert not (tmp_path / "missing").exists()


def test_provenance_is_idempotent_but_different_existing_content_is_preserved(tmp_path):
    path = tmp_path / "provenance.json"
    ASSETS._write_json_once(path, {"fixture": "original"})
    original_bytes = path.read_bytes()
    ASSETS._write_json_once(path, {"fixture": "original"})
    with pytest.raises(ValueError, match="refusing to overwrite"):
        ASSETS._write_json_once(path, {"fixture": "different"})
    assert path.read_bytes() == original_bytes
