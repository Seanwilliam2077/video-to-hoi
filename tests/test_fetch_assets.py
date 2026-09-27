"""Download preparation tests: synthetic files only, no network or model execution."""
import importlib.util
import io
import json
import stat
from pathlib import Path
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location("fetch_assets", Path(__file__).parents[1] / "tools/fetch_assets.py")
fetcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetcher)


def http_asset():
    return {"id": "example", "kind": "http", "target": "weights/example", "filename": "model.bin",
            "url": "https://example.org/model.bin", "license": "synthetic test", "required": True}


@pytest.mark.parametrize("path", ["../escape", "/etc/passwd", "C:/escape", "weights/track_2/a", "weights\\a"])
def test_manifest_paths_cannot_escape_or_include_track2(tmp_path, path):
    with pytest.raises(fetcher.AssetError):
        fetcher.safe_path(tmp_path, path)


def test_source_zip_traversal_rejected_before_writes(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("repo/safe.py", "ok")
        z.writestr("repo/../../escape.txt", "bad")
    with pytest.raises(fetcher.AssetError):
        fetcher.extract_source(archive, tmp_path / "source")
    assert not (tmp_path / "source/safe.py").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_public_source_design_tokens_are_not_credentials(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("repo/design/tokens/base.css", ":root { --color: blue; }")
    fetcher.extract_source(archive, tmp_path / "source")
    assert (tmp_path / "source/design/tokens/base.css").is_file()


def test_source_internal_symlinks_materialize_for_zip_transfer(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("repo/meshes/a.obj", "synthetic mesh")
        for name, target in [("repo/urdfs/meshes", "../meshes"), ("repo/alias.obj", "meshes/a.obj")]:
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(info, target)
    fetcher.extract_source(archive, tmp_path / "source")
    assert (tmp_path / "source/urdfs/meshes/a.obj").read_text() == "synthetic mesh"
    assert (tmp_path / "source/alias.obj").read_text() == "synthetic mesh"
    assert not (tmp_path / "source/alias.obj").is_symlink()


@pytest.mark.parametrize("target", ["../../escape", "/etc/passwd", "C:/outside", "."])
def test_source_unsafe_symlinks_fail_before_extraction(tmp_path, target):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("repo/safe.txt", "synthetic")
        info = zipfile.ZipInfo("repo/link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, target)
    with pytest.raises(fetcher.AssetError):
        fetcher.extract_source(archive, tmp_path / "source")
    assert not (tmp_path / "source/safe.txt").exists()


def test_receipt_verifies_actual_bytes_and_detects_tamper(tmp_path, monkeypatch):
    asset = http_asset()
    monkeypatch.setattr(fetcher, "request", lambda *args: io.BytesIO(b"independent synthetic fixture"))
    receipt = fetcher.fetch(asset, tmp_path)
    assert receipt["status"] == "complete"
    assert receipt["files"][0]["path"] == "weights/example/model.bin"
    assert fetcher.verify_receipt(tmp_path, asset)["id"] == "example"
    (tmp_path / "weights/example/model.bin").write_bytes(b"tampered")
    with pytest.raises(fetcher.AssetError, match="missing or changed"):
        fetcher.verify_receipt(tmp_path, asset)


def test_changed_asset_revision_cannot_reuse_receipt(tmp_path, monkeypatch):
    asset = http_asset()
    monkeypatch.setattr(fetcher, "request", lambda *args: io.BytesIO(b"synthetic"))
    fetcher.fetch(asset, tmp_path)
    with pytest.raises(fetcher.AssetError, match="different asset specification"):
        fetcher.verify_receipt(tmp_path, {**asset, "url": "https://example.org/replacement.bin"})


def test_gated_skip_remains_incomplete(tmp_path):
    asset = {**http_asset(), "gated": True}
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps({"schema_version": 1, "assets": [asset]}))
    destination = tmp_path / "downloads"
    result = fetcher.main(["fetch", "--manifest", str(manifest), "--root", str(destination), "--skip-gated"])
    report = json.loads((destination / "asset-status.json").read_text())
    assert result == 2
    assert report["complete_selected_scope"] is False
    assert report["incomplete_required"] == ["example"]
    assert not (destination / "weights").exists()


def test_dataset_disguised_as_model_is_rejected(tmp_path):
    manifest = tmp_path / "assets.json"
    asset = {"id": "dataset", "kind": "huggingface", "target": "weights/test",
             "repo_id": "nvidia/video_to_data_challenge", "repo_type": "dataset", "revision": "a" * 40}
    manifest.write_text(json.dumps({"schema_version": 1, "assets": [asset]}))
    with pytest.raises(fetcher.AssetError, match="separate Track 1 downloader"):
        fetcher.load_manifest(manifest)


def test_library_error_does_not_leak_credentials():
    assert "hf_private_example" not in fetcher.public_error(RuntimeError("https://host/?token=hf_private_example"))


def test_organization_block_never_attempts_network(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("blocked provider must not be contacted")
    monkeypatch.setattr(fetcher, "request", unexpected)
    asset = {"id": "restricted", "kind": "gdrive_folder", "target": "weights/restricted",
             "url": "https://drive.google.com/drive/folders/example",
             "download_policy": "blocked_by_organization"}
    with pytest.raises(fetcher.AssetError, match="organization policy blocks"):
        fetcher.fetch(asset, tmp_path)
    with pytest.raises(fetcher.AssetError, match="no network check attempted"):
        fetcher.access_check(asset)


def test_drive_cannot_be_reintroduced_as_http(monkeypatch):
    monkeypatch.setattr(fetcher.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network forbidden"))
    with pytest.raises(fetcher.AssetError, match="organization policy blocks"):
        fetcher.request("https://drive.google.com/uc?id=example")


def test_redirect_to_blocked_provider_is_rejected_without_network():
    with pytest.raises(fetcher.AssetError, match="organization policy blocks"):
        fetcher.PolicyRedirectHandler().redirect_request(
            None, None, 302, "Found", {}, "https://drive.google.com/example")
