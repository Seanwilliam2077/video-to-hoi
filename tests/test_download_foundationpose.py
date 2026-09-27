"""Offline manual-download tests; fake gdown never contacts Google Drive."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import download_foundationpose as manual


@pytest.fixture
def asset():
    manifest = manual.assets.load_manifest(manual.assets.DEFAULT_MANIFEST)
    return next(item for item in manifest["assets"] if item["id"] == "foundationpose-scorer")


def place_files(root, asset, *, html=False):
    target = root / asset["target"]
    target.mkdir(parents=True, exist_ok=True)
    (target / "model_best.pth").write_bytes(b"<!DOCTYPE html>denied" if html else b"synthetic checkpoint")
    (target / "config.yml").write_text("synthetic: true\n")
    return target


def test_manual_download_uses_fixed_paths_and_records_receipt(tmp_path, monkeypatch, asset):
    calls = []

    def list_folder(**kwargs):
        assert kwargs["url"] == asset["url"]
        assert kwargs["skip_download"] is True
        assert kwargs["use_cookies"] is False and kwargs["verify"] is True
        return [SimpleNamespace(id=name, path=name) for name in
                ("model_best.pth", "config.yml", "../../unrequested.py")]

    def download(**kwargs):
        calls.append(kwargs["id"])
        assert kwargs["resume"] is True
        assert kwargs["use_cookies"] is False and kwargs["verify"] is True
        path = Path(kwargs["output"])
        assert path.is_relative_to(tmp_path)
        path.write_bytes(b"synthetic model or config")
        return str(path)

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download_folder=list_folder, download=download))
    result = manual.prepare(asset, tmp_path)
    assert calls == ["config.yml", "model_best.pth"]
    assert result["acquisition"] == "user_manual"
    assert len(result["files"]) == 2
    assert manual.assets.verify_receipt(tmp_path, asset) == result
    # A second invocation validates existing files without importing gdown/network.
    monkeypatch.setitem(sys.modules, "gdown", None)
    assert manual.prepare(asset, tmp_path) == result


def test_browser_registration_is_offline_and_detects_tampering(tmp_path, monkeypatch, asset):
    monkeypatch.setitem(sys.modules, "gdown", None)
    target = place_files(tmp_path, asset)
    result = manual.prepare(asset, tmp_path, register_only=True)
    assert result["acquisition"] == "user_manual"
    (target / "model_best.pth").write_bytes(b"changed")
    with pytest.raises(manual.assets.AssetError, match="missing or changed"):
        manual.prepare(asset, tmp_path, register_only=True)


@pytest.mark.parametrize("failure", ["missing_config", "html", "empty"])
def test_invalid_files_never_get_complete_receipt(tmp_path, asset, failure):
    target = place_files(tmp_path, asset, html=failure == "html")
    if failure == "missing_config":
        (target / "config.yml").unlink()
    if failure == "empty":
        (target / "model_best.pth").write_bytes(b"")
    with pytest.raises(manual.assets.AssetError):
        manual.prepare(asset, tmp_path, register_only=True)
    assert not manual.assets.receipt_path(tmp_path, asset).exists()


def test_existing_manual_files_are_not_overwritten(tmp_path, monkeypatch, asset):
    monkeypatch.setitem(sys.modules, "gdown", None)
    target = place_files(tmp_path, asset)
    with pytest.raises(manual.assets.AssetError, match="register-only"):
        manual.prepare(asset, tmp_path)
    assert (target / "model_best.pth").read_bytes() == b"synthetic checkpoint"


def test_default_automation_still_cannot_fetch_or_probe_drive(tmp_path, monkeypatch, asset):
    monkeypatch.setattr(manual.assets, "request", lambda *a, **kw: pytest.fail("network attempted"))
    with pytest.raises(manual.assets.AssetError, match="automated download"):
        manual.assets.fetch(asset, tmp_path)
    with pytest.raises(manual.assets.AssetError, match="no network check attempted"):
        manual.assets.access_check(asset)


def test_cli_offline_registration_reports_incomplete_pair(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "gdown", None)
    assert manual.main(["--register-only", "--root", str(tmp_path)]) == 1
    assert not list(tmp_path.rglob("*.json"))
