"""CPU-only synthetic checks for the portable handoff bundle."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_bundle import BundleError, build_bundle  # noqa: E402
from fetch_assets import verify_receipt  # noqa: E402


def _put(root: Path, relative: str, data: bytes = b"example") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def test_github_zip_checkout_code_only_and_sha256(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    _put(root, "README.md", b"# project\n")
    _put(root, "src/v2hoi/stages/inputs.py", b"pass\n")
    _put(root, "tools/prepare_bundle.py", b"pass\n")
    _put(root, "deployment/assets.lock.json", b'{"schema_version":1,"assets":[]}')
    _put(root, "docs/track1-videos/media/ep00.mp4", b"video")
    _put(root, "data/v2d/track_2/secret.parquet", b"forbidden")
    _put(root, "weights/model.pt", b"weights")
    _put(root, "deployment/.env", b"TOKEN=private")
    _put(root, "deployment/.receipts/asset.json", b'{"status":"complete"}')
    (root / "third_party" / "empty-submodule").mkdir(parents=True)
    output = tmp_path / "code.zip"

    manifest = build_bundle(root, output)

    assert manifest["selection"] == "allowlist"
    assert manifest["gated_models_included"] is False
    assert manifest["track2_included"] is False
    assert _names(output) == {
        "README.md", "src/v2hoi/stages/inputs.py", "tools/prepare_bundle.py",
        "deployment/assets.lock.json", "bundle-manifest.json",
    }
    with zipfile.ZipFile(output) as archive:
        archived = json.loads(archive.read("bundle-manifest.json"))
        for item in archived["files"]:
            assert hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"]
            assert len(archive.read(item["path"])) == item["bytes"]
    companion = output.with_suffix(".zip.sha256")
    assert companion.read_text(encoding="ascii") == (
        f"{hashlib.sha256(output.read_bytes()).hexdigest()}  code.zip\n"
    )
    with pytest.raises(BundleError, match="already exists"):
        build_bundle(root, output)


def test_git_index_omits_untracked_source_but_includes_handoff_tools(tmp_path: Path):
    root = tmp_path / "clone"
    root.mkdir()
    if subprocess.run(["git", "--version"], capture_output=True).returncode:
        pytest.skip("git unavailable")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    _put(root, "README.md", b"hello")
    _put(root, "src/v2hoi/tracked.py", b"tracked")
    _put(root, "docs/track1-videos/media/ep00.mp4", b"tracked video")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Bundle Test",
         "-c", "user.email=bundle@example.invalid", "commit", "-qm", "fixture"],
        check=True,
    )
    _put(root, "src/v2hoi/private_untracked.py", b"not in git")
    _put(root, "deployment/install.sh", b"#!/bin/sh\n")
    _put(root, "tools/fetch_assets.py", b"pass\n")
    _put(root, "tools/prepare_wheelhouse.py", b"pass\n")
    output = tmp_path / "clone.zip"

    manifest = build_bundle(root, output)

    assert manifest["selection"] == "git-index+handoff-allowlist"
    names = _names(output)
    assert "src/v2hoi/tracked.py" in names
    assert "src/v2hoi/private_untracked.py" not in names
    assert "docs/track1-videos/media/ep00.mp4" not in names
    assert "deployment/install.sh" in names
    assert "tools/fetch_assets.py" in names
    assert "tools/prepare_wheelhouse.py" in names


def _receipt_fixture(tmp_path: Path, *, gated: bool = False):
    root = tmp_path / "source"
    root.mkdir()
    _put(root, "README.md", b"code")
    asset_root = tmp_path / "downloaded"
    content = b"upstream source code"
    sha = hashlib.sha256(content).hexdigest()
    path = _put(asset_root, "third_party/MoGe/data/v2.py", content)
    lock = {
        "schema_version": 1,
        "status": "draft-pending-final-audit",
        "assets": [{
            "id": "moge", "kind": "git", "target": "third_party/MoGe",
            "url": "https://github.com/Ruicheng/MoGe", "revision": "a" * 40,
            "license": "Apache-2.0", "gated": gated,
            "files": [{"path": "data/v2.py", "size": len(content), "sha256": sha}],
        }],
    }
    _put(root, "deployment/assets.lock.json", json.dumps(lock).encode())
    receipt = {
        "schema_version": 1, "id": "moge", "status": "complete",
        "source": {
            "kind": "git", "url": "https://github.com/Ruicheng/MoGe", "revision": "a" * 40,
            "license": "Apache-2.0",
        },
        "target": "third_party/MoGe",
        "files": [{"path": "third_party/MoGe/data/v2.py", "bytes": len(content), "sha256": sha}],
    }
    receipt["asset_spec_sha256"] = hashlib.sha256(
        json.dumps(lock["assets"][0], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipts = asset_root / ".receipts"
    _put(asset_root, ".receipts/moge.json", json.dumps(receipt).encode())
    return root, asset_root, receipts, path


def test_receipt_adds_only_verified_nongated_asset(tmp_path: Path):
    root, asset_root, receipts, path = _receipt_fixture(tmp_path)
    original = json.loads((receipts / "moge.json").read_text())
    original["private_note"] = "HF_TOKEN=must-not-be-packed"
    (receipts / "moge.json").write_text(json.dumps(original))
    output = tmp_path / "with-assets.zip"
    manifest = build_bundle(root, output, receipts=receipts, asset_root=asset_root)
    assert "third_party/MoGe/data/v2.py" in _names(output)
    assert ".receipts/moge.json" in _names(output)
    assert next(
        item for item in manifest["files"] if item["path"] == "third_party/MoGe/data/v2.py"
    )["asset_id"] == "moge"
    with zipfile.ZipFile(output) as archive:
        assert b"HF_TOKEN" not in archive.read(".receipts/moge.json")
        archive.extractall(tmp_path / "received")
    lock = json.loads((root / "deployment/assets.lock.json").read_text())
    assert verify_receipt(tmp_path / "received", lock["assets"][0])["id"] == "moge"

    path.write_bytes(b"tampered")
    with pytest.raises(BundleError, match="does not match receipt"):
        build_bundle(root, tmp_path / "tampered.zip", receipts=receipts, asset_root=asset_root)


def test_gated_receipt_never_adds_model(tmp_path: Path):
    root, asset_root, receipts, _ = _receipt_fixture(tmp_path, gated=True)
    output = tmp_path / "nongated.zip"
    manifest = build_bundle(root, output, receipts=receipts, asset_root=asset_root)
    assert manifest["gated_models_included"] is False
    assert all(not path.startswith("third_party/") for path in _names(output))


def test_receipt_source_must_match_asset_lock(tmp_path: Path):
    root, asset_root, receipts, _ = _receipt_fixture(tmp_path)
    receipt_path = receipts / "moge.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["source"]["revision"] = "b" * 40
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(BundleError, match="differs from asset lock"):
        build_bundle(root, tmp_path / "wrong-source.zip", receipts=receipts, asset_root=asset_root)


def test_partial_receipt_cannot_claim_complete_locked_asset(tmp_path: Path):
    root, asset_root, receipts, _ = _receipt_fixture(tmp_path)
    lock_path = root / "deployment/assets.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["assets"][0]["files"].append({"path": "data/second.py", "size": 5})
    lock_path.write_text(json.dumps(lock))
    receipt_path = receipts / "moge.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["asset_spec_sha256"] = hashlib.sha256(
        json.dumps(lock["assets"][0], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(BundleError, match="locked asset file is absent"):
        build_bundle(root, tmp_path / "partial.zip", receipts=receipts, asset_root=asset_root)


def test_only_pinned_upstream_public_names_bypass_secret_name_filter(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    _put(root, "README.md", b"code")
    asset_root = tmp_path / "assets"
    upstream = "third_party/video_to_data"
    public_names = [
        "docs/chord/_ds/basic-nvidia-design-system-f939d4e7-d0bb-4994-9e26-a471f04510c4/tokens/base.css",
        "reconstruction/workflows/mv_hoi/alembic/env.py",
    ]
    files = []
    for name in public_names:
        content = name.encode()
        _put(asset_root, f"{upstream}/{name}", content)
        files.append({"path": f"{upstream}/{name}", "bytes": len(content),
                      "sha256": hashlib.sha256(content).hexdigest()})
    asset = {
        "id": "video-to-data-source", "kind": "git", "target": upstream,
        "url": "https://github.com/nvidia-isaac/video_to_data",
        "revision": "33129dd0f2d2dcfd1164d43fd076542660756ed2",
        "license": "Apache-2.0", "files": [],
    }
    _put(root, "deployment/assets.lock.json", json.dumps({"schema_version": 1, "assets": [asset]}).encode())
    receipt = {
        "schema_version": 1, "id": asset["id"], "status": "complete",
        "source": {key: asset[key] for key in ("kind", "url", "revision", "license")},
        "target": upstream, "files": files,
        "asset_spec_sha256": hashlib.sha256(
            json.dumps(asset, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    _put(asset_root, ".receipts/video-to-data-source.json", json.dumps(receipt).encode())
    output = tmp_path / "public.zip"
    build_bundle(root, output, receipts=asset_root / ".receipts", asset_root=asset_root)
    assert all(f"{upstream}/{name}" in _names(output) for name in public_names)

    extra = "docs/unapproved/tokens/secret.css"
    content = b"looks public, but is not at the locked commit path"
    _put(asset_root, f"{upstream}/{extra}", content)
    receipt["files"].append({"path": f"{upstream}/{extra}", "bytes": len(content),
                             "sha256": hashlib.sha256(content).hexdigest()})
    (asset_root / ".receipts/video-to-data-source.json").write_text(json.dumps(receipt))
    with pytest.raises(BundleError, match="forbidden asset path"):
        build_bundle(root, tmp_path / "unapproved.zip", receipts=asset_root / ".receipts", asset_root=asset_root)


@pytest.mark.parametrize("asset_id,target,url,revision,public_name", [
    ("cari4d-dinov3-source", "weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov3_main",
     "https://github.com/facebookresearch/dinov3", "6876159a11b4df116f30f667f8c9888617df0751",
     "dinov3/env/__init__.py"),
    ("pybind11-source", "third_party/pybind11", "https://github.com/pybind/pybind11",
     "aa304c9c7d725ffb9d10af08a3b34cb372307020", "tests/env.py"),
    ("sam2-source", "third_party/sam2", "https://github.com/facebookresearch/sam2",
     "2b90b9f5ceec907a1c18123530e92e794ad901a4",
     "demo/frontend/src/theme/tokens.stylex.ts"),
])
def test_pinned_public_source_files_with_env_or_tokens_names(
    tmp_path: Path, asset_id: str, target: str, url: str, revision: str, public_name: str,
):
    root = tmp_path / "source"
    root.mkdir()
    _put(root, "README.md", b"code")
    asset_root = tmp_path / "assets"
    content = b"public upstream source"
    _put(asset_root, f"{target}/{public_name}", content)
    asset = {"id": asset_id, "kind": "git", "target": target, "url": url,
             "revision": revision, "files": []}
    _put(root, "deployment/assets.lock.json", json.dumps({"schema_version": 1, "assets": [asset]}).encode())
    receipt = {
        "schema_version": 1, "id": asset_id, "status": "complete",
        "source": {"kind": "git", "url": url, "revision": revision},
        "target": target,
        "files": [{"path": f"{target}/{public_name}", "bytes": len(content),
                   "sha256": hashlib.sha256(content).hexdigest()}],
        "asset_spec_sha256": hashlib.sha256(
            json.dumps(asset, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    _put(asset_root, f".receipts/{asset_id}.json", json.dumps(receipt).encode())
    output = tmp_path / "public.zip"
    build_bundle(root, output, receipts=asset_root / ".receipts", asset_root=asset_root)
    assert f"{target}/{public_name}" in _names(output)


def test_organization_blocked_drive_asset_is_rejected(tmp_path: Path):
    root, asset_root, receipts, _ = _receipt_fixture(tmp_path)
    lock_path = root / "deployment" / "assets.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["assets"][0]["kind"] = "gdrive_folder"
    lock["assets"][0]["download_policy"] = "blocked_by_organization"
    lock_path.write_text(json.dumps(lock))
    with pytest.raises(BundleError, match="organization-blocked"):
        build_bundle(root, tmp_path / "blocked.zip", receipts=receipts, asset_root=asset_root)


def _manual_drive_fixture(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    _put(root, "README.md", b"code")
    asset_root = tmp_path / "manual-assets"
    target = "weights/foundationpose/synthetic-folder"
    contents = {"model_best.pth": b"synthetic model bytes", "config.yml": b"synthetic config\n"}
    files = []
    locked_files = []
    for name, content in contents.items():
        relative = f"{target}/{name}"
        _put(asset_root, relative, content)
        digest = hashlib.sha256(content).hexdigest()
        files.append({"path": relative, "bytes": len(content), "sha256": digest})
        locked_files.append({"path": name, "size": len(content), "sha256": digest})
    asset = {
        "id": "manual-foundationpose", "kind": "gdrive_folder", "target": target,
        "url": "https://example.invalid/synthetic-folder",
        "download_policy": "manual_user_only", "license": "synthetic test license",
        "gated": False, "files": locked_files,
    }
    _put(root, "deployment/assets.lock.json", json.dumps({"schema_version": 1, "assets": [asset]}).encode())
    receipt = {
        "schema_version": 1, "id": asset["id"], "status": "complete",
        "acquisition": "user_manual", "target": target,
        "source": {"kind": "gdrive_folder", "url": asset["url"], "license": asset["license"]},
        "asset_spec_sha256": hashlib.sha256(
            json.dumps(asset, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "files": files,
        "private_note": "TOKEN=never-include-in-portable-receipt",
    }
    receipt_path = _put(asset_root, f".receipts/{asset['id']}.json", json.dumps(receipt).encode())
    return root, asset_root, receipt_path, asset


def test_manual_drive_receipt_is_sanitized_and_rebundles(tmp_path: Path):
    root, asset_root, receipt_path, asset = _manual_drive_fixture(tmp_path)
    output = tmp_path / "manual.zip"
    build_bundle(root, output, receipts=receipt_path.parent, asset_root=asset_root)
    with zipfile.ZipFile(output) as archive:
        assert f"{asset['target']}/model_best.pth" in archive.namelist()
        assert f"{asset['target']}/config.yml" in archive.namelist()
        portable = json.loads(archive.read(f".receipts/{asset['id']}.json"))
        assert portable["acquisition"] == "user_manual"
        assert "private_note" not in portable
        archive.extractall(tmp_path / "received")

    received = tmp_path / "received"
    assert verify_receipt(received, asset)["acquisition"] == "user_manual"
    rebundled = tmp_path / "rebundled.zip"
    build_bundle(received, rebundled, receipts=received / ".receipts", asset_root=received)
    with zipfile.ZipFile(rebundled) as archive:
        assert json.loads(archive.read(f".receipts/{asset['id']}.json"))["acquisition"] == "user_manual"
        assert f"{asset['target']}/model_best.pth" in archive.namelist()


@pytest.mark.parametrize("acquisition", [None, "automated"])
def test_manual_drive_receipt_requires_user_manual_marker(tmp_path: Path, acquisition: str | None):
    root, asset_root, receipt_path, _ = _manual_drive_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    if acquisition is None:
        receipt.pop("acquisition")
    else:
        receipt["acquisition"] = acquisition
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(BundleError, match="manual_user_only policy and user_manual acquisition"):
        build_bundle(root, tmp_path / "rejected.zip", receipts=receipt_path.parent, asset_root=asset_root)


def test_symlink_rejected(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    _put(root, "README.md")
    outside = _put(tmp_path, "outside.py")
    link = root / "src" / "leak.py"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(BundleError, match="symlink"):
        build_bundle(root, tmp_path / "bad.zip")
