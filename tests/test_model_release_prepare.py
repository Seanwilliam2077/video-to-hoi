"""Synthetic redistribution packaging checks; no model execution or network."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_model_release as release


@pytest.fixture
def sample(tmp_path):
    project, root = tmp_path / "project", tmp_path / "assets"
    target = "weights/cari4d/cari4d"
    payload = b"synthetic checkpoint\x00" * 100
    checkpoint = root / target / "model.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(payload)
    asset = {"id": "cari4d", "kind": "huggingface", "repo_id": "example/model",
             "revision": "a" * 40, "target": target, "gated": "auto", "license": "example-license",
             "files": [{"path": "model.bin", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}
    receipt = {"schema_version": 1, "id": "cari4d", "status": "complete", "target": target,
               "asset_spec_sha256": release.assets.spec_hash(asset),
               "source": {k: asset[k] for k in ("kind", "repo_id", "revision", "license")},
               "files": [{"path": target + "/model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}
    directory = root / ".receipts"
    directory.mkdir()
    (directory / "cari4d.json").write_text(json.dumps(receipt))
    license_path = project / "deployment/licenses/EXAMPLE.txt"
    license_path.parent.mkdir(parents=True)
    license_path.write_bytes(b"Full synthetic license for the fixture.")
    review = {"id": "cari4d", "revision": asset["revision"], "asset_spec_sha256": release.assets.spec_hash(asset),
              "redistribution_allowed": True, "license_name": "Example license",
              "notices": [{"path": "deployment/licenses/EXAMPLE.txt", "sha256": release.assets.sha256(license_path)}]}
    return project, root, asset, {"schema_version": 1, "assets": [review]}, checkpoint


def test_licensed_archive_and_parts_reconstruct_exact_model(tmp_path, sample):
    project, root, asset, policy, checkpoint = sample
    (checkpoint.parent / "unselected-token.txt").write_text("private fixture excluded")
    result = release.build_asset(asset, policy, project=project, asset_root=root,
                                 output_dir=tmp_path / "out", part_bytes=1024)
    assert len(result["parts"]) > 1
    archive = tmp_path / "out" / result["archive"]["name"]
    combined = b"".join((archive.parent / p["name"]).read_bytes() for p in result["parts"])
    assert combined == archive.read_bytes()
    assert hashlib.sha256(combined).hexdigest() == result["archive"]["sha256"]
    for part in result["parts"]:
        data = (archive.parent / part["name"]).read_bytes()
        assert len(data) == part["bytes"] <= 1024
        assert hashlib.sha256(data).hexdigest() == part["sha256"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert z.read(asset["target"] + "/model.bin") == checkpoint.read_bytes()
        assert "deployment/licenses/EXAMPLE.txt" in z.namelist()
        assert ".receipts/cari4d.json" in z.namelist()
        assert not any("token" in name for name in z.namelist())


@pytest.mark.parametrize("field,value", [("redistribution_allowed", False), ("revision", "b" * 40),
                                        ("asset_spec_sha256", "0" * 64)])
def test_exact_review_is_required(tmp_path, sample, field, value):
    project, root, asset, policy, _ = sample
    policy["assets"][0][field] = value
    with pytest.raises(release.assets.AssetError, match="does not authorize"):
        release.build_asset(asset, policy, project=project, asset_root=root, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_license_or_changed_model_cannot_be_published(tmp_path, sample):
    project, root, asset, policy, checkpoint = sample
    (project / policy["assets"][0]["notices"][0]["path"]).unlink()
    with pytest.raises(release.assets.AssetError, match="notice is missing"):
        release.build_asset(asset, policy, project=project, asset_root=root, output_dir=tmp_path / "out")
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(release.assets.AssetError, match="missing or changed"):
        release.build_asset(asset, policy, project=project, asset_root=root, output_dir=tmp_path / "out")
