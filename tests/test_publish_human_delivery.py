"""Independent synthetic transport fixtures; no network or challenge assets."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publish_human_delivery", ROOT / "tools/publish_human_delivery.py")
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)
COMMIT = "a" * 40
DELIVERY = "human-independent-synthetic-v1"


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def synthetic_manifest(directory, episode=16):
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"{DELIVERY}-ep{episode:02d}.zip"
    members = {"file-manifest.json": json.dumps({"schema_version": "human_delivery_files.v1",
                   "delivery_id": DELIVERY, "episode": episode, "files": {}}).encode(),
               "README.md": b"Independent synthetic transport fixture; no model prediction or challenge input."}
    members["SHA256SUMS"] = "".join(f"{hashlib.sha256(value).hexdigest()}  {name}\n" for name, value in members.items()).encode()
    with zipfile.ZipFile(archive, "x") as stream:
        for name, value in members.items():
            stream.writestr(name, value)
    side = directory / f"{DELIVERY}-ep{episode:02d}-checks.json"
    write(side, {"fixture": "independent synthetic check metadata"})
    assets = [{"path": str(path), "name": path.name, "bytes": path.stat().st_size,
               "sha256": publisher.digest(path), "content_type": publisher.MIMES[path.suffix],
               "intended_url": None, "upload_status": "not_uploaded"} for path in (archive, side)]
    manifest = directory / "upload-manifest.json"
    write(manifest, {"schema_version": "human_delivery_upload.v1", "delivery_id": DELIVERY,
                    "source_commit": COMMIT, "assets": assets, "upload_performed": False})
    return manifest


class FakeGitHub:
    def __init__(self, plan):
        self.plan = plan
        self.release = None
        self.assets = {}
        self.public_bytes = {}
        self.calls = []
        self.uploads = []
        self.lost_release_response = False
        self.lost_upload_response = False
        self.digest_missing = False
        self.tag_commit = COMMIT

    def create_release(self, payload):
        self.release = {"id": 17, "tag_name": self.plan["tag"], "draft": False, "prerelease": True,
                        "body": payload["body"], "target_commitish": payload["target_commitish"]}

    def json(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if method == "GET" and path.startswith("/git/commits/"):
            return {"sha": COMMIT}
        if method == "GET" and path.startswith("/git/ref/tags/"):
            if self.release is None:
                raise publisher.APIError(404)
            return {"object": {"type": "commit", "sha": self.tag_commit}}
        if method == "GET" and path.startswith("/releases/tags/"):
            if self.release is None:
                raise publisher.APIError(404)
            return dict(self.release)
        if method == "POST" and path == "/releases":
            self.create_release(payload)
            if self.lost_release_response:
                self.lost_release_response = False
                raise publisher.APIError(transient=True)
            return dict(self.release)
        if method == "GET" and path.startswith("/releases/17/assets?"):
            assert "page=1" in path
            return list(self.assets.values())
        if method == "PATCH" and path == "/releases/17":
            self.release.update(payload)
            return dict(self.release)
        raise AssertionError(f"Unexpected mocked request: {method} {path}")

    def upload(self, release_id, asset):
        assert release_id == 17
        assert asset["name"] not in self.assets, "A duplicate upload was attempted"
        data = Path(asset["path"]).read_bytes()
        self.uploads.append(asset["name"])
        remote = {"id": 100 + len(self.assets), "name": asset["name"], "state": "uploaded",
                  "size": len(data), "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
                  "browser_download_url": asset["url"]}
        if self.digest_missing:
            remote.pop("digest")
        self.assets[asset["name"]] = remote
        self.public_bytes[asset["url"]] = data
        if self.lost_upload_response:
            self.lost_upload_response = False
            raise publisher.APIError(transient=True)
        return dict(remote)

    def public_identity(self, url, expected_bytes):
        data = self.public_bytes[url]
        return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def setup(tmp_path):
    manifest = synthetic_manifest(tmp_path / "package")
    plan = publisher.load_plan(manifest, COMMIT)
    client = FakeGitHub(plan)
    instance = publisher.Publisher(client, plan, tmp_path / "receipts", sleep=lambda _: None)
    return manifest, plan, client, instance


def test_default_plan_performs_no_network_or_credential_lookup(tmp_path, monkeypatch, capsys):
    manifest = synthetic_manifest(tmp_path / "package")
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline plan attempted authentication or network")
    monkeypatch.setattr(publisher, "credential_token", forbidden)
    monkeypatch.setattr(publisher, "GitHub", forbidden)
    assert publisher.main(["--manifest", str(manifest), "--source-commit", COMMIT]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["publish_performed"] is False and result["network_used"] is False
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("field,value", [("bytes", 1), ("sha256", "b" * 64), ("content_type", "image/jpeg"),
                                        ("name", "weights.ckpt"), ("intended_url", "https://example.com/placeholder")])
def test_manifest_rejects_changed_bytes_wrong_mime_and_unknown_assets(tmp_path, field, value):
    manifest = synthetic_manifest(tmp_path / "package")
    obj = publisher.read_json(manifest)
    obj["assets"][0][field] = value
    write(manifest, obj)
    with pytest.raises(ValueError):
        publisher.load_plan(manifest, COMMIT)


def test_manifest_source_commit_and_tag_are_exact(tmp_path):
    manifest = synthetic_manifest(tmp_path / "package")
    with pytest.raises(ValueError, match="commit"):
        publisher.load_plan(manifest, "b" * 40)
    with pytest.raises(ValueError, match="tag"):
        publisher.load_plan(manifest, COMMIT, "main")


def test_windows_registry_zip_mime_alias_is_normalized_for_upload(tmp_path):
    manifest = synthetic_manifest(tmp_path / "package")
    obj = publisher.read_json(manifest)
    obj["assets"][0]["content_type"] = "application/x-zip-compressed"
    write(manifest, obj)
    assert publisher.load_plan(manifest, COMMIT)["assets"][0]["content_type"] == "application/zip"


@pytest.mark.parametrize("delivery_id", [DELIVERY, "sam-mediapipe-20261021"])
def test_actual_packager_output_is_consumable_without_rewriting_its_manifest(tmp_path, delivery_id):
    specification = importlib.util.spec_from_file_location("independent_packager_test_helper", ROOT / "tests/test_package_human_delivery.py")
    helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helper)
    batch, _, _ = helper.fixture_batch(tmp_path)
    output = tmp_path / "portable"
    assert helper.package.main(["--batch-root", str(batch), "--episodes", "16", "--output", str(output),
                               "--delivery-id", delivery_id, "--source-commit", COMMIT]) == 0
    plan = publisher.load_plan(output / "upload-manifest.json", COMMIT)
    assert plan["tag"] == plan["delivery_id"] == delivery_id
    assert len(plan["assets"]) >= 10
    assert {a["content_type"] for a in plan["assets"]} == {"application/zip", "application/json", "image/jpeg", "video/mp4"}


@pytest.mark.parametrize("member", ["weights/model.pt", "data/input.mp4", "track_2/derived.json", "../escape.json"])
def test_review_archive_rejects_input_model_or_unsafe_members(tmp_path, member):
    manifest = synthetic_manifest(tmp_path / "package")
    obj = publisher.read_json(manifest)
    path = Path(obj["assets"][0]["path"])
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(member, b"Independent synthetic forbidden-member fixture")
    obj["assets"][0].update(bytes=path.stat().st_size, sha256=publisher.digest(path))
    write(manifest, obj)
    with pytest.raises(ValueError, match="ZIP"):
        publisher.load_plan(manifest, COMMIT)


def test_unknown_release_and_upload_outcomes_recover_without_duplicate_posts(tmp_path):
    _, plan, client, instance = setup(tmp_path)
    client.lost_release_response = client.lost_upload_response = True
    result = instance.publish()
    assert result["passed"] is True
    assert len([c for c in client.calls if c[0] == "POST"]) == 1
    assert len(client.uploads) == len(set(client.uploads)) == 2 * len(plan["assets"])
    assert all(asset["verified"] and asset["anonymous_download_verified"] for asset in result["assets"])
    assert publisher.read_json(tmp_path / "receipts/publication.json")["passed"] is True
    for asset in plan["assets"]:
        data = client.public_bytes[asset["url"] + ".sha256"]
        assert data == f"{asset['sha256']}  {asset['name']}\n".encode()


def test_resuming_reverifies_remote_bytes_without_reupload_or_body_edits(tmp_path):
    _, plan, client, instance = setup(tmp_path)
    instance.publish(body="Initial release notes")
    uploads = list(client.uploads)
    result = instance.publish(body="Different notes are not implicitly authorized")
    assert result["passed"] is True and client.uploads == uploads
    assert not any(c[0] == "PATCH" for c in client.calls)
    assert client.release["body"].startswith("Initial release notes")


def test_missing_server_digest_requires_successful_fresh_public_download(tmp_path):
    _, _, client, instance = setup(tmp_path)
    client.digest_missing = True
    result = instance.publish()
    assert all(not asset["server_digest_verified"] and asset["anonymous_download_verified"] for asset in result["assets"])


def test_existing_digest_mismatch_fails_without_overwriting_asset(tmp_path):
    _, plan, client, instance = setup(tmp_path)
    client.create_release({"body": "untouched", "target_commitish": COMMIT})
    client.upload(17, plan["assets"][0])
    client.assets[plan["assets"][0]["name"]]["digest"] = "sha256:" + "f" * 64
    previous = list(client.uploads)
    with pytest.raises(ValueError, match="digest mismatch"):
        instance.publish()
    assert client.uploads == previous
    receipt = publisher.read_json(tmp_path / "receipts/publication.json")
    assert receipt["passed"] is False and receipt["assets"] == []


def test_incomplete_remote_upload_is_not_deleted_or_replaced(tmp_path):
    _, plan, client, instance = setup(tmp_path)
    client.create_release({"body": "untouched", "target_commitish": COMMIT})
    client.upload(17, plan["assets"][0])
    client.assets[plan["assets"][0]["name"]].update(state="starter", size=0)
    previous = list(client.uploads)
    with pytest.raises(ValueError, match="incomplete state"):
        instance.publish()
    assert client.uploads == previous
    assert not any(call[0] == "DELETE" for call in client.calls)


def test_wrong_tag_commit_fails_before_upload_or_release_modification(tmp_path):
    _, _, client, instance = setup(tmp_path)
    client.create_release({"body": "untouched", "target_commitish": COMMIT})
    client.tag_commit = "b" * 40
    with pytest.raises(ValueError, match="another commit"):
        instance.publish()
    assert client.uploads == [] and not any(c[0] != "GET" for c in client.calls)


def test_github_immutable_release_fails_clearly_before_asset_posts(tmp_path):
    _, _, client, instance = setup(tmp_path)
    client.create_release({"body": "untouched", "target_commitish": COMMIT})
    client.release["immutable"] = True
    with pytest.raises(ValueError, match="immutability prevents adding assets"):
        instance.publish()
    assert client.uploads == []
    assert not any(call[0] in {"POST", "PATCH", "DELETE"} for call in client.calls)


def test_anonymous_download_corruption_never_produces_a_success_receipt(tmp_path, monkeypatch):
    _, _, client, instance = setup(tmp_path)
    monkeypatch.setattr(client, "public_identity", lambda url, size: {"bytes": size, "sha256": "f" * 64})
    with pytest.raises(ValueError, match="anonymous"):
        instance.publish()
    assert publisher.read_json(tmp_path / "receipts/publication.json")["passed"] is False
    assets = list((tmp_path / "receipts/assets").glob("*.json"))
    assert publisher.read_json(assets[0])["verified"] is False


def test_local_file_changed_after_plan_is_rejected_before_network_writes(tmp_path):
    _, plan, client, instance = setup(tmp_path)
    Path(plan["assets"][0]["path"]).write_bytes(b"changed independent synthetic asset")
    with pytest.raises(ValueError, match="Local asset changed"):
        instance.publish()
    assert client.calls == [] and client.uploads == []


def test_append_new_episode_does_not_modify_previous_asset_or_release_body(tmp_path):
    _, _, client, first = setup(tmp_path)
    first.publish()
    originals = {k: dict(v) for k, v in client.assets.items()}
    manifest = synthetic_manifest(tmp_path / "next-package", episode=11)
    plan = publisher.load_plan(manifest, COMMIT)
    second = publisher.Publisher(client, plan, tmp_path / "receipts", sleep=lambda _: None)
    result = second.publish()
    assert result["passed"] is True
    assert all(client.assets[k] == v for k, v in originals.items())
    assert len(client.assets) == 8
    assert not any(c[0] == "PATCH" for c in client.calls)


def test_existing_release_notes_change_only_with_explicit_update(tmp_path):
    _, _, client, instance = setup(tmp_path)
    instance.publish(body="First notes")
    instance.publish(body="Explicitly updated notes", update_body=True)
    assert client.release["body"].startswith("Explicitly updated notes")
    assert len([c for c in client.calls if c[0] == "PATCH"]) == 1


def test_git_credential_is_captured_and_never_printed(monkeypatch, capsys):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    secret = "independent-synthetic-secret"
    def fake_run(command, **kwargs):
        assert command[-2:] == ["credential", "fill"]
        assert kwargs["capture_output"] is True
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        return SimpleNamespace(returncode=0, stdout=f"username=synthetic\npassword={secret}\n", stderr="")
    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    assert publisher.credential_token(ROOT) == secret
    assert secret not in capsys.readouterr().out


def test_upload_uses_stream_content_length_mime_and_no_asset_replacement(tmp_path, monkeypatch):
    _, plan, _, _ = setup(tmp_path)
    asset = plan["assets"][0]
    client = publisher.GitHub("independent-synthetic-token")
    captured = {}
    def fake_open(request, **kwargs):
        assert request.method == "POST"
        assert request.get_header("Content-length") == str(asset["bytes"])
        assert request.get_header("Content-type") == "application/zip"
        assert hasattr(request.data, "read")
        captured["first"] = request.data.read(7)
        return io.BytesIO(b'{"id":17}')
    monkeypatch.setattr(client, "_open", fake_open)
    assert client.upload(17, asset) == {"id": 17}
    assert captured["first"] == Path(asset["path"]).read_bytes()[:7]


def test_public_download_has_no_authorization_and_checks_every_byte(monkeypatch):
    client = publisher.GitHub("independent-synthetic-token")
    data = b"independent synthetic public transport bytes"
    def fake_open(request, **kwargs):
        assert kwargs["public"] is True
        assert request.get_header("Authorization") is None
        return io.BytesIO(data)
    monkeypatch.setattr(client, "_open", fake_open)
    actual = client.public_identity(f"https://github.com/{publisher.REPOSITORY}/releases/download/{DELIVERY}/fixture.json", len(data))
    assert actual == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_authenticated_requests_do_not_forward_credentials_on_redirect():
    assert publisher.NoAuthRedirect().redirect_request(None, None, 302, "redirect", {}, "https://untrusted.invalid/") is None
