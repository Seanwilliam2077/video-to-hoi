"""Restore only SAM 3D Body, the public MHR model, and Track 1 episode 16.

Local, hash-checked archives are used by default. --download permits missing
files from the exact pinned sources below. Every existing destination is
verified; changed files are never overwritten. No model is deserialized here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile


REPO = Path(__file__).resolve().parents[2]
DOWNLOADS = "work/human-runtime/downloads"
BODY_TARGET = "weights/human/sam-3d-body-dinov3"
BODY_MEMBERS = {"assets/mhr_model.pt", "LICENSE", "model.ckpt", "model_config.yaml", "README.md"}
MHR_URL = "https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip"
MHR_ARCHIVE = {
    "bytes": 198943157,
    "sha256": "e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94",
}
MHR_MODEL = {
    "bytes": 696110248,
    "sha256": "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc",
}
MHR_LICENSE = {
    "bytes": 11358,
    "sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}
TRACK1_REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
TRACK1_PREFIX = f"https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/{TRACK1_REVISION}/"
TRACK1_FILES = [
    {"path": "track_1/meta/info.json", "bytes": 1956,
     "sha256": "eefb27bd1238ed0574858bef8da632c5fe48c9bb4245af419be852372f6a59db"},
    {"path": "track_1/meta/episodes.jsonl", "bytes": 6850,
     "sha256": "11cc802c05ce62d83602e70d4f12991656e5701083a66d43e5452c16d3e4343b"},
    {"path": "track_1/meta/episodes_metadata.jsonl", "bytes": 7597,
     "sha256": "cce105292e2b670f32b49e0504b0ece67b8aa860ea69dc9e728fce97531b53fe"},
    {"path": "track_1/meta/tasks.jsonl", "bytes": 4493,
     "sha256": "1d12b398b55ee9587265fb11f674214f233fd955aeb6acefff09f4c889bd9f23"},
    {"path": "track_1/videos/chunk-000/observation.images.exo_camera/episode_000016.mp4", "bytes": 11744057,
     "sha256": "a37a978ef0f3d0eeaad74f2ed75fb32502d8a1a047587291c14433d052f9eb74"},
]


def _restore_module():
    spec = importlib.util.spec_from_file_location("human_release_restore", REPO / "tools/restore_model_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RESTORE = _restore_module()


def _under(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or ".." in parts.parts or "\\" in relative or ":" in relative:
        raise ValueError(f"unsafe relative asset path: {relative}")
    destination = root.joinpath(*parts.parts)
    if not destination.resolve().is_relative_to(root.resolve()) or RESTORE._has_linked_component(destination):
        raise ValueError(f"asset path leaves root or contains a link: {relative}")
    return destination


def _write_stream(source, destination: Path, record: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
    temporary = Path(temporary_name)
    digest, count = hashlib.sha256(), 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            while block := source.read(1024 * 1024):
                count += len(block)
                if count > record["bytes"]:
                    raise ValueError(f"{destination.name}: input exceeds pinned byte count")
                output.write(block)
                digest.update(block)
        if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"{destination.name}: size or SHA-256 differs from pinned record")
        RESTORE._publish_no_replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class _AssetRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, origin: str):
        self.origin = origin

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise ValueError("asset redirect must remain on approved HTTPS origins")
        host = parsed.hostname or ""
        if self.origin == "mhr":
            allowed = host in RESTORE.ASSET_REDIRECT_HOSTS
        else:
            allowed = host in {"huggingface.co", "cdn-lfs.huggingface.co", "cdn-lfs-us-1.huggingface.co", "cas-bridge.xethub.hf.co"}
        if not allowed:
            raise ValueError(f"asset redirect leaves approved download origins: {host}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _ensure_download(destination: Path, record: dict, url: str, download: bool, origin: str):
    if destination.exists():
        RESTORE._verify_file(destination, record)
        return
    if not download:
        raise FileNotFoundError(f"Missing {destination}; pass --download to fetch the pinned asset")
    approved = url == MHR_URL if origin == "mhr" else url in {TRACK1_PREFIX + f["path"] for f in TRACK1_FILES}
    if not approved:
        raise ValueError("download is not an explicitly pinned Human/Track 1 asset")
    print(f"Downloading {destination.name}", flush=True)
    opener = urllib.request.build_opener(_AssetRedirect(origin))
    request = urllib.request.Request(url, headers={"User-Agent": "video-to-hoi-human-bootstrap/1"})
    with opener.open(request, timeout=60) as response:
        _write_stream(response, destination, record)


def _extract_member(archive: zipfile.ZipFile, member: str, destination: Path, record: dict):
    if destination.exists():
        RESTORE._verify_file(destination, record)
        return
    if archive.namelist().count(member) != 1:
        raise ValueError(f"archive must contain exactly one {member}")
    info = archive.getinfo(member)
    if info.file_size != record["bytes"] or info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError(f"invalid selected archive member: {member}")
    print(f"Restoring {destination.name}", flush=True)
    with archive.open(info) as source:
        _write_stream(source, destination, record)


def _write_json_once(path: Path, content: dict):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != content:
            raise ValueError(f"Existing manifest differs; refusing to overwrite {path}")
        return
    import io
    data = (json.dumps(content, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_stream(io.BytesIO(data), path, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})


def prepare(root: Path = REPO, *, download: bool = False) -> dict:
    root = Path(root).absolute()
    if RESTORE._has_linked_component(root):
        raise ValueError("output root contains a symlink or junction")
    downloads = _under(root, DOWNLOADS)
    downloads.mkdir(parents=True, exist_ok=True)
    release_doc = json.loads((REPO / "deployment/releases/hf-models-20260927-manifest.json").read_text(encoding="utf-8"))
    body_entries = [entry for entry in release_doc["assets"] if entry["id"] == "sam3d-body"]
    if len(body_entries) != 1:
        raise ValueError("release must contain exactly one SAM 3D Body asset")
    filtered = {"schema_version": 1, "release_url": release_doc["release_url"], "assets": body_entries}
    manifest_path = _under(root, DOWNLOADS + "/body-release-manifest.json")
    _write_json_once(manifest_path, filtered)
    body_entry = body_entries[0]
    body_archive = _under(root, DOWNLOADS + "/" + body_entry["archive"]["name"])
    if body_archive.exists():
        RESTORE._verify_file(body_archive, body_entry["archive"])
    else:
        # The filtered manifest can never restore CARI4D, objects, or any dataset.
        RESTORE.restore(manifest_path, downloads, downloads, download=download)

    receipt = json.loads((REPO / "deployment/receipts/sam3d-body.json").read_text(encoding="utf-8"))
    prefix = receipt["target"] + "/"
    selected = {item["path"].removeprefix(prefix): item for item in receipt["files"] if item["path"].startswith(prefix)}
    if set(selected) != BODY_MEMBERS or len(receipt["files"]) != len(BODY_MEMBERS):
        raise ValueError("SAM receipt does not match the exact five allowed model files")
    restored = []
    with zipfile.ZipFile(body_archive) as archive:
        for name in sorted(BODY_MEMBERS):
            item = selected[name]
            destination = _under(root, BODY_TARGET + "/" + name)
            _extract_member(archive, item["path"], destination, item)
            restored.append({"path": destination.relative_to(root).as_posix(), "bytes": item["bytes"], "sha256": item["sha256"]})

    mhr_archive = _under(root, DOWNLOADS + "/mhr-assets-v1.0.1.zip")
    _ensure_download(mhr_archive, MHR_ARCHIVE, MHR_URL, download, "mhr")
    with zipfile.ZipFile(mhr_archive) as archive:
        for member, relative, record in (
            ("assets/mhr_model.pt", "weights/human/mhr_model.pt", MHR_MODEL),
            ("assets/LICENSE.txt", "weights/human/MHR_LICENSE.txt", MHR_LICENSE),
        ):
            _extract_member(archive, member, _under(root, relative), record)
            restored.append({"path": relative, **record})

    for item in TRACK1_FILES:
        destination = _under(root, "data/v2d/" + item["path"])
        _ensure_download(destination, item, TRACK1_PREFIX + item["path"], download, "track1")
    provenance = {
        "schema_version": 1,
        "data_policy": "Only Track 1 episode 16 and its four metadata files; no other challenge assets",
        "sam3d_body": {"source": receipt["source"], "release_url": release_doc["release_url"],
                       "archive": body_entry["archive"], "license_path": BODY_TARGET + "/LICENSE"},
        "mhr": {"source_url": MHR_URL, "release": "v1.0.1", "archive": MHR_ARCHIVE,
                "selected_members": ["assets/mhr_model.pt", "assets/LICENSE.txt"], "license": "Apache-2.0"},
        "model_files": restored,
        "track1": {"dataset": "nvidia/video_to_data_challenge", "revision": TRACK1_REVISION, "files": TRACK1_FILES},
    }
    _write_json_once(_under(root, "weights/human/ASSET_PROVENANCE.json"), provenance)
    print("Verified five SAM 3D Body files, public MHR model/license, and Track 1 episode 16 with metadata.", flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--download", action="store_true", help="fetch only missing pinned model and Track 1 episode-16 assets")
    args = parser.parse_args()
    prepare(args.root, download=args.download)


if __name__ == "__main__":
    main()
