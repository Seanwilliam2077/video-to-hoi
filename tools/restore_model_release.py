"""Restore split model ZIPs from a pinned release manifest without extracting them.

Local parts are used by default. --download explicitly permits missing parts to
be fetched only from this project's GitHub release and GitHub asset redirects.
No model package is installed, imported, or deserialized by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


RELEASE_PATH = "/Seanwilliam2077/video-to-hoi/releases/download/hf-models-20260927/"
RELEASE_PAGE = "https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/hf-models-20260927"
ASSET_REDIRECT_HOSTS = {
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
    "github-releases.githubusercontent.com",
}
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
WINDOWS_DEVICES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                   *(f"LPT{i}" for i in range(1, 10))}
CHUNK = 1024 * 1024


class RestoreError(ValueError):
    """Unsafe manifest, invalid part, or conflicting output."""


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _safe_name(value: object, label: str) -> str:
    if (not isinstance(value, str) or not SAFE_NAME.fullmatch(value) or ".." in value
            or value.endswith(".") or value.split(".", 1)[0].upper() in WINDOWS_DEVICES):
        raise RestoreError(f"unsafe {label}: {value!r}")
    return value


def _positive_bytes(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RestoreError(f"{label} must be a positive integer")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise RestoreError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def _validate_url(url: object, part_name: str, *, redirect: bool = False) -> str:
    if not isinstance(url, str):
        raise RestoreError(f"{part_name}: download URL must be a string")
    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError as error:
        raise RestoreError(f"{part_name}: invalid download URL port") from error
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or port not in (None, 443) or parsed.fragment or "\\" in parsed.path):
        raise RestoreError(f"{part_name}: download URL must use approved HTTPS origin")
    if parsed.hostname.lower() == "github.com":
        if parsed.path != RELEASE_PATH + urllib.parse.quote(part_name) or parsed.query:
            raise RestoreError(f"{part_name}: URL is outside the pinned GitHub release")
    elif not redirect or parsed.hostname.lower() not in ASSET_REDIRECT_HOSTS or not parsed.path.startswith("/"):
        raise RestoreError(f"{part_name}: redirect leaves approved GitHub asset origins")
    return url


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, part_name: str):
        self.part_name = part_name

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_url(newurl, self.part_name, redirect=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_release_url(url: str, part_name: str):
    """Open an approved URL; the redirect handler rejects other origins."""
    _validate_url(url, part_name)
    opener = urllib.request.build_opener(_SafeRedirect(part_name))
    request = urllib.request.Request(url, headers={"User-Agent": "video-to-hoi-release-restore/1"})
    return opener.open(request, timeout=60)


def read_manifest(path: Path) -> list[dict]:
    if _is_link(path) or not path.is_file():
        raise RestoreError(f"manifest is missing or a symlink: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RestoreError(f"cannot read manifest: {error}") from error
    if not isinstance(doc, dict) or doc.get("schema_version") != 1 or not isinstance(doc.get("assets"), list) or not doc["assets"]:
        raise RestoreError("manifest must have schema_version=1 and a nonempty assets list")
    if "release_url" in doc and doc["release_url"] != RELEASE_PAGE:
        raise RestoreError("manifest release_url is not the pinned GitHub release")
    ids, archives, parts_seen = set(), set(), set()
    assets = []
    for entry in doc["assets"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("archive"), dict) or not isinstance(entry.get("parts"), list):
            raise RestoreError("asset must contain archive and parts records")
        asset_id = _safe_name(entry.get("id"), "asset ID")
        archive = entry["archive"]
        archive_name = _safe_name(archive.get("name"), "archive filename")
        if not archive_name.endswith(".zip"):
            raise RestoreError(f"{archive_name}: output must be a .zip")
        archive_bytes = _positive_bytes(archive.get("bytes"), f"{archive_name} bytes")
        archive_hash = _sha256(archive.get("sha256"), f"{archive_name} SHA-256")
        if asset_id.casefold() in ids or archive_name.casefold() in archives:
            raise RestoreError(f"duplicate asset ID or archive name: {asset_id}")
        ids.add(asset_id.casefold())
        archives.add(archive_name.casefold())
        if not entry["parts"] or len(entry["parts"]) > 999:
            raise RestoreError(f"{asset_id}: expected 1-999 ordered parts")
        parts = []
        for index, item in enumerate(entry["parts"], 1):
            if not isinstance(item, dict):
                raise RestoreError(f"{asset_id}: invalid part record")
            name = _safe_name(item.get("name"), "part filename")
            if name != f"{archive_name}.part{index:03d}":
                raise RestoreError(f"{asset_id}: part names must be sequential in manifest order")
            if name.casefold() in parts_seen:
                raise RestoreError(f"duplicate part filename: {name}")
            parts_seen.add(name.casefold())
            size = _positive_bytes(item.get("bytes"), f"{name} bytes")
            digest = _sha256(item.get("sha256"), f"{name} SHA-256")
            url = item.get("url")
            if url is not None:
                _validate_url(url, name)
            parts.append({"name": name, "bytes": size, "sha256": digest, "url": url})
        if sum(part["bytes"] for part in parts) != archive_bytes:
            raise RestoreError(f"{asset_id}: part byte counts do not equal archive byte count")
        assets.append({"id": asset_id, "archive": {"name": archive_name, "bytes": archive_bytes,
                                                     "sha256": archive_hash}, "parts": parts})
    return assets


def _file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source:
        while chunk := source.read(CHUNK):
            count += len(chunk)
            digest.update(chunk)
    return count, digest.hexdigest()


def _verify_file(path: Path, record: dict) -> None:
    if _is_link(path) or not path.is_file():
        raise RestoreError(f"missing or linked file: {path}")
    if path.stat().st_size != record["bytes"]:
        raise RestoreError(f"{path.name}: byte count differs from manifest")
    if _file_digest(path)[1] != record["sha256"]:
        raise RestoreError(f"{path.name}: SHA-256 differs from manifest")


def _publish_no_replace(temporary: Path, final: Path) -> None:
    """Atomically expose a completed same-directory file without replacing a race winner."""
    try:
        os.link(temporary, final)
    except FileExistsError as error:
        raise RestoreError(f"{final.name}: destination appeared; refusing to overwrite") from error


def _download_part(record: dict, parts_dir: Path) -> None:
    url = record["url"]
    if url is None:
        raise RestoreError(f"{record['name']}: missing part has no approved release URL")
    final = parts_dir / record["name"]
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{record['name']}.", suffix=".partial", dir=parts_dir)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(descriptor, "wb") as target, open_release_url(url, record["name"]) as response:
            while chunk := response.read(CHUNK):
                count += len(chunk)
                if count > record["bytes"]:
                    raise RestoreError(f"{record['name']}: download exceeds locked byte count")
                target.write(chunk)
                digest.update(chunk)
        if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise RestoreError(f"{record['name']}: downloaded size or SHA-256 differs from manifest")
        if final.exists() or _is_link(final):
            _verify_file(final, record)
        else:
            _publish_no_replace(temporary, final)
    except (OSError, urllib.error.URLError) as error:
        raise RestoreError(f"{record['name']}: download failed: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _restore_asset(asset: dict, parts_dir: Path, output_dir: Path, download: bool) -> dict:
    for part in asset["parts"]:
        path = parts_dir / part["name"]
        if not path.exists() and not _is_link(path):
            if not download:
                raise RestoreError(f"missing part: {path}; pass --download to fetch approved release parts")
            _download_part(part, parts_dir)
        _verify_file(path, part)

    archive = asset["archive"]
    final = output_dir / archive["name"]
    if final.exists() or _is_link(final):
        _verify_file(final, archive)
        if not zipfile.is_zipfile(final):
            raise RestoreError(f"{final.name}: output is not a ZIP")
        return {"id": asset["id"], "archive": str(final), "status": "existing",
                "bytes": archive["bytes"], "sha256": archive["sha256"]}

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{archive['name']}.", suffix=".partial", dir=output_dir)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(descriptor, "wb") as target:
            for part in asset["parts"]:
                part_digest = hashlib.sha256()
                part_count = 0
                with (parts_dir / part["name"]).open("rb") as source:
                    while chunk := source.read(CHUNK):
                        target.write(chunk)
                        digest.update(chunk)
                        part_digest.update(chunk)
                        part_count += len(chunk)
                        count += len(chunk)
                if part_count != part["bytes"] or part_digest.hexdigest() != part["sha256"]:
                    raise RestoreError(f"{part['name']}: changed during restoration")
        if count != archive["bytes"] or digest.hexdigest() != archive["sha256"]:
            raise RestoreError(f"{archive['name']}: restored size or SHA-256 differs from manifest")
        if not zipfile.is_zipfile(temporary):
            raise RestoreError(f"{archive['name']}: restored file is not a ZIP")
        if final.exists() or _is_link(final):
            raise RestoreError(f"{archive['name']}: output appeared during restoration")
        _publish_no_replace(temporary, final)
    finally:
        temporary.unlink(missing_ok=True)
    return {"id": asset["id"], "archive": str(final), "status": "created",
            "bytes": archive["bytes"], "sha256": archive["sha256"]}


def restore(manifest: Path, parts_dir: Path, output_dir: Path, *, download: bool = False) -> list[dict]:
    assets = read_manifest(Path(manifest))
    parts_dir, output_dir = Path(parts_dir), Path(output_dir)
    if _is_link(parts_dir) or not parts_dir.is_dir():
        raise RestoreError(f"parts directory is missing or a symlink: {parts_dir}")
    if _is_link(output_dir):
        raise RestoreError(f"output directory is a symlink: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return [_restore_asset(asset, parts_dir, output_dir, download) for asset in assets]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="fetch only missing approved GitHub Release parts")
    args = parser.parse_args(argv)
    try:
        results = restore(args.manifest, args.parts_dir, args.output_dir, download=args.download)
    except (OSError, RestoreError) as error:
        parser.exit(2, f"restore error: {error}\n")
    print(json.dumps({"archives": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
