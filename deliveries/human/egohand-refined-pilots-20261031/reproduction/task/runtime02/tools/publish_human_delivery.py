"""Publish immutable Track 1 Human review assets; default to an offline plan.

Only the explicit --publish option authorizes network writes. Existing assets
are never deleted or replaced. Credentials stay in memory and out of receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import zipfile

REPOSITORY = "Seanwilliam2077/video-to-hoi"
API = "https://api.github.com/repos/" + REPOSITORY
API_VERSION = "2026-03-10"
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
MIMES = {".zip": "application/zip", ".mp4": "video/mp4", ".jpg": "image/jpeg", ".json": "application/json"}
SUFFIXES = re.compile(r"(?:\.zip|-(?:depth-scale|checks|selection|input-bindings|package)\.json"
                      r"|-(?:baseline-comparison|refined-comparison|hands-comparison)\.mp4"
                      r"|-(?:baseline-contact-sheet|refined-contact-sheet|hand-qa-frame-[0-9]{6})\.jpg)\Z")
TRANSIENT = {408, 425, 429, 500, 502, 503, 504}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def durable_json(path, value):
    """Atomic, flushed latest state; immutable run receipts retain history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def regular(path):
    for candidate in (path, *path.parents):
        info = candidate.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
                "Symbolic links and reparse paths are not publishable")
    require(path.is_file(), "Upload source is not a regular file")


def verify_local(asset):
    path = Path(asset["path"])
    regular(path)
    require(path.stat().st_size == asset["bytes"] and digest(path) == asset["sha256"],
            f"Local asset changed: {asset['name']}")


def verify_review_zip(path, delivery_id, episode):
    """Reject unknown bundles, model binaries, original inputs and unsafe paths."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(set(names)) == len(names), "Duplicate ZIP entries")
        banned = {"track_2", "track2", "weights", "checkpoints", "data", "human_masks"}
        extensions = {".pt", ".pth", ".ckpt", ".pkl", ".safetensors", ".onnx", ".parquet", ".h5", ".obj", ".ply", ".glb"}
        for name in names:
            parts = PurePosixPath(name).parts
            member = archive.getinfo(name)
            require(not member.is_dir() and not stat.S_ISLNK(member.external_attr >> 16), "ZIP links and directories are not result files")
            require(not name.startswith("/") and "\\" not in name and ":" not in name and ".." not in parts
                    and not banned.intersection(part.lower() for part in parts)
                    and PurePosixPath(name).suffix.lower() not in extensions,
                    "Unsafe or prohibited ZIP member")
            if name.lower().endswith(".mp4"):
                require(PurePosixPath(name).name in {"comparison.mp4", "hands-comparison.mp4"},
                        "Only generated comparison movies may appear in the review ZIP")
        require({"file-manifest.json", "SHA256SUMS"} <= set(names), "Not a packaged Human review ZIP")
        require(archive.getinfo("file-manifest.json").file_size <= 10 * 1024 * 1024, "Oversized review manifest")
        metadata = json.loads(archive.read("file-manifest.json"))
        require(metadata.get("schema_version") == "human_delivery_files.v1"
                and metadata.get("delivery_id") == delivery_id and metadata.get("episode") == episode,
                "Review ZIP belongs to another delivery or episode")
        checks = {}
        for line in archive.read("SHA256SUMS").decode("utf-8").splitlines():
            fields = line.split("  ", 1)
            require(len(fields) == 2 and SHA.fullmatch(fields[0]) and fields[1] not in checks, "Invalid ZIP checksum inventory")
            checks[fields[1]] = fields[0]
        require(set(checks) | {"SHA256SUMS"} == set(names), "Incomplete ZIP checksum inventory")
        for name, expected in checks.items():
            value = hashlib.sha256()
            with archive.open(name) as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    value.update(block)
            require(value.hexdigest() == expected, "Review ZIP member checksum mismatch")


def load_plan(manifest_path, source_commit, tag=None):
    manifest_path = Path(manifest_path).absolute()
    regular(manifest_path)
    manifest = read_json(manifest_path)
    delivery_id = manifest.get("delivery_id", "")
    require(manifest.get("schema_version") == "human_delivery_upload.v1", "Unsupported upload manifest")
    require(SAFE_ID.fullmatch(delivery_id) and not delivery_id.endswith("."), "A unique safe packaged delivery ID is required")
    require(COMMIT.fullmatch(source_commit) and manifest.get("source_commit") == source_commit, "Source commit must exactly match the packaged full SHA")
    tag = tag or delivery_id
    require(tag == delivery_id, "Release tag must equal the unique packaged delivery ID")
    base = f"https://github.com/{REPOSITORY}/releases/download/{quote(tag, safe='')}"
    entries = manifest.get("assets")
    require(isinstance(entries, list) and bool(entries), "Upload manifest has no assets")
    assets, names = [], set()
    for item in entries:
        require(isinstance(item, dict), "Invalid asset entry")
        name = item.get("name", "")
        prefix = re.fullmatch(re.escape(delivery_id) + r"-ep([0-9]{2})(.*)", name)
        require(prefix and int(prefix[1]) in range(30) and SUFFIXES.fullmatch(prefix[2]), "Asset name is outside the generated Human review allowlist")
        require(name not in names, "Duplicate upload asset name")
        names.add(name)
        require(type(item.get("bytes")) is int and item["bytes"] > 0
                and isinstance(item.get("sha256"), str) and SHA.fullmatch(item["sha256"]), "Invalid asset byte count or SHA256")
        content_type = MIMES[Path(name).suffix]
        # Windows' MIME registry returns this ZIP alias in existing packaging
        # manifests. Preserve those bytes and upload with the canonical MIME.
        accepted_types = {content_type, "application/x-zip-compressed"} if Path(name).suffix == ".zip" else {content_type}
        require(item.get("content_type") in accepted_types, "Incorrect asset MIME type")
        path = Path(item.get("path", ""))
        path = path if path.is_absolute() else manifest_path.parent / path
        regular(path)
        path = path.resolve(strict=True)
        require(path.is_relative_to(manifest_path.parent.resolve()) and path.name == name, "Asset must be an exact named file beside its upload manifest")
        url = base + "/" + quote(name, safe="")
        require(item.get("intended_url") in (None, url), "Intended URL differs from the exact public release destination")
        asset = {**item, "path": str(path), "url": url, "content_type": content_type}
        verify_local(asset)
        if path.suffix == ".zip":
            verify_review_zip(path, delivery_id, int(prefix[1]))
        assets.append(asset)
    require(any(Path(a["name"]).suffix == ".zip" for a in assets), "At least one actual portable review ZIP is required")
    return {"schema_version": "human_delivery_publication.v1", "repository": REPOSITORY, "delivery_id": delivery_id,
            "tag": tag, "source_commit": source_commit, "release_base_url": base,
            "upload_manifest_sha256": digest(manifest_path), "assets": assets}


def credential_token(repo_root):
    """Capture an existing authorized credential without printing its contents."""
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"}
        try:
            result = subprocess.run(["git", "-C", str(repo_root), "credential", "fill"],
                input=f"protocol=https\nhost=github.com\npath={REPOSITORY}.git\n\n", text=True,
                capture_output=True, timeout=30, env=environment, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise ValueError("Existing GitHub credential could not be read") from None
        require(result.returncode == 0, "Existing GitHub credential unavailable; set GH_TOKEN for this process")
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        token = fields.get("password", "")
    require(bool(token) and "\r" not in token and "\n" not in token, "A valid existing GitHub credential is required")
    return token


class APIError(RuntimeError):
    def __init__(self, status=None, *, transient=False):
        self.status, self.transient = status, transient
        super().__init__(f"GitHub request failed ({status if status is not None else 'transport error'})")


class NoAuthRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PublicRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlsplit(newurl)
        require(parsed.scheme == "https" and (parsed.hostname == "github.com" or (parsed.hostname or "").endswith(".githubusercontent.com")),
                "Unexpected public asset redirect host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class GitHub:
    def __init__(self, token):
        self._token = token
        self.auth = build_opener(NoAuthRedirect())
        self.public = build_opener(PublicRedirect())

    def headers(self):
        return {"Authorization": "Bearer " + self._token, "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": API_VERSION, "User-Agent": "video-to-hoi-human-delivery"}

    def _open(self, request, *, public=False):
        try:
            return (self.public if public else self.auth).open(request, timeout=300)
        except HTTPError as error:
            raise APIError(error.code, transient=error.code in TRANSIENT) from None
        except (URLError, OSError, http.client.HTTPException):
            raise APIError(transient=True) from None

    def json(self, method, path, payload=None):
        require(path.startswith("/") and not path.startswith("//"), "Invalid API path")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = self.headers()
        if data is not None:
            headers["Content-Type"] = "application/json"
        with self._open(Request(API + path, data=data, headers=headers, method=method)) as response:
            try:
                return json.load(response)
            except (ValueError, OSError, http.client.HTTPException):
                raise APIError(transient=True) from None

    def upload(self, release_id, asset):
        url = f"https://uploads.github.com/repos/{REPOSITORY}/releases/{release_id}/assets?" + urlencode({"name": asset["name"]})
        headers = {**self.headers(), "Content-Type": asset["content_type"], "Content-Length": str(asset["bytes"])}
        # urllib/http.client streams a readable body with the explicit length.
        with Path(asset["path"]).open("rb") as stream:
            with self._open(Request(url, data=stream, headers=headers, method="POST")) as response:
                try:
                    return json.load(response)
                except (ValueError, OSError, http.client.HTTPException):
                    raise APIError(transient=True) from None

    def public_identity(self, url, expected_bytes):
        require(url.startswith(f"https://github.com/{REPOSITORY}/releases/download/"), "Unexpected public download URL")
        value, size = hashlib.sha256(), 0
        # This request deliberately has no Authorization header.
        with self._open(Request(url, headers={"User-Agent": "video-to-hoi-public-verification"}), public=True) as response:
            while True:
                try:
                    block = response.read(8 * 1024 * 1024)
                except (OSError, http.client.HTTPException):
                    raise APIError(transient=True) from None
                if not block:
                    break
                size += len(block)
                require(size <= expected_bytes, "Public asset is larger than its declared bytes")
                value.update(block)
        return {"bytes": size, "sha256": value.hexdigest()}


class Publisher:
    def __init__(self, client, plan, receipt_dir, *, attempts=3, sleep=time.sleep):
        self.client, self.plan, self.root = client, plan, Path(receipt_dir)
        self.attempts, self.sleep = attempts, sleep
        self._remote_assets = None
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, path, *, missing=False):
        for attempt in range(self.attempts):
            try:
                return self.client.json("GET", path)
            except APIError as error:
                if missing and error.status == 404:
                    return None
                if not error.transient or attempt + 1 == self.attempts:
                    raise
                self.sleep(min(2 ** attempt, 8))

    def verify_tag(self, *, missing=False):
        ref = self.get("/git/ref/tags/" + quote(self.plan["tag"], safe=""), missing=missing)
        if ref is None:
            return
        obj = ref["object"]
        for _ in range(8):
            if obj["type"] == "commit":
                require(obj["sha"] == self.plan["source_commit"], "Release tag points to another commit; use a new delivery ID")
                return
            require(obj["type"] == "tag" and COMMIT.fullmatch(obj["sha"]), "Invalid annotated tag")
            obj = self.get("/git/tags/" + obj["sha"])["object"]
        raise ValueError("Excessively nested release tag")

    def ensure_release(self, *, title=None, body=None, update_body=False):
        commit = self.get("/git/commits/" + self.plan["source_commit"])
        require(commit.get("sha") == self.plan["source_commit"], "Source commit is not available in the target repository")
        self.verify_tag(missing=True)
        path = "/releases/tags/" + quote(self.plan["tag"], safe="")
        marker = f"\n\n<!-- human-delivery:{self.plan['delivery_id']} source-commit:{self.plan['source_commit']} -->"
        description = (body if body is not None else "Track 1 Human candidates in progress. Complete original-frame reviews and native MHR parameters are provided. Reconstruction quality, shared scene scale, physical contact, and official export remain pending. No model weights or original input assets are included.") + marker
        payload = {"tag_name": self.plan["tag"], "target_commitish": self.plan["source_commit"],
                   "name": title or self.plan["delivery_id"], "body": description,
                   "draft": False, "prerelease": True, "make_latest": "false"}
        release = None
        for attempt in range(self.attempts):
            release = self.get(path, missing=True)
            if release is not None:
                break
            try:
                release = self.client.json("POST", "/releases", payload)
                if not isinstance(release, dict) or type(release.get("id")) is not int:
                    raise APIError(transient=True)
                break
            except APIError as error:
                # POST may have committed before a timeout. Recover by exact tag
                # before considering another identical creation request.
                release = self.get(path, missing=True)
                if release is not None:
                    break
                if not (error.transient or error.status == 422) or attempt + 1 == self.attempts:
                    raise
                self.sleep(min(2 ** attempt, 8))
        require(isinstance(release, dict) and type(release.get("id")) is int
                and release.get("tag_name") == self.plan["tag"] and release.get("draft") is False,
                "Release is not the expected public delivery")
        require(release.get("immutable") is not True,
                "GitHub release immutability prevents adding assets after publication; this append workflow requires a release with immutable=false")
        self.verify_tag()
        if update_body and release.get("body") != description:
            require(body is not None, "Updating release notes requires an explicit body file")
            release = self.client.json("PATCH", f"/releases/{release['id']}", {"body": description})
        durable_json(self.root / "release.json", {"repository": REPOSITORY, "tag": self.plan["tag"],
                     "source_commit": self.plan["source_commit"], "release_id": release["id"],
                     "release_base_url": self.plan["release_base_url"], "draft": False,
                     "prerelease": release.get("prerelease"), "verified_at": now()})
        return release

    def find_asset(self, release_id, name, *, fresh=False):
        # One complete inventory avoids thousands of repeated listing requests
        # on an all-30 release. An unknown upload outcome always refreshes it.
        if self._remote_assets is not None and not fresh:
            return self._remote_assets.get(name)
        found = {}
        for page in range(1, 1001):
            assets = self.get(f"/releases/{release_id}/assets?per_page=100&page={page}")
            require(isinstance(assets, list), "Invalid release asset listing")
            for asset in assets:
                require(asset.get("name") not in found, "Duplicate remote asset name")
                found[asset.get("name")] = asset
            if len(assets) < 100:
                self._remote_assets = found
                return found.get(name)
        raise ValueError("Release asset pagination exceeded its bound")

    def event(self, receipt, status, **fields):
        receipt.update(status=status, verified=status == "verified", **fields)
        receipt.setdefault("events", []).append({"at": now(), "status": status})
        durable_json(self.root / "assets" / (receipt["name"] + ".json"), receipt)

    def verify_remote(self, remote, asset):
        require(remote.get("name") == asset["name"] and type(remote.get("id")) is int
                and remote.get("state") == "uploaded" and remote.get("size") == asset["bytes"],
                f"Existing asset has different bytes or incomplete state: {asset['name']}; use a new delivery ID")
        server_digest = remote.get("digest")
        if server_digest is not None:
            require(server_digest == "sha256:" + asset["sha256"], f"Server digest mismatch: {asset['name']}; use a new delivery ID")
        require(remote.get("browser_download_url") == asset["url"], "Remote public asset URL differs from the exact destination")
        # A fresh anonymous full download verifies accessibility, sidecar bytes,
        # and integrity even on servers that do not return a digest field.
        for attempt in range(self.attempts):
            try:
                actual = self.client.public_identity(asset["url"], asset["bytes"])
                break
            except APIError as error:
                if not (error.transient or error.status == 404) or attempt + 1 == self.attempts:
                    raise
                self.sleep(min(2 ** attempt, 8))
        require(actual == {"bytes": asset["bytes"], "sha256": asset["sha256"]}, "Fresh anonymous asset download differs from the local source")
        return {"asset_id": remote["id"], "server_digest": server_digest,
                "server_digest_verified": server_digest is not None,
                "anonymous_download_verified": True, "verification_method": "fresh_anonymous_full_download"}

    def publish_asset(self, release_id, asset):
        verify_local(asset)
        receipt_path = self.root / "assets" / (asset["name"] + ".json")
        binding = {"name": asset["name"], "bytes": asset["bytes"], "sha256": asset["sha256"],
                   "url": asset["url"], "source_commit": self.plan["source_commit"],
                   "release_id": release_id, "repository": REPOSITORY, "tag": self.plan["tag"]}
        previous = read_json(receipt_path) if receipt_path.exists() else None
        if previous is not None:
            require(all(previous.get(k) == v for k, v in binding.items()), "Durable asset receipt binds different bytes or release")
        receipt = {**(previous or {}), **binding}
        self.event(receipt, "checking_remote")
        try:
            for attempt in range(self.attempts):
                remote = self.find_asset(release_id, asset["name"])
                if remote is not None:
                    verification = self.verify_remote(remote, asset)
                    self.event(receipt, "verified", **verification)
                    return receipt
                verify_local(asset)
                self.event(receipt, "uploading", upload_attempt=attempt + 1)
                failure = None
                try:
                    response = self.client.upload(release_id, asset)
                except APIError as error:
                    failure = error
                    self.event(receipt, "outcome_unknown", http_status=error.status)
                # Never repeat a POST merely because its response was lost.
                if failure is None and isinstance(response, dict) and response.get("name") == asset["name"]:
                    remote = response
                    self._remote_assets[asset["name"]] = response
                else:
                    remote = self.find_asset(release_id, asset["name"], fresh=True)
                if remote is not None:
                    verification = self.verify_remote(remote, asset)
                    self.event(receipt, "verified", **verification)
                    return receipt
                if failure is not None and not (failure.transient or failure.status == 422):
                    raise failure
                if attempt + 1 == self.attempts:
                    raise failure or APIError(transient=True)
                self.sleep(min(2 ** attempt, 8))
        except Exception as error:
            self.event(receipt, "failed", error_type=type(error).__name__, error=str(error))
            raise

    def sidecar(self, asset):
        path = self.root / "sidecars" / (asset["name"] + ".sha256")
        content = f"{asset['sha256']}  {asset['name']}\n".encode("ascii")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            require(path.read_bytes() == content, "Existing local checksum sidecar differs")
        else:
            with path.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        return {"name": path.name, "path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                "content_type": "text/plain", "url": self.plan["release_base_url"] + "/" + quote(path.name, safe="")}

    def publish(self, *, title=None, body=None, update_body=False):
        result = {k: v for k, v in self.plan.items() if k != "assets"}
        result.update(passed=False, assets=[], started_at=now(), quality_accepted=False)
        attempt_path = self.root / ("attempt-" + uuid.uuid4().hex + ".json")
        durable_json(attempt_path, result)
        try:
            self._remote_assets = None
            for asset in self.plan["assets"]:
                verify_local(asset)
            release = self.ensure_release(title=title, body=body, update_body=update_body)
            result["release_id"] = release["id"]
            for asset in self.plan["assets"]:
                result["assets"].append(self.publish_asset(release["id"], asset))
                durable_json(attempt_path, result)
                result["assets"].append(self.publish_asset(release["id"], self.sidecar(asset)))
                durable_json(attempt_path, result)
            result.update(passed=True, completed_at=now(), all_manifest_assets_verified=True)
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error), failed_at=now())
            raise
        finally:
            durable_json(attempt_path, result)
            durable_json(self.root / "publication.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--tag", help="Must match the unique packaged delivery ID; defaults to that ID")
    parser.add_argument("--receipt-dir", type=Path, help="Durable publication state; required with --publish")
    parser.add_argument("--publish", action="store_true", help="Explicitly create/reuse the release and upload immutable assets")
    parser.add_argument("--title", help="Title for a newly created release only")
    parser.add_argument("--body-file", type=Path, help="Notes for a new release; existing notes stay unchanged by default")
    parser.add_argument("--update-body", action="store_true", help="Explicitly replace existing release notes with --body-file")
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        require(1 <= args.attempts <= 6, "Attempts must be in 1..6")
        require(not args.update_body or args.body_file is not None, "--update-body requires --body-file")
        plan = load_plan(args.manifest, args.source_commit, args.tag)
        if not args.publish:
            print(json.dumps({"publish_performed": False, "network_used": False, "release_base_url": plan["release_base_url"],
                              "source_commit": args.source_commit, "assets": [{k: a[k] for k in ("name", "bytes", "sha256", "content_type")} for a in plan["assets"]],
                              "checksum_sidecars_to_upload": len(plan["assets"])}))
            return 0
        require(args.receipt_dir is not None, "--publish requires --receipt-dir")
        body = args.body_file.read_text(encoding="utf-8") if args.body_file else None
        token = credential_token(Path(__file__).resolve().parents[1])
        result = Publisher(GitHub(token), plan, args.receipt_dir, attempts=args.attempts).publish(
            title=args.title, body=body, update_body=args.update_body)
        print(json.dumps({"passed": result["passed"], "release_base_url": result["release_base_url"],
                          "verified_assets_including_sidecars": len(result["assets"]),
                          "publication_receipt": str((args.receipt_dir / "publication.json").resolve())}))
        return 0
    except (ValueError, OSError, KeyError, APIError, zipfile.BadZipFile) as error:
        print(f"Human delivery publication failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
