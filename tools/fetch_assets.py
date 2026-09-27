"""Download the declared model/source assets without installing or executing them.

Works from a Git clone or an extracted source ZIP. Python 3.10+ is required.
Hugging Face downloads use the optional bootstrap dependencies. Google Drive
assets are inventoried only: this workspace's organization blocks that service.
Credentials are read by the download library from its normal environment/cache;
they are never stored in receipts or interpolated into shell commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "deployment" / "assets.lock.json"
CHUNK = 4 * 1024 * 1024


class AssetError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a manifest path without escaping the destination or using symlinks."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise AssetError("asset paths must be nonempty POSIX relative paths")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).anchor or ".." in parts:
        raise AssetError(f"unsafe asset path: {relative}")
    if any(p.lower() in {"track_2", "track2", ".git", ".env"} for p in parts):
        raise AssetError(f"prohibited asset path: {relative}")
    root = root.resolve()
    candidate = root.joinpath(*parts)
    for parent in [candidate, *candidate.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise AssetError(f"symlink asset path: {relative}")
    if not candidate.resolve().is_relative_to(root):
        raise AssetError(f"asset path leaves destination: {relative}")
    return candidate


def check_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise AssetError("asset URLs must use HTTPS without embedded credentials")
    if any(p.lower() in {"track_2", "track2"} for p in parsed.path.split("/")):
        raise AssetError("Track 2 URLs are prohibited")
    if any(k.lower() in {"token", "access_token", "api_key", "signature"}
           for k, _ in urllib.parse.parse_qsl(parsed.query)):
        raise AssetError("signed or credential-bearing URLs must not be committed")
    return url


def load_manifest(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1 or not isinstance(doc.get("assets"), list):
        raise AssetError("unsupported asset manifest")
    seen = set()
    for asset in doc["assets"]:
        name = asset.get("id", "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name) or name in seen:
            raise AssetError(f"invalid or duplicate asset id: {name}")
        seen.add(name)
        if asset.get("kind") not in {"huggingface", "http", "git", "gdrive_folder"}:
            raise AssetError(f"unknown asset kind for {name}")
        target = asset.get("target", "")
        safe_path(Path.cwd(), target)
        if PurePosixPath(target).parts[0] not in {"weights", "third_party"}:
            raise AssetError(f"model/source target must be in weights/ or third_party/: {name}")
        if asset["kind"] in {"git", "huggingface"}:
            if not re.fullmatch(r"[a-f0-9]{40}", asset.get("revision", "")):
                raise AssetError(f"{name} needs a full immutable commit SHA")
        if asset.get("url"):
            check_url(asset["url"])
        if asset["kind"] == "huggingface":
            if asset.get("repo_type", "model") != "model":
                raise AssetError("challenge datasets must use the separate Track 1 downloader")
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", asset.get("repo_id", "")):
                raise AssetError(f"invalid Hugging Face model id: {name}")
        for entry in asset.get("files", []):
            safe_path(Path.cwd(), entry["path"])
            if entry.get("destination"):
                safe_path(Path.cwd(), entry["destination"])
    return doc


def selected_assets(doc: dict, group: str, ids: list[str] | None, include_optional: bool = False) -> list[dict]:
    known = {a["id"] for a in doc["assets"]}
    if ids and set(ids) - known:
        raise AssetError(f"unknown asset ids: {sorted(set(ids) - known)}")
    return [a for a in doc["assets"] if (not ids or a["id"] in ids)
            and (a.get("required", True) or include_optional or bool(ids))
            and (group == "all" or a.get("group", "sources" if a["kind"] == "git" else "models") == group)]


def expected_files(asset: dict) -> list[dict]:
    return asset.get("files", [])


def request(url: str, method: str = "GET"):
    validate_network_url(url)
    opener = urllib.request.build_opener(PolicyRedirectHandler())
    return opener.open(urllib.request.Request(url, method=method,
                       headers={"User-Agent": "video-to-hoi-asset-preparer/1"}), timeout=60)


def validate_network_url(url: str) -> None:
    check_url(url)
    host = urllib.parse.urlsplit(url).hostname or ""
    if host in {"drive.google.com", "drive.usercontent.google.com", "docs.google.com"}:
        raise AssetError("organization policy blocks Google Drive; request an approved delivery method from IT")


class PolicyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_network_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_file(url: str, destination: Path, expected_hash: str | None = None) -> None:
    if destination.is_file() and expected_hash and sha256(destination) == expected_hash:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    with request(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, CHUNK)
    if temporary.stat().st_size == 0:
        raise AssetError("download returned an empty file")
    if expected_hash and sha256(temporary) != expected_hash:
        raise AssetError(f"checksum mismatch: {destination.name}")
    temporary.replace(destination)


def git_archive_url(asset: dict) -> str:
    url = check_url(asset["url"])
    parsed = urllib.parse.urlsplit(url)
    pieces = parsed.path.removesuffix(".git").strip("/").split("/")
    if parsed.hostname != "github.com" or len(pieces) != 2:
        raise AssetError("source archives currently require a public GitHub repository")
    return f"https://codeload.github.com/{'/'.join(pieces)}/zip/{asset['revision']}"


def extract_source(archive: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise AssetError(f"existing source directory has no valid receipt; use a fresh asset root: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        top_levels = {PurePosixPath(m.filename).parts[0] for m in members if m.filename}
        if len(top_levels) != 1:
            raise AssetError("source ZIP must have exactly one root directory")
        # Validate every member before extracting any file.
        outputs = []
        links = []
        for member in members:
            mode = member.external_attr >> 16
            parts = PurePosixPath(member.filename).parts
            if len(parts) < 2 or member.is_dir():
                continue
            out = safe_path(destination, "/".join(parts[1:]))
            if stat.S_ISLNK(mode):
                link = z.read(member).decode("utf-8")
                if "\\" in link or PurePosixPath(link).is_absolute() or PureWindowsPath(link).anchor:
                    raise AssetError("source symlink must have a relative internal target")
                target = (out.parent / link).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise AssetError("source symlink target leaves source tree")
                target = safe_path(destination, target.relative_to(destination.resolve()).as_posix())
                if out == target or out.is_relative_to(target) or target.is_relative_to(out):
                    raise AssetError("source symlink would create a recursive copy")
                links.append((out, target))
            else:
                outputs.append((member, out))
        # No archive member may write through a declared link, even though links
        # are materialized as ordinary copies for Windows/ZIP portability.
        for out, _ in links:
            if any(path == out or path.is_relative_to(out) for _, path in outputs):
                raise AssetError("source archive contains files beneath a symlink")
        for member, out in outputs:
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as inp, out.open("wb") as stream:
                shutil.copyfileobj(inp, stream, CHUNK)
            if mode := (member.external_attr >> 16) & 0o777:
                out.chmod(mode)
        for out, target in links:
            out.parent.mkdir(parents=True, exist_ok=True)
            if target.is_dir():
                shutil.copytree(target, out)
            elif target.is_file():
                shutil.copy2(target, out)
            else:
                raise AssetError("source symlink has a missing or unresolved target")


def _hf_download(asset: dict, root: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download
    target = safe_path(root, asset["target"])
    files = expected_files(asset)
    if not files:
        raise AssetError("Hugging Face downloads require an explicit locked file list")
    paths = []
    for entry in files:
        kwargs = {"repo_id": asset["repo_id"], "repo_type": "model",
                  "revision": asset["revision"], "filename": entry["path"]}
        if asset.get("storage") == "hf_cache":
            kwargs["cache_dir"] = str(target)
        else:
            kwargs["local_dir"] = str(target)
        downloaded = Path(hf_hub_download(**kwargs))
        # HF caches may use links to blob storage. Materialize the snapshot file
        # so the transfer ZIP can contain ordinary files only.
        if downloaded.is_symlink():
            actual = downloaded.resolve()
            if not actual.is_relative_to(target.resolve()):
                raise AssetError("HF cache link escapes the declared asset target")
            tmp = downloaded.with_name(downloaded.name + ".materializing")
            shutil.copyfile(actual, tmp)
            tmp.replace(downloaded)
        if entry.get("destination"):
            out = safe_path(target, entry["destination"])
            if out != downloaded:
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(downloaded, out)
                downloaded = out
        validate_expected(downloaded, entry)
        paths.append(downloaded)
    if asset.get("storage") == "hf_cache":
        model_cache = target / ("models--" + asset["repo_id"].replace("/", "--"))
        for name, revision in asset.get("cache_refs", {}).items():
            if revision != asset["revision"]:
                raise AssetError("cache alias must point at the locked revision")
            ref = safe_path(model_cache / "refs", name)
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(revision, encoding="utf-8")
            paths.append(ref)
    return paths


def validate_expected(path: Path, entry: dict) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssetError(f"missing or empty downloaded file: {entry['path']}")
    size = entry.get("size", entry.get("bytes"))
    if size is not None and path.stat().st_size != size:
        raise AssetError(f"size mismatch: {entry['path']}")
    if entry.get("sha256") and sha256(path) != entry["sha256"]:
        raise AssetError(f"SHA256 mismatch: {entry['path']}")


def receipt_path(root: Path, asset: dict) -> Path:
    return root / ".receipts" / f"{asset['id']}.json"


def verify_receipt(root: Path, asset: dict) -> dict:
    receipt = receipt_path(root, asset)
    if not receipt.is_file():
        raise AssetError("not downloaded: no receipt")
    doc = json.loads(receipt.read_text(encoding="utf-8"))
    if doc.get("status") != "complete" or doc.get("id") != asset["id"]:
        raise AssetError("invalid receipt")
    if doc.get("asset_spec_sha256") != spec_hash(asset):
        raise AssetError("receipt belongs to a different asset specification")
    if not doc.get("files"):
        raise AssetError("receipt contains no files")
    for entry in doc["files"]:
        path = safe_path(root, entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise AssetError(f"missing or changed asset: {entry['path']}")
    return doc


def spec_hash(asset: dict) -> str:
    return hashlib.sha256(json.dumps(asset, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fetch(asset: dict, root: Path) -> dict:
    if asset.get("download_policy") == "blocked_by_organization" or asset["kind"] == "gdrive_folder":
        raise AssetError("organization policy blocks this download; obtain IT approval and an approved delivery method")
    if receipt_path(root, asset).is_file():
        return verify_receipt(root, asset)
    destination = safe_path(root, asset["target"])
    kind = asset["kind"]
    if kind == "huggingface":
        paths = _hf_download(asset, root)
    elif kind == "http":
        destination = safe_path(destination, asset["filename"])
        entry = next((f for f in expected_files(asset) if f["path"] == asset["filename"]), {})
        download_file(asset["url"], destination, asset.get("sha256") or entry.get("sha256"))
        if asset.get("expected_bytes") is not None:
            validate_expected(destination, {"path": asset["target"], "size": asset["expected_bytes"]})
        if entry:
            validate_expected(destination, entry)
        paths = [destination]
    elif kind == "git":
        archive_dir = root / ".downloads"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f"{asset['id']}-{asset['revision']}.zip"
        archive_hash_file = archive.with_suffix(".zip.sha256")
        saved_hash = archive_hash_file.read_text(encoding="utf-8").strip() if archive_hash_file.is_file() else None
        if not (archive.is_file() and saved_hash and sha256(archive) == saved_hash):
            download_file(git_archive_url(asset), archive, asset.get("archive_sha256"))
            archive_hash_file.write_text(sha256(archive) + "\n", encoding="utf-8")
        # Extract to a new staging directory; interrupted extraction does not
        # turn the final source tree into an accepted asset.
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise AssetError("source target already exists without a verified receipt; choose a new asset root")
        with tempfile.TemporaryDirectory(prefix="v2hoi-source-", dir=destination.parent) as temp:
            stage = Path(temp) / "source"
            extract_source(archive, stage)
            stage.rename(destination)
        paths = sorted(p for p in destination.rglob("*") if p.is_file())
    else:
        raise AssetError("unsupported download kind")
    files = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(set(paths))]
    if not files:
        raise AssetError("asset contains no files")
    source = {k: asset[k] for k in ("kind", "repo_id", "revision", "url", "license", "gated") if k in asset}
    if kind == "git":
        source["archive_url"] = git_archive_url(asset)
    result = {"schema_version": 1, "id": asset["id"], "status": "complete", "source": source,
              "target": asset["target"], "asset_spec_sha256": spec_hash(asset), "files": files}
    receipt = receipt_path(root, asset)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def access_check(asset: dict) -> None:
    if asset.get("download_policy") == "blocked_by_organization" or asset["kind"] == "gdrive_folder":
        raise AssetError("organization policy blocks this source; no network check attempted")
    if asset["kind"] == "huggingface":
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        # Check each locked file so a readable README does not hide gated weights.
        for entry in expected_files(asset):
            get_hf_file_metadata(hf_hub_url(asset["repo_id"], entry["path"], revision=asset["revision"]),
                                 timeout=30)
    elif asset["kind"] == "git":
        with request(git_archive_url(asset), "HEAD"):
            pass
    elif asset["kind"] == "http":
        with request(asset["url"], "HEAD"):
            pass
    else:
        # A Drive HTML page can be accessible even when file downloads are not.
        raise AssetError("Google Drive access is verified only by a download, not by its folder landing page")


def public_error(exc: Exception) -> str:
    """Do not publish arbitrary library exceptions that could include auth URLs."""
    if isinstance(exc, AssetError):
        return str(exc)
    if isinstance(exc, ModuleNotFoundError):
        return f"missing bootstrap dependency: {exc.name}; install deployment/requirements/requirements.bootstrap.txt"
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) or getattr(exc, "code", None)
    if status in (401, 403):
        return f"HTTP {status}: model access or authentication required; use an approved account locally"
    if status:
        return f"HTTP {status}; check availability and retry"
    return f"{type(exc).__name__}; check connectivity, available disk space, and the locked asset specification"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "fetch", "verify"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--group", choices=("all", "sources", "models"), default="all")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--include-optional", action="store_true", help="also prepare standalone/export extras; explicit --ids can select them too")
    parser.add_argument("--skip-gated", action="store_true", help="fetch public assets only; skipped assets remain incomplete")
    args = parser.parse_args(argv)
    try:
        doc = load_manifest(args.manifest)
        assets = selected_assets(doc, args.group, args.ids, args.include_optional)
        if not assets:
            raise AssetError("no assets selected")
    except (AssetError, OSError, json.JSONDecodeError) as exc:
        parser.error(public_error(exc))
    root = args.root.resolve()
    if args.command == "plan":
        for asset in assets:
            size = asset.get("expected_bytes")
            amount = f"{size / 1024**3:.2f} GiB" if isinstance(size, (int, float)) else "size unknown"
            print(f"{asset['id']:<30} {asset['kind']:<14} {amount:<15} required={asset.get('required', True)} gated={asset.get('gated', False)}  {asset['target']}")
        print("Plan only. No model, dependency, or container was installed or executed.")
        return 0
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for asset in assets:
        result = {"id": asset["id"], "required": asset.get("required", True)}
        if args.skip_gated and asset.get("gated"):
            result.update(status="skipped", reason="gated asset explicitly deferred")
        else:
            try:
                print(f"{args.command}: {asset['id']}", flush=True)
                if args.command == "fetch":
                    receipt = fetch(asset, root)
                    result.update(status="complete", files=len(receipt["files"]),
                                  bytes=sum(f["bytes"] for f in receipt["files"]))
                elif args.command == "verify":
                    verify_receipt(root, asset)
                    result.update(status="complete")
                else:
                    access_check(asset)
                    result.update(status="accessible")
            except Exception as exc:
                result.update(status="incomplete", reason=public_error(exc))
        print(json.dumps(result, ensure_ascii=True), flush=True)
        results.append(result)
    incomplete = [r["id"] for r in results if r["required"] and r["status"] not in {"complete", "accessible"}]
    report = {"schema_version": 1, "command": args.command, "manifest_sha256": sha256(args.manifest),
              "scope": {"group": args.group, "ids": args.ids, "include_optional": args.include_optional}, "complete_selected_scope": not incomplete,
              "incomplete_required": incomplete, "assets": results,
              "note": "Access checks do not download files or establish runtime compatibility. Partial selections do not establish full baseline readiness."}
    (root / "asset-status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 2 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
