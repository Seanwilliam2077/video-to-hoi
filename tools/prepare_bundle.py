"""Build a portable, verified source ZIP from a Git clone or GitHub ZIP checkout.

The default bundle contains project code and deployment recipes, never dataset
contents, secrets, submodule placeholders, third-party trees, or model weights.
Explicit --receipts may add non-gated assets approved by deployment/assets.lock.json.
No third-party packages are needed to run this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
MANIFEST_NAME = "bundle-manifest.json"
ROOT_FILES = {
    ".gitignore", "README.md", "pyproject.toml", "uv.lock", "poetry.lock",
    "requirements.txt", "requirements-dev.txt", "environment.yml",
    "environment.yaml", "LICENSE", "LICENSE.md", "NOTICE", "NOTICE.md",
}
SOURCE_SUFFIXES = {".py", ".json", ".toml", ".yaml", ".yml", ".txt", ".html", ".svg", ".sh", ".ps1", ".lock"}
DEPLOY_SUFFIXES = SOURCE_SUFFIXES | {".md", ".dockerignore", ".in"}
FORBIDDEN_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".receipts", "work",
    "outputs", "runs", "scores", "vendor", "third_party", "weights", "dist",
}
SECRET_NAME = re.compile(r"(^|[._-])(env|tokens?|secret|credential|password|api[_-]?key|private[_-]?key)([._-]|$)", re.I)
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".kdbx"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_CHALLENGE_PARTS = {"track_2", "track2", "tier_1", "tier1", "tier_2", "tier2"}
VIDEO_TO_DATA_REVISION = "33129dd0f2d2dcfd1164d43fd076542660756ed2"
# These paths are public source files at the pinned upstream commit. Their
# design-token and Python/environment names would otherwise look like secrets.
VIDEO_TO_DATA_PUBLIC_NAMES = set("""\
docs/chord/_ds/basic-nvidia-design-system-f939d4e7-d0bb-4994-9e26-a471f04510c4/tokens/base.css
docs/chord/_ds/basic-nvidia-design-system-f939d4e7-d0bb-4994-9e26-a471f04510c4/tokens/colors.css
docs/chord/_ds/basic-nvidia-design-system-f939d4e7-d0bb-4994-9e26-a471f04510c4/tokens/fonts.css
docs/chord/_ds/basic-nvidia-design-system-f939d4e7-d0bb-4994-9e26-a471f04510c4/tokens/kaizen-mode.css
docs/chord/_ds/basic-nvidia-design-system-f939d4e7-d0bb-4994-9e26-a471f04510c4/tokens/spacing.css
docs/chord/_ds/basic-nvidia-design-system-f939d4e7-d0bb-4994-9e26-a471f04510c4/tokens/typography.css
docs/hydra-0/_ds/nvidia-kaizen-design-system-b84c2d07-b48f-47c6-9cd1-f3844d902748/kaizen-react/kui-tokens.css
docs/v2d_challenge/_ds/cohere-design-system-b8749743-ab65-4b45-b918-71b6fb5285b5/tokens/colors.css
docs/v2d_challenge/_ds/cohere-design-system-b8749743-ab65-4b45-b918-71b6fb5285b5/tokens/fonts.css
docs/v2d_challenge/_ds/cohere-design-system-b8749743-ab65-4b45-b918-71b6fb5285b5/tokens/spacing.css
docs/v2d_challenge/_ds/cohere-design-system-b8749743-ab65-4b45-b918-71b6fb5285b5/tokens/typography.css
docs/v2d_corl_tutorial/_ds/cohere-design-system-b8749743-ab65-4b45-b918-71b6fb5285b5/tokens/colors.css
docs/v2d_corl_tutorial/_ds/cohere-design-system-b8749743-ab65-4b45-b918-71b6fb5285b5/tokens/fonts.css
docs/v2d_corl_tutorial/_ds/cohere-design-system-b8749743-ab65-4b45-b918-71b6fb5285b5/tokens/spacing.css
docs/v2d_corl_tutorial/_ds/cohere-design-system-b8749743-ab65-4b45-b918-71b6fb5285b5/tokens/typography.css
reconstruction/scripts/setup_css_env.sh
reconstruction/scripts/setup_hitl_aws_env.sh
reconstruction/scripts/setup_kratos_drs_env.sh
reconstruction/scripts/setup_mv_hoi_status_publish_env.sh
reconstruction/workflows/mv_hoi/alembic/env.py
robotic_grounding/flash_chord/src/flash_chord/configs/env/rl.yaml
robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/scene_utils/scene_viewer_env_cfg.py
robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/config/sharpa_wave/recording/sharpa_v2d_record_env_cfg.py
robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/config/sharpa_wave/sharpa_v2d_dr_env_cfg.py
robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/config/sharpa_wave/sharpa_v2d_env_cfg.py
robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/config/sharpa_wave/sharpa_v2d_gr00t_env_cfg.py
robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d/v2d_hand_env_cfg.py
robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/base_env_cfg.py
robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/config/sonic/g1/g1_sonic_env_cfg.py
robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/config/vega_sharpa/vega_sharpa_env_cfg.py
robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/config/vega_sharpa/vega_sharpa_gr00t_env_cfg.py
robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/config/vega_sharpa/vega_sharpa_manip_env_cfg.py
robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/v2d_whole_body/tests/test_sonic_g1_env.py
robotic_grounding/workflow/dev_env.yaml
video_ingestion_agent/osmo_workflows/dev_env.yaml
""".splitlines())
PINNED_PUBLIC_SOURCE_NAMES = {
    ("video-to-data-source", VIDEO_TO_DATA_REVISION, "https://github.com/nvidia-isaac/video_to_data"):
        VIDEO_TO_DATA_PUBLIC_NAMES,
    ("cari4d-dinov3-source", "6876159a11b4df116f30f667f8c9888617df0751", "https://github.com/facebookresearch/dinov3"):
        {"dinov3/env/__init__.py"},
    ("pybind11-source", "aa304c9c7d725ffb9d10af08a3b34cb372307020", "https://github.com/pybind/pybind11"):
        {"tests/env.py"},
    ("sam2-source", "2b90b9f5ceec907a1c18123530e92e794ad901a4", "https://github.com/facebookresearch/sam2"):
        {"demo/frontend/src/theme/tokens.stylex.ts", "demo/frontend/src/vite-env.d.ts"},
}


class BundleError(ValueError):
    """Unsafe or inconsistent bundle input."""


def _relative_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw or ":" in raw or raw.startswith("/"):
        raise BundleError(f"invalid relative path: {raw!r}")
    path = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in raw.split("/")) or path.is_absolute():
        raise BundleError(f"unsafe relative path: {raw!r}")
    return path.as_posix()


def _safe_file(root: Path, relative: str) -> Path:
    rel = _relative_path(relative)
    root = root.resolve()
    path = root / rel
    current = root
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise BundleError(f"symlink/junction is not allowed: {current}")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise BundleError(f"file is missing or outside the allowed root: {path}")
    return path


def _allowed_code(relative: str) -> bool:
    rel = _relative_path(relative)
    path = PurePosixPath(rel)
    parts = path.parts
    if parts[0] == "data" or any(part.lower() in FORBIDDEN_CHALLENGE_PARTS for part in parts):
        return False
    if any(
        part.startswith(".")
        and not (i == 0 and part == ".github")
        and not (len(parts) == 1 and part == ".gitignore")
        and not (parts[0] == "deployment" and i == len(parts) - 1 and part == ".dockerignore")
        for i, part in enumerate(parts)
    ):
        return False
    if any(part in FORBIDDEN_PARTS or part.startswith(".env") or SECRET_NAME.search(part) for part in parts):
        return False
    if path.suffix.lower() in SECRET_SUFFIXES:
        return False
    if len(parts) == 1:
        return rel in ROOT_FILES
    top = parts[0]
    if top == ".github":
        return rel == ".github/CODEOWNERS" or (
            len(parts) == 3 and parts[1] == "workflows" and path.suffix in {".yml", ".yaml"}
        )
    if top == "docs":
        return not rel.startswith("docs/track1-videos/") and path.suffix in {".md", ".svg", ".html"}
    if top == "src":
        return path.suffix in SOURCE_SUFFIXES
    if top == "tests":
        return path.suffix in SOURCE_SUFFIXES
    if top == "tools":
        return path.suffix in SOURCE_SUFFIXES
    if top == "deployment":
        return path.suffix in DEPLOY_SUFFIXES or path.name == "Dockerfile"
    return False


def _extra_untracked(relative: str) -> bool:
    """Only these new, reviewable handoff files can bypass Git's index."""
    if relative.startswith("deployment/"):
        return _allowed_code(relative)
    return relative in {
        "tools/prepare_bundle.py", "tools/fetch_assets.py", "tools/bundle_utils.py",
        "tools/prepare_wheelhouse.py",
        "tests/test_bundle.py", "tests/test_fetch_assets.py", "docs/platform.md",
    }


def _git_index(root: Path) -> tuple[set[str], str | None] | None:
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if Path(top).resolve() != root.resolve():
            return None
        raw = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached"],
            capture_output=True, check=True,
        ).stdout
        paths = {name.decode("utf-8") for name in raw.split(b"\0") if name}
        rev = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return paths, rev
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return None


def _scan_code(root: Path) -> set[str]:
    """Strict fallback for GitHub source archives without .git."""
    paths = set()
    for name in ROOT_FILES:
        if (root / name).exists():
            paths.add(name)
    for dirname in (".github", "docs", "src", "tests", "tools", "deployment"):
        directory = root / dirname
        if not directory.exists():
            continue
        if directory.is_symlink() or (hasattr(directory, "is_junction") and directory.is_junction()):
            raise BundleError(f"symlink/junction is not allowed: {directory}")
        for parent, dirs, files in os.walk(directory, followlinks=False):
            for name in list(dirs):
                child = Path(parent) / name
                relative = child.relative_to(root).as_posix()
                if relative.startswith("docs/track1-videos") or name in FORBIDDEN_PARTS or name.startswith("."):
                    dirs.remove(name)
                elif child.is_symlink() or (hasattr(child, "is_junction") and child.is_junction()):
                    raise BundleError(f"symlink/junction is not allowed: {child}")
            for name in files:
                relative = (Path(parent) / name).relative_to(root).as_posix()
                if _allowed_code(relative):
                    paths.add(relative)
    return paths


def _code_paths(root: Path) -> tuple[set[str], str | None, str]:
    indexed = _git_index(root)
    if indexed is None:
        return _scan_code(root), None, "allowlist"
    tracked, revision = indexed
    candidates = {p for p in tracked if _allowed_code(p)}
    for path in _scan_code(root):
        if _extra_untracked(path):
            candidates.add(path)
    return candidates, revision, "git-index+handoff-allowlist"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise BundleError(f"{path}: expected a JSON object")
    return value


def _asset_paths(root: Path, asset_root: Path, receipts_dir: Path) -> dict[str, dict]:
    lock_path = _safe_file(root, "deployment/assets.lock.json")
    lock = _read_json(lock_path)
    if lock.get("schema_version") != SCHEMA_VERSION or not isinstance(lock.get("assets"), list):
        raise BundleError(f"{lock_path}: unsupported asset lock schema")
    approved = {}
    for asset in lock["assets"]:
        if not isinstance(asset, dict) or not isinstance(asset.get("id"), str):
            raise BundleError(f"{lock_path}: invalid asset entry")
        if asset["id"] in approved:
            raise BundleError(f"{lock_path}: duplicate asset ID {asset['id']}")
        approved[asset["id"]] = asset

    if not receipts_dir.is_dir() or receipts_dir.is_symlink():
        raise BundleError(f"receipts directory is missing or a symlink: {receipts_dir}")
    receipts = sorted(receipts_dir.glob("*.json"))
    if not receipts:
        raise BundleError(f"no asset receipts in {receipts_dir}")
    result = {}
    seen_assets = set()
    for receipt_path in receipts:
        if receipt_path.is_symlink():
            raise BundleError(f"symlink receipt is not allowed: {receipt_path}")
        receipt = _read_json(receipt_path)
        asset_id = receipt.get("id")
        if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("status") != "complete":
            raise BundleError(f"{receipt_path}: receipt is not complete with schema v1")
        if asset_id not in approved:
            raise BundleError(f"{receipt_path}: asset {asset_id!r} is not approved by lock")
        if receipt_path.name != f"{asset_id}.json" or asset_id in seen_assets:
            raise BundleError(f"{receipt_path}: receipt name or asset ID is invalid")
        seen_assets.add(asset_id)
        locked = approved[asset_id]
        if locked.get("download_policy") == "blocked_by_organization":
            raise BundleError(f"{receipt_path}: organization-blocked source cannot be bundled")
        manual_drive = locked.get("kind") == "gdrive_folder"
        if manual_drive and (
            locked.get("download_policy") != "manual_user_only"
            or receipt.get("acquisition") != "user_manual"
        ):
            raise BundleError(f"{receipt_path}: Google Drive asset requires manual_user_only policy and user_manual acquisition")
        if locked.get("gated"):
            # A handoff ZIP must never transport gated model weights.
            continue
        spec_hash = hashlib.sha256(
            json.dumps(locked, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if receipt.get("asset_spec_sha256") != spec_hash:
            raise BundleError(f"{receipt_path}: receipt belongs to another asset specification")
        target = _relative_path(locked.get("target"))
        if not (target.startswith("third_party/") or target.startswith("weights/")):
            raise BundleError(f"{receipt_path}: asset target must be under third_party/ or weights/")
        if receipt.get("target") != target:
            raise BundleError(f"{receipt_path}: target differs from asset lock")
        source = receipt.get("source")
        if not isinstance(source, dict) or source.get("kind") != locked.get("kind"):
            raise BundleError(f"{receipt_path}: source kind differs from asset lock")
        if locked.get("license") and source.get("license") != locked.get("license"):
            raise BundleError(f"{receipt_path}: license differs from asset lock")
        for field in ("repo_id", "revision", "url"):
            expected = locked.get(field)
            if expected and source.get(field) != expected:
                raise BundleError(f"{receipt_path}: source {field} differs from asset lock")
        if source["kind"] == "huggingface":
            if not source.get("repo_id") or not source.get("revision"):
                raise BundleError(f"{receipt_path}: source requires pinned repo_id and revision")
        elif source["kind"] == "git":
            if not str(source.get("url", "")).startswith("https://") or not source.get("revision"):
                raise BundleError(f"{receipt_path}: Git source requires HTTPS URL and pinned revision")
        elif source["kind"] in {"http", "gdrive_folder"}:
            url = str(source.get("url", ""))
            if not url.startswith("https://"):
                raise BundleError(f"{receipt_path}: source requires an HTTPS URL")
            approved_urls = {locked.get("url"), *(
                item if isinstance(item, str) else item.get("url")
                for item in (locked.get("evidence") or [])
                if isinstance(item, (str, dict))
            )}
            if url not in approved_urls:
                raise BundleError(f"{receipt_path}: URL is not approved by asset lock")
        else:
            raise BundleError(f"{receipt_path}: unsupported source kind")
        files = receipt.get("files")
        if not isinstance(files, list) or not files:
            raise BundleError(f"{receipt_path}: complete receipt has no files")
        locked_files = {entry["path"]: entry for entry in locked.get("files", []) if isinstance(entry, dict) and "path" in entry}
        seen_files = set()
        portable_files = []
        for entry in files:
            if not isinstance(entry, dict):
                raise BundleError(f"{receipt_path}: invalid file entry")
            relative = _relative_path(entry.get("path"))
            if not (relative == target or relative.startswith(target + "/")):
                raise BundleError(f"{receipt_path}: file outside approved target: {relative}")
            inside_target = relative[len(target):].lstrip("/")
            public_upstream_name = inside_target in PINNED_PUBLIC_SOURCE_NAMES.get(
                (asset_id, locked.get("revision"), locked.get("url")), set()
            )
            if any(
                part.lower() in {".git", ".venv", *FORBIDDEN_CHALLENGE_PARTS}
                or part.startswith(".env")
                or (SECRET_NAME.search(part) and not public_upstream_name)
                for part in PurePosixPath(relative).parts
            ) or PurePosixPath(relative).suffix.lower() in SECRET_SUFFIXES:
                raise BundleError(f"{receipt_path}: forbidden asset path: {relative}")
            expected_hash = entry.get("sha256")
            expected_bytes = entry.get("bytes")
            if not isinstance(expected_hash, str) or not SHA256.fullmatch(expected_hash.lower()):
                raise BundleError(f"{receipt_path}: {relative} has no SHA256")
            if not isinstance(expected_bytes, int) or expected_bytes < 0:
                raise BundleError(f"{receipt_path}: {relative} has no byte count")
            path = _safe_file(asset_root, relative)
            if path.stat().st_size != expected_bytes or _sha256(path) != expected_hash.lower():
                raise BundleError(f"{receipt_path}: {relative} does not match receipt")
            lock_entry = locked_files.get(inside_target)
            if lock_entry:
                if lock_entry.get("size") is not None and lock_entry["size"] != expected_bytes:
                    raise BundleError(f"{receipt_path}: {relative} differs from locked size")
                if lock_entry.get("sha256") and lock_entry["sha256"].lower() != expected_hash.lower():
                    raise BundleError(f"{receipt_path}: {relative} differs from locked SHA256")
            if relative in result:
                raise BundleError(f"duplicate asset path in receipts: {relative}")
            seen_files.add(relative)
            portable_files.append({"path": relative, "bytes": expected_bytes, "sha256": expected_hash.lower()})
            result[relative] = {
                "path": path, "kind": "asset", "asset_id": asset_id,
                "source": {key: source.get(key) for key in ("kind", "repo_id", "revision", "url", "license") if source.get(key)},
            }
        for locked_path, locked_entry in locked_files.items():
            if locked_entry.get("destination"):
                expected_inside = locked_entry["destination"]
            elif locked.get("storage") == "hf_cache":
                repo_cache = "models--" + locked["repo_id"].replace("/", "--")
                expected_inside = f"{repo_cache}/snapshots/{locked['revision']}/{locked_path}"
            else:
                expected_inside = locked_path
            expected_path = f"{target}/{expected_inside}"
            if expected_path not in seen_files:
                raise BundleError(f"{receipt_path}: locked asset file is absent: {expected_path}")
        if locked.get("storage") == "hf_cache":
            repo_cache = "models--" + locked["repo_id"].replace("/", "--")
            for name in locked.get("cache_refs", {}):
                expected_path = f"{target}/{repo_cache}/refs/{name}"
                if expected_path not in seen_files:
                    raise BundleError(f"{receipt_path}: locked cache ref is absent: {expected_path}")
        portable_receipt = {
            "schema_version": SCHEMA_VERSION, "id": asset_id, "status": "complete",
            "source": {key: source.get(key) for key in ("kind", "repo_id", "revision", "url", "license") if source.get(key)},
            "target": target, "asset_spec_sha256": spec_hash, "files": portable_files,
        }
        if manual_drive:
            portable_receipt["acquisition"] = "user_manual"
        result[f".receipts/{asset_id}.json"] = {
            "data": json.dumps(portable_receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            "kind": "receipt", "asset_id": asset_id,
        }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle(
    root: Path, output: Path, *, receipts: Path | None = None, asset_root: Path | None = None,
) -> dict:
    if Path(root).is_symlink():
        raise BundleError(f"checkout root is a symlink: {root}")
    root = Path(root).resolve()
    output = Path(output).resolve()
    if not root.is_dir():
        raise BundleError(f"checkout does not exist: {root}")
    if os.path.lexists(output) or os.path.lexists(output.with_suffix(output.suffix + ".sha256")):
        raise BundleError(f"bundle output or checksum already exists: {output}")
    code, revision, selection = _code_paths(root)
    if not code:
        raise BundleError(f"no allowlisted source files in {root}")
    payload = {}
    for relative in sorted(code):
        payload[relative] = {"path": _safe_file(root, relative), "kind": "code"}
    if receipts is not None:
        asset_root = Path(asset_root or root).resolve()
        for relative, record in _asset_paths(root, asset_root, Path(receipts).expanduser().absolute()).items():
            if relative in payload:
                raise BundleError(f"asset collides with code: {relative}")
            payload[relative] = record

    files = []
    for relative, record in sorted(payload.items()):
        data = record.get("data")
        size = len(data) if data is not None else record["path"].stat().st_size
        digest = hashlib.sha256(data).hexdigest() if data is not None else _sha256(record["path"])
        item = {
            "path": relative, "bytes": size, "sha256": digest,
            "kind": record["kind"],
        }
        if record["kind"] in {"asset", "receipt"}:
            item["asset_id"] = record["asset_id"]
        if record["kind"] == "asset":
            item["source"] = record["source"]
        files.append(item)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection": selection,
        "git_revision": revision,
        "gated_models_included": False,
        "track2_included": False,
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    companion = output.with_suffix(output.suffix + ".sha256")
    file_info = {item["path"]: item for item in files}
    created_output = False
    created_companion = False
    try:
        with zipfile.ZipFile(output, mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            created_output = True
            for relative, record in sorted(payload.items()):
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                digest = hashlib.sha256()
                length = 0
                with archive.open(info, "w") as target:
                    if "data" in record:
                        target.write(record["data"])
                        digest.update(record["data"])
                        length = len(record["data"])
                    else:
                        with record["path"].open("rb") as source:
                            while chunk := source.read(4 * 1024 * 1024):
                                target.write(chunk)
                                digest.update(chunk)
                                length += len(chunk)
                item = file_info[relative]
                if digest.hexdigest() != item["sha256"] or length != item["bytes"]:
                    raise BundleError(f"{relative} changed while the ZIP was written")
            info = zipfile.ZipInfo(MANIFEST_NAME, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        with companion.open("x", encoding="ascii") as checksum_file:
            created_companion = True
            checksum_file.write(f"{_sha256(output)}  {output.name}\n")
    except Exception:
        # Remove only files this invocation created, never a pre-existing user file.
        if created_output:
            output.unlink(missing_ok=True)
        if created_companion:
            companion.unlink(missing_ok=True)
        raise
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository or GitHub ZIP checkout")
    parser.add_argument("--output", type=Path, required=True, help="new ZIP path; existing files are not overwritten")
    parser.add_argument("--receipts", type=Path, help="directory of complete fetch receipts; enables approved assets")
    parser.add_argument("--asset-root", type=Path, help="root containing asset file paths; defaults to --root")
    args = parser.parse_args(argv)
    try:
        manifest = build_bundle(args.root, args.output, receipts=args.receipts, asset_root=args.asset_root)
    except BundleError as error:
        parser.exit(2, f"bundle error: {error}\n")
    print(f"{args.output}: {len(manifest['files'])} files")
    print(f"{args.output}.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
