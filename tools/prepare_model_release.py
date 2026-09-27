"""Build reviewed model ZIPs and GitHub-sized parts from verified local files.

This separate release path requires a revision-bound redistribution review.
The general handoff packager continues to exclude gated model weights.
No download, installation, checkpoint loading, or inference is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

if __package__:
    from . import fetch_assets as assets
else:
    import fetch_assets as assets

PROJECT = Path(__file__).resolve().parents[1]
POLICY = PROJECT / "deployment/model-redistribution.json"
DEFAULT_PART_BYTES = 1900 * 1024 * 1024
IDS = ("cari4d", "sam3d-body", "sam3d-objects")


def review_for(asset: dict, policy: dict) -> dict:
    if policy.get("schema_version") != 1:
        raise assets.AssetError("unsupported redistribution review schema")
    matches = [row for row in policy.get("assets", []) if row.get("id") == asset["id"]]
    if len(matches) != 1:
        raise assets.AssetError("one explicit redistribution review is required per asset")
    review = matches[0]
    if (review.get("redistribution_allowed") is not True
            or review.get("revision") != asset.get("revision")
            or review.get("asset_spec_sha256") != assets.spec_hash(asset)):
        raise assets.AssetError("redistribution review does not authorize this exact asset specification")
    if not review.get("notices"):
        raise assets.AssetError("redistribution requires license/notice files")
    return review


def build_asset(asset: dict, policy: dict, *, project: Path, asset_root: Path,
                output_dir: Path, part_bytes: int = DEFAULT_PART_BYTES,
                release_base: str = "") -> dict:
    if not 1 <= part_bytes < 2 * 1024**3:
        raise assets.AssetError("release parts must be nonempty and below 2 GiB")
    if asset["id"] not in IDS or asset.get("kind") != "huggingface":
        raise assets.AssetError("only the three declared Hugging Face models use this release tool")
    review = review_for(asset, policy)
    receipt = assets.verify_receipt(asset_root, asset)
    if receipt.get("schema_version") != 1 or receipt.get("target") != asset["target"]:
        raise assets.AssetError("receipt schema/target mismatch")
    for key in ("kind", "repo_id", "revision", "license"):
        if receipt.get("source", {}).get(key) != asset.get(key):
            raise assets.AssetError("receipt source differs from the asset lock")
    expected = {f"{asset['target']}/{entry['path']}" for entry in asset["files"]}
    actual = [entry["path"] for entry in receipt["files"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise assets.AssetError("receipt must cover exactly all declared model files")
    payload = {}
    for entry in receipt["files"]:
        path = assets.safe_path(asset_root, entry["path"])
        locked = next(f for f in asset["files"] if f"{asset['target']}/{f['path']}" == entry["path"])
        assets.validate_expected(path, locked)
        payload[entry["path"]] = (path, entry["bytes"], entry["sha256"])
    for entry in review["notices"]:
        name = entry["path"]
        if not name.startswith("deployment/licenses/") or name in payload:
            raise assets.AssetError("license notices must be unique files under deployment/licenses/")
        path = assets.safe_path(project, name)
        if not path.is_file() or not path.stat().st_size or assets.sha256(path) != entry["sha256"]:
            raise assets.AssetError("license notice is missing or changed: " + name)
        payload[name] = (path, path.stat().st_size, entry["sha256"])

    portable_receipt = {key: receipt[key] for key in (
        "schema_version", "id", "status", "source", "target", "asset_spec_sha256", "files")}
    portable_receipt["source"] = {key: asset[key] for key in (
        "kind", "repo_id", "revision", "license", "gated") if key in asset}
    portable_receipt["files"] = [{key: entry[key] for key in ("path", "bytes", "sha256")}
                                 for entry in receipt["files"]]
    metadata = {
        f".receipts/{asset['id']}.json": portable_receipt,
        "model-release-provenance.json": {"schema_version": 1, "asset": asset, "redistribution_review": review,
                                           "files": [{"path": name, "bytes": size, "sha256": digest}
                                                     for name, (_, size, digest) in payload.items()]},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = assets.safe_path(output_dir, asset["id"] + ".zip")
    partial = assets.safe_path(output_dir, asset["id"] + ".zip.building")
    if archive.exists() or partial.exists() or list(output_dir.glob(asset["id"] + ".zip.part*")):
        raise assets.AssetError("release output exists; choose an empty output directory for this asset")
    print(f"Building {asset['id']}: {sum(x[1] for x in payload.values())} bytes", flush=True)
    # STORE avoids spending hours recompressing dense numeric model weights.
    # Check the exact bytes written to the ZIP against the verified receipts.
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for name, (path, expected_size, expected_hash) in payload.items():
            digest, total = hashlib.sha256(), 0
            with path.open("rb") as source, z.open(name, "w", force_zip64=True) as target:
                for chunk in iter(lambda: source.read(assets.CHUNK), b""):
                    target.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
            if total != expected_size or digest.hexdigest() != expected_hash:
                raise assets.AssetError("file changed while building the ZIP: " + name)
        for name, record in metadata.items():
            z.writestr(name, json.dumps(record, indent=2) + "\n")
    partial.replace(archive)
    parts, archive_hash = [], hashlib.sha256()
    with archive.open("rb") as source:
        number = 0
        while source.tell() < archive.stat().st_size:
            number += 1
            name = f"{archive.name}.part{number:03d}"
            path = assets.safe_path(output_dir, name)
            digest, size = hashlib.sha256(), 0
            with path.open("xb") as target:
                while size < part_bytes:
                    chunk = source.read(min(assets.CHUNK, part_bytes - size))
                    if not chunk:
                        break
                    target.write(chunk)
                    digest.update(chunk)
                    archive_hash.update(chunk)
                    size += len(chunk)
            part = {"name": name, "bytes": size, "sha256": digest.hexdigest()}
            if release_base:
                part["url"] = release_base.rstrip("/") + "/" + name
            parts.append(part)
            print(f"Prepared {name}: {size} bytes", flush=True)
    return {"id": asset["id"], "repo_id": asset["repo_id"], "revision": asset["revision"],
            "license_name": review["license_name"], "asset_spec_sha256": assets.spec_hash(asset),
            "archive": {"name": archive.name, "bytes": archive.stat().st_size, "sha256": archive_hash.hexdigest()},
            "parts": parts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--ids", nargs="+", choices=IDS, default=list(IDS))
    parser.add_argument("--part-bytes", type=int, default=DEFAULT_PART_BYTES)
    parser.add_argument("--release-base", default="https://github.com/Seanwilliam2077/video-to-hoi/releases/download/hf-models-20260927")
    args = parser.parse_args(argv)
    try:
        doc = assets.load_manifest(args.project / "deployment/assets.lock.json")
        policy = json.loads((args.project / "deployment/model-redistribution.json").read_text(encoding="utf-8"))
        selected = assets.selected_assets(doc, "models", args.ids)
        output = args.output_dir.resolve()
        manifest_path = output / "model-release-manifest.json"
        record = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
            "schema_version": 1, "release_url": "https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/hf-models-20260927",
            "assets": [],
        }
        if record.get("schema_version") != 1 or any(a["id"] in args.ids for a in record["assets"]):
            raise assets.AssetError("release manifest already contains a selected asset or has an unsupported schema")
        for asset in selected:
            record["assets"].append(build_asset(asset, policy, project=args.project.resolve(),
                asset_root=args.asset_root.resolve(), output_dir=output, part_bytes=args.part_bytes,
                release_base=args.release_base))
            temporary = output / "model-release-manifest.json.tmp"
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            temporary.replace(manifest_path)
        checksums = [f"{p['sha256']}  {p['name']}" for a in record["assets"] for p in a["parts"]]
        (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        print(manifest_path)
    except (assets.AssetError, OSError, ValueError, KeyError) as exc:
        print(assets.public_error(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
