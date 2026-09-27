"""User-run FoundationPose download; never invoked by the automated asset fetcher.

Run in your own terminal: python -m pip install gdown==5.2.0
Then: python tools/download_foundationpose.py --root artifacts/preparation
For files already obtained in a browser, use --register-only (no network).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__:
    from . import fetch_assets as assets
else:
    import fetch_assets as assets

IDS = ("foundationpose-scorer", "foundationpose-refiner")


def checked_files(root: Path, asset: dict) -> list[dict]:
    """Check file presence and declared integrity without loading model code."""
    result = []
    for entry in asset["files"]:
        relative = f"{asset['target']}/{entry['path']}"
        path = assets.safe_path(root, relative)
        assets.validate_expected(path, entry)
        with path.open("rb") as stream:
            prefix = stream.read(256).lstrip().lower()
        if prefix.startswith((b"<!doctype html", b"<html")):
            raise assets.AssetError(f"HTML page saved instead of a model/config: {relative}")
        result.append({"path": relative, "bytes": path.stat().st_size,
                       "sha256": assets.sha256(path)})
    return result


def download_to_stage(asset: dict, stage: Path) -> None:
    """Only called when the user explicitly runs this standalone downloader."""
    try:
        import gdown
    except ImportError as exc:
        raise assets.AssetError("Install the manual-download dependency: python -m pip install gdown==5.2.0") from exc
    # List first, then download only the two declared files to fixed local paths.
    # Remote folder names are not used as output paths.
    listing = gdown.download_folder(
        url=asset["url"], output=str(stage), skip_download=True,
        quiet=False, use_cookies=False, remaining_ok=False, verify=True,
    )
    if not listing:
        raise assets.AssetError("Cannot list the official folder; download it in your browser and use --register-only")
    required = {entry["path"] for entry in asset["files"]}
    entries = {}
    for item in listing:
        if item.path in required:
            if item.path in entries:
                raise assets.AssetError(f"Duplicate required filename in source: {item.path}")
            entries[item.path] = item
    if required - entries.keys():
        raise assets.AssetError(f"Official folder is missing required files: {sorted(required - entries.keys())}")
    stage.mkdir(parents=True, exist_ok=True)
    for name in sorted(required):
        target = assets.safe_path(stage, name)
        result = gdown.download(id=entries[name].id, output=str(target), quiet=False,
                                use_cookies=False, verify=True, resume=True)
        if result is None:
            raise assets.AssetError(f"Download incomplete: {name}; rerun this command to resume")


def prepare(asset: dict, root: Path, *, register_only: bool = False) -> dict:
    if (asset.get("id") not in IDS or asset.get("kind") != "gdrive_folder"
            or asset.get("download_policy") != "manual_user_only"):
        raise assets.AssetError("This tool only accepts the two user-download FoundationPose entries")
    if {entry["path"] for entry in asset["files"]} != {"model_best.pth", "config.yml"}:
        raise assets.AssetError("FoundationPose requires exactly model_best.pth and config.yml")
    root = root.resolve()
    receipt = assets.safe_path(root, f".receipts/{asset['id']}.json")
    if receipt.exists():
        previous = assets.verify_receipt(root, asset)
        if previous.get("acquisition") != "user_manual":
            raise assets.AssetError("Existing receipt is not a user-manual receipt")
        if previous["files"] != checked_files(root, asset):
            raise assets.AssetError("Existing receipt does not cover the required files exactly")
        return previous
    if not register_only:
        destination = assets.safe_path(root, asset["target"])
        if any(assets.safe_path(destination, entry["path"]).exists() for entry in asset["files"]):
            raise assets.AssetError("Files already exist without a receipt; finish browser download and use --register-only")
        stage_relative = f".downloads/manual-foundationpose/{asset['id']}"
        stage = assets.safe_path(root, stage_relative)
        download_to_stage(asset, stage)
        # Both files must pass before moving either into the runtime layout.
        checked_files(root, {**asset, "target": stage_relative})
        destination.mkdir(parents=True, exist_ok=True)
        for entry in asset["files"]:
            assets.safe_path(stage, entry["path"]).replace(assets.safe_path(destination, entry["path"]))
    files = checked_files(root, asset)
    result = {
        "schema_version": 1, "id": asset["id"], "status": "complete",
        "acquisition": "user_manual",
        "source": {key: asset[key] for key in ("kind", "url", "license", "gated") if key in asset},
        "target": asset["target"], "asset_spec_sha256": assets.spec_hash(asset),
        "integrity_note": "Local size/SHA256 recorded for transfer integrity; no upstream SHA256 is available.",
        "files": files,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = assets.safe_path(root, f".receipts/{asset['id']}.json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(receipt)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/preparation"))
    parser.add_argument("--manifest", type=Path, default=assets.DEFAULT_MANIFEST)
    parser.add_argument("--ids", nargs="+", choices=IDS, default=list(IDS))
    parser.add_argument("--register-only", action="store_true", help="Validate existing files and write receipts without network access")
    args = parser.parse_args(argv)
    try:
        locked = {asset["id"]: asset for asset in assets.load_manifest(args.manifest)["assets"]}
        if any(name not in locked for name in args.ids):
            raise assets.AssetError("Manifest is missing a required FoundationPose entry")
    except (assets.AssetError, OSError, ValueError) as exc:
        parser.error(str(exc))
    failed = False
    for name in dict.fromkeys(args.ids):
        try:
            receipt = prepare(locked[name], args.root, register_only=args.register_only)
            print(f"{name}: complete ({sum(entry['bytes'] for entry in receipt['files'])} bytes)")
            print(f"  Receipt: {assets.receipt_path(args.root.resolve(), locked[name])}")
        except Exception as exc:
            failed = True
            print(f"{name}: {assets.public_error(exc)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
