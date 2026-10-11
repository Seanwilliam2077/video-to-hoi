"""Reuse verified project weights, otherwise restore only the needed releases."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def installed(root, asset):
    receipt = root / ".receipts" / f"{asset['id']}.json"
    if not receipt.is_file():
        return False
    record = json.loads(receipt.read_text())
    if record.get("status") != "complete" or record.get("asset_spec_sha256") != asset["asset_spec_sha256"]:
        return False
    if record.get("source", {}).get("revision") != asset["revision"] or not record.get("files"):
        return False
    for item in record["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != item["bytes"]:
            return False
        if digest(path) != item["sha256"]:
            return False
    return True


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "artifacts/baseline-downloads"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "deployment/releases/hf-models-20260927-manifest.json").read_text())
    needed = []
    for asset in manifest["assets"]:
        if asset["id"] not in ("cari4d", "sam3d-body"):
            continue
        if installed(root, asset):
            print(f"Reused SHA-256-verified {asset['id']} weights; no network download", flush=True)
        else:
            needed.append(asset)
    if not needed:
        return
    manifest["assets"] = needed
    selected = out / "selected-manifest.json"
    selected.write_text(json.dumps(manifest, indent=2))
    for asset in needed:
        for part in asset["parts"]:
            destination = out / part["name"]
            if destination.exists() and destination.stat().st_size == part["bytes"]:
                if digest(destination) != part["sha256"]:
                    raise ValueError(f"Cached part is corrupt; inspect {destination}")
                continue
            print(f"Downloading {part['name']}", flush=True)
            subprocess.run(["curl.exe", "-L", "--fail", "--retry", "3", "--connect-timeout", "30",
                            "-C", "-", "--output", str(destination), part["url"]], check=True)
    subprocess.run([sys.executable, str(root / "tools/restore_model_release.py"), str(selected),
                    "--parts-dir", str(out), "--output-dir", str(out / "archives")], cwd=root, check=True)
    for asset in needed:
        with zipfile.ZipFile(out / "archives" / asset["archive"]["name"]) as archive:
            for info in archive.infolist():
                if not (root / info.filename).resolve().is_relative_to(root):
                    raise ValueError(info.filename)
            for info in archive.infolist():
                if info.filename == "model-release-provenance.json":
                    (out / f"{asset['id']}-provenance.json").write_bytes(archive.read(info))
                else:
                    archive.extract(info, root)
        if not installed(root, asset):
            raise ValueError(f"Restored {asset['id']} did not pass its receipt checks")
        print(f"Restored and verified {asset['id']}", flush=True)


if __name__ == "__main__":
    main()
