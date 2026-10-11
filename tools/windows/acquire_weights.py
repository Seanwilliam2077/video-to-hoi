"""Reuse verified project weights, otherwise restore only the needed releases."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
import requests


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append", choices=["cari4d", "sam3d-body", "sam3d-objects"],
                        help="Repeat to select required releases; defaults to the existing CARI4D and body weights")
    parser.add_argument("--download-workers", type=int, default=1)
    parser.add_argument("--direct-download", action="store_true",
                        help="Bypass the proxy for these public Release transfers only")
    args = parser.parse_args()
    selected_ids = args.asset or ["cari4d", "sam3d-body"]
    root = Path(__file__).resolve().parents[2]
    out = root / "artifacts/baseline-downloads"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "deployment/releases/hf-models-20260927-manifest.json").read_text())
    needed = []
    for asset in manifest["assets"]:
        if asset["id"] not in selected_ids:
            continue
        if installed(root, asset):
            print(f"Reused SHA-256-verified {asset['id']} weights; no network download", flush=True)
        else:
            needed.append(asset)
    if not needed:
        return
    need_parts = []
    for asset in needed:
        archive = out / "archives" / asset["archive"]["name"]
        if archive.is_file():
            if archive.stat().st_size != asset["archive"]["bytes"] or digest(archive) != asset["archive"]["sha256"]:
                raise ValueError(f"Cached archive is corrupt; inspect {archive}")
            print(f"Reused SHA-256-verified {asset['id']} archive; no part downloads", flush=True)
        else:
            need_parts.append(asset)
    manifest["assets"] = need_parts
    selected = out / "selected-manifest.json"
    selected.write_text(json.dumps(manifest, indent=2))
    def download(part):
        destination = out / part["name"]
        if destination.exists() and destination.stat().st_size == part["bytes"]:
            if digest(destination) != part["sha256"]:
                raise ValueError(f"Cached part is corrupt; inspect {destination}")
            return
        print(f"Downloading/resuming {part['name']}", flush=True)
        session = requests.Session()
        session.trust_env = not args.direct_download
        failures = 0
        while (destination.stat().st_size if destination.exists() else 0) < part["bytes"]:
            offset = destination.stat().st_size if destination.exists() else 0
            end = min(offset + 128 * 1024**2, part["bytes"]) - 1
            try:
                with session.get(part["url"], params={"v2hoi_resume": offset},
                    headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"},
                    stream=True, timeout=(30, 45)) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{part['bytes']}":
                        raise RuntimeError(f"Server ignored exact resume range for {part['name']}; existing bytes retained")
                    with destination.open("ab") as output:
                        for chunk in response.iter_content(1024**2):
                            if output.tell() + len(chunk) > end + 1:
                                raise ValueError("Resume response exceeded requested range")
                            output.write(chunk)
                if destination.stat().st_size != end + 1:
                    raise requests.exceptions.ChunkedEncodingError("Resume response ended before the requested range completed")
                failures = 0
                print(f"{part['name']}: {destination.stat().st_size}/{part['bytes']} bytes", flush=True)
            except requests.RequestException:
                failures += 1
                print(f"Retrying interrupted range for {part['name']}; all received bytes retained", flush=True)
                if failures >= 12:
                    raise RuntimeError(f"Repeated transfer failures for {part['name']}; cached partial retained")
        session.close()
        if digest(destination) != part["sha256"]:
            raise ValueError(f"Downloaded part hash failed; inspect {destination}")
        print(f"Downloaded {part['name']}", flush=True)
    if need_parts:
        with ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as pool:
            list(pool.map(download, [part for asset in need_parts for part in asset["parts"]]))
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
