"""Upload finished input-evidence zips to the project GitHub release.

Reads the git credential helper without printing it. Skips assets that are
already present with the same byte size. Does not upload a partial zip.
"""
from __future__ import annotations

import http.client
import json
import subprocess
from pathlib import Path
import time
import urllib.error
import urllib.request

ROOT = Path(r"D:\V2H\object-depth-20261022\publish")
REPO = "Seanwilliam2077/video-to-hoi"
TAG = "input-evidence-20261022"


def credential() -> str:
    result = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                            text=True, capture_output=True, check=True)
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = values.get("password") or ""
    if not token:
        raise RuntimeError("git credential helper did not provide a GitHub token")
    return token


def request(token: str, url: str, method: str = "GET", data: bytes | None = None, headers: dict | None = None):
    all_headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if headers:
        all_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=all_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            body = response.read()
            return response.status, json.loads(body.decode()) if body else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise RuntimeError(f"GitHub {method} {url} failed: {error.code} {detail}") from None


def release_id(token: str) -> int:
    try:
        _status, payload = request(token, f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}")
        return payload["id"]
    except RuntimeError as error:
        if "404" not in str(error):
            raise
    _status, payload = request(token, f"https://api.github.com/repos/{REPO}/releases", "POST",
                               json.dumps({"tag_name": TAG, "name": "Track 1 object masks and depth evidence",
                                           "body": "Per-episode object masks and compressed MoGe2 depth for the 30 Track 1 videos. Monocular depth is not metric scale. Object masks are SAM2 propagations from reviewed frame-0 points. This does not replace the verified Body input package or accept reconstruction quality.",
                                           "prerelease": True, "draft": False}).encode(),
                               {"Content-Type": "application/json"})
    return payload["id"]


def existing_assets(token: str, release: int) -> dict[str, int]:
    assets = {}
    page = 1
    while True:
        _status, payload = request(token, f"https://api.github.com/repos/{REPO}/releases/{release}/assets?per_page=100&page={page}")
        if not payload:
            return assets
        for asset in payload:
            assets[asset["name"]] = asset["size"]
        if len(payload) < 100:
            return assets
        page += 1


def upload(token: str, release: int, path: Path) -> None:
    size = path.stat().st_size
    connection = http.client.HTTPSConnection("uploads.github.com", timeout=600)
    with path.open("rb") as stream:
        connection.request("POST", f"/repos/{REPO}/releases/{release}/assets?name={path.name}", body=stream, headers={
            "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "Content-Type": "application/zip", "Content-Length": str(size), "X-GitHub-Api-Version": "2022-11-28"})
    response = connection.getresponse()
    detail = response.read(500).decode(errors="replace")
    if response.status not in (201,):
        raise RuntimeError(f"upload {path.name} failed: {response.status} {detail}")


def main() -> int:
    token = credential()
    release = release_id(token)
    print(f"release {release}", flush=True)
    while True:
        present = existing_assets(token, release)
        uploaded = 0
        for receipt_path in sorted(ROOT.glob("*.sha256.json")):
            recorded = json.loads(receipt_path.read_text(encoding="utf-8"))
            zip_path = Path(str(receipt_path).removesuffix(".sha256.json") + ".zip")
            if not zip_path.is_file() or zip_path.stat().st_size != recorded.get("bytes"):
                continue
            if present.get(zip_path.name) == zip_path.stat().st_size:
                continue
            print(f"uploading {zip_path.name} {zip_path.stat().st_size}", flush=True)
            upload(token, release, zip_path)
            uploaded += 1
            print(f"uploaded {zip_path.name}", flush=True)
        complete = len(list(ROOT.glob(f"{TAG}-ep*-depth.zip"))) 
        masks = len(list(ROOT.glob(f"{TAG}-ep*-object-masks.zip")))
        print(f"uploaded_batch {uploaded} local_depth {complete} local_masks {masks}", flush=True)
        if complete == 30 and masks == 30 and uploaded == 0:
            return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
