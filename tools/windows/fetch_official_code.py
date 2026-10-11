"""Fetch only hash-pinned, allowlisted code members from the official kit.

Never download the complete archive or challenge reference assets. The public
Track 1 sample contains the submission row schema, not reference geometry.
"""
import hashlib
import argparse
import io
import json
import struct
import zipfile
import zlib
from pathlib import Path
import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-updated-code", action="store_true", help="Save changed allowlisted source for review; never execute it")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    evidence = json.loads((root / "docs/evidence/official-track1-20261002.json").read_text())
    url = evidence["sources"]["kit"]["url"]
    pin_path = root / "deployment/windows/official-kit.json"
    pin = json.loads(pin_path.read_text()) if pin_path.exists() else None
    output = root / "third_party/official-track1-kit"
    if pin and not args.review_updated_code and all((output / m["path"]).is_file() and hashlib.sha256((output / m["path"]).read_bytes()).hexdigest() == m["sha256"] for m in pin["members"]):
        print(f"Reused {len(pin['members'])} hash-verified official Track 1 files; no network download")
        return
    session = requests.Session()

    def get_range(start, end):
        response = session.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}, timeout=60)
        if response.status_code != 206 or response.headers.get("Content-Range", "").split("/")[0] != f"bytes {start}-{end}":
            raise RuntimeError("Server did not honor an exact range; refusing an archive download")
        if len(response.content) != end - start + 1:
            raise RuntimeError("Unexpected range size")
        return response.content

    head = session.head(url, timeout=30)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    end = get_range(size - 22, size - 1)
    signature, disk, cd_disk, entries, total, cd_size, cd_offset, comment = struct.unpack("<4s4H2IH", end)
    if signature != b"PK\x05\x06" or disk or cd_disk or comment:
        raise ValueError("Unsupported archive directory")
    central = get_range(cd_offset, cd_offset + cd_size - 1)
    directory = zipfile.ZipFile(io.BytesIO(central + struct.pack("<4s4H2IH", signature, 0, 0, entries, total, cd_size, 0, 0)))
    infos = {i.filename: i for i in directory.infolist()}
    receipts = []
    extras = ["v2dlb/report_schema.py", "v2dlb/mesh_quality.py", "v2dlb/mesh_common.py", "v2dlb/mhr_metrics.py", "data/track_1_sample_submission.parquet"]
    members = evidence["members"] + [{"path": name, "sha256": None} for name in extras if name not in {m["path"] for m in evidence["members"]}]
    if pin:
        members = pin["members"]
    for member in members:
        relative = member["path"]
        if "track_2" in relative or "track2" in relative or (not relative.endswith((".py", ".md", ".txt")) and relative != "data/track_1_sample_submission.parquet"):
            continue
        info = infos["v2d_submission_kit/" + relative]
        offset = info.header_offset
        local = get_range(offset, offset + 29)
        if local[:4] != b"PK\x03\x04":
            raise ValueError("Invalid local header")
        name_size, extra_size = struct.unpack_from("<HH", local, 26)
        begin = offset + 30 + name_size + extra_size
        compressed = get_range(begin, begin + info.compress_size - 1)
        content = zlib.decompress(compressed, -15) if info.compress_type == 8 else compressed
        digest = hashlib.sha256(content).hexdigest()
        if digest != member["sha256"] and not args.review_updated_code:
            raise ValueError(f"Official code changed: {relative}; review before use")
        target = output / relative
        if not target.resolve().is_relative_to(output.resolve()):
            raise ValueError(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        receipts.append({"path": relative, "sha256": digest, "previous_sha256": member["sha256"], "changed": digest != member["sha256"]})
    (output / "acquisition.json").write_text(json.dumps({"url": url, "whole_archive_downloaded": False, "members": receipts}, indent=2))
    print(f"Retrieved {len(receipts)} allowlisted code/documentation/Track 1 schema files; SHA-256 recorded")


if __name__ == "__main__":
    main()
