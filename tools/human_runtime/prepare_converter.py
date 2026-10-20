"""Retrieve only the official Track 1 mesh-to-MHR converter, never a whole kit.

The ZIP directory and exact allowlisted code member are read by HTTP ranges.
No sample submission, reference arrays or other tracks' file bodies are fetched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen
import zlib

URL = "https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip"
MEMBER = "v2d_submission_kit/tools/track1/mesh_to_mhr_params.py"
SHA256 = "c799ad612fca19620563fcb93bf61e5a4adad0a04251482358746b5f27f8a52e"
ROOT = Path(__file__).resolve().parents[2]


def fetch_range(start: int, length: int) -> bytes:
    request = Request(URL, headers={"Range": f"bytes={start}-{start+length-1}", "Accept-Encoding": "identity"})
    with urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError("Server did not honor byte ranges; refusing a full archive download")
        data = response.read(length + 1)
    if len(data) != length:
        raise ValueError("Unexpected byte-range length")
    return data


def retrieve() -> bytes:
    with urlopen(Request(URL, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])
    tail = fetch_range(size - 22, 22)
    if tail[:4] != b"PK\x05\x06" or struct.unpack_from("<H", tail, 20)[0]:
        raise ValueError("Unexpected ZIP layout; inspect a new kit before updating this helper")
    cd_size, cd_offset = struct.unpack_from("<II", tail, 12)
    directory = fetch_range(cd_offset, cd_size)
    position = 0
    while position < len(directory):
        if directory[position:position + 4] != b"PK\x01\x02":
            raise ValueError("Invalid ZIP central directory")
        method = struct.unpack_from("<H", directory, position + 10)[0]
        compressed_size = struct.unpack_from("<I", directory, position + 20)[0]
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", directory, position + 28)
        offset = struct.unpack_from("<I", directory, position + 42)[0]
        name = directory[position + 46:position + 46 + name_len].decode("utf-8")
        position += 46 + name_len + extra_len + comment_len
        if name != MEMBER:
            continue
        header = fetch_range(offset, 30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid selected file header")
        name_bytes, extra_bytes = struct.unpack_from("<HH", header, 26)
        body = fetch_range(offset + 30 + name_bytes + extra_bytes, compressed_size)
        if method == 8:
            return zlib.decompress(body, -15)
        if method == 0:
            return body
        raise ValueError("Unsupported compression method")
    raise FileNotFoundError(MEMBER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "third_party" / "track1-mhr-converter")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    destination = args.destination / "mesh_to_mhr_params.py"
    if destination.exists():
        data = destination.read_bytes()
    elif args.cache is not None:
        data = args.cache.read_bytes()
    else:
        data = retrieve()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Converter hash differs from the reviewed version; no file installed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    destination.with_name("provenance.json").write_text(json.dumps({
        "source_url": URL, "member": MEMBER, "sha256": SHA256,
        "scope": "Track 1 conversion code only; no reference/data payloads",
    }, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
