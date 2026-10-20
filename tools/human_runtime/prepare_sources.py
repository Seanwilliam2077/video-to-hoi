"""Install a pinned, source-only SAM 3D Body runtime without repository clones.

Only the exact files and SHA-256 hashes in source_manifest.json are accepted.
The small patches select a device and use the pinned local DINOv3 implementation;
they do not replace model operations, weights, or native MHR parameter math.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_DESTINATION = REPO / "third_party" / "sam3d-body-runtime"


def _replace(text: str, old: str, new: str, count: int) -> str:
    if text.count(old) != count:
        raise ValueError(f"Pinned compatibility patch expected {count} occurrences of {old!r}")
    return text.replace(old, new)


def patch_source(path: str, original: bytes) -> bytes:
    text = original.decode("utf-8")
    if path == "sam_3d_body/sam_3d_body_estimator.py":
        text = _replace(text, 'recursive_to(batch, "cuda")', 'recursive_to(batch, self.device)', 2)
        text = _replace(text, 'recursive_to(out, DEVICE)', 'recursive_to(out, self.device)', 2)
        text = _replace(text, '        torch.cuda.empty_cache()', '        if self.device.type == "cuda":\n            torch.cuda.empty_cache()', 1)
    elif path == "sam_3d_body/models/meta_arch/sam3d_body.py":
        text = _replace(text, '.cuda()', '.to(self.device)', 5)
        text = _replace(text, 'recursive_to(batch_lhand, "cuda")', 'recursive_to(batch_lhand, self.device)', 1)
        text = _replace(text, 'recursive_to(batch_rhand, "cuda")', 'recursive_to(batch_rhand, self.device)', 1)
    elif path == "sam_3d_body/models/heads/mhr_head.py":
        # Stage the TorchScript model on the CPU; the caller's model.to(device)
        # transfers it with the rest of SAM. Avoid an implicit CUDA allocation.
        text = _replace(text, 'map_location=("cuda" if torch.cuda.is_available() else "cpu")', 'map_location="cpu"', 1)
    elif path == "sam_3d_body/models/backbones/dinov3.py":
        text = _replace(text,
            '        self.encoder = torch.hub.load(\n            "facebookresearch/dinov3",\n            self.name,\n            source="github",',
            '        from dinov3.hub import backbones as local_backbones\n\n        self.encoder = getattr(local_backbones, self.name)(', 1)
    return text.encode("utf-8")


def prepare(destination: Path, offline: bool = False, research_cache: Path | None = None) -> dict:
    destination = destination.resolve()
    manifest = json.loads(HERE.joinpath("source_manifest.json").read_text(encoding="utf-8-sig"))
    upstream_cache = destination / ".upstream"

    def fetch(record):
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source path: {relative}")
        cache_path = upstream_cache / relative
        original = None
        if cache_path.is_file():
            original = cache_path.read_bytes()
        elif research_cache is not None:
            for directory in ("sam_body_source", "dinov3_source"):
                name = "LICENSE.md" if relative.name == "DINOv3-LICENSE.md" else relative
                candidate = research_cache / directory / name
                if candidate.is_file():
                    data = candidate.read_bytes()
                    if hashlib.sha256(data).hexdigest() == record["sha256"]:
                        original = data
                        break
        if original is None:
            if offline:
                raise FileNotFoundError(f"Missing verified source cache: {cache_path}")
            request = urllib.request.Request(record["url"], headers={"User-Agent": "video-to-hoi-Track1-runtime"})
            with urllib.request.urlopen(request, timeout=60) as response:
                original = response.read()
        if hashlib.sha256(original).hexdigest() != record["sha256"]:
            raise ValueError(f"Upstream checksum mismatch: {relative}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(original)
        patched = patch_source(relative.as_posix(), original)
        local_path = destination / relative
        if local_path.exists() and local_path.read_bytes() not in (original, patched):
            raise ValueError(f"Refusing to overwrite modified runtime source: {local_path}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(patched)
        return {**record, "installed_sha256": hashlib.sha256(patched).hexdigest(), "patched": patched != original}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        installed = list(pool.map(fetch, manifest["files"]))
    result = {**manifest, "destination": str(destination), "files": installed}
    destination.joinpath("installed_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--research-cache", type=Path)
    args = parser.parse_args()
    result = prepare(args.destination, args.offline, args.research_cache)
    print(json.dumps({"destination": result["destination"], "source_files": len(result["files"]), "patched_files": sum(x["patched"] for x in result["files"])}, indent=2))


if __name__ == "__main__":
    main()
