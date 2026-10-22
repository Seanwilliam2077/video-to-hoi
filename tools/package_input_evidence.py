"""Zip completed object-mask and depth episodes without touching a live writer.

A depth episode is packed only after depth-report.json says complete. An object
episode is packed only after its mask manifest lists every frame. Existing zips
are left unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

ROOT = Path(r"D:\V2H\object-depth-20261022")
RELEASE = "input-evidence-20261022"
BASE = f"https://github.com/Seanwilliam2077/video-to-hoi/releases/download/{RELEASE}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_zip(destination: Path, files: list[tuple[Path, str]]) -> None:
    temporary = destination.with_suffix(".zip.partial")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for source, name in files:
            archive.write(source, name)
    temporary.replace(destination)


def object_ready(episode: Path, frames: int) -> bool:
    manifest_path = episode / "masks-manifest.json"
    masks = episode / "object_masks"
    if not manifest_path.is_file() or not masks.is_dir():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("frame_count") == frames and len(list(masks.glob("*.png"))) == frames


def depth_ready(episode: Path, frames: int) -> bool:
    report_path = episode / "depth-report.json"
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return (report.get("status") == "complete" and report.get("frames_written") == frames
            and (episode / "depth.f16").is_file() and (episode / "valid.u8").is_file())


def pack(kind: str, index: int, frames: int, output: Path) -> dict | None:
    name = f"{RELEASE}-ep{index:02d}-{kind}.zip"
    destination = output / name
    receipt = destination.with_suffix(".sha256.json")
    if destination.is_file() and receipt.is_file() and destination.stat().st_size > 0:
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
        if recorded.get("bytes") == destination.stat().st_size and recorded.get("sha256"):
            return {"name": name, "bytes": recorded["bytes"], "sha256": recorded["sha256"], "frames": frames, "cached": True}
    if kind == "object-masks":
        episode = ROOT / "object-masks" / f"episode_{index:06d}"
        if not object_ready(episode, frames):
            return None
        files = [(episode / "masks-manifest.json", "masks-manifest.json"),
                 (episode / "masks-provenance.json", "masks-provenance.json"),
                 (episode / "observations.json", "observations.json")]
        files.extend((path, f"object_masks/{path.name}") for path in sorted((episode / "object_masks").glob("*.png")))
        files.extend((path, f"mask-qa/{path.name}") for path in sorted((episode / "mask-qa").glob("*.jpg")))
    else:
        episode = ROOT / "depth" / f"episode_{index:06d}"
        if not depth_ready(episode, frames):
            return None
        files = [(episode / "depth-report.json", "depth-report.json"),
                 (episode / "depth.f16", "depth.f16"),
                 (episode / "valid.u8", "valid.u8")]
    print(f"packing {name}", flush=True)
    write_zip(destination, files)
    packed = {"name": name, "bytes": destination.stat().st_size, "sha256": sha256(destination), "frames": frames}
    receipt.write_text(json.dumps({"bytes": packed["bytes"], "sha256": packed["sha256"]}) + "\n", encoding="utf-8")
    print(f"packed {name} {packed['bytes']} {packed['sha256'][:12]}", flush=True)
    return packed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "publish")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "package" / "data" / "prepared" / "track1" / "manifest.json").read_text(encoding="utf-8"))
    frames = {row["episode_index"]: row["frame_count"] for row in manifest["episodes"]}
    objects = {episode["ep"]: episode["object"] for episode in json.loads(Path(r"C:\Users\seanzjxiao\video-to-hoi\tracking\board.json").read_text(encoding="utf-8"))["episodes"]}
    while True:
        entries = []
        for index in range(30):
            masks = pack("object-masks", index, frames[index], args.output)
            depth = pack("depth", index, frames[index], args.output)
            if not masks and not depth:
                continue
            entry = {"episode_id": f"episode_{index:06d}", "frame_count": frames[index], "object_id": objects[index]}
            if masks:
                entry["object_masks"] = {"url": f"{BASE}/{masks['name']}", "sha256": masks["sha256"], "bytes": masks["bytes"], "frames": frames[index]}
            if depth:
                entry["depth"] = {"url": f"{BASE}/{depth['name']}", "sha256": depth["sha256"], "bytes": depth["bytes"], "frames": frames[index]}
            entries.append(entry)
        document = {"schema_version": "hoi_input_evidence.v1", "delivery_id": RELEASE, "entries": entries}
        target = args.output / "input-evidence.json"
        target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        complete = sum(1 for entry in entries if "object_masks" in entry and "depth" in entry)
        print(f"evidence entries {len(entries)} complete pairs {complete}", flush=True)
        if not args.watch or complete == 30:
            return 0
        time.sleep(20)


if __name__ == "__main__":
    raise SystemExit(main())
