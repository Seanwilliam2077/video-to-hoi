"""Build docs/track1-videos/: a browsable page of the 30 Track 1 videos with small previews.

    python -m v2hoi.download --videos          # the full videos, into data/v2d/ (not committed)
    python tools/track1_gallery/build.py       # previews and index.html, into docs/track1-videos/

Previews are re-encoded at 768x576 and 15 fps so that the whole set stays
small enough to commit; the page links each full video both as a local copy
and on Hugging Face. Existing previews are kept; pass --force to redo them.
Needs ffmpeg on PATH (or the imageio-ffmpeg package).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACK1 = ROOT / "data" / "v2d" / "track_1"
OUT = ROOT / "docs" / "track1-videos"
HF = "https://huggingface.co/datasets/nvidia/video_to_data_challenge"
VIDEO_DIR = "videos/chunk-000/observation.images.exo_camera"

NAMES = {
    "hula_hoop": "Hula hoop", "big_red_bowl": "Big red bowl", "iron": "Iron", "white_desk": "White desk",
    "black_pan": "Black pan", "foam_grass_block": "Foam grass block", "paint_roller": "Paint roller",
    "pink_foam_roll": "Pink foam roll", "short_wood_stool": "Short wood stool", "white_laptop_cart": "White laptop cart",
}
# Episode groups from docs/design.md, section 4.
GROUPS = [
    (range(0, 3), "Thin ring", "thin"), (range(3, 9), "Hand-held", "hand"), (range(9, 12), "Large furniture", "furniture"),
    (range(12, 24), "Hand-held", "hand"), (range(24, 27), "Body contact", "contact"), (range(27, 30), "Large furniture", "furniture"),
]


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found: install it or `pip install imageio-ffmpeg`")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recorded(sequence_id: str) -> str:
    stamp = datetime.strptime(sequence_id[:19], "%Y-%m-%d_%H-%M-%S")
    return f"{stamp:%b} {stamp.day}, {stamp:%Y %H:%M}"


def encode(exe: str, src: Path, poster: Path, preview: Path, force: bool) -> None:
    if force or not poster.exists():
        subprocess.run([exe, "-v", "error", "-y", "-ss", "4", "-i", str(src), "-frames:v", "1",
                        "-vf", "scale=768:-2", "-q:v", "4", str(poster)], check=True)
    if force or not preview.exists():
        subprocess.run([exe, "-v", "error", "-y", "-i", str(src), "-vf", "scale=768:-2,fps=15", "-an",
                        "-c:v", "libx264", "-crf", "24", "-preset", "slow", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(preview)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-encode previews that already exist")
    args = parser.parse_args()

    lengths = {r["episode_index"]: r for r in jsonl(TRACK1 / "meta" / "episodes.jsonl")}
    meta = sorted(jsonl(TRACK1 / "meta" / "episodes_metadata.jsonl"), key=lambda m: m["episode_index"])
    missing = [m["episode_index"] for m in meta if not (TRACK1 / VIDEO_DIR / f"episode_{m['episode_index']:06d}.mp4").is_file()]
    if missing:
        raise SystemExit(f"videos missing for episodes {missing}: run python -m v2hoi.download --videos")

    exe = ffmpeg()
    (OUT / "media").mkdir(parents=True, exist_ok=True)
    videos = []
    for m in meta:
        e = m["episode_index"]
        src = TRACK1 / VIDEO_DIR / f"episode_{e:06d}.mp4"
        poster, preview = OUT / "media" / f"ep{e:02d}.jpg", OUT / "media" / f"ep{e:02d}.mp4"
        encode(exe, src, poster, preview, args.force)
        name, key = next((g, k) for r, g, k in GROUPS if e in r)
        frames = lengths[e]["length"]
        videos.append({
            "ep": e, "object": m["object"], "objectName": NAMES.get(m["object"], m["object"]),
            "prompt": m["object_prompt"], "camera": m["camera"], "frames": frames, "seconds": frames / 30,
            "sequence": m["sequence_id"], "recorded": recorded(m["sequence_id"]), "task": lengths[e]["tasks"][0],
            "group": name, "groupKey": key,
            "poster": f"media/ep{e:02d}.jpg", "preview": f"media/ep{e:02d}.mp4",
            "local": f"../../data/v2d/track_1/{VIDEO_DIR}/episode_{e:06d}.mp4",
            "hf": f"{HF}/blob/main/track_1/{VIDEO_DIR}/episode_{e:06d}.mp4",
        })
        print(f"episode {e:2d} {m['object']:<18} preview {preview.stat().st_size / 1e6:.2f} MB")

    frames = sum(v["frames"] for v in videos)
    cams = {}
    for v in videos:
        cams[v["camera"].split("_")[0]] = cams.get(v["camera"].split("_")[0], 0) + 1
    data = {
        "intro": "The 30 challenge videos of V2D Track 1: one static third-person camera each, ten objects with three clips "
                 "per object. Each card plays a small preview; the full 1536 x 1152 video is linked as a local copy "
                 "(after python -m v2hoi.download --videos) and on Hugging Face.",
        "stats": {"episodes": len(videos), "frames": frames, "duration": f"{frames // 30 // 60} min {frames // 30 % 60} s",
                  "cameras": ", ".join(f"{k} {n}" for k, n in sorted(cams.items())), "preview": "768 x 576, 15 fps"},
        "objects": [{"id": k, "name": NAMES.get(k, k)} for k in dict.fromkeys(v["object"] for v in videos)],
        "cameras": sorted({v["camera"] for v in videos}),
        "videos": videos,
        "attribution": "Videos: NVIDIA Video to Data (V2D) Challenge dataset, nvidia/video_to_data_challenge, Track 1; "
                       "previews re-encoded from the originals. Licensed under",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "regenerate": "Regenerate with python tools/track1_gallery/build.py.",
    }
    template = (Path(__file__).parent / "template.html").read_text(encoding="utf-8")
    html = template.replace("__GALLERY_JSON__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    (OUT / "index.html").write_text(html, encoding="utf-8", newline="\n")
    total = sum(p.stat().st_size for p in (OUT / "media").iterdir())
    print(f"wrote {OUT / 'index.html'}; media {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
