"""The episodes a run works on, read from a dataset root's LeRobot metadata.

Track 1 is the challenge input. Tier 1 is the development set with ground
truth; it may be used for scoring only, never as input to a submission.
Both list the object and the frame count; only Track 1 names the physical
camera.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from v2hoi.dataset import TIER1_ROOT, _child_path, read_metadata

TRACK1_ROOT = Path("data/v2d/track_1")
DATASETS = {"track1": TRACK1_ROOT, "tier1": TIER1_ROOT}


@dataclass(frozen=True)
class Clip:
    dataset: str
    episode: int
    object_name: str
    prompt: str  # text description of the object, for detection
    camera: str | None  # physical camera; None when the dataset does not say
    n_frames: int
    fps: float
    width: int
    height: int
    video: Path


def _lengths(root: Path) -> dict[int, int]:
    with open(root / "meta" / "episodes.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["episode_index"]: row["length"] for row in rows}


def load_clips(dataset: str, episodes: list[int] | None = None, root: Path | None = None) -> list[Clip]:
    """Clips of ``dataset`` ("track1" or "tier1"), all episodes by default."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}")
    root = Path(root or DATASETS[dataset])
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    meta = read_metadata(root)
    lengths = _lengths(root)
    prompts = {o["name"]: o.get("prompt", "") for o in info.get("objects", [])}

    episodes = sorted(meta) if episodes is None else episodes
    unknown = [e for e in episodes if e not in meta]
    if unknown:
        raise ValueError(f"{dataset} has no episodes {unknown}")

    clips = []
    for e in episodes:
        row = meta[e]
        name = row["object"]
        video_key = row.get("video_key", "observation.images.exo_camera")
        height, width = info["features"][video_key]["shape"][:2]
        rel = info["video_path"].format(
            episode_chunk=e // info.get("chunks_size", 1000), video_key=video_key, episode_index=e
        )
        clips.append(Clip(
            dataset=dataset,
            episode=e,
            object_name=name,
            prompt=row.get("object_prompt") or prompts.get(name, name),
            camera=row.get("camera"),
            n_frames=int(row.get("frames") or lengths[e]),
            fps=float(info.get("fps", 30)),
            width=int(width),
            height=int(height),
            video=_child_path(root, rel, "video_path"),
        ))
    return clips


def group_by_object(clips: list[Clip]) -> dict[str, list[Clip]]:
    grouped: dict[str, list[Clip]] = {}
    for clip in clips:
        grouped.setdefault(clip.object_name, []).append(clip)
    return grouped
