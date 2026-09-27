"""Track 1 clips selected from LeRobot metadata, without reference data."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from v2hoi.dataset import _child_path, _object_name, read_metadata

TRACK1_ROOT = Path("data/v2d/track_1")
DATASETS = {"track1": TRACK1_ROOT}


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
    """Track 1 clips, all episodes by default."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(DATASETS)}")
    root = Path(root or DATASETS[dataset]).resolve()
    if "track_2" in {part.lower() for part in root.parts}:
        raise ValueError(f"Track 2 root is prohibited: {root}")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    # Hand-authored synthetic fixtures may omit robot_type; published non-Track 1 roots may not.
    if info.get("robot_type") not in (None, "video_only_object_tracking"):
        raise ValueError(f"{root} is not Track 1 metadata (robot_type={info.get('robot_type')!r})")
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
        name = _object_name(row["object"])
        video_key = row.get("video_key", "observation.images.exo_camera")
        height, width = info["features"][video_key]["shape"][:2]
        rel = info["video_path"].format(
            episode_chunk=e // info.get("chunks_size", 1000), video_key=video_key, episode_index=e
        )
        n_frames = int(row.get("frames") or lengths[e])
        if e in lengths and row.get("frames") is not None and n_frames != lengths[e]:
            raise ValueError(f"episode {e}: metadata frames {n_frames} != episodes.jsonl length {lengths[e]}")
        fps = float(info.get("fps", 30))
        if n_frames <= 0 or not math.isfinite(fps) or fps <= 0 or min(height, width) <= 0:
            raise ValueError(f"episode {e}: invalid frame count, fps, or image size")
        clips.append(Clip(
            dataset=dataset,
            episode=e,
            object_name=name,
            prompt=row.get("object_prompt") or prompts.get(name, name),
            camera=row.get("camera"),
            n_frames=n_frames,
            fps=fps,
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
