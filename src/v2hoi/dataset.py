"""Read LeRobot v2.1 roots in the Track 2 Tier 1 schema.

Ground truth (Tier 1), noisy labels (Tier 2) and this project's predictions
share one layout, so the scorer reads them all the same way:

    <root>/meta/info.json
    <root>/meta/episodes_metadata.jsonl      episode -> object, mesh, ground plane
    <root>/data/chunk-000/episode_000007.parquet
    <root>/mesh/<object>/<object>.glb

Human columns are SOMA-X (MHR identity) parameters in the SOMA Y-up world.
Object poses are world_T_object in the OpenCV world; invisible frames are
zero-filled and flagged by ``observation.object.visible``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

from v2hoi.geometry import pose7_to_matrix

TIER1_ROOT = Path("data/v2d/track_2/tier_1_multiview_caption")
TIER2_ROOT = Path("data/v2d/track_2/tier_2_synthetic_noise")


@dataclass
class Episode:
    index: int
    sequence_id: str
    object_name: str
    mesh_path: Path
    ground_plane: np.ndarray | None  # (4,) [a, b, c, d], OpenCV world
    pose: np.ndarray  # (T, 77, 3) local rotation vectors
    transl: np.ndarray  # (T, 3) metres, SOMA world
    identity: np.ndarray  # (T, 45)
    scale: np.ndarray  # (T, 68)
    bone_flex: np.ndarray  # (T, 6)
    obj_T: np.ndarray  # (T, 4, 4) world_T_object, NaN where invisible
    obj_visible: np.ndarray  # (T,) bool

    def __len__(self) -> int:
        return len(self.pose)


def read_metadata(root: Path) -> dict[int, dict]:
    path = Path(root) / "meta" / "episodes_metadata.jsonl"
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {row["episode_index"]: row for row in rows}


def list_episodes(root: Path) -> list[int]:
    return sorted(read_metadata(root))


def _child_path(root: Path, relative: str, field: str) -> Path:
    """Resolve a metadata path without allowing it to leave its selected root."""
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} must be a nonempty relative path")
    rel_path = Path(relative)
    if rel_path.is_absolute() or PureWindowsPath(relative).anchor:
        raise ValueError(f"{field} must be relative: {relative!r}")
    base = Path(root).resolve()
    path = (base / rel_path).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"{field} escapes {base}: {relative!r}")
    return path


def _object_name(name: str) -> str:
    """Object identifiers are directory names, not paths."""
    if (
        not isinstance(name, str)
        or name in ("", ".", "..")
        or "/" in name
        or "\\" in name
        or PureWindowsPath(name).anchor
    ):
        raise ValueError(f"invalid object name: {name!r}")
    return name


def _parquet_path(root: Path, index: int) -> Path:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    rel = info["data_path"].format(
        episode_chunk=index // info.get("chunks_size", 1000), episode_index=index
    )
    return _child_path(root, rel, "data_path")


def load_episode(root: Path, index: int, mesh_dir: Path | None = None) -> Episode:
    """Load one episode. ``mesh_dir`` overrides where ``<object>/<object>.glb`` is found."""
    root = Path(root)
    meta = read_metadata(root)[index]
    obj = _object_name(meta["object"])
    df = pd.read_parquet(_parquet_path(root, index))

    def col(name: str) -> np.ndarray:
        return np.stack(df[name].to_numpy()).astype(np.float32)

    if mesh_dir is not None:
        mesh_path = _child_path(mesh_dir, f"{obj}/{obj}.glb", "object mesh")
    else:
        mesh_path = _child_path(root, meta.get("mesh", f"mesh/{obj}/{obj}.glb"), "mesh")

    plane = None
    plane_rel = meta.get("ground_plane")
    if plane_rel:
        plane_path = _child_path(root, plane_rel, "ground_plane")
        if plane_path.is_file():
            plane = np.asarray(json.loads(plane_path.read_text())["plane"], dtype=np.float64)

    visible = df["observation.object.visible"].to_numpy().astype(bool)
    obj_T = pose7_to_matrix(col("observation.object.pose"))
    obj_T[~visible] = np.nan

    return Episode(
        index=index,
        sequence_id=meta.get("sequence_id", str(index)),
        object_name=obj,
        mesh_path=mesh_path,
        ground_plane=plane,
        pose=col("observation.human.pose").reshape(len(df), -1, 3),
        transl=col("observation.human.translation"),
        identity=col("observation.human.identity_coeffs"),
        scale=col("observation.human.scale_params"),
        bone_flex=col("observation.human.bone_length_flexibles"),
        obj_T=obj_T,
        obj_visible=visible,
    )
