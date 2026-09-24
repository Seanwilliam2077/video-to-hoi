"""Test pipeline for the Track 1 glue we own.

Heavy models (MoGe, SAM, FoundationPose, CARI4D) stay behind a backend.
``DemoBackend`` is deterministic and needs no GPU, so the object layer,
the video layer, and the export layout can be checked locally.

One object gets one mesh scale, the median of its clips. One clip locks
one camera and one body identity. Poses stay in the camera frame until export.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from v2hoi.geometry import matrix_to_pose7


@dataclass(frozen=True)
class Clip:
    episode_index: int
    object_name: str
    prompt: str
    camera: str
    n_frames: int


@dataclass
class ClipResult:
    clip: Clip
    intrinsics: dict
    scale: float
    identity: np.ndarray
    obj_pose7: np.ndarray  # camera frame, (T, 7)
    world_T_cam: np.ndarray  # (4, 4)


# Short clips from the plan: black pan ep12, foam block ep16.
DEMO_CLIPS = (
    Clip(12, "black_pan", "a black frying pan.", "front_stereo_camera_left", 8),
    Clip(13, "black_pan", "a black frying pan.", "left_stereo_camera_left", 8),
    Clip(16, "foam_grass_block", "a foam block.", "front_stereo_camera_left", 6),
)


def median_intrinsics(samples: list[dict]) -> dict:
    keys = ("fx", "fy", "cx", "cy", "width", "height")
    return {key: float(np.median([row[key] for row in samples])) for key in keys}


def median_scale(values: list[float]) -> float:
    if not values:
        raise ValueError("scale list is empty")
    return float(np.median(values))


def lock_vector(samples: np.ndarray) -> np.ndarray:
    """(T, D) or (N, D) -> one (D,) median vector."""
    return np.median(np.asarray(samples, dtype=np.float64), axis=0)


def camera_to_world(up: np.ndarray) -> np.ndarray:
    """First-frame camera, with ``up`` mapped to world +Y."""
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    target = np.array([0.0, 1.0, 0.0])
    axis = np.cross(up, target)
    sine = np.linalg.norm(axis)
    cosine = float(np.dot(up, target))
    T = np.eye(4)
    if sine < 1e-8:
        if cosine < 0:
            T[1, 1] = T[2, 2] = -1.0
        return T
    axis = axis / sine
    angle = np.arctan2(sine, cosine)
    k = axis
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    T[:3, :3] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return T


class DemoBackend:
    """Stand-in for UniDepth, SAM 3D and FoundationPose."""

    def frame_intrinsics(self, clip: Clip) -> list[dict]:
        base = 1000.0 + clip.episode_index
        return [
            {"fx": base + i * 0.4, "fy": base + 2, "cx": 768, "cy": 576, "width": 1536, "height": 1152}
            for i in range(clip.n_frames)
        ]

    def frame_scales(self, clip: Clip) -> list[float]:
        return [1.0 + 0.01 * clip.episode_index + 0.001 * i for i in range(clip.n_frames)]

    def body_identity(self, clip: Clip) -> np.ndarray:
        rng = np.random.default_rng(clip.episode_index)
        return rng.normal(size=(clip.n_frames, 45))

    def gravity(self, clip: Clip) -> np.ndarray:
        return np.array([0.0, 0.98, 0.2])

    def object_poses_cam(self, clip: Clip) -> np.ndarray:
        T = np.repeat(np.eye(4)[None], clip.n_frames, axis=0)
        T[:, 0, 3] = 0.01 * np.arange(clip.n_frames)
        T[:, 2, 3] = 1.5
        pose = matrix_to_pose7(T)
        if clip.n_frames > 2:
            pose[1] = 0.0  # one invisible frame
        return pose


def group_by_object(clips: list[Clip]) -> dict[str, list[Clip]]:
    grouped: dict[str, list[Clip]] = {}
    for clip in clips:
        grouped.setdefault(clip.object_name, []).append(clip)
    return grouped


def run(clips: list[Clip], backend: DemoBackend, out_dir: Path) -> list[ClipResult]:
    out_dir = Path(out_dir)
    results: list[ClipResult] = []
    for object_name, group in group_by_object(clips).items():
        per_clip_scale = [median_scale(backend.frame_scales(clip)) for clip in group]
        scale = median_scale(per_clip_scale)
        obj_dir = out_dir / "objects" / object_name
        obj_dir.mkdir(parents=True, exist_ok=True)
        (obj_dir / "scale.json").write_text(
            json.dumps({"object": object_name, "scale": scale, "episodes": [c.episode_index for c in group]}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        for clip in group:
            intrinsics = median_intrinsics(backend.frame_intrinsics(clip))
            identity = lock_vector(backend.body_identity(clip))
            pose = backend.object_poses_cam(clip)
            world = camera_to_world(backend.gravity(clip))
            ep_dir = out_dir / "episodes" / f"{clip.episode_index:06d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            (ep_dir / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2) + "\n", encoding="utf-8")
            np.save(ep_dir / "identity.npy", identity)
            np.save(ep_dir / "object_pose_cam.npy", pose)
            np.save(ep_dir / "world_T_cam.npy", world)
            (ep_dir / "clip.json").write_text(
                json.dumps(
                    {
                        "episode_index": clip.episode_index,
                        "object": object_name,
                        "prompt": clip.prompt,
                        "camera": clip.camera,
                        "scale": scale,
                        "n_frames": clip.n_frames,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            results.append(ClipResult(clip, intrinsics, scale, identity, pose, world))
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the local Track 1 test pipeline.")
    parser.add_argument("--out", type=Path, default=Path("work/demo"))
    args = parser.parse_args()
    run(list(DEMO_CLIPS), DemoBackend(), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
