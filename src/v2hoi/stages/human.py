"""Human: the person in each clip, as MHR (for submission) and SOMA-X (for local scoring).

Planned real backend: SAM 3D Body → MHR, then the toolkit's export_soma.py →
SOMA-X; one identity per clip (lock_identity); hands smoothed on their own,
since finger jitter would dominate the joint acceleration (design 3.4).
"""
from __future__ import annotations

import numpy as np

from v2hoi.body import SOMA_TO_OPENCV
from v2hoi.clips import Clip
from v2hoi.contracts import Human, Run


def lock_identity(samples: np.ndarray) -> np.ndarray:
    """One clip has one person: per-frame identity estimates (T, D) → one (D,) vector, the median."""
    return np.median(np.asarray(samples, dtype=np.float64), axis=0)


class FakeHuman:
    """A person in the rest pose, standing still 3 m in front of the camera."""

    DISTANCE_M = 3.0

    def run(self, run: Run, clips: list[Clip]) -> None:
        for clip in clips:
            T = clip.n_frames
            in_front = np.array([0.0, 0.0, self.DISTANCE_M])

            def zeros(*shape):
                return np.zeros((T, *shape), np.float32)

            human = Human(
                pose=zeros(77, 3),
                transl=np.tile(in_front * SOMA_TO_OPENCV, (T, 1)),  # the flip is its own inverse
                identity=zeros(45),
                scale=zeros(68),
                bone_flex=zeros(6),
                mhr_global_rot=zeros(3),
                mhr_body_pose=zeros(133),
                mhr_hand_pose=zeros(108),
                mhr_scale=zeros(28),
                mhr_shape=zeros(45),
                mhr_transl=np.tile(in_front, (T, 1)),
            )
            run.save(human, episode=clip.episode)


BACKENDS = {"fake": FakeHuman}
