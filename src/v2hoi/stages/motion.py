"""Motion (module 3, object): the object's pose on every frame.

The target emits a finite pose for every object on every frame, including
occlusions and out-of-view spans. Holding a pose is only an initialization;
evidence from contact, motion and reappearance must constrain the final path.
"""
from __future__ import annotations

import numpy as np

from v2hoi.body import SOMA_TO_OPENCV
from v2hoi.clips import Clip
from v2hoi.contracts import Human, Motion, Run
from v2hoi.stages import dev_only


def hold_missing(T_cam_obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill frames without a pose (NaN) with the nearest earlier pose, or the first
    pose at the start; returns the filled poses and a confidence of 0 on filled frames."""
    ok = np.isfinite(T_cam_obj).all(axis=(1, 2))
    if not ok.any():
        raise ValueError("no frame has an object pose")
    idx = np.where(ok, np.arange(len(ok)), -1)
    idx = np.maximum.accumulate(idx)
    idx[idx < 0] = np.flatnonzero(ok)[0]
    return T_cam_obj[idx], ok.astype(np.float32)


class FakeMotion:
    """The object hangs still, 40 cm to the side of the person's root on the first frame."""

    OFFSET_M = np.array([0.4, 0.0, 0.0])

    def run(self, run: Run, clips: list[Clip]) -> None:
        for clip in clips:
            human = run.load(Human, episode=clip.episode)
            root = human.transl[0] * SOMA_TO_OPENCV  # SOMA convention → camera frame
            T = np.tile(np.eye(4), (clip.n_frames, 1, 1))
            T[:, :3, 3] = root + self.OFFSET_M
            run.save(Motion(T_cam_obj=T, confidence=np.ones(clip.n_frames)), episode=clip.episode)


class Tier2Motion:
    """Disabled compatibility name; always refuses before accessing data."""

    def run(self, run: Run, clips: list[Clip]) -> None:
        dev_only(clips, "Tier 2 trajectories")


BACKENDS = {"fake": FakeMotion, "tier2": Tier2Motion}
