"""Motion: the object's pose on every frame, and joint human-object refinement.

Planned real backends: FoundationPose registration and tracking on the object
mask, with occluded frames filled (interpolation or continued tracking, never
missing), static segments locked, and CARI4D's contact refinement. A backend
that also refines the human saves a RefinedHuman; export prefers it.
"""
from __future__ import annotations

import numpy as np

from v2hoi.body import SOMA_TO_OPENCV
from v2hoi.clips import Clip
from v2hoi.contracts import Human, Motion, Run


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


BACKENDS = {"fake": FakeMotion}
