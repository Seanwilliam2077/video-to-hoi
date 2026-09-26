"""Human (module 2): the person in each clip, as MHR (for submission) and SOMA-X
(for local scoring), and the depth scale that makes the human the metric anchor.

Planned real backend: SAM 3D Body → MHR, then the toolkit's export_soma.py →
SOMA-X; one identity per clip (lock_identity); DepthScale from aligning the
clip's Depth to the human (CARI4D step 3, design 3.2). The first frame
matters most, because the official Sim(3) is fitted on it.
"""
from __future__ import annotations

import numpy as np

from v2hoi.body import SOMA_TO_OPENCV
from v2hoi.clips import Clip
from v2hoi.contracts import DepthScale, Human, Run
from v2hoi.dataset import TIER2_ROOT, load_episode
from v2hoi.stages import dev_only

MHR_SHAPES = {
    "mhr_global_rot": (3,), "mhr_body_pose": (133,), "mhr_hand_pose": (108,), "mhr_scale": (28,),
    "mhr_shape": (45,), "mhr_transl": (3,),
}


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
            mhr = {name: np.zeros((T, *shape), np.float32) for name, shape in MHR_SHAPES.items()}
            mhr["mhr_transl"] = np.tile(in_front, (T, 1))
            human = Human(
                pose=np.zeros((T, 77, 3), np.float32),
                transl=np.tile(in_front * SOMA_TO_OPENCV, (T, 1)),  # the flip is its own inverse
                identity=np.zeros((T, 45), np.float32),
                scale=np.zeros((T, 68), np.float32),
                bone_flex=np.zeros((T, 6), np.float32),
                **mhr,
            )
            run.save(human, episode=clip.episode)
            run.save(DepthScale(1.0, "fake"), episode=clip.episode)


class Tier2Human:
    """Development only: the Tier 2 human, i.e. Tier 1 with the organizer's
    Track 1-like noise, as realistic input for the refine stage.

    SOMA-X only; the MHR fields are zeros. The frame is the Tier 1 world
    rather than a camera, which the scorer's alignment does not mind.
    """

    root = TIER2_ROOT

    def run(self, run: Run, clips: list[Clip]) -> None:
        dev_only(clips, "Tier 2 trajectories")
        for clip in clips:
            ep = load_episode(self.root, clip.episode, mask_hidden=False)
            T = len(ep)
            human = Human(
                pose=ep.pose, transl=ep.transl, identity=ep.identity, scale=ep.scale, bone_flex=ep.bone_flex,
                **{name: np.zeros((T, *shape), np.float32) for name, shape in MHR_SHAPES.items()},
            )
            run.save(human, episode=clip.episode)
            run.save(DepthScale(1.0, "tier2: no depth"), episode=clip.episode)


BACKENDS = {"fake": FakeHuman, "tier2": Tier2Human}
