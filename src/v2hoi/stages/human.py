"""Human (module 2): current fake v1 state and an estimated depth-scale prior.

The target is authoritative native MHR with fixed identity and structural
scales, plus a verified conversion bridge. The mhr_* fields below are old
decomposed placeholders and are not the final submission representation.
No real human reconstruction backend is implemented. See docs/design.md.
"""
from __future__ import annotations

import numpy as np

from v2hoi.body import SOMA_TO_OPENCV
from v2hoi.clips import Clip
from v2hoi.contracts import DepthScale, Human, Run
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
    """Disabled compatibility name; always refuses before accessing data."""

    def run(self, run: Run, clips: list[Clip]) -> None:
        dev_only(clips, "Tier 2 trajectories")


BACKENDS = {"fake": FakeHuman, "tier2": Tier2Human}
