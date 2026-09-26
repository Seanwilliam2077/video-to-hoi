"""Inputs (module 1, platform & perception): camera intrinsics, human and object
masks, and depth for each clip.

Planned real backends: intrinsics from MoGe or GeoCalib, merged over all clips
of one physical camera with merge_intrinsics; masks from GroundingDINO
detection and SAM2 propagation, prompted with the clip's object prompt; depth
from MoGe2.
"""
from __future__ import annotations

import math

import numpy as np

from v2hoi.clips import Clip
from v2hoi.contracts import Camera, ContractError, Depth, Masks, Run


def merge_intrinsics(samples: list[Camera]) -> Camera:
    """One physical camera, one set of intrinsics: the median of its estimates."""
    if not samples:
        raise ContractError("no intrinsic estimates to merge")
    sizes = {(c.width, c.height) for c in samples}
    if len(sizes) != 1:
        raise ContractError(f"intrinsic estimates disagree on the image size: {sorted(sizes)}")

    def median(key: str) -> float:
        return float(np.median([getattr(c, key) for c in samples]))

    first = samples[0]
    return Camera(first.width, first.height, median("fx"), median("fy"), median("cx"), median("cy"), first.camera)


class FakeInputs:
    """A 60 degree horizontal field of view, the principal point at the centre,
    empty masks, and a flat wall 3 m away."""

    DEPTH_STRIDE = 8

    def run(self, run: Run, clips: list[Clip]) -> None:
        for clip in clips:
            f = 0.5 * clip.width / math.tan(math.radians(30))
            camera = Camera(clip.width, clip.height, f, f, clip.width / 2, clip.height / 2, clip.camera)
            run.save(camera, episode=clip.episode)
            run.save(Masks.empty(clip.n_frames, clip.height, clip.width), episode=clip.episode)
            depth = Depth.constant(clip.n_frames, clip.height, clip.width, self.DEPTH_STRIDE, 3.0)
            run.save(depth, episode=clip.episode)


BACKENDS = {"fake": FakeInputs}
