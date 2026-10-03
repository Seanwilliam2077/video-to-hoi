"""Objects: the current fake backend and a disabled legacy backend name.

The target reconstructs canonical geometry and scale from Track 1 observations,
jointly checks posed surfaces against video evidence, and fixes the mesh origin.
Cross-clip sharing requires verified instance identity and permitted evidence;
the fake backend's name-based grouping is not the real reconstruction design.
"""
from __future__ import annotations

import numpy as np

from v2hoi.clips import Clip, group_by_object
from v2hoi.contracts import ContractError, ObjectAsset, Run
from v2hoi.stages import dev_only


def merge_scale(values: list[float]) -> float:
    """An object's clips each estimate its scale; the object gets one, the median."""
    if not values:
        raise ContractError("no scale estimates to merge")
    return float(np.median(values))


class FakeObjects:
    """A 20 cm cube for every object."""

    SIZE_M = 0.2

    def run(self, run: Run, clips: list[Clip]) -> None:
        import trimesh

        for name, group in group_by_object(clips).items():
            asset = ObjectAsset(name=name, scale=1.0, source="fake: 20 cm cube",
                                episodes=[c.episode for c in group])
            path = run.save(asset, name=name)
            trimesh.creation.box(extents=[self.SIZE_M] * 3).export(path.parent / ObjectAsset.MESH)


class ReferenceObjects:
    """Disabled compatibility name; Track 2 meshes have no permitted use."""

    def run(self, run: Run, clips: list[Clip]) -> None:
        dev_only(clips, "reference meshes")


BACKENDS = {"fake": FakeObjects, "reference": ReferenceObjects}
