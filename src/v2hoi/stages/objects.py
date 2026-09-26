"""Objects: one metric mesh per object, shared by all of its clips.

Planned real backend: candidate meshes from SAM 3D Objects and Hunyuan3D-2 on
several frames, chosen by reprojection; each clip estimates a scale on
human-aligned depth and the object takes their median (merge_scale, design
3.2); symmetry recorded for the tracker.
"""
from __future__ import annotations

import shutil

import numpy as np

from v2hoi.clips import Clip, group_by_object
from v2hoi.contracts import ContractError, ObjectAsset, Run
from v2hoi.dataset import TIER1_ROOT, _child_path


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
    """Development only: the Tier 1 ground-truth mesh, so tracking can be worked on
    before generated meshes exist. Refuses Track 1, where Track 2 assets are banned;
    the scorer marks runs that use it as invalid submissions."""

    root = TIER1_ROOT

    def run(self, run: Run, clips: list[Clip]) -> None:
        for name, group in group_by_object(clips).items():
            if any(c.dataset != "tier1" for c in group):
                raise ContractError("reference meshes are Track 2 assets; only Tier 1 dev runs may use them")
            mesh = _child_path(self.root, f"mesh/{name}/{name}.glb", "reference mesh")
            asset = ObjectAsset(name=name, scale=1.0, source=f"reference: {mesh}",
                                episodes=[c.episode for c in group])
            path = run.save(asset, name=name)
            shutil.copyfile(mesh, path.parent / ObjectAsset.MESH)


BACKENDS = {"fake": FakeObjects, "reference": ReferenceObjects}
