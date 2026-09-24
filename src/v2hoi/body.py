"""SOMA-X forward pass: Tier 1 human parameters -> joints and vertices in the OpenCV world.

Pinned to the same package and asset revision as the official toolkit
(robotic_grounding/scripts/setup_soma_assets.py), with procedural transforms
off, which matches how the toolkit reads the published Track 2 parquet.
"""
from __future__ import annotations

import warnings

import numpy as np

from v2hoi.dataset import Episode

SOMA_ASSET_REVISION = "466879a83d57eabf3d875ded2d869f2075f90348"

# SOMA lives in a Y-up world; object poses and ground planes are in the OpenCV
# (Y-down, Z-forward) world. Checked on Tier 1 episode 7: with this flip the
# wrists stay ~0.15 m from the iron and the feet sit 2-7 cm above the ground
# plane; without it both are metres off.
SOMA_TO_OPENCV = np.array([1.0, -1.0, -1.0], dtype=np.float32)


class SomaBody:
    def __init__(self, device: str | None = None, batch_size: int = 128):
        import torch
        from soma import SOMALayer
        from soma.assets import get_assets_dir

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.layer = SOMALayer(
                data_root=str(get_assets_dir(revision=SOMA_ASSET_REVISION)),
                identity_model_type="mhr",
                device=self.device,
                enable_procedural_transforms=False,
            )
        # rig_data includes a virtual "Root" before the 77 exported joints.
        self.joint_names = [str(n) for n in self.layer.rig_data["joint_names"]][1:]
        wrists = ("LeftHand", "RightHand")
        self.finger_ids = np.array(
            [i for i, n in enumerate(self.joint_names) if n.startswith(wrists) and n not in wrists]
        )
        self.body_ids = np.setdiff1d(np.arange(len(self.joint_names)), self.finger_ids)

    def __call__(self, ep: Episode) -> tuple[np.ndarray, np.ndarray]:
        """Returns joints (T, 77, 3) and vertices (T, V, 3), float32, OpenCV world, metres."""
        torch = self._torch

        def t(x: np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)

        joints, verts = [], []
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for s in range(0, len(ep), self.batch_size):
                sl = slice(s, s + self.batch_size)
                out = self.layer(
                    t(ep.pose[sl]),
                    t(ep.identity[sl]),
                    scale_params=t(ep.scale[sl]),
                    transl=t(ep.transl[sl]),
                    apply_correctives=False,
                    kwargs={"bone_length_flexibles": t(ep.bone_flex[sl])},
                )
                joints.append(out["joints"].cpu().numpy())
                verts.append(out["vertices"].cpu().numpy())
        return np.concatenate(joints) * SOMA_TO_OPENCV, np.concatenate(verts) * SOMA_TO_OPENCV
