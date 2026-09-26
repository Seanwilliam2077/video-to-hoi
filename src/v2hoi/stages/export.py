"""Export (module 1, platform & perception): a run → the Tier 1 layout that v2hoi.score reads.

    export/meta/info.json
    export/meta/episodes_metadata.jsonl
    export/data/chunk-000/episode_XXXXXX.parquet   Tier 1 columns
    export/mesh/<object>/<object>.glb
    export/mhr/episode_XXXXXX.npz                   MHR parameters, for the official format

It reads the refine stage's output. The world frame is the camera frame (see
v2hoi.contracts), so poses are copied, not transformed. Once
eval_reconstruction.py is published, a second backend writes the official
artifact from the same run.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import fields

import numpy as np
import pandas as pd

from v2hoi.clips import Clip
from v2hoi.contracts import (
    CONTRACT_VERSION, ContractError, Human, Motion, ObjectAsset, RefinedHuman, RefinedMotion, Run,
)
from v2hoi.geometry import matrix_to_pose7

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"


def _rows(x: np.ndarray) -> list[np.ndarray]:
    return list(np.asarray(x, dtype=np.float32).reshape(len(x), -1))


def load_refined(run: Run, refined, raw, **keys):
    """The refine stage's output, unless its input was re-run more recently.

    Without this check, a run that re-runs motion but not refine would export
    the upstream run's refined trajectory and silently ignore the new tracking.
    """
    if run.origin(raw, **keys) < run.origin(refined, **keys):
        raise ContractError(
            f"{raw.REL.format(**keys)} is newer than {refined.REL.format(**keys)}; "
            "run the refine stage again (add refine to --stages)"
        )
    return run.load(refined, **keys)


class Tier1Export:
    def run(self, run: Run, clips: list[Clip]) -> None:
        out = run.root / "export"
        if out.exists():
            shutil.rmtree(out)
        (out / "meta").mkdir(parents=True)
        (out / "mhr").mkdir()

        rows, index = [], 0
        for clip in clips:
            e, name, T = clip.episode, clip.object_name, clip.n_frames
            human = load_refined(run, RefinedHuman, Human, episode=e)
            human.validate(T)
            motion = load_refined(run, RefinedMotion, Motion, episode=e)
            motion.validate(T)
            asset = run.load(ObjectAsset, name=name)
            mesh = run.mesh(name)
            asset.validate(mesh)

            mesh_rel = f"mesh/{name}/{name}.glb"
            if not (out / mesh_rel).is_file():
                (out / mesh_rel).parent.mkdir(parents=True)
                shutil.copyfile(mesh, out / mesh_rel)

            frames = np.arange(T)
            data = out / DATA_PATH.format(episode_chunk=e // 1000, episode_index=e)
            data.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "observation.human.pose": _rows(human.pose),
                "observation.human.translation": _rows(human.transl),
                "observation.human.identity_coeffs": _rows(human.identity),
                "observation.human.scale_params": _rows(human.scale),
                "observation.human.bone_length_flexibles": _rows(human.bone_flex),
                "observation.object.pose": _rows(matrix_to_pose7(motion.T_cam_obj)),
                # Every frame has a pose; the scorer ignores this flag for predictions.
                "observation.object.visible": np.ones(T, dtype=bool),
                "timestamp": (frames / clip.fps).astype(np.float32),
                "frame_index": frames,
                "episode_index": np.full(T, e),
                "index": index + frames,
            }).to_parquet(data)
            index += T

            mhr = {f.name: getattr(human, f.name) for f in fields(human) if f.name.startswith("mhr_")}
            np.savez_compressed(out / "mhr" / f"episode_{e:06d}.npz", **mhr)
            rows.append({"episode_index": e, "object": name, "mesh": mesh_rel, "frames": T})

        info = {
            "codebase_version": "v2.1",
            "fps": clips[0].fps if clips else 30,
            "chunks_size": 1000,
            "data_path": DATA_PATH,
            "total_episodes": len(clips),
            "total_frames": index,
            "source_dataset": clips[0].dataset if clips else None,
            "contract": CONTRACT_VERSION,
            "world_frame": "camera (OpenCV); human SOMA-X parameters in the SOMA convention",
        }
        (out / "meta" / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        (out / "meta" / "episodes_metadata.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )


BACKENDS = {"tier1": Tier1Export}
