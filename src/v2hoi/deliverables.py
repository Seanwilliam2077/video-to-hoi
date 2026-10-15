"""Reload checks for the six internal v1 stages, not native/submission acceptance.

Passing these checks establishes structural delivery readiness only. Fake model
outputs can pass; this is not evidence of reconstruction quality or provenance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from v2hoi.clips import Clip, group_by_object
from v2hoi.contracts import (
    CONTRACT_VERSION, Camera, Depth, DepthScale, Human, Masks, Motion, ObjectAsset,
    RefinedHuman, RefinedMotion, Run,
)
from v2hoi.dataset import _child_path, _object_name
from v2hoi.geometry import MESH_EXTENT_M, load_mesh, matrix_to_pose7, pose7_to_matrix
from v2hoi.stages.export import DATA_PATH

STAGES = ("inputs", "human", "objects", "motion", "refine", "export")
HUMAN_COLUMNS = {
    "pose": "pose", "transl": "translation", "identity": "identity_coeffs",
    "scale": "scale_params", "bone_flex": "bone_length_flexibles",
}


@dataclass
class DeliveryInspection:
    files: dict[str, Path] = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check["passed"] for check in self.checks)


def _specs(stage: str, clips: list[Clip]):
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    for clip in clips:
        _object_name(clip.object_name)
        if type(clip.episode) is not int or clip.episode < 0:
            raise ValueError("episode IDs must be nonnegative integers")
    if stage == "objects":
        for name, group in group_by_object(clips).items():
            rel = ObjectAsset.REL.format(name=name)
            yield rel, ObjectAsset, {"name": name}, group[0]
            yield str(Path(rel).with_name(ObjectAsset.MESH)).replace("\\", "/"), None, {"name": name}, group[0]
    elif stage == "export":
        yield "export/meta/info.json", None, {}, None
        yield "export/meta/episodes_metadata.jsonl", None, {}, None
        for clip in clips:
            e, name = clip.episode, clip.object_name
            yield "export/" + DATA_PATH.format(episode_chunk=e // 1000, episode_index=e), None, {}, clip
            yield f"export/mhr/episode_{e:06d}.npz", None, {}, clip
        for name in group_by_object(clips):
            yield f"export/mesh/{name}/{name}.glb", None, {}, None
    else:
        classes = {"inputs": (Camera, Masks, Depth), "human": (Human, DepthScale),
                   "motion": (Motion,), "refine": (RefinedHuman, RefinedMotion)}[stage]
        for clip in clips:
            for cls in classes:
                yield cls.REL.format(episode=clip.episode), cls, {"episode": clip.episode}, clip


def required_paths(stage: str, clips: list[Clip]) -> list[str]:
    """Expected run-relative filenames, including JSON sidecars and meshes."""
    return list(dict.fromkeys(spec[0] for spec in _specs(stage, clips)))


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _version(value) -> None:
    value = np.asarray(value)
    _require(value.shape == () and np.issubdtype(value.dtype, np.integer)
             and int(value) == CONTRACT_VERSION, f"expected contract version {CONTRACT_VERSION}")


def _mesh(path: Path) -> None:
    mesh = load_mesh(path)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    _require(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices)
             and np.isfinite(vertices).all(), "mesh vertices must be nonempty finite (V, 3)")
    _require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces)
             and np.issubdtype(faces.dtype, np.integer) and (faces >= 0).all()
             and (faces < len(vertices)).all(), "mesh faces must be valid integer triangles")
    triangles = vertices[faces]
    _require(np.any(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1) > 0),
             "mesh must contain nondegenerate triangles")
    lo, hi = MESH_EXTENT_M
    _require(lo <= np.ptp(vertices, axis=0).max() <= hi, "mesh extent is outside metre sanity bounds")


def _identical_files(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            chunk = a.read(1024 * 1024)
            if chunk != b.read(1024 * 1024):
                return False
            if not chunk:
                return True


def _artifact(path: Path, cls, clip: Clip, clips: list[Clip]) -> None:
    if path.suffix == ".json":
        _version(_json(path)["contract"])
    else:
        with np.load(path, allow_pickle=False) as arrays:
            _version(arrays["contract"])
            for key in ("stride",) if cls is Depth else (("width",) if cls is Masks else ()):
                scalar = arrays[key]
                _require(scalar.shape == () and np.issubdtype(scalar.dtype, np.integer),
                         f"{key} must be an integer scalar")
    value = cls.load(path)
    if cls is Camera:
        _require(type(value.width) is int and type(value.height) is int
                 and (value.width, value.height) == (clip.width, clip.height), "camera dimensions differ from clip")
        _require(np.isfinite([value.fx, value.fy, value.cx, value.cy]).all(), "camera intrinsics must be finite")
        value.validate()
    elif cls is Masks:
        value.validate(clip.n_frames, clip.height, clip.width)
    elif cls is Depth:
        _require(value.stride > 0, "depth stride must be positive")
        value.validate(clip.n_frames, clip.height, clip.width)
    elif cls is DepthScale:
        value.validate()
    elif cls is ObjectAsset:
        _require(value.name == clip.object_name, "object name differs from requested object")
        _require(np.isfinite(value.scale) and value.scale > 0, "object scale must be finite and positive")
        expected = {c.episode for c in clips if c.object_name == value.name}
        _require(isinstance(value.episodes, list) and all(type(e) is int and e >= 0 for e in value.episodes)
                 and len(value.episodes) == len(set(value.episodes)) and expected <= set(value.episodes),
                 "object episode list must uniquely contain every requested episode")
    else:
        value.validate(clip.n_frames)
        if isinstance(value, Motion):
            _require(((value.confidence >= 0) & (value.confidence <= 1)).all(), "motion confidence must be in [0, 1]")


def inspect_deliverables(run: Run, stage: str, clips: list[Clip], *, local_only: bool = True) -> DeliveryInspection:
    """Aggregate missing/malformed delivery errors; never substitute upstream by default.

    Unknown stages and invalid caller clip identifiers raise ValueError. File
    failures are checks, including unsafe symlinks; no pickle payload is loaded.
    Export paths always belong to this run, irrespective of ``local_only``.
    """
    specs = list(_specs(stage, clips))
    result = DeliveryInspection()
    if not clips:
        result.checks.append({"check_id": "selection", "artifact": stage, "passed": False,
                              "message": "at least one clip is required"})

    def check(check_id, rel, action, episode=None):
        item = {"check_id": check_id, "artifact": rel, "passed": True, "message": "passed"}
        if episode is not None:
            item["episode_id"] = episode
        try:
            action()
        except Exception as exc:
            item.update(passed=False, message=f"{type(exc).__name__}: {exc}")
        result.checks.append(item)

    for rel, cls, keys, clip in specs:
        result.files[rel] = run.root / rel

        def inspect_one(rel=rel, cls=cls, keys=keys, clip=clip):
            owner = run
            if not local_only and stage != "export":
                hops = run.origin(cls or ObjectAsset, **keys)
                for _ in range(hops):
                    owner = owner.upstream
            path = _child_path(owner.root, rel, "delivery")
            result.files[rel] = path
            _require(path.is_file(), f"missing required file: {rel}")
            if cls is not None:
                _artifact(path, cls, clip, clips)
            elif path.suffix == ".glb":
                _mesh(path)

        check("reload:" + rel, rel, inspect_one, clip.episode if clip else None)
    if stage == "export":
        _inspect_export(run, clips, result, check)
    return result


def _inspect_export(run, clips, result, check) -> None:
    """Only fixed internal-v1 paths are opened; metadata cannot redirect reads."""
    import pandas as pd

    def path(rel):
        return _child_path(run.root, "export/" + rel, "export")

    def metadata():
        info = _json(path("meta/info.json"))
        _version(info["contract"])
        _require(info["source_dataset"] == run.dataset == "track1", "export source dataset must be track1")
        _require(info["data_path"] == DATA_PATH and info["chunks_size"] == 1000,
                 "export uses an unsupported data layout")
        _require(np.isfinite(info["fps"]) and info["fps"] > 0, "export fps must be finite and positive")
        rows = [json.loads(line) for line in path("meta/episodes_metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        ids, frames, offset = set(), 0, 0
        for row in rows:
            e, n, name = row["episode_index"], row["frames"], _object_name(row["object"])
            _require(type(e) is int and e >= 0 and e not in ids, "export episode IDs must be unique nonnegative integers")
            _require(type(n) is int and n > 0, "export frame count must be a positive integer")
            _require(row["mesh"] == f"mesh/{name}/{name}.glb", "export mesh path differs from fixed layout")
            rel_data = DATA_PATH.format(episode_chunk=e // 1000, episode_index=e)
            result.files["export/" + rel_data] = path(rel_data)
            df = pd.read_parquet(path(rel_data))
            _require(len(df) == n, f"episode {e}: metadata frame count differs from actual parquet")
            _require(np.array_equal(df["index"].to_numpy(), np.arange(offset, offset + n)), "global row indices have gaps or duplicates")
            ids.add(e)
            frames += len(df)
            offset += n
        _require(type(info["total_frames"]) is int and info["total_frames"] == frames
                 and type(info["total_episodes"]) is int and info["total_episodes"] == len(rows),
                 "export totals differ from actual episode rows")
        by_id = {row["episode_index"]: row for row in rows}
        for clip in clips:
            row = by_id[clip.episode]
            _require(row["object"] == clip.object_name and row["frames"] == clip.n_frames,
                     f"episode {clip.episode}: export metadata differs from clip")
            _require(info["fps"] == clip.fps, "export fps differs from clip")

    check("export:metadata", "export/meta/info.json", metadata)
    for clip in clips:
        e, n = clip.episode, clip.n_frames
        rel = "export/" + DATA_PATH.format(episode_chunk=e // 1000, episode_index=e)

        def parquet(clip=clip, e=e, n=n, rel=rel):
            df = pd.read_parquet(_child_path(run.root, rel, "parquet"))
            _require(len(df) == n, "parquet does not cover the complete clip")
            _require(np.array_equal(df["frame_index"].to_numpy(), np.arange(n)), "frame indices must be exactly 0..T-1")
            _require(np.array_equal(df["episode_index"].to_numpy(), np.full(n, e)), "parquet episode IDs differ from clip")
            _require(np.allclose(df["timestamp"].to_numpy(), np.arange(n) / clip.fps, rtol=1e-6, atol=1e-7),
                     "timestamps differ from the clip frame timeline")
            source = run.load(RefinedHuman, episode=e) if run.has(RefinedHuman, episode=e) else None
            for f in fields(Human):
                if f.name not in HUMAN_COLUMNS:
                    continue
                x = np.stack(df["observation.human." + HUMAN_COLUMNS[f.name]].to_numpy())
                _require(x.shape == (n, int(np.prod(f.metadata["shape"]))) and np.isfinite(x).all(),
                         f"human column {f.name} has wrong shape or non-finite values")
                if source is not None:
                    expected = np.asarray(getattr(source, f.name), np.float32).reshape(n, -1)
                    _require(np.array_equal(x, expected), f"export {f.name} differs from refined human")
            pose = np.stack(df["observation.object.pose"].to_numpy())
            _require(pose.shape == (n, 7) and np.isfinite(pose).all(), "object poses must be finite (T, 7)")
            _require(np.allclose(np.linalg.norm(pose[:, 3:], axis=1), 1, atol=1e-5), "object quaternions must have unit length")
            Motion(pose7_to_matrix(pose), np.ones(n)).validate(n)
            if run.has(RefinedMotion, episode=e):
                expected = matrix_to_pose7(run.load(RefinedMotion, episode=e).T_cam_obj).astype(np.float32)
                _require(np.array_equal(pose, expected), "export object poses differ from refined motion")
            visible = df["observation.object.visible"].to_numpy()
            _require(visible.shape == (n,) and visible.dtype == np.bool_, "visibility must be a boolean per-frame column")

        check("export:parquet:" + str(e), rel, parquet, e)
        rel_mhr = f"export/mhr/episode_{e:06d}.npz"

        def mhr(e=e, n=n, rel_mhr=rel_mhr):
            source = run.load(RefinedHuman, episode=e) if run.has(RefinedHuman, episode=e) else None
            with np.load(_child_path(run.root, rel_mhr, "mhr"), allow_pickle=False) as arrays:
                for f in fields(Human):
                    if not f.name.startswith("mhr_"):
                        continue
                    x = arrays[f.name]
                    _require(x.shape == (n, *f.metadata["shape"]) and np.isfinite(x).all(),
                             f"{f.name} has wrong shape or non-finite values")
                    if source is not None:
                        _require(np.array_equal(x, getattr(source, f.name)), f"export {f.name} differs from refined human")

        check("export:mhr:" + str(e), rel_mhr, mhr, e)
    for name in group_by_object(clips):
        rel_mesh = f"export/mesh/{name}/{name}.glb"

        def mesh_source(name=name, rel_mesh=rel_mesh):
            if run.has(ObjectAsset, name=name):
                _require(_identical_files(_child_path(run.root, rel_mesh, "mesh"), run.mesh(name)),
                         "export mesh differs from source mesh")

        check("export:mesh-source:" + name, rel_mesh, mesh_source)
