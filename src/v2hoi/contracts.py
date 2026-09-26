"""What the pipeline stages hand to each other, and where it lives in a run.

    runs/<run_id>/
      run.json                          dataset, upstream run, history of invocations
      inputs/<episode>/camera.json      Camera                 platform
      inputs/<episode>/masks.npz        Masks                  platform
      human/<episode>/human.npz         Human                  human
      objects/<object>/object.json      ObjectAsset            objects
      objects/<object>/mesh.glb         the object's mesh      objects
      motion/<episode>/motion.npz       Motion                 motion
      motion/<episode>/human.npz        RefinedHuman, optional motion
      export/                           Tier 1 layout for v2hoi.score

Frames: everything is in the clip's camera frame (OpenCV: x right, y down,
z forward, metres). The camera is static, so this is also the submission's
world frame; the official first-frame Sim(3) makes every rigid choice of
world equivalent. Human SOMA-X parameters use the SOMA convention (y up),
which body.SOMA_TO_OPENCV maps onto the camera frame, as in the Tier 1 layout.

Every file records CONTRACT_VERSION. A change to a field's meaning or shape
bumps it, in a PR of its own that the consuming stages' owners approve.
See docs/contracts.md.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import numpy as np

CONTRACT_VERSION = 1


class ContractError(ValueError):
    pass


def _check_version(found, path: Path) -> None:
    if int(found) != CONTRACT_VERSION:
        raise ContractError(f"{path} has contract version {found}, this code reads {CONTRACT_VERSION}")


def _per_frame(*shape: int):
    """A per-frame array field: (n_frames, *shape)."""
    return field(metadata={"shape": shape})


class _Arrays:
    """Per-frame arrays stored together in one .npz."""

    REL: ClassVar[str]

    def validate(self, n_frames: int) -> None:
        for f in fields(self):
            x = np.asarray(getattr(self, f.name))
            want = (n_frames, *f.metadata["shape"])
            if x.shape != want:
                raise ContractError(f"{type(self).__name__}.{f.name} has shape {x.shape}, expected {want}")
            if not np.isfinite(x).all():
                raise ContractError(f"{type(self).__name__}.{f.name} has non-finite values")

    def save(self, path: Path) -> None:
        arrays = {f.name: np.asarray(getattr(self, f.name), dtype=np.float32) for f in fields(self)}
        np.savez_compressed(path, contract=np.int64(CONTRACT_VERSION), **arrays)

    @classmethod
    def load(cls, path: Path):
        with np.load(path) as z:
            _check_version(z["contract"], path)
            missing = [f.name for f in fields(cls) if f.name not in z]
            if missing:
                raise ContractError(f"{path} lacks {missing}")
            return cls(**{f.name: z[f.name] for f in fields(cls)})


@dataclass
class Camera:
    """Pinhole intrinsics of the clip's video, in pixels."""

    REL: ClassVar[str] = "inputs/{episode:06d}/camera.json"

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera: str | None = None  # physical camera, when known

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def validate(self) -> None:
        if min(self.fx, self.fy) <= 0:
            raise ContractError(f"focal length must be positive, got fx={self.fx} fy={self.fy}")
        if not (0 <= self.cx <= self.width and 0 <= self.cy <= self.height):
            raise ContractError(f"principal point ({self.cx}, {self.cy}) lies outside the image")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"contract": CONTRACT_VERSION, **asdict(self)}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Camera":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        _check_version(data.pop("contract"), path)
        return cls(**data)


@dataclass
class Masks:
    """Per-frame human and object masks, bits packed along the width.

    ``human`` and ``obj`` are (n_frames, height, ceil(width / 8)) uint8.
    """

    REL: ClassVar[str] = "inputs/{episode:06d}/masks.npz"

    human: np.ndarray
    obj: np.ndarray
    width: int

    @classmethod
    def pack(cls, human: np.ndarray, obj: np.ndarray) -> "Masks":
        """From (n_frames, height, width) boolean masks."""
        return cls(np.packbits(human, axis=-1), np.packbits(obj, axis=-1), human.shape[-1])

    @classmethod
    def empty(cls, n_frames: int, height: int, width: int) -> "Masks":
        shape = (n_frames, height, (width + 7) // 8)
        return cls(np.zeros(shape, np.uint8), np.zeros(shape, np.uint8), width)

    def frame(self, which: str, t: int) -> np.ndarray:
        """One frame of ``which`` ("human" or "obj") as a (height, width) boolean mask."""
        return np.unpackbits(getattr(self, which)[t], axis=-1)[:, : self.width].astype(bool)

    def validate(self, n_frames: int, height: int, width: int) -> None:
        want = (n_frames, height, (width + 7) // 8)
        for name in ("human", "obj"):
            x = getattr(self, name)
            if x.shape != want or x.dtype != np.uint8:
                raise ContractError(f"Masks.{name} is {x.dtype}{x.shape}, expected uint8{want}")
        if self.width != width:
            raise ContractError(f"Masks.width is {self.width}, expected {width}")

    def save(self, path: Path) -> None:
        np.savez_compressed(path, contract=np.int64(CONTRACT_VERSION), human=self.human, obj=self.obj,
                            width=np.int64(self.width))

    @classmethod
    def load(cls, path: Path) -> "Masks":
        with np.load(path) as z:
            _check_version(z["contract"], path)
            return cls(z["human"], z["obj"], int(z["width"]))


@dataclass
class Human(_Arrays):
    """The person in one clip, camera frame.

    SOMA-X parameters (the Tier 1 columns) feed export and local scoring.
    MHR parameters are what the official submission asks for; their shapes
    follow SAM 3D Body's output (toolkit v2d_sam3d_body) until the official
    format is published.
    """

    REL: ClassVar[str] = "human/{episode:06d}/human.npz"

    pose: np.ndarray = _per_frame(77, 3)  # SOMA-X local rotation vectors
    transl: np.ndarray = _per_frame(3)  # SOMA-X root translation, SOMA convention
    identity: np.ndarray = _per_frame(45)  # MHR identity coefficients
    scale: np.ndarray = _per_frame(68)  # SOMA-X scale parameters
    bone_flex: np.ndarray = _per_frame(6)  # SOMA-X bone length flexibles
    mhr_global_rot: np.ndarray = _per_frame(3)  # Euler ZYX
    mhr_body_pose: np.ndarray = _per_frame(133)
    mhr_hand_pose: np.ndarray = _per_frame(108)
    mhr_scale: np.ndarray = _per_frame(28)
    mhr_shape: np.ndarray = _per_frame(45)
    mhr_transl: np.ndarray = _per_frame(3)  # camera frame, metres


@dataclass
class RefinedHuman(Human):
    """The human after joint human-object refinement; export prefers it over Human."""

    REL: ClassVar[str] = "motion/{episode:06d}/human.npz"


@dataclass
class Motion(_Arrays):
    """The object's pose on every frame, occluded ones included, camera frame."""

    REL: ClassVar[str] = "motion/{episode:06d}/motion.npz"

    T_cam_obj: np.ndarray = _per_frame(4, 4)  # maps object mesh coordinates to the camera frame
    confidence: np.ndarray = _per_frame()  # 0..1, how much the tracker trusts the frame

    def validate(self, n_frames: int) -> None:
        super().validate(n_frames)
        R = self.T_cam_obj[:, :3, :3]
        if not np.allclose(R @ np.swapaxes(R, 1, 2), np.eye(3), atol=1e-4) or not np.allclose(
            np.linalg.det(R), 1.0, atol=1e-4
        ):
            raise ContractError("Motion.T_cam_obj rotations are not proper rotations")
        if not np.allclose(self.T_cam_obj[:, 3], [0.0, 0.0, 0.0, 1.0]):
            raise ContractError("Motion.T_cam_obj last row must be [0, 0, 0, 1]")


@dataclass
class ObjectAsset:
    """One object's mesh, shared by all of its clips. The mesh sits next to this file as mesh.glb,
    in metres, in the object's canonical frame."""

    REL: ClassVar[str] = "objects/{name}/object.json"
    MESH: ClassVar[str] = "mesh.glb"

    name: str
    scale: float  # already applied to the mesh; kept for provenance
    source: str = ""  # generator and candidate that produced the mesh
    symmetry: str | None = None  # e.g. "continuous-z"; free text until a consumer needs more
    episodes: list[int] = field(default_factory=list)

    def validate(self, mesh_path: Path) -> None:
        from v2hoi.geometry import load_mesh
        from v2hoi.score import MESH_EXTENT_M

        if not Path(mesh_path).is_file():
            raise ContractError(f"object {self.name} has no mesh at {mesh_path}")
        extent = float(np.ptp(load_mesh(mesh_path).vertices, axis=0).max())
        lo, hi = MESH_EXTENT_M
        if not lo <= extent <= hi:
            raise ContractError(f"object {self.name} mesh is {extent:.3g} across; expected metres ({lo}-{hi})")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"contract": CONTRACT_VERSION, **asdict(self)}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ObjectAsset":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        _check_version(data.pop("contract"), path)
        return cls(**data)


def _git_rev() -> str | None:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass
class Run:
    """A run directory. Reads fall back to the upstream run, so a stage can be
    developed against a fixed snapshot of the stages before it."""

    root: Path
    dataset: str
    upstream: Run | None = None

    def path(self, cls, **keys) -> Path:
        """Where this run writes an artifact."""
        return self.root / cls.REL.format(**keys)

    def save(self, artifact, **keys) -> Path:
        path = self.path(type(artifact), **keys)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact.save(path)
        return path

    def find(self, cls, **keys) -> Path:
        """This run's copy of an artifact, else the nearest upstream's."""
        run = self
        while run is not None:
            path = run.path(cls, **keys)
            if path.is_file():
                return path
            run = run.upstream
        raise ContractError(
            f"{cls.REL.format(**keys)} not found in run {self.root} or its upstream runs; run that stage first"
        )

    def has(self, cls, **keys) -> bool:
        try:
            self.find(cls, **keys)
            return True
        except ContractError:
            return False

    def load(self, cls, **keys):
        return cls.load(self.find(cls, **keys))

    def mesh(self, name: str) -> Path:
        return self.find(ObjectAsset, name=name).parent / ObjectAsset.MESH

    @classmethod
    def open(cls, root: Path) -> "Run":
        root = Path(root)
        meta_path = root / "run.json"
        if not meta_path.is_file():
            raise ContractError(f"{root} is not a run (no run.json)")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _check_version(meta["contract"], meta_path)
        upstream = cls.open(meta["upstream"]) if meta.get("upstream") else None
        return cls(root, meta["dataset"], upstream)

    @classmethod
    def start(cls, root: Path, dataset: str, upstream: "Run | None" = None, **invocation) -> "Run":
        """Open or create a run and append this invocation to its history."""
        root = Path(root)
        meta_path = root / "run.json"
        if upstream is not None and upstream.dataset != dataset:
            raise ContractError(f"upstream run {upstream.root} is on {upstream.dataset}, not {dataset}")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            _check_version(meta["contract"], meta_path)
            if meta["dataset"] != dataset:
                raise ContractError(f"run {root} is on {meta['dataset']}, not {dataset}")
            if upstream is not None and Path(meta.get("upstream") or "") != upstream.root:
                raise ContractError(f"run {root} already has upstream {meta.get('upstream')}")
            if upstream is None and meta.get("upstream"):
                upstream = cls.open(meta["upstream"])
        else:
            meta = {"contract": CONTRACT_VERSION, "dataset": dataset,
                    "upstream": str(upstream.root) if upstream else None, "history": []}
        meta["history"].append({
            "created": datetime.now().isoformat(timespec="seconds"), "git": _git_rev(), **invocation,
        })
        root.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return cls(root, dataset, upstream)
