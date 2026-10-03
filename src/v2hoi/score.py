"""Reference-based Track 1 diagnostics from native MHR and posed object meshes.

No reference, decoder, role indices or model assets are supplied implicitly.
Public Track 1 videos do not include reconstruction ground truth. Supply a
permitted Track 1 reference or independently constructed synthetic fixtures.
Provenance declarations are not proof of origin or official equivalence.

python -m v2hoi.score --pred prediction.json --reference reference.json
    --decoder package.module:function --roles roles.json --out report.json
The local decoder takes NativeMHR and returns DecodedHuman in metres. This
module does not download models or reference data. See docs/evaluation.md.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import trimesh

from v2hoi import metrics as M
from v2hoi.geometry import Similarity, sample_surface

SCORER_VERSION = 2
CM = 100.0
METRICS = ("cd_h_cm", "cd_o_cm", "acc_h_cm_frame2", "acc_o_cm_frame2", "pen_cm")
ALLOWED_PROVENANCE = {"track1", "independent_synthetic"}


def _finite(value, shape, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def _frame_ids(value, name: str, *, consecutive: bool) -> np.ndarray:
    ids = np.asarray(value)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or (ids < 0).any():
        raise ValueError(f"{name} must contain nonnegative integer frame IDs")
    if (ids > np.iinfo(np.int64).max).any():
        raise ValueError(f"{name} exceeds the supported integer frame range")
    differences = np.diff(ids.astype(np.int64))
    if (differences <= 0).any() or (consecutive and (differences != 1).any()):
        raise ValueError(f"{name} must be ordered, unique" + (" and consecutive" if consecutive else ""))
    return ids.astype(np.int64)


def _reject_prohibited(value: str) -> None:
    if re.search(r"track[\s_-]*2|tier[\s_-]*[12]", value, re.IGNORECASE):
        raise ValueError("prohibited challenge-data provenance or path")


@dataclass(frozen=True)
class Provenance:
    kind: str
    evidence: str

    def validate(self) -> None:
        if self.kind not in ALLOWED_PROVENANCE or not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("provenance requires track1 or independent_synthetic and nonempty evidence")
        _reject_prohibited(self.evidence)


@dataclass(frozen=True)
class NativeMHR:
    """Native parameters: their units follow the supplied rig, not SI by default."""

    pose: np.ndarray
    scales: np.ndarray
    shape: np.ndarray

    def validate(self, frames: int) -> None:
        _finite(self.pose, (frames, 136), "native pose")
        _finite(self.scales, (68,), "native static scales")
        _finite(self.shape, (45,), "native static shape")


@dataclass(frozen=True)
class DecodedHuman:
    """Corresponding native-rig vertices and joints, in the scene frame in metres."""

    vertices: np.ndarray
    joints: np.ndarray

    def validate(self, frames: int) -> None:
        for name in ("vertices", "joints"):
            array = np.asarray(getattr(self, name))
            if array.ndim != 3 or array.shape[0] != frames or array.shape[2] != 3 or not array.shape[1]:
                raise ValueError(f"decoded {name} must have shape [T,N,3]")
            if not np.isfinite(array).all():
                raise ValueError(f"decoded {name} contains nonfinite geometry")


@dataclass(frozen=True)
class Roles:
    alignment_vertices: Sequence[int]
    surface_vertices: Sequence[int]
    body_joints22: Sequence[int]
    hand_vertices: Sequence[int]
    source_id: str

    def validate(self, human: DecodedHuman) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("body roles require a declared rig/role source_id")
        _reject_prohibited(self.source_id)
        for name in ("alignment_vertices", "surface_vertices", "body_joints22", "hand_vertices"):
            indices = np.asarray(getattr(self, name))
            limit = human.joints.shape[1] if name == "body_joints22" else human.vertices.shape[1]
            if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                    or len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= limit).any()):
                raise ValueError(f"{name} requires unique in-range integer indices")
            if name == "body_joints22" and len(indices) != 22:
                raise ValueError("body_joints22 requires exactly 22 explicit joint indices")
        if len(self.alignment_vertices) < 3:
            raise ValueError("alignment requires at least three corresponding vertices")


@dataclass(frozen=True)
class Reconstruction:
    """Programmatic diagnostics accept caller-sampled canonical object points.

    The CLI requires a mesh and samples its surface internally. Object scale
    is applied once: scene = scale * R * canonical_point + t. Visibility is
    metadata only and never excludes a pose or a scored frame.
    """

    frame_indices: np.ndarray
    human: NativeMHR
    object_points: np.ndarray
    object_poses: np.ndarray
    object_scale: float
    object_id: str
    provenance: Provenance
    object_visible: np.ndarray | None = None
    point_sampling: str = "caller-supplied diagnostic points; sampling not independently verified"

    def validate(self, expected_frames: np.ndarray) -> None:
        self.provenance.validate()
        frames = _frame_ids(self.frame_indices, "reconstruction frame_indices", consecutive=True)
        if not np.array_equal(frames, expected_frames):
            raise ValueError("reconstruction must cover every expected source frame, including occlusions")
        if not isinstance(self.human, NativeMHR):
            raise ValueError("human input must be NativeMHR, not a legacy body representation")
        self.human.validate(len(frames))
        points = np.asarray(self.object_points)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError("object_points must be a nonempty finite [N,3] array")
        poses = _finite(self.object_poses, (len(frames), 4, 4), "all object poses")
        rotations = poses[:, :3, :3]
        if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6, rtol=0)
                or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-6, rtol=0)
                or not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0)):
            raise ValueError("every object pose must be a proper rigid transform")
        if np.asarray(self.object_scale).shape != () or not np.isfinite(self.object_scale) or self.object_scale <= 0:
            raise ValueError("object_scale must be finite and positive")
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("object_id is required")
        _reject_prohibited(self.object_id)
        if self.object_visible is not None:
            visible = np.asarray(self.object_visible)
            if visible.shape != (len(frames),) or visible.dtype.kind != "b":
                raise ValueError("object_visible must be one boolean per source frame")


Decoder = Callable[[NativeMHR], DecodedHuman]


def _posed(reconstruction: Reconstruction, index: int) -> np.ndarray:
    pose = np.asarray(reconstruction.object_poses[index], dtype=np.float64)
    return reconstruction.object_scale * (np.asarray(reconstruction.object_points) @ pose[:3, :3].T) + pose[:3, 3]


def score_episode(prediction: Reconstruction, reference: Reconstruction, decoder: Decoder, roles: Roles, *,
                  expected_frame_indices: Sequence[int], scored_frame_indices: Sequence[int] | None = None,
                  signed_distance: Callable | None = None) -> dict:
    """Align at the first REFERENCE frame, even if the scoring span starts later.

    signed_distance evaluates canonical submitted-mesh distances, negative
    inside, and accepts (points, workers). No approximate default is selected.
    """
    expected = _frame_ids(expected_frame_indices, "expected_frame_indices", consecutive=True)
    prediction.validate(expected)
    reference.validate(expected)
    if prediction.object_id != reference.object_id:
        raise ValueError("prediction and reference object IDs differ")
    if prediction.provenance.kind != reference.provenance.kind:
        raise ValueError("prediction and reference provenance kinds must agree")
    scored = expected if scored_frame_indices is None else _frame_ids(scored_frame_indices, "scored_frame_indices", consecutive=False)
    if not np.isin(scored, expected).all():
        raise ValueError("scored_frame_indices must map explicitly into the complete reference timeline")
    if len(scored) < 3 or not ((np.diff(scored)[:-1] == 1) & (np.diff(scored)[1:] == 1)).any():
        raise ValueError("ACC requires at least one consecutive three-frame scoring interval")
    selected = np.searchsorted(expected, scored)

    # All declarations and arrays are checked before a decoder runs.
    ph, rh = decoder(prediction.human), decoder(reference.human)
    if not isinstance(ph, DecodedHuman) or not isinstance(rh, DecodedHuman):
        raise ValueError("decoder must return DecodedHuman from native MHR")
    ph.validate(len(expected))
    rh.validate(len(expected))
    if ph.vertices.shape != rh.vertices.shape or ph.joints.shape != rh.joints.shape:
        raise ValueError("native decoder must preserve corresponding vertex and joint topology")
    roles.validate(ph)
    align_ids = np.asarray(roles.alignment_vertices)
    first_pred = np.asarray(ph.vertices[0, align_ids], dtype=np.float64)
    first_ref = np.asarray(rh.vertices[0, align_ids], dtype=np.float64)
    for points in (first_pred, first_ref):
        if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-10) < 2:
            raise ValueError("first-reference-frame human alignment is degenerate")
    align = Similarity.fit(first_pred, first_ref, with_scale=True)
    if not np.isfinite(align.s) or align.s <= 0 or not np.isfinite(align.R).all() or not np.isfinite(align.t).all():
        raise ValueError("first-reference-frame alignment is not a finite positive Sim(3)")
    surface, body22 = np.asarray(roles.surface_vertices), np.asarray(roles.body_joints22)
    p_vertices = align.points(np.asarray(ph.vertices, dtype=np.float64))
    p_joints = align.points(np.asarray(ph.joints, dtype=np.float64))
    p_translation = align.points(np.asarray(prediction.object_poses[:, :3, 3], dtype=np.float64))
    r_translation = np.asarray(reference.object_poses[:, :3, 3], dtype=np.float64)
    values = {
        "cd_h_cm": float(np.mean([M.chamfer_sum(p_vertices[t, surface], rh.vertices[t, surface], workers=1) for t in selected])) * CM,
        "cd_o_cm": float(np.mean([M.chamfer_sum(align.points(_posed(prediction, t)), _posed(reference, t), workers=1) for t in selected])) * CM,
        "acc_h_cm_frame2": M.accel_error(p_joints[selected][:, body22], np.asarray(rh.joints)[selected][:, body22], frame_indices=scored) * CM,
        "acc_o_cm_frame2": M.accel_error(p_translation[selected], r_translation[selected], frame_indices=scored) * CM,
        "pen_cm": None,
    }
    pen_reason = "No submitted-mesh signed-distance function supplied; PEN is unavailable."
    if signed_distance is not None:
        hands, depths = np.asarray(roles.hand_vertices), []
        for t in selected:
            pose = np.asarray(prediction.object_poses[t])
            canonical_hands = (np.asarray(ph.vertices[t, hands]) - pose[:3, 3]) @ pose[:3, :3] / prediction.object_scale
            depths.append(M.hand_penetration_mean(canonical_hands, signed_distance, workers=1))
        values["pen_cm"] = float(np.mean(depths)) * prediction.object_scale * align.s * CM
        pen_reason = None
    if any(value is not None and not np.isfinite(value) for value in values.values()):
        raise ValueError("metric calculation produced a nonfinite value; no partial score is returned")
    return {
        "scorer_version": SCORER_VERSION, "kind": "reference_based_track1_diagnostics", "official_equivalence": False,
        "limitations": ["Decoder, role indices and source provenance are caller-supplied, not independently authenticated.",
                        "Surface samples and mesh compilation are not certified equivalent to the official evaluator."],
        "object_id": prediction.object_id, "provenance_kind": prediction.provenance.kind,
        "provenance_verification": "caller_declared_only", "source_frame_count": len(expected),
        "scored_frame_indices": scored.tolist(),
        "alignment": {"reference_frame_id": int(expected[0]), "source": "human_vertices_only", "shared_by_human_and_object": True,
                      "scale": float(align.s), "rotation": align.R.tolist(), "translation": align.t.tolist()},
        "roles_source": roles.source_id,
        "object_sampling": {"prediction": prediction.point_sampling, "reference": reference.point_sampling,
                            "prediction_count": len(prediction.object_points), "reference_count": len(reference.object_points)},
        "metrics": values, "pen_unavailable_reason": pen_reason,
    }


def score_dataset(predictions: Mapping[str, Reconstruction], references: Mapping[str, Reconstruction],
                  decoder: Decoder, roles: Roles, *, expected_frames: Mapping[str, Sequence[int]],
                  required_objects: Sequence[str], signed_distances: Mapping[str, Callable] | None = None) -> dict:
    """Equal-episode means over the declared complete episode and object set."""
    if not expected_frames or set(predictions) != set(expected_frames) or set(references) != set(expected_frames):
        raise ValueError("prediction and reference must contain every declared episode exactly once")
    required = set(required_objects)
    if not required or len(required) != len(required_objects):
        raise ValueError("required_objects must be a nonempty unique list")
    if {r.object_id for r in predictions.values()} != required or {r.object_id for r in references.values()} != required:
        raise ValueError("prediction and reference must cover every required object")
    if len({r.provenance.kind for r in (*predictions.values(), *references.values())}) != 1:
        raise ValueError("one dataset report cannot mix Track 1 and independent synthetic provenance")
    for key in expected_frames:
        timeline = _frame_ids(expected_frames[key], f"expected_frames[{key}]", consecutive=True)
        predictions[key].validate(timeline)
        references[key].validate(timeline)
    episodes = {key: score_episode(predictions[key], references[key], decoder, roles,
                expected_frame_indices=expected_frames[key], signed_distance=(signed_distances or {}).get(key))
                for key in sorted(expected_frames)}
    mean = {}
    for metric in METRICS:
        values = [episode["metrics"][metric] for episode in episodes.values()]
        mean[metric] = None if any(value is None for value in values) else float(np.mean(values))
    return {"scorer_version": SCORER_VERSION, "kind": "reference_based_track1_diagnostics", "official_equivalence": False,
            "aggregation": "equal_episode_mean_no_composite", "required_objects": sorted(required), "episodes": episodes, "mean": mean,
            "pen_unavailable_reason": "PEN unavailable in at least one episode; no partial mean." if mean["pen_cm"] is None else None}


def _safe_path(path: Path) -> Path:
    _reject_prohibited(str(path))
    resolved = path.resolve()
    _reject_prohibited(str(resolved))
    return resolved


def _read_json(path: Path) -> dict:
    def unique_keys(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    value = json.loads(_safe_path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys,
                       parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    _reject_prohibited(json.dumps(value))
    return value


def _declares_fake(value) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"fake", "smoke", "contains_fake_outputs"} and item is True:
                return True
            if _declares_fake(item):
                return True
    elif isinstance(value, list):
        return any(_declares_fake(item) for item in value)
    elif isinstance(value, str):
        return value.lower() in {"fake", "smoke", "synthetic_smoke", "fake_backend", "interface_fixture"}
    return False


def _manifest(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    data = _read_json(path)
    if data.get("schema_version") != 2:
        raise ValueError("scoring manifest schema_version must be 2")
    Provenance(**data["provenance"]).validate()
    if data["provenance"]["kind"] == "track1" and _declares_fake(data):
        raise ValueError("Track 1 manifest contradicts its fake/smoke declaration")
    if not isinstance(data.get("episodes"), dict) or not data["episodes"] or not isinstance(data.get("expected_frames"), dict):
        raise ValueError("manifest needs episodes and expected_frames objects")
    if set(data["episodes"]) != set(data["expected_frames"]):
        raise ValueError("manifest episodes must exactly match expected_frames")
    required = data.get("required_objects")
    if not isinstance(required, list) or not required or any(not isinstance(x, str) or not x for x in required) or len(set(required)) != len(required):
        raise ValueError("manifest requires a nonempty unique required_objects list")
    expected = {}
    for key, frames in data["expected_frames"].items():
        start, count = frames["start"], frames["count"]
        if (type(start) is not int or type(count) is not int or start < 0 or count < 3
                or start + count - 1 > np.iinfo(np.int64).max):
            raise ValueError("expected_frames requires nonnegative integer start and count >= 3")
        expected[key] = start + np.arange(count, dtype=np.int64)
        entry = data["episodes"][key]
        provenance = Provenance(**entry.get("provenance", data["provenance"]))
        provenance.validate()
        if provenance.kind != data["provenance"]["kind"]:
            raise ValueError("episode provenance must match the manifest provenance kind")
        for field in ("artifact", "mesh"):
            entry[field] = _safe_path(path.parent / entry[field])
            if not entry[field].is_file():
                raise ValueError(f"missing explicit {field} for episode {key}")
        if entry["mesh"].suffix.lower() not in {".glb", ".ply", ".stl"}:
            raise ValueError("mesh must be a self-contained GLB, PLY or STL; sidecar formats are not accepted")
        count = entry.get("sample_count", 2048)
        if type(count) is not int or count <= 0:
            raise ValueError("sample_count must be a positive integer")
    return data, expected


def _load_reconstructions(data: dict) -> tuple[dict[str, Reconstruction], dict]:
    outputs, meshes = {}, {}
    for key, entry in data["episodes"].items():
        mesh_bytes = entry["mesh"].read_bytes()
        mesh_type = entry["mesh"].suffix.lower()[1:]
        if mesh_type == "glb":
            if len(mesh_bytes) < 20:
                raise ValueError("invalid GLB header")
            magic, version, length = struct.unpack("<4sII", mesh_bytes[:12])
            chunk_length, chunk_type = struct.unpack("<II", mesh_bytes[12:20])
            if magic != b"glTF" or version != 2 or length != len(mesh_bytes) or chunk_type != 0x4E4F534A:
                raise ValueError("invalid self-contained GLB")
            gltf = json.loads(mesh_bytes[20:20 + chunk_length].decode("utf-8"))
            for collection in ("buffers", "images"):
                if any("uri" in item and not item["uri"].startswith("data:") for item in gltf.get(collection, [])):
                    raise ValueError("external GLB buffers or textures are prohibited")
        if mesh_type == "ply" and b"TextureFile" in mesh_bytes.split(b"end_header", 1)[0]:
            raise ValueError("external PLY texture declarations are prohibited")
        mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type=mesh_type, force="mesh", process=False,
                            resolver={}, skip_materials=True)
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in "iu"
                or (faces < 0).any() or (faces >= len(vertices)).any() or not np.isfinite(mesh.area) or mesh.area <= 0):
            raise ValueError(f"episode {key} requires a finite nonempty triangle mesh")
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
        count = entry.get("sample_count", 2048)
        points, _ = sample_surface(mesh, count=count, seed=seed)
        digest = hashlib.sha256(mesh_bytes).hexdigest()
        with np.load(entry["artifact"], allow_pickle=False) as arrays:
            scale = arrays["object_scale"]
            if scale.shape != ():
                raise ValueError("native artifact object_scale must be scalar")
            outputs[key] = Reconstruction(arrays["frame_indices"], NativeMHR(arrays["pose"], arrays["scales"], arrays["shape"]),
                points, arrays["object_poses"], float(scale), entry["object_id"],
                Provenance(**entry.get("provenance", data["provenance"])),
                arrays["object_visible"] if "object_visible" in arrays else None,
                f"uniform triangle surface; count={count}; seed={seed}; mesh_sha256={digest}; no official mesh compilation")
        meshes[key] = mesh
    return outputs, meshes


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--pred", type=Path, required=True, help="prediction manifest; no implicit data root")
    parser.add_argument("--reference", type=Path, required=True, help="explicit permitted reference manifest")
    parser.add_argument("--decoder", required=True, help="local module:function; NativeMHR -> DecodedHuman")
    parser.add_argument("--roles", type=Path, required=True, help="native-rig role indices JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        # Preflight both manifests and roles before mesh/NPZ reads or decoder import.
        pred_data, expected = _manifest(args.pred)
        ref_data, ref_expected = _manifest(args.reference)
        if set(expected) != set(ref_expected) or any(not np.array_equal(expected[k], ref_expected[k]) for k in expected):
            raise ValueError("prediction/reference expected frame mappings differ")
        if set(pred_data["required_objects"]) != set(ref_data["required_objects"]):
            raise ValueError("prediction/reference required_objects differ")
        roles = Roles(**_read_json(args.roles))
        output_path = _safe_path(args.out)
        input_paths = [_safe_path(args.pred), _safe_path(args.reference), _safe_path(args.roles)]
        input_paths += [entry[field] for data in (pred_data, ref_data) for entry in data["episodes"].values()
                        for field in ("artifact", "mesh")]
        if any(output_path == source or (output_path.exists() and output_path.samefile(source)) for source in input_paths):
            raise ValueError("output report must not overwrite a manifest, role file, mesh or native artifact")
        _reject_prohibited(args.decoder)
        module_name, separator, function_name = args.decoder.partition(":")
        if not separator or not module_name or not function_name or not function_name.isidentifier():
            raise ValueError("decoder must be module:function")
        predictions, meshes = _load_reconstructions(pred_data)
        references, _ = _load_reconstructions(ref_data)
        for key in expected:
            predictions[key].validate(expected[key])
            references[key].validate(expected[key])
        decoder = getattr(importlib.import_module(module_name), function_name)
        if not callable(decoder):
            raise ValueError("decoder must be callable")
        from v2hoi.geometry import ExactTriangleSDF
        distances = {key: ExactTriangleSDF(mesh) for key, mesh in meshes.items()}
        report = score_dataset(predictions, references, decoder, roles, expected_frames=expected,
                               required_objects=pred_data["required_objects"], signed_distances=distances)
        report["decoder"] = args.decoder
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Diagnostic report: {args.out}; official equivalence is not established.")
    except (ValueError, KeyError, TypeError, OSError, ImportError, AttributeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    # Decoder plugins import these dataclasses from v2hoi.score, not __main__.
    from v2hoi.score import main as canonical_main
    canonical_main()
