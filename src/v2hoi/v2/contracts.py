"""Executable structural profile for the proposed v2 records.

This module verifies structure, bindings and declared evidence. It does not run a
decoder, assess reconstruction quality, authenticate a reviewer's statements or
implement the official packer. Passing it never creates a passing quality report.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar
from uuid import UUID

import numpy as np

from . import IMPLEMENTATION_PROFILE, SPEC_REVISION
from .artifacts import ArtifactRef, ArtifactStore, ContractError, canonical_json_bytes, content_hash, strict_json_loads
from .geometry import validate_native_human, validate_object_trajectory


def require(condition, message):
    if not condition:
        raise ContractError(message)


def fields(value, required, where):
    require(isinstance(value, dict), f"{where} must be an object")
    require(set(required) <= value.keys(), f"{where} lacks {sorted(set(required) - value.keys())}")


def text(value, where):
    require(isinstance(value, str) and bool(value.strip()), f"{where} must be a nonempty string")


def integer(value, where, minimum=0):
    require(type(value) is int and minimum <= value <= 2**53 - 1, f"{where} must be an exact integer >= {minimum}")


def enum(value, options, where):
    require(value in options, f"unsupported {where}: {value!r}")


def timestamp(value):
    text(value, "timestamp")
    require(bool(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z", value)), "timestamp must be UTC RFC3339")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("invalid timestamp") from exc


def frame_ids(value, count, where, complete=False):
    require(isinstance(value, list), f"{where} must be a list")
    for frame in value:
        integer(frame, where)
        require(frame < count, f"{where} exceeds source timeline")
    require(value == sorted(set(value)), f"{where} must have sorted unique original frame IDs")
    if complete:
        require(value == list(range(count)), f"{where} must cover all original frames")


def interval(value, count):
    fields(value, ("start_frame_id", "end_frame_id"), "interval")
    a, b = value["start_frame_id"], value["end_frame_id"]
    integer(a, "interval start")
    integer(b, "interval end")
    require(a <= b < count, "interval must use inclusive source frame IDs")
    return set(range(a, b + 1))


def refs_in(value):
    """Visit every declared file reference, including supplemental dependencies."""
    if isinstance(value, dict):
        if {"path", "sha256", "media_type"} <= value.keys():
            yield ArtifactRef.from_dict(value)
        else:
            for child in value.values():
                yield from refs_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from refs_in(child)


def scene_hash(record):
    return content_hash({"spec_revision": record["spec_revision"], "provenance": record["provenance"], "scene": record["payload"]["scene"]})


@dataclass(frozen=True)
class Record:
    """Canonical immutable value; callers receive copies, never mutable internals."""

    _data: bytes
    TYPE: ClassVar[str] = ""

    @classmethod
    def from_dict(cls, value, context):
        snapshot = canonical_json_bytes(value)
        parsed = strict_json_loads(snapshot)
        require(parsed.get("record_type") == cls.TYPE, f"expected {cls.TYPE}")
        context.validate(parsed)
        return cls(snapshot)

    @classmethod
    def load(cls, ref, context):
        return cls.from_dict(context.store.read_json(ref), context)

    def to_dict(self):
        return strict_json_loads(self._data)

    def publish(self, context):
        return context.store.publish_record(self.to_dict(), context.validate)


class EpisodeEvidence(Record):
    TYPE = "EpisodeEvidence"


class SceneState(Record):
    TYPE = "SceneState"


class SolvePlan(Record):
    TYPE = "SolvePlan"


class ValidationContext:
    """One store and one source domain. Synthetic records cannot enter real mode.

    Report trust is an explicit caller input: validator/configuration hash pairs
    must be reviewed externally. Structural checks cannot prove their truth.
    """

    def __init__(self, store: ArtifactStore, *, source_kind="track1", trusted_reports=()):
        enum(source_kind, {"track1", "independent_synthetic"}, "source kind")
        self.store = store
        self.source_kind = source_kind
        self.trusted_reports = frozenset(trusted_reports)
        self._active = set()

    def json(self, ref):
        value = self.store.read_json(ArtifactRef.from_dict(ref))
        # Typed records check their own source policy before traversing payloads.
        if "record_type" not in value:
            self._json_dependencies(value, {ref["sha256"]})
        return value

    def _json_dependencies(self, value, stack):
        for ref in refs_in(value):
            raw = self.store.read_bytes(ref)
            if ref.media_type == "application/json" or ref.media_type.endswith("+json"):
                require(ref.sha256 not in stack, "JSON dependency cycle")
                child = strict_json_loads(raw)
                if "record_type" not in child:
                    self._json_dependencies(child, stack | {ref.sha256})
                else:
                    self.validate(child)

    def arrays(self, ref):
        require(ref.get("media_type") == "application/x-npz", "array ref must declare application/x-npz")
        raw = self.store.read_bytes(ArtifactRef.from_dict(ref))
        try:
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                result = {name: archive[name] for name in archive.files}
        except (ValueError, OSError, EOFError, TypeError) as exc:
            raise ContractError("invalid or unsafe NPZ payload") from exc
        require(all(a.dtype.kind in "biuf" for a in result.values()), "only real numeric arrays are supported")
        return result

    def referenced_record(self, ref, kind):
        record = self.json(ref)
        require(record.get("record_type") == kind, f"expected {kind} reference")
        self.validate(record)
        return record

    def validate(self, record):
        try:
            self._validate(record)
        except ContractError:
            raise
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise ContractError(f"malformed v2 record: {exc}") from exc

    def _validate(self, record):
        # Enforce JCS's exact numbers/strings for callers passing Python objects too.
        canonical_json_bytes(record)
        fields(record, ("record_type", "spec_revision", "implementation_profile", "id", "created_at", "provenance", "payload"), "record")
        enum(record["record_type"], {"EpisodeEvidence", "SceneState", "SolvePlan"}, "record type")
        require(record["spec_revision"] == SPEC_REVISION, "unsupported spec revision")
        require(record["implementation_profile"] == IMPLEMENTATION_PROFILE, "unsupported implementation profile")
        try:
            identity = UUID(record["id"])
        except (ValueError, TypeError, AttributeError) as exc:
            raise ContractError("record id must be UUID v4") from exc
        require(identity.version == 4 and str(identity) == record["id"], "record id must be canonical UUID v4")
        timestamp(record["created_at"])
        require(record["id"] not in self._active, "record dependency cycle")
        self._active.add(record["id"])
        try:
            self._provenance(record["provenance"])
            # Source policy is checked before parsing any scene/data payload.
            for ref in refs_in(record):
                if ref.media_type == "application/json" or ref.media_type.endswith("+json"):
                    dependency = self.json(ref.to_dict())
                    if "record_type" in dependency:
                        self.validate(dependency)
                else:
                    self.store.read_bytes(ref)
            for ref in record["provenance"]["parent_refs"]:
                require("record_type" in self.json(ref), "parent refs must identify v2 records")
            getattr(self, "_" + record["record_type"])(record)
        finally:
            self._active.remove(record["id"])

    def _provenance(self, p):
        fields(p, ("source_kind", "source_refs", "parent_refs", "operation", "code_revision", "configuration_ref", "configuration_hash", "manual_edits", "policy_review_ref", "producer", "dependencies"), "provenance")
        require(p["source_kind"] == self.source_kind, "source domain is prohibited or mismatched")
        require(isinstance(p["source_refs"], list) and p["source_refs"], "source refs are required")
        require(isinstance(p["parent_refs"], list), "parent refs must be a list")
        text(p["operation"], "operation")
        require(p["operation"] != "fake", "fake backends are not valid v2 scene sources")
        text(p["code_revision"], "code revision")
        require(bool(re.fullmatch(r"[0-9a-f]{64}", p["configuration_hash"])), "configuration hash required")
        require(isinstance(p["manual_edits"], list), "manual edit history required")
        fields(p["producer"], ("invocation", "seed", "environment", "nondeterminism"), "producer")
        require(isinstance(p["producer"]["invocation"], list) and p["producer"]["invocation"], "invocation required")
        integer(p["producer"]["seed"], "seed")
        require(isinstance(p["dependencies"], dict) and p["dependencies"], "pinned dependencies required")
        review = self.json(p["policy_review_ref"])
        fields(review, ("decision", "source_kind", "source_hashes", "reviewer", "limitations"), "source policy review")
        require(review["decision"] == "approved" and review["source_kind"] == self.source_kind, "source policy review is not approved")
        text(review["reviewer"], "policy reviewer")
        require(set(review["source_hashes"]) == {ArtifactRef.from_dict(r).sha256 for r in p["source_refs"]}, "source policy review does not cover exact source refs")
        require(content_hash(self.json(p["configuration_ref"])) == p["configuration_hash"], "producer configuration hash mismatch")

    def _EpisodeEvidence(self, record):
        p = record["payload"]
        fields(p, ("episode_id", "video", "timeline", "entities", "observations", "events", "partitions", "evidence_policy_ref"), "evidence")
        integer(p["episode_id"], "episode id")
        require(p["episode_id"] < 30, "episode outside Track 1 inventory")
        require(p["video"] in record["provenance"]["source_refs"], "video must be a reviewed source")
        timeline = self.json(p["timeline"])
        fields(timeline, ("frame_count", "frame_ids", "pts", "time_base", "width", "height", "decode_ref"), "timeline")
        count = timeline["frame_count"]
        for key in ("frame_count", "width", "height"):
            integer(timeline[key], key, 1)
        frame_ids(timeline["frame_ids"], count, "timeline", complete=True)
        require(len(timeline["pts"]) == count, "PTS must cover all frames")
        require(all(type(x) is int for x in timeline["pts"]), "PTS must be exact integers")
        require(all(a < b for a, b in zip(timeline["pts"], timeline["pts"][1:])), "PTS must increase in presentation order")
        require(len(timeline["time_base"]) == 2 and all(type(x) is int and x > 0 for x in timeline["time_base"]), "positive rational time base required")
        for ref in refs_in(timeline):
            self.store.read_bytes(ref)
        entities = {}
        for entity in p["entities"]:
            fields(entity, ("id", "role", "evidence_ref"), "entity")
            text(entity["id"], "entity id")
            require(entity["id"] not in entities, "duplicate entity id")
            enum(entity["role"], {"target_human", "target_object", "context"}, "entity role")
            entities[entity["id"]] = entity["role"]
        require(list(entities.values()).count("target_human") == 1 and "target_object" in entities.values(), "one target human and at least one target object required")
        observations = {}
        for obs in p["observations"]:
            fields(obs, ("observation_id", "entity_id", "frame_ids", "kind", "origin", "data_ref", "producer", "coordinate_frame", "visibility", "validity_ref", "derivation_refs", "derived_from", "lineage_roots", "image_transform"), "observation")
            oid = obs["observation_id"]
            text(oid, "observation id")
            require(oid not in observations, "duplicate observation id")
            require(obs["entity_id"] in entities, "unknown observation entity")
            enum(obs["kind"], {"pixels", "mask", "keypoints", "depth", "flow", "feature_matches", "landmarks", "metadata"}, "observation kind")
            enum(obs["origin"], {"raw", "model_prediction", "human_annotation"}, "observation origin")
            enum(obs["coordinate_frame"], {"source_pixels", "processed_pixels"}, "observation coordinate frame")
            enum(obs["visibility"], {"visible", "partially_occluded", "fully_occluded", "out_of_view", "unknown"}, "visibility")
            frame_ids(obs["frame_ids"], count, "observation frames")
            require(obs["frame_ids"], "observation must identify source frames")
            require(isinstance(obs["lineage_roots"], list) and obs["lineage_roots"], "observation lineage roots required")
            require(isinstance(obs["derived_from"], list), "derived_from must name observation IDs")
            tr = obs["image_transform"]
            fields(tr, ("source_size", "processed_size", "pixel_centres", "T_processed_from_source"), "image transform")
            require(tr["source_size"] == [timeline["width"], timeline["height"]], "source image dimensions mismatch")
            require(len(tr["processed_size"]) == 2 and all(type(x) is int and x > 0 for x in tr["processed_size"]), "invalid processed image dimensions")
            matrix = np.asarray(tr["T_processed_from_source"], dtype=float)
            require(matrix.shape == (3, 3) and np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > 1e-12, "invalid image transform")
            require(np.allclose(matrix[2], [0, 0, 1]), "foundation profile supports affine image transforms only")
            text(tr["pixel_centres"], "pixel centre convention")
            observations[oid] = obs
        lineage_cache = {}
        def lineage(oid, stack=()):
            require(oid in observations, "unknown derived observation")
            require(oid not in stack, "observation lineage cycle")
            if oid not in lineage_cache:
                obs = observations[oid]
                roots = {"declared:" + str(x) for x in obs["lineage_roots"]}
                roots.add("bytes:" + obs["data_ref"]["sha256"])
                roots.update("bytes:" + ref["sha256"] for ref in obs["derivation_refs"])
                for parent in obs["derived_from"]:
                    roots |= lineage(parent, stack + (oid,))
                lineage_cache[oid] = roots
            return lineage_cache[oid]
        for oid in observations:
            lineage(oid)
        partition = self.json(p["partitions"])
        fields(partition, ("fit", "review", "diagnostic", "unused", "scope", "selection_config_ref", "selected_at", "temporal_exclusion_frames", "limitations", "whole_video_model_consumed_review"), "partitions")
        enum(partition["scope"], {"local_repair", "baseline_review"}, "review scope")
        timestamp(partition["selected_at"])
        integer(partition["temporal_exclusion_frames"], "exclusion buffer")
        assigned = []
        for role in ("fit", "review", "diagnostic", "unused"):
            require(isinstance(partition[role], list), "partition IDs must be lists")
            require(all(x in observations for x in partition[role]), "partition contains unknown observation")
            assigned.extend(partition[role])
        require(len(assigned) == len(set(assigned)) and set(assigned) == set(observations), "each observation must have exactly one partition role")
        fit_lineage = set().union(*(lineage(x) for x in partition["fit"]))
        review_lineage = set().union(*(lineage(x) for x in partition["review"]))
        require(not fit_lineage & review_lineage, "fit/review observation lineage overlaps")
        fit_frames = set(f for oid in partition["fit"] for f in observations[oid]["frame_ids"])
        review_frames = set(f for oid in partition["review"] for f in observations[oid]["frame_ids"])
        buffer = partition["temporal_exclusion_frames"]
        require(not any(abs(a - b) <= buffer for a in fit_frames for b in review_frames), "fit/review temporal exclusion violated")
        if partition["whole_video_model_consumed_review"]:
            require(partition["limitations"], "whole-video review leakage must be declared")
        for ref in refs_in(partition):
            self.store.read_bytes(ref)
        require(type(partition["whole_video_model_consumed_review"]) is bool, "whole-video exposure must be explicit boolean")
        for event in p["events"]:
            fields(event, ("id", "type", "entity_ids", "interval", "state", "evidence_refs"), "event")
            text(event["id"], "event id")
            enum(event["type"], {"grasp", "slide", "support", "foot_push", "release", "rolling", "visibility_transition"}, "event type")
            require(all(e in entities for e in event["entity_ids"]), "unknown event entity")
            enum(event["state"], {"candidate", "unresolved"}, "event state")
            require(event["evidence_refs"], "event evidence required")
            interval(event["interval"], count)

    def _SceneState(self, record):
        p = record["payload"]
        fields(p, ("scene", "scene_hash", "status", "validation_reports"), "scene payload")
        require(p["scene_hash"] == scene_hash(record), "scene hash mismatch")
        s = p["scene"]
        fields(s, ("episode_id", "evidence_ref", "timeline_ref", "coordinate_frame", "camera_ref", "humans", "objects", "support_contact_ref", "hypothesis_lineage_ref", "execution_manifest_ref", "preparation_refs", "dependencies"), "scene")
        require(s["coordinate_frame"] == "opencv_camera_m", "scene frame must be OpenCV camera metres")
        evidence = self.referenced_record(s["evidence_ref"], "EpisodeEvidence")
        require(evidence["payload"]["episode_id"] == s["episode_id"], "scene/evidence episode mismatch")
        require(s["timeline_ref"] == evidence["payload"]["timeline"], "scene/evidence timeline mismatch")
        timeline = self.json(s["timeline_ref"])
        expected = timeline["frame_ids"]
        entities = {e["id"]: e["role"] for e in evidence["payload"]["entities"]}
        camera = self.json(s["camera_ref"])
        fields(camera, ("width", "height", "K", "distortion", "evidence_ref"), "camera")
        require([camera["width"], camera["height"]] == [timeline["width"], timeline["height"]], "camera image size mismatch")
        K = np.asarray(camera["K"], dtype=float)
        require(K.shape == (3, 3) and np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0 and np.allclose(K[2], [0, 0, 1]), "invalid camera intrinsics")
        seen = []
        required_outputs = {s["camera_ref"]["sha256"], s["support_contact_ref"]["sha256"]}
        for human in s["humans"]:
            fields(human, ("entity_id", "native_ref", "rig_ref", "decoder_ref", "conversion_ref", "native_units"), "native human")
            require(entities.get(human["entity_id"]) == "target_human", "wrong human binding")
            seen.append(human["entity_id"])
            require(human["native_units"] == "decoder_defined", "native units cannot be inferred from v1")
            binding = self.json(human["conversion_ref"])
            require(binding.get("rig_sha256") == human["rig_ref"]["sha256"] and binding.get("decoder_sha256") == human["decoder_ref"]["sha256"], "rig/decoder conversion binding mismatch")
            a = self.arrays(human["native_ref"])
            fields(a, ("pose", "shape", "scales", "frame_ids"), "native arrays")
            validate_native_human(a["pose"], a["shape"], a["scales"], a["frame_ids"], expected)
            required_outputs.add(human["native_ref"]["sha256"])
        meshes = set()
        for obj in s["objects"]:
            fields(obj, ("entity_id", "mesh_ref", "trajectory_ref", "canonical_transform_ref", "scale_mode", "scale", "source_episodes", "symmetry", "pose_binding", "evidence_status", "uncertainty_ref"), "object")
            require(entities.get(obj["entity_id"]) == "target_object", "wrong object binding")
            seen.append(obj["entity_id"])
            require(obj["source_episodes"] == [s["episode_id"]], "cross-episode pooling disabled")
            binding = obj["pose_binding"]
            require(binding == {"mesh_sha256": obj["mesh_ref"]["sha256"], "canonical_sha256": obj["canonical_transform_ref"]["sha256"], "scale_mode": obj["scale_mode"], "scale": obj["scale"]}, "mesh/canonical/scale pose binding mismatch")
            canonical = self.json(obj["canonical_transform_ref"])
            transform = np.asarray(canonical.get("T_new_from_source"), dtype=float)
            require(transform.shape == (4, 4) and np.isfinite(transform).all() and np.allclose(transform[3], [0, 0, 0, 1]), "invalid canonical transform")
            rotation = transform[:3, :3]
            require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) and abs(np.linalg.det(rotation) - 1) < 1e-6, "canonical transform must be proper rigid geometry")
            mesh, motion = self.arrays(obj["mesh_ref"]), self.arrays(obj["trajectory_ref"])
            fields(mesh, ("vertices", "faces"), "mesh arrays")
            fields(motion, ("rotations", "translations", "frame_ids"), "object motion arrays")
            faces = mesh["faces"]
            require(faces.ndim == 2 and faces.shape[1] == 3 and faces.dtype.kind in "iu" and len(faces) > 0 and faces.min() >= 0 and faces.max() < len(mesh["vertices"]), "invalid triangle indices")
            validate_object_trajectory(mesh["vertices"], motion["rotations"], motion["translations"], motion["frame_ids"], expected, obj["scale_mode"], obj["scale"])
            require(len(obj["evidence_status"]) == len(expected) and all(x in {"observed", "inferred", "unresolved"} for x in obj["evidence_status"]), "object evidence status must cover all frames")
            enum(obj["symmetry"].get("type"), {"none", "finite", "continuous"}, "symmetry type")
            symmetry = obj["symmetry"]
            if symmetry["type"] == "finite":
                rotations = np.asarray(symmetry.get("rotations"), dtype=float)
                require(rotations.ndim == 3 and rotations.shape[1:] == (3, 3) and len(rotations) > 0 and np.isfinite(rotations).all(), "finite symmetry requires explicit rotations")
                require(np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=1e-6, rtol=0) and np.allclose(np.linalg.det(rotations), 1, atol=1e-6, rtol=0), "symmetry rotations must be proper")
                require("evidence_ref" in symmetry, "symmetry evidence required")
            elif symmetry["type"] == "continuous":
                axis, origin = np.asarray(symmetry.get("axis"), dtype=float), np.asarray(symmetry.get("origin"), dtype=float)
                require(axis.shape == origin.shape == (3,) and np.isfinite(axis).all() and np.isfinite(origin).all() and abs(np.linalg.norm(axis) - 1) < 1e-6, "continuous symmetry needs unit axis and origin")
                require("evidence_ref" in symmetry, "symmetry evidence required")
            meshes.add(obj["mesh_ref"]["sha256"])
            required_outputs.add(obj["trajectory_ref"]["sha256"])
        require(len(seen) == len(set(seen)) and set(seen) == {e for e, role in entities.items() if role != "context"}, "scene must contain each target exactly once")
        contact = self.json(s["support_contact_ref"])
        fields(contact, ("supports", "edges"), "support/contact")
        support_ids = set()
        for support in contact["supports"]:
            fields(support, ("id", "geometry_ref", "uncertainty_ref", "evidence_refs"), "support")
            require(support["id"] not in support_ids and support["id"] not in entities, "duplicate support identity")
            support_ids.add(support["id"])
        edge_ids = set()
        for edge in contact["edges"]:
            fields(edge, ("id", "entity_ids", "type", "state", "interval", "evidence_refs", "patches", "normal_freedom", "tangential_freedom", "weight_policy_ref"), "contact edge")
            require(edge["id"] not in edge_ids, "duplicate contact edge")
            edge_ids.add(edge["id"])
            require(len(edge["entity_ids"]) == 2 and len(set(edge["entity_ids"])) == 2 and all(e in entities or e in support_ids for e in edge["entity_ids"]), "unknown contact entity")
            enum(edge["type"], {"grasp", "sliding", "body_support", "foot_push", "ground_support", "rolling", "no_contact"}, "contact type")
            enum(edge["state"], {"candidate", "active", "inactive", "unresolved"}, "contact state")
            interval(edge["interval"], len(expected))
            require(edge["evidence_refs"] and len(edge["patches"]) == 2, "contact evidence and two patches required")
            for freedom in ("normal_freedom", "tangential_freedom"):
                enum(edge[freedom], {"free", "soft", "constrained"}, "contact freedom")
            require(edge["type"] != "sliding" or edge["tangential_freedom"] != "constrained", "sliding cannot impose a no-slip lock")
        for ref in refs_in(contact):
            self.store.read_bytes(ref)
        execution = self.json(s["execution_manifest_ref"])
        fields(execution, ("job_id", "backend", "source_kind", "timeline_ref", "inputs", "object_inputs", "outputs", "state"), "execution manifest")
        enum(execution["backend"], {"native", "independent_synthetic"}, "execution backend")
        require(execution["source_kind"] == self.source_kind, "execution source mismatch")
        require(self.source_kind != "track1" or execution["backend"] == "native", "synthetic invocation in real scene")
        require(execution["timeline_ref"] == s["timeline_ref"], "invocation timeline mismatch")
        require(required_outputs <= {r["sha256"] for r in execution["outputs"]}, "atomic invocation does not bind every coupled output")
        require(meshes <= {r["sha256"] for r in execution["inputs"]}, "metric meshes not recorded as native inputs")
        require(s["evidence_ref"] in execution["inputs"], "invocation does not bind the exact EpisodeEvidence input")
        expected_bindings = [{"entity_id": obj["entity_id"], **obj["pose_binding"]} for obj in s["objects"]]
        require(execution["object_inputs"] == expected_bindings, "invocation object scale/canonical bindings mismatch")
        for nested in (camera, execution):
            for ref in refs_in(nested):
                self.store.read_bytes(ref)
        status = p["status"]
        fields(status, ("execution_valid", "quality_accepted", "export_valid", "selection"), "scene status")
        for flag in ("execution_valid", "quality_accepted", "export_valid"):
            require(type(status[flag]) is bool, "status flags must be booleans")
        require(status["execution_valid"] or not (status["quality_accepted"] or status["export_valid"]), "quality/export require execution validity")
        enum(status["selection"], {"candidate", "accepted", "rejected", "unresolved"}, "selection")
        if status["selection"] == "accepted":
            require(status["execution_valid"] and status["quality_accepted"], "accepted selection requires execution and quality")
        reports = {}
        for ref in p["validation_reports"]:
            report = self.json(ref)
            fields(report, ("kind", "scene_hash", "passed", "validator_ref", "configuration_ref", "checks", "limitations"), "validation report")
            enum(report["kind"], {"execution", "quality", "export"}, "report kind")
            require(report["scene_hash"] == p["scene_hash"], "stale validation report")
            require(report["kind"] not in reports, "duplicate validation report kind")
            require(type(report["passed"]) is bool and isinstance(report["checks"], list) and report["checks"], "report must record checks and explicit decision")
            require(not report["passed"] or all(isinstance(check, dict) and check.get("passed") is True for check in report["checks"]), "passing report contains failed or unrun checks")
            for dependency in refs_in(report):
                self.store.read_bytes(dependency)
            reports[report["kind"]] = report
        for flag, kind in (("execution_valid", "execution"), ("quality_accepted", "quality"), ("export_valid", "export")):
            if status[flag]:
                report = reports.get(kind)
                require(report is not None and report["passed"], f"{flag} requires a passing report")
                require((kind, report["validator_ref"]["sha256"], report["configuration_ref"]["sha256"]) in self.trusted_reports, "untrusted report validator/configuration")
        if status["execution_valid"]:
            require(execution["state"] == "complete", "incomplete native invocation")
        if any(status[flag] for flag in ("execution_valid", "quality_accepted", "export_valid")):
            require("review_lineage" in p, "validity attestation requires an immutable prior scene and review lineage")
        if "review_lineage" in p:
            history = p["review_lineage"]
            fields(history, ("previous_ref", "operation", "report_refs"), "review lineage")
            text(history["operation"], "review operation")
            require(history["report_refs"] == p["validation_reports"], "review lineage must bind the exact current report refs")
            previous = self.referenced_record(history["previous_ref"], "SceneState")
            require(previous["payload"]["scene_hash"] == p["scene_hash"] and previous["provenance"] == record["provenance"], "status-only revision changed scene provenance")
            require(previous["id"] != record["id"], "status revisions require new IDs")

    def _SolvePlan(self, record):
        p = record["payload"]
        fields(p, ("input_scene_ref", "evidence_ref", "operation", "entity_ids", "intervals", "mutable_variables", "fixed_variables", "bounds", "fit_partition_ref", "review_partition_ref", "budgets", "acceptance_config_ref", "acceptance_config_hash", "rollback_scene_ref", "failure_action", "unresolved_action", "operator_ref"), "solve plan")
        enum(p["operation"], {"orientation_branch", "contact_transition", "occlusion_recovery"}, "repair operation")
        require(p["failure_action"] == "retain_parent" and p["unresolved_action"] == "retain_parent_and_record_uncertainty", "unsupported failure action")
        parent = self.referenced_record(p["input_scene_ref"], "SceneState")
        require(parent["payload"]["status"]["execution_valid"], "repair requires execution-valid parent")
        require(p["rollback_scene_ref"] == p["input_scene_ref"], "foundation repairs roll back to their exact parent")
        scene = parent["payload"]["scene"]
        require(p["evidence_ref"] == scene["evidence_ref"], "plan/evidence scene mismatch")
        evidence = self.referenced_record(p["evidence_ref"], "EpisodeEvidence")["payload"]
        count = self.json(evidence["timeline"])["frame_count"]
        require(p["fit_partition_ref"] == evidence["partitions"] == p["review_partition_ref"], "plan must pin the shared fit/review partition manifest")
        partition = self.json(evidence["partitions"])
        require(partition["scope"] == "local_repair" and partition["fit"] and partition["review"], "repair requires fitting and reserved review observations")
        entities = {e["id"]: e["role"] for e in evidence["entities"]}
        require(p["entity_ids"] and all(e in entities and entities[e] != "context" for e in p["entity_ids"]), "unknown repair target")
        require(p["intervals"], "repair intervals required")
        allowed = set().union(*(interval(i, count) for i in p["intervals"]))
        require(p["mutable_variables"], "repair mutable blocks required")
        seen = set()
        for variable in p["mutable_variables"]:
            fields(variable, ("entity_id", "block", "frame_ids"), "mutable block")
            entity, block = variable["entity_id"], variable["block"]
            require(entity in p["entity_ids"], "mutable block outside repair targets")
            frame_ids(variable["frame_ids"], count, "mutable frames")
            require(variable["frame_ids"] and set(variable["frame_ids"]) <= allowed, "mutable frames exceed declared intervals")
            role = entities[entity]
            blocks = {"rotation", "translation"} if role == "target_object" else set()
            if p["operation"] == "contact_transition":
                blocks = {"rotation", "translation", "contact"} if role == "target_object" else {"root", "body", "hand", "contact"}
            require(block in blocks, "operation cannot mutate this variable block")
            if role == "target_human" and block != "contact":
                require("control_map_ref" in variable, "human edits require a decoder-pinned control map")
                controls = self.json(variable["control_map_ref"])
                fields(controls, ("decoder_sha256", "rig_sha256", "block", "indices", "units", "review_ref"), "native control map")
                human = next(h for h in scene["humans"] if h["entity_id"] == entity)
                require(controls["decoder_sha256"] == human["decoder_ref"]["sha256"] and controls["rig_sha256"] == human["rig_ref"]["sha256"] and controls["block"] == block, "native control map binding mismatch")
                require(isinstance(controls["indices"], list) and controls["indices"], "native control indices required")
                frame_ids(controls["indices"], 136, "native control indices")
                text(controls["units"], "native control units")
            for f in variable["frame_ids"]:
                key = (entity, block, f)
                require(key not in seen, "overlapping mutable blocks")
                seen.add(key)
        fixed = p["fixed_variables"]
        required_fixed = {"camera", "mesh", "scale", "human_shape", "human_scales", "all_undeclared"}
        require(isinstance(fixed, list) and required_fixed <= set(fixed), "global and undeclared fixed blocks required")
        valid_fixed = required_fixed | {"human"} | {f"{entity}.{block}" for entity in entities for block in ("rotation", "translation", "root", "body", "hand", "contact")}
        require(set(fixed) <= valid_fixed, "unknown fixed block")
        for entity, block, _ in seen:
            require(f"{entity}.{block}" not in fixed and not (entities[entity] == "target_human" and "human" in fixed), "fixed/mutable blocks overlap")
        if p["operation"] == "orientation_branch":
            require(any(block == "rotation" for _, block, _ in seen), "orientation repair requires a rotation block")
        require(isinstance(p["bounds"], dict) and p["bounds"], "explicit bounds required")
        for value in p["bounds"].values():
            require(type(value) in (int, float) and np.isfinite(value) and value >= 0, "bounds must be finite and nonnegative")
        bound_names = {"rotation": "max_rotation_change_rad", "translation": "max_translation_change_m", "root": "max_native_control_change", "body": "max_native_control_change", "hand": "max_native_control_change", "contact": "max_contact_changes"}
        require({bound_names[block] for _, block, _ in seen} <= p["bounds"].keys(), "each mutable block requires an explicit bound with units")
        budgets = p["budgets"]
        for key in ("max_candidates", "max_iterations", "max_wall_seconds", "max_gpu_memory_bytes"):
            require(key in budgets, f"missing {key}")
            integer(budgets[key], key, 1)
        acceptance = self.json(p["acceptance_config_ref"])
        require(content_hash(acceptance) == p["acceptance_config_hash"], "acceptance configuration hash mismatch")
        fields(acceptance, ("frozen", "metrics", "tolerances", "allowed_regressions", "review_criteria", "full_clip", "boundary_checks", "required_reports", "limitations"), "acceptance policy")
        require(acceptance["frozen"] is True and acceptance["full_clip"] is True and acceptance["boundary_checks"] is True, "repair requires frozen full-clip acceptance policy")

    def select_scene(self, name, ref, *, expected=None, require_export=False):
        """CAS selection of a checked scene; never performs or certifies submission."""
        def validate(record):
            require(record.get("record_type") == "SceneState", "selection requires SceneState")
            self.validate(record)
            status = record["payload"]["status"]
            require(status["selection"] == "accepted" and status["execution_valid"] and status["quality_accepted"], "cannot select an unaccepted baseline")
            require(not require_export or status["export_valid"], "export validity required")
        return self.store.compare_and_swap_selection(name, ref, expected, validate)
