"""Controller-owned delivery gates for the internal v1 runner.

These receipts prove covered local format/lineage checks, not reconstruction
quality, authentic provenance, native MHR conversion, or official export.
Native distributed stage names deliberately have no implicit validator here.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from v2hoi.clips import Clip
from v2hoi.contracts import ContractError, Run
from v2hoi.deliverables import inspect_deliverables

PROFILE = "v1-internal-delivery.1"
DEPENDENCIES = {
    "inputs": (),
    "human": ("inputs",),
    "objects": ("inputs", "human"),
    "motion": ("inputs", "human", "objects"),
    "refine": ("inputs", "human", "objects", "motion"),
    "export": ("inputs", "human", "objects", "motion", "refine"),
}
LIMITATIONS = [
    "Internal v1 format readiness only; fake outputs may pass these checks.",
    "Not native distributed execution, reconstruction quality, or official submission validation.",
    "Local controller receipts are not signatures or proof of authentic source provenance.",
]


class GateError(ContractError):
    """A delivery failed; dependent execution must not proceed."""


def _digest(path: Path) -> dict:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
            size += len(block)
    return {"sha256": h.hexdigest(), "bytes": size}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _child(root: Path, relative: str) -> Path:
    from v2hoi.dataset import _child_path
    resolved = _child_path(root, relative, "gate path")
    candidate = root.resolve()
    for part in Path(relative).parts:
        if part in (".", ".."):
            raise GateError("gate paths must use plain relative components")
        candidate = candidate / part
        if candidate.is_symlink() or (candidate.exists() and getattr(candidate.lstat(), "st_file_attributes", 0)
                                     & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise GateError(f"gate paths cannot contain links or reparse points: {candidate}")
    return resolved


def _write_json(path: Path, value: dict, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(value) + b"\n"
    if immutable:
        with path.open("xb") as stream:
            stream.write(data)
        return
    temporary = path.with_name(path.name + "." + str(uuid4()) + ".tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _chain(run: Run):
    seen = set()
    while run is not None:
        root = run.root.resolve()
        if root in seen:
            raise GateError("cyclic upstream run chain")
        if run.dataset != "track1" or "track_2" in {p.lower() for p in root.parts}:
            raise GateError("only Track 1 run roots are permitted")
        seen.add(root)
        yield run
        run = run.upstream


def _effective_file(run: Run, relative: str) -> Path:
    for current in _chain(run):
        path = _child(current.root, relative)
        if path.is_file():
            return path
    raise GateError(f"missing required artifact: {relative}")


def _validator_identity() -> str:
    root = Path(__file__).parent
    # Git checkout line endings must not change validator identity across hosts.
    inventory = {name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                 for name in ("gates.py", "deliverables.py", "contracts.py")}
    return hashlib.sha256(_canonical({"profile": PROFILE, "dependencies": DEPENDENCIES,
                                      "implementation": inventory})).hexdigest()


def _context(clip: Clip) -> dict:
    if clip.dataset != "track1" or "track_2" in {p.lower() for p in clip.video.parts}:
        raise GateError("only Track 1 or independent internal fixture inputs are permitted")
    return {
        "episode": clip.episode, "object": clip.object_name, "frames": clip.n_frames,
        "fps": clip.fps, "width": clip.width, "height": clip.height,
        "camera": clip.camera, "prompt": clip.prompt,
        "video": _digest(clip.video) if clip.video.is_file() else None,
    }


@contextmanager
def run_lock(run: Run):
    """Serialize cooperating writers on one local run, never across machines."""
    run.root.mkdir(parents=True, exist_ok=True)
    path = _child(run.root, ".gate.lock")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise GateError(f"run is locked: {path}; inspect the active/crashed writer before retrying") from exc
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


class DeliveryGate:
    """Execute fresh attempts and revalidate every dependency before reuse.

    The complete clip catalog is needed to verify receipts from larger batches.
    Reports are immutable; current pointers may change, invalidating descendants.
    No v2 scene status or selection is modified by this class.
    """

    def __init__(self, run: Run, clips: list[Clip], *, catalog: list[Clip] | None = None):
        self.run, self.clips = run, list(clips)
        self.catalog = {c.episode: c for c in (catalog if catalog is not None else clips)}
        if not self.clips or len({c.episode for c in self.clips}) != len(self.clips):
            raise GateError("delivery gates require a nonempty, unique episode selection")
        for clip in self.clips:
            if self.catalog.get(clip.episode) != clip:
                raise GateError("selected clips do not match the supplied catalog")
        list(_chain(run))
        self.validator = _validator_identity()

    @staticmethod
    def _stage(stage: str) -> None:
        if stage not in DEPENDENCIES:
            raise GateError(f"no registered delivery validator for stage {stage!r}")

    def _pointer(self, stage: str, episode: int) -> tuple[Run, dict]:
        relative = f".gates/current/{stage}/{episode:06d}.json"
        for owner in _chain(self.run):
            path = _child(owner.root, relative)
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise GateError("gate pointer must be an object")
                if value.get("state") != "passed":
                    raise GateError(f"{stage} episode {episode}: latest attempt is {value.get('state', 'unknown')}")
                identity = value.get("id")
                if not isinstance(identity, str) or str(UUID(identity)) != identity:
                    raise GateError("invalid gate receipt ID")
                return owner, value
        raise GateError(f"{stage} episode {episode}: missing delivery receipt; rerun the producer with gates")

    def _verify(self, stage: str, episodes: list[int], cache: dict, active: set) -> dict:
        self._stage(stage)
        files, refs = {}, {}
        for episode in episodes:
            owner, pointer = self._pointer(stage, episode)
            receipt_path = _child(owner.root, f".gates/receipts/{pointer['id']}.json")
            if _digest(receipt_path)["sha256"] != pointer.get("sha256"):
                raise GateError(f"{stage} episode {episode}: receipt hash mismatch")
            key = (owner.root.resolve(), pointer["id"])
            if key in active:
                raise GateError("cyclic gate dependencies")
            if key not in cache:
                active.add(key)
                value = json.loads(receipt_path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise GateError("gate receipt must be an object")
                if (value.get("profile") != PROFILE or value.get("stage") != stage
                        or value.get("id") != pointer["id"] or value.get("passed") is not True
                        or value.get("validator") != self.validator):
                    raise GateError(f"{stage}: incompatible, failed, or obsolete delivery receipt")
                checks = value.get("checks")
                if not isinstance(checks, list) or not checks or any(
                    not isinstance(c, dict) or c.get("passed") is not True for c in checks
                ):
                    raise GateError(f"{stage}: failed, empty, or unrun receipt checks")
                batch = value.get("episodes")
                if (not isinstance(batch, list) or not batch or len(set(batch)) != len(batch)
                        or any(type(e) is not int or e not in self.catalog for e in batch)):
                    raise GateError(f"{stage}: receipt needs a matching complete metadata catalog")
                clips = [self.catalog[e] for e in batch]
                for member in batch:
                    member_owner, member_pointer = self._pointer(stage, member)
                    if member_owner.root.resolve() != owner.root.resolve() or member_pointer != pointer:
                        raise GateError(f"{stage}: original batch was partially replaced; rerun this batch or use isolated episode runs")
                contexts = {str(c.episode): _context(c) for c in clips}
                if value.get("contexts") != contexts:
                    raise GateError(f"{stage}: stale source video or clip metadata")
                prior_files, prior_refs = {}, {}
                for dependency in DEPENDENCIES[stage]:
                    prior = self._verify(dependency, batch, cache, active)
                    prior_files.update(prior["files"])
                    prior_refs.update(prior["refs"])
                if value.get("inputs") != prior_files or value.get("predecessors") != prior_refs:
                    raise GateError(f"{stage}: stale predecessors; rerun {stage} against the current inputs")
                inspected = inspect_deliverables(owner, stage, clips, local_only=True)
                if not inspected.passed:
                    failed = [c for c in inspected.checks if c.get("passed") is not True]
                    raise GateError(f"{stage}: delivery validation failed: {json.dumps(failed)}")
                actual = {rel: _digest(path) for rel, path in inspected.files.items()}
                if stage == "export":
                    self._export_inventory(owner.root.resolve(), actual, batch)
                if value.get("outputs") != actual:
                    raise GateError(f"{stage}: output changed, missing, or not covered by its receipt")
                for relative, expected in actual.items():
                    effective = _effective_file(self.run, relative)
                    original = _child(owner.root, relative)
                    if effective != original or _digest(effective) != expected:
                        raise GateError(f"{stage}: unverified overlay or stale output: {relative}")
                own_refs = {f"{stage}:{e}": {"id": pointer["id"], "sha256": pointer["sha256"]} for e in batch}
                cache[key] = {"files": {**prior_files, **actual}, "refs": {**prior_refs, **own_refs},
                              "episodes": batch}
                active.remove(key)
            verified = cache[key]
            if episode not in verified["episodes"]:
                raise GateError(f"{stage}: receipt does not cover episode {episode}")
            files.update(verified["files"])
            refs.update(verified["refs"])
        return {"files": files, "refs": refs}

    def _snapshot(self, stage: str, *, include_self: bool = False) -> dict:
        self._stage(stage)
        files, refs, cache = {}, {}, {}
        for dependency in ((stage,) if include_self else DEPENDENCIES[stage]):
            prior = self._verify(dependency, [c.episode for c in self.clips], cache, set())
            files.update(prior["files"])
            refs.update(prior["refs"])
        return {"files": files, "refs": refs,
                "contexts": {str(c.episode): _context(c) for c in self.clips}}

    def check(self, stage: str, *, before: bool = False) -> dict:
        """Read-only local revalidation; a receipt flag is never sufficient."""
        self._stage(stage)
        report = {"profile": PROFILE, "stage": stage, "mode": "before" if before else "after",
                  "passed": False, "limitations": LIMITATIONS, "checks": []}
        try:
            report["snapshot"] = self._snapshot(stage, include_self=not before)
            report["passed"] = True
            report["checks"].append({"check_id": "delivery_and_lineage", "passed": True,
                                     "message": "Required local bytes, structures and current predecessors verified."})
        except Exception as exc:
            report["checks"].append({"check_id": "delivery_and_lineage", "passed": False,
                                     "message": str(exc)})
        return report

    def execute(self, stage: str, backend_name: str, backend) -> Path:
        """The caller holds run_lock; failed attempts never unlock consumers."""
        self._stage(stage)
        identity = str(uuid4())
        pointers = [_child(self.run.root, f".gates/current/{stage}/{c.episode:06d}.json") for c in self.clips]
        report = {
            "profile": PROFILE, "id": identity, "stage": stage, "backend": backend_name,
            "backend_class": f"{type(backend).__module__}.{type(backend).__qualname__}",
            "created_at": datetime.now(timezone.utc).isoformat(), "validator": self.validator,
            "episodes": [c.episode for c in self.clips], "passed": False, "checks": [],
            "contexts": {}, "inputs": {}, "predecessors": {}, "outputs": {},
            "quality_accepted": False, "official_export_valid": False, "limitations": LIMITATIONS,
        }
        for path in pointers:
            _write_json(path, {"state": "running", "id": identity})
        error = None
        try:
            prior = self._snapshot(stage)
            report.update(contexts=prior["contexts"], inputs=prior["files"], predecessors=prior["refs"])
            report["checks"].append({"check_id": "predecessors", "passed": True,
                                     "message": "Current prerequisite deliveries verified before execution."})
            output_root = _child(self.run.root, f".gates/attempts/{identity}/outputs")
            output_root.mkdir(parents=True)
            input_root = _child(self.run.root, f".gates/attempts/{identity}/input-snapshot")
            input_root.mkdir()
            for relative, expected in prior["files"].items():
                destination = _child(input_root, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_effective_file(self.run, relative), destination)
                if _digest(destination) != expected:
                    raise GateError(f"input changed while freezing the attempt: {relative}")
            attempt = Run(output_root, self.run.dataset, Run(input_root, self.run.dataset))
            backend.run(attempt, self.clips)
            inspected = inspect_deliverables(attempt, stage, self.clips, local_only=True)
            report["checks"].extend(inspected.checks)
            if not inspected.passed:
                raise GateError("required deliverables are missing or invalid")
            outputs = {relative: _digest(path) for relative, path in inspected.files.items()}
            if stage == "export":
                self._export_inventory(output_root, outputs, report["episodes"])
            frozen = {rel: _digest(_child(input_root, rel)) for rel in prior["files"]}
            if frozen != prior["files"] or self._snapshot(stage) != prior:
                raise GateError("predecessor bytes, receipts or source inputs changed during execution")
            # Publish only enumerated, verified outputs. Preserve fresh attempt bytes.
            for relative, source in inspected.files.items():
                if source.resolve() != _child(output_root, relative):
                    raise GateError(f"output was not produced in the fresh attempt: {relative}")
            if stage == "export":
                # All directory moves stay inside this explicitly checked run root.
                prepared = _child(self.run.root, f".gates/publication/{identity}/export")
                shutil.copytree(_child(output_root, "export"), prepared)
                destination = _child(self.run.root, "export")
                if destination.exists():
                    backup = _child(self.run.root, f".gates/previous/{identity}/export")
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(destination, backup)
                os.replace(prepared, destination)
            for relative, source in (() if stage == "export" else inspected.files.items()):
                destination = _child(self.run.root, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + "." + identity + ".tmp")
                try:
                    shutil.copyfile(source, temporary)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            published = {rel: _digest(_child(self.run.root, rel)) for rel in outputs}
            if published != outputs or self._snapshot(stage) != prior:
                raise GateError("inputs or outputs changed during publication")
            report["outputs"] = outputs
            report["checks"].append({"check_id": "publication_and_stable_inputs", "passed": True,
                                     "message": "Published bytes match the fresh attempt; inputs remained unchanged."})
            report["passed"] = True
        except Exception as exc:
            error = exc
            report["checks"].append({"check_id": "stage_completion", "passed": False, "message": str(exc)})
        report_path = _child(self.run.root, f".gates/receipts/{identity}.json")
        _write_json(report_path, report, immutable=True)
        pointer = {"state": "passed" if report["passed"] else "failed", "id": identity,
                   "sha256": _digest(report_path)["sha256"]}
        for path in pointers:
            _write_json(path, pointer)
        if error is not None:
            failures = "; ".join(c.get("message", "failed check") for c in report["checks"] if c.get("passed") is not True)
            raise GateError(f"{stage} delivery gate FAILED: {failures}. Report: {report_path}") from error
        return report_path

    @staticmethod
    def _export_inventory(root: Path, outputs: dict, episodes: list[int]) -> None:
        rows = [json.loads(line) for line in _child(root, "export/meta/episodes_metadata.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()]
        if sorted(row["episode_index"] for row in rows) != sorted(episodes):
            raise GateError("export episode set differs from the complete requested batch")
        entries = list(_child(root, "export").rglob("*"))
        for entry in entries:
            _child(root, entry.relative_to(root).as_posix())
        actual = {p.relative_to(root).as_posix() for p in entries if p.is_file()}
        if actual != set(outputs):
            raise GateError("export contains unexpected or unrecorded files")
        for path in actual:
            _child(root, path)
