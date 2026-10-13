"""Strict JSON and immutable, content-addressed storage for the opt-in v2 contracts.

This layer verifies bytes and identities, not permitted scene provenance or reconstruction
quality. Callers supply semantic validators before publishing records or selections.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from typing import Any, Callable, Iterator
from uuid import UUID, uuid4

import rfc8785


class ContractError(ValueError):
    """An artifact violates the v2 storage or contract requirements."""


_SAFE_INTEGER = 2**53 - 1
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MEDIA_TYPE = re.compile(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+\Z")
_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_DEVICE = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z", re.I)
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _check_json(value: Any) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > _SAFE_INTEGER:
            raise ContractError("JSON integer exceeds the exactly interoperable safe integer range")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("JSON contains a non-finite number")
    elif isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ContractError("JSON contains a lone surrogate")
    elif isinstance(value, list):
        for item in value:
            _check_json(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("JSON object keys must be strings")
            _check_json(key)
            _check_json(item)
    else:
        raise ContractError(f"Unsupported JSON value type: {type(value).__name__}")


def strict_json_loads(data: bytes) -> dict[str, Any]:
    """Parse a UTF-8 object without duplicate keys, non-finite or unsafe numbers."""
    if not isinstance(data, bytes):
        raise ContractError("JSON input must be bytes")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ContractError(f"Duplicate JSON key: {key!r}")
            result[key] = item
        return result

    def constant(value: str) -> None:
        raise ContractError(f"JSON contains a non-finite number: {value}")

    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
        _check_json(result)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ContractError(f"Invalid UTF-8 JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ContractError("A JSON record must be an object")
    return result


def canonical_json_bytes(value: Any) -> bytes:
    """Return RFC 8785 bytes; sorting keys with json.dumps is not sufficient."""
    try:
        _check_json(value)
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, RecursionError) as exc:
        raise ContractError(f"Cannot canonicalize JSON: {exc}") from exc


def content_hash(value: Any) -> str:
    """SHA-256 of RFC 8785 JSON, for semantic and configuration identities."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _relative_path(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError("Artifact path must be a nonempty relative string")
    if any(ord(c) < 32 or c in '\\:*?"<>|' or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ContractError("Artifact path contains a drive, stream, separator or forbidden character")
    for component in value.split("/"):
        if component in ("", ".", "..") or component[-1:] in (" ", ".") or _DEVICE.fullmatch(component):
            raise ContractError("Artifact path contains an unsafe or reserved component")


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        _relative_path(self.path)
        if not isinstance(self.sha256, str) or not _HASH.fullmatch(self.sha256):
            raise ContractError("Artifact SHA-256 must be 64 lowercase hexadecimal characters")
        if not isinstance(self.media_type, str) or not _MEDIA_TYPE.fullmatch(self.media_type):
            raise ContractError("Artifact media_type must be an explicit bare MIME type")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ArtifactRef:
        if not isinstance(value, dict) or set(value) != {"path", "sha256", "media_type"}:
            raise ContractError("ArtifactRef requires exactly path, sha256 and media_type")
        return cls(**value)

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256, "media_type": self.media_type}


def _is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _comparison_path(path: Path) -> Path:
    # Windows resolve(strict=False) can retain its extended prefix when another
    # writer creates a previously absent directory during resolution. These are
    # the same filesystem paths; compare them with one spelling, without changing
    # the path used for IO. ArtifactRef itself never accepts either absolute form.
    value = str(path)
    if os.name == "nt" and value.startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + value[8:])
    if os.name == "nt" and value.startswith("\\\\?\\") and re.match(r"[A-Za-z]:\\", value[4:]):
        return Path(value[4:])
    return path


def _record_id(value: Any) -> str:
    try:
        parsed = UUID(value) if isinstance(value, str) else None
    except ValueError as exc:
        raise ContractError("Record ID must be a canonical UUID v4") from exc
    if parsed is None or parsed.version != 4 or str(parsed) != value:
        raise ContractError("Record ID must be a canonical UUID v4")
    return value


class ArtifactStore:
    """A local immutable blob store with locked record identities and selection CAS.

    Link/reparse components inside the declared root are rejected conservatively, even
    when their target is contained. Local filesystem atomic hard links, rename and
    advisory locks are required; network filesystems with weaker semantics are unsupported.
    A selection is a mutable pointer; its referenced record and blobs remain immutable.
    """

    def __init__(self, root: str | Path):
        requested = Path(root).expanduser()
        requested.mkdir(parents=True, exist_ok=True)
        self.root = requested.resolve(strict=True)
        if not self.root.is_dir():
            raise ContractError("Artifact-store root must be a directory")
        root_stat = self.root.stat()
        self._root_identity = (root_stat.st_dev, root_stat.st_ino)
        with _LOCKS_GUARD:
            self._thread_lock = _LOCKS.setdefault(os.path.normcase(str(_comparison_path(self.root))), threading.RLock())

    def _path(self, relative: str, *, must_exist: bool = True) -> Path:
        _relative_path(relative)
        try:
            root_info = self.root.lstat()
            if _is_link(root_info) or (root_info.st_dev, root_info.st_ino) != self._root_identity:
                raise ContractError("Artifact-store root identity changed")
            current = self.root
            components = relative.split("/")
            for index, component in enumerate(components):
                current = current / component
                try:
                    info = current.lstat()
                except FileNotFoundError:
                    if must_exist:
                        raise ContractError(f"Missing artifact: {relative}")
                    continue
                if _is_link(info):
                    raise ContractError(f"Artifact path has a link or reparse point: {relative}")
                if index < len(components) - 1 and not stat.S_ISDIR(info.st_mode):
                    raise ContractError(f"Artifact parent is not a directory: {relative}")
            resolved = current.resolve(strict=must_exist)
            if not _comparison_path(resolved).is_relative_to(_comparison_path(self.root)):
                raise ContractError(f"Artifact path escapes its store: {relative}")
            return current
        except OSError as exc:
            raise ContractError(f"Cannot resolve artifact path: {relative}") from exc

    def _directory(self, relative: str) -> Path:
        path = self._path(relative, must_exist=False)
        path.mkdir(parents=True, exist_ok=True)
        path = self._path(relative)
        if not path.is_dir():
            raise ContractError("Artifact storage directory is not a directory")
        return path

    def _read(self, relative: str) -> bytes:
        path = self._path(relative)
        try:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise ContractError(f"Artifact is not a regular file: {relative}")
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            with os.fdopen(os.open(path, flags), "rb") as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                    raise ContractError(f"Artifact identity changed while opening: {relative}")
                self._path(relative)
                result = stream.read()
                after = self._path(relative).lstat()
                if (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
                    raise ContractError(f"Artifact identity changed while reading: {relative}")
                return result
        except OSError as exc:
            raise ContractError(f"Cannot read artifact: {relative}") from exc

    def _write(self, relative: str, data: bytes, *, replace: bool = False) -> None:
        parent = relative.rsplit("/", 1)[0] if "/" in relative else None
        if parent:
            self._directory(parent)
        path = self._path(relative, must_exist=False)
        temporary = path.with_name(f".tmp-{uuid4()}")
        temp_relative = temporary.relative_to(self.root).as_posix()
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if self._read(temp_relative) != data:
                raise ContractError("Temporary artifact failed byte verification")
            self._path(relative, must_exist=False)
            if replace:
                os.replace(temporary, path)
            else:
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    if self._read(relative) != data:
                        raise ContractError(f"Immutable artifact already exists with different bytes: {relative}")
        except OSError as exc:
            raise ContractError(f"Cannot atomically publish artifact: {relative}") from exc
        finally:
            # Only remove the unique temporary name created by this invocation.
            if temporary.exists():
                temporary.unlink()

    def put_bytes(self, data: bytes, media_type: str, suffix: str = "") -> ArtifactRef:
        if not isinstance(data, bytes):
            raise ContractError("Artifact payload must be bytes")
        if not isinstance(suffix, str) or (suffix and not re.fullmatch(r"\.[A-Za-z0-9][A-Za-z0-9._-]{0,30}[A-Za-z0-9]", suffix)):
            # One-character extensions are useful and do not match the longer form.
            if not isinstance(suffix, str) or not re.fullmatch(r"\.[A-Za-z0-9]", suffix):
                raise ContractError("Artifact suffix must be a short filename extension")
        digest = hashlib.sha256(data).hexdigest()
        ref = ArtifactRef(f"objects/{digest[:2]}/{digest}{suffix}", digest, media_type)
        self._write(ref.path, data)
        return ref

    def put_json(self, record: dict[str, Any]) -> ArtifactRef:
        if not isinstance(record, dict):
            raise ContractError("A JSON record must be an object")
        return self.put_bytes(canonical_json_bytes(record), "application/json", ".json")

    def read_bytes(self, ref: ArtifactRef) -> bytes:
        if not isinstance(ref, ArtifactRef):
            raise ContractError("An ArtifactRef is required")
        data = self._read(ref.path)
        if hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ContractError(f"Artifact hash mismatch: {ref.path}")
        return data

    def read_json(self, ref: ArtifactRef) -> dict[str, Any]:
        if not isinstance(ref, ArtifactRef):
            raise ContractError("An ArtifactRef is required")
        if ref.media_type != "application/json" and not ref.media_type.endswith("+json"):
            raise ContractError("JSON artifact requires a JSON media type")
        return strict_json_loads(self.read_bytes(ref))

    @contextmanager
    def _registry_lock(self) -> Iterator[None]:
        with self._thread_lock:
            lock_path = self._path(".registry.lock", must_exist=False)
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(lock_path, flags, 0o600)
            acquired = False
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ContractError("Artifact registry lock is not a regular file")
                # Both OS primitives permit a range beyond EOF. Avoid initializing
                # a byte before locking: another process may already hold that range.
                deadline = time.monotonic() + 10
                while not acquired:
                    try:
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        if os.name == "nt":
                            import msvcrt
                            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                    except OSError as exc:
                        if time.monotonic() >= deadline:
                            raise ContractError("Timed out acquiring artifact registry lock") from exc
                        time.sleep(0.01)
                self._path(".registry.lock")
                yield
            finally:
                if acquired:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _pointer(self, path: str) -> ArtifactRef | None:
        resolved = self._path(path, must_exist=False)
        if not resolved.exists():
            return None
        ref = ArtifactRef.from_dict(strict_json_loads(self._read(path)))
        self.read_bytes(ref)
        return ref

    def lookup_record(self, record_id: str) -> ArtifactRef | None:
        record_id = _record_id(record_id)
        ref = self._pointer(f".records/{record_id}.json")
        if ref is not None and self.read_json(ref).get("id") != record_id:
            raise ContractError("Record registry identity does not match the referenced record")
        return ref

    @staticmethod
    def _validate_snapshot(data: bytes, validate: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        if not callable(validate):
            raise ContractError("A semantic validator callback is required")
        snapshot = strict_json_loads(data)
        if validate(snapshot) is False:
            raise ContractError("Semantic validator rejected the record")
        if canonical_json_bytes(snapshot) != data:
            raise ContractError("Semantic validator mutated its input record")
        return snapshot

    def publish_record(self, record: dict[str, Any], validate: Callable[[dict[str, Any]], Any]) -> ArtifactRef:
        """Publish a validated record once; same-ID identical retries are idempotent."""
        data = canonical_json_bytes(record)
        with self._registry_lock():
            snapshot = self._validate_snapshot(data, validate)
            record_id = _record_id(snapshot.get("id"))
            existing = self.lookup_record(record_id)
            if existing is not None:
                if self.read_bytes(existing) != data:
                    raise ContractError("Record UUID was already published with different bytes")
                return existing
            ref = self.put_bytes(data, "application/json", ".json")
            self._write(f".records/{record_id}.json", canonical_json_bytes(ref.to_dict()))
            return ref

    @staticmethod
    def _selection_name(name: str) -> str:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ContractError("Selection name must contain 1-128 letters, digits, underscores or hyphens")
        return name

    def get_selection(self, name: str) -> ArtifactRef | None:
        return self._pointer(f".selections/{self._selection_name(name)}.json")

    def compare_and_swap_selection(
        self, name: str, ref: ArtifactRef, expected: ArtifactRef | None,
        validate: Callable[[dict[str, Any]], Any],
    ) -> ArtifactRef:
        """Validate an exact published record and atomically replace a named pointer.

        ``expected=None`` means no prior selection. Stale expectations fail without
        retry or promotion. The callback must check scene/report acceptance semantics.
        """
        name = self._selection_name(name)
        with self._registry_lock():
            if self.get_selection(name) != expected:
                raise ContractError("Selection changed; expected reference is stale")
            data = self.read_bytes(ref)
            snapshot = self._validate_snapshot(data, validate)
            if self.lookup_record(_record_id(snapshot.get("id"))) != ref:
                raise ContractError("Selection requires the exact registered immutable record")
            self._write(f".selections/{name}.json", canonical_json_bytes(ref.to_dict()), replace=True)
            return ref
