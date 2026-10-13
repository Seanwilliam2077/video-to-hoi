"""Independent synthetic storage fixtures; no challenge data or model execution."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import subprocess
import sys
import threading
from uuid import uuid4

import pytest

from v2hoi.v2.artifacts import (
    ArtifactRef, ArtifactStore, ContractError, canonical_json_bytes, content_hash,
    strict_json_loads,
)


def accepted_record(value=0, record_id=None):
    return {"id": record_id or str(uuid4()), "value": value, "accepted": True}


def require_accepted(record):
    if record.get("accepted") is not True:
        raise ContractError("Not accepted")


@pytest.mark.parametrize("payload", [
    b'{"a":1,"a":2}', b'{"nested":{"a":1,"a":2}}',
    b'{"n":NaN}', b'{"n":Infinity}', b'{"n":-Infinity}', b'{"n":1e400}',
    b'{"n":9007199254740992}', b'{"n":-9007199254740992}',
    b'{"x":"\\ud800"}', b'{"\\udfff":1}', b'{"bad":"\xff"}',
    b'[]', b'null', b'{} trailing', b'\xef\xbb\xbf{}',
])
def test_strict_json_rejects_ambiguous_or_noninteroperable_bytes(payload):
    with pytest.raises(ContractError):
        strict_json_loads(payload)


def test_strict_json_preserves_safe_ids_and_unicode_pairs():
    parsed = strict_json_loads(b'{"id":9007199254740991,"emoji":"\\ud83d\\ude00","unknown":null,"empty":[]}')
    assert parsed == {"id": 9007199254740991, "emoji": "\U0001f600", "unknown": None, "empty": []}
    with pytest.raises(ContractError):
        strict_json_loads("{}")


def test_canonical_json_uses_jcs_float_and_utf16_ordering():
    # RFC 8785 section 3.2.2.3 examples, including ECMAScript float rendering.
    value = {"numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27, -0.0]}
    expected = b'{"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27,0]}'
    assert canonical_json_bytes(value) == expected
    assert canonical_json_bytes({"\ue000": 1, "\U0001f600": 2}) == '{"\U0001f600":2,"\ue000":1}'.encode()
    assert content_hash(value) == hashlib.sha256(expected).hexdigest()
    assert content_hash({"b": 2, "a": 1.0}) == content_hash({"a": 1, "b": 2})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2**53, "\ud800", {1: "value"}, {"tuple": (1, 2)}])
def test_canonical_json_rejects_non_json_values(value):
    with pytest.raises(ContractError):
        canonical_json_bytes(value)


@pytest.mark.parametrize("path", [
    "", "/absolute", "//server/share", "C:/drive", "C:relative", "a\\b", "\\\\server\\share",
    "a/../b", "a/./b", "a//b", "a/", "../outside", "https://host/path", "a:stream",
    "a\x00b", "a\nb", "a?b", "a ", "a.", "CON", "a/NUL.txt", "LPT1.json",
])
def test_artifact_ref_rejects_unsafe_paths(path):
    with pytest.raises(ContractError):
        ArtifactRef(path, "a" * 64, "application/json")


def test_artifact_ref_is_strict_and_round_trips():
    ref = ArtifactRef("objects/ab/example.json", "a" * 64, "application/json")
    assert ArtifactRef.from_dict(ref.to_dict()) == ref
    with pytest.raises(ContractError):
        ArtifactRef.from_dict({**ref.to_dict(), "unchecked": True})
    with pytest.raises(ContractError):
        ArtifactRef("x", "A" * 64, "application/json")
    with pytest.raises(ContractError):
        ArtifactRef("x", "a" * 64, "unspecified")


def test_content_addressed_round_trip_and_idempotency(tmp_path):
    store = ArtifactStore(tmp_path)
    ref = store.put_json({"float": 1.0, "name": "test"})
    assert ref == store.put_json({"name": "test", "float": 1})
    assert store.read_json(ref) == {"float": 1, "name": "test"}
    assert store.read_bytes(ref) == b'{"float":1,"name":"test"}'
    assert ref.sha256 in ref.path
    assert not list(tmp_path.rglob(".tmp-*"))


def test_tampered_same_path_is_never_adopted(tmp_path):
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"original", "application/octet-stream", ".bin")
    (tmp_path / ref.path).write_bytes(b"modified")
    with pytest.raises(ContractError, match="hash mismatch"):
        store.read_bytes(ref)
    with pytest.raises(ContractError, match="different bytes"):
        store.put_bytes(b"original", "application/octet-stream", ".bin")
    assert (tmp_path / ref.path).read_bytes() == b"modified"


def test_reference_hash_and_file_type_are_checked_before_json(tmp_path):
    store = ArtifactStore(tmp_path)
    ref = store.put_bytes(b"not JSON", "application/json")
    wrong_hash = ArtifactRef(ref.path, "0" * 64, ref.media_type)
    with pytest.raises(ContractError, match="hash mismatch"):
        store.read_json(wrong_hash)
    with pytest.raises(ContractError, match="Invalid UTF-8 JSON"):
        store.read_json(ref)
    (tmp_path / "directory").mkdir()
    with pytest.raises(ContractError, match="regular file"):
        store.read_bytes(ArtifactRef("directory", "0" * 64, "application/json"))
    with pytest.raises(ContractError, match="Missing artifact"):
        store.read_bytes(ArtifactRef("missing", "0" * 64, "application/json"))


def test_link_or_reparse_escape_is_rejected(tmp_path):
    root, outside = tmp_path / "store", tmp_path / "outside"
    outside.mkdir()
    data = b"outside"
    (outside / "payload.bin").write_bytes(data)
    store = ArtifactStore(root)
    link = root / "escape"
    junction = False
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            pytest.skip("Directory symlinks unavailable")
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True)
        if result.returncode:
            pytest.skip("Windows junction creation unavailable")
        junction = True
    try:
        ref = ArtifactRef("escape/payload.bin", hashlib.sha256(data).hexdigest(), "application/octet-stream")
        with pytest.raises(ContractError, match="link or reparse"):
            store.read_bytes(ref)
    finally:
        if junction:
            link.rmdir()
        else:
            link.unlink()
    assert (outside / "payload.bin").read_bytes() == data


def test_publish_record_requires_validation_and_new_uuid_for_changed_bytes(tmp_path):
    store = ArtifactStore(tmp_path)
    record = accepted_record()
    ref = store.publish_record(record, require_accepted)
    assert store.lookup_record(record["id"]) == ref
    assert store.publish_record(record, require_accepted) == ref
    with pytest.raises(ContractError, match="UUID.*different bytes"):
        store.publish_record({**record, "value": 1}, require_accepted)
    with pytest.raises(ContractError, match="Not accepted"):
        store.publish_record({**accepted_record(), "accepted": False}, require_accepted)
    assert store.read_json(ref) == record
    with pytest.raises(ContractError, match="UUID v4"):
        store.publish_record({**record, "id": "not-a-uuid"}, require_accepted)


def test_callback_cannot_mutate_published_record(tmp_path):
    store = ArtifactStore(tmp_path)
    record = accepted_record()

    def mutate(value):
        value["accepted"] = False

    with pytest.raises(ContractError, match="mutated"):
        store.publish_record(record, mutate)
    assert record["accepted"] is True
    assert store.lookup_record(record["id"]) is None


def test_record_registry_cannot_rebind_uuid_to_other_valid_record(tmp_path):
    store = ArtifactStore(tmp_path)
    first = accepted_record()
    store.publish_record(first, require_accepted)
    second = store.publish_record(accepted_record(), require_accepted)
    (tmp_path / ".records" / f"{first['id']}.json").write_bytes(canonical_json_bytes(second.to_dict()))
    with pytest.raises(ContractError, match="registry identity"):
        store.lookup_record(first["id"])


def test_concurrent_blob_puts_are_idempotent(tmp_path):
    barrier = threading.Barrier(8)

    def put(_):
        barrier.wait()
        return ArtifactStore(tmp_path).put_bytes(b"same immutable bytes", "application/octet-stream")

    with ThreadPoolExecutor(max_workers=8) as executor:
        refs = list(executor.map(put, range(8)))
    assert len(set(refs)) == 1
    assert ArtifactStore(tmp_path).read_bytes(refs[0]) == b"same immutable bytes"
    assert not list(tmp_path.rglob(".tmp-*"))


def test_concurrent_uuid_collision_has_one_winner(tmp_path):
    record_id = str(uuid4())
    barrier = threading.Barrier(8)

    def publish(index):
        store = ArtifactStore(tmp_path)
        barrier.wait()
        try:
            return store.publish_record(accepted_record(index, record_id), require_accepted)
        except ContractError as exc:
            assert "UUID" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(publish, range(8)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert ArtifactStore(tmp_path).lookup_record(record_id) == winners[0]


def test_cross_process_uuid_collision_preserves_published_identity(tmp_path):
    record_id = str(uuid4())
    script = """
import sys
from v2hoi.v2.artifacts import ArtifactStore, ContractError
record = {"id": sys.argv[2], "accepted": True, "value": int(sys.argv[3])}
try:
    ArtifactStore(sys.argv[1]).publish_record(record, lambda record: None)
except ContractError as exc:
    if "UUID" not in str(exc):
        raise
    sys.exit(7)
"""
    children = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path), record_id, str(index)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE) for index in range(4)]
    results = [child.communicate(timeout=30) for child in children]
    codes = [child.returncode for child in children]
    assert codes.count(0) == 1, results
    assert codes.count(7) == 3, results
    store = ArtifactStore(tmp_path)
    assert store.read_json(store.lookup_record(record_id))["id"] == record_id


def test_selection_cas_rejects_stale_expected_and_unregistered_records(tmp_path):
    store = ArtifactStore(tmp_path)
    first = store.publish_record(accepted_record(1), require_accepted)
    second = store.publish_record(accepted_record(2), require_accepted)
    assert store.get_selection("final") is None
    assert store.compare_and_swap_selection("final", first, None, require_accepted) == first
    with pytest.raises(ContractError, match="stale"):
        store.compare_and_swap_selection("final", second, None, require_accepted)
    assert store.get_selection("final") == first
    assert store.compare_and_swap_selection("final", second, first, require_accepted) == second
    unregistered = store.put_json(accepted_record(3))
    with pytest.raises(ContractError, match="registered"):
        store.compare_and_swap_selection("final", unregistered, second, require_accepted)
    assert store.get_selection("final") == second
    with pytest.raises(ContractError, match="Selection name"):
        store.get_selection("../escape")


def test_failed_selection_validation_retains_previous_pointer(tmp_path):
    store = ArtifactStore(tmp_path)
    first = store.publish_record(accepted_record(), require_accepted)
    second = store.publish_record({**accepted_record(), "accepted": False}, lambda record: None)
    store.compare_and_swap_selection("final", first, None, require_accepted)
    with pytest.raises(ContractError, match="Not accepted"):
        store.compare_and_swap_selection("final", second, first, require_accepted)
    assert store.get_selection("final") == first
    (tmp_path / second.path).write_bytes(b"{}")
    with pytest.raises(ContractError, match="hash mismatch"):
        store.compare_and_swap_selection("final", second, first, require_accepted)
    assert store.get_selection("final") == first


def test_concurrent_selection_cas_does_not_silently_retry(tmp_path):
    store = ArtifactStore(tmp_path)
    original = store.publish_record(accepted_record(), require_accepted)
    candidates = [store.publish_record(accepted_record(index), require_accepted) for index in range(8)]
    store.compare_and_swap_selection("final", original, None, require_accepted)
    barrier = threading.Barrier(8)

    def select(ref):
        barrier.wait()
        try:
            return ArtifactStore(tmp_path).compare_and_swap_selection("final", ref, original, require_accepted)
        except ContractError as exc:
            assert "stale" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(select, candidates))
    winners = [ref for ref in results if ref is not None]
    assert len(winners) == 1
    assert store.get_selection("final") == winners[0]
