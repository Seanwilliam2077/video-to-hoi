"""Offline metadata/declaration tests; no challenge pixels, models or references."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("progress_board", ROOT / "tools/progress_board.py")
board = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(board)


@pytest.fixture
def repository(tmp_path, monkeypatch):
    """Independent synthetic metadata in the same 30-slot coordination format."""
    directory = tmp_path / "docs/track1-videos"
    directory.mkdir(parents=True)
    videos = [{"ep": ep, "object": "synthetic_cube", "objectName": "Synthetic cube",
               "camera": "synthetic_camera", "frames": 10, "seconds": 1.0,
               "task": "Independent synthetic metadata fixture", "poster": f"media/ep{ep:02d}.jpg",
               "hf": f"https://example.test/track_1/videos/episode_{ep:06d}.mp4"} for ep in range(30)]
    (directory / "index.html").write_text("const DATA = " + json.dumps({"videos": videos}) + ";\n", encoding="utf-8")
    monkeypatch.setattr(board, "revision", lambda root: "a" * 40)
    return tmp_path


def record(record_id="inputs-a", stage="inputs", state="planned", *, iteration="baseline", created_at="2026-10-15T01:00:00Z"):
    result = {"id": record_id, "episode_id": "episode_000000", "stage": stage,
              "iteration": iteration, "state": state, "kind": "real_track1", "created_at": created_at,
              "contributor": "Fixture contributor", "reviewer": "", "summary": "Independent declaration fixture",
              "delivery_url": "", "source_commit": "", "upstream": [], "artifacts": [],
              "coverage": {"frames": None, "objects_complete": None},
              "checks": {kind: {"status": "not_run", "evidence_url": ""} for kind in board.CHECKS},
              "blocker": "", "next_action": ""}
    if state in {"checked", "accepted"}:
        result.update(delivery_url="https://example.test/delivery", source_commit="a" * 40,
                      coverage={"frames": 10, "objects_complete": True})
        result["artifacts"] = [{"requirement_id": requirement["id"], "label": requirement["label"],
                                "url": "https://example.test/artifact", "sha256": "b" * 64, "bytes": 100}
                               for requirement in board.STAGE_MAP[stage]["requirements"]]
        result["checks"]["structural"] = {"status": "pass", "evidence_url": "https://example.test/structural"}
    if state == "accepted":
        result["reviewer"] = "Fixture reviewer"
        result["checks"]["quality"] = {"status": "pass", "evidence_url": "https://example.test/quality"}
        if stage == "export":
            result["checks"]["export"] = {"status": "pass", "evidence_url": "https://example.test/export"}
    return result


def index(repository):
    return {episode["id"]: episode for episode in board.load_episodes(repository)}


def incoming(repository, value, name="incoming.json"):
    path = repository / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_existing_metadata_inventory_has_exact_30_track1_episodes():
    # Repository metadata only; this does not load images, video or challenge outputs.
    episodes = board.load_episodes(ROOT)
    assert [episode["id"] for episode in episodes] == [f"episode_{ep:06d}" for ep in range(30)]
    assert sum(episode["frames"] for episode in episodes) == 16563
    assert episodes[0]["object"] == "hula_hoop"
    assert all("/track_1/videos/" in episode["video_url"] for episode in episodes)


def test_empty_snapshot_has_no_fabricated_progress(repository):
    snapshot = board.build_snapshot(repository)
    assert snapshot["records"] == []
    assert len(snapshot["episodes"]) == 30
    assert len(snapshot["stages"]) == 6
    assert {capability["module"] for capability in snapshot["capabilities"]} == {"platform", "human", "object", "physics"}
    board.write_snapshot(repository, snapshot)
    assert board.validate_snapshot(repository) == snapshot


def test_import_is_append_only_and_identical_retry_is_safe(repository):
    value = record()
    path = incoming(repository, value)
    assert board.import_record(repository, path) == "registered"
    assert board.import_record(repository, path) == "already registered"
    original = (repository / "tracking/records/inputs-a.json").read_bytes()
    value["state"] = "running"
    path = incoming(repository, value)
    with pytest.raises(board.BoardError, match="Immutable record collision"):
        board.import_record(repository, path)
    assert (repository / "tracking/records/inputs-a.json").read_bytes() == original
    assert len(board.validate_snapshot(repository)["records"]) == 1


@pytest.mark.parametrize("unsafe", ["javascript:alert(1)", "file:///tmp/data", "data:text/html,test", "//example.test/file", "https://user:password@example.test/file", "https://example.test/unsafe\npath", "https:\\example.test/file"])
def test_unsafe_links_are_rejected(repository, unsafe):
    value = record(state="checked")
    value["artifacts"][0]["url"] = unsafe
    with pytest.raises(board.BoardError):
        board.validate_record(value, index(repository))


@pytest.mark.parametrize("kind", ["synthetic", "engineering"])
def test_synthetic_or_engineering_cannot_be_accepted_video_progress(repository, kind):
    value = record(state="accepted")
    value["kind"] = kind
    with pytest.raises(board.BoardError, match="real_track1"):
        board.validate_record(value, index(repository))


@pytest.mark.parametrize("mutation", [
    lambda r: r["artifacts"].pop(),
    lambda r: r["artifacts"][0].update(sha256=""),
    lambda r: r["artifacts"][0].update(bytes=0),
    lambda r: r["artifacts"][0].update(requirement_id="unknown"),
    lambda r: r.update(source_commit="abc123"),
    lambda r: r.update(delivery_url=""),
    lambda r: r["checks"]["structural"].update(evidence_url=""),
    lambda r: r["checks"]["structural"].update(status="not_run"),
    lambda r: r["coverage"].update(frames=9),
    lambda r: r["coverage"].update(objects_complete=None),
])
def test_checked_requires_complete_declarations(repository, mutation):
    value = record(state="checked")
    mutation(value)
    with pytest.raises(board.BoardError):
        board.validate_record(value, index(repository))


def test_accepted_requires_actual_reviewer_and_quality_link(repository):
    value = record(state="accepted")
    board.validate_record(value, index(repository))
    value["reviewer"] = ""
    with pytest.raises(board.BoardError, match="reviewer"):
        board.validate_record(value, index(repository))
    value["reviewer"] = "Reviewer"
    value["checks"]["quality"]["evidence_url"] = ""
    with pytest.raises(board.BoardError):
        board.validate_record(value, index(repository))


def test_full_chain_and_export_acceptance(repository):
    records = []
    for position, stage in enumerate(board.STAGES):
        value = record(stage["id"] + "-a", stage["id"], "accepted", created_at=f"2026-10-15T0{position + 1}:00:00Z")
        value["upstream"] = [{"stage": parent["stage"], "record_id": parent["id"]} for parent in records]
        records.append(value)
    board.validate_records(records, index(repository))
    assert not board.stale_record_ids(records)
    records[-1]["checks"]["export"] = {"status": "not_run", "evidence_url": ""}
    with pytest.raises(board.BoardError, match="export pass"):
        board.validate_records(records, index(repository))


def test_cross_iteration_reuse_and_recursive_staleness(repository):
    inputs = record("inputs-a", state="checked")
    human = record("human-a", "human", "checked", iteration="candidate", created_at="2026-10-15T02:00:00Z")
    human["upstream"] = [{"stage": "inputs", "record_id": "inputs-a"}]
    objects = record("objects-a", "objects", "checked", iteration="candidate", created_at="2026-10-15T03:00:00Z")
    objects["upstream"] = [{"stage": "inputs", "record_id": "inputs-a"}, {"stage": "human", "record_id": "human-a"}]
    records = [inputs, human, objects]
    board.validate_records(records, index(repository))
    assert board.stale_record_ids(records) == set()
    separate = record("inputs-experiment", state="running", iteration="other", created_at="2026-10-15T04:00:00Z")
    assert board.stale_record_ids(records + [separate]) == set()
    replacement = record("inputs-b", state="failed", created_at="2026-10-15T05:00:00Z")
    board.validate_records(records + [replacement], index(repository))
    assert board.stale_record_ids(records + [replacement]) == {"human-a", "objects-a"}


@pytest.mark.parametrize("mutation", [
    lambda r: r[1]["upstream"][0].update(record_id="missing"),
    lambda r: r[0].update(episode_id="episode_000001"),
    lambda r: r[0].update(created_at="2026-10-15T03:00:00Z"),
    lambda r: r[0].update(state="running"),
    lambda r: r[1]["upstream"].clear(),
    lambda r: r[1]["upstream"][0].update(stage="human", record_id="human-a"),
])
def test_invalid_predecessor_graph_rejected(repository, mutation):
    records = [record(state="checked"), record("human-a", "human", "checked", created_at="2026-10-15T02:00:00Z")]
    records[1]["upstream"] = [{"stage": "inputs", "record_id": "inputs-a"}]
    mutation(records)
    with pytest.raises(board.BoardError):
        board.validate_records(records, index(repository))


@pytest.mark.parametrize("unsafe_id", ["../escape", "UPPER", "foo/bar", "con", "nul.txt", "bad."])
def test_record_id_cannot_escape_or_alias_paths(repository, unsafe_id):
    value = record(unsafe_id)
    with pytest.raises(board.BoardError):
        board.import_record(repository, incoming(repository, value))
    assert not (repository / "tracking/records").exists()


def test_snapshot_tamper_and_direct_unbuilt_record_fail_validation(repository):
    board.write_snapshot(repository, board.build_snapshot(repository))
    snapshot_path = repository / "tracking/board.json"
    snapshot = board.load_json(snapshot_path)
    snapshot["episodes"][0]["frames"] = 999
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(board.BoardError, match="episodes"):
        board.validate_snapshot(repository)
    board.write_snapshot(repository, board.build_snapshot(repository))
    directory = repository / "tracking/records"
    directory.mkdir(parents=True)
    (directory / "inputs-a.json").write_text(json.dumps(record()), encoding="utf-8")
    with pytest.raises(board.BoardError, match="records"):
        board.validate_snapshot(repository)


def test_inventory_cannot_drop_or_duplicate_an_episode(repository):
    path = repository / board.INVENTORY_PATH
    text = path.read_text(encoding="utf-8").replace('"ep": 29', '"ep": 28')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(board.BoardError, match="exactly Track 1"):
        board.load_episodes(repository)


def test_duplicate_json_keys_and_record_ids_fail(repository):
    path = repository / "duplicate.json"
    path.write_text('{"id":"one","id":"two"}', encoding="utf-8")
    with pytest.raises(board.BoardError, match="Duplicate JSON key"):
        board.load_json(path)
    with pytest.raises(board.BoardError, match="Duplicate record ID"):
        board.validate_records([record(), record()], index(repository))


def test_checked_without_dependencies_is_not_silently_registered(repository):
    path = incoming(repository, record("human-a", "human", "checked"))
    with pytest.raises(board.BoardError):
        board.import_record(repository, path)
    assert not (repository / "tracking/records/human-a.json").exists()


def test_cli_does_not_call_artifact_verification(repository, capsys):
    assert board.main(["build", "--root", str(repository)]) == 0
    assert board.main(["validate", "--root", str(repository)]) == 0
    output = capsys.readouterr().out
    assert "artifact bytes and quality were not verified" in output
    assert board.main(["import", "--root", str(repository)]) == 1


def test_snapshot_revision_is_build_provenance_not_rebuild_per_commit(repository, monkeypatch):
    board.write_snapshot(repository, board.build_snapshot(repository))
    monkeypatch.setattr(board, "revision", lambda root: "c" * 40)
    assert board.validate_snapshot(repository)["source"]["revision"] == "a" * 40


def envelope(records):
    return {"schema_version": board.SCHEMA_VERSION, "records": records}


def test_batch_import_supports_unordered_in_batch_dependencies_and_retry(repository):
    inputs = record(state="checked")
    human = record("human-a", "human", "checked", created_at="2026-10-15T02:00:00Z")
    human["upstream"] = [{"stage": "inputs", "record_id": "inputs-a"}]
    path = incoming(repository, envelope([human, inputs, inputs]))
    assert board.import_record(repository, path) == "registered"
    assert [r["id"] for r in board.validate_snapshot(repository)["records"]] == ["inputs-a", "human-a"]
    assert board.import_record(repository, path) == "already registered"


def test_batch_collision_does_not_write_any_new_records(repository):
    original = record()
    board.import_record(repository, incoming(repository, original))
    before = (repository / "tracking/board.json").read_bytes()
    changed = record(state="running")
    path = incoming(repository, envelope([record("new-record"), changed]))
    with pytest.raises(board.BoardError, match="Immutable record collision"):
        board.import_record(repository, path)
    assert not (repository / "tracking/records/new-record.json").exists()
    assert (repository / "tracking/board.json").read_bytes() == before


def test_invalid_batch_is_fully_validated_before_any_record_is_written(repository):
    bad = record("invalid-stage")
    bad["stage"] = "unknown"
    path = incoming(repository, envelope([record(), bad]))
    with pytest.raises(board.BoardError):
        board.import_record(repository, path)
    assert not (repository / "tracking/records").exists()


def test_conflicting_duplicate_within_batch_is_rejected(repository):
    path = incoming(repository, envelope([record(), record(state="running")]))
    with pytest.raises(board.BoardError, match="Immutable record collision"):
        board.import_record(repository, path)
    assert not (repository / "tracking/records").exists()


@pytest.mark.parametrize("payload", [
    {"schema_version": "unknown", "records": []},
    {"schema_version": board.SCHEMA_VERSION, "records": [], "episodes": []},
    {"schema_version": board.SCHEMA_VERSION, "records": {}},
])
def test_import_envelope_is_strict(repository, payload):
    with pytest.raises(board.BoardError):
        board.import_record(repository, incoming(repository, payload))


def test_failed_snapshot_write_rolls_back_only_this_batch(repository, monkeypatch):
    board.import_record(repository, incoming(repository, record()))
    previous = (repository / "tracking/board.json").read_bytes()
    def fail_write(*args):
        raise OSError("Independent write-failure fixture")
    monkeypatch.setattr(board, "write_snapshot", fail_write)
    path = incoming(repository, envelope([record("new-a"), record("new-b")]))
    with pytest.raises(OSError):
        board.import_record(repository, path)
    assert (repository / "tracking/records/inputs-a.json").exists()
    assert not (repository / "tracking/records/new-a.json").exists()
    assert not (repository / "tracking/records/new-b.json").exists()
    assert (repository / "tracking/board.json").read_bytes() == previous


@pytest.mark.parametrize("value", ["20261015T010000Z", "2026-W42-4T01:00:00Z", "2026-10-15T01:00Z", "2026-10-15T01:00:00.000001Z", "2026-10-15T01:00:00+00:60", "2026-10-15T01:00:00+24:00", "2026-02-30T01:00:00Z"])
def test_timestamp_precision_and_format_match_browser(repository, value):
    with pytest.raises(board.BoardError):
        board.validate_record(record(created_at=value), index(repository))


@pytest.mark.parametrize("value,microsecond,offset_seconds", [
    ("2026-10-15T01:00:00Z", 0, 0),
    ("2026-10-15T01:00:00.1Z", 100000, 0),
    ("2026-10-15T01:00:00.12-05:30", 120000, -19800),
    ("2026-10-15T01:00:00.123+08:00", 123000, 28800),
])
def test_timestamp_accepts_timezone_and_milliseconds(repository, value, microsecond, offset_seconds):
    board.validate_record(record(created_at=value), index(repository))
    parsed = board.timestamp(value, "fixture.created_at")
    assert parsed.microsecond == microsecond
    assert parsed.utcoffset().total_seconds() == offset_seconds


def test_integer_declarations_must_be_exact_in_browser(repository):
    value = record(state="checked")
    value["artifacts"][0]["bytes"] = 9007199254740992
    with pytest.raises(board.BoardError, match="safe integer"):
        board.validate_record(value, index(repository))
    value = record()
    value["coverage"]["frames"] = 9007199254740992
    with pytest.raises(board.BoardError, match="safe integer"):
        board.validate_record(value, index(repository))


def test_checked_cannot_mix_incompatible_frozen_predecessor_chains(repository):
    baseline = record("inputs-baseline", state="checked")
    candidate = record("inputs-candidate", state="checked", iteration="candidate")
    human = record("human-a", "human", "checked", iteration="candidate", created_at="2026-10-15T02:00:00Z")
    human["upstream"] = [{"stage": "inputs", "record_id": "inputs-candidate"}]
    objects = record("objects-a", "objects", "checked", iteration="candidate", created_at="2026-10-15T03:00:00Z")
    objects["upstream"] = [{"stage": "inputs", "record_id": "inputs-baseline"}, {"stage": "human", "record_id": "human-a"}]
    with pytest.raises(board.BoardError, match="coherent exact"):
        board.validate_records([baseline, candidate, human, objects], index(repository))
    objects["upstream"][0]["record_id"] = "inputs-candidate"
    board.validate_records([baseline, candidate, human, objects], index(repository))


@pytest.mark.parametrize("hostname", ["exa<mple.test", "example.test%2f.evil", "%65xample.test", "127.1", "1.2.3.4.", "999.1.1.1", "example.123", "example.0xff"])
def test_url_host_matches_browser_conservative_grammar(repository, hostname):
    value = record()
    value["delivery_url"] = f"https://{hostname}/evidence"
    with pytest.raises(board.BoardError):
        board.validate_record(value, index(repository))


@pytest.mark.parametrize("value", ["https://localhost/evidence", "https://1.2.3.4/evidence", "https://[::1]/evidence", "HTTPS://example.test/evidence", "https://0xgarbage/evidence"])
def test_url_supports_conventional_domains_and_explicit_ips(repository, value):
    candidate = record()
    candidate["delivery_url"] = value
    board.validate_record(candidate, index(repository))
