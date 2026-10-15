"""Build an evidence-index dashboard without opening videos or model outputs.

This validates declarations and their lineage, not artifact bytes or scientific
quality. Actual files still need the pipeline's delivery and acceptance checks.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlsplit


SCHEMA_VERSION = "hoi_progress.v1"
REPOSITORY = "https://github.com/Seanwilliam2077/video-to-hoi"
INVENTORY_PATH = "docs/track1-videos/index.html"
SCOPE = "Track 1 metadata and manually registered evidence declarations; six internal stage boundaries, native adapters and official acceptance remain separate"
STATES = {"planned", "running", "blocked", "submitted", "checked", "accepted", "failed"}
KINDS = {"real_track1", "synthetic", "engineering"}
CHECKS = ("structural", "quality", "export")
ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)\Z")
MAX_SAFE_INTEGER = 9007199254740991
RESERVED_IDS = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def _stage(stage_id, module, name, label, requirements, dependencies):
    return {"id": stage_id, "module": module, "name": name, "label": label,
            "requirements": [{"id": key, "label": value} for key, value in requirements],
            "dependencies": dependencies}


STAGES = [
    _stage("inputs", "platform", "Input evidence", "Inputs", [
        ("camera", "Camera and source frame identity"), ("masks", "Per-frame segmentation masks"),
        ("depth", "Depth and scale evidence")], []),
    _stage("human", "human", "Human reconstruction", "Human", [
        ("human", "Human trajectory and representation contract"), ("depth_scale", "Depth-scale record")], ["inputs"]),
    _stage("objects", "object", "Object reconstruction", "Objects", [
        ("object_record", "Object identity, frame and scale record"), ("mesh", "Object mesh and provenance")], ["inputs", "human"]),
    _stage("motion", "object", "Object motion", "Motion", [
        ("transforms", "Full-frame object transforms"), ("confidence", "Per-frame confidence")], ["inputs", "human", "objects"]),
    _stage("refine", "physics", "Interaction refinement", "Refine", [
        ("refined_human", "Refined human trajectory"), ("refined_objects", "Refined object trajectory")], ["inputs", "human", "objects", "motion"]),
    _stage("export", "platform", "Submission export", "Export", [
        ("metadata", "Episode, object and frame manifest"), ("parquet", "Complete trajectory tables"),
        ("mhr_arrays", "Human parameter arrays and representation contract"), ("meshes", "Object meshes and pose pairing")], ["inputs", "human", "objects", "motion", "refine"]),
]
STAGE_MAP = {stage["id"]: stage for stage in STAGES}
CAPABILITIES = [
    {"module": "platform", "status": "infrastructure_available",
     "summary": "Internal delivery gates and delivery folders exist. Native import, package relocation and official export acceptance remain pending.",
     "evidence_url": f"{REPOSITORY}/blob/main/docs/platform-plan.md"},
    {"module": "human", "status": "infrastructure_available",
     "summary": "Internal human-array structural checks exist. Real native human adapter and per-video accepted results are not registered.",
     "evidence_url": f"{REPOSITORY}/blob/main/docs/delivery-gates.md"},
    {"module": "object", "status": "infrastructure_available",
     "summary": "Internal mesh and motion structural checks exist. Real reconstruction, tracking and per-video accepted results are not registered.",
     "evidence_url": f"{REPOSITORY}/blob/main/docs/delivery-gates.md"},
    {"module": "physics", "status": "infrastructure_available",
     "summary": "Internal refined-output structural checks exist. Real optimization, interaction quality and per-video accepted results are not registered.",
     "evidence_url": f"{REPOSITORY}/blob/main/docs/delivery-gates.md"},
]


class BoardError(ValueError):
    """A declaration, inventory or output location is invalid."""


def require(condition, message):
    if not condition:
        raise BoardError(message)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_json_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(BoardError(f"Invalid JSON constant: {value}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BoardError(f"Cannot read JSON {path}: {exc}") from exc


def fields(value, expected, label):
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == set(expected), f"{label} fields must be exactly {', '.join(sorted(expected))}")


def string(value, label, *, nonempty=False):
    require(isinstance(value, str), f"{label} must be a string")
    require(not nonempty or bool(value.strip()), f"{label} cannot be empty")
    return value


def identifier(value, label):
    string(value, label, nonempty=True)
    require(bool(ID_PATTERN.fullmatch(value)) and not value.endswith(".") and value.split(".")[0] not in RESERVED_IDS,
            f"{label} must be a safe lowercase ID (letters, digits, dots, hyphens, underscores)")


def timestamp(value, label):
    string(value, label, nonempty=True)
    require(bool(TIMESTAMP_PATTERN.fullmatch(value)), f"{label} requires full ISO date/time, timezone and at most millisecond precision")
    try:
        # Python 3.10's fromisoformat accepts only 3 or 6 fractional digits.
        # strptime preserves the supported 1-3 digit browser timestamps too.
        date_format = "%Y-%m-%dT%H:%M:%S.%f%z" if "." in value else "%Y-%m-%dT%H:%M:%S%z"
        parsed = datetime.strptime(value.replace("Z", "+00:00"), date_format)
    except ValueError as exc:
        raise BoardError(f"{label} must be an ISO timestamp") from exc
    require("T" in value and parsed.tzinfo is not None, f"{label} requires date, time and timezone")
    return parsed


def url(value, label, *, nonempty=False):
    string(value, label, nonempty=nonempty)
    if not value:
        return
    require(not any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value) and "\\" not in value, f"{label} contains unsafe URL characters")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise BoardError(f"{label} is not a valid HTTP(S) URL") from exc
    require(parsed.scheme in {"http", "https"} and bool(parsed.hostname) and parsed.username is None and parsed.password is None,
            f"{label} must be an HTTP(S) URL without embedded credentials")
    require(port is None or 0 < port <= 65535, f"{label} has an invalid port")
    hostname = parsed.hostname
    require(bool(re.fullmatch(r"[a-zA-Z0-9._:-]+", hostname)), f"{label} requires an ASCII hostname (use punycode for international domains)")
    try:
        if ":" in hostname:
            ipaddress.IPv6Address(hostname)
        else:
            final_label = hostname.rstrip(".").split(".")[-1]
            if final_label.isdigit() or re.fullmatch(r"0x[0-9a-f]+", final_label, re.IGNORECASE):
                require(str(ipaddress.IPv4Address(hostname)) == hostname, f"{label} requires canonical dotted IPv4")
    except ValueError as exc:
        raise BoardError(f"{label} has an invalid or noncanonical IP hostname") from exc


def load_episodes(root):
    """Read the existing Track 1 gallery's JSON metadata; never load its media."""
    text = (Path(root) / INVENTORY_PATH).read_text(encoding="utf-8")
    match = re.search(r"^const DATA = (\{.*\});\s*$", text, flags=re.MULTILINE)
    require(match is not None, "Track 1 gallery DATA inventory was not found")
    data = json.loads(match.group(1), object_pairs_hook=_json_object)
    require(isinstance(data.get("videos"), list), "Gallery videos must be a list")
    videos = data["videos"]
    require(len(videos) == 30 and {v.get("ep") for v in videos} == set(range(30)), "Expected exactly Track 1 episodes 0 through 29")
    episodes = []
    for video in sorted(videos, key=lambda v: v["ep"]):
        ep = video["ep"]
        require(type(ep) is int, "Episode number must be an integer")
        require(type(video["frames"]) is int and video["frames"] > 0, "Frame count must be positive")
        require(type(video["seconds"]) in {int, float} and math.isfinite(video["seconds"]) and video["seconds"] > 0, "Duration must be positive and finite")
        require(video["poster"] == f"media/ep{ep:02d}.jpg", "Unexpected gallery poster path")
        url(video["hf"], "video_url", nonempty=True)
        require(f"/track_1/videos/" in video["hf"] and video["hf"].endswith(f"/episode_{ep:06d}.mp4"), "Video link must identify its Track 1 episode")
        for key in ("object", "objectName", "camera", "task"):
            string(video[key], f"video.{key}", nonempty=True)
        episodes.append({"id": f"episode_{ep:06d}", "ep": ep, "object": video["object"],
                         "object_name": video["objectName"], "frames": video["frames"],
                         "seconds": video["seconds"], "camera": video["camera"], "task": video["task"],
                         "poster_url": f"https://raw.githubusercontent.com/Seanwilliam2077/video-to-hoi/main/docs/track1-videos/{video['poster']}",
                         "video_url": video["hf"]})
    return episodes


RECORD_FIELDS = {"id", "episode_id", "stage", "iteration", "state", "kind", "created_at", "contributor", "summary",
                 "delivery_url", "source_commit", "upstream", "artifacts", "checks", "coverage", "reviewer", "blocker", "next_action"}


def validate_record(record, episode_index):
    fields(record, RECORD_FIELDS, "record")
    identifier(record["id"], "record.id")
    require(record["episode_id"] in episode_index, "Unknown episode_id")
    require(record["stage"] in STAGE_MAP, "Unknown stage")
    identifier(record["iteration"], "record.iteration")
    require(record["state"] in STATES, "Unknown state")
    require(record["kind"] in KINDS, "Unknown evidence kind")
    timestamp(record["created_at"], "record.created_at")
    for key in ("contributor", "reviewer", "summary", "blocker", "next_action"):
        string(record[key], f"record.{key}", nonempty=key in {"contributor", "summary"})
    require(record["state"] != "blocked" or record["blocker"].strip(), "Blocked records need a blocker")
    url(record["delivery_url"], "record.delivery_url")
    string(record["source_commit"], "record.source_commit")
    require(not record["source_commit"] or COMMIT_PATTERN.fullmatch(record["source_commit"]), "source_commit must be a full lowercase Git SHA")
    require(isinstance(record["upstream"], list), "upstream must be a list")
    upstream_stages = set()
    for item in record["upstream"]:
        fields(item, {"stage", "record_id"}, "upstream item")
        require(item["stage"] in STAGE_MAP[record["stage"]]["dependencies"], "Upstream stage is not a declared dependency")
        require(item["stage"] not in upstream_stages, "Duplicate upstream stage")
        upstream_stages.add(item["stage"])
        identifier(item["record_id"], "upstream.record_id")
    require(isinstance(record["artifacts"], list), "artifacts must be a list")
    expected = {item["id"] for item in STAGE_MAP[record["stage"]]["requirements"]}
    artifact_ids = set()
    for artifact in record["artifacts"]:
        fields(artifact, {"requirement_id", "label", "url", "sha256", "bytes"}, "artifact")
        key = artifact["requirement_id"]
        require(key in expected and key not in artifact_ids, "Unknown or duplicate artifact requirement_id")
        artifact_ids.add(key)
        string(artifact["label"], "artifact.label", nonempty=True)
        url(artifact["url"], "artifact.url", nonempty=True)
        require(isinstance(artifact["sha256"], str) and SHA_PATTERN.fullmatch(artifact["sha256"]), "artifact.sha256 must be a lowercase SHA-256")
        require(type(artifact["bytes"]) is int and 0 < artifact["bytes"] <= MAX_SAFE_INTEGER, "artifact.bytes must be a positive safe integer")
    fields(record["checks"], CHECKS, "checks")
    for key in CHECKS:
        check = record["checks"][key]
        fields(check, {"status", "evidence_url"}, f"checks.{key}")
        require(check["status"] in {"not_run", "pass", "fail"}, f"Unknown {key} status")
        url(check["evidence_url"], f"checks.{key}.evidence_url", nonempty=check["status"] != "not_run")
    fields(record["coverage"], {"frames", "objects_complete"}, "coverage")
    coverage = record["coverage"]
    require(coverage["frames"] is None or (type(coverage["frames"]) is int and 0 <= coverage["frames"] <= MAX_SAFE_INTEGER), "coverage.frames must be null or a nonnegative safe integer")
    require(coverage["objects_complete"] is None or type(coverage["objects_complete"]) is bool, "coverage.objects_complete must be null or boolean")
    if record["state"] in {"checked", "accepted"}:
        require(record["kind"] == "real_track1", "Checked/accepted video progress requires real_track1 evidence")
        require(artifact_ids == expected, "Checked/accepted records require every stage output")
        require(bool(record["delivery_url"]) and bool(record["source_commit"]), "Checked/accepted records require delivery_url and source_commit")
        require(record["checks"]["structural"]["status"] == "pass", "Checked/accepted records require structural pass evidence")
        require(upstream_stages == set(STAGE_MAP[record["stage"]]["dependencies"]), "Checked/accepted records require every predecessor binding")
        require(coverage["frames"] == episode_index[record["episode_id"]]["frames"] and coverage["objects_complete"] is True, "Checked/accepted records require complete declared frame and object coverage")
    if record["state"] == "accepted":
        require(bool(record["reviewer"].strip()), "Accepted records require a reviewer")
        require(record["checks"]["quality"]["status"] == "pass", "Accepted records require quality pass evidence")
        if record["stage"] == "export":
            require(record["checks"]["export"]["status"] == "pass", "Accepted export requires export pass evidence")


def validate_records(records, episode_index):
    require(isinstance(records, list), "records must be a list")
    by_id = {}
    for record in records:
        validate_record(record, episode_index)
        require(record["id"] not in by_id, f"Duplicate record ID: {record['id']}")
        by_id[record["id"]] = record
    for record in records:
        for item in record["upstream"]:
            require(item["record_id"] in by_id, f"Unknown predecessor: {item['record_id']}")
            parent = by_id[item["record_id"]]
            require(parent["stage"] == item["stage"], "Predecessor stage mismatch")
            require(parent["episode_id"] == record["episode_id"], "Predecessor must match episode")
            require(timestamp(parent["created_at"], "parent.created_at") <= timestamp(record["created_at"], "created_at"), "Predecessor cannot be newer than its consumer")
            if record["state"] in {"checked", "accepted"}:
                require(parent["state"] in {"checked", "accepted"} and parent["kind"] == "real_track1", "Checked/accepted consumer needs checked/accepted real predecessors")
                direct_bindings = {entry["stage"]: entry["record_id"] for entry in record["upstream"]}
                require(all(direct_bindings.get(entry["stage"]) == entry["record_id"] for entry in parent["upstream"]),
                        "Checked/accepted consumer must bind a coherent exact predecessor chain")
    # Explicit graph traversal also protects future schema changes to dependencies.
    visiting, visited = set(), set()
    def visit(record_id):
        require(record_id not in visiting, "Cyclic predecessor references")
        if record_id in visited:
            return
        visiting.add(record_id)
        for item in by_id[record_id]["upstream"]:
            visit(item["record_id"])
        visiting.remove(record_id)
        visited.add(record_id)
    for record_id in by_id:
        visit(record_id)


def stale_record_ids(records):
    """Derive lineage staleness without rewriting immutable history.

    A new failed/running predecessor also masks an older passing predecessor.
    Attempts from other iterations do not replace a baseline's predecessors.
    """
    by_id = {record["id"]: record for record in records}
    latest = {}
    for record in sorted(records, key=lambda r: (timestamp(r["created_at"], "created_at"), r["id"])):
        latest[(record["episode_id"], record["iteration"], record["stage"])] = record["id"]
    stale = set()
    for _ in range(len(records)):
        previous = len(stale)
        for record in records:
            for item in record["upstream"]:
                parent = by_id.get(item["record_id"])
                key = (parent["episode_id"], parent["iteration"], item["stage"]) if parent else None
                if parent is None or latest.get(key) != item["record_id"] or item["record_id"] in stale:
                    stale.add(record["id"])
        if len(stale) == previous:
            break
    return stale


def contained_path(root, *parts):
    root = Path(root).resolve()
    candidate = root.joinpath(*parts)
    require(candidate.resolve().is_relative_to(root), "Path escapes the repository")
    current = candidate
    while current != root:
        require(not current.is_symlink(), f"Symlink outputs are not allowed: {current}")
        current = current.parent
    return candidate


def read_records(root, episodes):
    directory = contained_path(root, "tracking", "records")
    records = []
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            require(not path.is_symlink(), "Record files cannot be symlinks")
            record = load_json(path)
            require(isinstance(record, dict) and path.name == f"{record.get('id')}.json", "Record filename must equal its ID")
            records.append(record)
    validate_records(records, {episode["id"]: episode for episode in episodes})
    return sorted(records, key=lambda r: (timestamp(r["created_at"], "created_at"), r["id"]))


def revision(root):
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    value = result.stdout.strip()
    require(result.returncode == 0 and bool(COMMIT_PATTERN.fullmatch(value)), "Cannot determine inventory source Git revision")
    return value


def build_snapshot(root):
    episodes = load_episodes(root)
    return {"schema_version": SCHEMA_VERSION, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "source": {"inventory_path": INVENTORY_PATH + "#DATA", "repository": REPOSITORY, "revision": revision(root), "scope": SCOPE},
            "episodes": episodes, "stages": STAGES, "records": read_records(root, episodes), "capabilities": CAPABILITIES}


def write_snapshot(root, snapshot):
    path = contained_path(root, "tracking", "board.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(snapshot, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_snapshot(root):
    snapshot = load_json(contained_path(root, "tracking", "board.json"))
    fields(snapshot, {"schema_version", "updated_at", "source", "episodes", "stages", "records", "capabilities"}, "snapshot")
    require(snapshot["schema_version"] == SCHEMA_VERSION, "Unknown board schema")
    timestamp(snapshot["updated_at"], "updated_at")
    fields(snapshot["source"], {"inventory_path", "repository", "revision", "scope"}, "source")
    require(isinstance(snapshot["source"]["revision"], str) and COMMIT_PATTERN.fullmatch(snapshot["source"]["revision"]), "Invalid snapshot source revision")
    expected = build_snapshot(root)
    for key in ("inventory_path", "repository", "scope"):
        require(snapshot["source"][key] == expected["source"][key], f"Snapshot source.{key} is inconsistent")
    for key in ("episodes", "stages", "records", "capabilities"):
        require(snapshot[key] == expected[key], f"Snapshot {key} is out of date; run build")
    return snapshot


def import_record(root, incoming):
    """Validate a complete batch before appending; preserve identical retries."""
    episodes = load_episodes(root)
    episode_index = {episode["id"]: episode for episode in episodes}
    payload = load_json(incoming)
    if isinstance(payload, dict) and "schema_version" in payload:
        fields(payload, {"schema_version", "records"}, "import envelope")
        require(payload["schema_version"] == SCHEMA_VERSION, "Unknown import schema")
        require(isinstance(payload["records"], list), "Import records must be a list")
        batch = payload["records"]
    else:
        batch = [payload]
    records = read_records(root, episodes)
    by_id = {record["id"]: record for record in records}
    additions = []
    for record in batch:
        validate_record(record, episode_index)
        if record["id"] in by_id:
            require(by_id[record["id"]] == record, f"Immutable record collision: {record['id']}; use a new ID")
        else:
            additions.append(record)
            by_id[record["id"]] = record
    validate_records(list(by_id.values()), episode_index)
    snapshot = build_snapshot(root)  # Validate provenance and every declaration before appending.
    snapshot["records"] = sorted(by_id.values(), key=lambda r: (timestamp(r["created_at"], "created_at"), r["id"]))
    paths = [contained_path(root, "tracking", "records", record["id"] + ".json") for record in additions]
    require(not any(path.exists() for path in paths), "A record output path already exists")
    created = []
    try:
        for record, path in zip(additions, paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                created.append(path)
                json.dump(record, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        write_snapshot(root, snapshot)
    except BaseException:
        # Roll back only files created by this invocation; never remove a collision.
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return "registered" if additions else "already registered"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "build", "import"))
    parser.add_argument("record", nargs="?", help="JSON record to append (import only)")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        require((args.command == "import") == bool(args.record), "Only import requires a record JSON path")
        if args.command == "validate":
            board = validate_snapshot(args.root)
            message = "valid declarations and synchronized snapshot"
        elif args.command == "build":
            board = build_snapshot(args.root)
            write_snapshot(args.root, board)
            message = "snapshot built"
        else:
            message = import_record(args.root, Path(args.record))
            board = validate_snapshot(args.root)
        print(json.dumps({"result": message, "episodes": len(board["episodes"]), "records": len(board["records"]),
                          "stale_record_ids": sorted(stale_record_ids(board["records"])),
                          "scope": "Evidence declarations only; artifact bytes and quality were not verified."}))
        return 0
    except (BoardError, OSError, TypeError, KeyError) as exc:
        print(f"progress-board: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
