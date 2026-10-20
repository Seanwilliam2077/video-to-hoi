"""Build a read-only, evidence-bound catalog of the published Body input delivery.

This does not run a model, certify scientific quality, or update stage records.
The fixed delivery revision and receipt hashes are the trust roots. New deliveries
need a separate reviewed catalog version; unrelated new tracking records are ignored.
Only --copy-previews needs the ignored local data package. Normal builds and --check
work offline from committed receipts and byte-identical, tracked QA previews.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
DELIVERY_REVISION = "19c11378d1f2e0f9aac2ac579743bb8cf14b8213"
DELIVERY = Path("deliveries/platform/body-inputs-track1-20261016")
RAW_DELIVERY = Path("deliveries/platform/raw-track1-20261015")
PREVIEWS = Path("docs/progress-board/previews")
CATALOG = Path("tracking/delivery-catalog.json")
PROFILE = "sam3d_body_masks_camera.v1"
DATASET_REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
RECORDS_DIGEST = "5f463bd29a24d3851759c2218abccfccc77a64fc2f5690b863bd137911f638ae"
RAW_BASE = f"https://raw.githubusercontent.com/Seanwilliam2077/video-to-hoi/{DELIVERY_REVISION}/"
TRUST_ROOTS = {
    DELIVERY / "inventory.json": "a63de9b4441df5065540326f96e6faf26530ba2ae0a8bceb1219169cbf53adf9",
    DELIVERY / "checks.json": "d4553996ef3eff9c8f022c82707ef48905690116ad8ed7fb99eb11ff169ce4c1",
    DELIVERY / "native-reader-checks.json": "6491eac34c6ed75d08d2248082e0ddfb7e8e496f2c2d9f697ccc460e4530cf94",
    RAW_DELIVERY / "inventory.json": "1c610ef47eca9b4e204cc97c9c4ddaf817b4ad1f94b66f0ff19bf1cf5bb27962",
    RAW_DELIVERY / "checks.json": "8265b5676586a4f1cd188dfc5294d0ef8c660581f5d479c3daa0353dc43959d8",
}
SMALL_FILES = (
    ("input.json", "receipts", "Body input manifest"),
    ("intrinsics.json", "camera", "Estimated camera intrinsics"),
    ("camera-estimation.json", "camera", "Camera estimation evidence"),
    ("masks-provenance.json", "masks", "Human mask provenance"),
    ("review.json", "qa", "Body input review"),
    ("visual-review-receipt.json", "qa", "Sampled visual review receipt"),
)
PREVIEW_FILES = (
    ("review-001.jpg", "review", "mask_review", "Sampled RGB and mask overview"),
    ("targets-001.jpg", "targets", "target_review", "Sampled target-person detail"),
)


class CatalogError(ValueError):
    """The catalog cannot establish the declared immutable evidence identity."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path, *, git_text: bool = False) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CatalogError(f"Missing or unreadable file: {path}") from exc
    # The old raw delivery uses Git text normalization. The Body delivery is
    # explicitly -text in .gitattributes, so its exact CRLF bytes stay untouched.
    return data.replace(b"\r\n", b"\n") if git_text else data


def read_json(path: Path):
    try:
        return json.loads(read_bytes(path))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Invalid JSON: {path}") from exc


def verify_file(path: Path, expected: str, *, git_text: bool = False) -> bytes:
    data = read_bytes(path, git_text=git_text)
    require(sha256(data) == expected, f"SHA256 mismatch: {path}")
    return data


def indexed_episodes(document: dict, label: str) -> dict:
    episodes = document.get("episodes", [])
    require(len(episodes) == 30, f"{label}: expected exactly 30 episodes")
    indexed = {item.get("episode_index"): item for item in episodes}
    require(set(indexed) == set(range(30)), f"{label}: missing or duplicate episode")
    return indexed


def evidence_file(root: Path, relative: Path, identifier: str, label: str) -> dict:
    data = read_bytes(root / relative, git_text=RAW_DELIVERY in relative.parents)
    return {"id": identifier, "label": label, "url": RAW_BASE + relative.as_posix(),
            "sha256": sha256(data), "bytes": len(data)}


def build_catalog(root: Path = ROOT, *, preview_source: Path | None = None) -> dict:
    """Verify the committed hash chain and return a deterministic catalog."""
    root = Path(root)
    for relative, digest in TRUST_ROOTS.items():
        verify_file(root / relative, digest, git_text=RAW_DELIVERY in relative.parents)

    inventory = read_json(root / DELIVERY / "inventory.json")
    checks = read_json(root / DELIVERY / "checks.json")
    native = read_json(root / DELIVERY / "native-reader-checks.json")
    raw_inventory = read_json(root / RAW_DELIVERY / "inventory.json")
    raw_checks = read_json(root / RAW_DELIVERY / "checks.json")
    inv_rows = indexed_episodes(inventory, "inventory")
    check_rows = indexed_episodes(checks, "handoff checks")
    native_rows = indexed_episodes(native, "native reader checks")
    raw_rows = indexed_episodes(raw_inventory, "raw inventory")

    require(inventory["profile"] == checks["profile"] == PROFILE, "Unexpected Body profile")
    require(checks["passed"] and raw_checks["passed"], "Delivery checks did not pass")
    require(inventory["all_30_body_inputs_verified"] and checks["all_30_body_inputs_verified"],
            "Body input delivery is not verified")
    require(inventory["all_30_native_readers_passed"] and native["all_30_native_readers_passed"],
            "Native readers did not pass")
    require(not any(doc["body_execution_completed"] for doc in (inventory, checks, native)),
            "This catalog describes input preparation only")
    require(native["handoff_checks_sha256"] == TRUST_ROOTS[DELIVERY / "checks.json"],
            "Native checks refer to a different handoff report")
    require(checks["raw_manifest_sha256"] == raw_inventory["manifest_sha256"]
            == raw_checks["manifest_sha256"], "Raw manifest identity mismatch")
    require(checks["source_inventory_sha256"] == raw_inventory["source_inventory_sha256"],
            "Raw source inventory identity mismatch")
    require(raw_inventory["repository"] == "nvidia/video_to_data_challenge"
            and raw_inventory["revision"] == DATASET_REVISION, "Unexpected dataset source")
    require(inventory["frames_verified"] == checks["frames_verified"]
            == raw_inventory["total_frames"] == raw_checks["frames_verified"] == 16563,
            "Total frame count mismatch")

    records = [read_json(root / f"tracking/records/body-inputs-20261016-ep{ep:02d}.json")
               for ep in range(30)]
    digest = sha256(json.dumps(records, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode("utf-8"))
    require(digest == RECORDS_DIGEST, "Registered immutable Body input records changed")
    board = read_json(root / "tracking/board.json")
    board_records = board.get("records", [])
    for record in records:
        matches = [value for value in board_records if value.get("id") == record["id"]]
        require(matches == [record], f"Board record missing, duplicated, or mismatched: {record['id']}")

    shared_evidence = [
        evidence_file(root, RAW_DELIVERY / "inventory.json", "raw_inventory", "Raw video inventory"),
        evidence_file(root, RAW_DELIVERY / "checks.json", "raw_checks", "Raw video checks"),
        evidence_file(root, DELIVERY / "inventory.json", "inventory", "Body delivery inventory"),
        evidence_file(root, DELIVERY / "checks.json", "handoff_checks", "Body input gate report"),
        evidence_file(root, DELIVERY / "native-reader-checks.json", "native_readers", "Native reader report"),
    ]
    entries = []
    preview_bytes = 0
    for ep, record in enumerate(records):
        episode_id = f"episode_{ep:06d}"
        folder = DELIVERY / "episodes" / episode_id
        row, gate, reader, raw = inv_rows[ep], check_rows[ep], native_rows[ep], raw_rows[ep]
        require(row["body_inputs_verified"] and gate["body_inputs_verified"]
                and gate["inputs_structurally_verified"] and gate["visual_review_verified"]
                and row["native_readers_passed"] and reader["passed"],
                f"{episode_id}: incomplete verification")
        require(not gate["errors"] and not reader["errors"], f"{episode_id}: verification errors")
        frame_count = row["frame_count"]
        require(frame_count == gate["frame_count"] == gate["masks_verified"]
                == reader["frames_verified"] == raw["frame_count"]
                == row["human_mask_count"] == record["coverage"]["frames"],
                f"{episode_id}: frame count mismatch")
        require(record["episode_id"] == episode_id and record["stage"] == "inputs"
                and record["state"] == "submitted" and record["kind"] == "real_track1",
                f"{episode_id}: incompatible registered record")
        require(row["input_sha256"] == gate["input_sha256"] == reader["input_sha256"],
                f"{episode_id}: input receipt identity mismatch")
        require(row["review_sha256"] == gate["review_sha256"], f"{episode_id}: review identity mismatch")
        verify_file(root / folder / "input.json", row["input_sha256"])
        verify_file(root / folder / "review.json", row["review_sha256"])
        manifest = read_json(root / folder / "input.json")
        review = read_json(root / folder / "review.json")
        source = manifest["source"]
        require(manifest["episode_index"] == review["episode_index"] == ep
                and manifest["profile"] == PROFILE, f"{episode_id}: manifest identity mismatch")
        require(source["repository"] == raw_inventory["repository"]
                and source["revision"] == DATASET_REVISION
                and source["video_path"] == raw["video_path"],
                f"{episode_id}: source video identity mismatch")
        require(source["video_sha256"] == row["video_sha256"] == gate["video_sha256"]
                == raw["video"]["sha256"], f"{episode_id}: video hash mismatch")
        require(source["timeline_sha256"] == raw["timeline_sha256"]
                == manifest["artifacts"]["timeline.json"], f"{episode_id}: timeline identity mismatch")
        require(review["status"] == "approved" and not review["unresolved_issues"],
                f"{episode_id}: unresolved input review")
        require(review["reviewed_artifacts"]["input.json"] == row["input_sha256"],
                f"{episode_id}: review refers to a different input")
        verify_file(root / folder / "visual-review-receipt.json", review["source_receipt_sha256"])
        receipt = read_json(root / folder / "visual-review-receipt.json")
        require(receipt["episode_index"] == ep and receipt["video_sha256"] == source["video_sha256"]
                and receipt["status"] == "approved" and not receipt["unresolved_issues"],
                f"{episode_id}: visual review identity mismatch")
        require(receipt["sample_frame_ids"] == review["sample_frame_ids"]
                == row["review_sample_frame_ids"], f"{episode_id}: review sample mismatch")
        for name in ("camera-estimation.json", "intrinsics.json"):
            expected = manifest["artifacts"][name]
            require(review["reviewed_artifacts"][name] == receipt["reviewed_artifacts"][name] == expected,
                    f"{episode_id}: reviewed {name} identity mismatch")
            verify_file(root / folder / name, expected)
        verify_file(root / folder / "masks-provenance.json", manifest["producer"]["provenance_sha256"])
        require(receipt["reviewed_artifacts"]["masks-provenance.json"]
                == manifest["producer"]["provenance_sha256"], f"{episode_id}: mask provenance mismatch")
        camera = read_json(root / folder / "intrinsics.json")
        require(camera["fx"] == row["camera_fx_pixels"] and camera["fy"] == row["camera_fy_pixels"]
                and camera["width"] == reader["width"] == raw["width"]
                and camera["height"] == reader["height"] == raw["height"],
                f"{episode_id}: camera values mismatch")
        archive_path = f"body-inputs/{episode_id}"
        require(row["archive_episode_path"] == archive_path, f"{episode_id}: unexpected archive path")
        source_url = (f"https://huggingface.co/datasets/{source['repository']}/resolve/"
                      f"{DATASET_REVISION}/track_1/{source['video_path']}")
        files = [{"path": source["video_path"], "group": "raw", "kind": "video",
                  "url": source_url, "sha256": source["video_sha256"],
                  "bytes": raw["video"]["size_bytes"],
                  "archive_path": "data/v2d/track_1/" + source["video_path"]}]
        evidence = list(shared_evidence)
        for name, group, label in SMALL_FILES:
            item = evidence_file(root, folder / name, name.removesuffix(".json"), label)
            evidence.append(item)
            files.append({"path": name, "group": group, "kind": "json", "url": item["url"],
                          "sha256": item["sha256"], "bytes": item["bytes"],
                          "archive_path": f"{archive_path}/{name}"})
        for name, group in (("timeline.json", "timeline"), ("observations.json", "observations"),
                            ("masks-manifest.json", "masks")):
            expected = manifest["artifacts"][name]
            require(review["reviewed_artifacts"][name] == expected,
                    f"{episode_id}: reviewed {name} identity mismatch")
            if name != "timeline.json":
                require(receipt["reviewed_artifacts"][name] == expected,
                        f"{episode_id}: visual review {name} identity mismatch")
            files.append({"path": name, "group": group, "kind": "json",
                          "sha256": expected, "archive_path": f"{archive_path}/{name}"})
        files.append({"path": "human_masks/", "group": "masks", "kind": "directory",
                      "archive_path": f"{archive_path}/human_masks/"})
        previews = []
        for source_name, suffix, kind, label in PREVIEW_FILES:
            expected = receipt["reviewed_artifacts"][f"mask-qa/{source_name}"]
            filename = f"{episode_id}-{suffix}.jpg"
            destination = root / PREVIEWS / filename
            if preview_source is not None:
                source_path = preview_source / episode_id / "mask-qa" / source_name
                verify_file(source_path, expected)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, destination)
            preview_data = verify_file(destination, expected)
            preview_bytes += len(preview_data)
            previews.append({"url": f"previews/{filename}", "sha256": expected,
                             "kind": kind, "label": label})
        require(record["artifacts"][0]["sha256"] == manifest["artifacts"]["intrinsics.json"],
                f"{episode_id}: registered camera identity mismatch")
        archive = inventory["archive"]
        require(record["artifacts"][1]["sha256"] == archive["sha256"]
                and record["artifacts"][1]["bytes"] == archive["bytes"]
                and record["artifacts"][1]["url"] == archive["url"],
                f"{episode_id}: registered archive identity mismatch")
        sampling_limitation = receipt.get(
            "sampling_limitation",
            "Visual review covers the listed sample frames, not every source frame. "
            "Full file coverage and hashes were checked automatically. "
            "Fine boundary accuracy and downstream Body reconstruction quality remain unverified.",
        )
        if receipt.get("quality_limitations"):
            sampling_limitation += " " + " ".join(receipt["quality_limitations"])
        entries.append({
            "record": record, "episode_id": episode_id, "profile": PROFILE,
            "frame_count": frame_count, "raw_verified": True, "body_inputs_verified": True,
            "native_readers_passed": True, "body_execution_completed": False,
            "camera": camera,
            "review": {"sample_frame_ids": receipt["sample_frame_ids"],
                       "method": receipt["review_method"], "findings": receipt["findings"],
                       "sampling_limitation": sampling_limitation},
            "archive": {"url": archive["url"], "bytes": archive["bytes"],
                        "sha256": archive["sha256"], "episode_path": archive_path},
            "source_video_url": source_url, "evidence": evidence, "files": files, "previews": previews,
        })
    require(preview_bytes < 25_000_000, "Preview payload exceeds the 25 MB publication budget")
    return {"schema_version": "hoi_delivery_catalog.v1",
            "delivery_revision": DELIVERY_REVISION, "entries": entries}


def serialize(catalog: dict) -> bytes:
    return (json.dumps(catalog, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root")
    parser.add_argument("--check", action="store_true", help="Validate without writing files")
    parser.add_argument("--copy-previews", type=Path, metavar="BODY_INPUT_DIRECTORY",
                        help="Copy verified existing QA JPEGs from the local 30-episode Body package")
    args = parser.parse_args(argv)
    try:
        require(not (args.check and args.copy_previews), "--check cannot copy or modify previews")
        result = build_catalog(args.root, preview_source=args.copy_previews)
        expected = serialize(result)
        target = args.root / CATALOG
        if args.check:
            require(read_bytes(target, git_text=True) == expected,
                    "Delivery catalog is stale; run python tools/build_delivery_catalog.py")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(expected)
        print(f"Delivery catalog {'verified' if args.check else 'built'}: "
              f"{len(result['entries'])} episodes, 16563 frames, 60 unchanged QA previews")
        return 0
    except (CatalogError, KeyError, TypeError) as exc:
        print(f"Delivery catalog error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
