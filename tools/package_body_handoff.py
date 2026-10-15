"""Assemble reviewed Body inputs and verify a portable all-30 receiver archive.

Reviews are supplied evidence: this tool never generates approval, findings or
sample frame IDs. Assembly writes a fresh directory. Packaging rechecks all 30
episodes, validates every ZIP member hash, extracts into a fresh directory and
runs the bundled receiver gate before publishing the final archive filename.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import zipfile

_SPEC = importlib.util.spec_from_file_location("package_body_gate", Path(__file__).with_name("check_body_handoff.py"))
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)
raw = gate.raw
require = gate.require

RAW_PREFIX = "data/v2d/track_1"
PREPARED_PREFIX = "data/prepared/track1"
RECEIPT_ARTIFACTS = {"observations.json", "masks-manifest.json", "masks-provenance.json",
                     "intrinsics.json", "camera-estimation.json"}
ASSEMBLED_NAMES = {"input.json", "timeline.json", "review.json", "visual-review-receipt.json", "SHA256SUMS", "checks.json"}
REVIEW_FIELDS = ("episode_index", "status", "reviewer", "review_method", "reviewed_at",
                 "sample_frame_ids", "findings", "unresolved_issues")
PACKAGES = ("numpy", "Pillow", "av", "pandas", "pyarrow")


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def file_tree(root: Path) -> dict[str, Path]:
    raw.reject_links(root)
    require(root.is_dir(), f"missing directory: {root}")
    result = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            path = Path(directory) / name
            raw.reject_links(path)
            raw.safe_relative(path.relative_to(root).as_posix())
        for name in files:
            path = Path(directory) / name
            require(path.is_file(), f"not a regular file: {path}")
            result[path.relative_to(root).as_posix()] = path
    return result


def fresh(path: Path) -> None:
    raw.reject_links(path)
    require(not path.exists(), f"destination already exists; choose a fresh path: {path}")


def validate_receipt(receipt: dict, source: Path, row: dict, video_sha: str) -> None:
    require(receipt.get("schema_version") == "body_visual_review_receipt.v1", "unknown visual review receipt schema")
    require(type(receipt.get("episode_index")) is int and receipt["episode_index"] == row["episode_index"], "receipt episode mismatch")
    require(receipt.get("video_sha256") == video_sha and receipt.get("target_id") == "human_0", "receipt source/target mismatch")
    # Reuse the final gate's review requirements, supplying only temporary hash
    # placeholders for fields whose real values exist after assembly. The actual
    # approval, frame IDs, findings and identity are never filled in here.
    temporary = {key: receipt.get(key) for key in REVIEW_FIELDS}
    temporary.update(schema_version="body_input_review.v1",
                     reviewed_artifacts={name: "0" * 64 for name in gate.REVIEW_ARTIFACTS})
    gate.validate_review(temporary, row["episode_index"], row["frame_count"], temporary["reviewed_artifacts"])
    provenance = gate.read_json(source / "masks-provenance.json")
    require(provenance.get("schema_version") == "body_masks_provenance.v1"
            and provenance.get("episode_index") == row["episode_index"]
            and provenance.get("source_video_sha256") == video_sha
            and provenance.get("timeline_sha256") == row["timeline_sha256"], "mask provenance source mismatch")
    gate.validate_producer(provenance)
    require(provenance.get("invalid_frame_ids") == [], "mask producer reports invalid frames")
    required_ids = gate.frame_ids(provenance.get("visual_review_frame_ids"), row["frame_count"], "mask QA frame IDs")
    require(set(required_ids) <= set(receipt["sample_frame_ids"]), "review receipt does not cover all required QA/flagged frames")
    qa_files = file_tree(source / "mask-qa")
    require(any(name.endswith((".jpg", ".png")) for name in qa_files), "missing actual visual review images")
    required_artifacts = RECEIPT_ARTIFACTS | {"mask-qa/" + name for name in qa_files}
    bindings = receipt.get("reviewed_artifacts")
    require(isinstance(bindings, dict) and required_artifacts <= set(bindings), "receipt must bind model outputs and every QA artifact")
    for relative, expected in bindings.items():
        raw.safe_relative(relative)
        require(relative not in ASSEMBLED_NAMES, "receipt must refer to producer artifacts, not future assembled files")
        gate.digest(expected, relative)
        raw.reject_links(source / relative)
        require(raw.sha256(source / relative) == expected, f"stale visual review receipt: {relative}")


def assembly_context(raw_root: Path, source_inventory: Path) -> dict:
    manifest = gate.read_json(raw_root / "manifest.json")
    checks = gate.read_json(raw_root / "checks.json")
    raw.load_inventory(source_inventory)
    require(manifest.get("schema_version") == "track1_raw_preparation.v1" and manifest.get("raw_data_ready") is True
            and manifest.get("track") == "track_1" and manifest.get("repository") == raw.REPOSITORY
            and manifest.get("revision") == raw.REVISION, "requires verified pinned Track 1 raw preparation")
    require(checks.get("passed") is True and checks.get("manifest_sha256") == raw.sha256(raw_root / "manifest.json"),
            "raw preparation checks changed")
    require(manifest.get("source_inventory_sha256") == raw.sha256(source_inventory), "raw source lock changed")
    require([row.get("episode_index") for row in manifest.get("episodes", [])] == list(range(30)),
            "raw preparation must contain exactly all 30 ordered episodes")
    return manifest


def assemble(inputs_root: Path, dataset_root: Path, raw_root: Path, review_root: Path,
             source_inventory: Path, output: Path, episodes: list[int] | None = None) -> dict:
    selected = list(range(30)) if episodes is None else episodes
    gate.frame_ids(selected, 30, "assembly episodes")
    fresh(output)
    require(not any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve())
                    for p in (inputs_root, dataset_root, raw_root, review_root)), "assembly output and inputs must not overlap")
    manifest = assembly_context(raw_root, source_inventory)
    planned = []
    # Check every requested approval before creating an output directory.
    for episode in selected:
        row = manifest["episodes"][episode]
        source = inputs_root / f"episode_{episode:06d}"
        files = file_tree(source)
        require(not ASSEMBLED_NAMES & set(files), "source already contains assembled files; use a fresh producer directory")
        receipt_path = review_root / f"episode_{episode:06d}.json"
        receipt = gate.read_json(receipt_path)
        video_sha = manifest["source_files"][row["video_path"]]["sha256"]
        validate_receipt(receipt, source, row, video_sha)
        timeline_path = raw_root / raw.safe_relative(row["timeline_path"])
        require(raw.sha256(timeline_path) == row["timeline_sha256"], "verified timeline changed")
        planned.append((episode, row, source, files, receipt_path, receipt, video_sha, timeline_path))
    output.mkdir(parents=True, exist_ok=False)
    for episode, row, source, files, receipt_path, receipt, video_sha, timeline_path in planned:
        destination = output / f"episode_{episode:06d}"
        destination.mkdir()
        for relative, path in files.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        shutil.copyfile(timeline_path, destination / "timeline.json")
        shutil.copyfile(receipt_path, destination / "visual-review-receipt.json")
        # Reject source changes between reviewing inputs and copying them.
        validate_receipt(receipt, destination, row, video_sha)
        require(gate.read_json(destination / "visual-review-receipt.json") == receipt, "receipt changed during assembly")
        provenance = gate.read_json(destination / "masks-provenance.json")
        producer = {name: provenance[name] for name in ("method", "code_revision", "checkpoint_sha256", "environment")}
        producer.update(provenance_path="masks-provenance.json", provenance_sha256=raw.sha256(destination / "masks-provenance.json"))
        identity = {"schema_version": "body_input.v1", "profile": gate.PROFILE, "episode_index": episode,
                    "target_id": "human_0", "source": {"repository": raw.REPOSITORY, "revision": raw.REVISION,
                    "video_path": row["video_path"], "video_sha256": video_sha, "timeline_sha256": row["timeline_sha256"]},
                    "producer": producer, "artifacts": {name: raw.sha256(destination / name) for name in sorted(gate.ARTIFACTS)}}
        write_json(destination / "input.json", identity)
        review = {key: receipt[key] for key in REVIEW_FIELDS}
        review.update(schema_version="body_input_review.v1",
                      reviewed_artifacts={name: raw.sha256(destination / name) for name in sorted(gate.REVIEW_ARTIFACTS)},
                      source_receipt_path="visual-review-receipt.json",
                      source_receipt_sha256=raw.sha256(destination / "visual-review-receipt.json"))
        write_json(destination / "review.json", review)
        checksums = "".join(f"{raw.sha256(path)}  {name}\n" for name, path in sorted(file_tree(destination).items()))
        (destination / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    report = gate.check_package(output, dataset_root, raw_root, source_inventory, selected)
    write_json(output / "assembly-checks.json", report)
    require(report["passed"], "assembled inputs failed the Body gate; inspect assembly-checks.json")
    return report


def byte_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(document: dict) -> bytes:
    return (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8")


def archive_inventory(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    result = {}
    folded = set()
    for info in archive.infolist():
        name = raw.safe_relative(info.filename)
        require(not info.is_dir(), "archive must contain files only")
        require(name not in result and name.casefold() not in folded, "duplicate/case-colliding ZIP member")
        require(not info.flag_bits & 1, "encrypted ZIP entries are not allowed")
        mode = info.external_attr >> 16
        require(not stat.S_ISLNK(mode) and (stat.S_IFMT(mode) in {0, stat.S_IFREG}), "nonregular ZIP entry")
        result[name] = info
        folded.add(name.casefold())
    return result


def verify_archive(archive_path: Path) -> dict:
    """Verify content before extraction. The outer ZIP SHA256 is separate evidence."""
    raw.reject_links(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive_inventory(archive)
        require({"bundle-manifest.json", "SHA256SUMS"} <= set(members), "archive lacks complete bundle inventory")
        require(members["bundle-manifest.json"].file_size < 32 * 1024 * 1024, "bundle inventory is unexpectedly large")
        manifest = json.loads(archive.read("bundle-manifest.json"), parse_constant=gate._reject_constant,
                              object_pairs_hook=gate._unique_object)
        require(isinstance(manifest, dict) and manifest.get("schema_version") == "body_handoff_bundle.v1",
                "unknown bundle manifest schema")
        require(manifest.get("all_30_body_inputs_verified") is True, "bundle is not an all-30 delivery")
        listed = {}
        for row in manifest.get("files", []):
            require(isinstance(row, dict), "invalid bundle inventory row")
            name = raw.safe_relative(row.get("path"))
            require(name not in listed and name not in {"bundle-manifest.json", "SHA256SUMS"}, "duplicate/reserved bundle inventory path")
            gate.digest(row.get("sha256"), name)
            require(type(row.get("bytes")) is int and row["bytes"] >= 0, "invalid bundle byte size")
            listed[name] = row
        require(set(members) == set(listed) | {"bundle-manifest.json", "SHA256SUMS"}, "ZIP file set differs from bundle inventory")
        actual_hashes = {}
        for name, info in members.items():
            checksum = hashlib.sha256()
            with archive.open(info) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    checksum.update(block)
            actual_hashes[name] = checksum.hexdigest()
            if name in listed:
                require(info.file_size == listed[name]["bytes"] and actual_hashes[name] == listed[name]["sha256"],
                        f"ZIP member hash/size mismatch: {name}")
        expected_lines = {name: sha for name, sha in actual_hashes.items() if name != "SHA256SUMS"}
        checksum_lines = {}
        for line in archive.read("SHA256SUMS").decode("utf-8").splitlines():
            parts = line.split("  ", 1)
            require(len(parts) == 2 and parts[1] not in checksum_lines, "invalid bundle checksum inventory")
            checksum_lines[parts[1]] = parts[0]
        require(checksum_lines == expected_lines, "bundle SHA256SUMS coverage/hash mismatch")
    return manifest


def extract_verified(archive_path: Path, destination: Path) -> dict:
    archive_sha = raw.sha256(archive_path)
    manifest = verify_archive(archive_path)
    fresh(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        for name, info in archive_inventory(archive).items():
            target = destination / name
            require(target.resolve().is_relative_to(destination.resolve()), "unsafe extraction destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            raw.reject_links(target)
            with archive.open(info) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
    expected = {row["path"]: row for row in manifest["files"]}
    require(set(file_tree(destination)) == set(expected) | {"SHA256SUMS", "bundle-manifest.json"}, "extracted file set mismatch")
    for name, row in expected.items():
        require((destination / name).stat().st_size == row["bytes"] and raw.sha256(destination / name) == row["sha256"],
                f"extracted byte mismatch: {name}")
    require(raw.sha256(archive_path) == archive_sha, "archive changed during extraction")
    return manifest


def run_receiver(extracted_root: Path, output: Path) -> dict:
    fresh(output)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-B", str(extracted_root / "runtime/check_body_handoff.py"),
               "--package-root", str(extracted_root / "body-inputs"),
               "--dataset-root", str(extracted_root / RAW_PREFIX),
               "--raw-preparation-root", str(extracted_root / PREPARED_PREFIX),
               "--source-inventory", str(extracted_root / "source/track1-source.json"), "--output", str(output)]
    completed = subprocess.run(command, env=env, check=False)
    require(completed.returncode == 0 and output.is_file(), "extracted receiver gate failed")
    report = gate.read_json(output)
    require(report.get("all_30_body_inputs_verified") is True and report.get("videos_verified") == 30,
            "receiver did not verify all 30 episodes")
    return report


def readme() -> bytes:
    return f"""# Track 1 Body input handoff

This archive contains the original 30 Track 1 videos, original timelines, full
human masks, per-episode estimated camera intrinsics, producer identities,
visual review receipts and portable receiver checks. It contains no Body model
weights or Body inference outputs. Camera estimates are not physical calibration.

Dataset attribution: NVIDIA Video to Data Challenge, Track 1,
https://huggingface.co/datasets/nvidia/video_to_data_challenge/tree/{raw.REVISION}/track_1
The pinned dataset is licensed under Creative Commons Attribution 4.0:
https://creativecommons.org/licenses/by/4.0/
Raw videos are unchanged. Human masks and camera estimates were generated by
the video-to-hoi project. No NVIDIA endorsement is implied. The dataset's
original README and metadata are preserved alongside the raw videos.

Use Python 3.12. From the extracted archive root:

```sh
python -m pip install -r requirements.txt
python runtime/check_body_handoff.py --package-root body-inputs --dataset-root {RAW_PREFIX} --raw-preparation-root {PREPARED_PREFIX} --source-inventory source/track1-source.json --output receiver-checks.json
```

The output path must be fresh. Proceed only when `all_30_body_inputs_verified`
is true. A diagnostic `--episodes` subset never certifies the full delivery.
Keep this folder layout; video paths in each input.json are relative to
`{RAW_PREFIX}`. The original raw manifest also resolves its relative dataset
dependency within this archive. File checksums cover all shipped evidence.

The selected standalone Body code revision is `{gate.BODY_SOURCE_REVISION}`.
For each episode, supply its original MP4, `human_masks` directory and
`intrinsics.json` to the pinned adapter. The receiving worker must separately
verify its Body runtime and licensed model weights before running inference.
See [receiving-guide.md](receiving-guide.md) for the exact Body invocation,
weight-directory layout and optional native-reader probe.
Input readiness is not model execution, reconstruction quality, or submission
compliance. Review identities and actual sample IDs are preserved in each
visual-review-receipt.json; machine-assisted review is named explicitly.

If supplied, requirements-native-readers.txt records the additional dependencies
for runtime/check_body_native_readers.py from an actual successful native-reader
report. That optional probe also requires the separately obtained pinned upstream
source checkout. The native probe loads readers only and does not run Body.
""".encode("utf-8")


def validate_native_report(report: dict, current_checks: dict) -> None:
    """Bind optional native-reader evidence to the current immutable inputs."""
    require(report.get("schema_version") == "body_native_reader_checks.v1"
            and report.get("all_30_native_readers_passed") is True, "native reader report is not an all-30 success")
    source = report.get("upstream_common_source")
    require(isinstance(source, dict) and source.get("repository") == "nvidia-isaac/video_to_data"
            and source.get("revision") == gate.BODY_SOURCE_REVISION, "native reader source pin mismatch")
    rows = report.get("episodes")
    require(isinstance(rows, list) and len(rows) == 30, "native reader report must cover all 30 episodes")
    expected = {row["episode_index"]: row for row in current_checks["episodes"]}
    seen = set()
    for row in rows:
        require(isinstance(row, dict) and type(row.get("episode_index")) is int
                and row["episode_index"] in range(30) and row["episode_index"] not in seen,
                "native reader episode coverage is invalid")
        index = row["episode_index"]
        seen.add(index)
        require(row.get("passed") is True, f"native readers did not pass episode {index}")
        require(type(row.get("frames_verified")) is int and row["frames_verified"] == expected[index]["frame_count"],
                f"native reader frame count is stale: episode {index}")
        require(row.get("input_sha256") == expected[index]["input_sha256"],
                f"native reader input identity is stale: episode {index}")
    require(type(report.get("frames_verified")) is int and report["frames_verified"] == current_checks["frames_verified"],
            "native reader frame total mismatch")


def package(assembled_root: Path, dataset_root: Path, raw_root: Path, source_inventory: Path,
            output: Path, extract_root: Path) -> dict:
    fresh(output)
    temporary = output.with_name(output.name + ".incomplete")
    fresh(temporary)
    fresh(extract_root)
    receipt_path = output.with_name(output.name + ".verification.json")
    receiver_path = output.with_name(output.name + ".receiver-checks.json")
    fresh(receipt_path)
    fresh(receiver_path)
    inputs = (assembled_root, dataset_root, raw_root)
    require(not any(output.resolve().is_relative_to(path.resolve()) or extract_root.resolve().is_relative_to(path.resolve())
                    or path.resolve().is_relative_to(extract_root.resolve()) for path in inputs), "archive/extraction must be separate from source inputs")
    manifest = assembly_context(raw_root, source_inventory)
    require(manifest.get("dataset_root_relative") == "../../v2d/track_1", "raw manifest layout is not the supported portable Track 1 layout")
    producer_checks = gate.check_package(assembled_root, dataset_root, raw_root, source_inventory)
    require(producer_checks["all_30_body_inputs_verified"], "all 30 inputs must pass before packaging")
    source_entries = raw.load_inventory(source_inventory)
    raw_paths = raw.validate_tree(dataset_root, source_entries)
    file_sources = {}
    for name in raw_paths:
        raw.verify_file(dataset_root / name, source_entries[name])
        file_sources[f"{RAW_PREFIX}/{name}"] = dataset_root / name
    expected_prepared = {"manifest.json", "checks.json", "SHA256SUMS"} | {f"timelines/episode_{i:06d}.json" for i in range(30)}
    prepared_files = file_tree(raw_root)
    require(set(prepared_files) == expected_prepared, "unexpected/missing raw preparation files")
    file_sources.update({f"{PREPARED_PREFIX}/{name}": path for name, path in prepared_files.items()})
    for episode in range(30):
        directory = assembled_root / f"episode_{episode:06d}"
        file_sources.update({f"body-inputs/{directory.name}/{name}": path for name, path in file_tree(directory).items()})
    native_packages = None
    for path in assembled_root.iterdir():
        raw.reject_links(path)
        if path.is_file():
            require(path.suffix.lower() in {".json", ".jpg", ".png", ".md"}, "unexpected root evidence file")
            file_sources[f"body-inputs/{path.name}"] = path
            if path.suffix.lower() == ".json":
                evidence = gate.read_json(path)
                if evidence.get("schema_version") == "body_native_reader_checks.v1":
                    validate_native_report(evidence, producer_checks)
                    native_packages = evidence.get("environment", {}).get("packages")
        else:
            require(path.name in {f"episode_{i:06d}" for i in range(30)}, "unexpected root evidence directory")
    file_sources["source/track1-source.json"] = source_inventory
    receiving_guide = Path(__file__).resolve().parents[1] / "docs/body-input-handoff.md"
    raw.reject_links(receiving_guide)
    require(receiving_guide.is_file(), "missing receiving guide")
    file_sources["receiving-guide.md"] = receiving_guide
    for name in ("check_body_handoff.py", "prepare_track1_data.py", "package_body_handoff.py"):
        file_sources[f"runtime/{name}"] = Path(__file__).with_name(name)
    native_reader = Path(__file__).with_name("check_body_native_readers.py")
    if native_reader.is_file():
        file_sources["runtime/check_body_native_readers.py"] = native_reader
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    generated = {"README.md": readme(), "requirements.txt": "".join(f"{name}=={version}\n" for name, version in versions.items()).encode(),
                 "producer-checks.json": json_bytes(producer_checks),
                 "runtime-environment.json": json_bytes({"python": platform.python_version(), "platform": platform.platform(), "packages": versions})}
    if native_packages:
        allowed = {"numpy", "Pillow", "scipy", "av", "imageio", "tqdm", "h5py", "opencv-python", "opencv-python-headless"}
        require(isinstance(native_packages, dict) and set(native_packages) <= allowed, "unexpected native-reader package names")
        for version in native_packages.values():
            require(isinstance(version, str) and version and all(c.isalnum() or c in ".+-_" for c in version),
                    "invalid native-reader dependency version")
        generated["requirements-native-readers.txt"] = (
            "# Additional packages from the verified native-reader environment.\n"
            + "".join(f"{name}=={version}\n" for name, version in sorted(native_packages.items()) if name not in PACKAGES)).encode()
    records = [{"path": name, "bytes": path.stat().st_size, "sha256": raw.sha256(path)} for name, path in sorted(file_sources.items())]
    records += [{"path": name, "bytes": len(content), "sha256": byte_sha256(content)} for name, content in sorted(generated.items())]
    bundle = {"schema_version": "body_handoff_bundle.v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "profile": gate.PROFILE, "episode_indices": list(range(30)), "all_30_body_inputs_verified": True,
              "body_execution_completed": False, "scientific_accuracy_certified": False,
              "files": sorted(records, key=lambda row: row["path"])}
    generated["bundle-manifest.json"] = json_bytes(bundle)
    sums = {row["path"]: row["sha256"] for row in records}
    sums["bundle-manifest.json"] = byte_sha256(generated["bundle-manifest.json"])
    generated["SHA256SUMS"] = "".join(f"{checksum}  {name}\n" for name, checksum in sorted(sums.items())).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, path in sorted(file_sources.items()):
            archive.write(path, name)
        for name, content in sorted(generated.items()):
            archive.writestr(name, content)
    extract_verified(temporary, extract_root)
    receiver = run_receiver(extract_root, receiver_path)
    # Run the receiver in the extracted copy. Keep both evidence and extracted
    # bytes available for review; no recursive cleanup is performed here.
    # Exclusive hard-link creation publishes the already verified file without
    # replacing an existing destination on either Windows or POSIX.
    os.link(temporary, output)
    temporary.unlink()
    receipt = {"schema_version": "body_bundle_verification.v1", "verified_at": datetime.now(timezone.utc).isoformat(),
               "archive_name": output.name, "bytes": output.stat().st_size, "sha256": raw.sha256(output),
               "archive_members_verified": len(records) + 2, "extracted_receiver_passed": True,
               "all_30_body_inputs_verified": receiver["all_30_body_inputs_verified"],
               "receiver_checks_sha256": raw.sha256(receiver_path), "body_execution_completed": False}
    write_json(receipt_path, receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    assembly = subcommands.add_parser("assemble", help="Consume explicit reviews and create fresh complete episode packages")
    assembly.add_argument("--inputs-root", type=Path, required=True)
    assembly.add_argument("--review-root", type=Path, required=True)
    assembly.add_argument("--episodes", type=int, nargs="+")
    shipping = subcommands.add_parser("package", help="Verify all 30, create ZIP, and rerun the extracted receiver")
    shipping.add_argument("--package-root", type=Path, required=True)
    shipping.add_argument("--verify-extract-root", type=Path, required=True)
    for command in (assembly, shipping):
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--raw-preparation-root", type=Path, required=True)
        command.add_argument("--source-inventory", type=Path, default=Path(__file__).resolve().parents[1] / "configs/track1-source.json")
        command.add_argument("--output", type=Path, required=True)
    verification = subcommands.add_parser("verify", help="Verify every member hash of an existing archive")
    verification.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "assemble":
            result = assemble(args.inputs_root, args.dataset_root, args.raw_preparation_root, args.review_root,
                              args.source_inventory, args.output, args.episodes)
        elif args.command == "package":
            result = package(args.package_root, args.dataset_root, args.raw_preparation_root, args.source_inventory,
                             args.output, args.verify_extract_root)
        else:
            result = verify_archive(args.archive)
        print(json.dumps({key: result[key] for key in ("schema_version", "all_30_body_inputs_verified")}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(f"Body packaging failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
