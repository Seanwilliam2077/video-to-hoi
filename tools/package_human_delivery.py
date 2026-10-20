"""Package completed standalone Track 1 Human candidates for review and upload.

Large raw meshes, model weights and original input assets stay outside the
portable review. Native parameters and their exact decoder identities remain
authoritative. This tool does not upload or accept reconstruction quality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import importlib.util
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import quote, unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
STAGES = {"inference", "baseline", "observations", "refined", "baseline_review", "refined_review", "hand_comparison", "checks"}
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def file_identity(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def contained(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root.resolve(strict=True)), f"Path escapes episode outputs: {path}")
    require(not path.is_symlink(), f"Symlink is not a portable result file: {path}")
    return resolved


def stage_receipt(episode_root: Path, stage: str, output: Path) -> tuple[dict, Path]:
    contained(output, episode_root)
    candidates = []
    for path in sorted((episode_root / stage).glob("attempt-*.finished.json")):
        receipt = read_json(path)
        if receipt.get("status") == "completed" and Path(receipt.get("output", "")).resolve() == output:
            candidates.append((receipt, path))
    require(bool(candidates), f"No completed attempt receipt for {stage}: {output}")
    receipt, path = candidates[-1]
    require(receipt.get("stage") == stage and receipt.get("returncode") == 0,
            f"Invalid successful attempt receipt for {stage}")
    return receipt, path


def verified_copy(source: Path, destination: Path, output: Path, receipt: dict) -> None:
    source = contained(source, output)
    relative = source.relative_to(output).as_posix()
    expected = receipt["outputs"].get(relative)
    require(isinstance(expected, dict), f"File absent from completed stage inventory: {source}")
    require(file_identity(source) == expected, f"Source result changed after stage completion: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), f"Refusing to overwrite portable file: {destination}")
    shutil.copy2(source, destination)
    require(file_identity(destination) == expected, f"Copy verification failed: {destination}")


def verified_identity(source: Path, output: Path, receipt: dict) -> dict:
    source = contained(source, output)
    expected = receipt["outputs"].get(source.relative_to(output).as_posix())
    require(isinstance(expected, dict) and file_identity(source) == expected,
            f"Source result changed after stage completion: {source}")
    return expected


def copy_review(source: Path, destination: Path, receipt: dict, *, frames: int | None = None) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            require(path.suffix.lower() in {".html", ".json", ".csv", ".jpg", ".jpeg", ".png", ".webp", ".mp4", ".log", ".txt"},
                    f"Unexpected file in visual review: {path}")
            verified_copy(path, destination / path.relative_to(source), source, receipt)
    require((destination / "index.html").is_file(), "Missing portable review page")
    require(any(p.stat().st_size > 0 for p in destination.glob("*.mp4")), "Missing comparison video")
    if frames is not None:
        expected = {f"{index:06d}.jpg" for index in range(frames)}
        actual = {p.name for p in (destination / "frames").iterdir() if p.is_file()}
        require(actual == expected, f"Review JPEGs do not cover all {frames} original frames")


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        fields = dict(attrs)
        for name in ("href", "src", "poster"):
            if name in fields and fields[name]:
                self.links.append(fields[name])


def verify_local_links(directory: Path) -> None:
    """Verify pages after relocation, including every copied viewer's own links."""
    root = directory.resolve()
    for page in directory.rglob("*.html"):
        parser = LocalLinks()
        parser.feed(page.read_text(encoding="utf-8"))
        for value in parser.links:
            parsed = urlsplit(value)
            if parsed.scheme in {"https", "http", "data", "mailto"} or value.startswith("#"):
                continue
            require(not parsed.scheme and not parsed.netloc, f"Unsafe link in portable page: {value}")
            target = (page.parent / unquote(parsed.path)).resolve()
            require(target.is_relative_to(root) and target.exists(), f"Broken portable page link: {page.name}: {value}")


def portable_index(directory: Path, episode: int, frames: int, selected: str, reason: str) -> None:
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Track 1 episode {episode:02d} Human review</title><style>body{{font:16px/1.6 system-ui;max-width:1100px;margin:24px auto;padding:0 18px;background:#10151d;color:#e4eaf3}}a{{color:#9cc5ff}}video{{width:100%;height:auto;background:#000}}section{{margin:32px 0}}</style><h1>Track 1 episode {episode:02d}: Human reconstruction candidates</h1><p>All {frames} original frames. Selected local candidate: <strong>{html.escape(selected)}</strong>. {html.escape(reason)}</p><p>Native structure was checked. Reconstruction quality, true 3D hand contact, metric/shared scene scale and official export remain pending. These image diagnostics are not official scores.</p><nav><a href="checks.json">Native and hand checks</a> · <a href="depth-scale.json">Scale limitations</a> · <a href="selection.json">Exact candidate selection</a> · <a href="parameters/selectedhuman.npz">Selected native parameters</a> · <a href="README.md">Package contents and decoder</a></nav><section><h2>Hand close-ups: same source frame and same crop</h2><video controls preload="metadata" src="reviews/hands/hands-comparison.mp4"></video><p><a href="reviews/hands/index.html">Open hand snapshots and observation evidence</a></p></section><section><h2>Fixed identity and skeleton baseline</h2><video controls preload="metadata" src="reviews/baseline/comparison.mp4"></video><p><a href="reviews/baseline/index.html">Inspect every original frame</a></p></section><section><h2>Independent MediaPipe hand candidate</h2><video controls preload="metadata" src="reviews/refined/comparison.mp4"></video><p><a href="reviews/refined/index.html">Inspect every original frame</a></p></section><p>Green hand points are independent accepted 2D observations. A missing hand observation means uncertainty. Cyan geometry is a prediction; a close overlay does not prove correct depth or physical contact. Read each side's held-out evidence and the abstention receipt where present.</p></html>'''
    (directory / "index.html").write_text(page, encoding="utf-8")


def verify_zip(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as stream:
        require(stream.testzip() is None, "ZIP CRC verification failed")
        names = stream.namelist()
        require(len(set(names)) == len(names), "Duplicate ZIP entries")
        require(all(not Path(name).is_absolute() and ".." not in Path(name).parts and "\\" not in name for name in names),
                "Unsafe ZIP entry")
        checksums = {}
        for line in stream.read("SHA256SUMS").decode("utf-8").splitlines():
            expected, name = line.split("  ", 1)
            require(SHA.fullmatch(expected) and name not in checksums, "Malformed ZIP checksum inventory")
            checksums[name] = expected
        require(set(names) == set(checksums) | {"SHA256SUMS"}, "ZIP checksum coverage mismatch")
        for name, expected in checksums.items():
            value = hashlib.sha256()
            with stream.open(name) as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    value.update(block)
            require(value.hexdigest() == expected, f"ZIP content hash differs: {name}")
    return {"crc_verified": True, "all_member_sha256_verified": True, "files": len(names)}


def package_episode(batch_root: Path, episode: int, output: Path, delivery_id: str, *, source_commit: str = "") -> dict:
    """Create a fresh archive; never mutate the inference or previous delivery."""
    import numpy as np
    from v2hoi.human_result import load_human_result, validate_human_result

    episode_root = batch_root / f"episode_{episode:06d}"
    current = read_json(episode_root / "current-result.json")
    require(current.get("episode") == episode and current.get("status") == "completed_candidate", "Episode is not a completed batch candidate")
    require(set(current.get("outputs", {})) == STAGES, "Completed batch result has missing or unknown stages")
    outputs = {name: contained(Path(value), episode_root) for name, value in current["outputs"].items()}
    receipts = {name: stage_receipt(episode_root, name, path) for name, path in outputs.items()}
    checks = read_json(outputs["checks"] / "checks.json")
    manifest = read_json(outputs["inference"] / "manifest.json")
    verified_identity(outputs["checks"] / "checks.json", outputs["checks"], receipts["checks"][0])
    verified_identity(outputs["inference"] / "manifest.json", outputs["inference"], receipts["inference"][0])
    require(checks.get("schema_version") == "human_batch_episode_checks.v1" and checks.get("track") == 1
            and checks.get("episode") == episode and checks.get("structural_checks_passed") is True,
            "A passing real native-structure check is required")
    count = checks["frames"]
    require(type(count) is int and count > 0 and count == current["frames"] == manifest["video_frames"], "Frame coverage mismatch")
    require(manifest.get("track") == 1 and manifest.get("episode") == episode
            and manifest.get("profile") == "sam3d_body_masks_camera.v1", "Wrong input policy/profile")
    require("/track_1/" in manifest.get("video_url", "") and manifest["video_url"].endswith(f"episode_{episode:06d}.mp4"),
            "Input URL does not identify the explicit Track 1 episode")
    require(checks["source_video_sha256"] == manifest["video_sha256"] and checks["mhr_model_sha256"] == manifest["mhr_model_sha256"],
            "Checks do not bind the actual model and Track 1 video")
    selected = checks["selected_candidate"]
    require(selected in {"baseline", "refined"} and selected == current["selected_candidate"], "Candidate selection mismatch")
    target = output / f"{delivery_id}-ep{episode:02d}"
    target.mkdir(parents=True, exist_ok=False)
    (target / "parameters").mkdir()
    parameters = f"episode_{episode:06d}_human.npz"
    for name in ("baseline", "refined"):
        result = load_human_result(outputs[name] / parameters)
        validate_human_result(result, count)
        native_report = result["report"]
        require(native_report.get("video_sha256", native_report.get("source_video_sha256")) == manifest["video_sha256"]
                and native_report.get("mhr_model_sha256") == manifest["mhr_model_sha256"], "Native parameters have different source/model provenance")
        require(np.array_equal(result["frame_indices"], np.arange(count)), "Native parameters omit original frames")
        expected_sha = checks["native_results"][name]["parameters_sha256"]
        require(digest(outputs[name] / parameters) == expected_sha, "Native parameters changed since decode check")
        verified_copy(outputs[name] / parameters, target / "parameters" / f"{name}.npz", outputs[name], receipts[name][0])
        verified_copy(outputs[name] / "report.json", target / "parameters" / f"{name}-report.json", outputs[name], receipts[name][0])
        abstention = outputs[name] / "hand-refinement-abstention.json"
        if abstention.is_file():
            verified_copy(abstention, target / "parameters" / abstention.name, outputs[name], receipts[name][0])
        snapshot = outputs[name] / "source_snapshot"
        if snapshot.is_dir():
            for path in sorted(snapshot.rglob("*.py")):
                verified_copy(path, target / "source_snapshot" / name / path.relative_to(snapshot), outputs[name], receipts[name][0])
    source_selected = target / "parameters" / f"{selected}.npz"
    shutil.copy2(source_selected, target / "parameters/selectedhuman.npz")
    selection = {"schema_version": "human_delivery_selection.v1", "episode": episode, "frames": count,
                 "selected_candidate": selected, "selection_reason": checks["selection_reason"],
                 "selected_native_parameters": "parameters/selectedhuman.npz", **file_identity(source_selected),
                 "selected_is_exact_candidate_copy": digest(target / "parameters/selectedhuman.npz") == digest(source_selected),
                 "quality_accepted": False, "hand_contact_verified": False}
    write_json(target / "selection.json", selection)
    verified_copy(outputs["checks"] / "checks.json", target / "checks.json", outputs["checks"], receipts["checks"][0])
    obs = outputs["observations"]
    for path in sorted(obs.rglob("*")):
        if path.is_file():
            require(path.suffix.lower() in {".npz", ".json", ".jpg", ".png"}, f"Unexpected independent observation artifact: {path}")
            verified_copy(path, target / "observations" / path.relative_to(obs), obs, receipts["observations"][0])
    for name, label in (("baseline_review", "baseline"), ("refined_review", "refined"), ("hand_comparison", "hands")):
        copy_review(outputs[name], target / "reviews" / label, receipts[name][0], frames=count if name != "hand_comparison" else None)
    for stage in sorted(STAGES):
        for path in sorted((episode_root / stage).glob("attempt-*")):
            if path.is_file() and (path.name.endswith(".json") or path.name.endswith(".log")):
                contained(path, episode_root)
                _copy_new(path, target / "attempts" / stage / path.name)
    environments = []
    for path in sorted(batch_root.glob("runtime-*.json")):
        contained(path, batch_root)
        runtime = read_json(path)
        compact = {"recorded_at": runtime.get("recorded_at"), "asset_identities": runtime.get("asset_identities"),
                   "environments": runtime.get("environments"), "original_receipt_sha256": digest(path)}
        if "execution_sources" in runtime:
            compact["execution_sources"] = runtime["execution_sources"]
        write_json(target / "environment" / path.name, compact)
        environments.append({"path": f"environment/{path.name}", "original_receipt_sha256": digest(path)})
    input_binding = {key: manifest[key] for key in ("track", "episode", "video_url", "video_sha256", "video_width", "video_height", "video_fps", "video_frames", "profile",
                     "checkpoint_sha256", "mhr_model_sha256", "config_sha256", "runtime_source_manifest_sha256", "camera_intrinsics", "intrinsics_status",
                     "receiver_checks_sha256", "input_inventory_sha256", "input_artifact_sha256", "target_id", "consumer_code_sha256") if key in manifest}
    input_binding.update({"inference_manifest_sha256": digest(outputs["inference"] / "manifest.json"),
                          "current_result_sha256": digest(episode_root / "current-result.json"),
                          "source_commit": source_commit, "runtime_receipts": environments,
                          "selected_successful_attempt_receipts": {name: {"file": f"attempts/{name}/{path.name}", "sha256": digest(path)} for name, (_, path) in receipts.items()}})
    write_json(target / "input-bindings.json", input_binding)
    depth = {"schema_version": "human_depth_scale_evidence.v1", "track": 1, "episode": episode,
             "source_video_sha256": manifest["video_sha256"], "status": "unverified_monocular_scale",
             "camera_intrinsics": manifest["camera_intrinsics"], "intrinsics_status": manifest["intrinsics_status"],
             "intrinsics_source_sha256": manifest.get("input_artifact_sha256", {}).get("intrinsics.json"),
             "native_geometry_unit": "metres after the documented MHR decoder coordinate/unit conversion",
             "native_root_parameter_unit_metres": .1, "metric_scale_verified": False, "shared_human_object_scale_verified": False,
             "scene_alignment_performed": False, "reference_alignment_available": False,
             "limits": "Monocular camera/human depth and learned human size are estimates. No object-based scale measurement or shared scene-scale verification is delivered. No arbitrary scale factor or hidden-reference alignment is supplied."}
    write_json(target / "depth-scale.json", depth)
    portable_index(target, episode, count, selected, checks["selection_reason"])
    readme = f"""# Track 1 episode {episode:02d} portable Human review

Delivery: `{delivery_id}`. Coverage: {count} consecutive original frames, starting
at frame 0. Open `index.html` locally or through a static HTTP server. All view
links are relative; no model server or network request is needed for review.

`parameters/selectedhuman.npz` is an exact byte copy of the selected {selected}
candidate. `selection.json` identifies it. Both fixed-body baseline and independent
MediaPipe hand candidate remain available, including an explicit abstention
receipt if independent hand evidence could not support optimization. The files
store true native MHR pose136, one scales68 vector and one shape45 vector.

The completed native checks, observed-side image diagnostics, and source/runtime
identities are in `checks.json`, `input-bindings.json`, `attempts/`, `environment/`
and the native reports. Attempt logs retain original execution paths as evidence;
those paths are not required to play the packaged videos or viewers.

Large raw meshes, per-frame SAM caches, model/checkpoint weights, source videos,
input masks and other original input assets are omitted. The full video/image
comparisons and independent 2D observations are included. Native geometry can be
regenerated from the native parameters with the exact MHR model whose SHA256 is
`{manifest['mhr_model_sha256']}` and the coordinate/unit conventions recorded in
the native reports. Obtain permitted input/model dependencies separately; their
identities are retained. Omission does not replace those dependencies for reruns.

Execution/native structure has passed the included checks. Reconstruction quality,
true 3D hand contact, metric/shared human-object scale, official scene export and
receiver acceptance remain pending. `depth-scale.json` explicitly records the
unverified monocular scale. No hidden-reference competition score is available.
The scene's required first-reference-frame alignment is not invented or fitted.

The package contains only results from the explicitly bound Track 1 source and
generic model observations. It supplies no Track 2 data or derived references.
"""
    (target / "README.md").write_text(readme, encoding="utf-8")
    verify_local_links(target)
    files = {p.relative_to(target).as_posix(): file_identity(p) for p in sorted(target.rglob("*")) if p.is_file()}
    write_json(target / "file-manifest.json", {"schema_version": "human_delivery_files.v1", "delivery_id": delivery_id,
               "episode": episode, "files": files, "scope": "Portable review plus authoritative native Human candidates; not a complete HOI submission"})
    all_paths = sorted(p for p in target.rglob("*") if p.is_file())
    (target / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.relative_to(target).as_posix()}\n" for path in all_paths), encoding="utf-8")
    archive = output / f"{target.name}.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as stream:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                stream.write(path, path.relative_to(target).as_posix())
    verification = verify_zip(archive)
    assets = []
    for source, suffix in ((target / "depth-scale.json", "depth-scale.json"), (target / "checks.json", "checks.json"),
                           (target / "selection.json", "selection.json"), (target / "input-bindings.json", "input-bindings.json")):
        destination = output / f"{target.name}-{suffix}"
        _copy_new(source, destination)
        assets.append({"path": str(destination), "name": destination.name, **file_identity(destination)})
    descriptor = {"schema_version": "human_delivery_package.v1", "delivery_id": delivery_id, "episode": episode,
                  "frames": count, "selected_candidate": selected, "source_commit": source_commit,
                  "source_video_sha256": manifest["video_sha256"], "archive": {"path": str(archive), "name": archive.name, **file_identity(archive)},
                  "input_intrinsics_sha256": manifest.get("input_artifact_sha256", {}).get("intrinsics.json"),
                  "zip_verification": verification, "assets": assets, "structural_check": "pass",
                  "quality_review": "pending", "depth_scale": "unverified_monocular_scale", "official_export": "not_run"}
    write_json(output / f"{target.name}-package.json", descriptor)
    return descriptor


def _copy_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), f"Refusing to overwrite artifact: {destination}")
    shutil.copy2(source, destination)
    require(file_identity(source) == file_identity(destination), f"Artifact copy differs: {source}")


def board_module():
    spec = importlib.util.spec_from_file_location("human_delivery_board", ROOT / "tools/progress_board.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def published_records(packages: list[dict], base_url: str, source_commit: str, delivery_id: str,
                      upstream_directory: Path, contributor: str) -> dict:
    """Prepare submitted declarations for explicit real asset URLs; never upload."""
    board = board_module()
    board.url(base_url, "release base URL", nonempty=True)
    parsed = urlsplit(base_url)
    require(not parsed.query and not parsed.fragment, "Release base URL must not contain a query or fragment")
    require(bool(COMMIT.fullmatch(source_commit)), "Publishing records requires a full lowercase source commit SHA")
    records = []
    for package in packages:
        episode = package["episode"]
        upstream_path = upstream_directory / f"body-inputs-20261016-ep{episode:02d}.json"
        upstream = read_json(upstream_path)
        require(upstream.get("id") == f"body-inputs-20261016-ep{episode:02d}" and upstream.get("stage") == "inputs"
                and upstream.get("episode_id") == f"episode_{episode:06d}" and upstream.get("kind") == "real_track1",
                "Missing or incorrect actual Body-preparation upstream record")
        cameras = [artifact for artifact in upstream.get("artifacts", []) if artifact.get("requirement_id") == "camera"]
        require(len(cameras) == 1 and cameras[0].get("sha256") == package["input_intrinsics_sha256"],
                "Dashboard upstream camera record does not bind the consumed intrinsics")
        assets = {asset["name"].rsplit(f"-ep{episode:02d}-", 1)[-1]: asset for asset in package["assets"]}
        url = lambda name: base_url.rstrip("/") + "/" + quote(name)
        human = package["archive"]
        depth, check = assets["depth-scale.json"], assets["checks.json"]
        records.append({"id": f"{delivery_id}-ep{episode:02d}", "episode_id": f"episode_{episode:06d}",
            "stage": "human", "iteration": delivery_id, "state": "submitted", "kind": "real_track1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "contributor": contributor,
            "summary": f"{package['frames']} original frames reconstructed with fixed native MHR and independent MediaPipe hand observations; local candidate {package['selected_candidate']}. Structural checks passed; quality, scene scale and official export remain pending.",
            "delivery_url": url(human["name"]), "source_commit": source_commit,
            "upstream": [{"stage": "inputs", "record_id": upstream["id"]}],
            "artifacts": [{"requirement_id": "human", "label": "Portable review and authoritative native Human parameters", "url": url(human["name"]), "sha256": human["sha256"], "bytes": human["bytes"]},
                          {"requirement_id": "depth_scale", "label": "Explicit unverified monocular-scale record; shared scene scale pending", "url": url(depth["name"]), "sha256": depth["sha256"], "bytes": depth["bytes"]}],
            "checks": {"structural": {"status": "pass", "evidence_url": url(check["name"])},
                       "quality": {"status": "not_run", "evidence_url": ""}, "export": {"status": "not_run", "evidence_url": ""}},
            "coverage": {"frames": package["frames"], "objects_complete": None}, "reviewer": "",
            "blocker": "Quality/receiver review, shared scene-scale verification and complete upstream Inputs acceptance remain pending.",
            "next_action": "Verify published package bytes and inspect complete videos before quality acceptance; retain submitted state while dependencies remain pending."})
        board.validate_record(records[-1], {f"episode_{episode:06d}": {"frames": package["frames"]}})
    return {"schema_version": "hoi_progress.v1", "records": records}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh package directory; never overwrites prior packages")
    parser.add_argument("--delivery-id", required=True)
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--release-base-url", help="Actual HTTP(S) asset-download base; emits submitted records only when supplied")
    parser.add_argument("--upstream-records", type=Path, default=ROOT / "tracking/records")
    parser.add_argument("--contributor", default="Codex on the owner's local workstation")
    args = parser.parse_args(argv)
    try:
        require(bool(SAFE_ID.fullmatch(args.delivery_id)) and not args.delivery_id.endswith("."), "Unsafe delivery ID")
        require(len(set(args.episodes)) == len(args.episodes) and all(0 <= i < 30 for i in args.episodes), "Unique Track 1 episode IDs 0..29 are required")
        require(not args.output.exists(), "Package output already exists; choose a unique fresh delivery directory")
        require(not args.source_commit or COMMIT.fullmatch(args.source_commit), "Source commit must be a full lowercase 40-character SHA")
        if args.release_base_url:
            board_module().url(args.release_base_url, "release base URL", nonempty=True)
            require(bool(COMMIT.fullmatch(args.source_commit)), "Release records require --source-commit")
        args.output.mkdir(parents=True)
        packages = [package_episode(args.batch_root.resolve(), episode, args.output.resolve(), args.delivery_id,
                                     source_commit=args.source_commit) for episode in args.episodes]
        uploads = []
        for package in packages:
            for asset in [package["archive"], *package["assets"]]:
                uploads.append({**asset, "content_type": mimetypes.guess_type(asset["name"])[0] or "application/octet-stream",
                                "intended_url": args.release_base_url.rstrip("/") + "/" + quote(asset["name"]) if args.release_base_url else None,
                                "upload_status": "not_uploaded"})
            descriptor = args.output / f"{args.delivery_id}-ep{package['episode']:02d}-package.json"
            uploads.append({"path": str(descriptor.resolve()), "name": descriptor.name, **file_identity(descriptor),
                            "content_type": "application/json",
                            "intended_url": args.release_base_url.rstrip("/") + "/" + quote(descriptor.name) if args.release_base_url else None,
                            "upload_status": "not_uploaded"})
        write_json(args.output / "upload-manifest.json", {"schema_version": "human_delivery_upload.v1", "delivery_id": args.delivery_id,
                   "source_commit": args.source_commit, "assets": uploads, "upload_performed": False,
                   "dashboard_records_generated": bool(args.release_base_url)})
        if args.release_base_url:
            envelope = published_records(packages, args.release_base_url, args.source_commit, args.delivery_id,
                                         args.upstream_records, args.contributor)
            write_json(args.output / "dashboard-submitted-records.json", envelope)
        write_json(args.output / "packages.json", {"schema_version": "human_delivery_set.v1", "delivery_id": args.delivery_id,
                   "packages": packages, "quality_accepted": False, "uploaded": False})
        print(json.dumps({"output": str(args.output.resolve()), "packages": len(packages), "files_to_upload": len(uploads),
                          "dashboard_records_generated": bool(args.release_base_url), "uploaded": False}))
        return 0
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Human delivery packaging failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
