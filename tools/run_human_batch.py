"""Run resumable, provenance-bound Track 1 Human reconstruction episodes.

This is a standalone real-model runner, not the legacy fake backend or an
official submission exporter. The dashboard handoff stays a local draft until
real artifacts have a published, receiver-accessible location.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DRIVER = Path(__file__).resolve()


def verified_stage_snapshot(root: Path) -> dict:
    """Require exact, hash-verified stage source, with no undeclared code files."""
    root = root.resolve(strict=True)
    manifest = root / "snapshot.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("Stage runtime root requires a regular snapshot.json")
    files = read_json(manifest).get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Stage snapshot must declare a nonempty file inventory")
    for name, expected in files.items():
        if (not isinstance(name, str) or "\\" in name or ":" in name
                or any(part in ("", ".", "..") for part in name.split("/"))
                or Path(name).is_absolute() or not name.startswith(("tools/", "src/"))
                or not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            raise ValueError(f"Unsafe stage snapshot file identity: {name}")
        path = root / name
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root)
                or digest(path) != expected):
            raise ValueError(f"Stage snapshot file changed or missing: {name}")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Stage snapshot contains a symlink: {path}")
        relative = path.relative_to(root)
        if path.is_file() and "__pycache__" not in relative.parts and relative.as_posix() != "snapshot.json":
            actual.add(relative.as_posix())
    if actual != set(files):
        raise ValueError("Stage snapshot inventory differs from actual files; undeclared files are not allowed")
    required = {"tools/run_sam3d_body.py", "tools/finalize_human.py", "tools/refine_human_hands.py",
                "tools/review_human.py", "tools/compare_human_hands.py", "tools/human_hand/observations.py",
                "tools/human_review/template.html", "src/v2hoi/__init__.py", "src/v2hoi/sam_body_inputs.py",
                "src/v2hoi/human_result.py", "src/v2hoi/hand_refinement.py"}
    if not required <= set(files):
        raise ValueError("Stage snapshot does not declare the required Human tools, modules and review template")
    return {"root": str(root), "snapshot_manifest_sha256": digest(manifest), "files_verified": len(files)}


def configure_stage_runtime(root: Path | None) -> dict:
    """Select source before any native module import; preserve the driver identity."""
    global ROOT
    driver_root = DRIVER.parents[1]
    driver = {"path": str(DRIVER), "sha256": digest(DRIVER), "snapshot_manifest_sha256": None}
    if (driver_root / "snapshot.json").is_file():
        driver["snapshot_manifest_sha256"] = verified_stage_snapshot(driver_root)["snapshot_manifest_sha256"]
    if root is not None:
        stage = verified_stage_snapshot(root)
        selected = Path(stage["root"])
    else:
        selected = driver_root
        stage = (verified_stage_snapshot(selected) if (selected / "snapshot.json").is_file()
                 else {"root": str(selected), "snapshot_manifest_sha256": None, "files_verified": None})
    source = selected / "src"
    for name, module in list(sys.modules.items()):
        if (name == "v2hoi" or name.startswith("v2hoi.")) and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
                raise ValueError("Stage runtime must be selected before importing v2hoi modules from another source tree")
    ROOT = selected
    sys.path[:] = [entry for entry in sys.path if entry not in {str(driver_root / "src"), str(source)}]
    sys.path.insert(0, str(source))
    return {"driver": driver, "stage_source": stage}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)


def inventory(directory: Path) -> dict:
    return {str(p.relative_to(directory)).replace("\\", "/"): {
        "bytes": p.stat().st_size, "sha256": digest(p)
    } for p in sorted(directory.rglob("*")) if p.is_file()}


def verify_inventory(directory: Path, expected: dict) -> None:
    actual = {str(p.relative_to(directory)).replace("\\", "/") for p in directory.rglob("*") if p.is_file()}
    if actual != set(expected):
        raise ValueError(f"Output inventory changed: {directory}")
    for name, identity in expected.items():
        path = directory / name
        if path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
            raise ValueError(f"Output bytes changed: {path}")


def source_identities(paths: list[Path]) -> dict:
    return {str(path.resolve()): digest(path) for path in sorted(set(paths))}


def runtime_identity(python: Path) -> dict:
    """Record installed distributions and executable identity without importing ML."""
    code = ("import importlib.metadata as m,json,platform,sys;"
            "print(json.dumps({'python':sys.version,'platform':platform.platform(),"
            "'distributions':sorted((d.metadata['Name'],d.version) for d in m.distributions())},sort_keys=True))")
    report = json.loads(subprocess.check_output([str(python), "-I", "-c", code], text=True))
    report["executable"] = str(python)
    report["executable_sha256"] = digest(python)
    report["fingerprint_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    return report


def verified_sam_source(source: Path) -> str:
    """A manifest hash alone cannot authorize mutated source at the same path."""
    manifest = source / "installed_manifest.json"
    for record in read_json(manifest)["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe installed runtime source path")
        path = source / relative
        if not path.is_file() or digest(path) != record["installed_sha256"]:
            raise ValueError(f"Runtime source differs from its installed manifest: {path}")
    return digest(manifest)


class StageError(RuntimeError):
    pass


def execute_stage(parent: Path, name: str, command_for_output, *, inputs: dict,
                  source_paths: list[Path], resume_inference: bool = False) -> Path:
    """Cache only hash-verified success; preserve separate attempts and logs.

    Inference alone may continue an interrupted per-frame cache. A previous
    failed receipt/log is never rewritten. Other tools require fresh output
    directories, so each retry gets a new attempt directory.
    """
    stage = parent / name
    stage.mkdir(parents=True, exist_ok=True)
    sources = source_identities(source_paths)
    signature_data = {"inputs": inputs, "sources": sources,
                      "command_template": list(map(str, command_for_output(Path("__OUTPUT__"))))}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    receipts = sorted(stage.glob("attempt-*.finished.json"))
    for path in reversed(receipts):
        receipt = read_json(path)
        if receipt["signature"] == signature and receipt["status"] == "completed":
            output = Path(receipt["output"])
            verify_inventory(output, receipt["outputs"])
            print(f"REUSE {parent.name}/{name}: {output}", flush=True)
            return output
    attempts = sorted(stage.glob("attempt-*.started.json"))
    attempt = len(attempts) + 1
    label = f"attempt-{attempt:04d}"
    output = stage / label
    if resume_inference:
        for path in reversed(attempts):
            previous = read_json(path)
            previous_output = Path(previous["output"])
            if previous["signature"] == signature and (previous_output / "manifest.json").is_file():
                output = previous_output
                break
    command = list(map(str, command_for_output(output)))
    started = {"schema_version": "human_batch_attempt.v1", "stage": name, "attempt": attempt,
               "signature": signature, **signature_data, "command": command, "output": str(output),
               "started_at": utcnow(), "executor": "Codex on the owner's local workstation",
               "cwd": str(ROOT), "pid": os.getpid()}
    write_json(stage / f"{label}.started.json", started, exclusive=True)
    log_path = stage / f"{label}.log"
    print(f"START {parent.name}/{name}: {' '.join(command)}", flush=True)
    begin = time.monotonic()
    returncode = None
    error = None
    try:
        with log_path.open("xb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            try:
                returncode = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if returncode:
            raise StageError(f"{name} exited {returncode}; see {log_path}")
        if source_identities(source_paths) != sources:
            raise StageError("Source code changed during stage execution; preserve this attempt and rerun")
        if not output.is_dir():
            raise StageError(f"{name} did not create its output directory")
        outputs = inventory(output)
        if not outputs:
            raise StageError(f"{name} produced no files")
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        raise
    finally:
        finished = {**started, "finished_at": utcnow(), "wall_seconds": round(time.monotonic() - begin, 3),
                    "returncode": returncode, "status": "failed" if error else "completed",
                    "error": error, "log": str(log_path), "log_sha256": digest(log_path) if log_path.exists() else None,
                    "outputs": outputs if not error else {}}
        write_json(stage / f"{label}.finished.json", finished, exclusive=True)
    print(f"DONE {parent.name}/{name}: {finished['wall_seconds']} seconds", flush=True)
    return output


def load_verified_episode(root: Path, gate_path: Path, episode: int) -> dict:
    """Bind each consumer to the receiver gate and exact Track 1 inputs again."""
    gate = read_json(gate_path)
    if (gate.get("schema_version") != "body_handoff_checks.v1" or not gate.get("passed")
            or gate.get("all_30_body_inputs_verified") is not True or gate.get("videos_verified") != 30):
        raise ValueError("A successful all-30 receiver gate is required, not a subset or producer assertion")
    rows = {r["episode_index"]: r for r in gate["episodes"]}
    if set(rows) != set(range(30)) or any(r.get("body_inputs_verified") is not True for r in rows.values()):
        raise ValueError("Receiver gate does not verify exactly all 30 episodes")
    directory = root / "body-inputs" / f"episode_{episode:06d}"
    identity = read_json(directory / "input.json")
    row = rows[episode]
    if identity.get("schema_version") != "body_input.v1" or identity.get("episode_index") != episode:
        raise ValueError("Wrong episode or unsupported input schema")
    if digest(directory / "input.json") != row["input_sha256"]:
        raise ValueError("Input identity changed after receiver gate")
    source = identity["source"]
    relative = Path(source["video_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe video path")
    video = root / "data/v2d/track_1" / relative
    if digest(video) != source["video_sha256"] or source["video_sha256"] != row["video_sha256"]:
        raise ValueError("Source video changed after receiver gate")
    for name, expected in identity["artifacts"].items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError(f"Input artifact changed after receiver gate: {name}")
    return {"directory": directory, "video": video, "identity": identity,
            "frames": row["frame_count"], "input_sha256": row["input_sha256"],
            "video_sha256": row["video_sha256"], "receiver_sha256": digest(gate_path)}


def refine_or_preserve(baseline: Path, observations: Path, output: Path, command: list[str]) -> None:
    """Abstain honestly if the independent observer cannot support hand fitting."""
    import numpy as np
    from v2hoi.hand_refinement import HandFitConfig, supported_frames
    with np.load(observations, allow_pickle=False) as archive:
        confidence = archive["confidence"].copy()
    config = HandFitConfig()
    # Match the refiner's training split; fitting cannot borrow held-out frames.
    confidence[np.arange(len(confidence)) % config.holdout_stride == 0] = 0
    if supported_frames(confidence, config).any():
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise StageError(f"Native hand refiner exited {result.returncode}")
        return
    output.mkdir(parents=True)
    shutil.copy2(baseline, output / baseline.name)
    shutil.copytree(baseline.parent / "review_arrays", output / "review_arrays")
    report = read_json(baseline.parent / "report.json")
    report.update({"hand_refinement_status": "abstained_insufficient_independent_observations",
                   "baseline_preserved_unchanged": True, "baseline_sha256": digest(baseline),
                   "observations_sha256": digest(observations),
                   "reason": "No sufficiently supported training-frame hand observations after holdout; no hand optimizer was run"})
    write_json(output / "report.json", report, exclusive=True)
    write_json(output / "hand-refinement-abstention.json", {"status": report["hand_refinement_status"],
               "baseline_preserved_unchanged": True, "no_hand_optimizer_run": True,
               "baseline_sha256": digest(baseline), "observations_sha256": digest(observations), "at": utcnow()}, exclusive=True)


def hand_candidate_guards(report: dict) -> dict:
    """Sparse evidence cannot prove improvement, but can still veto regression."""
    guards = {}
    abstained = report.get("hand_refinement_status") == "abstained_insufficient_independent_observations"
    for side in ("left", "right"):
        if abstained:
            guards[side] = {"heldout_observed_frames": None, "sufficient": False,
                            "before_px": None, "after_px": None, "nonregression": None,
                            "reason": "Independent evidence insufficient; optimizer abstained"}
            continue
        before, after = report["before_heldout"][side], report["after_heldout"][side]
        count, after_count = before["observed_frames"], after["observed_frames"]
        if type(count) is not int or type(after_count) is not int or count < 0 or count != after_count:
            raise ValueError(f"{side}: invalid or inconsistent held-out observation counts")
        pre, post = before["weighted_reprojection_pixels"], after["weighted_reprojection_pixels"]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (pre, post)):
            raise ValueError(f"{side}: held-out image metrics must be finite and nonnegative")
        guards[side] = {"heldout_observed_frames": count, "sufficient": count >= 5,
                        "before_px": pre, "after_px": post,
                        "nonregression": bool(post <= pre * 1.05 + .25) if count > 0 else None}
    usable = any(value["sufficient"] for value in guards.values()) and not any(value["nonregression"] is False for value in guards.values())
    return {"per_side": guards, "candidate_usable": usable,
            "selected_candidate": "refined" if usable else "baseline",
            "selection_reason": ("Sufficient held-out evidence on at least one side and no observed-side image regression"
                                 if usable else "Insufficient held-out evidence or image regression on an observed side; preserve fixed-body baseline")}


def check_episode_results(inference: Path, baseline: Path, refined: Path, observations: Path,
                          model: Path, output: Path, *, device: str = "cpu") -> dict:
    """Check true native decode, original frame mapping, and honest quality gates."""
    import numpy as np
    from v2hoi.human_result import load_human_result, native_mhr_forward, validate_human_result

    manifest = read_json(inference / "manifest.json")
    count = manifest["video_frames"]
    expected = np.arange(count)
    results = {}
    model_hash = digest(model)
    for label, directory in (("baseline", baseline), ("refined", refined)):
        path = directory / f"episode_{manifest['episode']:06d}_human.npz"
        result = load_human_result(path)
        structural = validate_human_result(result, count)
        if not np.array_equal(result["frame_indices"], expected):
            raise ValueError(f"{label}: missing original frame IDs")
        report = result["report"]
        if report.get("video_sha256", report.get("source_video_sha256")) != manifest["video_sha256"]:
            raise ValueError(f"{label}: wrong source video identity")
        if report.get("mhr_model_sha256") != model_hash:
            raise ValueError(f"{label}: native model identity mismatch")
        review = directory / "review_arrays"
        for field in ("frame_indices", "camera_intrinsics", "fps"):
            if not np.array_equal(np.load(review / f"{field}.npy", allow_pickle=False),
                                  np.load(inference / "raw" / f"{field}.npy", allow_pickle=False)):
                raise ValueError(f"{label}: changed {field}")
        vertices = np.load(review / "vertices_m.npy", mmap_mode="r", allow_pickle=False)
        if vertices.shape != (count, 18439, 3):
            raise ValueError(f"{label}: unexpected native geometry shape")
        # Scan all saved geometry for validity, decode a deterministic 4-frame sample.
        for start in range(0, count, 32):
            if not np.isfinite(vertices[start:start + 32]).all():
                raise ValueError(f"{label}: nonfinite native geometry")
        sample = np.unique([0, count // 3, 2 * count // 3, count - 1])
        sampled = {"pose": result["pose"][sample], "scales": result["scales"], "shape": result["shape"]}
        decoded = native_mhr_forward(sampled, model, device=device, batch_size=1)["vertices_m"]
        difference = float(np.max(np.abs(decoded - vertices[sample])))
        if difference > 1e-5:
            raise ValueError(f"{label}: stored mesh disagrees with native decoder by {difference} metres")
        results[label] = {**structural, "parameters_sha256": digest(path), "native_decode_max_abs_metres": difference,
                          "native_decode_sample_original_frame_ids": sample.tolist()}
    before = load_human_result(baseline / f"episode_{manifest['episode']:06d}_human.npz")
    after = load_human_result(refined / f"episode_{manifest['episode']:06d}_human.npz")
    for field in ("shape", "scales"):
        if not np.array_equal(before[field], after[field]):
            raise ValueError(f"Hand fitting changed fixed {field}")
    with np.load(observations / "observations.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["frame_indices"], expected):
            raise ValueError("Independent observations do not cover every original frame")
        observed = (archive["confidence"].sum(axis=2) > 0).sum(axis=0).tolist()
    report = read_json(refined / "report.json")
    decision = hand_candidate_guards(report)
    result = {"schema_version": "human_batch_episode_checks.v1", "track": 1, "episode": manifest["episode"],
              "frames": count, "source_video_sha256": manifest["video_sha256"], "mhr_model_sha256": model_hash,
              "structural_checks_passed": True, "native_results": results,
              "independent_observed_hand_frames_left_right": observed, "hand_heldout_nonregression": decision["per_side"],
              "selected_candidate": decision["selected_candidate"], "selection_reason": decision["selection_reason"],
              "reconstruction_quality_accepted": False, "quality_review": "pending human review; image diagnostics do not establish true 3D contact or hidden accuracy",
              "depth_scale": {"status": "unverified_monocular_scale", "camera": "per-episode Track 1 estimate", "shared_scene_scale_verified": False},
              "official_score_available": False, "official_export_accepted": False, "checked_at": utcnow()}
    output.mkdir(parents=True)
    write_json(output / "checks.json", result, exclusive=True)
    return result


def html_report(episode: int, outputs: dict[str, Path], checks: dict, destination: Path) -> None:
    """Portable relative links; no runtime file upload, tracking, or hidden scores."""
    destination.mkdir(parents=True)
    relative = lambda path: html.escape(os.path.relpath(path, destination).replace("\\", "/"), quote=True)
    checks_link = relative(outputs["checks"] / "checks.json")
    sections = []
    for name, label in (("baseline_review", "Fixed identity and skeleton baseline"), ("refined_review", "Independent MediaPipe hand fitting candidate")):
        review = outputs[name]
        # review_human normally produces comparison.mp4; discover actual output.
        videos = sorted(review.glob("*.mp4"))
        video = (f'<video controls preload="metadata" src="{relative(videos[0])}"></video>' if videos else "<p>No comparison video was produced.</p>")
        sections.append(f'<section><h2>{label}</h2>{video}<p><a href="{relative(review / "index.html")}">Open every-frame review</a></p></section>')
    hands = outputs["hand_comparison"]
    hand_section = f'<section><h2>Synchronized hand close-ups</h2><video controls preload="metadata" src="{relative(hands / "hands-comparison.mp4")}"></video><p><a href="{relative(hands / "index.html")}">Hand comparison and source-frame snapshots</a></p></section>'
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Track 1 episode {episode:02d} Human review</title><style>body{{font:16px system-ui;margin:24px auto;max-width:1100px;padding:0 16px;background:#10151d;color:#e4eaf3}}a{{color:#91bfff}}video{{width:100%;background:#000}}section{{margin:32px 0}}table{{border-collapse:collapse}}th,td{{border:1px solid #546070;padding:10px;text-align:left}}</style><h1>Track 1 episode {episode:02d}: Human candidates</h1><p>{checks['frames']} original frames. Selected local candidate: <strong>{checks['selected_candidate']}</strong>. {html.escape(checks['selection_reason'])}.</p><p>Execution and native structure are checked. Reconstruction quality, shared scene scale, object interaction, and official export remain pending. No hidden-reference challenge score is computed.</p><p><a href="{checks_link}">Native and hand checks</a></p>{hand_section}{''.join(sections)}<h2>Interpretation</h2><p>Compare the same source moment in the original and overlay. Check wrists, fingers, occlusion, body motion and any abrupt changes. A low image error does not prove correct 3D depth or physical contact. Missing MediaPipe evidence stays uncertain, rather than becoming a fitted target.</p></html>'''
    (destination / "index.html").write_text(content, encoding="utf-8")
    write_json(destination / "dashboard-local-draft.json", {
        "schema_version": "human_dashboard_local_draft.v1", "episode": episode,
        "state": "local_candidate_not_published", "report": "index.html",
        "checks": os.path.relpath(outputs["checks"] / "checks.json", destination).replace("\\", "/"),
        "structural_check": "pass", "quality_review": "pending", "depth_scale_review": "unverified",
        "dashboard_record_generated": False,
        "next_action": "Publish exact artifacts and depth-scale limitations with real URLs; register submitted Human evidence against the input chain. Do not mark accepted."})


def run_episode(args, episode: int) -> dict:
    item = load_verified_episode(args.input_root, args.receiver_checks, episode)
    # Revalidate actual mask pixels and the whole episode inventory even on a
    # cache hit: a receipt/manifest hash cannot authorize changed PNG bytes.
    import cv2
    from v2hoi.sam_body_inputs import load_body_inputs
    capture = cv2.VideoCapture(str(item["video"]))
    if not capture.isOpened():
        raise ValueError("Cannot open received source video")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    load_body_inputs(item["directory"], args.receiver_checks, video=item["video"], episode=episode,
                     width=width, height=height, frame_count=count, fps=fps, video_sha256=item["video_sha256"])
    parent = args.output / f"episode_{episode:06d}"
    parent.mkdir(parents=True, exist_ok=True)
    common = {"episode": episode, "video_sha256": item["video_sha256"], "input_sha256": item["input_sha256"],
              "receiver_sha256": item["receiver_sha256"], "model_sha256": args.identities["mhr_model"],
              "checkpoint_sha256": args.identities["checkpoint"], "model_config_sha256": args.identities["model_config"],
              "sam_runtime_source_manifest_sha256": args.identities["sam_source_manifest"],
              "sam_python_runtime_fingerprint": args.identities["sam_python_runtime"],
              "mediapipe_python_runtime_fingerprint": args.identities["mediapipe_python_runtime"]}
    tool = lambda name: ROOT / "tools" / name
    modules = [ROOT / "src/v2hoi/human_result.py"]
    runner = DRIVER
    driver_stage_args = (["--stage-runtime-root", args.stage_runtime_root]
                         if args.stage_runtime_root is not None else [])
    source_url = f"https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/{item['identity']['source']['revision']}/track_1/{item['identity']['source']['video_path']}"
    output = {}
    output["inference"] = execute_stage(parent, "inference", lambda out: [args.sam_python, tool("run_sam3d_body.py"),
        "--video", item["video"], "--source-url", source_url, "--episode", episode, "--output", out,
        "--checkpoint", args.checkpoint, "--mhr-path", args.mhr_model, "--source", args.sam_source,
        "--body-input-dir", item["directory"], "--receiver-checks", args.receiver_checks,
        "--device", args.inference_device, "--threads", args.inference_threads,
        "--cuda-cache-policy", args.cuda_cache_policy, "--resume"],
        inputs=common, source_paths=[tool("run_sam3d_body.py"), ROOT / "src/v2hoi/sam_body_inputs.py", *modules], resume_inference=True)
    inf = output["inference"]
    baseline_inputs = {**common, "inference_manifest_sha256": digest(inf / "manifest.json"),
                       "inference_completed_sha256": digest(inf / "raw/frame_indices.npy"), "converter_sha256": args.identities["converter"]}
    output["baseline"] = execute_stage(parent, "baseline", lambda out: [args.sam_python, tool("finalize_human.py"),
        "--run", inf, "--output", out, "--model", args.mhr_model, "--converter", args.converter,
        "--device", args.fit_device, "--method", "fit", "--threads", 1, "--pose-workers", args.pose_workers,
        "--iterations", args.fit_iterations, "--keyframes", args.identity_keyframes, "--model-batch", args.fit_model_batch],
        inputs=baseline_inputs, source_paths=[tool("finalize_human.py"), *modules])
    output["observations"] = execute_stage(parent, "observations", lambda out: [args.mediapipe_python,
        tool("human_hand/observations.py"), "--video", item["video"], "--sam-keypoints", inf / "raw/pred_keypoints_2d.npy",
        "--model", args.mediapipe_model, "--output", out, "--episode", episode, "--mirror-check"],
        inputs={**common, "sam_seeds_sha256": digest(inf / "raw/pred_keypoints_2d.npy"), "mediapipe_model_sha256": args.identities["mediapipe_model"]},
        source_paths=[tool("human_hand/observations.py")])
    base, obs = output["baseline"], output["observations"]
    parameters = f"episode_{episode:06d}_human.npz"
    output["refined"] = execute_stage(parent, "refined", lambda out: [args.sam_python, runner, "hands",
        *driver_stage_args,
        "--baseline", base / parameters, "--observations", obs / "observations.npz", "--output", out,
        "--", args.sam_python, tool("refine_human_hands.py"),
        "--baseline", base / parameters, "--observations", obs / "observations.npz", "--manifest", inf / "manifest.json",
        "--review-arrays", base / "review_arrays", "--output", out, "--model", args.mhr_model,
        "--device", args.hand_device, "--threads", 1, "--model-batch", args.hand_model_batch,
        "--iterations", args.hand_iterations, "--optimize-wrist-chain", "--unobserved-policy", "preserve"],
        inputs={**common, "baseline_sha256": digest(base / parameters), "observations_sha256": digest(obs / "observations.npz")},
        source_paths=[runner, tool("refine_human_hands.py"), ROOT / "src/v2hoi/hand_refinement.py", *modules])
    refined = output["refined"]
    for label in ("baseline", "refined"):
        arrays = output[label] / "review_arrays"
        output[f"{label}_review"] = execute_stage(parent, f"{label}_review", lambda out, arrays=arrays, label=label: [
            args.sam_python, tool("review_human.py"), "--video", item["video"], "--predictions", arrays,
            "--episode", episode, "--output", out, "--panel-width", args.panel_width,
            "--title", f"Track 1 episode {episode:02d}: {label}, all original frames"],
            inputs={**common, "parameters_sha256": digest(output[label] / parameters)}, source_paths=[tool("review_human.py"), *sorted(p for p in (ROOT / "tools/human_review").glob("*") if p.is_file())])
    output["hand_comparison"] = execute_stage(parent, "hand_comparison", lambda out: [args.sam_python,
        tool("compare_human_hands.py"), "--video", item["video"], "--observations", obs / "observations.npz",
        "--episode", episode, "--expected-frames", item["frames"], "--size", 256, "--mesh-only", "--output", out,
        "--variant", f"Fixed body={base / 'review_arrays'}", "--variant", f"MediaPipe hand candidate={refined / 'review_arrays'}"],
        inputs={**common, "baseline_sha256": digest(base / parameters), "refined_sha256": digest(refined / parameters),
                "observations_sha256": digest(obs / "observations.npz")},
        source_paths=[tool("compare_human_hands.py"), tool("review_human.py"), *sorted(p for p in (ROOT / "tools/human_review").glob("*") if p.is_file())])
    output["checks"] = execute_stage(parent, "checks", lambda out: [args.sam_python, runner, "check", "--inference", inf,
        "--baseline", base, "--refined", refined, "--observations", obs, "--model", args.mhr_model,
        "--output", out, "--device", args.fit_device, *driver_stage_args],
        inputs={**common, "baseline_sha256": digest(base / parameters), "refined_sha256": digest(refined / parameters),
                "observations_sha256": digest(obs / "observations.npz")}, source_paths=[runner, *modules])
    checks = read_json(output["checks"] / "checks.json")
    # Reports are derived navigation, not immutable scientific evidence.
    html_report(episode, output, checks, parent / "report")
    result = {"episode": episode, "status": "completed_candidate", "frames": item["frames"],
              "selected_candidate": checks["selected_candidate"], "quality_review": "pending",
              "outputs": {key: str(value) for key, value in output.items()}, "report": str(parent / "report/index.html")}
    write_json(parent / "current-result.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    hands = sub.add_parser("hands", help="Run hand fitting or explicitly preserve the baseline if observations are insufficient")
    hands.add_argument("--stage-runtime-root", type=Path, help="Verified frozen tools/src snapshot used by native stages")
    for name in ("baseline", "observations", "output"):
        hands.add_argument(f"--{name}", type=Path, required=True)
    hands.add_argument("command", nargs=argparse.REMAINDER)
    check = sub.add_parser("check", help="Validate one finished episode without hidden references")
    check.add_argument("--stage-runtime-root", type=Path, help="Verified frozen tools/src snapshot used by native stages")
    for name in ("inference", "baseline", "refined", "observations", "model", "output"):
        check.add_argument(f"--{name}", type=Path, required=True)
    check.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run = sub.add_parser("run", help="Run complete episodes, reusing verified immutable successes")
    run.add_argument("--stage-runtime-root", type=Path, help="Reuse unchanged external tools/modules from this verified snapshot; the current driver runs hands/checks")
    for name in ("input-root", "receiver-checks", "output", "sam-python", "mediapipe-python", "checkpoint", "mhr-model", "converter", "sam-source", "mediapipe-model"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--episodes", nargs="+", required=True, help="Original episode IDs, or all")
    run.add_argument("--inference-device", choices=("cpu", "cuda"), default="cuda")
    run.add_argument("--cuda-cache-policy", choices=("upstream", "retain"), default="upstream")
    run.add_argument("--fit-device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--hand-device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--inference-threads", type=int, default=8)
    run.add_argument("--pose-workers", type=int, default=4)
    run.add_argument("--fit-iterations", type=int, default=8)
    run.add_argument("--fit-model-batch", type=int, default=1)
    run.add_argument("--identity-keyframes", type=int, default=5)
    run.add_argument("--hand-iterations", type=int, default=180)
    run.add_argument("--hand-model-batch", type=int, default=8)
    run.add_argument("--panel-width", type=int, default=720)
    run.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    try:
        execution_sources = configure_stage_runtime(args.stage_runtime_root)
        if args.stage_runtime_root is not None:
            args.stage_runtime_root = Path(execution_sources["stage_source"]["root"])
        if args.operation == "hands":
            command = args.command[1:] if args.command and args.command[0] == "--" else args.command
            if not command:
                raise ValueError("Hand refinement command is required after --")
            refine_or_preserve(args.baseline, args.observations, args.output, command)
            return 0
        if args.operation == "check":
            check_episode_results(args.inference, args.baseline, args.refined, args.observations, args.model, args.output, device=args.device)
            return 0
        episodes = list(range(30)) if args.episodes == ["all"] else list(map(int, args.episodes))
        if len(set(episodes)) != len(episodes) or any(i not in range(30) for i in episodes):
            raise ValueError("Episodes must be unique Track 1 IDs 0..29, or all")
        for name in ("inference_threads", "pose_workers", "fit_iterations", "fit_model_batch", "identity_keyframes", "hand_iterations", "hand_model_batch", "panel_width"):
            if getattr(args, name) < 1:
                raise ValueError(f"{name} must be positive")
        if args.fit_device != "cpu" and args.pose_workers != 1:
            raise ValueError("GPU fitting requires --pose-workers 1")
        for name in ("input_root", "receiver_checks", "output", "sam_python", "mediapipe_python", "checkpoint", "mhr_model", "converter", "sam_source", "mediapipe_model"):
            setattr(args, name, getattr(args, name).resolve())
        args.identities = {name: digest(getattr(args, name)) for name in ("checkpoint", "mhr_model", "converter", "mediapipe_model")}
        args.identities["model_config"] = digest(args.checkpoint.parent / "model_config.yaml")
        args.identities["sam_source_manifest"] = verified_sam_source(args.sam_source)
        environments = {name: runtime_identity(getattr(args, name)) for name in ("sam_python", "mediapipe_python")}
        for name, report in environments.items():
            args.identities[f"{name}_runtime"] = report["fingerprint_sha256"]
        args.output.mkdir(parents=True, exist_ok=True)
        # Exclusive local lock deliberately survives an unclean termination. An
        # operator must inspect the recorded PID before removing a stale lock.
        lock = args.output / "batch.lock"
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "started_at": utcnow(), "episodes": episodes}, stream)
        environment_path = args.output / f"runtime-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        results = []
        try:
            write_json(environment_path, {"recorded_at": utcnow(), "environments": environments,
                       "asset_identities": args.identities, "execution_sources": execution_sources}, exclusive=True)
            for episode in episodes:
                try:
                    results.append(run_episode(args, episode))
                except Exception as exc:
                    results.append({"episode": episode, "status": "failed", "error": str(exc)})
                    print(f"FAILED episode {episode:02d}: {exc}", file=sys.stderr, flush=True)
                    if not args.continue_on_error:
                        raise
                finally:
                    write_json(args.output / "batch-progress.json", {"schema_version": "human_batch_progress.v1", "updated_at": utcnow(),
                        "requested_episodes": episodes, "results": results, "quality_acceptance": "not implied by execution completion"})
        finally:
            lock.unlink()
        return 1 if any(r["status"] == "failed" for r in results) else 0
    except (ValueError, OSError, KeyError, StageError) as exc:
        print(f"Human batch failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
