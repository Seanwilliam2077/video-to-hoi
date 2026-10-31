"""Durable, source-pinned postprocessing for only refined EP16 and EP11.

Preflight creates reviewable commands without launching anything. After explicit
root approval, run mode requires that exact preflight/config/helper identity.
Upstream MP and Ego inference are owned by other jobs; this controller only waits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import numpy as np


TASK = Path(__file__).resolve().parent
RESULTS = Path("J:/video-to-hoi/runs/egohand-refined-pilots-20261031").resolve()
ATTEMPT = "02"
COUNTS = {16: 360, 11: 877}
UPSTREAM = "9dfc9f25fe0b8c69a2f4f0bc01eb669097f7d1b9"
CHECKPOINT = "f940ecb6e3b73d5bd9e562cff232550e1c072f6c063eb4f705951122b15d8a75"
STATE_LOCK = threading.Lock()
STOP = threading.Event()
PROGRESS = {}
STOP_REQUEST = TASK / f"postprocessing-stop{ATTEMPT}.request.json"


class StopRequested(RuntimeError):
    """Stop new stage launches while existing owned jobs finish normally."""


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path):
    return {"path": str(path), "sha256": digest(path)}


def check_stop(concurrent_fit=None):
    if STOP_REQUEST.exists():
        if concurrent_fit is not None:
            concurrent_fit.result()
        request = read(STOP_REQUEST)
        raise StopRequested(str(request.get("reason", "Explicit task-local stop requested")))


def bound(item):
    path = Path(item["path"])
    if digest(path) != item["sha256"]:
        raise ValueError(f"Changed frozen input: {path}")
    return path


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def state(episode, status, **values):
    with STATE_LOCK:
        PROGRESS[str(episode)] = {"recorded_at": now(), "status": status, **values}
        data = {"recorded_at": now(), "controller_pid": os.getpid(), "attempt": ATTEMPT, "episodes": PROGRESS,
                "all30_execution_authorized": False, "automatic_promotion": False}
        path = TASK / f"postprocessing-progress{ATTEMPT}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def fit_command(cfg, item, spec, kind):
    runtime = Path(cfg["runtime_root"])
    source = read(item["manifest"])
    config = read(bound(spec["fit_configuration_reference"]))["config"]
    overrides = {key: config[key] for key in (
        "iterations", "learning_rate", "holdout_stride", "unobserved_policy", "acceleration_weight",
        "pose_prior_weight", "max_delta_radians", "relative_weight", "optimize_wrist_chain",
        "arm_delta_limit_radians", "wrist_max_displacement_metres", "arm_update_temporal_weight",
        "finger_update_temporal_weight", "support_transition_frames", "correction_velocity_budget_weight",
        "correction_budget_safety_fraction", "spatial_acceleration_weight")}
    sys.path.insert(0, str(runtime / "src"))
    from v2hoi.hand_refinement import HandFitConfig
    if asdict(replace(HandFitConfig(), **overrides)) != config:
        raise ValueError("Frozen CLI/defaults cannot reproduce the complete preserved fit configuration")
    for key, value in {"iterations": 600, "learning_rate": .001, "holdout_stride": 5,
                       "support_transition_frames": 8, "correction_velocity_budget_weight": 1.,
                       "correction_budget_safety_fraction": .2}.items():
        if config[key] != value:
            raise ValueError(f"Preserved pilot condition changed: {key}")
    root = RESULTS / f"episode_{item['episode']:06d}"
    target = Path(spec["fresh_mp_dir"]) / "observations.npz" if kind == "mp" else root / f"observations-smoothed{ATTEMPT}/observations.npz"
    output = root / (f"candidate-mp{ATTEMPT}" if kind == "mp" else f"candidate-egohand{ATTEMPT}")
    command = [cfg["human_python"], "-B", str(runtime / "tools/refine_human_hands.py"),
        "--baseline", item["baseline"]["path"], "--observations", str(target),
        "--manifest", item["manifest"], "--review-arrays", str(Path(item["baseline"]["path"]).parent / "review_arrays"),
        "--model", source["mhr_model"], "--device", "cpu", "--threads", "1", "--model-batch", "8", "--output", str(output)]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command += [flag, str(value)]
    return command


def adapter_command(cfg, item, spec, variant):
    root = RESULTS / f"episode_{item['episode']:06d}"
    prediction = Path(spec["prediction_root"])
    return [cfg["human_python"], "-B", str(Path(cfg["runtime_root"]) / "tools/adapt_refined_egohand.py"),
        "--left", str(prediction / variant / "left_results.npz"),
        "--right", str(prediction / variant / "right_results.npz"),
        "--scale-reference-left", str(prediction / "raw/left_results.npz"),
        "--scale-reference-right", str(prediction / "raw/right_results.npz"),
        "--fresh-observations", str(Path(spec["fresh_mp_dir"]) / "observations.npz"),
        "--manifest", item["manifest"], "--baseline-review-arrays", str(Path(item["baseline"]["path"]).parent / "review_arrays"),
        "--crop-policy-receipt", spec["crop_receipt"]["path"],
        "--prediction-receipt", str(prediction / "prediction-receipt.json"),
        "--upstream-commit", UPSTREAM, "--variant", variant,
        "--output", str(root / f"observations-{variant}{ATTEMPT}")]


def cache_plan(cfg, item, spec):
    """Independently derive reuse from exact old/new original-frame crop bytes."""
    old = read(bound(spec["legacy_raw_receipt"]))
    for key, value in {"status": "completed", "episode": item["episode"], "source_frames": item["frames"],
        "frames": item["frames"], "source_video_sha256": item["video_sha256"], "complete_source_coverage": True,
        "upstream_commit": UPSTREAM, "checkpoint_sha256": CHECKPOINT, "device": "cpu", "dtype": "float32",
        "runner_sha256": "9cbfdd91c492da8d54f661d02f3cfba567e09ec6206ff477b4f2a36eac75770b",
        "perspective_cropping": False, "pixel_aspect_rectification": False, "box_expansion": False}.items():
        if old.get(key) != value:
            raise ValueError(f"Legacy RAW reuse source contract differs: {key}")
    previous = old["input_bindings"]
    if previous["camera_sha256"] != item["camera_sha256"] or previous["original_observations_sha256"] != item["original_observations_sha256"]:
        raise ValueError("Legacy RAW camera or input crop source changed")
    bound({"path": previous["camera"], "sha256": previous["camera_sha256"]})
    boxes_old = npz(bound({"path": previous["original_observations"], "sha256": previous["original_observations_sha256"]}))["crop_boxes"]
    boxes_new = np.load(bound(spec["crop_boxes"]), allow_pickle=False)
    for saved in previous["saved_boxes"]:
        values = read(bound(saved))
        side = ("left", "right").index(saved["side"])
        for frame, box in enumerate(boxes_old[:, side]):
            sampling = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0]], np.float32)
            if not np.array_equal(sampling, np.asarray(values[f"{frame:06d}"], np.float32)):
                raise ValueError("Legacy crop bounds do not reproduce its executed neural sampling")
    mask = np.all(boxes_old == boxes_new, axis=-1)
    if mask.shape != (item["frames"], 2):
        raise ValueError("Cache reuse original frame/side dimensions differ")
    for row in old["raw_outputs"]:
        bound(row)
        if row["variant"] != "raw" or not row["complete_source_coverage"]:
            raise ValueError("Only full RAW outputs can supply cached predictions")
    return {"legacy_receipt": spec["legacy_raw_receipt"],
            "cache_hit_original_frame_ids_per_side": {side: np.flatnonzero(mask[:, s]).tolist() for s, side in enumerate(("left", "right"))},
            "neural_recomputed_original_frame_ids_per_side": {side: np.flatnonzero(~mask[:, s]).tolist() for s, side in enumerate(("left", "right"))},
            "cache_hits": int(mask.sum()), "neural_recomputed_views": int((~mask).sum()),
            "smoothed_predictions_reused": False, "matching": "Exact original source-video/frame/anatomical-side/estimated-camera and executed source-pixel crop"}


def preflight(cfg, config_path):
    if cfg.get("track") != 1 or cfg.get("authorized_episodes") != [16, 11] or cfg.get("all30_execution_authorized") is not False:
        raise ValueError("Wrong two-pilot controller scope")
    if cfg.get("attempt") != ATTEMPT:
        raise ValueError("Explicit controller/config attempt identity differs")
    runtime = Path(cfg["runtime_root"]).resolve()
    if runtime != (TASK / f"runtime{ATTEMPT}").resolve():
        raise ValueError("Require the root-approved new task/runtime02 snapshot")
    snapshot = read(bound(cfg["runtime_snapshot"]))
    for entry in snapshot["files"]:
        path = (runtime / entry["path"]).resolve()
        if not path.is_relative_to(runtime):
            raise ValueError("Snapshot source escaped its frozen runtime")
        bound({"path": str(path), "sha256": entry["sha256"]})
    for name in ("tools/adapt_refined_egohand.py", "tools/refine_human_hands.py", "tools/evaluate_refined_egohand_pilots.py"):
        if not (runtime / name).is_file():
            raise ValueError(f"Missing frozen tool: {name}")
    inputs = read(bound(cfg["input_manifest"]))
    if inputs.get("track") != 1 or inputs.get("all30_execution_authorized") is not False:
        raise ValueError("Wrong source inputs")
    bound(cfg["protocol"])
    bound(cfg["renderer"])
    bound(cfg["inference_runner"])
    bound(cfg["raw_cache_contract"])
    proof = read(bound(cfg["raw_cache_proof"]))
    if not proof.get("passed") or proof.get("neural_forward_calls") != 0 or proof["proof_source_sha256"] != cfg["raw_cache_contract"]["sha256"]:
        raise ValueError("RAW restoration/smoothing proof failed or source changed")
    sparse_proof = read(bound(cfg["raw_cache_sparse_export_proof"]))
    if not sparse_proof.get("passed") or sparse_proof.get("neural_forward_calls") != 0 or sparse_proof.get("runner_sha256") != cfg["inference_runner"]["sha256"] or sparse_proof.get("all_exported_parameter_and_geometry_arrays_unchanged") is not True:
        raise ValueError("Sparse changed-view chunk export proof failed or source changed")
    for row in sparse_proof["chunk_output"]:
        bound(row)
    if not Path(cfg["human_python"]).is_file():
        raise ValueError("Human runtime interpreter unavailable")
    if [entry["episode"] for entry in cfg["episodes"]] != [16, 11]:
        raise ValueError("Controller must contain exactly EP16 then EP11")
    sources = {entry["episode"]: entry for entry in inputs["episodes"]}
    planned, originals, caches = [], [], []
    for spec in cfg["episodes"]:
        episode = spec["episode"]
        item = sources[episode]
        if item["frames"] != COUNTS[episode] or item["fps"] != 30:
            raise ValueError("Unexpected original source coverage")
        for part in ("fresh_mp_dir", "prediction_root"):
            path = Path(spec[part]).resolve()
            if not path.is_relative_to(RESULTS / f"episode_{episode:06d}"):
                raise ValueError("Upstream result path escaped the authorized episode")
        bound({"path": item["manifest"], "sha256": item["manifest_sha256"]})
        bound({"path": item["video"], "sha256": item["video_sha256"]})
        bound({"path": item["original_observations"], "sha256": item["original_observations_sha256"]})
        bound(item["original_bindings"]["old_active_hands"])
        originals.append(binding(bound(item["baseline"])))
        for role in ("legacy_mediapipe", "legacy_egohand"):
            native = bound(spec[role])
            bound({"path": str(native.parent / "report.json"), "sha256": spec[role]["report_sha256"]})
            originals.append({"role": role, **binding(native)})
        if spec["legacy_mediapipe"]["sha256"] != item["mediapipe_candidate_sha256"] or spec["fit_configuration_reference"]["sha256"] != spec["legacy_mediapipe"]["report_sha256"]:
            raise ValueError("Legacy MP or its preserved fit-config reference changed")
        for name in ("crop_boxes", "crop_receipt", "crop_anchors", "review_windows", "fit_configuration_reference"):
            bound(spec[name])
        caches.append({"episode": episode, **cache_plan(cfg, item, spec)})
        crop = read(spec["crop_receipt"]["path"])
        if crop["source_video_sha256"] != item["video_sha256"] or crop["frames"] != item["frames"]:
            raise ValueError("Wrong crop source identity or coverage")
        if crop["crop_boxes_sha256"] != spec["crop_boxes"]["sha256"] or crop["crop_anchors_sha256"] != spec["crop_anchors"]["sha256"]:
            raise ValueError("Crop arrays differ from their receipt")
        anchors = npz(spec["crop_anchors"]["path"])
        if not np.array_equal(anchors["frame_indices"], np.arange(item["frames"])) or np.any(anchors["training_anchor_weights"][::5] > 0):
            raise ValueError("Crop construction used heldout hand anchors or changed original IDs")
        for kind in ("mp", "egohand"):
            planned.append({"episode": episode, "stage": "fit-" + kind, "command": fit_command(cfg, item, spec, kind)})
        for variant in ("raw", "smoothed"):
            planned.append({"episode": episode, "stage": "adapt-" + variant, "command": adapter_command(cfg, item, spec, variant)})
        for dirname in (f"observations-raw{ATTEMPT}", f"observations-smoothed{ATTEMPT}", f"candidate-mp{ATTEMPT}", f"candidate-egohand{ATTEMPT}", f"review{ATTEMPT}"):
            if (RESULTS / f"episode_{episode:06d}" / dirname).exists():
                raise ValueError(f"Preserve prior postprocessing attempt: {dirname}")
    if sum(row["cache_hits"] for row in caches) != 2160 or sum(row["neural_recomputed_views"] for row in caches) != 314:
        raise ValueError("Final routing02 exact RAW reuse plan differs from reviewed2160+314 hand views")
    return {"created_at": now(), "track": 1, "attempt": ATTEMPT, "episodes": [16, 11], "config": binding(config_path),
        "controller_sha256": digest(__file__), "job_runner_sha256": digest(TASK / "run_postprocessing_job.py"),
        "runtime_snapshot": cfg["runtime_snapshot"], "protocol": cfg["protocol"], "planned_commands": planned,
        "preserved_originals": originals, "independently_derived_raw_cache_plans": caches,
        "all30_execution_authorized": False, "automatic_promotion": False,
        "launched": False, "quality_accepted": False}


def wait_upstream(path, episode, description, concurrent_fit=None):
    state(episode, "waiting_for_" + description, receipt=str(path))
    check_stop(concurrent_fit)
    while not Path(path).exists():
        check_stop(concurrent_fit)
        if concurrent_fit is not None and concurrent_fit.done():
            concurrent_fit.result()
        if STOP.wait(5):
            raise RuntimeError("Another authorized pilot failed; no new stages will launch")
    # A producer can expose its exclusive receipt before its final JSON write
    # closes. Retry only a short incomplete-read window, never a failed job.
    deadline = time.monotonic() + 1
    while True:
        try:
            receipt = read(path)
            break
        except json.JSONDecodeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.05)
    if receipt.get("returncode") != 0:
        raise RuntimeError(f"Upstream job failed; preserve its attempt: {path}")
    return receipt


def make_job(cfg, episode, stage, command, paths):
    path = TASK / f"ep{episode}-{stage}.json"
    if path.exists():
        raise ValueError(f"Preserve an existing stage definition: {path}")
    bindings = list(dict.fromkeys(map(str, [Path(__file__), TASK / "run_postprocessing_job.py", Path(cfg["runtime_snapshot"]["path"]),
                                            Path(cfg["protocol"]["path"]), Path(cfg["input_manifest"]["path"]), *paths])))
    write(path, {"episode": episode, "command": list(map(str, command)), "cwd": cfg["runtime_root"],
                 "log": str(path.with_suffix(".log")), "input_bindings": [binding(value) for value in bindings],
                 "automatic_promotion": False, "all30_execution_authorized": False})
    return path


def run_job(cfg, path):
    check_stop()
    if STOP.is_set():
        raise RuntimeError("Another pilot failed; refusing a new stage")
    result = subprocess.run([cfg["human_python"], "-B", str(TASK / "run_postprocessing_job.py"), str(path)], cwd=TASK)
    finished = path.with_suffix(".finished.json")
    if result.returncode or not finished.exists() or read(finished)["returncode"] != 0:
        raise RuntimeError(f"Stage failed; preserve attempt and inspect actual PID/receipt: {path}")
    return read(finished)


def verify_mp(item, spec):
    root = Path(spec["fresh_mp_dir"])
    data, report = npz(root / "observations.npz"), read(root / "report.json")
    expected = np.arange(item["frames"])
    if not np.array_equal(data["frame_indices"], expected) or list(data["side_order"].astype(str)) != ["left", "right"]:
        raise ValueError("Fresh MP lacks full original-frame anatomical mapping")
    if str(data["source_video_sha256"].item()) != item["video_sha256"] or report["source_video_sha256"] != item["video_sha256"] or report["frames"] != item["frames"]:
        raise ValueError("Fresh MP source identity differs")
    if report["crop_boxes_sha256"] != spec["crop_boxes"]["sha256"] or report["crop_policy_receipt_sha256"] != spec["crop_receipt"]["sha256"]:
        raise ValueError("Fresh MP did not bind the frozen new crop policy")
    if not np.array_equal(data["crop_boxes"], np.load(spec["crop_boxes"]["path"], allow_pickle=False)):
        raise ValueError("Fresh MP actual source crop arrays differ")
    return [root / "observations.npz", root / "report.json", spec["fresh_mp_finished_receipt"]]


def verify_prediction(cfg, item, spec):
    root = Path(spec["prediction_root"])
    receipt = read(root / "prediction-receipt.json")
    for key, value in {"status": "completed", "episode": item["episode"], "source_frames": item["frames"],
        "frames": item["frames"], "decoded_source_frames": item["frames"], "source_video_sha256": item["video_sha256"],
        "complete_source_coverage": True, "upstream_commit": UPSTREAM, "checkpoint_sha256": CHECKPOINT,
        "runner_sha256": cfg["inference_runner"]["sha256"], "protocol_sha256": cfg["protocol"]["sha256"],
        "input_manifest_sha256": cfg["input_manifest"]["sha256"], "same_original_camera": True,
        "perspective_cropping": False, "pixel_aspect_rectification": False, "box_expansion": False,
        "all30_execution_started": False}.items():
        if receipt.get(key) != value:
            raise ValueError(f"Full inference receipt contract differs: {key}")
    if receipt["selected_original_frame_ids"] != list(range(item["frames"])):
        raise ValueError("Smoke/subset predictions cannot be treated as complete")
    for key, config_key in (("crop_boxes_sha256", "crop_boxes"), ("crop_receipt_sha256", "crop_receipt")):
        if receipt["input_bindings"][key] != spec[config_key]["sha256"]:
            raise ValueError("Inference used different new crops")
    if receipt["model_load"]["missing_learned_keys"] or receipt["model_load"]["unexpected_keys"]:
        raise ValueError("Model learned-key loading was incomplete")
    plan = cache_plan(cfg, item, spec)
    lineage = receipt["raw_cache_lineage"]
    old_receipt = read(bound(spec["legacy_raw_receipt"]))
    if Path(lineage["receipt"]).resolve() != Path(spec["legacy_raw_receipt"]["path"]).resolve():
        raise ValueError("RAW reuse receipt path differs from its approved source")
    for old_row in old_receipt["raw_outputs"]:
        cached_row = lineage["raw_outputs"][old_row["side"]]
        if cached_row["sha256"] != old_row["sha256"] or Path(cached_row["path"]).resolve() != Path(old_row["path"]).resolve():
            raise ValueError("RAW reuse cannot substitute a different or smoothed source output")
    for key, expected in (("receipt_sha256", spec["legacy_raw_receipt"]["sha256"]),
                          ("cache_contract_source_sha256", cfg["raw_cache_contract"]["sha256"]),
                          ("cache_proof_sha256", cfg["raw_cache_proof"]["sha256"]),
                          ("smoothed_cache_reused", False), ("prediction_interpolation", False)):
        if lineage.get(key) != expected:
            raise ValueError(f"RAW reuse lineage differs: {key}")
    for key in ("cache_hit_original_frame_ids_per_side", "neural_recomputed_original_frame_ids_per_side"):
        if receipt[key] != plan[key]:
            raise ValueError("RAW reuse/recomputation IDs differ from exact input equality")
    for field, ids_field in (("cache_hits_per_side", "cache_hit_original_frame_ids_per_side"),
                            ("neural_recomputed_views_per_side", "neural_recomputed_original_frame_ids_per_side")):
        if receipt[field] != {side: len(ids) for side, ids in plan[ids_field].items()}:
            raise ValueError("RAW cache/recomputation count metadata differs from actual planned IDs")
    if receipt.get("raw_cache_enabled") is not True or receipt.get("official_smoothing_recomputed_over_full_merged_raw_clip") is not True or receipt.get("all_original_hand_views_neurally_recomputed") is not False:
        raise ValueError("Cache/smoothing semantics differ from the reviewed pipeline")
    if receipt.get("smoothing_seconds", 0) <= 0:
        raise ValueError("No full-clip smoothing execution recorded")
    files = [root / "prediction-receipt.json", spec["prediction_finished_receipt"]]
    reused_checks = []
    for variant, key, label in (("raw", "raw_outputs", "raw"), ("smoothed", "smoothed_outputs", "upstream_adaptive_smoothed")):
        rows = receipt[key]
        if {row["side"] for row in rows} != {"left", "right"} or len(rows) != 2:
            raise ValueError("Full prediction needs both anatomical hands")
        for row in rows:
            path = bound(row)
            if path.resolve() != (root / variant / f"{row['side']}_results.npz").resolve() or row["variant"] != label:
                raise ValueError("Prediction variant/path differs")
            data = npz(path)
            if list(data["frame_names"].astype(str)) != [f"{i:06d}" for i in range(item["frames"])]:
                raise ValueError("Prediction original frame IDs differ")
            joints, translation = data["joints"], data["cam_t"]
            if joints.shape != (item["frames"], 25, 3) or translation.shape != (item["frames"], 3) or not np.isfinite(joints).all() or not np.isfinite(translation).all():
                raise ValueError("Invalid full SOMA hand geometry")
            if np.linalg.norm(joints[:, 0], axis=-1).max() >= 1e-4 or np.any((joints + translation[:, None])[..., 2] <= 0):
                raise ValueError("Prediction wrist-local or camera-depth convention differs")
            if variant == "raw":
                previous = npz(bound(lineage["raw_outputs"][row["side"]]))
                if previous["frame_names"].astype(str).tolist() != [f"{i:06d}" for i in range(item["frames"])]:
                    raise ValueError("Legacy RAW output actual original-frame row mapping differs")
                ids = plan["cache_hit_original_frame_ids_per_side"][row["side"]]
                fields = ("hand_poses", "identity_coeffs", "global_scale", "vertices", "joints", "cam_t", "scale_params",
                          "pred_keypoints_2d", "pred_projection_valid", "pred_kp2d_crop_presence", "pred_kp2d_sigma")
                if any(not np.array_equal(data[field][ids], previous[field][ids]) for field in fields):
                    raise ValueError("A reused RAW row differs from its immutable source output")
                reused_checks.append({"side": row["side"], "cached_rows_exactly_equal": len(ids), "fields": list(fields)})
            files.append(path)
    write(TASK / f"ep{item['episode']}-prediction-verification{ATTEMPT}.json", {"recorded_at": now(), "episode": item["episode"],
          "frames": item["frames"], "receipt": binding(root / "prediction-receipt.json"), "files": [binding(path) for path in files],
          "raw_cache_plan": plan, "reused_raw_value_checks": reused_checks,
          "full_smoothing_recomputed_from_merged_raw_source_and_receipt_evidence": True,
          "scope": "Source, model, exact RAW lineage, variant, coverage and geometry conventions; not hand accuracy", "quality_accepted": False})
    return files


def evaluation_inputs(cfg, item, spec):
    root = RESULTS / f"episode_{item['episode']:06d}"
    variants = []
    for role, label, path in (
        ("legacy_mediapipe", "Preserved bounded MediaPipe MHR", Path(spec["legacy_mediapipe"]["path"])),
        ("legacy_egohand", "Preserved EgoHand smoothed MHR", Path(spec["legacy_egohand"]["path"])),
        ("fresh_mediapipe", "Fresh same-new-crop MediaPipe MHR", root / f"candidate-mp{ATTEMPT}/episode_{item['episode']:06d}_human.npz"),
        ("refined_egohand", "Revised EgoHand smoothed/gated MHR", root / f"candidate-egohand{ATTEMPT}/episode_{item['episode']:06d}_human.npz")):
        value = {"role": role, "label": label, **binding(path), "report_sha256": digest(path.parent / "report.json")}
        if role in ("fresh_mediapipe", "refined_egohand"):
            value.update(evidence_role=role, support=binding(path.parent / "active_hands.npy"),
                         weights=binding(path.parent / "optimization_weights.npy"))
        variants.append(value)
    def evidence(path, kind, label, shared):
        return {**binding(path), "kind": kind, "label": label, "shared_inputs": shared,
                "report": binding(path.parent / "report.json")}
    result = {key: item[key] for key in ("episode", "frames", "fps", "video", "video_sha256", "manifest", "manifest_sha256", "baseline", "original_bindings")}
    result.update(fit_configuration_reference=spec["fit_configuration_reference"], controlled_pair=["fresh_mediapipe", "refined_egohand"], variants=variants,
        crop_formation_provenance={key: spec[key] for key in ("crop_boxes", "crop_receipt", "crop_anchors")},
        evidence={
            "legacy_mediapipe": evidence(Path(item["original_observations"]), "image_detector", "Legacy MP image reference", []),
            "fresh_mediapipe": evidence(Path(spec["fresh_mp_dir"]) / "observations.npz", "image_detector", "Fresh MP: shared wrist/gate reference", ["EgoHand wrist anchoring", "EgoHand image-support gating"]),
            "refined_raw": evidence(root / f"observations-raw{ATTEMPT}/observations.npz", "generated_proposal", "Raw Ego retarget/gated proposal", ["fresh_mediapipe training wrists/gates"]),
            "refined_egohand": evidence(root / f"observations-smoothed{ATTEMPT}/observations.npz", "generated_proposal", "Smoothed Ego retarget/gated proposal", ["fresh_mediapipe training wrists/gates"])})
    path = TASK / f"ep{item['episode']}-evaluation-inputs{ATTEMPT}.json"
    write(path, {"track": 1, "authorized_episodes": [16, 11], "all30_execution_authorized": False, "episodes": [result]})
    return path


def verify_retarget(item):
    root = RESULTS / f"episode_{item['episode']:06d}"
    native = np.load(Path(item["baseline"]["path"]).parent / "review_arrays/joints_3d.npy", mmap_mode="r", allow_pickle=False)
    native_depth = native[:, [78, 42], 2]
    rows, scalars = {}, []
    for variant in ("raw", "smoothed"):
        path = root / f"observations-{variant}{ATTEMPT}/observations.npz"
        data = npz(path)
        if not np.array_equal(data["frame_indices"], np.arange(item["frames"])):
            raise ValueError("Retargeted original frame IDs differ")
        if str(data["refinement_kind"].item()) != "native_depth_training_wrist_image_gate.v1":
            raise ValueError("Unexpected retargeting variant")
        for name, shape in (("shared_wrist_anchor_weights", (item["frames"], 2)),
                            ("shared_visibility_gate_weights", (item["frames"], 2)),
                            ("shared_image_joint_gate_weights", (item["frames"], 2, 21)),
                            ("fixed_scale_anchor_weights", (item["frames"], 2))):
            value = np.asarray(data[name])
            if value.shape != shape or not np.isfinite(value).all() or np.any(value < 0) or np.any(value[::5] > 0):
                raise ValueError("Shared crop/anchor/scale/gate used heldout targets or has invalid provenance")
        if np.any(data["confidence"][::5] > 0):
            raise ValueError("Retargeting admitted downstream-development-holdout targets")
        geometry = data["proposal_joints3d_camera"]
        if geometry.shape != (item["frames"], 2, 21, 3) or not np.isfinite(geometry).all():
            raise ValueError("Retargeted proposal geometry differs")
        depth_error = float(np.abs(geometry[:, :, 0, 2] - native_depth).max())
        if depth_error > 1e-5:
            raise ValueError("Retargeting changed baseline native wrist depth")
        scalar = data["fixed_native_to_ego_palm_scalar"]
        if scalar.shape != (2,) or not np.isfinite(scalar).all() or np.any(scalar <= 0):
            raise ValueError("Invalid per-hand fixed scalar")
        scalars.append(scalar)
        rows[variant] = {"observations": binding(path), "report": binding(path.parent / "report.json"),
            "heldout_shared_input_weights_zero": True, "native_wrist_depth_max_difference_metres": depth_error,
            "fixed_scalar": scalar.tolist(), "fixed_scale_calibrated": data["fixed_scale_calibrated"].tolist(),
            "new_target_frames_per_hand": (data["confidence"].sum(-1) > 0).sum(0).tolist(),
            "scope": "Shared-input and coordinate provenance, not actual hand visibility, physical scale or accuracy"}
    if not np.array_equal(scalars[0], scalars[1]):
        raise ValueError("Raw and smoothed retarget variants used different fixed raw-reference scales")
    path = TASK / f"ep{item['episode']}-retarget-verification{ATTEMPT}.json"
    write(path, {"recorded_at": now(), "episode": item["episode"], "variants": rows,
                 "same_raw_reference_fixed_scales": True, "quality_accepted": False})
    return path


def episode(cfg, item, spec, fits):
    number = item["episode"]
    root = RESULTS / f"episode_{number:06d}"
    runtime = Path(cfg["runtime_root"])
    try:
        wait_upstream(spec["fresh_mp_finished_receipt"], number, "complete_fresh_MP")
        mp_files = verify_mp(item, spec)
        state(number, "fitting_fresh_same_crop_MP_CPU1")
        def fit(kind, files):
            command = fit_command(cfg, item, spec, kind)
            job = make_job(cfg, number, "fit-" + kind + ATTEMPT, command, [*files, item["manifest"], item["baseline"]["path"],
                spec["fit_configuration_reference"]["path"], runtime / "tools/refine_human_hands.py", runtime / "src/v2hoi/hand_refinement.py"])
            return run_job(cfg, job)
        mp_fit = fits.submit(fit, "mp", mp_files)
        wait_upstream(spec["prediction_finished_receipt"], number, "complete_EgoHand", mp_fit)
        prediction_files = verify_prediction(cfg, item, spec)
        state(number, "adapting_raw_and_smoothed_with_shared_training_MP")
        for variant in ("raw", "smoothed"):
            job = make_job(cfg, number, "adapt-" + variant + ATTEMPT, adapter_command(cfg, item, spec, variant),
                [*prediction_files, *mp_files, spec["crop_receipt"]["path"], item["manifest"],
                 runtime / "tools/adapt_refined_egohand.py", runtime / "src/v2hoi/refined_egohand.py", runtime / "src/v2hoi/egohand_adapter.py"])
            run_job(cfg, job)
        retarget_verification = verify_retarget(item)
        state(number, "fitting_revised_smoothed_EgoHand_CPU1")
        ego_fit = fits.submit(fit, "egohand", [root / f"observations-smoothed{ATTEMPT}/observations.npz", root / f"observations-smoothed{ATTEMPT}/report.json", retarget_verification])
        mp_fit.result()
        ego_fit.result()
        for directory in (f"candidate-mp{ATTEMPT}", f"candidate-egohand{ATTEMPT}"):
            report = read(root / directory / "report.json")
            if report["config"] != read(spec["fit_configuration_reference"]["path"])["config"] or report["frames"] != item["frames"]:
                raise ValueError("Actual native fit configuration or coverage differs")
        state(number, "fresh_all_variant_native_evaluation")
        inputs = evaluation_inputs(cfg, item, spec)
        evaluation = root / f"evaluation{ATTEMPT}.json"
        command = [cfg["human_python"], "-B", str(runtime / "tools/evaluate_refined_egohand_pilots.py"),
            "--episode", str(number), "--input-manifest", str(inputs), "--review-windows", spec["review_windows"]["path"],
            "--output", str(evaluation), "--tool-root", str(runtime), "--device", "cpu", "--threads", "1", "--model-batch", "8"]
        run_job(cfg, make_job(cfg, number, "evaluation" + ATTEMPT, command, [inputs, spec["review_windows"]["path"],
            runtime / "tools/evaluate_refined_egohand_pilots.py", runtime / "tools/evaluate_human_hand_pilots.py"]))
        state(number, "rendering_shared_ROI_native_and_proposal_comparisons")
        command = [cfg["human_python"], "-B", cfg["renderer"]["path"], "--episode", str(number),
            "--input-manifest", str(inputs), "--evaluation", str(evaluation), "--review-windows", spec["review_windows"]["path"],
            "--tool-root", str(runtime), "--output", str(root / f"review{ATTEMPT}"),
            "--proposal-roles", "fresh_mediapipe", "refined_raw", "refined_egohand"]
        run_job(cfg, make_job(cfg, number, "render" + ATTEMPT, command, [inputs, evaluation, cfg["renderer"]["path"],
            spec["review_windows"]["path"], runtime / "tools/compare_human_hands.py", runtime / "tools/review_human.py"]))
        for role in ("legacy_mediapipe", "legacy_egohand"):
            bound(spec[role])
            bound({"path": str(Path(spec[role]["path"]).parent / "report.json"), "sha256": spec[role]["report_sha256"]})
        bound(item["baseline"])
        summary = {"completed_at": now(), "episode": number, "frames": item["frames"],
                   "status": "complete_numeric_and_video_evidence_awaiting_visual_review", "evaluation": binding(evaluation),
                   "video_summary": binding(root / f"review{ATTEMPT}/summary.json"), "review_index": str(root / f"review{ATTEMPT}/index.html"),
                   "quality_accepted": False, "contact_quality_accepted": False, "official_score_available": False,
                   "automatic_promotion": False, "all30_execution_authorized": False}
        write(root / f"postprocessing-completion{ATTEMPT}.json", summary)
        state(number, summary["status"], completion=summary)
        return summary
    except StopRequested as error:
        STOP.set()
        state(number, "stop_requested_no_new_stages_owned_jobs_draining", reason=str(error), stop_request=binding(STOP_REQUEST))
        raise
    except Exception as error:
        STOP.set()
        state(number, "failed_preserve_attempt", error=str(error), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--preflight-sha256")
    args = parser.parse_args()
    bound({"path": str(args.config), "sha256": args.config_sha256})
    cfg = read(args.config)
    if args.mode == "preflight":
        result = preflight(cfg, args.config)
        path = TASK / f"postprocessing-preflight{ATTEMPT}.json"
        write(path, result)
        print(json.dumps({"preflight": str(path), "sha256": digest(path), "launched": False}), flush=True)
        return
    if not args.preflight_receipt or not args.preflight_sha256:
        parser.error("Run mode requires exact reviewed preflight receipt/hash and root approval")
    prior = read(bound({"path": str(args.preflight_receipt), "sha256": args.preflight_sha256}))
    if prior["config"]["sha256"] != args.config_sha256 or prior["controller_sha256"] != digest(__file__) or prior["job_runner_sha256"] != digest(TASK / "run_postprocessing_job.py"):
        raise ValueError("Reviewed commands/config/controller identity changed; repeat preflight before launch")
    preflight(cfg, args.config)
    started = {"started_at": now(), "pid": os.getpid(), "attempt": ATTEMPT, "command": sys.argv, "config": binding(args.config),
               "preflight": binding(args.preflight_receipt), "controller_sha256": digest(__file__),
               "all30_execution_authorized": False, "automatic_promotion": False}
    write(TASK / f"postprocessing-started{ATTEMPT}.json", started)
    sources = {entry["episode"]: entry for entry in read(cfg["input_manifest"]["path"])["episodes"]}
    try:
        with ThreadPoolExecutor(max_workers=2) as fits, ThreadPoolExecutor(max_workers=2) as episodes:
            futures = [episodes.submit(episode, cfg, sources[spec["episode"]], spec, fits) for spec in cfg["episodes"]]
            completed = [future.result() for future in futures]
        write(TASK / f"postprocessing-finished{ATTEMPT}.json", {**started, "finished_at": now(), "returncode": 0, "episodes": completed})
    except StopRequested as error:
        write(TASK / f"postprocessing-finished{ATTEMPT}.json", {**started, "finished_at": now(), "returncode": 2,
              "status": "stopped_after_owned_jobs_drained", "reason": str(error), "stop_request": binding(STOP_REQUEST)})
        raise SystemExit(2)
    except Exception as error:
        write(TASK / f"postprocessing-finished{ATTEMPT}.json", {**started, "finished_at": now(), "returncode": 1,
              "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
