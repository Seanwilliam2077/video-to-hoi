"""Freeze and run the EP16 contact/no-contact native-MHR controlled pair.

The numerical fitter is unchanged. This wrapper binds one prepared object,
one RGB-reviewed phase schedule and identical observation-supported variables
for both arms. Only B adds the existing weak contact terms. It neither expands
hidden-hand controls nor calibrates an object's unknown physical scale.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v2hoi.hand_contact import phase_weights, validate_object_candidate
from v2hoi.hand_refinement import (
    HandFitConfig, supported_frames, support_transition_envelope, validate_observations,
)
from v2hoi.human_result import load_human_result, validate_human_result

SCHEMA = "ep16_contact_ab_plan.v1"
SOURCE_FILES = ("tools/refine_human_hands.py", "src/v2hoi/hand_refinement.py",
                "src/v2hoi/hand_contact.py", "src/v2hoi/human_result.py")
VALUE_OPTIONS = (
    "baseline", "observations", "manifest", "review-arrays", "output", "model",
    "device", "threads", "model-batch", "iterations", "learning-rate", "holdout-stride",
    "unobserved-policy", "acceleration-weight", "pose-prior-weight", "max-delta-radians",
    "relative-weight", "arm-delta-limit-radians", "wrist-max-displacement-metres",
    "arm-update-temporal-weight", "finger-update-temporal-weight", "support-transition-frames",
    "correction-velocity-budget-weight", "correction-budget-safety-fraction",
    "spatial-acceleration-weight",
)
REQUIRED_VALUES = {
    "device": "cpu", "threads": "1", "model-batch": "8", "iterations": "600",
    "learning-rate": "0.001", "holdout-stride": "5", "unobserved-policy": "preserve",
    "acceleration-weight": "0.025", "pose-prior-weight": "0.03",
    "max-delta-radians": "0.65", "relative-weight": "1.0",
    "arm-delta-limit-radians": "0.1", "wrist-max-displacement-metres": "0.04",
    "arm-update-temporal-weight": "0.08", "finger-update-temporal-weight": "0.08",
    "support-transition-frames": "8", "correction-velocity-budget-weight": "1.0",
    "correction-budget-safety-fraction": "0.2", "spatial-acceleration-weight": "0.0",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def binding(path):
    original = Path(path)
    path = original.resolve()
    if not path.is_file() or original.is_symlink():
        raise ValueError(f"Require an ordinary existing source file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def parse_reference_command(reference):
    """Accept only the explicit frozen MP02 fit configuration, never new flags."""
    if reference.get("episode") != 16:
        raise ValueError("Only the authorized EP16 reference fit is allowed")
    command = reference["command"]
    if len(command) < 4 or command[1] != "-B" or Path(command[2]).name != "refine_human_hands.py":
        raise ValueError("Reference must invoke the existing native hand fitter")
    parsed, positions = {}, {}
    position = 3
    while position < len(command):
        flag = command[position]
        if flag in positions:
            raise ValueError("Duplicate reference options are not allowed")
        positions[flag] = position
        if flag == "--optimize-wrist-chain":
            parsed[flag[2:]] = True
            position += 1
        elif flag.startswith("--") and flag[2:] in VALUE_OPTIONS and position + 1 < len(command):
            parsed[flag[2:]] = command[position + 1]
            position += 2
        else:
            raise ValueError(f"Unexpected reference option: {flag}")
    if set(parsed) != set(VALUE_OPTIONS) | {"optimize-wrist-chain"}:
        raise ValueError("Reference must state the complete frozen MP02 configuration")
    for name, expected in REQUIRED_VALUES.items():
        actual = parsed[name]
        try:
            equal = float(actual) == float(expected)
        except ValueError:
            equal = actual == expected
        if not equal:
            raise ValueError(f"Controlled configuration changed: {name}")
    return parsed, positions


def validate_manual_phases(phases, video_sha256):
    if (phases.get("track") != 1 or phases.get("episode") != 16 or
            phases.get("frame_count") != 360 or phases.get("video_sha256") != video_sha256):
        raise ValueError("Manual phases must identify the exact complete EP16 source")
    if not isinstance(phases.get("annotation_method"), str) or not phases["annotation_method"].strip():
        raise ValueError("Contact phases need an explicit manual annotation method")
    if not phases.get("source_review_receipt") or not phases.get("source_review_receipt_sha256"):
        raise ValueError("Contact phases need a pinned source-review receipt")
    for item in phases["phase_intervals_inclusive"]:
        if float(item.get("anchor_weight", 0)) > 0 and item.get("constraint_type") != "grasp":
            raise ValueError("Only reviewed grasp intervals may use fixed no-slip attraction")
    weights = phase_weights(phases, 360)
    if not weights.any():
        raise ValueError("The manual-contact arm requires at least one positive grasp interval")
    return weights


def paired_commands(reference, source_root, object_path, phases_path, a_output, b_output,
                    *, require_fresh_outputs=True):
    values, positions = parse_reference_command(reference)
    source_root = Path(source_root).resolve()
    output_paths = [Path(a_output).resolve(), Path(b_output).resolve()]
    if output_paths[0] == output_paths[1] or (require_fresh_outputs and any(path.exists() for path in output_paths)):
        raise ValueError("A/B outputs must be different fresh attempt directories")
    common = list(reference["command"])
    common[2] = str(source_root / "tools/refine_human_hands.py")
    commands = {}
    for role, output in zip(("A", "B"), output_paths):
        command = common.copy()
        command[positions["--output"] + 1] = str(output)
        if role == "B":
            command += ["--contact-object", str(Path(object_path).resolve()),
                        "--contact-phases", str(Path(phases_path).resolve()),
                        "--contact-anchor-weight", "0.01", "--contact-penetration-weight", "0.005",
                        "--contact-scale", "1.0", "--contact-scale-mode", "fixed_pose_geometry"]
        commands[role] = command
    return commands, values


def prepare(args):
    reference = read(args.reference_fit_job)
    commands, values = paired_commands(reference, args.source_root, args.object, args.phases,
                                        args.a_output, args.b_output)
    output_plan = Path(args.output_plan).resolve()
    if output_plan.exists():
        raise ValueError("Preserve previous plans; output plan must not exist")
    baseline_path, observations_path, manifest_path = [Path(values[name]).resolve()
                                                     for name in ("baseline", "observations", "manifest")]
    baseline = load_human_result(baseline_path)
    validate_human_result(baseline, expected_frames=360)
    expected_ids = np.arange(360)
    if not np.array_equal(baseline["frame_indices"], expected_ids):
        raise ValueError("Require every consecutive original EP16 frame")
    manifest = read(manifest_path)
    if manifest.get("track") != 1 or manifest.get("episode") != 16 or manifest.get("video_frames") != 360:
        raise ValueError("Only the complete explicit Track 1 EP16 manifest is permitted")
    video_sha = manifest["video_sha256"]
    source_video = Path(manifest["video"]).resolve()
    if "track_1" not in source_video.parts:
        raise ValueError("Require the permitted Track 1 source path")
    if baseline["report"].get("source_video_sha256", baseline["report"].get("video_sha256")) != video_sha:
        raise ValueError("Native initialization source identity differs")
    with np.load(observations_path, allow_pickle=False) as archive:
        observations = validate_observations({name: archive[name] for name in archive.files}, expected_ids)
    observer_report = read(observations_path.parent / "report.json")
    if observer_report.get("source_video_sha256") != video_sha or observer_report.get("frames") != 360:
        raise ValueError("Image observations identify a different source or frame coverage")
    k = np.asarray(manifest["camera_intrinsics"], dtype=np.float32)
    review = Path(values["review-arrays"]).resolve()
    if not np.array_equal(np.load(review / "frame_indices.npy", allow_pickle=False), expected_ids):
        raise ValueError("Review arrays use different original frame IDs")
    if not np.array_equal(np.load(review / "camera_intrinsics.npy", allow_pickle=False), k):
        raise ValueError("Review camera must exactly match the frozen human camera")
    for name, expected_shape in (("vertices_m", (360, 18439, 3)), ("joints_3d", (360, 127, 3))):
        geometry = np.load(review / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if geometry.shape != expected_shape or geometry.dtype.kind != "f":
            raise ValueError("Require complete native baseline review geometry")
    review_fps = np.load(review / "fps.npy", allow_pickle=False)
    if review_fps.shape != () or not np.isclose(float(review_fps), float(manifest["video_fps"]), atol=1e-6, rtol=0):
        raise ValueError("Review frame rate differs from the original video manifest")
    object_path = Path(args.object).resolve()
    with np.load(object_path, allow_pickle=False) as archive:
        obj = {name: archive[name] for name in archive.files}
    validated_object = validate_object_candidate(obj, expected_ids)
    if not np.array_equal(np.asarray(obj["camera_intrinsics"], dtype=np.float32), k):
        raise ValueError("Prepared object camera differs from the human camera")
    object_report = read(object_path.parent / "report.json")
    if object_report.get("track") != 1 or object_report.get("episode") != 16 or object_report.get("source_video_sha256", object_report.get("video_sha256")) != video_sha:
        raise ValueError("Prepared object must have explicit same-source Track 1 provenance")
    phases_path = Path(args.phases).resolve()
    phases = read(phases_path)
    manual_weights = validate_manual_phases(phases, video_sha)
    phase_review_path = Path(phases["source_review_receipt"])
    if digest(phase_review_path) != phases["source_review_receipt_sha256"]:
        raise ValueError("Manual annotation source-review receipt changed")
    reference_report_path = Path(values["output"]) / "report.json"
    reference_report = read(reference_report_path)
    if (reference_report.get("source_video_sha256") != video_sha or
            reference_report.get("baseline_sha256") != digest(baseline_path) or
            reference_report.get("observations_sha256") != digest(observations_path)):
        raise ValueError("Reference fit does not bind the same human and image observations")
    sources = [binding(Path(args.source_root) / name) for name in SOURCE_FILES]
    paths = [Path(args.reference_fit_job), reference_report_path, baseline_path, observations_path,
             observations_path.parent / "report.json", manifest_path, object_path,
             object_path.parent / "report.json", phases_path, phase_review_path, Path(values["model"]), source_video]
    paths += [review / f"{name}.npy" for name in ("vertices_m", "joints_3d", "faces", "camera_intrinsics", "frame_indices", "fps")]
    inputs = [binding(path) for path in paths]
    model_binding = next(row for row in inputs if Path(row["path"]) == Path(values["model"]).resolve())
    if model_binding["sha256"] != reference_report["mhr_model_sha256"]:
        raise ValueError("Native decoder differs from the reference fit")
    if next(row["sha256"] for row in inputs if Path(row["path"]) == source_video) != video_sha:
        raise ValueError("Original video bytes differ from the declared Track 1 source")
    training_confidence = observations["confidence"].copy()
    training_confidence[expected_ids % 5 == 0] = 0
    config = HandFitConfig(iterations=600, learning_rate=0.001, optimize_wrist_chain=True,
                           support_transition_frames=8, correction_velocity_budget_weight=1.0,
                           correction_budget_safety_fraction=0.2)
    config.validate()
    support = supported_frames(training_confidence, config)
    envelope = support_transition_envelope(support, 8)
    manual_usable = (manual_weights > 0) & validated_object["pose_usable"][:, None]
    plan = {
        "schema_version": SCHEMA, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "track": 1, "episode": 16, "frames": 360, "source_video_sha256": video_sha,
        "source_root": str(Path(args.source_root).resolve()), "input_bindings": inputs,
        "reference_fit_job": str(Path(args.reference_fit_job).resolve()),
        "prepared_object": str(object_path), "manual_phases": str(phases_path),
        "fitter_source_bindings": sources, "wrapper_source_sha256": digest(__file__),
        "commands": commands, "outputs": {role: command[command.index("--output") + 1] for role, command in commands.items()},
        "frozen_config": asdict(config), "object_scale_factor": 1.0,
        "manual_phase_frames": (manual_weights > 0).sum(0).tolist(),
        "manual_phase_object_usable_frames": manual_usable.sum(0).tolist(),
        "manual_phase_with_mutable_finger_controls": (manual_usable & (envelope > 0)).sum(0).tolist(),
        "manual_phase_without_mutable_finger_controls": (manual_usable & (envelope == 0)).sum(0).tolist(),
        "variable_support_policy": "Both arms retain identical image-supported plus8-frame-collar masks. Manual contact does not activate hidden-hand controls or add image targets.",
        "limitations": "Prepared object geometry/relative scale and manual phases are hypotheses, not physical contact truth. Fixed local anchors model stable grasp only. Contact penalties are absent outside admitted intervals; other unchanged temporal terms may still couple neighboring frames.",
        "quality_accepted": False, "official_scores_available": False, "all30_execution_authorized": False,
    }
    write_new(output_plan, plan)
    print(json.dumps({"plan": str(output_plan), "sha256": digest(output_plan),
                      "phase_frames_without_mutable_fingers": plan["manual_phase_without_mutable_finger_controls"]}), flush=True)


def verify_plan(plan):
    if (plan.get("schema_version") != SCHEMA or plan.get("track") != 1 or
            plan.get("episode") != 16 or plan.get("frames") != 360 or
            plan.get("all30_execution_authorized") is not False or plan.get("object_scale_factor") != 1.0 or
            plan.get("quality_accepted") is not False):
        raise ValueError("Execution plan violates the frozen EP16 A/B scope")
    if digest(__file__) != plan["wrapper_source_sha256"]:
        raise ValueError("Controlled-pair wrapper source changed after preparation")
    expected, _ = paired_commands(read(plan["reference_fit_job"]), plan["source_root"],
                                  plan["prepared_object"], plan["manual_phases"],
                                  plan["outputs"]["A"], plan["outputs"]["B"],
                                  require_fresh_outputs=False)
    if expected != plan["commands"]:
        raise ValueError("A/B commands differ from the frozen contact-only intervention")
    for row in plan["input_bindings"] + plan["fitter_source_bindings"]:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"Frozen input/source bytes changed: {path}")


def run(args):
    plan_path = Path(args.plan).resolve()
    if digest(plan_path) != args.plan_sha256:
        raise ValueError("Execution plan bytes differ from the approved frozen identity")
    plan = read(plan_path)
    verify_plan(plan)
    output = Path(plan["outputs"][args.arm])
    stem = plan_path.parent / (plan_path.stem + "-" + args.arm)
    started, child_started, finished, log = [stem.with_name(stem.name + suffix) for suffix in (".started.json", ".child-started.json", ".finished.json", ".log")]
    if output.exists() or any(path.exists() for path in (started, child_started, finished, log)):
        raise ValueError("Preserve prior attempts; output and run receipts must not exist")
    record = {"recorded_at": datetime.now(timezone.utc).isoformat(), "arm": args.arm,
              "plan": str(plan_path), "plan_sha256": args.plan_sha256,
              "command": plan["commands"][args.arm], "cwd": plan["source_root"],
              "wrapper_pid": os.getpid(), "contact_enabled": args.arm == "B"}
    write_new(started, record)
    before = time.monotonic()
    with log.open("xb") as stream:
        child = subprocess.Popen(record["command"], cwd=record["cwd"], stdout=stream,
                                 stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        write_new(child_started, {**record, "child_pid": child.pid,
                                 "child_started_at": datetime.now(timezone.utc).isoformat()})
        returncode = child.wait()
    verify_plan(plan)
    record.update(finished_at=datetime.now(timezone.utc).isoformat(), child_pid=child.pid,
                  returncode=returncode, wall_seconds=time.monotonic() - before,
                  frozen_inputs_and_sources_unchanged=True,
                  status="fit_completed" if returncode == 0 else "contact_abstained" if returncode == 2 and (output / "contact-rejection.json").exists() else "fit_failed")
    write_new(finished, record)
    print(json.dumps(record), flush=True)
    raise SystemExit(returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    for name in ("reference-fit-job", "source-root", "object", "phases", "a-output", "b-output", "output-plan"):
        prep.add_argument("--" + name, type=Path, required=True)
    execute = modes.add_parser("run")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--plan-sha256", required=True)
    execute.add_argument("--arm", choices=("A", "B"), required=True)
    args = parser.parse_args()
    try:
        prepare(args) if args.mode == "prepare" else run(args)
    except (ValueError, KeyError) as error:
        parser.exit(2, f"Controlled pair rejected: {error}\n")


if __name__ == "__main__":
    main()
