"""Fresh native comparison with separately identified MP and EgoHand evidence.

EP11/EP16 comparison helper. No fitting, publication, promotion or batch launch.
Original policy functions are imported from a pinned source file and recorded.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROLE_NAMES = {"old_candidate": "mediapipe", "candidate": "egohand"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def renamed(value):
    if isinstance(value, dict):
        return {ROLE_NAMES.get(k, k): renamed(v) for k, v in value.items()}
    if isinstance(value, list):
        return [renamed(v) for v in value]
    return value


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def generated_target_vocabulary(value):
    """Prevent reused diagnostic labels from calling model proposals independent."""
    if isinstance(value, dict):
        def key_name(key):
            key = key.replace("independently_observed", "generated_target")
            key = key.replace("independent_observed", "generated_target")
            key = key.replace("observed", "target")
            return key
        return {key_name(k): generated_target_vocabulary(v) for k, v in value.items()}
    if isinstance(value, list):
        return [generated_target_vocabulary(v) for v in value]
    return value


def edge_list(edges):
    return [{"from_original_frame_id": frame, "to_original_frame_id": frame + 1,
             "side": ("left", "right")[side]} for frame, side in sorted(edges)]


def load_policy(tool_root):
    source = tool_root / "tools/evaluate_human_hand_pilots.py"
    spec = importlib.util.spec_from_file_location("frozen_pilot_evaluator", source)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=(11, 16), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--egohand-observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, default=ROOT)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--review-windows", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Override decoder location; bytes must match original manifest")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model-batch", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Preserve existing evaluation; output must be new")
    if args.threads < 1 or args.model_batch < 1:
        parser.error("Positive threads and model-batch required")
    source = args.tool_root / "tools/evaluate_human_hand_pilots.py"
    evaluator = load_policy(args.tool_root)
    sys.path.insert(0, str(args.tool_root / "src"))
    manifest_input = read(args.input_manifest)
    if manifest_input.get("track") != 1 or manifest_input.get("all30_execution_authorized") is not False:
        raise ValueError("Wrong task scope")
    item = next(x for x in manifest_input["episodes"] if x["episode"] == args.episode)
    manifest = read(item["manifest"])
    if manifest.get("track") != 1 or manifest["episode"] != args.episode:
        raise ValueError("Wrong episode manifest")
    if evaluator.digest(Path(item["manifest"])) != item["manifest_sha256"]:
        raise ValueError("Original manifest identity changed")
    expected = np.arange(manifest["video_frames"])
    baseline = Path(item["baseline"]["path"]).parent
    mediapipe = Path(item["mediapipe_candidate"]).parent
    original_observations = Path(item["original_observations"])
    model_path = args.model or Path(manifest["mhr_model"])
    model_hash = evaluator.digest(model_path)
    if model_hash != manifest["mhr_model_sha256"]:
        raise ValueError("Original native decoder identity changed")
    for path, digest in [(original_observations, item["original_observations_sha256"]),
                         (Path(item["baseline"]["path"]), item["baseline"]["sha256"]),
                         (Path(item["mediapipe_candidate"]), item["mediapipe_candidate_sha256"])]:
        if evaluator.digest(path) != digest:
            raise ValueError(f"Frozen comparison input changed: {path}")
    evidence = {"mediapipe": load_npz(original_observations),
                "egohand": load_npz(args.egohand_observations)}
    points, confidence = {}, {}
    for name, observations in evidence.items():
        if str(observations["source_video_sha256"].item()) != manifest["video_sha256"]:
            raise ValueError(f"Wrong source-video identity: {name}")
        points[name], confidence[name] = evaluator.validate_evidence(observations, expected)
    if not np.array_equal(evidence["mediapipe"]["crop_boxes"], evidence["egohand"]["crop_boxes"]):
        raise ValueError("Original crop bounds changed")
    reports = {"old_candidate": read(mediapipe / "report.json"),
               "candidate": read(args.candidate / "report.json")}
    baseline_hash = item["baseline"]["sha256"]
    for key, report in reports.items():
        if report.get("source_video_sha256") != manifest["video_sha256"] or report.get("baseline_sha256") != baseline_hash:
            raise ValueError(f"Candidate provenance differs: {key}")
        obs_path = original_observations if key == "old_candidate" else args.egohand_observations
        if report.get("observations_sha256") != evaluator.digest(obs_path):
            raise ValueError(f"Candidate observation binding differs: {key}")
    holdout_ids = reports["old_candidate"]["heldout_frame_indices"]
    if reports["candidate"]["heldout_frame_indices"] != holdout_ids:
        raise ValueError("Original heldout frame split changed")
    heldout = np.isin(expected, holdout_ids)
    if reports["candidate"]["config"] != reports["old_candidate"]["config"]:
        raise ValueError("Controlled fit configuration changed")
    threshold = reports["old_candidate"]["config"]["min_confidence"]
    support = {name: np.load(path / "active_hands.npy", allow_pickle=False) for name, path in
               (("mediapipe", mediapipe), ("egohand", args.candidate))}
    edges = {name: set(evaluator.support_edges(value)) for name, value in support.items()}
    older = Path(item["original_bindings"]["old_active_hands"]["path"])
    if evaluator.digest(older) != item["original_bindings"]["old_active_hands"]["sha256"]:
        raise ValueError("Frozen original MediaPipe support changed")
    edges["original_rejected_mediapipe"] = set(evaluator.support_edges(np.load(older, allow_pickle=False)))
    weights = {name: np.load(path / "optimization_weights.npy", allow_pickle=False) for name, path in
               (("mediapipe", mediapipe), ("egohand", args.candidate))}
    for name, value in weights.items():
        if value.shape != confidence[name].shape or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("Invalid optimization weights")
        if np.any(value[heldout] > 0) or np.any((value > 0) & (confidence[name] < threshold)):
            raise ValueError(f"Unsupported or heldout image target used: {name}")
    import torch
    torch.set_num_threads(args.threads)
    model = torch.jit.load(str(model_path), map_location=args.device).float().eval()
    parameters, hands, parity = {}, {}, {}
    for name, path in (("baseline", baseline), ("old_candidate", mediapipe), ("candidate", args.candidate)):
        print(f"Fresh original-frame native decode: {ROLE_NAMES.get(name, name)}", flush=True)
        parameters[name], hands[name], parity[name] = evaluator.decode_variant(
            path, manifest, model, model_hash, device=args.device, batch=args.model_batch)
    intrinsics = np.asarray(manifest["camera_intrinsics"], dtype=np.float64)
    projected = {name: evaluator.project_hands(value, intrinsics) for name, value in hands.items()}
    motion = {}
    image = {}
    window_path = args.review_windows
    windows = read(window_path)
    if windows.get("track") != 1 or windows.get("episode") != args.episode or windows["source_video_sha256"] != manifest["video_sha256"]:
        raise ValueError("Manual review windows identify a different video")
    for evidence_name in ("mediapipe", "egohand"):
        rows = evaluator.motion_rows(hands["baseline"], {name: hands[name] for name in ("old_candidate", "candidate")},
            points[evidence_name], confidence[evidence_name], intrinsics, threshold, heldout,
            edges["original_rejected_mediapipe"])
        motion[evidence_name] = {
            "interpretation": ("Independent of EgoHand fit but method-dependent MediaPipe reference; not ground truth"
                               if evidence_name == "mediapipe" else
                               "Self-consistency against generated EgoHand targets; not independent accuracy"),
            "all_adjacent_original_frames": evaluator.summarize_motion(rows),
            "original_mediapipe_boundaries": evaluator.summarize_motion([r for r in rows if r["original_support_boundary"]]),
            "current_mediapipe_boundaries": evaluator.summarize_motion([r for r in rows if
                (r["from_original_frame_id"], ("left", "right").index(r["side"])) in edges["mediapipe"]]),
            "egohand_boundaries": evaluator.summarize_motion([r for r in rows if
                (r["from_original_frame_id"], ("left", "right").index(r["side"])) in edges["egohand"]]),
            "fast_observed_motion": evaluator.summarize_motion([r for r in rows if r["fast_observed_motion"]]),
            "holdout_touching_observed_motion": evaluator.summarize_motion([r for r in rows if
                r["touches_original_holdout"] and r["independent_observed_motion_sufficient"]]),
            "manual_review_windows": evaluator.review_window_metrics(windows["windows"], rows, len(expected)),
            "all_adjacent_frame_rows": rows,
        }
        image[evidence_name] = evaluator.image_agreement(projected, points[evidence_name], confidence[evidence_name], heldout)
        if evidence_name == "egohand":
            motion[evidence_name] = generated_target_vocabulary(motion[evidence_name])
            image[evidence_name] = generated_target_vocabulary(image[evidence_name])
    self_motion = {}
    for name, joints in hands.items():
        self_motion[name] = {}
        for side, label in enumerate(("left", "right")):
            relative_m = evaluator.relative(joints)[:, side]
            relative_px = evaluator.relative(projected[name])[:, side]
            self_motion[name][label] = {
                "wrist_relative_step_px": evaluator.summary(np.linalg.norm(np.diff(relative_px, axis=0), axis=-1)),
                "wrist_relative_second_difference_px_per_frame_squared": evaluator.summary(np.linalg.norm(np.diff(relative_px, n=2, axis=0), axis=-1)),
                "wrist_relative_step_native_estimated_metres": evaluator.summary(np.linalg.norm(np.diff(relative_m, axis=0), axis=-1)),
                "wrist_relative_second_difference_native_estimated_metres_per_frame_squared": evaluator.summary(np.linalg.norm(np.diff(relative_m, n=2, axis=0), axis=-1)),
                "interpretation": "Prediction-only motion, not reference-relative acceleration error; freezing can reduce this quantity",
            }
    preservation = {name: evaluator.parameter_preservation(parameters["baseline"], parameters[name],
        bool(reports[name]["config"]["optimize_wrist_chain"])) for name in reports}
    result = {
        "schema_version": "egohand_mediapipe_pilot_evaluation.v1", "track": 1, "episode": args.episode,
        "frames": len(expected), "original_frame_coverage_complete": True,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_video_sha256": manifest["video_sha256"], "mhr_model_sha256": model_hash,
        "evaluation_helper_sha256": evaluator.digest(Path(__file__)),
        "reused_independent_policy_source": {"path": str(source), "sha256": evaluator.digest(source)},
        "unchanged_original_guard_policy": evaluator.POLICY,
        "input_bindings": {name: {"path": str(path), "sha256": evaluator.digest(path)} for name, path in {
            "task_inputs": args.input_manifest, "manifest": Path(item["manifest"]),
            "mediapipe_observations": original_observations, "egohand_observations": args.egohand_observations,
            "mediapipe_report": mediapipe / "report.json", "egohand_report": args.candidate / "report.json",
            "review_windows": window_path}.items()},
        "role_labels": {"baseline": "Original fixed-identity SAM MHR", "mediapipe": "Current bounded MediaPipe candidate",
                        "egohand": "EgoHand-driven MHR candidate"},
        "frozen_conditions": {"same_camera": True, "same_source_crop_bounds": True, "same_fit_config": True,
            "same_original_holdout_frame_ids": holdout_ids, "both_holdout_targets_excluded": True,
            "positive_target_masks_identical": bool(np.array_equal(weights["mediapipe"] > 0, weights["egohand"] > 0)),
            "different_observation_values_and_support_expected": True,
            "development_validation_not_untouched_test": True},
        "support_boundaries": {key: edge_list(value) for key, value in edges.items()},
        "native_decoder": parity, "parameter_preservation": preservation,
        "image_agreement_by_target_source": image,
        "motion_screens_by_target_source": motion, "prediction_self_motion": self_motion,
        "automatic_candidate_promotion": False, "all30_execution_authorized": False,
        "quality_accepted": False, "contact_quality_accepted": False, "official_score_available": False,
        "limitations": [
            "MediaPipe evidence can favour the MediaPipe-fitted method and is not ground truth.",
            "EgoHand targets are generated predictions; agreement with them is fit consistency, not independent accuracy.",
            "Passing unchanged added-correction screens does not establish real grasp, hidden motion, physical contact or calibrated scale.",
            "Native decoder parity is representation agreement, not geometric accuracy.",
            "Source-visible image review, grasp timing and opening/closure amplitude remain required.",
            "The all-30 batch remains outside this authorized pilot task.",
        ],
    }
    result = renamed(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "frames": len(expected), "real_quality_accepted": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
