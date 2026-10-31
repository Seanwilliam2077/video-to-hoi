"""Independent synthetic orchestration and failure checks; no challenge assets."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    cli = module_at("refined_egohand_cli", ROOT / "tools/evaluate_refined_egohand_pilots.py")
    policy = cli.load_policy(ROOT)
    monkeypatch.setattr(cli, "load_policy", lambda _: policy)
    count = 10
    K = np.diag([100., 100., 1.])
    joints = np.zeros((count, 2, 21, 3))
    joints[..., 2] = 1.
    joints[:, :, :, 0] = np.arange(21) * .002
    joints[:, :, :, 1] = np.arange(21) * .001
    points = policy.project_hands(joints, K)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"Independent synthetic source identity; not an actual video")
    source_hash = policy.digest(source)
    model = tmp_path / "model.pt"
    model.write_bytes(b"Independent synthetic injected decoder identity")
    model_hash = policy.digest(model)

    def write(path, data):
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def binding(path):
        return {"path": str(path), "sha256": policy.digest(path)}

    model_manifest = tmp_path / "original-manifest.json"
    write(model_manifest, {"track": 1, "episode": 16, "video_frames": count,
          "video_sha256": source_hash, "camera_intrinsics": K.tolist(),
          "mhr_model": str(model), "mhr_model_sha256": model_hash})
    config = {"min_confidence": .05, "optimize_wrist_chain": True,
              "iterations": 600, "learning_rate": .001, "holdout_stride": 5,
              "max_delta_radians": .65, "arm_delta_limit_radians": .1,
              "wrist_max_displacement_metres": .04}
    reference = tmp_path / "preserved-fit-report.json"
    write(reference, {"config": config})
    baseline = tmp_path / "baseline/episode_000016_human.npz"
    baseline.parent.mkdir()
    baseline.write_bytes(b"Independent synthetic baseline parameters; decoder injected")
    baseline_hash = policy.digest(baseline)
    evidence = {}
    observations = {}
    for role, kind in (("legacy_mp", "image_detector"), ("fresh_mp", "image_detector"),
                       ("revised_ego", "generated_proposal")):
        confidence = np.ones((count, 2, 21))
        confidence[{"legacy_mp": 1, "fresh_mp": 2, "revised_ego": 3}[role]] = 0
        crop = np.broadcast_to(np.array([-10., -10., 20., 20.]), (count, 2, 4)).copy()
        if role == "legacy_mp":
            crop[..., :2] -= 1
        arrays = {"frame_indices": np.arange(count), "side_order": np.array(["left", "right"]),
                  "joints2d": points.copy(), "confidence": confidence,
                  "crop_boxes": crop, "source_video_sha256": np.array(source_hash)}
        if role == "revised_ego":
            values = np.ones((count, 2))
            values[::5] = 0
            arrays.update(shared_wrist_anchor_weights=values.copy(), shared_visibility_gate_weights=values.copy())
        path = tmp_path / f"{role}-observations.npz"
        np.savez(path, **arrays)
        observations[role] = arrays
        evidence[role] = {**binding(path), "kind": kind,
                          "shared_inputs": [] if role == "legacy_mp" else ["fresh MP wrist/gate shared with Ego"]}
    variants = []
    directories = {}
    reports = {}
    for role in ("legacy_mp", "legacy_ego", "fresh_mp", "revised_ego"):
        directory = tmp_path / role
        directory.mkdir()
        directories[role] = directory
        path = directory / "episode_000016_human.npz"
        path.write_bytes(b"Independent synthetic candidate " + role.encode())
        report = {"source_video_sha256": source_hash, "baseline_sha256": baseline_hash,
                  "mhr_model_sha256": model_hash, "config": config.copy(),
                  "heldout_frame_indices": [0, 5]}
        entry = {"role": role, "label": role, **binding(path)}
        if role in ("fresh_mp", "revised_ego"):
            report["observations_sha256"] = evidence[role]["sha256"]
            entry["evidence_role"] = role
            weights = observations[role]["confidence"].copy()
            weights[::5] = 0
            np.save(directory / "optimization_weights.npy", weights)
            np.save(directory / "active_hands.npy", (weights[:, :, 1:].sum(-1) >= 8))
            entry["weights"] = binding(directory / "optimization_weights.npy")
            entry["support"] = binding(directory / "active_hands.npy")
        report_path = directory / "report.json"
        write(report_path, report)
        entry["report_sha256"] = policy.digest(report_path)
        reports[role] = report
        variants.append(entry)
    original_active = tmp_path / "original-active.npy"
    np.save(original_active, np.array([[False, False]] + [[True, True]] * (count - 1)))
    task = {"track": 1, "all30_execution_authorized": False, "authorized_episodes": [16, 11],
            "episodes": [{"episode": 16, "frames": count, "video": str(source), "video_sha256": source_hash,
                          "manifest": str(model_manifest), "manifest_sha256": policy.digest(model_manifest),
                          "baseline": binding(baseline), "fit_configuration_reference": binding(reference),
                          "controlled_pair": ["fresh_mp", "revised_ego"], "variants": variants,
                          "evidence": evidence, "original_bindings": {"old_active_hands": binding(original_active)}}]}
    inputs = tmp_path / "inputs.json"
    write(inputs, task)
    windows = tmp_path / "windows.json"
    write(windows, {"track": 1, "episode": 16, "source_video_sha256": source_hash,
                    "windows": [{"first_original_frame_id": 1, "last_original_frame_id": 8,
                                 "side": "both", "label": "Independent synthetic inspection interval"}]})
    decoded = []
    failure = set()

    def decode(directory, *args, **kwargs):
        role = directory.name
        decoded.append(role)
        if role in failure:
            raise ValueError("Injected native parity failure")
        pose = np.zeros((count, 136))
        if role != "revised_ego" and role != "baseline":
            pose[:, 70] = .01
        parameters = {"pose": pose, "shape": np.zeros(45), "scales": np.zeros(68)}
        return parameters, joints.copy(), {"passed": True, "frames_freshly_decoded": count,
            "interpretation": "Independent injected synthetic native contract; no real model loaded"}

    monkeypatch.setattr(policy, "decode_variant", decode)
    class FakeModel:
        def float(self):
            return self
        def eval(self):
            return self
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None,
        jit=SimpleNamespace(load=lambda *a, **k: FakeModel())))
    output = tmp_path / "evaluation.json"
    argv = ["--episode", "16", "--input-manifest", str(inputs), "--review-windows", str(windows),
            "--output", str(output), "--tool-root", str(ROOT)]

    def save_task():
        write(inputs, task)

    def update_evidence(role):
        path = Path(evidence[role]["path"])
        np.savez(path, **observations[role])
        evidence[role]["sha256"] = policy.digest(path)
        if role in reports:
            reports[role]["observations_sha256"] = evidence[role]["sha256"]
            update_report(role)
        save_task()

    def update_report(role):
        path = directories[role] / "report.json"
        write(path, reports[role])
        next(v for v in variants if v["role"] == role)["report_sha256"] = policy.digest(path)
        save_task()

    return SimpleNamespace(cli=cli, policy=policy, task=task, item=task["episodes"][0], argv=argv,
        output=output, inputs=inputs, decoded=decoded, observations=observations, reports=reports,
        evidence=evidence, directories=directories, variants=variants, update_evidence=update_evidence,
        update_report=update_report, save_task=save_task, binding=binding, source=source, failure=failure)


def test_fresh_pair_and_historical_crops_are_separate_and_zero_update_is_not_a_win(experiment):
    assert experiment.cli.main(experiment.argv) == 0
    result = json.loads(experiment.output.read_text())
    assert set(experiment.decoded) == {"baseline", "legacy_mp", "legacy_ego", "fresh_mp", "revised_ego"}
    assert len(experiment.decoded) == 5
    assert result["frozen_conditions"]["same_fresh_source_crop_bounds"]
    assert not result["frozen_conditions"]["crop_comparison_by_evidence_role"]["legacy_mp"]["identical_to_fresh_pair"]
    assert not result["frozen_conditions"]["positive_target_masks_identical"]
    assert result["retained_native_updates"]["revised_ego"]["retained_zero_update"]
    assert "abstention" in result["retained_native_updates"]["revised_ego"]["interpretation"]
    generated = result["generated_target_fit_consistency"]["revised_ego"]
    assert "no motion or acceptance guards" in generated["interpretation"]
    assert "candidate_image_nonregression" not in json.dumps(generated)
    assert "revised_ego" not in result["image_reference_motion_screens"]
    assert "independent_observed" not in json.dumps(result["image_reference_motion_screens"])
    assert result["image_reference_motion_screens"]["fresh_mp"]["all_adjacent_original_frames"]["edge_count"] == 18
    assert not result["quality_accepted"] and not result["official_score_available"]
    assert not result["automatic_candidate_promotion"] and not result["all30_execution_authorized"]


@pytest.mark.parametrize("change,message", [
    ("crop", "fresh source crop"),
    ("config", "fit configuration"),
    ("holdout_target", "heldout image target"),
    ("shared_anchor_holdout", "Heldout fresh MP"),
    ("native_hash", "native parameters binding"),
    ("source_hash", "source video binding"),
])
def test_contract_failures_precede_any_native_decode(experiment, change, message):
    if change == "crop":
        experiment.observations["revised_ego"]["crop_boxes"][4, 0, 0] += 1
        experiment.update_evidence("revised_ego")
    elif change == "config":
        for role in ("fresh_mp", "revised_ego"):
            experiment.reports[role]["config"]["iterations"] = 1
            experiment.update_report(role)
    elif change == "holdout_target":
        variant = next(v for v in experiment.variants if v["role"] == "revised_ego")
        path = Path(variant["weights"]["path"])
        weights = np.load(path)
        weights[0] = 1
        np.save(path, weights)
        variant["weights"] = experiment.binding(path)
        experiment.save_task()
    elif change == "shared_anchor_holdout":
        experiment.observations["revised_ego"]["shared_wrist_anchor_weights"][0] = 1
        experiment.update_evidence("revised_ego")
    elif change == "native_hash":
        Path(experiment.variants[-1]["path"]).write_bytes(b"Changed independent synthetic native bytes")
    else:
        experiment.source.write_bytes(b"Changed independent synthetic source identity")
    with pytest.raises(ValueError, match=message):
        experiment.cli.main(experiment.argv)
    assert experiment.decoded == []
    assert not experiment.output.exists()


def test_native_decode_failure_cannot_write_completed_evaluation(experiment):
    experiment.failure.add("fresh_mp")
    with pytest.raises(ValueError, match="native parity failure"):
        experiment.cli.main(experiment.argv)
    assert "fresh_mp" in experiment.decoded
    assert not experiment.output.exists()


def test_generic_reduction_matches_original_frozen_policy_for_each_variant(experiment):
    policy = experiment.policy
    base = np.zeros((4, 2, 21, 3))
    base[..., 2] = 1
    first, second = base.copy(), base.copy()
    first[1:, :, 1:, 0] = .01
    second[2:, :, :, 1] = .02
    K = np.diag([100., 100., 1.])
    observations = policy.project_hands(first, K)
    confidence = np.ones((4, 2, 21))
    confidence[2] = 0
    rows = policy.motion_rows(base, {"old_candidate": first, "candidate": second}, observations,
                              confidence, K, .05, np.array([True, False, False, False]), {(0, 0)})
    original = policy.summarize_motion(rows)
    generic = experiment.cli.summarize_multi_motion(rows, ["baseline", "old_candidate", "candidate"], policy)
    assert generic["variants"] == original["variants"]
    assert generic["absolute_wrist_motion"]["variants"] == original["absolute_wrist_motion"]["variants"]


def test_tiny_projected_palm_width_keeps_distances_and_invalidates_ratios(experiment):
    points = np.zeros((2, 2, 21, 2))
    points[..., 4, 0] = 8
    points[..., 8, 0] = 1
    points[..., 12, 0] = 2
    points[..., 17, 0] = .12
    result = experiment.cli.opening_curves(points)
    row = result["sides"]["right"]["rows"][0]
    assert row["thumb_index_distance"] == 7
    assert row["palm_width"] == .12
    assert not row["normalized_ratio_valid"]
    assert row["thumb_index_over_palm"] is None
    assert result["sides"]["right"]["tiny_palm_denominator_frames"] == 2


def test_existing_evaluation_is_preserved(experiment):
    experiment.output.write_text("preserve independent earlier attempt")
    with pytest.raises(SystemExit):
        experiment.cli.main(experiment.argv)
    assert experiment.decoded == []
    assert experiment.output.read_text() == "preserve independent earlier attempt"
