"""Independent synthetic pilot orchestration; no Torch or challenge assets."""
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
    cli = module_at("egohand_pilot_cli", ROOT / "tools/evaluate_egohand_pilots.py")
    policy = cli.load_policy(ROOT)
    monkeypatch.setattr(cli, "load_policy", lambda _: policy)
    count = 10
    indices = np.arange(count)
    K = np.diag([100., 100., 1.])
    hands = np.zeros((count, 2, 21, 3))
    hands[..., 2] = 1.
    hands[:, :, 1:, 0] = np.arange(1, 21) * .001
    points = policy.project_hands(hands, K)
    camera_shape = (count, 2, 4)
    boxes = np.broadcast_to(np.array([0., 0., 50., 50.]), camera_shape).copy()
    base = tmp_path / "baseline"
    mp = tmp_path / "mediapipe"
    ego = tmp_path / "egohand"
    for directory in (base, mp, ego):
        directory.mkdir()
        (directory / "episode_000016_human.npz").write_bytes(b"independent synthetic placeholder; decoder injected")
    model = tmp_path / "model.pt"
    model.write_bytes(b"independent synthetic decoder identity; not an actual model")
    source_hash = "a" * 64

    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def observations(path, confidence, target_points):
        np.savez(path, frame_indices=indices, side_order=np.array(["left", "right"]),
                 joints2d=target_points, confidence=confidence, crop_boxes=boxes,
                 source_video_sha256=np.array(source_hash))

    mp_conf = np.ones((count, 2, 21))
    mp_conf[1] = 0
    ego_conf = np.ones_like(mp_conf)
    ego_conf[2] = 0
    mp_obs = tmp_path / "mp-observations.npz"
    ego_obs = tmp_path / "ego-observations.npz"
    observations(mp_obs, mp_conf, points)
    observations(ego_obs, ego_conf, points + .1)
    heldout = np.array([0, 5])
    config = {"min_confidence": .05, "optimize_wrist_chain": True, "iterations": 600}
    reports = {}
    for name, directory, obs, conf in (("mediapipe", mp, mp_obs, mp_conf), ("egohand", ego, ego_obs, ego_conf)):
        weights = conf.copy()
        weights[heldout] = 0
        np.save(directory / "optimization_weights.npy", weights)
        active = np.ones((count, 2), bool)
        active[1 if name == "mediapipe" else 2] = False
        np.save(directory / "active_hands.npy", active)
        reports[name] = {"source_video_sha256": source_hash,
                         "baseline_sha256": policy.digest(base / "episode_000016_human.npz"),
                         "observations_sha256": policy.digest(obs),
                         "heldout_frame_indices": heldout.tolist(), "config": config.copy()}
        write(directory / "report.json", reports[name])
    original_active = tmp_path / "old-support.npy"
    np.save(original_active, np.array([[False, False]] + [[True, True]] * (count - 1)))
    manifest = tmp_path / "manifest.json"
    write(manifest, {"track": 1, "episode": 16, "video_frames": count,
                     "video_sha256": source_hash, "camera_intrinsics": K.tolist(),
                     "mhr_model": str(model), "mhr_model_sha256": policy.digest(model)})
    inputs = tmp_path / "inputs.json"
    write(inputs, {"track": 1, "all30_execution_authorized": False, "episodes": [{
        "episode": 16, "manifest": str(manifest), "manifest_sha256": policy.digest(manifest),
        "baseline": {"path": str(base / "episode_000016_human.npz"), "sha256": policy.digest(base / "episode_000016_human.npz")},
        "mediapipe_candidate": str(mp / "episode_000016_human.npz"),
        "mediapipe_candidate_sha256": policy.digest(mp / "episode_000016_human.npz"),
        "original_observations": str(mp_obs), "original_observations_sha256": policy.digest(mp_obs),
        "original_bindings": {"old_active_hands": {"path": str(original_active), "sha256": policy.digest(original_active)}}
    }]})
    windows = tmp_path / "windows.json"
    write(windows, {"track": 1, "episode": 16, "source_video_sha256": source_hash,
                    "windows": [{"first_original_frame_id": 2, "last_original_frame_id": 8,
                                 "side": "both", "label": "Synthetic source-review interval"}]})
    decoded = []

    def decode(directory, *args, **kwargs):
        decoded.append(directory)
        parameters = {"pose": np.zeros((count, 136)), "shape": np.zeros(45), "scales": np.zeros(68)}
        return parameters, hands.copy(), {"passed": True, "frames_freshly_decoded": count,
                                           "interpretation": "Injected independent synthetic fixture, not a real decoder"}

    monkeypatch.setattr(policy, "decode_variant", decode)
    class FakeModel:
        def float(self):
            return self
        def eval(self):
            return self
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None,
        jit=SimpleNamespace(load=lambda *a, **k: FakeModel())))
    output = tmp_path / "evaluation.json"
    argv = ["--episode", "16", "--candidate", str(ego), "--egohand-observations", str(ego_obs),
            "--output", str(output), "--input-manifest", str(inputs), "--review-windows", str(windows),
            "--tool-root", str(ROOT)]
    return SimpleNamespace(cli=cli, policy=policy, argv=argv, output=output, decoded=decoded,
                           ego=ego, ego_obs=ego_obs, reports=reports, write=write,
                           original_active=original_active, manifest=manifest, model=model, inputs=inputs)


def test_different_evidence_is_allowed_without_calling_generated_targets_independent(experiment):
    assert experiment.cli.main(experiment.argv) == 0
    result = json.loads(experiment.output.read_text())
    assert len(experiment.decoded) == 3
    assert not result["frozen_conditions"]["positive_target_masks_identical"]
    assert result["frozen_conditions"]["both_holdout_targets_excluded"]
    assert set(result["native_decoder"]) == {"baseline", "mediapipe", "egohand"}
    assert result["motion_screens_by_target_source"]["mediapipe"]["all_adjacent_original_frames"]["edge_count"] == 18
    generated = result["motion_screens_by_target_source"]["egohand"]
    assert "not independent accuracy" in generated["interpretation"]
    assert "generated_target_motion_sufficient" in generated["all_adjacent_frame_rows"][0]
    assert "independent_observed_motion_sufficient" not in generated["all_adjacent_frame_rows"][0]
    assert not result["quality_accepted"]
    assert not result["official_score_available"]
    assert not result["all30_execution_authorized"]


@pytest.mark.parametrize("change,message", [
    ("holdout", "heldout image target"),
    ("config", "fit configuration"),
    ("support", "original MediaPipe support"),
    ("model", "decoder identity"),
    ("target_binding", "observation binding"),
])
def test_changed_comparison_contract_fails_before_any_native_decode(experiment, change, message):
    if change == "holdout":
        path = experiment.ego / "optimization_weights.npy"
        weights = np.load(path)
        weights[0] = 1
        np.save(path, weights)
    elif change == "config":
        report = experiment.reports["egohand"]
        report["config"]["iterations"] = 1
        experiment.write(experiment.ego / "report.json", report)
    elif change == "support":
        np.save(experiment.original_active, np.ones((10, 2), bool))
    elif change == "model":
        experiment.model.write_bytes(b"different synthetic decoder")
    else:
        report = experiment.reports["egohand"]
        report["observations_sha256"] = "0" * 64
        experiment.write(experiment.ego / "report.json", report)
    with pytest.raises(ValueError, match=message):
        experiment.cli.main(experiment.argv)
    assert experiment.decoded == []
    assert not experiment.output.exists()


def test_changed_camera_freeze_input_manifest_is_rejected(experiment):
    manifest = json.loads(experiment.manifest.read_text())
    manifest["camera_intrinsics"][0][0] += 1
    experiment.write(experiment.manifest, manifest)
    with pytest.raises(ValueError, match="manifest identity"):
        experiment.cli.main(experiment.argv)
    assert experiment.decoded == []


def test_prior_evaluation_is_never_overwritten(experiment):
    experiment.output.write_text("preserved previous attempt")
    with pytest.raises(SystemExit):
        experiment.cli.main(experiment.argv)
    assert experiment.output.read_text() == "preserved previous attempt"
    assert experiment.decoded == []
