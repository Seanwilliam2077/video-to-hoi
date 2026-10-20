"""Independent synthetic process and transfer failures for the Human runner."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("human_batch", Path(__file__).resolve().parents[1] / "tools/run_human_batch.py")
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def script(tmp_path, text):
    path = tmp_path / "worker.py"
    path.write_text(text, encoding="utf-8")
    return path


def test_completed_process_reused_and_output_tampering_rejected(tmp_path):
    worker = script(tmp_path, "import pathlib,sys\np=pathlib.Path(sys.argv[1]);p.mkdir();(p/'result.txt').write_text('synthetic geometry')\n")
    command = lambda out: [sys.executable, worker, out]
    first = batch.execute_stage(tmp_path / "episode", "fit", command, inputs={"synthetic": True}, source_paths=[worker])
    second = batch.execute_stage(tmp_path / "episode", "fit", command, inputs={"synthetic": True}, source_paths=[worker])
    assert first == second
    assert len(list((tmp_path / "episode/fit").glob("*.started.json"))) == 1
    (first / "result.txt").write_text("damaged transfer")
    with pytest.raises(ValueError, match="Output bytes changed"):
        batch.execute_stage(tmp_path / "episode", "fit", command, inputs={"synthetic": True}, source_paths=[worker])


def test_failed_inference_continues_partial_cache_and_preserves_receipt(tmp_path):
    worker = script(tmp_path, """import pathlib,sys
p=pathlib.Path(sys.argv[1]);p.mkdir(exist_ok=True)
if not (p/'manifest.json').exists():
 (p/'manifest.json').write_text('{}');sys.exit(3)
(p/'completed.json').write_text('{}')
""")
    command = lambda out: [sys.executable, worker, out]
    with pytest.raises(batch.StageError, match="exited 3"):
        batch.execute_stage(tmp_path / "episode", "inference", command, inputs={"synthetic": True}, source_paths=[worker], resume_inference=True)
    old_receipt = tmp_path / "episode/inference/attempt-0001.finished.json"
    previous = old_receipt.read_bytes()
    output = batch.execute_stage(tmp_path / "episode", "inference", command, inputs={"synthetic": True}, source_paths=[worker], resume_inference=True)
    assert output.name == "attempt-0001"
    assert (output / "completed.json").is_file()
    assert old_receipt.read_bytes() == previous
    assert json.loads(old_receipt.read_text())["returncode"] == 3
    assert json.loads((old_receipt.parent / "attempt-0002.finished.json").read_text())["status"] == "completed"


def test_changed_input_starts_new_attempt_and_preserves_old_result(tmp_path):
    worker = script(tmp_path, "import pathlib,sys\np=pathlib.Path(sys.argv[1]);p.mkdir();(p/'result.txt').write_text('synthetic')\n")
    command = lambda out: [sys.executable, worker, out]
    first = batch.execute_stage(tmp_path / "episode", "fit", command, inputs={"source_video_sha256": "a" * 64}, source_paths=[worker])
    second = batch.execute_stage(tmp_path / "episode", "fit", command, inputs={"source_video_sha256": "b" * 64}, source_paths=[worker])
    assert first != second
    assert (first / "result.txt").read_text() == "synthetic"


def test_subset_receipt_cannot_authorize_all30_runner(tmp_path):
    gate = tmp_path / "receiver.json"
    gate.write_text(json.dumps({"schema_version": "body_handoff_checks.v1", "passed": True,
                                "all_30_body_inputs_verified": False, "videos_verified": 1}))
    with pytest.raises(ValueError, match="all-30 receiver gate"):
        batch.load_verified_episode(tmp_path, gate, 16)


def test_changed_received_camera_rejected_before_execution(tmp_path):
    """A valid-looking old receipt must not authorize changed camera bytes."""
    episode = tmp_path / "body-inputs/episode_000016"
    episode.mkdir(parents=True)
    video = tmp_path / "data/v2d/track_1/videos/episode_000016.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"independent synthetic video identity fixture")
    camera = episode / "intrinsics.json"
    camera.write_text('{"fx":100}')
    original_camera = batch.digest(camera)
    identity = {"schema_version": "body_input.v1", "episode_index": 16,
                "source": {"video_path": "videos/episode_000016.mp4", "video_sha256": batch.digest(video)},
                "artifacts": {"intrinsics.json": original_camera}}
    (episode / "input.json").write_text(json.dumps(identity))
    rows = [{"episode_index": i, "body_inputs_verified": True, "frame_count": 2,
             "input_sha256": batch.digest(episode / "input.json"), "video_sha256": batch.digest(video)} for i in range(30)]
    gate = tmp_path / "receiver.json"
    gate.write_text(json.dumps({"schema_version": "body_handoff_checks.v1", "passed": True,
                                "all_30_body_inputs_verified": True, "videos_verified": 30, "episodes": rows}))
    camera.write_text('{"fx":999}')
    with pytest.raises(ValueError, match="artifact changed"):
        batch.load_verified_episode(tmp_path, gate, 16)


def test_no_independent_hand_evidence_preserves_baseline_without_running_fit(tmp_path):
    baseline = tmp_path / "baseline/episode_000000_human.npz"
    baseline.parent.mkdir()
    # Synthetic file identity fixture: this test never loads challenge assets.
    baseline.write_bytes(b"synthetic native baseline identity")
    (baseline.parent / "review_arrays").mkdir()
    (baseline.parent / "review_arrays/vertices_m.npy").write_bytes(b"synthetic rendered identity")
    (baseline.parent / "report.json").write_text('{"track":1,"episode":0}')
    observation = tmp_path / "observations.npz"
    np.savez(observation, confidence=np.zeros((7, 2, 21), dtype=np.float32))
    marker = tmp_path / "optimizer-was-run"
    worker = script(tmp_path, f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    output = tmp_path / "refined"
    batch.refine_or_preserve(baseline, observation, output, [sys.executable, str(worker)])
    assert not marker.exists()
    assert (output / baseline.name).read_bytes() == baseline.read_bytes()
    receipt = json.loads((output / "hand-refinement-abstention.json").read_text())
    assert receipt["no_hand_optimizer_run"] is True
    assert receipt["baseline_preserved_unchanged"] is True


def test_runtime_manifest_cannot_hide_changed_source_at_same_path(tmp_path):
    source = tmp_path / "generic-runtime"
    source.mkdir()
    module = source / "generic_model.py"
    module.write_text("# independent synthetic runtime fixture\n")
    manifest = source / "installed_manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": module.name, "installed_sha256": batch.digest(module)}]}))
    original = batch.verified_sam_source(source)
    module.write_text("# modified source, same manifest identity\n")
    assert batch.digest(manifest) == original
    with pytest.raises(ValueError, match="differs from its installed manifest"):
        batch.verified_sam_source(source)


def heldout_report(left=(2, 3., 3.), right=(12, 5., 4.)):
    result = {"before_heldout": {}, "after_heldout": {}}
    for side, (frames, before, after) in zip(("left", "right"), (left, right)):
        result["before_heldout"][side] = {"observed_frames": frames, "weighted_reprojection_pixels": before}
        result["after_heldout"][side] = {"observed_frames": frames, "weighted_reprojection_pixels": after}
    return result


def test_sparse_left_hand_regression_vetoes_well_observed_right_improvement():
    decision = batch.hand_candidate_guards(heldout_report(left=(2, 3., 7.), right=(12, 5., 4.)))
    assert decision["per_side"]["left"]["sufficient"] is False
    assert decision["per_side"]["left"]["nonregression"] is False
    assert decision["per_side"]["right"]["sufficient"] is True
    assert decision["per_side"]["right"]["nonregression"] is True
    assert decision["selected_candidate"] == "baseline"


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -1., None])
def test_nonfinite_or_invalid_heldout_metric_cannot_select_a_candidate(metric):
    report = heldout_report()
    report["after_heldout"]["left"]["weighted_reprojection_pixels"] = metric
    with pytest.raises(ValueError, match="finite and nonnegative"):
        batch.hand_candidate_guards(report)


def test_sparse_nonregression_stays_uncertain_without_vetoing_supported_side():
    decision = batch.hand_candidate_guards(heldout_report())
    assert decision["per_side"]["left"]["sufficient"] is False
    assert decision["per_side"]["left"]["nonregression"] is True
    assert decision["selected_candidate"] == "refined"
    unsupported = batch.hand_candidate_guards(heldout_report(left=(2, 3., 2.), right=(4, 5., 4.)))
    assert unsupported["selected_candidate"] == "baseline"


def snapshot_fixture(tmp_path):
    root = tmp_path / "snapshot"
    names = {"tools/run_sam3d_body.py", "tools/finalize_human.py", "tools/refine_human_hands.py",
             "tools/review_human.py", "tools/compare_human_hands.py", "tools/human_hand/observations.py",
             "tools/human_review/template.html", "src/v2hoi/__init__.py", "src/v2hoi/sam_body_inputs.py",
             "src/v2hoi/human_result.py", "src/v2hoi/hand_refinement.py"}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# independent synthetic snapshot source fixture\n")
    manifest = {"files": {name: batch.digest(root / name) for name in names}}
    (root / "snapshot.json").write_text(json.dumps(manifest))
    return root


def test_stage_snapshot_rejects_changed_source_undeclared_code_and_path_escape(tmp_path):
    root = snapshot_fixture(tmp_path)
    assert batch.verified_stage_snapshot(root)["files_verified"] == 11
    extra = root / "src/v2hoi/undeclared.py"
    extra.write_text("# independent undeclared module fixture\n")
    with pytest.raises(ValueError, match="undeclared files"):
        batch.verified_stage_snapshot(root)
    extra.unlink()
    module = root / "src/v2hoi/human_result.py"
    original = module.read_bytes()
    module.write_text("# changed after source freeze\n")
    with pytest.raises(ValueError, match="changed or missing"):
        batch.verified_stage_snapshot(root)
    module.write_bytes(original)
    manifest = json.loads((root / "snapshot.json").read_text())
    manifest["files"]["../outside.py"] = "a" * 64
    (root / "snapshot.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe stage snapshot"):
        batch.verified_stage_snapshot(root)


def test_new_driver_checks_keep_unchanged_external_stage_cache(tmp_path):
    """A check-only driver fix must not silently change or recompute model tools."""
    external_tool = script(tmp_path, "import pathlib,sys\np=pathlib.Path(sys.argv[1]);p.mkdir();(p/'model-result').write_text('synthetic model stage')\n")
    driver = tmp_path / "new-driver.py"
    driver.write_text("import pathlib,sys\np=pathlib.Path(sys.argv[1]);p.mkdir();(p/'guard-result').write_text('guard version 1')\n")
    model_command = lambda out: [sys.executable, external_tool, out]
    check_command = lambda out: [sys.executable, driver, out]
    episode = tmp_path / "episode"
    first_model = batch.execute_stage(episode, "baseline", model_command, inputs={"synthetic": True}, source_paths=[external_tool])
    first_check = batch.execute_stage(episode, "checks", check_command, inputs={"model_result": batch.digest(first_model / "model-result")}, source_paths=[driver])
    original_model_receipt = (episode / "baseline/attempt-0001.finished.json").read_bytes()
    driver.write_text("import pathlib,sys\np=pathlib.Path(sys.argv[1]);p.mkdir();(p/'guard-result').write_text('guard version 2, including sparse veto')\n")
    second_model = batch.execute_stage(episode, "baseline", model_command, inputs={"synthetic": True}, source_paths=[external_tool])
    second_check = batch.execute_stage(episode, "checks", check_command, inputs={"model_result": batch.digest(second_model / "model-result")}, source_paths=[driver])
    assert first_model == second_model
    assert first_check != second_check
    assert (episode / "baseline/attempt-0001.finished.json").read_bytes() == original_model_receipt
    assert len(list((episode / "baseline").glob("*.started.json"))) == 1


def test_comparison_override_uses_its_own_helper_and_template_without_changing_default(tmp_path, monkeypatch):
    """Renderer changes do not redirect model-stage sources or default commands."""
    stage = tmp_path / "model-snapshot"
    override = tmp_path / "new-renderer-snapshot"
    for root in (stage, override):
        (root / "tools/human_review").mkdir(parents=True)
        (root / "tools/compare_human_hands.py").write_text("# independent renderer fixture\n")
        (root / "tools/review_human.py").write_text("# independent renderer helper fixture\n")
        (root / "tools/human_review/template.html").write_text("<p>independent template fixture</p>\n")
    monkeypatch.setattr(batch, "ROOT", stage)
    default_tool, default_sources = batch.hand_comparison_sources()
    actual_tool, actual_sources = batch.hand_comparison_sources(override / "tools/compare_human_hands.py")
    assert default_tool == stage / "tools/compare_human_hands.py"
    assert default_sources == [stage / "tools/compare_human_hands.py", stage / "tools/review_human.py", stage / "tools/human_review/template.html"]
    assert actual_tool == override / "tools/compare_human_hands.py"
    assert all(path.is_relative_to(override) for path in actual_sources)
    assert batch.ROOT == stage
    assert batch.hand_comparison_sources() == (default_tool, default_sources)
