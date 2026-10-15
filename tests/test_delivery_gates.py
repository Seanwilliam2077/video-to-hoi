"""Independent synthetic controller/lineage tests; no challenge assets or models."""
import hashlib
import json
import sys
from dataclasses import replace

import pytest

from v2hoi.check import main as check_main
from v2hoi.clips import load_clips
from v2hoi.contracts import Camera, Run
from v2hoi.gates import DeliveryGate, GateError, run_lock
from v2hoi.run import run_pipeline
from v2hoi.stages import make


@pytest.fixture
def catalog_root(tmp_path):
    root = tmp_path / "independent-synthetic" / "track_1"
    (root / "meta").mkdir(parents=True)
    info = {"robot_type": "video_only_object_tracking", "fps": 30, "chunks_size": 1000,
            "video_path": "videos/episode_{episode_index:06d}.mp4",
            "features": {"observation.images.exo_camera": {"shape": [8, 16, 3]}}}
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/episodes_metadata.jsonl").write_text("".join(
        json.dumps({"episode_index": e, "object": "synthetic-box", "frames": n}) + "\n"
        for e, n in [(3, 3), (4, 2)]), encoding="utf-8")
    (root / "meta/episodes.jsonl").write_text("".join(
        json.dumps({"episode_index": e, "length": n}) + "\n" for e, n in [(3, 3), (4, 2)]), encoding="utf-8")
    return root


@pytest.fixture
def completed(tmp_path, catalog_root):
    run = pipeline(tmp_path / "complete", catalog_root)
    return run, load_clips("track1", root=catalog_root), catalog_root


def pipeline(root, catalog_root, **kwargs):
    return run_pipeline(root, dataset_root=catalog_root, log=lambda *_: None, **kwargs)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def pointer(run, stage, episode=3):
    return run.root / f".gates/current/{stage}/{episode:06d}.json"


def receipt(run, stage):
    return run.root / f".gates/receipts/{read(pointer(run, stage))['id']}.json"


class NoOutputs:
    def run(self, run, clips):
        pass


def test_no_output_backend_stops_next_and_records_failure(tmp_path, catalog_root, monkeypatch):
    import v2hoi.run as runner
    calls = []

    def factory(stage, name):
        calls.append(stage)
        return NoOutputs()

    monkeypatch.setattr(runner, "make", factory)
    root = tmp_path / "no-outputs"
    with pytest.raises(GateError, match="required deliverables"):
        pipeline(root, catalog_root)
    run = Run.open(root)
    assert calls == ["inputs"]
    report = read(receipt(run, "inputs"))
    assert report["passed"] is False and report["outputs"] == {}
    assert any(c["passed"] is False for c in report["checks"])
    assert read(pointer(run, "inputs"))["state"] == "failed"
    assert not pointer(run, "human").exists()


def test_old_outputs_cannot_mask_noop_and_failure_shadows_upstream(completed, tmp_path):
    base, clips, _ = completed
    old = base.path(Camera, episode=3).read_bytes()
    with run_lock(base), pytest.raises(GateError):
        DeliveryGate(base, clips).execute("inputs", "noop", NoOutputs())
    assert base.path(Camera, episode=3).read_bytes() == old
    assert not DeliveryGate(base, clips).check("inputs")["passed"]
    # An independently failed local receipt must block even a passing upstream.
    other = Run(tmp_path / "other", "track1")
    with run_lock(other):
        DeliveryGate(other, clips).execute("inputs", "fake", make("inputs", "fake"))
    child = Run(tmp_path / "child", "track1", other)
    with run_lock(child), pytest.raises(GateError):
        DeliveryGate(child, clips).execute("inputs", "noop", NoOutputs())
    report = DeliveryGate(child, clips).check("human", before=True)
    assert not report["passed"] and "latest attempt is failed" in str(report)


def test_input_mutation_during_backend_blocks_publication(completed):
    run, clips, _ = completed

    class MutatesInput:
        def run(self, attempt, selected):
            path = attempt.upstream.path(Camera, episode=3)
            value = read(path)
            value["fx"] *= 1.1
            write(path, value)
            make("human", "fake").run(attempt, selected)

    with run_lock(run), pytest.raises(GateError, match="changed"):
        DeliveryGate(run, clips).execute("human", "mutator", MutatesInput())
    assert read(pointer(run, "human"))["state"] == "failed"


def test_identical_motion_rerun_invalidates_refine(completed):
    run, clips, root = completed
    old = (run.root / "motion/000003/motion.npz").read_bytes()
    pipeline(run.root, root, stages=["motion"])
    assert (run.root / "motion/000003/motion.npz").read_bytes() == old
    report = DeliveryGate(run, clips).check("refine")
    assert not report["passed"] and "stale predecessors" in str(report)
    assert not DeliveryGate(run, clips).check("export", before=True)["passed"]


def test_valid_but_changed_output_fails_hash_recheck(completed):
    run, clips, _ = completed
    path = run.path(Camera, episode=3)
    value = read(path)
    value["fx"] *= 1.1
    write(path, value)
    report = DeliveryGate(run, clips).check("inputs")
    assert not report["passed"] and "output changed" in str(report)


@pytest.mark.parametrize("mutation", ["hash", "empty_checks", "false_check", "false_top"])
def test_tampered_receipt_rejected(completed, mutation):
    run, clips, _ = completed
    path = receipt(run, "inputs")
    report = read(path)
    if mutation == "empty_checks":
        report["checks"] = []
    elif mutation == "false_check":
        report["checks"][0]["passed"] = False
    elif mutation == "false_top":
        report["passed"] = False
    else:
        report["backend"] = "edited"
    write(path, report)
    if mutation != "hash":
        for clip in clips:
            p = pointer(run, "inputs", clip.episode)
            value = read(p)
            value["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            write(p, value)
    assert not DeliveryGate(run, clips).check("inputs")["passed"]


def test_false_pass_flag_cannot_promote_failed_receipt(tmp_path, catalog_root):
    clips = load_clips("track1", root=catalog_root)
    run = Run(tmp_path / "failed", "track1")
    with run_lock(run), pytest.raises(GateError):
        DeliveryGate(run, clips).execute("inputs", "noop", NoOutputs())
    path = receipt(run, "inputs")
    report = read(path)
    report["passed"] = True
    write(path, report)
    for clip in clips:
        p = pointer(run, "inputs", clip.episode)
        value = read(p)
        value.update(state="passed", sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        write(p, value)
    assert not DeliveryGate(run, clips).check("inputs")["passed"]


def test_changed_clip_metadata_stales_receipt(completed):
    run, clips, _ = completed
    changed = [replace(c, fps=24) for c in clips]
    report = DeliveryGate(run, changed).check("inputs")
    assert not report["passed"] and "clip metadata" in str(report)


def test_unknown_native_gate_rejected(completed):
    run, clips, _ = completed
    with pytest.raises(GateError, match="no registered delivery validator"):
        DeliveryGate(run, clips).check("native-01-04")


def test_four_independent_roots_handoff_and_final_export(tmp_path, catalog_root):
    m1 = pipeline(tmp_path / "m1", catalog_root, stages=["inputs"])
    m2 = pipeline(tmp_path / "m2", catalog_root, stages=["human"], upstream=m1.root)
    m3 = pipeline(tmp_path / "m3", catalog_root, stages=["objects", "motion"], upstream=m2.root)
    m4 = pipeline(tmp_path / "m4", catalog_root, stages=["refine", "export"], upstream=m3.root)
    clips = load_clips("track1", root=catalog_root)
    assert DeliveryGate(m4, clips).check("export")["passed"]
    assert not (m2.root / "inputs").exists() and not (m3.root / "human").exists()


def test_subset_export_replaces_complete_tree_without_old_files(completed):
    run, clips, root = completed
    pipeline(run.root, root, episodes=[3], stages=["export"])
    assert not (run.root / "export/data/chunk-000/episode_000004.parquet").exists()
    assert not (run.root / "export/mhr/episode_000004.npz").exists()
    assert DeliveryGate(run, clips[:1], catalog=clips).check("export")["passed"]


def test_extra_export_file_fails_revalidation(completed):
    run, clips, _ = completed
    (run.root / "export/unrecorded.txt").write_text("extra", encoding="utf-8")
    report = DeliveryGate(run, clips).check("export")
    assert not report["passed"] and "unexpected or unrecorded" in str(report)


def test_unrecorded_export_output_blocks_publication(completed):
    run, clips, _ = completed
    before = (run.root / "export/meta/info.json").read_bytes()

    class ExtraExport:
        def run(self, attempt, selected):
            make("export", "tier1").run(attempt, selected)
            (attempt.root / "export/extra.txt").write_text("extra", encoding="utf-8")

    with run_lock(run), pytest.raises(GateError, match="unexpected or unrecorded"):
        DeliveryGate(run, clips).execute("export", "extra", ExtraExport())
    assert (run.root / "export/meta/info.json").read_bytes() == before
    assert not (run.root / "export/extra.txt").exists()


def test_malformed_pointer_produces_structured_failure(completed):
    run, clips, _ = completed
    pointer(run, "human").write_text("malformed", encoding="utf-8")
    report = DeliveryGate(run, clips).check("human")
    assert report["passed"] is False and report["checks"][0]["passed"] is False


def test_run_lock_is_acquired_before_run_metadata_changes(completed):
    run, _, root = completed
    before = (run.root / "run.json").read_bytes()
    with run_lock(run), pytest.raises(GateError, match="locked"):
        pipeline(run.root, root, stages=["inputs"])
    assert (run.root / "run.json").read_bytes() == before


def test_upstream_cycle_rejected(completed):
    run, clips, _ = completed
    run.upstream = run
    with pytest.raises(GateError, match="cyclic upstream"):
        DeliveryGate(run, clips)


@pytest.mark.parametrize("args,code", [(["--stage", "human"], 0),
                                       (["--before-stage", "motion"], 0),
                                       (["--stage", "native-05"], 2)])
def test_check_cli_exit_codes_and_no_writes(completed, monkeypatch, capsys, args, code):
    run, _, root = completed
    before = {p: p.read_bytes() for p in run.root.rglob("*") if p.is_file()}
    monkeypatch.setattr(sys, "argv", ["v2hoi.check", "--run", str(run.root), "--dataset-root", str(root), *args])
    with pytest.raises(SystemExit) as exited:
        check_main()
    assert exited.value.code == code
    captured = capsys.readouterr()
    if code == 0:
        assert json.loads(captured.out)["passed"] is True
    assert before == {p: p.read_bytes() for p in run.root.rglob("*") if p.is_file()}


def test_check_cli_failure_exit_one_without_creating_receipt(tmp_path, catalog_root, monkeypatch, capsys):
    run = pipeline(tmp_path / "only-inputs", catalog_root, stages=["inputs"])
    before = {p: p.read_bytes() for p in run.root.rglob("*") if p.is_file()}
    monkeypatch.setattr(sys, "argv", ["v2hoi.check", "--run", str(run.root), "--dataset-root", str(catalog_root), "--status"])
    with pytest.raises(SystemExit) as exited:
        check_main()
    assert exited.value.code == 1 and json.loads(capsys.readouterr().out)["passed"] is False
    assert before == {p: p.read_bytes() for p in run.root.rglob("*") if p.is_file()}


def test_partially_replaced_grouped_batch_fails_conservatively(completed):
    run, clips, root = completed
    pipeline(run.root, root, episodes=[3], stages=["human"])
    assert DeliveryGate(run, clips[:1], catalog=clips).check("human")["passed"]
    report = DeliveryGate(run, clips[1:], catalog=clips).check("human")
    assert not report["passed"] and "partially replaced" in str(report)
