"""Independent synthetic delivery checks; no challenge assets or model inference."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from v2hoi.clips import Clip
from v2hoi.contracts import Camera, Depth, Human, Motion, ObjectAsset, RefinedMotion, Run
from v2hoi.deliverables import STAGES, inspect_deliverables, required_paths
from v2hoi.stages import DEFAULT_BACKENDS, make


@pytest.fixture
def delivery(tmp_path):
    clips = [Clip("track1", e, name, "synthetic object", None, n, 30, 16, 12,
                  tmp_path / f"unused_{e}.mp4") for e, name, n in [(3, "box", 4), (8, "cup", 3)]]
    run = Run(tmp_path / "run", "track1")
    for stage in STAGES:
        make(stage, DEFAULT_BACKENDS[stage]).run(run, clips)
    return run, clips


def failed(result):
    return [c for c in result.checks if not c["passed"]]


def rewrite_npz(path, **updates):
    with np.load(path, allow_pickle=False) as source:
        arrays = dict(source)
    arrays.update(updates)
    np.savez_compressed(path, **arrays)


def rewrite_json(path, **updates):
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(updates)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("stage", STAGES)
def test_fake_deliveries_reload_and_enumerate_every_output(delivery, stage):
    run, clips = delivery
    result = inspect_deliverables(run, stage, clips)
    assert result.passed, failed(result)
    assert set(result.files) == set(required_paths(stage, clips))
    assert all(p.is_file() for p in result.files.values())
    assert all(type(c["passed"]) is bool for c in result.checks)


@pytest.mark.parametrize("stage", STAGES)
def test_missing_each_required_output_fails_even_with_complete_upstream(delivery, tmp_path, stage):
    run, clips = delivery
    child = Run(tmp_path / "child", "track1", run)
    result = inspect_deliverables(child, stage, clips)
    assert not result.passed
    assert all(not c["passed"] for c in result.checks if c["check_id"].startswith("reload:"))
    inherited = inspect_deliverables(child, stage, clips, local_only=False)
    assert inherited.passed is (stage != "export")


def test_all_errors_aggregate_and_unknown_stage_rejected(delivery):
    run, clips = delivery
    run.path(Camera, episode=3).write_text("not json", encoding="utf-8")
    run.path(Depth, episode=8).unlink()
    result = inspect_deliverables(run, "inputs", clips)
    assert len(failed(result)) == 2
    assert {c["episode_id"] for c in failed(result)} == {3, 8}
    with pytest.raises(ValueError, match="unknown stage"):
        inspect_deliverables(run, "native", clips)
    assert not inspect_deliverables(run, "human", []).passed


@pytest.mark.parametrize("update", [{"width": 15}, {"width": 16.0}, {"fx": float("inf")},
                                    {"fy": 0}, {"cx": -1}, {"contract": 1.5}])
def test_invalid_camera(delivery, update):
    run, clips = delivery
    rewrite_json(run.path(Camera, episode=3), **update)
    assert not inspect_deliverables(run, "inputs", clips).passed


@pytest.mark.parametrize("update", [{"stride": np.int64(0)}, {"stride": np.float64(8.2)},
                                    {"depth": np.zeros((3, 2, 2))},
                                    {"depth": np.full((4, 2, 2), -1)},
                                    {"depth": np.full((4, 2, 2), np.nan)}, {"contract": np.int64(2)}])
def test_invalid_depth(delivery, update):
    run, clips = delivery
    rewrite_npz(run.path(Depth, episode=3), **update)
    assert not inspect_deliverables(run, "inputs", clips).passed


def test_mask_dtype_and_missing_human_sidecar(delivery):
    run, clips = delivery
    mask = run.root / "inputs/000003/masks.npz"
    rewrite_npz(mask, human=np.zeros((4, 12, 2), dtype=np.float32))
    assert not inspect_deliverables(run, "inputs", clips).passed
    (run.root / "human/000003/depth_scale.json").unlink()
    assert not inspect_deliverables(run, "human", clips).passed


@pytest.mark.parametrize("field,value", [("scale", 0), ("scale", float("inf"))])
def test_depth_scale_must_be_positive_finite(delivery, field, value):
    run, clips = delivery
    rewrite_json(run.root / "human/000003/depth_scale.json", **{field: value})
    assert not inspect_deliverables(run, "human", clips).passed


@pytest.mark.parametrize("update", [{"name": "different"}, {"episodes": [8]}, {"episodes": [3, 3]},
                                    {"scale": -1}, {"scale": float("nan")}])
def test_object_identity_episode_scope_and_scale(delivery, update):
    run, clips = delivery
    rewrite_json(run.path(ObjectAsset, name="box"), **update)
    assert not inspect_deliverables(run, "objects", clips).passed


def test_object_mesh_is_required_and_reloadable(delivery):
    run, clips = delivery
    run.mesh("box").write_bytes(b"not a glb")
    result = inspect_deliverables(run, "objects", clips)
    assert not result.passed and any(c["artifact"].endswith("mesh.glb") for c in failed(result))


@pytest.mark.parametrize("stage,cls", [("motion", Motion), ("refine", RefinedMotion)])
def test_motion_full_frames_confidence_and_rigidity(delivery, stage, cls):
    run, clips = delivery
    path = run.path(cls, episode=3)
    rewrite_npz(path, confidence=np.full(4, 1.5))
    assert not inspect_deliverables(run, stage, clips).passed
    rewrite_npz(path, confidence=np.ones(4), T_cam_obj=np.zeros((4, 4, 4)))
    assert not inspect_deliverables(run, stage, clips).passed
    rewrite_npz(path, T_cam_obj=np.tile(np.eye(4), (3, 1, 1)))
    assert not inspect_deliverables(run, stage, clips).passed


def test_human_bad_shapes_and_object_payload_refused(delivery):
    run, clips = delivery
    path = run.path(Human, episode=3)
    rewrite_npz(path, scale=np.zeros((4, 67)))
    assert not inspect_deliverables(run, "human", clips).passed
    rewrite_npz(path, scale=np.array([object()] * 4, dtype=object))
    result = inspect_deliverables(run, "human", clips)
    assert not result.passed and "allow_pickle=False" in str(failed(result))


def test_refine_requires_human_and_motion(delivery):
    run, clips = delivery
    (run.root / "refine/000003/human.npz").unlink()
    assert not inspect_deliverables(run, "refine", clips).passed


@pytest.mark.parametrize("column,value", [("frame_index", [0, 1, 1, 3]),
                                         ("episode_index", [8] * 4),
                                         ("timestamp", [0.0] * 4),
                                         ("index", [0, 1, 2, 2])])
def test_export_frame_timeline_integrity(delivery, column, value):
    run, clips = delivery
    path = run.root / "export/data/chunk-000/episode_000003.parquet"
    df = pd.read_parquet(path)
    df[column] = value
    df.to_parquet(path)
    assert not inspect_deliverables(run, "export", clips).passed


def test_export_must_match_refined_arrays_and_mesh(delivery):
    run, clips = delivery
    path = run.root / "export/data/chunk-000/episode_000003.parquet"
    df = pd.read_parquet(path)
    df["observation.human.translation"] = [np.ones(3)] * 4
    df.to_parquet(path)
    rewrite_npz(run.root / "export/mhr/episode_000003.npz", mhr_shape=np.ones((4, 45)))
    import trimesh
    trimesh.creation.box(extents=[0.3] * 3).export(run.root / "export/mesh/box/box.glb")
    result = inspect_deliverables(run, "export", clips)
    assert len(failed(result)) == 3
    assert "differs from" in str(failed(result))


@pytest.mark.parametrize("update", [{"total_frames": 8}, {"total_episodes": 3},
                                    {"contract": 2}, {"data_path": "../../secret.parquet"}])
def test_export_metadata_totals_and_fixed_paths(delivery, update):
    run, clips = delivery
    rewrite_json(run.root / "export/meta/info.json", **update)
    assert not inspect_deliverables(run, "export", clips).passed


def test_export_duplicate_metadata_and_subset_checks(delivery):
    run, clips = delivery
    result = inspect_deliverables(run, "export", clips[:1])
    assert result.passed, failed(result)
    assert "export/data/chunk-000/episode_000008.parquet" in result.files
    path = run.root / "export/meta/episodes_metadata.jsonl"
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    assert not inspect_deliverables(run, "export", clips).passed


def test_export_schema_without_source_still_checked(delivery, tmp_path):
    run, clips = delivery
    import shutil
    standalone = Run(tmp_path / "standalone", "track1")
    shutil.copytree(run.root / "export", standalone.root / "export")
    assert inspect_deliverables(standalone, "export", clips).passed
    rewrite_npz(standalone.root / "export/mhr/episode_000003.npz", mhr_shape=np.full((4, 45), np.nan))
    assert not inspect_deliverables(standalone, "export", clips).passed


def test_unsafe_object_identifier_rejected_before_read(delivery):
    run, clips = delivery
    with pytest.raises(ValueError, match="invalid object name"):
        inspect_deliverables(run, "objects", [replace(clips[0], object_name="../outside")])
