"""Independent synthetic geometry/evidence checks without challenge assets."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "tools/evaluate_contact_ab.py"
    spec = importlib.util.spec_from_file_location("contact_ab_evaluation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_phase_is_preserved_even_with_zero_attraction(cli):
    rows = cli.phase_windows({"phase_intervals_inclusive": [
        {"phase": "hold", "first": 0, "last": 1, "anchor_weight": 1},
        {"phase": "release", "first": 2, "last": 3, "anchor_weight": 0}]}, 4)
    assert len(rows) == 2 and rows[1]["label"] == "release"
    assert rows[1]["first"] == 2 and rows[1]["anchor_weight"] == 0


@pytest.mark.parametrize("intervals", [
    [{"phase": "gap", "first": 1, "last": 3}],
    [{"phase": "one", "first": 0, "last": 2}, {"phase": "two", "first": 2, "last": 3}],
    [{"phase": "bad", "first": False, "last": 3}],
])
def test_phase_gaps_overlap_and_boolean_frame_ids_are_rejected(cli, intervals):
    with pytest.raises(ValueError):
        cli.phase_windows({"phase_intervals_inclusive": intervals}, 4)


def controlled():
    support = {role: np.ones((5, 2), bool) for role in ("a", "b")}
    confidence = np.ones((5, 2, 21))
    weights = {role: confidence.copy() for role in ("a", "b")}
    heldout = np.array([True, False, False, False, False])
    for value in weights.values():
        value[heldout] = 0
    return support, weights, confidence, heldout


def test_different_positive_weight_values_cannot_pass_same_mask_comparison(cli):
    support, weights, confidence, heldout = controlled()
    weights["b"][1, 0, 1] = .9
    with pytest.raises(ValueError, match="identical"):
        cli.validate_controlled_arrays(support, weights, confidence, heldout, .05)


def test_b_only_hidden_hand_degrees_of_freedom_are_rejected(cli):
    support, weights, confidence, heldout = controlled()
    support["a"][2, 0] = False
    with pytest.raises(ValueError, match="identical"):
        cli.validate_controlled_arrays(support, weights, confidence, heldout, .05)


def test_holdout_weights_are_rejected_even_when_a_and_b_match(cli):
    support, weights, confidence, heldout = controlled()
    for value in weights.values():
        value[0] = 1
    with pytest.raises(ValueError, match="heldout"):
        cli.validate_controlled_arrays(support, weights, confidence, heldout, .05)


def test_unsupported_native_finger_changes_are_rejected(cli):
    base, changed = np.zeros((4, 136)), np.zeros((4, 136))
    active = np.ones((4, 2), bool)
    active[2, 0] = False
    changed[2, 95] = .01
    with pytest.raises(ValueError, match="Unsupported"):
        cli.inactive_finger_abstention(base, changed, active)


def test_object_local_motion_cancels_rigid_translation_and_rotation(cli):
    local = np.array([[[[.1, .2, .3]]]] * 3)
    pose = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    pose[1, :3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    pose[:, :3, 3] = [[0., 0., 0.], [2., 0., 1.], [4., 2., 1.]]
    world = np.einsum("t...i,tji->t...j", local, pose[:, :3, :3]) + pose[:, None, None, :3, 3]
    assert np.linalg.norm(np.diff(world, axis=0)).max() > 1
    np.testing.assert_allclose(cli.to_object(world, pose), local, atol=1e-15)


def summary(value):
    value = np.asarray(value)
    return {"count": int(value.size), "mean": float(value.mean()) if value.size else None,
            "max": float(value.max()) if value.size else None}


def fake_geometry(cli, depth):
    frames = 4
    hands = np.zeros((frames, 2, 21, 3))
    obj = {"object_to_camera": np.broadcast_to(np.eye(4), (frames, 4, 4)).copy(),
           "vertices": np.zeros((4, 3)), "faces": np.zeros((4, 3), int),
           "pose_usable": np.array([True, True, False, False])}
    skin = {"per_frame_mean_depth_m": np.asarray(depth)[:, None] * np.ones((frames, 2)),
            "per_frame_max_depth_m": np.asarray(depth)[:, None] * 2 * np.ones((frames, 2)),
            "per_frame_penetrating_vertices": np.ones((frames, 2), int)}
    windows = [{"label": "hold", "first": 0, "last": 1, "anchor_weight": 1},
               {"label": "released", "first": 2, "last": 3, "anchor_weight": 0}]
    nearest = lambda points, *_: (points.copy(), np.ones(len(points)) * .01)
    return cli.geometry_diagnostics(hands, obj, windows, skin, None, None, nearest, summary)


def test_full_skin_release_penetration_cannot_hide_behind_contact_or_pose_gate(cli):
    value = fake_geometry(cli, [0., 0., .02, .03])
    released = value["phase_summaries"][2]["sides"]["left"]
    assert released["frames"] == 2 and released["object_pose_usable_frames"] == 0
    assert released["skin_mean_depth_over_selected_hand_vertices_mm"]["mean"] == 25
    assert released["skin_maximum_vertex_depth_mm"]["max"] == 60
    assert released["usable_pose_skin_maximum_vertex_depth_mm"]["count"] == 0
    assert len(value["all_original_frame_rows"]) == 4


def test_skin_nonregression_keeps_failed_unfiltered_and_unknown_usable_scopes(cli):
    a = fake_geometry(cli, [0., 0., 0., 0.])
    b = fake_geometry(cli, [0., 0., .02, .03])
    result = cli.skin_nonregression(a, b, {"skin_b_versus_a": {
        "maximum_mean_depth_increase_mm": .5, "maximum_worst_vertex_depth_increase_mm": 2.}})
    release = [r for r in result["rows"] if r["phase"]["label"] == "released" and r["side"] == "left"]
    assert release[0]["sufficient"] and release[0]["mean_depth_nonregression"] is False
    assert release[0]["maximum_depth_nonregression"] is False
    assert release[1]["sufficient"] is False and release[1]["mean_depth_nonregression"] is None


def test_native_numeric_parity_ignores_metadata_and_rejects_changed_pose(cli):
    a = {key: np.zeros(3) for key in ("pose", "shape", "scales", "fps", "frame_indices")}
    b = {key: value.copy() for key, value in a.items()}
    a["report"], b["report"] = "original", "new receipt"
    assert all(cli.matched_numeric_arrays(a, b).values())
    b["pose"][0] = 1e-5
    assert cli.matched_numeric_arrays(a, b)["pose"] is False


def anchor_fixture(cli, tmp_path):
    count = 4
    baseline = {"pose": np.zeros((count, 136)), "shape": np.zeros(45), "scales": np.zeros(68),
                "fps": np.asarray(30.), "frame_indices": np.arange(count)}
    reference = {k: v.copy() for k, v in baseline.items()}
    reference["pose"][:, 68] = .1
    path = tmp_path / "synthetic-native.npz"
    np.savez(path, **reference)
    review = tmp_path / "review_arrays"
    review.mkdir()
    camera = np.diag([100., 100., 1.])
    np.save(review / "camera_intrinsics.npy", camera)
    manifest = {"video_frames": count, "video_sha256": "synthetic-source-only",
                "mhr_model_sha256": "injected-synthetic-decoder", "camera_intrinsics": camera}
    (tmp_path / "report.json").write_text(json.dumps({"source_video_sha256": manifest["video_sha256"],
        "mhr_model_sha256": manifest["mhr_model_sha256"]}), encoding="utf-8")
    entry = {"path": str(path), "sha256": cli.digest(path)}
    task = {"contact_anchor_reference": entry}
    contact = {"anchor_reference": {**entry, "same_optimizer_initialization": True}}
    return task, contact, baseline, manifest, reference, path


def test_explicit_anchor_prior_can_change_hand_but_not_initialization_identity(cli, tmp_path):
    task, contact, base, manifest, _, _ = anchor_fixture(cli, tmp_path)
    result = cli.validate_anchor_reference(task, contact, base, manifest)
    assert result["used"] and result["optimizer_initialization_and_image_targets_unchanged"]


def test_unbound_anchor_prior_is_rejected(cli, tmp_path):
    _, contact, base, manifest, _, _ = anchor_fixture(cli, tmp_path)
    with pytest.raises(ValueError, match="explicitly bound"):
        cli.validate_anchor_reference({}, contact, base, manifest)


def test_anchor_prior_with_different_root_is_rejected_even_if_sha_matches(cli, tmp_path):
    task, contact, base, manifest, reference, path = anchor_fixture(cli, tmp_path)
    reference["pose"][0, 0] = .01
    np.savez(path, **reference)
    task["contact_anchor_reference"]["sha256"] = contact["anchor_reference"]["sha256"] = cli.digest(path)
    with pytest.raises(ValueError, match="root"):
        cli.validate_anchor_reference(task, contact, base, manifest)


def test_anchor_prior_with_different_camera_is_rejected(cli, tmp_path):
    task, contact, base, manifest, _, path = anchor_fixture(cli, tmp_path)
    np.save(path.parent / "review_arrays/camera_intrinsics.npy", np.eye(3))
    with pytest.raises(ValueError, match="camera"):
        cli.validate_anchor_reference(task, contact, base, manifest)
