"""Independent synthetic kinematics only; no model or challenge assets loaded."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def evaluator():
    path = Path(__file__).resolve().parents[1] / "tools/evaluate_human_hand_pilots.py"
    spec = importlib.util.spec_from_file_location("independent_hand_pilot_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene(frames=5):
    joints = np.zeros((frames, 2, 21, 3), dtype=float)
    joints[..., 2] = 1
    joints[:, :, 1:, 0] = np.arange(1, 21) * .001
    confidence = np.ones((frames, 2, 21))
    camera = np.diag([100., 100., 1.])
    return joints, confidence, camera


def test_original_support_edges_survive_full_sequence_extension(evaluator):
    old = np.array([[False, False], [True, False], [True, False], [False, False]])
    extended = np.ones((4, 2), dtype=bool)
    assert evaluator.support_edges(old) == [(0, 0), (2, 0)]
    assert evaluator.support_edges(extended) == []
    joints, confidence, camera = scene(4)
    candidate = joints.copy()
    candidate[1:3, 0, 1:, 0] += .05
    original = set(evaluator.support_edges(old))
    rows = evaluator.motion_rows(joints, {"old_candidate": candidate, "candidate": candidate},
                                 evaluator.project_hands(joints, camera), confidence, camera, .05,
                                 np.zeros(4, bool), original)
    frozen = [row for row in rows if row["original_support_boundary"]]
    assert len(frozen) == 2
    assert all(not row["variants"]["candidate"]["passed"] for row in frozen)


def test_global_wrist_translation_is_not_finger_articulation(evaluator):
    joints, confidence, camera = scene()
    moving = joints.copy()
    moving[..., 0] += np.arange(len(joints))[:, None, None] * .1
    observed = evaluator.project_hands(moving, camera)
    rows = evaluator.motion_rows(joints, {"old_candidate": moving, "candidate": moving}, observed,
                                 confidence, camera, .05, np.zeros(5, bool), set())
    assert max(row["variants"]["candidate"]["correction_only_mean_wrist_relative_step_px"] for row in rows) < 1e-12
    assert not any(row["fast_observed_motion"] for row in rows)


def test_absolute_wrist_guard_detects_jump_hidden_by_wrist_relative_fingers(evaluator):
    joints, confidence, camera = scene(4)
    candidate = joints.copy()
    candidate[2:, :, :, 0] += .03  # Entire hand jumps, its relative fingers do not.
    rows = evaluator.motion_rows(joints, {"old_candidate": joints, "candidate": candidate},
                                 evaluator.project_hands(joints, camera), confidence, camera, .05,
                                 np.zeros(4, bool), {(1, 0)})
    edge = next(row for row in rows if row["side"] == "left" and row["from_original_frame_id"] == 1)
    assert edge["variants"]["candidate"]["passed"]
    assert edge["variants"]["candidate"]["correction_only_mean_wrist_relative_step_px"] < 1e-12
    assert not edge["wrist_variants"]["candidate"]["passed"]
    assert edge["wrist_variants"]["candidate"]["absolute_wrist_velocity_vector_discrepancy_px"] == pytest.approx(3)
    report = evaluator.summarize_motion([edge])
    assert report["absolute_wrist_motion"]["variants"]["candidate"]["observed_wrist_failed"] == 1


def test_unknown_absolute_wrist_budget_does_not_require_visible_fingers(evaluator):
    joints, confidence, camera = scene(3)
    candidate = joints.copy()
    candidate[1:, :, :, 0] += .006  # Below 1 pixel, above 5 native estimated mm.
    confidence[0, :, 0] = 0
    rows = evaluator.motion_rows(joints, {"old_candidate": joints, "candidate": candidate},
                                 evaluator.project_hands(joints, camera), confidence, camera, .05,
                                 np.zeros(3, bool), {(0, 0)})
    edge = rows[0]
    assert edge["variants"]["candidate"]["passed"]
    assert edge["wrist_variants"]["candidate"]["guard"] == "unknown_motion_absolute_wrist_correction_budget"
    assert edge["wrist_variants"]["candidate"]["absolute_wrist_velocity_vector_discrepancy_px"] is None
    assert not edge["wrist_variants"]["candidate"]["passed"]
    report = evaluator.summarize_wrist_motion(rows)
    assert report["variants"]["candidate"]["unknown_wrist_budget_failed"] == 2
    assert report["variants"]["candidate"]["observed_wrist_failed"] == 0


def test_frozen_fingers_fail_against_real_fast_observed_change(evaluator):
    joints, confidence, camera = scene()
    moving = joints.copy()
    moving[:, :, 1:, 0] += np.arange(len(joints))[:, None, None] * .05
    points = evaluator.project_hands(moving, camera)
    rows = evaluator.motion_rows(moving, {"old_candidate": joints, "candidate": joints}, points,
                                 confidence, camera, .05, np.array([True, False, False, False, False]), set())
    assert all(row["fast_observed_motion"] for row in rows)
    assert all(not row["variants"]["candidate"]["passed"] for row in rows)
    assert all(abs(row["variants"]["candidate"]["observed_vector_projection_gain"]) < 1e-12 for row in rows)
    assert all(row["variants"]["baseline"]["observed_vector_projection_gain"] == pytest.approx(1) for row in rows)
    assert sum(row["touches_original_holdout"] for row in rows) == 2


def test_unknown_motion_budget_is_separate_from_observed_regression(evaluator):
    joints, confidence, camera = scene(3)
    candidate = joints.copy()
    candidate[1:, :, 1:, 0] += .006  # 0.6px, but 6mm exceeds the unchanged budget.
    confidence[0] = 0
    rows = evaluator.motion_rows(joints, {"old_candidate": candidate, "candidate": candidate},
                                 evaluator.project_hands(joints, camera), confidence, camera, .05,
                                 np.zeros(3, bool), {(0, 0)})
    boundary = rows[0]
    assert boundary["variants"]["candidate"]["guard"] == "unknown_motion_correction_budget"
    assert not boundary["variants"]["candidate"]["passed"]
    assert boundary["variants"]["candidate"]["mean_velocity_vector_discrepancy_px"] is None
    assert rows[2]["variants"]["candidate"]["guard"] == "observed_velocity_nonregression"
    report = evaluator.summarize_motion(rows)
    assert report["variants"]["candidate"]["unknown_motion_budget_failed"] == 2
    assert report["variants"]["candidate"]["observed_motion_failed"] == 0


def test_missing_wrist_or_seven_fingers_cannot_claim_observed_motion(evaluator):
    joints, confidence, camera = scene(3)
    confidence[0, 0, 0] = 0
    confidence[0, 1, 8:] = 0
    rows = evaluator.motion_rows(joints, {"old_candidate": joints, "candidate": joints},
                                 evaluator.project_hands(joints, camera), confidence, camera, .05,
                                 np.zeros(3, bool), set())
    assert rows[0]["common_accepted_finger_count"] == 0
    assert rows[1]["common_accepted_finger_count"] == 7
    assert not rows[0]["independent_observed_motion_sufficient"]
    assert not rows[1]["independent_observed_motion_sufficient"]


def test_holdout_image_error_cannot_be_hidden_by_training_improvement(evaluator):
    joints, weights, camera = scene(10)
    points = evaluator.project_hands(joints, camera)
    baseline = points + 2
    candidate = points.copy()
    heldout = np.arange(10) % 2 == 0
    candidate[heldout] += 8
    report = evaluator.image_agreement({"baseline": baseline, "old_candidate": baseline, "candidate": candidate}, points, weights, heldout)
    assert report["original_training"]["left"]["candidate_image_nonregression"]
    assert not report["original_holdout"]["left"]["candidate_image_nonregression"]
    assert report["original_holdout"]["left"]["observed_frames"] == 5


def test_body_root_and_identity_changes_reported_even_with_allowed_finger_update(evaluator):
    baseline = {"pose": np.zeros((3, 136)), "shape": np.zeros(45), "scales": np.zeros(68)}
    candidate = {key: value.copy() for key, value in baseline.items()}
    candidate["pose"][:, 70] = .2
    allowed = evaluator.parameter_preservation(baseline, candidate, False)
    assert allowed["all_unselected_controls_unchanged"]
    assert allowed["root_controls_unchanged"]
    candidate["pose"][1, 3] = .01
    candidate["shape"][0] = .1
    bad = evaluator.parameter_preservation(baseline, candidate, False)
    assert not bad["all_unselected_controls_unchanged"]
    assert not bad["root_controls_unchanged"]
    assert not bad["fixed_identity_unchanged"]


def test_invalid_observation_and_support_contract_rejected(evaluator):
    _, weights, _ = scene(3)
    obs = {"frame_indices": np.arange(3), "side_order": np.array(["left", "right"]),
           "joints2d": np.zeros((3, 2, 21, 2)), "confidence": weights}
    obs["joints2d"][1, 0, 1] = np.nan
    with pytest.raises(ValueError, match="Accepted observations"):
        evaluator.validate_evidence(obs, np.arange(3))
    with pytest.raises(ValueError, match="boolean"):
        evaluator.support_edges(np.zeros((3, 2), dtype=int))
    obs["confidence"][1, 0, 1] = 0
    obs["joints2d"][1, 0, 1] = np.inf
    points, _ = evaluator.validate_evidence(obs, np.arange(3))
    assert np.isfinite(points).all()
    with pytest.raises(ValueError, match="original source"):
        evaluator.validate_evidence(obs, np.array([0, 2, 3]))


def test_source_review_windows_keep_original_edges_and_anatomical_side(evaluator):
    joints, confidence, camera = scene(5)
    rows = evaluator.motion_rows(joints, {"old_candidate": joints, "candidate": joints},
                                 evaluator.project_hands(joints, camera), confidence, camera, .05,
                                 np.zeros(5, bool), {(1, 0), (3, 0)})
    window = {"first_original_frame_id": 1, "last_original_frame_id": 3,
              "side": "left", "label": "Manually reviewed source action"}
    report = evaluator.review_window_metrics([window], rows, 5)[0]
    assert report["all_edges"]["edge_count"] == 2
    assert report["original_support_boundaries"]["edge_count"] == 1
    assert report["annotation"] == window
    with pytest.raises(ValueError, match="original-frame"):
        evaluator.review_window_metrics([{**window, "last_original_frame_id": 5}], rows, 5)
