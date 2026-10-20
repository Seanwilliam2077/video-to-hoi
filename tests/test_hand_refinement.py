"""Independent synthetic fixtures; no challenge data, references or model files."""
from dataclasses import replace

import numpy as np
import pytest

from v2hoi.hand_refinement import (
    HAND_JOINTS, HandFitConfig, hand_diagnostics, observation_motion_gate,
    refine_hands, supported_frames, validate_observations,
)


def observations(frames=12):
    return {"frame_indices": np.arange(frames), "side_order": np.array(["left", "right"]),
            "joints2d": np.zeros((frames, 2, 21, 2), dtype=np.float32),
            "confidence": np.ones((frames, 2, 21), dtype=np.float32)}


@pytest.mark.parametrize("change", [
    {"iterations": 0}, {"iterations": True}, {"min_landmarks": 21},
    {"max_gap_frames": -1}, {"holdout_stride": 1}, {"holdout_stride": True},
    {"max_delta_radians": float("inf")}, {"learning_rate": 0}, {"min_confidence": 0},
    {"relative_weight": 1.1}, {"acceleration_weight": -1}, {"unobserved_policy": "guess"},
    {"optimize_wrist_chain": "yes"}, {"arm_delta_limit_radians": 0},
    {"wrist_max_displacement_metres": float("nan")},
])
def test_invalid_configuration_rejected(change):
    with pytest.raises(ValueError):
        replace(HandFitConfig(), **change).validate()


def test_observations_require_original_frame_alignment_and_anatomical_sides():
    obs = observations()
    with pytest.raises(ValueError, match="exactly match"):
        validate_observations(obs, np.arange(12) + 1)
    obs["side_order"] = np.array(["right", "left"])
    with pytest.raises(ValueError, match="anatomical"):
        validate_observations(obs, np.arange(12))


def test_missing_nan_allowed_only_at_zero_weight_and_inputs_unchanged():
    obs = observations()
    obs["joints2d"][2, 0, 4] = np.nan
    with pytest.raises(ValueError, match="positive-confidence"):
        validate_observations(obs, np.arange(12))
    obs["confidence"][2, 0, 4] = 0
    validated = validate_observations(obs, np.arange(12))
    assert np.isfinite(validated["joints2d"]).all()
    assert np.isnan(obs["joints2d"][2, 0, 4]).all()
    validated["confidence"][:] = 0
    assert obs["confidence"].sum() > 0


@pytest.mark.parametrize("invalid", [-0.1, 1.1, float("nan")])
def test_reliability_weights_validated(invalid):
    obs = observations()
    obs["confidence"][0, 0, 0] = invalid
    with pytest.raises(ValueError, match="confidence"):
        validate_observations(obs, np.arange(12))


def test_support_bridges_only_short_internal_gaps_without_extrapolation():
    confidence = np.zeros((15, 2, 21), dtype=np.float32)
    confidence[[2, 5, 12], 0] = 1
    active = supported_frames(confidence, HandFitConfig(max_gap_frames=2))
    np.testing.assert_array_equal(np.flatnonzero(active[:, 0]), [2, 3, 4, 5, 12])
    assert not active[:, 1].any()


def test_partial_or_wristless_observations_do_not_activate_a_hand():
    obs = observations(2)
    obs["confidence"][0, :, 0] = 0
    obs["confidence"][1, :, 5:] = 0
    assert not supported_frames(obs["confidence"], HandFitConfig()).any()


def test_motion_gate_ignores_translation_but_retains_observed_articulation():
    obs = observations(6)
    obs["joints2d"][:] = np.arange(6)[:, None, None, None] * 50
    translated = observation_motion_gate(obs["joints2d"], obs["confidence"], HandFitConfig())
    np.testing.assert_array_equal(translated, np.ones((4, 2)))
    obs["joints2d"][:, :, 1:, 0] += np.arange(6)[:, None, None] * 10
    articulated = observation_motion_gate(obs["joints2d"], obs["confidence"], HandFitConfig())
    assert (articulated < 0.1).all()
    obs["confidence"][:] = 0
    missing = observation_motion_gate(obs["joints2d"], obs["confidence"], HandFitConfig())
    np.testing.assert_array_equal(missing, np.ones((4, 2)))


def test_relative_diagnostic_separates_frozen_wrist_error():
    obs = observations(3)
    projected = np.full((3, 2, 21, 2), 10, dtype=np.float32)
    diagnostic = hand_diagnostics(projected, obs, np.ones((3, 2), dtype=bool))
    assert diagnostic["left"]["weighted_reprojection_pixels"] == pytest.approx(2**0.5 * 10)
    assert diagnostic["left"]["weighted_wrist_relative_reprojection_pixels"] == 0
    assert "not hidden-reference" in diagnostic["interpretation"]


def test_verified_joint_mapping_is_unique_and_side_specific():
    assert HAND_JOINTS.shape == (2, 21)
    assert len(np.unique(HAND_JOINTS)) == 42
    np.testing.assert_array_equal(HAND_JOINTS[:, 0], [78, 42])


def _synthetic_problem():
    torch = pytest.importorskip("torch", reason="Optional Torch runtime needed for synthetic optimization")
    frames = 12
    baseline = {"pose": np.zeros((frames, 136), dtype=np.float32), "shape": np.zeros(45, dtype=np.float32),
                "scales": np.zeros(68, dtype=np.float32), "frame_indices": np.arange(frames), "fps": np.array(30)}
    obs = observations(frames)
    k = np.diag([1000, 1000, 1]).astype(np.float32)

    def forward(pose, scales):
        outputs = []
        for offset in (95, 68):
            controls = torch.cat([pose.new_zeros(len(pose), 1), pose[:, offset:offset + 20]], dim=1)
            x = controls * 0.04 + torch.arange(21, device=pose.device)[None] * 0.01
            outputs.append(torch.stack([x, x * 0, x * 0 + 5], dim=-1))
        return torch.stack(outputs, dim=1)

    with torch.no_grad():
        target_pose = torch.zeros(frames, 136)
        target_pose[:, 68:122] = 0.35
        xyz = forward(target_pose, None).numpy()
        obs["joints2d"] = (xyz @ k.T)[..., :2] / xyz[..., 2:]
    return torch, baseline, obs, k, forward


def test_actual_finger_optimization_improves_evidence_without_moving_body_or_identity():
    _, baseline, obs, k, forward = _synthetic_problem()
    config = HandFitConfig(iterations=70, holdout_stride=0, acceleration_weight=0, pose_prior_weight=0)
    fitted = refine_hands(baseline, obs, k, forward, config=config, log=lambda _: None)
    report = fitted["report"]
    assert report["after"]["right"]["weighted_wrist_relative_reprojection_pixels"] < 0.1
    assert report["best_objective"] < report["initial_objective"] / 100
    np.testing.assert_array_equal(fitted["pose"][:, :68], baseline["pose"][:, :68])
    np.testing.assert_array_equal(fitted["pose"][:, 122:], baseline["pose"][:, 122:])
    np.testing.assert_array_equal(fitted["shape"], baseline["shape"])
    np.testing.assert_array_equal(fitted["scales"], baseline["scales"])
    assert not baseline["pose"].any()


def test_heldout_targets_do_not_leak_into_position_velocity_or_motion_gates():
    _, baseline, obs, k, forward = _synthetic_problem()
    config = HandFitConfig(iterations=10, holdout_stride=3)
    first = refine_hands(baseline, obs, k, forward, config=config, log=lambda _: None)
    obs["joints2d"][::3] += 10000
    second = refine_hands(baseline, obs, k, forward, config=config, log=lambda _: None)
    np.testing.assert_array_equal(first["pose"], second["pose"])
    assert not first["optimization_weights"][::3].any()
    assert second["report"]["after_heldout"]["right"]["weighted_reprojection_pixels"] > 1000


def test_preserve_policy_keeps_unsupported_long_gaps_exactly_unchanged():
    _, baseline, obs, k, forward = _synthetic_problem()
    obs["confidence"][:] = 0
    obs["confidence"][2:7, 1] = 1
    fitted = refine_hands(baseline, obs, k, forward,
                          config=HandFitConfig(iterations=5, holdout_stride=0, max_gap_frames=1), log=lambda _: None)
    np.testing.assert_array_equal(fitted["pose"][:2], baseline["pose"][:2])
    np.testing.assert_array_equal(fitted["pose"][7:], baseline["pose"][7:])
    np.testing.assert_array_equal(fitted["pose"][:, 95:122], baseline["pose"][:, 95:122])


def test_coupled_arm_updates_obey_control_and_wrist_trust_region_bounds():
    torch, baseline, obs, k, fingers_only = _synthetic_problem()

    def coupled_forward(pose, scales):
        fingers = fingers_only(pose, scales)
        translation = torch.stack([torch.sin(pose[:, 43]), torch.sin(pose[:, 33])], dim=1) * 0.2
        shift = torch.stack([translation, translation * 0, translation * 0], dim=-1)
        return fingers + shift[:, :, None]

    with torch.no_grad():
        target = torch.tensor(baseline["pose"])
        target[:, [33, 43]] = 0.1
        target[:, 68:122] = 0.35
        xyz = coupled_forward(target, None).numpy()
        obs["joints2d"] = (xyz @ k.T)[..., :2] / xyz[..., 2:]
    config = HandFitConfig(iterations=60, holdout_stride=3, optimize_wrist_chain=True,
                           relative_weight=0.5, wrist_max_displacement_metres=0.01)
    fitted = refine_hands(baseline, obs, k, coupled_forward, config=config, log=lambda _: None)
    report = fitted["report"]
    assert report["max_wrist_displacement_metres"] <= 0.01001
    assert report["arm_wrist_max_absolute_control_delta_radians"] <= 0.10001
    assert report["after"]["right"]["weighted_reprojection_pixels"] < report["before"]["right"]["weighted_reprojection_pixels"]
    assert not report["body_root_wrist_controls_unchanged"]
    np.testing.assert_array_equal(fitted["pose"][:, :6], baseline["pose"][:, :6])
    assert np.mean(np.abs(fitted["pose"][[3, 6, 9], 33])) > 0.01


def test_spatial_hand_prior_corrects_inherited_arm_twist_only_with_explicit_phases():
    torch, baseline, obs, k, _ = _synthetic_problem()
    def forward(pose, scales):
        hands = []
        length = torch.arange(21, device=pose.device)[None] * 0.005
        for index in (43, 33):
            angle = pose[:, index:index + 1]
            x, y = length * torch.cos(angle), length * torch.sin(angle)
            hands.append(torch.stack([x, y, x * 0 + 5], dim=-1))
        return torch.stack(hands, dim=1)
    target = forward(torch.tensor(baseline["pose"]), None).numpy()
    obs["joints2d"] = (target @ k.T)[..., :2] / target[..., 2:]
    baseline["pose"][4, 33] = 0.3
    config = HandFitConfig(iterations=60, holdout_stride=2, optimize_wrist_chain=True,
                           arm_delta_limit_radians=0.35, spatial_acceleration_weight=0.2,
                           arm_update_temporal_weight=0, arm_prior_weight=0)
    with pytest.raises(ValueError, match="explicit admitted"):
        refine_hands(baseline, obs, k, forward, config=config, log=lambda _: None)
    weights = np.zeros((12, 2), np.float32)
    weights[3:8] = 1
    result = refine_hands(baseline, obs, k, forward, config=config, motion_prior_weights=weights, log=lambda _: None)
    assert abs(result["pose"][4, 33]) < 0.1
    assert result["report"]["admitted_motion_prior_frame_indices"]["right"] == list(range(3, 8))
    assert result["report"]["final_terms"]["spatial_hand_acceleration"] < result["report"]["initial_terms"]["spatial_hand_acceleration"]
    np.testing.assert_array_equal(result["pose"][[0, 11], :6], baseline["pose"][[0, 11], :6])
