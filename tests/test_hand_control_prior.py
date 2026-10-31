"""Independent synthetic control-eligibility tests; no challenge data/assets."""
from copy import deepcopy

import numpy as np
import pytest

from v2hoi.hand_refinement import (
    ARM_CONTROLS, HandFitConfig, control_prior_support_weights,
    project_wrist_delta_gauss_newton, refine_hands,
)


def annotation():
    return {
        "schema_version": "native_hand_control_prior.v1",
        "role": "parameter_eligibility_only_no_image_targets",
        "track": 1, "episode": 16, "frame_count": 12, "video_sha256": "a" * 64,
        "annotation_method": "Independent synthetic manual-eligibility fixture",
        "source_review_receipt": "synthetic-review.json",
        "source_review_receipt_sha256": "b" * 64,
        "eligible_intervals_inclusive": [
            {"first": 2, "last": 9, "eligible_sides": ["left"],
             "transition_frames": 2, "control_weight": 1.0}],
    }


def build(value):
    return control_prior_support_weights(value, track=1, episode=16, frame_count=12,
                                         video_sha256="a" * 64)


def test_eligibility_is_side_specific_with_internal_cosine_ramps_and_no_extrapolation():
    original = annotation()
    copy = deepcopy(original)
    support = build(original)
    np.testing.assert_allclose(support[:, 0], [0, 0, .25, .75, 1, 1, 1, 1, .75, .25, 0, 0])
    assert not support[:, 1].any()
    assert original == copy


@pytest.mark.parametrize("key,value", [
    ("track", 2), ("episode", 11), ("frame_count", 13),
    ("video_sha256", "c" * 64), ("schema_version", "other"),
    ("role", "hidden_pose_observations"), ("annotation_method", ""),
    ("source_review_receipt_sha256", ""), ("eligible_intervals_inclusive", []),
])
def test_source_and_semantics_are_explicit(key, value):
    item = annotation()
    item[key] = value
    with pytest.raises(ValueError):
        build(item)


@pytest.mark.parametrize("change", [
    {"first": -1}, {"last": 12}, {"first": True}, {"last": 1},
    {"transition_frames": -1}, {"transition_frames": 1.5},
    {"eligible_sides": ["left", "left"]}, {"eligible_sides": ["near"]},
    {"control_weight": 0}, {"control_weight": float("nan")},
    {"control_weight": 1.01}, {"control_weight": True},
])
def test_malformed_or_unbounded_prior_is_rejected(change):
    item = annotation()
    item["eligible_intervals_inclusive"][0].update(change)
    with pytest.raises(ValueError):
        build(item)


def test_overlap_is_rejected_for_same_hand_but_distinct_hands_can_overlap():
    item = annotation()
    item["eligible_intervals_inclusive"].append(deepcopy(item["eligible_intervals_inclusive"][0]))
    with pytest.raises(ValueError, match="overlap"):
        build(item)
    item["eligible_intervals_inclusive"][1]["eligible_sides"] = ["right"]
    support = build(item)
    np.testing.assert_array_equal(support[:, 0], support[:, 1])


def problem():
    torch = pytest.importorskip("torch", reason="Optional native optimization runtime")
    baseline = {"pose": np.zeros((12, 136), np.float32), "shape": np.zeros(45, np.float32),
                "scales": np.zeros(68, np.float32), "frame_indices": np.arange(12), "fps": np.array(30)}
    k = np.diag([1000, 1000, 1]).astype(np.float32)

    def forward(pose, scales):
        hands = []
        for fingers, arm in ((95, 43), (68, 33)):
            controls = torch.cat([pose.new_zeros(len(pose), 1), pose[:, fingers:fingers + 20]], dim=1)
            x = controls * .04 + torch.arange(21, device=pose.device)[None] * .005
            x = x + torch.sin(pose[:, arm:arm + 1]) * .2
            hands.append(torch.stack([x, x * 0, x * 0 + 5], dim=-1))
        return torch.stack(hands, dim=1)

    with torch.no_grad():
        xyz = forward(torch.tensor(baseline["pose"]), None).numpy()
    obs = {"frame_indices": np.arange(12), "side_order": np.array(["left", "right"]),
           "joints2d": (xyz @ k.T)[..., :2] / xyz[..., 2:],
           "confidence": np.ones((12, 2, 21), np.float32)}
    obs["confidence"][:, 0] = 0
    return torch, baseline, obs, k, forward


class SyntheticConstraint:
    """A differentiable synthetic force; not an actual contact geometry model."""
    anchor_weight = 1.0
    penetration_weight = 0.0
    before = {}

    def loss(self, hands):
        return {"contact_total": ((hands[2:10, 0, :, 0] - .15) ** 2).mean() * 1000}

    def diagnostics(self, hands):
        return {"synthetic_only": True}


def test_opt_in_hidden_hand_can_move_without_creating_image_evidence_or_freeing_other_controls():
    _, baseline, obs, k, forward = problem()
    support = build(annotation())
    config = HandFitConfig(iterations=80, learning_rate=.01, holdout_stride=3,
                           optimize_wrist_chain=True, support_transition_frames=0,
                           max_gap_frames=0, acceleration_weight=0)
    a = refine_hands(baseline, obs, k, forward, config=config,
                     control_prior_support=support, log=lambda _: None)
    b = refine_hands(baseline, obs, k, forward, config=config,
                     control_prior_support=support, contact_constraint=SyntheticConstraint(), log=lambda _: None)
    np.testing.assert_array_equal(a["optimization_weights"], b["optimization_weights"])
    assert not b["optimization_weights"][:, 0].any()
    assert not b["optimization_weights"][::3].any()
    assert not b["original_active_hands"][:, 0].any()
    np.testing.assert_array_equal(a["active_hands"], b["active_hands"])
    assert b["active_hands"][2:10, 0].all()
    assert (b["arm_control_envelope"][2:10, 0] > 0).all()
    assert np.abs(b["pose"][2:10, 95:122]).max() > .01
    assert np.abs(b["pose"][2:10, 43]).max() > .01
    np.testing.assert_array_equal(b["pose"][[0, 1, 10, 11], 95:122], baseline["pose"][[0, 1, 10, 11], 95:122])
    allowed = np.r_[68:122, ARM_CONTROLS.reshape(-1)]
    fixed = np.setdiff1d(np.arange(136), allowed)
    np.testing.assert_array_equal(b["pose"][:, fixed], baseline["pose"][:, fixed])
    np.testing.assert_array_equal(b["shape"], baseline["shape"])
    np.testing.assert_array_equal(b["scales"], baseline["scales"])
    assert b["report"]["max_wrist_displacement_metres"] <= config.wrist_max_displacement_metres + 1e-5
    assert b["report"]["arm_wrist_max_absolute_control_delta_radians"] <= config.arm_delta_limit_radians + 1e-6
    assert b["report"]["explicit_control_prior_frame_indices"]["left"] == list(range(2, 10))


def test_zero_prior_preserves_numerical_default_and_heldout_targets_never_enter_fit():
    _, baseline, obs, k, forward = problem()
    config = HandFitConfig(iterations=5, learning_rate=.01, holdout_stride=3,
                           optimize_wrist_chain=True)
    kwargs = dict(config=config, contact_constraint=SyntheticConstraint(), log=lambda _: None)
    original = refine_hands(baseline, obs, k, forward, **kwargs)
    zero = refine_hands(baseline, obs, k, forward, control_prior_support=np.zeros((12, 2)), **kwargs)
    np.testing.assert_array_equal(original["pose"], zero["pose"])
    np.testing.assert_array_equal(original["optimization_weights"], zero["optimization_weights"])
    first = refine_hands(baseline, obs, k, forward, control_prior_support=build(annotation()), **kwargs)
    obs["joints2d"][::3] += 10000
    second = refine_hands(baseline, obs, k, forward, control_prior_support=build(annotation()), **kwargs)
    np.testing.assert_array_equal(first["pose"], second["pose"])
    np.testing.assert_array_equal(first["arm_control_envelope"], second["arm_control_envelope"])


@pytest.mark.parametrize("support", [np.zeros((12, 1)), np.full((12, 2), np.nan), np.full((12, 2), 1.1)])
def test_invalid_runtime_support_is_rejected(support):
    _, baseline, obs, k, forward = problem()
    with pytest.raises(ValueError, match="Control-prior support"):
        refine_hands(baseline, obs, k, forward, config=HandFitConfig(iterations=1),
                     control_prior_support=support, log=lambda _: None)


def test_prior_only_bounds_expand_only_admitted_hand_and_keep_ray_prior_separate_from_evidence():
    _, baseline, obs, k, forward = problem()
    prior = build(annotation())
    config = HandFitConfig(iterations=12, learning_rate=.01, holdout_stride=3,
                           optimize_wrist_chain=True, control_prior_arm_delta_limit_radians=.35,
                           control_prior_wrist_max_displacement_metres=.12,
                           control_prior_wrist_ray_weight=.5,
                           control_prior_wrist_max_projection_pixels=3)
    result = refine_hands(baseline, obs, k, forward, config=config, control_prior_support=prior,
                          contact_constraint=SyntheticConstraint(), log=lambda _: None)
    np.testing.assert_allclose(result["wrist_displacement_limits_m"][:, 0], .04 + .08 * prior[:, 0])
    np.testing.assert_allclose(result["wrist_displacement_limits_m"][:, 1], .04)
    np.testing.assert_allclose(result["arm_control_delta_limits"][:, 0],
                               np.repeat((.1 + .25 * prior[:, 0])[:, None], 7, axis=1))
    np.testing.assert_allclose(result["arm_control_delta_limits"][:, 1], .1)
    assert not result["optimization_weights"][:, 0].any()
    assert "control_prior_wrist_ray" in result["report"]["final_terms"]
    assert result["report"]["prior_wrist_max_projection_displacement_pixels"] <= 3.0001
    with pytest.raises(ValueError, match="explicit nonzero"):
        refine_hands(baseline, obs, k, forward, config=config, log=lambda _: None)


def test_native_jacobian_normal_projection_preserves_depth_direction_that_radial_scaling_loses():
    torch = pytest.importorskip("torch")
    def forward(values):
        return torch.stack([values[..., 0], values[..., 0] * 0, 5 + values[..., 1]], -1)
    value = torch.tensor([[[.1, .1], [.1, .1]]])
    reference = forward(torch.zeros_like(value))
    k = torch.diag(torch.tensor([1000., 1000., 1.]))
    projected = project_wrist_delta_gauss_newton(value, forward, reference, k, 3.,
                                                torch.tensor([[True, False]]), torch.full_like(value, .35))
    image = forward(projected) @ k.T
    radius = torch.linalg.vector_norm(image[..., :2] / image[..., 2:], dim=-1)
    assert radius[0, 0] <= 3.0001
    assert projected[0, 0, 1] > .09
    radial_depth = .1 * 3 / (.1 / 5.1 * 1000)
    assert float(projected[0, 0, 1]) > radial_depth * 5
    np.testing.assert_array_equal(projected[:, 1], value[:, 1])
