"""Independent synthetic fixtures; no challenge assets, weights or video access."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def cli():
    source = Path(__file__).resolve().parents[1] / "tools/evaluate_occluded_contact_ab.py"
    spec = importlib.util.spec_from_file_location("occluded_contact_evaluation_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def annotation():
    return {"schema_version": "native_hand_control_prior.v1", "track": 1, "episode": 16,
        "frame_count": 12, "video_sha256": "independent-synthetic-source",
        "role": "parameter_eligibility_only_no_image_targets", "annotation_method": "Synthetic interval",
        "eligible_intervals_inclusive": [{"first": 3, "last": 8, "eligible_sides": ["left"],
            "transition_frames": 2, "control_weight": 1.}]}


def summary(value):
    values = np.asarray(value)
    return {"count": int(values.size), "mean": float(values.mean()) if values.size else None,
        "max": float(values.max()) if values.size else None}


def test_manual_prior_reconstructs_cosine_inside_both_boundaries(cli):
    result = cli.manual_control_envelope(annotation(), 12, "independent-synthetic-source")
    np.testing.assert_allclose(result[3:9, 0], [.25, .75, 1., 1., .75, .25], atol=1e-7)
    assert not result[:3].any() and not result[9:].any() and not result[:, 1].any()


@pytest.mark.parametrize("change", ["wrong_role", "wrong_source", "right", "boolean_frame", "overlap", "negative_ramp", "duplicate_side"])
def test_manual_prior_rejects_ambiguous_or_unbound_support(cli, change):
    item = annotation()
    row = item["eligible_intervals_inclusive"][0]
    if change == "wrong_role": item["role"] = "image_observation"
    elif change == "wrong_source": item["video_sha256"] = "different-source"
    elif change == "right": row["eligible_sides"] = ["right"]
    elif change == "boolean_frame": row["first"] = True
    elif change == "overlap": item["eligible_intervals_inclusive"].append(row.copy())
    elif change == "negative_ramp": row["transition_frames"] = -1
    elif change == "duplicate_side": row["eligible_sides"] = ["left", "left"]
    with pytest.raises(ValueError):
        cli.manual_control_envelope(item, 12, "independent-synthetic-source")


def fixture_arrays(cli):
    old, _, _ = cli.helpers(Path(__file__).resolve().parents[1])
    manual = cli.manual_control_envelope(annotation(), 12, "independent-synthetic-source")
    original = np.zeros((12, 2), bool)
    original[:, 1] = True
    confidence = np.zeros((12, 2, 21))
    confidence[:, 1] = 1
    heldout = np.arange(12) % 5 == 0
    weights = confidence.copy()
    weights[heldout] = 0
    old_arrays = {"original_support": original, "weights": weights, "effective_envelope": original.astype(np.float32)}
    envelope = np.maximum(old_arrays["effective_envelope"], manual)
    arrays = {role: {"support": envelope > 0, "weights": weights.copy(), "original_support": original.copy(),
        "control_prior_array": manual.copy(), "image_envelope": old_arrays["effective_envelope"].copy(),
        "effective_envelope": envelope.copy(), "arm_envelope": np.repeat(envelope[:, :, None], 7, axis=2)} for role in ("a", "b")}
    return arrays, manual, old_arrays, confidence, heldout, old


def test_hidden_control_eligibility_is_shared_without_invented_image_targets(cli):
    arrays, manual, old_arrays, confidence, heldout, old = fixture_arrays(cli)
    result = cli.validate_prior_arrays(arrays, manual, old_arrays, confidence, heldout, .05, old)
    assert result["manual_prior_positive_frames"] == [6, 0]
    assert result["image_support_and_exact_weights_equal_old_a"]
    assert not arrays["a"]["weights"][:, 0].any()
    assert not arrays["a"]["weights"][heldout].any()


def test_old_arm_support_keeps_internal_gaps_but_not_extrapolated_detections(cli):
    confidence = np.zeros((12, 2, 21))
    confidence[[3, 6], 0] = 1
    config = {"min_confidence": .05, "min_landmarks": 8, "max_gap_frames": 3,
        "support_transition_frames": 0, "unobserved_policy": "preserve"}
    result = cli.image_arm_envelope(confidence, np.zeros(12, bool), config)
    assert result[3:7, 0].all()
    assert not result[:3].any() and not result[7:].any() and not result[:, 1].any()


def test_shared_arm_mask_cannot_expand_beyond_both_allowed_sources(cli):
    arrays, manual, old_arrays, confidence, heldout, old = fixture_arrays(cli)
    expected_arm = np.repeat(old_arrays["effective_envelope"][:, :, None], 7, axis=2)
    for current in arrays.values():
        current["arm_envelope"][1, 0, 0] = 1
    with pytest.raises(ValueError, match="expanded outside"):
        cli.validate_prior_arrays(arrays, manual, old_arrays, confidence, heldout, .05, old, expected_arm=expected_arm)


@pytest.mark.parametrize("change", ["b_only_arm", "image_weights", "unsupported_weight", "saved_manual", "effective_envelope", "image_collar"])
def test_prior_support_cannot_hide_changed_controls_or_evidence(cli, change):
    arrays, manual, old_arrays, confidence, heldout, old = fixture_arrays(cli)
    if change == "b_only_arm": arrays["b"]["arm_envelope"][4, 0, 0] = .1
    else:
        for current in arrays.values():
            if change == "image_weights": current["weights"][1, 1, 1] = .9
            elif change == "unsupported_weight": current["weights"][4, 0, 1] = .2
            elif change == "saved_manual": current["control_prior_array"][4, 0] = .1
            elif change == "effective_envelope": current["effective_envelope"][4, 0] = .1
            elif change == "image_collar": current["image_envelope"][4, 0] = .1
    with pytest.raises(ValueError):
        cli.validate_prior_arrays(arrays, manual, old_arrays, confidence, heldout, .05, old)


def test_ramped_finger_and_arm_budget_rejects_delta_near_mask_boundary(cli):
    base, pose = np.zeros((12, 136)), np.zeros((12, 136))
    hands = np.zeros((12, 2, 21, 3))
    envelope = cli.manual_control_envelope(annotation(), 12, "independent-synthetic-source")
    arm = np.repeat(envelope[:, :, None], 7, axis=2)
    config = {"max_delta_radians": .65, "arm_delta_limit_radians": .1, "wrist_max_displacement_metres": .04}
    pose[3, 95] = .2  # The full .65 bound would pass; the .25 ramp permits only .1625.
    with pytest.raises(ValueError, match="ramped envelope"):
        cli.update_diagnostics(base, pose, hands, hands, summary, [], envelope=envelope, arm_envelope=arm, config=config)
    pose[3, 95] = .1
    pose[3, 43] = .03
    with pytest.raises(ValueError, match="Arm updates"):
        cli.update_diagnostics(base, pose, hands, hands, summary, [], envelope=envelope, arm_envelope=arm, config=config)


def test_cumulative_updates_are_exposed_even_when_final_step_is_zero(cli):
    base = np.zeros((12, 136))
    seed = base.copy()
    seed[:, 95] = .4
    hands = np.zeros((12, 2, 21, 3))
    baseline = cli.update_diagnostics(base, seed, hands, hands, summary, [])
    step = cli.update_diagnostics(seed, seed, hands, hands, summary, [])
    assert baseline["sides"]["left"]["phase_summaries"][0]["finger_absolute_delta_radians"]["max"] == .4
    assert step["sides"]["left"]["phase_summaries"][0]["finger_absolute_delta_radians"]["max"] == 0


def test_expanded_solver_limits_are_left_prior_only_and_do_not_relax_motion_guards(cli):
    manual = cli.manual_control_envelope(annotation(), 12, "independent-synthetic-source")
    cfg = {"arm_delta_limit_radians": .1, "wrist_max_displacement_metres": .04,
        "control_prior_arm_delta_limit_radians": .35, "control_prior_wrist_max_displacement_metres": .12}
    arm, wrist = cli.declared_control_limits(manual, cfg)
    np.testing.assert_allclose(arm[:, 1], .1)
    np.testing.assert_allclose(wrist[:, 1], .04)
    assert wrist[5, 0] == pytest.approx(.12) and arm[5, 0, 0] == pytest.approx(.35)
    assert wrist[3, 0] == pytest.approx(.06) and arm[3, 0, 0] == pytest.approx(.1625)
    assert cli.validate_frozen_criteria(criteria())


def test_inactive_fingers_cannot_change_and_wrist_budget_is_absolute(cli):
    base, pose = np.zeros((12, 136)), np.zeros((12, 136))
    hands = np.zeros((12, 2, 21, 3))
    envelope = cli.manual_control_envelope(annotation(), 12, "independent-synthetic-source")
    pose[2, 95] = .01
    with pytest.raises(ValueError, match="outside declared"):
        cli.update_diagnostics(base, pose, hands, hands, summary, [], envelope=envelope)
    pose[2, 95] = 0
    moved = hands.copy()
    moved[5, 0, 0, 2] = .041
    with pytest.raises(ValueError, match="wrist displacement"):
        cli.update_diagnostics(base, pose, hands, moved, summary, [], envelope=envelope,
            arm_envelope=np.repeat(envelope[:, :, None], 7, axis=2),
            config={"max_delta_radians": .65, "arm_delta_limit_radians": .1, "wrist_max_displacement_metres": .04})


def test_boundary_windows_retain_both_entry_exit_and_original_ids(cli):
    manual = cli.manual_control_envelope(annotation(), 12, "independent-synthetic-source")
    windows = cli.boundary_windows(manual, radius=2)
    entries = [row for row in windows if row["boundary_kind"] == "eligibility_entry_or_exit"]
    assert [row["eligibility_boundary_original_frame_id"] for row in entries] == [3, 9]
    assert [(r["first"], r["last"]) for r in entries] == [(1, 5), (7, 11)]
    assert [r["eligibility_boundary_original_frame_id"] for r in windows if r["boundary_kind"] == "ramp_plateau_endpoint"] == [5, 6]


def test_projective_wrist_trust_is_checked_without_inventing_image_targets(cli):
    pose = np.zeros((12,136))
    before = np.zeros((12,2,21,3))
    before[...,2] = 3
    after = before.copy()
    after[5,0,0,0] = .1
    manual = cli.manual_control_envelope(annotation(),12,"independent-synthetic-source")
    with pytest.raises(ValueError,match="projective trust"):
        cli.update_diagnostics(pose,pose,before,after,summary,[],config={"control_prior_wrist_max_projection_pixels":3.0},
            camera=np.diag([300.,300.,1.]),manual=manual)


def test_same_anchor_diagnostics_do_not_mix_hands_or_invent_missing_contact(cli):
    hands = np.zeros((3, 2, 21, 3))
    targets = np.zeros((3, 2, 5, 3))
    targets[:, 0, :, 0] = .01
    targets[:, 1, :, 0] = .03
    weights = np.ones((3, 2, 5))
    candidate = {"object_to_camera": np.repeat(np.eye(4)[None], 3, axis=0)}
    old, _, _ = cli.helpers(Path(__file__).resolve().parents[1])
    result = cli.anchor_consistency_by_side(hands, candidate, targets, weights, old.to_object)
    assert result["sides"]["left"]["weighted_distance_mm"] == pytest.approx(10.)
    assert result["sides"]["right"]["weighted_distance_mm"] == pytest.approx(30.)
    assert not cli.anchor_consistency_by_side(hands, candidate, None, None, old.to_object)["available"]


def test_distributed_native_selection_is_stable_for_candidate_order_and_ties(cli):
    rest = np.array([[-1., 0., 0.], [1., 0., 0.], [0., -1., 0.], [0., 1., 0.], [0., 0., 0.]])
    first = cli.distributed_native_ids(rest, np.array([4, 3, 2, 1, 0]), count=3)
    second = cli.distributed_native_ids(rest, np.arange(5), count=3)
    np.testing.assert_array_equal(first, second)
    assert 0 in first and 1 in first


def test_surface_patch_selection_uses_training_and_shared_seed_only(cli):
    signed = np.zeros((12, 2, 128))
    signed[:] = np.arange(128)[None, None] / 1000
    weights = np.ones((12, 2))
    signed[np.arange(12) % 5 == 0] = signed[np.arange(12) % 5 == 0, :, ::-1]
    selected = cli.select_patch_positions(signed, weights, np.arange(12))
    np.testing.assert_array_equal(selected, np.tile(np.arange(16), (2, 1)))


def test_surface_worst_term_uses_temporal_max_not_weighted_mean(cli):
    normal, offset = planes()
    candidate = {"object_to_camera": np.repeat(np.eye(4)[None], 3, axis=0), "normals": normal, "offsets": offset}
    candidate["object_to_camera"][:, 2, 3] = 3
    hands = np.zeros((3, 2, 21, 3))
    hands[..., 2] = 5
    signed = np.ones((3, 2, 128)) * .01
    signed[1, :, 0] = -.013  # One1cm penetration beyond the3mm soft margin.
    weights = np.ones((3, 2))
    configuration = {"penetration_weight": .1, "surface_weight": .01, "far_order_weight": .1}
    result = cli.surface_surrogate_terms(signed, hands, candidate, weights, np.tile(np.arange(16), (2, 1)), configuration)
    assert result["surface_skin_penetration"] == pytest.approx(1 / (3 * 128) + .25)
    assert result["surface_left_wrist_palm_order"] == 0


def test_fixed_seed_far_order_cannot_escape_by_moving_current_ray_outside_object(cli):
    normal, offset = planes()
    candidate = {"object_to_camera": np.repeat(np.eye(4)[None], 3, axis=0), "normals": normal, "offsets": offset}
    candidate["object_to_camera"][:, 2, 3] = 3
    seed = np.zeros((3, 2, 21, 3))
    seed[..., 2] = 3
    escaped = seed.copy()
    escaped[:, 0, :, 0] = 6
    signed = np.ones((3, 2, 128)) * .01
    weights = np.ones((3, 2))
    configuration = {"penetration_weight": .1, "surface_weight": .01, "far_order_weight": .1}
    dynamic = cli.surface_surrogate_terms(signed, escaped, candidate, weights, np.tile(np.arange(16), (2, 1)), configuration)
    configuration["order_ray_mode"] = "fixed_shared_seed"
    fixed = cli.surface_surrogate_terms(signed, escaped, candidate, weights, np.tile(np.arange(16), (2, 1)), configuration, seed)
    assert dynamic["surface_left_wrist_palm_order"] == 0
    assert fixed["surface_left_wrist_palm_order"] > 100
    assert fixed["surface_order_reference_ray_intersection_fraction"] == 1
    assert fixed["surface_order_current_ray_intersection_fraction"] == 0


def test_exact_triangle_audit_does_not_call_max_plane_outside_exact_distance(cli):
    from v2hoi.hand_contact import closest_mesh_points
    vertices = np.zeros((3, 18439, 3))
    vertices[:] = [1.5, 1.5, 3.]
    normal, offset = planes()
    cube = np.array([[-1.,-1.,-1.],[-1.,-1.,1.],[-1.,1.,-1.],[-1.,1.,1.],
                     [1.,-1.,-1.],[1.,-1.,1.],[1.,1.,-1.],[1.,1.,1.]])
    faces = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                      [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
    candidate = {"object_to_camera": np.repeat(np.eye(4)[None], 3, axis=0), "normals": normal,
        "offsets": offset, "vertices": cube, "faces": faces, "pose_usable": np.array([1,0,1], bool)}
    candidate["object_to_camera"][:, 2, 3] = 3
    hands = np.zeros((3, 2, 21, 3))
    hands[..., 2] = 5
    ids = np.arange(256).reshape(2, 128)
    patches = np.tile(np.arange(16), (2, 1))
    value, signed = cli.surface_patch_diagnostics(vertices, hands, candidate, ids, patches, np.ones((3, 2)),
        closest_mesh_points, summary, [0, 2], {"penetration_weight": .1, "surface_weight": .01, "far_order_weight": .1})
    assert signed[0, 0, 0] == .5
    row = value["predefined_exact_triangle_sample_rows"][0]
    assert row["absolute_max_plane_surrogate_mm"][0] == 500
    assert row["exact_triangle_distances_mm"][0] == pytest.approx(1000 * np.sqrt(.5), rel=1e-6)
    assert len(value["all_original_frame_rows"]) == 3


def test_independent_numpy_terms_match_torch_surface_intervention_on_synthetic_geometry(cli):
    torch = pytest.importorskip("torch")
    from v2hoi.hand_surface_contact import NativeHandSurfaceContact
    from v2hoi.hand_contact import validate_object_candidate
    count = 6
    cube = np.array([[-1.,-1.,-1.],[-1.,-1.,1.],[-1.,1.,-1.],[-1.,1.,1.],
                     [1.,-1.,-1.],[1.,-1.,1.],[1.,1.,-1.],[1.,1.,1.]], np.float32)
    faces = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                      [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    poses[:, 2, 3] = 3
    raw = {"vertices": cube, "faces": faces, "object_to_camera": poses,
        "frame_indices": np.arange(count), "pose_usable": np.ones(count, np.uint8)}
    phase = {"track": 1, "frame_count": count, "phase_intervals_inclusive": [
        {"first": 0, "last": count-1, "anchor_weight": 1., "eligible_sides": ["left", "right"]}]}
    points = np.zeros((count, 256, 3), np.float32)
    points[:, :, 0] = np.linspace(-1.02, 1.02, 256)
    points[:, :, 1] = .1
    points[:, :, 2] = 3.1

    class SyntheticSurface:
        device = "cpu"
        count_per_hand = 128
        hand_output_indices = [np.arange(128), np.arange(128, 256)]
        hand_vertex_ids = hand_output_indices

        def __call__(self, pose):
            return torch.as_tensor(points[np.rint(pose[:, 0].cpu().numpy()).astype(int)])

    pose = np.zeros((count, 136), np.float32)
    pose[:, 0] = np.arange(count)
    joints = np.zeros((count, 2, 21, 3), np.float32)
    joints[..., 2] = 2.9
    configuration = {"method": "native_hand_surface_contact.v1", "contact_patch_vertices_per_hand": 16, "sample_vertices_per_hand": 128,
        "penetration_weight": .1, "surface_weight": .02, "far_order_weight": .1}
    constraint = NativeHandSurfaceContact(raw, phase, SyntheticSurface(), pose, joints, np.arange(count), configuration)
    declared = constraint.diagnostics(pose, joints)["terms"]
    candidate = validate_object_candidate(raw, np.arange(count))
    full = np.zeros((count, 18439, 3), np.float32)
    full[:, :256] = points
    ids = np.arange(256).reshape(2, 128)
    _, signed = cli.selected_surface_signed(full, candidate, ids)
    patches = np.array([[np.flatnonzero(ids[side] == vertex)[0] for vertex in constraint.patch_native_vertex_ids[side]] for side in range(2)])
    actual = cli.surface_surrogate_terms(signed, joints.astype(float), candidate, constraint.weights, patches, configuration)
    for key in declared:
        assert actual[key] == pytest.approx(declared[key], rel=2e-5, abs=1e-5)


def planes():
    normals = np.r_[np.eye(3), -np.eye(3)]
    return normals, np.ones(6)


def test_camera_ray_returns_near_far_depth_and_abstains_for_missing_box(cli):
    normal, offset = planes()
    pose = np.eye(4)
    pose[2, 3] = 3
    points = np.array([[0., 0., 2.5], [0., 0., 5.], [5., 0., 2.]])
    near, far, hit = cli.convex_ray_bounds(points, pose, normal, offset)
    np.testing.assert_allclose(near[:2], 2.)
    np.testing.assert_allclose(far[:2], 4.)
    assert hit.tolist() == [True, True, False]
    assert np.isnan(near[2]) and np.isnan(far[2])


def test_camera_ray_proper_rigid_inverse_and_rotated_surface(cli):
    normal, offset = planes()
    angle = np.pi / 4
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    pose[:3, 3] = [.3, .2, 4.]
    points = np.array([[.3, .2, 4.], [.1, .1, 3.]])
    near, far, hit = cli.convex_ray_bounds(points, pose, normal, offset)
    assert hit.all() and (near < far).all()
    for depths in (near, far):
        surface = (points / points[:, 2:] * depths[:, None] - pose[:3, 3]) @ pose[:3, :3]
        clearance = surface @ normal.T - offset
        assert np.max(clearance) <= 1e-10
        np.testing.assert_allclose(clearance.max(-1), 0, atol=1e-10)


def test_invalid_camera_depth_is_not_a_hidden_visibility_label(cli):
    normal, offset = planes()
    with pytest.raises(ValueError, match="positive-depth"):
        cli.convex_ray_bounds(np.zeros((1, 3)), np.eye(4), normal, offset)


def test_ordering_summaries_retain_unreliable_frames_and_explicit_abstention(cli):
    normal, offset = planes()
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 2, 3] = 3
    hands = np.zeros((3, 2, 21, 3))
    hands[:, 0, :, 2] = 5  # Behind the whole object: not a measured true contact.
    hands[:, 1, :, 2] = 1  # In front.
    candidate = {"object_to_camera": poses, "normals": normal, "offsets": offset,
        "pose_usable": np.array([True, False, True])}
    result = cli.ordering_diagnostics(hands, candidate, summary, first=0, last=2)
    assert len(result["rows"]) == 6
    assert result["sides"]["left"]["all_original_frames"]["intersecting_joint_samples"] == 63
    assert result["sides"]["left"]["same_fixed_pose_usable_gate"]["intersecting_joint_samples"] == 42
    assert result["sides"]["left"]["training"]["frames"] == 2
    assert result["sides"]["left"]["stride5_holdout"]["frames"] == 1
    assert not result["sides"]["left"]["all_original_frames"]["coarse_order_conflict_samples"]
    assert "not skin visibility" in result["interpretation"]


def criteria():
    return {"image_reference_policy": {"relative_tolerance": 1.05, "absolute_tolerance_pixels": .25},
        "motion_policy": {"observed_velocity_relative_tolerance": 1.05, "observed_velocity_absolute_tolerance_pixels": .25,
            "unknown_motion_max_correction_step_pixels": 1., "unknown_motion_max_correction_step_native_estimated_metres": .005},
        "skin_b_versus_a": {"maximum_mean_depth_increase_mm": .5, "maximum_worst_vertex_depth_increase_mm": 2.}}


def test_frozen_guard_limits_cannot_be_relaxed_for_new_hidden_hand(cli):
    values = criteria()
    assert cli.validate_frozen_criteria(values)
    relaxed = copy.deepcopy(values)
    relaxed["motion_policy"]["unknown_motion_max_correction_step_pixels"] = 2
    with pytest.raises(ValueError, match="tolerance changed"):
        cli.validate_frozen_criteria(relaxed)
