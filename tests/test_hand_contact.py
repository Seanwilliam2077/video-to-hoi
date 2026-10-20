"""Synthetic convex objects only; no challenge meshes, trajectories or labels."""
import numpy as np
import pytest

from v2hoi.hand_contact import (
    WeakContactConstraint, closest_mesh_points, convex_mesh_planes,
    native_hand_vertex_indices, native_mesh_penetration_proxy, phase_weights,
    validate_object_candidate,
)


def cube(frames=5):
    vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float32) * 0.1
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 5, 1], [0, 4, 5],
                      [3, 2, 6], [3, 6, 7], [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]])
    transform = np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0)
    transform[:, 2, 3] = 5
    return {"vertices": vertices, "faces": faces, "object_to_camera": transform, "frame_indices": np.arange(frames)}


def phases(frames=5):
    return {"track": 1, "frame_count": frames, "phase_intervals_inclusive": [
        {"first": 0, "last": 0, "anchor_weight": 0},
        {"first": 1, "last": frames - 2, "anchor_weight": 1, "eligible_sides": ["left", "right"]},
        {"first": frames - 1, "last": frames - 1, "anchor_weight": 0},
    ]}


def test_convex_supporting_planes_recover_interior_clearance_regardless_of_winding():
    candidate = cube()
    normals, offsets = convex_mesh_planes(candidate["vertices"], candidate["faces"])
    for point, expected in [([0, 0, 0], -0.1), ([0.08, 0, 0], -0.02), ([0.12, 0, 0], 0.02)]:
        assert np.max(np.asarray(point) @ normals.T - offsets) == pytest.approx(expected, abs=1e-6)


def test_open_and_nonconvex_meshes_are_rejected():
    candidate = cube()
    with pytest.raises(ValueError, match="closed"):
        convex_mesh_planes(candidate["vertices"], candidate["faces"][:-1])
    candidate["vertices"][0] = [0, 0, 0]
    with pytest.raises(ValueError, match="convex"):
        convex_mesh_planes(candidate["vertices"], candidate["faces"])


def test_exact_surface_projection_handles_faces_edges_corners_and_interior():
    candidate = cube()
    points = np.array([[0.12, 0, 0], [0.2, 0.2, 0], [0.2, 0.2, 0.2], [0.09, 0, 0]])
    closest, distance = closest_mesh_points(points, candidate["vertices"], candidate["faces"], chunk_size=2)
    np.testing.assert_allclose(closest, [[0.1, 0, 0], [0.1, 0.1, 0], [0.1, 0.1, 0.1], [0.1, 0, 0]], atol=1e-6)
    np.testing.assert_allclose(distance, [0.02, 2**0.5 * 0.1, 3**0.5 * 0.1, 0.01], atol=1e-6)


def test_object_frames_and_proper_rotation_are_required():
    candidate = cube()
    with pytest.raises(ValueError, match="indices"):
        validate_object_candidate(candidate, np.arange(5) + 1)
    candidate["object_to_camera"][0, 0, 0] = -1
    with pytest.raises(ValueError, match="proper"):
        validate_object_candidate(candidate, np.arange(5))


def test_contact_phase_does_not_attract_through_pickup_or_release():
    weight = phase_weights(phases(), 5)
    assert not weight[[0, 4]].any()
    assert (weight[1:4] > 0).all()
    bad = phases()
    bad["phase_intervals_inclusive"][1]["first"] = 0
    with pytest.raises(ValueError, match="overlap"):
        phase_weights(bad, 5)


def test_actual_native_mesh_vertex_depth_is_separate_from_contact_point_proxy():
    candidate = validate_object_candidate(cube(1), np.arange(1))
    vertices = np.zeros((1, 18439, 3), dtype=np.float32)
    vertices[0, 0] = [0, 0, 5]
    vertices[0, 1] = [0.12, 0, 5]
    result = native_mesh_penetration_proxy(vertices, [np.array([0]), np.array([1])], candidate, np.ones((1, 2)))
    assert result["report"]["left"]["max_vertex_depth_mm"] == pytest.approx(100, abs=1e-3)
    assert result["report"]["right"]["max_vertex_depth_mm"] == 0
    assert result["per_frame_penetrating_vertices"].tolist() == [[1, 0]]
    assert "not volume" in result["report"]["interpretation"]


def _contact_problem():
    torch = pytest.importorskip("torch", reason="Optional Torch runtime for differentiable contact")
    baseline = np.zeros((5, 2, 21, 3), dtype=np.float32)
    baseline[..., 0] = 0.105
    baseline[..., 2] = 5
    return torch, baseline


def test_contact_selects_few_existing_tips_and_has_no_gradient_outside_grasp():
    torch, baseline = _contact_problem()
    constraint = WeakContactConstraint(cube(), phases(), baseline, np.arange(5))
    assert len(constraint.admitted) == 4
    assert not constraint.anchor_weights[[0, 4]].any()
    variable = torch.tensor(baseline, requires_grad=True)
    terms = constraint.loss(variable)
    terms["contact_total"].backward()
    assert torch.isfinite(variable.grad).all()
    assert variable.grad[1:4].abs().sum() > 0
    assert variable.grad[[0, 4]].abs().sum() == 0
    assert constraint.before["attraction_weight_outside_admitted_phases"] == 0


def test_contact_abstains_for_no_nearby_fingertips_or_object_pose_jumps():
    _, baseline = _contact_problem()
    with pytest.raises(ValueError, match="No baseline fingertip"):
        WeakContactConstraint(cube(), phases(), baseline + [1, 0, 0], np.arange(5))
    candidate = cube()
    candidate["object_to_camera"][2, 0, 3] = 0.2
    with pytest.raises(ValueError, match="trajectory jumps"):
        WeakContactConstraint(candidate, phases(), baseline, np.arange(5))


def test_object_pose_reliability_gate_disables_contact_even_inside_grasp():
    torch = pytest.importorskip("torch")
    candidate = cube(8)
    candidate["pose_usable"] = np.ones(8, dtype=np.uint8)
    candidate["pose_usable"][3] = 0
    baseline = np.zeros((8, 2, 21, 3), np.float32) + [0.105, 0, 5]
    constraint = WeakContactConstraint(candidate, phases(8), baseline, np.arange(8))
    assert not constraint.weights[3].any()
    variable = torch.tensor(baseline, dtype=torch.float32, requires_grad=True)
    constraint.loss(variable)["contact_total"].backward()
    assert variable.grad[3].abs().sum() == 0


def test_native_hand_skin_selection_uses_real_weight_sums_not_joint_as_vertex_indices():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    names = [""] * 127
    names[78], names[100], names[42], names[64] = "l_wrist", "l_thumb_null", "r_wrist", "r_thumb_null"
    skin = SimpleNamespace(skin_indices_flattened=torch.tensor([78, 97, 42, 56, 5]),
                           vert_indices_flattened=torch.tensor([0, 0, 1, 1, 2]),
                           skin_weights_flattened=torch.tensor([0.4, 0.6, 0.2, 0.3, 1.0]))
    model = SimpleNamespace(character_torch=SimpleNamespace(skeleton=SimpleNamespace(joint_names=names), linear_blend_skinning=skin))
    selected = native_hand_vertex_indices(model)
    np.testing.assert_array_equal(selected[0], [0])
    np.testing.assert_array_equal(selected[1], [1])


def test_contact_enters_true_hand_sequence_objective_without_disconnected_mesh_updates():
    torch = pytest.importorskip("torch")
    from v2hoi.hand_refinement import HandFitConfig, refine_hands

    frames = 8
    baseline = {"pose": np.zeros((frames, 136), np.float32), "shape": np.zeros(45, np.float32),
                "scales": np.zeros(68, np.float32), "frame_indices": np.arange(frames), "fps": np.array(30)}
    def forward(pose, scales):
        hands = []
        for offset in (95, 68):
            controls = torch.cat([pose.new_zeros(len(pose), 1), pose[:, offset:offset + 20]], dim=1)
            x = 0.105 + controls * 0.02
            hands.append(torch.stack([x, x * 0, x * 0 + 5], dim=-1))
        return torch.stack(hands, dim=1)
    xyz = forward(torch.tensor(baseline["pose"]), None).numpy()
    k = np.diag([1000, 1000, 1]).astype(np.float32)
    obs = {"frame_indices": np.arange(frames), "side_order": np.array(["left", "right"]),
           "joints2d": (xyz @ k.T)[..., :2] / xyz[..., 2:], "confidence": np.ones((frames, 2, 21), np.float32)}
    constraint = WeakContactConstraint(cube(frames), phases(frames), xyz, np.arange(frames))
    result = refine_hands(baseline, obs, k, forward, config=HandFitConfig(iterations=5, holdout_stride=0),
                          contact_constraint=constraint, log=lambda _: None)
    assert result["report"]["object_contact_constraints"]
    assert result["report"]["contact"]["status"].startswith("experimental")
    assert result["report"]["final_terms"]["contact_total"] >= 0
    assert np.isfinite(result["pose"]).all()
    np.testing.assert_array_equal(result["pose"][:, :68], baseline["pose"][:, :68])
