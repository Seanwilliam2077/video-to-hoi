"""Independent cube/skin fixtures; no challenge data, model weights or labels."""
import numpy as np
import pytest

from v2hoi.hand_surface_contact import (
    NativeHandSurfaceContact, convex_far_ray_depth, distributed_vertex_selection,
)


def cube(frames):
    vertices = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]], np.float32)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int64)
    transforms = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    transforms[:, 2, 3] = 5
    return {"vertices": vertices, "faces": faces, "frame_indices": np.arange(frames),
            "object_to_camera": transforms, "pose_usable": np.ones(frames, np.uint8)}


def test_rest_sampling_is_deterministic_complete_spatial_coverage_without_mutation():
    xyz = np.array([[x, y, z] for x in range(4) for y in range(4) for z in range(4)], np.float32)
    candidates = np.arange(64)[::-1]
    sample = distributed_vertex_selection(xyz, candidates, 8)
    np.testing.assert_array_equal(sample, distributed_vertex_selection(xyz, candidates[::-1], 8))
    assert len(np.unique(sample)) == 8
    assert np.max(np.min(np.linalg.norm(xyz[:, None] - xyz[sample][None], axis=-1), axis=1)) < 2.5
    assert candidates[0] == 63


@pytest.mark.parametrize("count", [0, 65, True, 2.5])
def test_invalid_sampling_count_is_rejected(count):
    with pytest.raises(ValueError):
        distributed_vertex_selection(np.zeros((64, 3)), np.arange(64), count)


def test_convex_ray_far_intersection_is_depth_correct_and_missed_rays_supply_no_target():
    torch = pytest.importorskip("torch")
    normals = torch.tensor(np.r_[np.eye(3), -np.eye(3)], dtype=torch.float32)
    offsets = torch.ones(6)
    rotation = torch.eye(3)[None]
    translation = torch.tensor([[0., 0., 5.]])
    queries = torch.tensor([[[0., 0., 4.5], [4., 0., 5.]]], requires_grad=True)
    far, valid = convex_far_ray_depth(queries, rotation, translation, normals, offsets)
    assert valid.tolist() == [[True, False]]
    assert float(far[0, 0].detach()) == pytest.approx(6)
    assert far[0, 1] == 0
    (far.sum() + queries.sum() * 0).backward()
    assert torch.isfinite(queries.grad).all()


def synthetic_surface():
    torch = pytest.importorskip("torch")

    class Surface:
        device = torch.device("cpu")
        count_per_hand = 128
        hand_vertex_ids = [np.arange(128), np.arange(128, 256)]
        hand_output_indices = hand_vertex_ids

        def __call__(self, pose):
            y = torch.linspace(-.1, .1, 128)[None].expand(len(pose), -1)
            z = torch.full_like(y, 5)
            shift = pose[:, :1] * .01
            inward = pose[:, 1:2]
            left = torch.stack([y * 0 + 1.01 + inward, y + shift, z], -1)
            right = torch.stack([y * 0 - 1.01 - inward, y + shift, z], -1)
            return torch.cat([left, right], 1)

    return Surface()


def make_constraint(mode="current_query"):
    torch = pytest.importorskip("torch")
    frames = 10
    poses = np.zeros((frames, 136), np.float32)
    hands = np.zeros((frames, 2, 21, 3), np.float32)
    hands[..., 2] = 6.2
    phases = {"track": 1, "frame_count": frames, "phase_intervals_inclusive": [
        {"first": 0, "last": 9, "eligible_sides": ["left", "right"], "anchor_weight": 1}]}
    configuration = {"method": "native_hand_surface_contact.v1", "sample_vertices_per_hand": 128,
                     "contact_patch_vertices_per_hand": 16, "penetration_weight": .1,
                     "surface_weight": .02, "far_order_weight": .1, "order_ray_mode": mode}
    result = NativeHandSurfaceContact(cube(frames), phases, synthetic_surface(), poses, hands,
                                      np.arange(frames), configuration)
    return torch, result, poses, hands


def test_frozen_surface_patch_allows_tangential_slide_without_a_no_slip_point_target():
    torch, constraint, poses, hands = make_constraint()
    first = constraint.loss(torch.tensor(poses), torch.tensor(hands))
    shifted = torch.tensor(poses)
    shifted[:, 0] = 1  # Tangential 1cm, staying inside the same box face.
    second = constraint.loss(shifted, torch.tensor(hands))
    assert first["surface_sliding_attraction"] == pytest.approx(float(second["surface_sliding_attraction"]))
    assert first["surface_skin_penetration"] == second["surface_skin_penetration"] == 0
    assert constraint.patch_native_vertex_ids[0].tolist() == list(range(16))


def test_actual_selected_skin_penetration_and_palm_order_have_finite_nonzero_gradients():
    torch, constraint, poses, hands = make_constraint()
    pose = torch.tensor(poses, requires_grad=True)
    pose.data[:, 1] = -.2
    joints = torch.tensor(hands, requires_grad=True)
    joints.data[:, 0, :, 2] = 5.8
    values = constraint.loss(pose, joints)
    assert values["surface_skin_penetration"] > 0
    assert values["surface_left_wrist_palm_order"] > 0
    values["surface_contact_total"].backward()
    assert torch.isfinite(pose.grad).all() and torch.isfinite(joints.grad).all()
    assert pose.grad[:, 1].abs().max() > 0
    assert joints.grad[:, 0, :, 2].abs().max() > 0
    assert not joints.grad[:, 1].any()


def test_unreliable_object_frames_and_unadmitted_phases_do_not_receive_contact_gradient():
    torch, constraint, poses, hands = make_constraint()
    pose = torch.tensor(poses, requires_grad=True)
    pose.data[:, 1] = -.2
    constraint.phase[3:5] = 0
    values = constraint.loss(pose, torch.tensor(hands))
    values["surface_contact_total"].backward()
    assert not pose.grad[3:5].any()
    assert pose.grad[1].abs().max() > 0


def test_fixed_seed_order_cannot_be_removed_by_moving_a_query_ray_off_the_object():
    torch, current_constraint, poses, hands = make_constraint()
    _, fixed_constraint, _, _ = make_constraint("fixed_shared_seed")
    query = torch.tensor(hands, requires_grad=True)
    query.data[:, 0, :, 2] = 5.8
    query.data[:, 0, :, 0] = 4  # Current rays miss; seed reference rays still hit.
    current = current_constraint.loss(torch.tensor(poses), query)
    fixed = fixed_constraint.loss(torch.tensor(poses), query)
    assert current["surface_left_wrist_palm_order"] == 0
    assert fixed["surface_left_wrist_palm_order"] > 0
    assert fixed["surface_order_current_ray_intersection_fraction"] == 0
    assert fixed["surface_order_reference_ray_intersection_fraction"] == 1
    assert fixed["surface_order_ray_intersection_fraction"] == 1
    fixed["surface_contact_total"].backward()
    assert torch.isfinite(query.grad).all() and query.grad[:, 0, :, 2].abs().max() > 0
