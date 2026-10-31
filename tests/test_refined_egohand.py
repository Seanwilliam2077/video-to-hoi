"""Independent synthetic geometry only; no model or challenge asset loading."""
from dataclasses import asdict

import numpy as np
import pytest

from v2hoi.egohand_adapter import project_camera_points
from v2hoi.refined_egohand import RefinedHandPolicy, build_refined_crops, retarget_egohand_proposals


def fixture(count=25):
    k = np.array([[300., 0, 256], [0, 300., 192], [0, 0, 1]])
    local = np.zeros((count, 2, 21, 3))
    for joint in range(1, 21):
        local[:, :, joint, 0] = ((joint-1)//4-2)*.025
        local[:, :, joint, 1] = ((joint-1)%4+1)*.025
    native = local + np.array([0, 0, 3.])
    native[:, 1, :, 0] += .6
    ego = local + np.array([0, 0, 1.])
    ego[:, 1, :, 0] += .2
    pixels, _ = project_camera_points(native, k)
    boxes = np.tile([200, 144, 312, 256], (count, 2, 1)).astype(np.int32)
    boxes[:, 1, [0, 2]] += 60
    obs = {"frame_indices": np.arange(count), "side_order": np.array(["left", "right"]),
           "joints2d": pixels.copy(), "confidence": np.ones((count, 2, 21)),
           "image_size": np.array([512, 384]), "crop_boxes": boxes}
    proposals = {"frame_indices": np.arange(count), "side_order": obs["side_order"].copy(),
                 "crop_boxes": boxes.copy(), "proposal_joints3d_camera": ego,
                 "proposal_camera_geometry_valid": np.ones((count, 2, 21), bool),
                 "pred_kp2d_crop_presence_soma25": np.ones((count, 2, 25))}
    return obs, proposals, native, k


def test_declared_defaults_are_fixed():
    policy = asdict(RefinedHandPolicy())
    assert policy["minimum_crop_side"] == 64
    assert policy["maximum_crop_side"] == 208
    assert policy["crop_padding"] == 1.5
    assert policy["center_median_window"] == 5
    assert policy["side_median_window"] == 15
    assert policy["maximum_crop_missing_frames"] == 8


def test_crop_excludes_heldout_positions_weights_and_preserves_input():
    obs, _, _, _ = fixture()
    before = obs["joints2d"].copy()
    boxes, arrays, _ = build_refined_crops(obs)
    assert np.array_equal(obs["joints2d"], before)
    assert not arrays["training_anchor_weights"][::5].any()
    changed = {k: v.copy() for k, v in obs.items()}
    changed["joints2d"][::5] = 100000
    changed["confidence"][::5] = 0
    boxes2, arrays2, _ = build_refined_crops(changed)
    assert np.array_equal(boxes, boxes2)
    assert np.array_equal(arrays["training_anchor_weights"], arrays2["training_anchor_weights"])


def test_tightened_boxes_cover_accepted_points_with_integer_margin():
    obs, _, _, _ = fixture()
    obs["joints2d"][12, :, :, 0] += 35
    boxes, arrays, _ = build_refined_crops(obs)
    assert boxes.dtype == np.int32
    assert np.all(boxes[..., :2] >= 0)
    assert np.all(boxes[..., 2:] <= obs["image_size"])
    assert np.array_equal(boxes[..., 2]-boxes[..., 0], boxes[..., 3]-boxes[..., 1])
    assert np.median(boxes[..., 2]-boxes[..., 0]) == 64
    for frame, side in np.argwhere(arrays["training_anchor_weights"] > 0):
        p = obs["joints2d"][frame, side]
        assert np.all(p >= boxes[frame, side, :2]+2)
        assert np.all(p <= boxes[frame, side, 2:]-2)


def test_only_short_crop_gaps_interpolate_without_observation_targets():
    obs, _, _, _ = fixture(30)
    obs["confidence"][:] = 0
    obs["confidence"][[1, 9, 21], :, :] = 1
    boxes, arrays, report = build_refined_crops(obs)
    assert arrays["crop_source_class"][5, 0] == "training_neighbor_crop_interpolation"
    assert not arrays["training_anchor_weights"][5].any()
    assert np.array_equal(boxes[0], obs["crop_boxes"][0])
    assert np.array_equal(boxes[10:21], obs["crop_boxes"][10:21])
    assert report["training_anchor_original_frame_ids"]["left"] == [1, 9, 21]


def test_full_detector_predictions_localize_crop_but_do_not_become_targets():
    obs, _, _, _ = fixture()
    obs["confidence"][:, :, 20] = 0
    obs["joints2d"][:, :, 20, 0] += 60
    _, arrays, _ = build_refined_crops(obs)
    assert np.all(arrays["training_anchor_weights"][1:5] == 1)
    assert np.array_equal(obs["confidence"][:, :, 20], np.zeros((25, 2)))


def test_no_supported_hand_keeps_original_crops():
    obs, _, _, _ = fixture()
    obs["confidence"][:] = 0
    boxes, arrays, _ = build_refined_crops(obs)
    assert np.array_equal(boxes, obs["crop_boxes"])
    assert not arrays["training_anchor_weights"].any()


def test_crop_boundary_coverage_and_infeasible_anchor_abstention():
    obs, _, _, _ = fixture()
    obs["joints2d"][:, :, :, 0] -= 246
    boxes, arrays, _ = build_refined_crops(obs)
    assert np.all(boxes[1:5, 0, 0] == 0)
    assert arrays["training_anchor_weights"][1, 0] == 1
    obs["joints2d"][1, 0, 2, 0] = -10
    _, arrays, _ = build_refined_crops(obs)
    assert arrays["training_anchor_weights"][1, 0] == 0


@pytest.mark.parametrize("bad", ["ids", "side", "box", "bounds"])
def test_invalid_crop_contract_rejected(bad):
    obs, _, _, _ = fixture()
    if bad == "ids": obs["frame_indices"][1] = 0
    elif bad == "side": obs["side_order"] = obs["side_order"][::-1]
    elif bad == "box": obs["crop_boxes"][0, 0, 2] += 1
    else: obs["crop_boxes"][0, 0] -= 300
    with pytest.raises(ValueError): build_refined_crops(obs)


def test_retarget_places_equal_size_hand_at_native_depth_and_mp_wrist():
    obs, proposal, native, k = fixture()
    obs["joints2d"][:, :, :, 0] += 4
    arrays, report = retarget_egohand_proposals(proposal, obs, native, intrinsics=k)
    assert np.allclose(arrays["fixed_native_to_ego_palm_scalar"], 1)
    assert report["fixed_scale_anchor_frames"] == [20, 20]
    assert np.allclose(arrays["proposal_joints3d_camera"][:, :, 0, 2], native[:, :, 0, 2])
    assert np.allclose(arrays["joints2d"][1:5], obs["joints2d"][1:5])
    assert not arrays["confidence"][::5].any()
    assert not arrays["shared_wrist_anchor_weights"][::5].any()
    assert not arrays["shared_visibility_gate_weights"][::5].any()
    assert str(arrays["observation_kind"]) == "egohand_geometric_proposals.v1"


def test_fixed_scale_and_articulation_are_preserved_not_flattened():
    obs, proposal, native, k = fixture()
    proposal["proposal_joints3d_camera"][:, :, 4, 1] += np.arange(25)[:, None]*.001
    source = proposal["proposal_joints3d_camera"].copy()
    arrays, _ = retarget_egohand_proposals(proposal, obs, native, intrinsics=k)
    actual = arrays["proposal_joints3d_camera"]-arrays["proposal_joints3d_camera"][:, :, :1]
    expected = (source-source[:, :, :1])*arrays["fixed_native_to_ego_palm_scalar"][None, :, None, None]
    assert np.allclose(actual, expected)
    assert np.ptp(actual[:, 0, 4, 1]) > .02
    assert np.array_equal(proposal["proposal_joints3d_camera"], source)


def test_mp_fingers_are_not_copied_and_no_residual_selection_gate():
    obs, proposal, native, k = fixture()
    original = obs["joints2d"].copy()
    obs["joints2d"][:, :, 4, 0] += 20
    arrays, _ = retarget_egohand_proposals(proposal, obs, native, intrinsics=k)
    assert np.allclose(arrays["joints2d"][1:5, :, 4], original[1:5, :, 4])
    assert arrays["confidence"][1, 0, 4] == 1


def test_heldout_mp_cannot_change_any_retargeted_geometry_or_gate():
    obs, proposal, native, k = fixture()
    arrays, _ = retarget_egohand_proposals(proposal, obs, native, intrinsics=k)
    obs["joints2d"][::5] = 100000
    obs["confidence"][::5] = 0
    mutated, _ = retarget_egohand_proposals(proposal, obs, native, intrinsics=k)
    for key in ("joints2d", "proposal_joints3d_camera", "confidence", "fixed_native_to_ego_palm_scalar",
                "shared_wrist_anchor_weights", "shared_visibility_gate_weights"):
        assert np.array_equal(arrays[key], mutated[key])


def test_raw_reference_fixes_same_scalar_for_smoothing_ablation():
    obs, raw, native, k = fixture()
    smooth = {key: value.copy() for key, value in raw.items()}
    smooth["proposal_joints3d_camera"] = raw["proposal_joints3d_camera"]*1.1
    a, _ = retarget_egohand_proposals(raw, obs, native, intrinsics=k, scale_reference=raw)
    b, _ = retarget_egohand_proposals(smooth, obs, native, intrinsics=k, scale_reference=raw)
    assert np.array_equal(a["fixed_native_to_ego_palm_scalar"], b["fixed_native_to_ego_palm_scalar"])


def test_insufficient_scale_one_hand_does_not_discard_other_hand_or_timeline():
    obs, proposal, native, k = fixture()
    obs["confidence"][:, 0] = 0
    arrays, report = retarget_egohand_proposals(proposal, obs, native, intrinsics=k)
    assert report["fixed_scale_calibrated"] == [False, True]
    assert not arrays["confidence"][:, 0].any()
    assert arrays["confidence"][:, 1].any()
    assert np.array_equal(arrays["frame_indices"], np.arange(25))


def test_missing_joint_or_wrist_has_no_image_weight_despite_model_presence():
    obs, proposal, native, k = fixture()
    obs["confidence"][1, 0, 4] = 0
    obs["confidence"][2, 0, 0] = 0
    arrays, _ = retarget_egohand_proposals(proposal, obs, native, intrinsics=k)
    assert arrays["confidence"][1, 0, 4] == 0
    assert not arrays["confidence"][2, 0].any()
    assert not arrays["shared_visibility_gate_weights"][2, 0]
    assert np.isfinite(arrays["proposal_joints3d_camera"][2, 0]).all()


def test_no_target_clamping_or_geometry_rewrite_outside_crop():
    obs, proposal, native, k = fixture()
    proposal["proposal_joints3d_camera"][:, :, 4, 0] += 20
    arrays, _ = retarget_egohand_proposals(proposal, obs, native, intrinsics=k)
    assert not arrays["confidence"][:, :, 4].any()
    assert np.all(arrays["proposal_joints3d_camera"][:, :, 4, 0] > 10)


def test_negative_depth_or_changed_crops_rejected_or_unweighted():
    obs, proposal, native, k = fixture()
    native[0, 0, 0, 2] = -1
    with pytest.raises(ValueError): retarget_egohand_proposals(proposal, obs, native, intrinsics=k)
    native[0, 0, 0, 2] = 3
    proposal["crop_boxes"][0, 0] += 1
    with pytest.raises(ValueError): retarget_egohand_proposals(proposal, obs, native, intrinsics=k)
