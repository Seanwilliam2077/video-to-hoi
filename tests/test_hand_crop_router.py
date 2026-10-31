"""Independent synthetic router tests; no videos, models or challenge assets."""
import numpy as np
import pytest

from v2hoi.hand_crop_router import CropRouterPolicy, route_hand_crops


def evidence(count=26):
    points = np.zeros((count, 2, 21, 2), dtype=float)
    for side in range(2):
        points[:, side, :, 0] = 200+side*100+np.arange(21)%5
        points[:, side, :, 1] = 200+np.arange(21)//5
    old = {"frame_indices": np.arange(count), "side_order": np.array(["left", "right"]),
           "image_size": np.array([640, 480]), "joints2d": points,
           "confidence": np.ones((count, 2, 21)),
           "crop_boxes": np.tile([[160,160,272,272], [260,160,372,272]], (count,1,1)).astype(np.int32)}
    trial = {key: value.copy() for key, value in old.items()}
    trial["crop_boxes"] = np.tile([[184,184,248,248], [284,184,348,248]], (count,1,1)).astype(np.int32)
    return old, trial


def test_three_training_successes_and_heldout_neighbor_routing():
    old, trial = evidence(16)
    boxes, arrays, report = route_hand_crops(old, trial)
    assert not arrays["route_tight"][:3].any()
    assert arrays["route_tight"][3:15].all()
    assert not arrays["route_tight"][15].any()  # No later training neighbor.
    assert arrays["route_tight"][5].all()
    assert not arrays["training_anchor_weights"][::5].any()
    assert np.array_equal(boxes[3], trial["crop_boxes"][3])
    assert np.array_equal(boxes[0], old["crop_boxes"][0])
    assert report["simulated_training_finger_count_nonregression"]


def test_heldout_detector_mutation_cannot_change_any_route_count_or_duplicate_check():
    old, trial = evidence()
    a, first, _ = route_hand_crops(old, trial)
    for obs in (old, trial):
        obs["joints2d"][::5] = 100000
        obs["confidence"][::5] = 0
    b, second, _ = route_hand_crops(old, trial)
    assert np.array_equal(a, b)
    for key in first:
        if first[key].dtype.kind in "fc":
            assert np.array_equal(first[key], second[key], equal_nan=True)
        else:
            assert np.array_equal(first[key], second[key])


@pytest.mark.parametrize("failure", ["missing", "missing_wrist", "finger_regression"])
def test_training_failure_immediately_widens_then_requires_three_successes(failure):
    old, trial = evidence()
    if failure == "missing": trial["confidence"][8, 0] = 0
    elif failure == "missing_wrist": trial["confidence"][8, 0, 0] = 0
    else: trial["confidence"][8, 0, 4] = 0  # Still reliable, but19<20.
    boxes, arrays, _ = route_hand_crops(old, trial)
    assert arrays["route_tight"][7, 0]
    assert not arrays["route_tight"][8:12, 0].any()
    assert arrays["route_tight"][12, 0]
    assert np.array_equal(boxes[8, 0], old["crop_boxes"][8, 0])
    assert arrays["consecutive_training_successes"][8, 0] == 0
    assert arrays["route_tight"][8, 1]  # Independent non-conflicting hand.


def test_heldout_slot_does_not_reset_training_success_counter():
    old, trial = evidence()
    trial["confidence"][:4] = 0
    _, arrays, _ = route_hand_crops(old, trial)
    assert arrays["consecutive_training_successes"][4, 0] == 1
    assert arrays["consecutive_training_successes"][6, 0] == 2
    assert arrays["route_tight"][7, 0]
    assert not arrays["route_tight"][5, 0]


def test_both_unknown_fall_back_without_new_evidence():
    old, trial = evidence()
    old["confidence"][:] = 0
    trial["confidence"][:] = 0
    boxes, arrays, _ = route_hand_crops(old, trial)
    assert np.array_equal(boxes, old["crop_boxes"])
    assert not arrays["training_anchor_weights"].any()
    assert not arrays["route_tight"].any()


def test_exact_box_selection_never_blends_crops_or_modifies_observations():
    old, trial = evidence()
    before_old = old["joints2d"].copy()
    before_trial = trial["joints2d"].copy()
    trial["confidence"][8, 0] = 0
    boxes, arrays, _ = route_hand_crops(old, trial)
    assert np.array_equal(boxes, np.where(arrays["route_tight"][...,None], trial["crop_boxes"], old["crop_boxes"]))
    assert np.array_equal(old["joints2d"], before_old)
    assert np.array_equal(trial["joints2d"], before_trial)


def test_new_mixed_pair_duplicate_widens_both_even_zero_weight_opposite_hand():
    old, trial = evidence()
    # Each original pair is100px apart, but tight-left + wide-right coincide.
    trial["joints2d"][:, :, :, 0] += 100
    old["confidence"][:, 1] = 0
    trial["confidence"][:, 1] = 0
    boxes, arrays, report = route_hand_crops(old, trial)
    assert arrays["new_mixed_pair_duplicate"][3]
    assert not arrays["route_tight"][3].any()
    assert not arrays["consecutive_training_successes"][3].any()
    assert np.array_equal(boxes[3], old["crop_boxes"][3])
    assert 3 in report["new_mixed_pair_duplicate_original_frame_ids"]
    assert not arrays["new_mixed_pair_duplicate"][::5].any()


def test_duplicate_distance_uses_strict_original_twelve_pixel_boundary():
    old, trial = evidence()
    trial["joints2d"][:, 0, :, 0] += 112
    trial["confidence"][:, 1] = 0
    _, arrays, _ = route_hand_crops(old, trial)
    assert arrays["selected_raw_wrist_distance_pixels"][3] == 12
    assert arrays["route_tight"][3, 0]
    assert not arrays["new_mixed_pair_duplicate"][3]


def test_raw_finite_precondition_matches_original_observer():
    old, trial = evidence()
    trial["joints2d"][:, 0, :, 0] += 100
    trial["confidence"][:, 1] = 0
    old["confidence"][:, 1] = 0
    old["joints2d"][:, 1, 20] = np.nan
    _, arrays, _ = route_hand_crops(old, trial)
    assert arrays["route_tight"][3, 0]
    assert not arrays["new_mixed_pair_duplicate"][3]


def test_mismatched_original_frame_side_and_crop_contracts_rejected():
    old, trial = evidence()
    trial["frame_indices"][2] = 1
    with pytest.raises(ValueError): route_hand_crops(old, trial)
    old, trial = evidence()
    trial["side_order"] = np.array(["right", "left"])
    with pytest.raises(ValueError): route_hand_crops(old, trial)
    old, trial = evidence()
    trial["crop_boxes"][0, 0, 2] += 1
    with pytest.raises(ValueError): route_hand_crops(old, trial)


@pytest.mark.parametrize("kwargs", [{"consecutive_training_successes":0}, {"holdout_stride":1},
                                    {"mixed_pair_duplicate_distance_pixels":float("nan")}])
def test_invalid_policy_rejected(kwargs):
    with pytest.raises(ValueError): CropRouterPolicy(**kwargs)
