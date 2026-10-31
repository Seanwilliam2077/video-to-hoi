"""Independent synthetic geometry and CLI fixtures; no challenge/model assets."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from v2hoi.egohand_adapter import (
    SOMA25_TO_LANDMARKS21, adapt_egohand_proposals, parse_prediction_frame_names,
    project_camera_points, validate_frame_indices,
)
from v2hoi.hand_refinement import validate_observations


def prediction(ids=(0, 2)):
    joints = np.zeros((len(ids), 25, 3))
    joints[:, :, 0] = np.arange(25) * .001
    return {"frame_names": np.array([f"{i:06d}" for i in ids]), "num_frames": np.array(len(ids)),
            "joints": joints, "cam_t": np.tile([0., 0., 2.], (len(ids), 1))}


def inputs(count=4):
    return {"frame_indices": np.arange(count), "intrinsics": np.array([[100., 0, 100], [0, 100, 100], [0, 0, 1]]),
            "image_size": np.array([200, 200]),
            "crop_boxes": np.tile([0, 0, 200, 200], (count, 2, 1)), "variant": "raw"}


def test_verified_anatomical_mapping_skips_extra_metacarpals_and_preserves_sparse_frames():
    p = prediction((2, 0))
    result, report = adapt_egohand_proposals({"left": p}, **inputs())
    expected = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    assert SOMA25_TO_LANDMARKS21.tolist() == expected
    np.testing.assert_allclose(result["joints2d"][0, 0, :, 0], 100 + np.array(expected) * .05)
    np.testing.assert_allclose(result["proposal_joints3d_camera"][2, 0, :, 2], 2)
    assert result["confidence"][[1, 3]].sum() == 0
    assert result["confidence"][:, 1].sum() == 0
    assert result["prediction_row_present"][:, 0].tolist() == [True, False, True, False]
    assert report["hands"]["left"]["missing_prediction_frames"] == [1, 3]
    assert report["hands"]["right"]["missing_prediction_frames"] == [0, 1, 2, 3]
    assert result["independent_image_observations"].item() is False
    assert "not visibility" in report["confidence_policy"]
    validate_observations(result, np.arange(4))


def test_empty_upstream_hand_has_no_fabricated_geometry_or_predictions():
    empty = {"frame_names": np.array([], dtype="U1"), "num_frames": np.array(0)}
    result, report = adapt_egohand_proposals({"right": empty}, **inputs())
    assert not result["prediction_row_present"].any()
    assert not result["confidence"].any()
    assert report["hands"]["right"]["saved_prediction_rows"] == 0


@pytest.mark.parametrize("names", [["000000", "0"], ["000004"], ["1.jpg"], ["-1"], [" 1"], ["١"]])
def test_ambiguous_duplicate_or_out_of_video_frame_ids_fail(names):
    with pytest.raises(ValueError):
        parse_prediction_frame_names(np.array(names), 4)


def test_integer_prediction_rows_are_not_assumed_to_be_source_frame_ids():
    with pytest.raises(ValueError, match="string vector"):
        parse_prediction_frame_names(np.array([0, 1]), 4)


@pytest.mark.parametrize("indices", [np.array([1, 2]), np.array([0, 2]), np.array([0., 1.]), np.array([], dtype=int)])
def test_reindexed_partial_timelines_are_rejected(indices):
    with pytest.raises(ValueError):
        validate_frame_indices(indices)


def test_translation_units_projection_and_reflection_round_trip():
    K = np.array([[120., 0, 70], [0, 80, 90], [0, 0, 1]])
    xyz = np.array([[.3, -.1, 2.], [-.2, .4, 1.]])
    pixel, valid = project_camera_points(xyz, K)
    np.testing.assert_allclose(pixel, [[88, 86], [46, 122]])
    assert valid.all()
    shifted, _ = project_camera_points(xyz + [0, 0, 1], K)
    assert shifted[0, 0] == pytest.approx(82)
    mirror_K = K.copy()
    mirror_K[0, 2] = 199 - K[0, 2]
    mirror_xyz = xyz * [-1, 1, 1]
    mirrored, _ = project_camera_points(mirror_xyz, mirror_K)
    np.testing.assert_allclose(mirrored[:, 0], 199 - pixel[:, 0])
    np.testing.assert_allclose(mirrored[:, 1], pixel[:, 1])
    left, right = prediction((0,)), prediction((0,))
    left["cam_t"][0, 0] = -.1
    right["cam_t"][0, 0] = .1
    result, _ = adapt_egohand_proposals({"left": left, "right": right}, **inputs())
    assert result["joints2d"][0, 0, 0, 0] == 95
    assert result["joints2d"][0, 1, 0, 0] == 105


def test_invalid_depth_image_and_fixed_crop_bounds_have_zero_weights():
    p = prediction((0,))
    p["joints"][0, 1] = [0, 0, -3]  # negative final z
    p["joints"][0, 2] = [np.nan, 0, 0]
    p["joints"][0, 3] = [4, 0, 0]  # image-exterior
    p["joints"][0, 4] = [.3, 0, 0]  # image-interior, crop-exterior
    args = inputs()
    args["crop_boxes"][0, 0] = [90, 90, 110, 110]
    result, report = adapt_egohand_proposals({"left": p}, **args)
    np.testing.assert_array_equal(result["confidence"][0, 0, 1:5], 0)
    assert np.isfinite(result["joints2d"]).all()
    assert report["hands"]["left"]["nonfinite_or_nonpositive_depth_landmarks"] == 2
    assert report["hands"]["left"]["outside_image_landmarks"] == 1
    assert report["hands"]["left"]["outside_crop_landmarks"] == 1


def test_presence_is_not_visibility_and_optional_sigma_is_not_a_calibrated_weight():
    p = prediction((0,))
    p["pred_kp2d_crop_presence"] = np.zeros((1, 25))
    p["pred_kp2d_sigma"] = np.full((1, 25, 2), 5.)
    before = p["joints"].copy()
    args = inputs()
    boxes = args["crop_boxes"].copy()
    result, report = adapt_egohand_proposals({"right": p}, **args)
    assert result["confidence"][0, 1].sum() == 21
    assert report["hands"]["right"]["presence_available"]
    assert result["pred_kp2d_sigma_soma25"][0, 1, 0, 0] == 5
    np.testing.assert_array_equal(p["joints"], before)
    result["crop_boxes"][:] = 1
    np.testing.assert_array_equal(args["crop_boxes"], boxes)


@pytest.mark.parametrize("change", [
    {"num_frames": np.array(3)}, {"joints": np.zeros((2, 21, 3))},
    {"cam_t": np.zeros((2, 1, 3))}, {"pred_kp2d_crop_presence": np.full((2, 25), 1.1)},
    {"pred_kp2d_sigma": np.full((2, 25, 2), -1.)},
])
def test_corrupt_or_wrong_joint_schema_fails(change):
    p = prediction()
    p.update(change)
    with pytest.raises(ValueError):
        adapt_egohand_proposals({"left": p}, **inputs())


def test_side_and_variant_are_explicit_and_camera_is_never_guessed():
    with pytest.raises(ValueError, match="anatomical"):
        adapt_egohand_proposals({"0": prediction()}, **inputs())
    args = inputs()
    args["variant"] = "unknown"
    with pytest.raises(ValueError, match="variant"):
        adapt_egohand_proposals({}, **args)
    args = inputs()
    args["intrinsics"][0, 0] = 0
    with pytest.raises(ValueError, match="focal"):
        adapt_egohand_proposals({}, **args)


def test_cli_preserves_inputs_records_generated_targets_and_refuses_overwrite(tmp_path):
    args = inputs()
    sha = "a" * 64
    frozen = tmp_path / "frozen.npz"
    np.savez(frozen, frame_indices=args["frame_indices"], side_order=np.array(["left", "right"]),
             crop_boxes=args["crop_boxes"], image_size=args["image_size"],
             fps=np.array(30.), source_video_sha256=np.array(sha))
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    np.savez(left, **prediction())
    np.savez(right, **prediction((1, 3)))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"track": 1, "episode": 16, "video_sha256": sha,
        "video_frames": 4, "video_width": 200, "video_height": 200, "video_fps": 30.,
        "camera_intrinsics": args["intrinsics"].tolist()}))
    receipt = tmp_path / "prediction-receipt.json"
    receipt.write_text(json.dumps({"source_video_sha256": sha, "frames": 4}))
    intrinsics = tmp_path / "intrinsics.npy"
    np.save(intrinsics, args["intrinsics"])
    out = tmp_path / "adapted"
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "tools/adapt_egohand_observations.py"),
        "--left", str(left), "--right", str(right), "--frozen-observations", str(frozen),
        "--manifest", str(manifest), "--intrinsics", str(intrinsics),
        "--prediction-receipt", str(receipt), "--upstream-commit", "b" * 40,
        "--variant", "raw", "--output", str(out)]
    original = frozen.read_bytes()
    result = subprocess.run(command, cwd=root, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / "report.json").read_text())
    assert report["source_video_sha256"] == sha and report["frames"] == 4
    assert report["quality_accepted"] is False and report["official_scores_computed"] is False
    assert report["all30_execution_authorized"] is False
    assert "not independent image detections" in report["targets"]
    with np.load(out / "observations.npz", allow_pickle=False) as z:
        validate_observations({k: z[k] for k in z.files}, np.arange(4))
        np.testing.assert_array_equal(z["crop_boxes"], args["crop_boxes"])
    assert frozen.read_bytes() == original
    repeat = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert repeat.returncode != 0 and "must not exist" in repeat.stderr
