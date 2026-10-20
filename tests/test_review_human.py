"""Independent synthetic projection/frame fixtures; no model, video, or challenge data."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location("review_human", Path(__file__).parents[1] / "tools/review_human.py")
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def test_projection_uses_positive_opencv_depth_and_full_intrinsics():
    camera = np.array([[100, 5, 320], [0, 120, 240], [0, 0, 1]], dtype=float)
    points = np.array([[1, 2, 4], [0, 0, -1], [0, 0, 0], [np.nan, 0, 2]])
    pixels, valid = review.project(points, camera)
    np.testing.assert_allclose(pixels[0], [347.5, 300])
    np.testing.assert_array_equal(valid, [True, False, False, False])
    assert np.isnan(pixels[1:]).all()


def test_explicit_indices_preserve_noncontiguous_source_mapping(tmp_path):
    path = tmp_path / "predictions.npz"
    np.savez(path, frame_indices=np.array([4, 1]), pred_keypoints_2d=np.zeros((2, 3, 2)))
    predictions = review.Predictions(path)
    try:
        assert predictions.lookup == {4: 0, 1: 1}
        assert 0 not in predictions.lookup
        assert predictions.at("joints_2d", 0).shape == (3, 2)
    finally:
        predictions.close()


@pytest.mark.parametrize("indices", [np.array([0, 0]), np.array([-1, 2]), np.array([0.0, 1.0])])
def test_invalid_source_indices_rejected(tmp_path, indices):
    path = tmp_path / "predictions.npz"
    np.savez(path, frame_indices=indices, joints_2d=np.zeros((2, 3, 2)))
    with pytest.raises(ValueError, match="frame_indices"):
        review.Predictions(path)


def test_3d_requires_explicit_camera_and_rejects_invalid_faces(tmp_path):
    path = tmp_path / "predictions.npz"
    data = dict(frame_indices=np.arange(2), vertices_m=np.zeros((2, 3, 3)))
    np.savez(path, **data)
    with pytest.raises(ValueError, match="camera_intrinsics"):
        review.Predictions(path)
    np.savez(path, **data, K=np.eye(3), faces=np.array([[0, 1, 3]]))
    with pytest.raises(ValueError, match="invalid geometry"):
        review.Predictions(path)


def test_npy_directory_supports_memory_mapped_geometry(tmp_path):
    np.save(tmp_path / "frame_indices.npy", np.arange(3))
    np.save(tmp_path / "joints_3d.npy", np.ones((3, 2, 3)))
    np.save(tmp_path / "camera_intrinsics.npy", np.eye(3))
    predictions = review.Predictions(tmp_path)
    assert isinstance(predictions.arrays["joints_3d"], np.memmap)
    assert predictions.at("camera_intrinsics", 1).shape == (3, 3)


def test_diagnostic_vector_mean_excludes_nonfinite_samples():
    assert review.average_length(np.array([[3, 4], [np.nan, 2], [0, 0]])) == 2.5
    assert review.average_length(np.array([[np.nan, 0]])) is None


def test_contact_sheet_includes_actual_sparse_predictions_and_excludes_outside_video():
    np.testing.assert_array_equal(review.contact_sheet_indices(360, [0, 120, 240, 500]), [0, 120, 240])
    assert len(review.contact_sheet_indices(360, range(360))) == 9
    np.testing.assert_array_equal(review.contact_sheet_indices(3, []), [0, 1, 2])
