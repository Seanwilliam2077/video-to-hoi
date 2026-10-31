"""Independent synthetic geometry only; no challenge assets or references."""
import numpy as np
import pytest

from v2hoi.object_depth_gauge import apply_projective_gauge, far_side_rigid_shift, fit_projective_gauge, forward_ray_interval, projection


def cube_planes():
    normals = np.r_[np.eye(3), -np.eye(3)]
    offsets = np.asarray([1, 1, 4, 1, 1, -2], float)
    return normals, offsets


def test_projective_gauge_preserves_all_projected_points():
    T = np.repeat(np.eye(4)[None], 3, axis=0)
    T[:, :3, 3] = [[0, 0, 3], [1, -.3, 4], [-.8, .4, 3.2]]
    v = np.asarray([[-.2, -.2, -.2], [.2, -.2, -.2], [-.2, .2, .2], [.2, .2, .2]])
    K = np.asarray([[800, 0, 400], [0, 800, 300], [0, 0, 1]])
    source = dict(vertices=v, object_to_camera=T, translation=T[:, :3, 3].copy(), rotation=T[:, :3, :3].copy(), camera_intrinsics=K, frame_indices=np.arange(3))
    scaled = apply_projective_gauge(source, 1.04)
    assert np.array_equal(source["vertices"], v)
    assert np.array_equal(source["object_to_camera"], T)
    assert np.array_equal(scaled["camera_intrinsics"], K)
    assert np.array_equal(scaled["rotation"], source["rotation"])
    assert np.array_equal(scaled["object_to_camera"][:, 3], T[:, 3])
    for i in range(3):
        np.testing.assert_allclose(projection(v, T[i], K), projection(scaled["vertices"], scaled["object_to_camera"][i], K), atol=1e-12)


@pytest.mark.parametrize("factor", [0, -1, float("nan"), True])
def test_invalid_gauge_rejected(factor):
    with pytest.raises(ValueError):
        apply_projective_gauge({}, factor)


def test_duplicate_pose_representation_must_agree():
    with pytest.raises(ValueError, match="translations disagree"):
        apply_projective_gauge(dict(vertices=np.zeros((4, 3)), object_to_camera=np.eye(4)[None], translation=np.ones((1, 3))), 1.01)


def test_robust_fit_recovers_known_gauge_without_prior_when_strongly_supported():
    q = np.asarray([[.1, .2, 3], [.2, -.2, 3.2], [-.2, .1, 2.8]])
    fit = fit_projective_gauge(q * 1.025, q, np.arange(3), prior_sigma=100)
    assert fit["factor"] == pytest.approx(1.025, abs=.0003)
    assert fit["data_only_factor"] == pytest.approx(1.025, abs=.0003)
    assert fit["frames"] == 3
    assert len(fit["curve"]) > 100
    assert len(fit["prior_sensitivity"]) == 3


def test_frame_balancing_prevents_dense_frame_from_dominating():
    q = np.tile([0, 0, 3.], (102, 1))
    p = q.copy()
    p[:100] *= 1.05
    p[100:] *= .97
    frames = np.r_[np.zeros(100, int), [1, 2]]
    fit = fit_projective_gauge(p, q, frames, prior_sigma=100)
    assert fit["factor"] < .99


def test_camera_ray_intersection_orders_front_and_back():
    normals, offsets = cube_planes()
    near, far, valid = forward_ray_interval(np.asarray([[0, 0, 3], [4, 0, 3], [0, .1, 1]]), normals, offsets)
    assert valid.tolist() == [True, False, True]
    np.testing.assert_allclose(near[[0, 2]], [2, 2])
    np.testing.assert_allclose(far[[0, 2]], [4, 4])


def test_rigid_far_side_shift_preserves_wrist_ray_but_reports_finger_change():
    normals, offsets = cube_planes()
    points = np.asarray([[0, 0, 3], [.2, 0, 3.5], [0, .2, 4.5]])
    report = far_side_rigid_shift(points, np.asarray([0, 0, 3]), normals, offsets, margin_m=.01)
    assert report["rigid_delta_z_m"] == pytest.approx(1.01)
    assert report["rigid_wrist_displacement_m"] == pytest.approx(1.01)
    assert report["already_far_side_count"] == 1
    assert report["normalized_projection_max_change"] > 0
    assert report["wrist_projection_change"] == 0


def test_empty_gauge_samples_abstain():
    with pytest.raises(ValueError, match="nonempty"):
        fit_projective_gauge(np.empty((0, 3)), np.empty((0, 3)), np.empty(0, int))
