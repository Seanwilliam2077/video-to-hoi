import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from v2hoi import metrics as M
from v2hoi.geometry import SurfaceSDF


def test_chamfer():
    a = np.zeros((1, 3))
    assert M.chamfer(a, a) == 0.0
    assert np.isclose(M.chamfer(a, np.array([[0.0, 0.0, 1.0]])), 1.0)


def test_icp_residual_removes_rigid_offset():
    pts = np.random.default_rng(0).uniform(-0.1, 0.1, size=(2000, 3)) * [1.0, 2.0, 3.0]
    R = Rotation.from_rotvec([0.05, 0.02, -0.04]).as_matrix()
    moved = pts @ R.T + [0.01, -0.02, 0.005]
    assert M.chamfer(moved, pts) > 1e-3
    assert M.icp_residual(moved, pts) < 1e-4


def test_accel_error():
    T = 20
    gt = np.random.default_rng(0).normal(size=(T, 4, 3))
    assert M.accel_error(gt, gt) == 0.0
    # a constant velocity offset has zero acceleration
    drift = gt + np.arange(T)[:, None, None] * np.array([0.01, 0.0, 0.0])
    assert np.isclose(M.accel_error(drift, gt), 0.0)
    # one spike of size e at frame 5 shows up at frames 4, 5, 6 as e, 2e, e
    spike = gt.copy()
    spike[5] += [0.0, 0.0, 1.0]
    assert np.isclose(M.accel_error(spike, gt), 4.0 / (T - 2))
    # triplets touching an invalid frame are skipped
    valid = np.ones(T, dtype=bool)
    valid[5] = False
    assert np.isclose(M.accel_error(spike, gt, valid), 0.0)


def test_accel_magnitude():
    T = 20
    line = np.arange(T)[:, None, None] * np.array([0.01, 0.02, 0.0]) * np.ones((1, 3, 1))
    assert np.isclose(M.accel_magnitude(line), 0.0)
    spike = line.copy()
    spike[5] += [0.0, 0.0, 1.0]
    assert np.isclose(M.accel_magnitude(spike), 4.0 / (T - 2))
    valid = np.ones(T, dtype=bool)
    valid[5] = False
    assert np.isclose(M.accel_magnitude(spike, valid), 0.0)


def test_angular_accel_magnitude():
    T = 30
    t = np.arange(T)
    steady = Rotation.from_rotvec(np.stack([0.05 * t, np.zeros(T), np.zeros(T)], 1)).as_matrix()
    valid = np.ones(T, dtype=bool)
    assert M.angular_accel_magnitude(steady, valid) < 1e-9
    C = Rotation.from_rotvec([0.3, -1.2, 0.7]).as_matrix()
    speeding = Rotation.from_rotvec(np.stack([0.01 * t**2, np.zeros(T), np.zeros(T)], 1)).as_matrix()
    assert np.isclose(M.angular_accel_magnitude(speeding, valid), 0.02)
    assert np.isclose(M.angular_accel_magnitude(speeding @ C, valid), 0.02)


def test_angular_accel_ignores_canonical_offset():
    T = 30
    t = np.arange(T)
    R_gt = Rotation.from_rotvec(np.stack([0.02 * t**1.5, 0.01 * t, np.zeros(T)], 1)).as_matrix()
    C = Rotation.from_rotvec([0.3, -1.2, 0.7]).as_matrix()
    valid = np.ones(T, dtype=bool)
    assert M.angular_accel_error(R_gt @ C, R_gt, valid) < 1e-9
    R_noisy = R_gt @ Rotation.from_rotvec(np.random.default_rng(0).normal(scale=0.02, size=(T, 3))).as_matrix()
    assert M.angular_accel_error(R_noisy, R_gt, valid) > 1e-3


def test_penetration_and_ground():
    sdf = SurfaceSDF(trimesh.creation.box(extents=[1.0, 1.0, 1.0]), count=20_000)
    assert abs(M.penetration_depth(np.array([[0.0, 0.0, 0.3], [2.0, 0.0, 0.0]]), sdf) - 0.2) < 0.02
    assert M.penetration_depth(np.array([[2.0, 0.0, 0.0]]), sdf) == 0.0

    # OpenCV world: ground at y = 1.5, people above it have y < 1.5
    people = np.array([[0.0, 0.5, 3.0], [0.0, 1.0, 3.0]])
    plane = M.orient_plane(np.array([0.0, 1.0, 0.0, -1.5]), people)
    assert np.allclose(plane, [0.0, -1.0, 0.0, 1.5])
    assert M.plane_penetration(people, plane) == 0.0
    assert np.isclose(M.plane_penetration(np.array([[0.0, 1.52, 3.0]]), plane), 0.02)
