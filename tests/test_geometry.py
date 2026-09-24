import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from v2hoi.geometry import Similarity, SurfaceSDF, matrix_to_pose7, pose7_to_matrix, umeyama


def test_pose7_roundtrip_and_invisible():
    rng = np.random.default_rng(0)
    q = Rotation.random(5, random_state=1).as_quat()  # xyzw
    pose7 = np.concatenate([rng.normal(size=(5, 3)), q[:, [3, 0, 1, 2]]], axis=1)
    pose7[2] = 0.0  # invisible frame: zero quaternion
    T = pose7_to_matrix(pose7)
    assert np.isnan(T[2]).all()
    assert np.allclose(T[0, :3, :3], Rotation.from_quat(q[0]).as_matrix())
    back = matrix_to_pose7(T)
    assert np.allclose(back[2], 0.0)
    ok = [0, 1, 3, 4]
    # q and -q are the same rotation
    same = np.minimum(np.abs(back[ok, 3:] - pose7[ok, 3:]).max(1), np.abs(back[ok, 3:] + pose7[ok, 3:]).max(1))
    assert np.allclose(back[ok, :3], pose7[ok, :3]) and same.max() < 1e-9


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(100, 3))
    R = Rotation.from_rotvec([0.2, -0.5, 0.9]).as_matrix()
    dst = 1.3 * src @ R.T + [0.5, -1.0, 2.0]
    s, R_hat, t = umeyama(src, dst, with_scale=True)
    assert np.isclose(s, 1.3) and np.allclose(R_hat, R) and np.allclose(t, [0.5, -1.0, 2.0])
    sim = Similarity.fit(src, dst, with_scale=True)
    assert np.allclose(sim.points(src), dst)


def test_surface_sdf_box():
    sdf = SurfaceSDF(trimesh.creation.box(extents=[1.0, 1.0, 1.0]), count=20_000)
    d = sdf(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.45], [0.0, 0.0, 0.49], [3.0, 0.0, 0.0]]))
    assert abs(d[0] + 0.5) < 0.02
    assert abs(d[1] + 0.05) < 0.02
    assert d[2] < 0
    assert np.isinf(d[3])
