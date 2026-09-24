"""Scorer end to end: synthetic episodes with a stand-in body model, plus Tier 1 if downloaded."""
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from v2hoi.dataset import TIER1_ROOT, Episode, load_episode
from v2hoi.score import Config, ObjectModels, score_episode


class FakeBody:
    """Rigid point clouds that follow the root translation; enough to exercise the scorer."""

    body_ids = np.arange(4)
    finger_ids = np.arange(4, 6)

    def __init__(self):
        rng = np.random.default_rng(0)
        self.joint_offsets = rng.normal(scale=0.3, size=(6, 3)).astype(np.float32)
        self.vert_offsets = rng.normal(scale=0.3, size=(300, 3)).astype(np.float32)

    def __call__(self, ep: Episode):
        # the world rotation of the fake human lives in pose[:, 0]
        R = Rotation.from_rotvec(ep.pose[:, 0]).as_matrix().astype(np.float32)
        joints = np.einsum("tij,nj->tni", R, self.joint_offsets) + ep.transl[:, None]
        verts = np.einsum("tij,nj->tni", R, self.vert_offsets) + ep.transl[:, None]
        return joints, verts


def make_episode(mesh_path, T=40, world=np.eye(4)) -> Episode:
    t = np.arange(T, dtype=np.float32)
    transl = np.stack([0.01 * t, np.zeros(T), 3.0 + 0.002 * t**1.2], 1).astype(np.float32)
    obj_T = np.tile(np.eye(4), (T, 1, 1))
    obj_T[:, :3, :3] = Rotation.from_rotvec(np.stack([0.03 * t, 0.01 * t, 0.0 * t], 1)).as_matrix()
    obj_T[:, :3, 3] = transl + [0.3, 0.0, 0.0]
    pose = np.zeros((T, 77, 3), np.float32)
    # move the whole episode into another world frame
    obj_T = world @ obj_T
    pose[:, 0] = Rotation.from_matrix(world[:3, :3]).as_rotvec()
    transl = transl @ world[:3, :3].T.astype(np.float32) + world[:3, 3].astype(np.float32)
    zeros = np.zeros((T, 1), np.float32)
    return Episode(
        index=0, sequence_id="synthetic", object_name="box", mesh_path=mesh_path, ground_plane=None,
        pose=pose, transl=transl, identity=zeros, scale=zeros, bone_flex=zeros,
        obj_T=obj_T, obj_visible=np.ones(T, dtype=bool),
    )


@pytest.fixture
def box_path(tmp_path):
    path = tmp_path / "box.glb"
    trimesh.creation.box(extents=[0.2, 0.1, 0.3]).export(path)
    return path


def test_rigidly_moved_prediction_scores_zero_after_se3(box_path):
    world = np.eye(4)
    world[:3, :3] = Rotation.from_rotvec([0.1, 0.4, -0.2]).as_matrix()
    world[:3, 3] = [0.5, -0.3, 0.8]
    gt = make_episode(box_path)
    pred = make_episode(box_path, world=world)
    body, objects = FakeBody(), ObjectModels(3000)

    res = score_episode(gt, pred, body, objects, Config(align="se3", object_samples=3000))
    assert np.isclose(res["align"]["rot_deg"], np.degrees(0.4583), atol=0.05)
    assert res["human"]["chamfer_mm"] < 1e-3
    assert res["human"]["accel_err_mm_f2"] < 1e-3
    assert res["object_metrics"]["chamfer_mm"] < 1e-3
    assert res["object_metrics"]["ang_accel_err_deg_f2"] < 1e-6
    assert res["object_metrics"]["shape_chamfer_mm"] < 1e-3

    raw = score_episode(gt, pred, body, objects, Config(align="none", object_samples=3000))
    assert raw["human"]["chamfer_mm"] > 100
    assert raw["object_metrics"]["chamfer_mm"] > 100


def test_hidden_frames_reduce_coverage(box_path):
    gt = make_episode(box_path)
    pred = make_episode(box_path)
    pred.obj_visible[10:20] = False
    pred.obj_T[10:20] = np.nan
    res = score_episode(gt, pred, FakeBody(), ObjectModels(3000), Config(object_samples=3000))
    assert np.isclose(res["object_metrics"]["coverage"], 0.75)
    assert res["object_metrics"]["chamfer_mm"] < 1e-3


@pytest.mark.skipif(not (TIER1_ROOT / "meta" / "info.json").is_file(), reason="Tier 1 not downloaded")
def test_tier1_ground_truth_against_itself():
    pytest.importorskip("soma")
    from v2hoi.body import SomaBody

    gt = load_episode(TIER1_ROOT, 7)
    res = score_episode(gt, load_episode(TIER1_ROOT, 7), SomaBody(), ObjectModels(5000),
                        Config(stride=50, object_samples=5000))
    assert res["human"]["chamfer_mm"] < 0.01
    assert res["human"]["accel_err_mm_f2"] < 0.01
    assert res["object_metrics"]["chamfer_mm"] < 0.01
    assert res["contact"]["penetration_err_mm"] < 0.01
    assert abs(res["align"]["sim3_scale"] - 1.0) < 1e-4
