"""Scorer end to end: synthetic episodes with a stand-in body model, plus Tier 1 if downloaded."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from v2hoi.dataset import TIER1_ROOT, Episode, load_episode
from v2hoi.geometry import matrix_to_pose7
from v2hoi.score import LEADERBOARD, Config, ObjectModels, aggregate, check_mesh, score, score_episode

BOX = np.array([0.2, 0.1, 0.3])


class FakeBody:
    """Rigid point clouds that follow the root; enough to exercise the scorer.

    The world rotation lives in pose[:, 0] and a size factor in identity[:, 0]
    (1 + value), so a prediction can be a similarity transform of the truth.
    """

    body_ids = np.arange(4)
    finger_ids = np.arange(4, 6)

    def __init__(self):
        rng = np.random.default_rng(0)
        self.joint_offsets = rng.normal(scale=0.3, size=(6, 3)).astype(np.float32)
        self.vert_offsets = rng.normal(scale=0.3, size=(300, 3)).astype(np.float32)

    def __call__(self, ep: Episode):
        R = Rotation.from_rotvec(ep.pose[:, 0]).as_matrix().astype(np.float32)
        size = (1.0 + ep.identity[:, :1])[:, :, None]
        joints = size * np.einsum("tij,nj->tni", R, self.joint_offsets) + ep.transl[:, None]
        verts = size * np.einsum("tij,nj->tni", R, self.vert_offsets) + ep.transl[:, None]
        return joints, verts


def make_episode(mesh_path, T=40, world=np.eye(4), scale=1.0) -> Episode:
    """A walking human next to a turning box; the whole scene is scaled, then moved by ``world``."""
    t = np.arange(T, dtype=np.float32)
    transl = scale * np.stack([0.01 * t, np.zeros(T), 3.0 + 0.002 * t**1.2], 1).astype(np.float32)
    obj_T = np.tile(np.eye(4), (T, 1, 1))
    obj_T[:, :3, :3] = Rotation.from_rotvec(np.stack([0.03 * t, 0.01 * t, 0.0 * t], 1)).as_matrix()
    obj_T[:, :3, 3] = transl + scale * np.array([0.3, 0.0, 0.0])
    obj_T = world @ obj_T
    pose = np.zeros((T, 77, 3), np.float32)
    pose[:, 0] = Rotation.from_matrix(world[:3, :3]).as_rotvec()
    transl = transl @ world[:3, :3].T.astype(np.float32) + world[:3, 3].astype(np.float32)
    identity = np.full((T, 1), scale - 1.0, np.float32)
    zeros = np.zeros((T, 1), np.float32)
    return Episode(
        index=0, sequence_id="synthetic", object_name="box", mesh_path=mesh_path, ground_plane=None,
        pose=pose, transl=transl.astype(np.float32), identity=identity, scale=zeros, bone_flex=zeros,
        obj_T=obj_T, obj_visible=np.ones(T, dtype=bool),
    )


def some_world() -> np.ndarray:
    world = np.eye(4)
    world[:3, :3] = Rotation.from_rotvec([0.1, 0.4, -0.2]).as_matrix()
    world[:3, 3] = [0.5, -0.3, 0.8]
    return world


def box_mesh(path, scale=1.0):
    trimesh.creation.box(extents=BOX * scale).export(path)
    return path


@pytest.fixture
def box_path(tmp_path):
    return box_mesh(tmp_path / "box.glb")


def test_rigidly_moved_prediction_scores_zero_after_se3(box_path):
    gt = make_episode(box_path)
    pred = make_episode(box_path, world=some_world())
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


def test_first_frame_sim3_undoes_a_similarity(tmp_path, box_path):
    gt = make_episode(box_path)
    pred = make_episode(box_mesh(tmp_path / "big.glb", 1.25), world=some_world(), scale=1.25)

    res = score_episode(gt, pred, FakeBody(), ObjectModels(3000), Config(object_samples=3000))
    assert res["align"]["mode"] == "first"
    assert np.isclose(res["align"]["scale"], 0.8)
    assert res["human"]["chamfer_mm"] < 1e-3
    assert res["object_metrics"]["chamfer_mm"] < 1e-3
    # depths are measured in the prediction's own units, then rescaled
    assert np.isclose(res["contact"]["penetration_pred_mm"], res["contact"]["penetration_gt_mm"], atol=1e-3)
    # float32 positions near 3 m: second differences agree to ~1e-4
    assert np.isclose(res["human"]["accel_mm_f2"], res["human"]["accel_ref_mm_f2"], rtol=1e-3)


def test_first_frame_alignment_keeps_later_drift(box_path):
    gt = make_episode(box_path)
    pred = make_episode(box_path)
    drift = np.linspace(0.0, 0.1, len(pred))[:, None] * np.array([1.0, 0.0, 0.0])
    pred.transl = (pred.transl + drift).astype(np.float32)

    first = score_episode(gt, pred, FakeBody(), ObjectModels(3000), Config(object_samples=3000))
    whole = score_episode(gt, pred, FakeBody(), ObjectModels(3000), Config(align="se3", object_samples=3000))
    assert first["align"]["trans_m"] < 1e-6
    assert np.isclose(first["human"]["mpjpe_mm"], 50.0, atol=0.5)
    # fitting the whole clip hides half of the drift
    assert whole["human"]["mpjpe_mm"] < 0.6 * first["human"]["mpjpe_mm"]


def test_smoothness_is_measured_on_the_prediction_alone(box_path):
    gt = make_episode(box_path)
    pred = make_episode(box_path)
    body, objects = FakeBody(), ObjectModels(3000)
    same = score_episode(gt, pred, body, objects, Config(object_samples=3000))
    assert np.isclose(same["human"]["accel_mm_f2"], same["human"]["accel_ref_mm_f2"])
    assert np.isclose(same["object_metrics"]["accel_mm_f2"], same["object_metrics"]["accel_ref_mm_f2"])

    jitter = np.random.default_rng(0).normal(scale=0.005, size=(len(pred), 3))
    pred.transl = (pred.transl + jitter).astype(np.float32)
    pred.obj_T[:, :3, 3] += jitter
    noisy = score_episode(gt, pred, body, objects, Config(object_samples=3000))
    assert noisy["human"]["accel_mm_f2"] > 2 * same["human"]["accel_mm_f2"]
    assert noisy["object_metrics"]["accel_mm_f2"] > 2 * same["object_metrics"]["accel_mm_f2"]
    assert noisy["human"]["accel_ref_mm_f2"] == same["human"]["accel_ref_mm_f2"]


def test_missing_object_poses_are_a_violation(box_path):
    gt = make_episode(box_path)
    pred = make_episode(box_path)
    pred.obj_T[10:20] = np.nan
    res = score_episode(gt, pred, FakeBody(), ObjectModels(3000), Config(object_samples=3000))
    assert np.isclose(res["object_metrics"]["coverage"], 0.75)
    assert res["object_metrics"]["chamfer_mm"] < 1e-3
    assert np.isfinite(res["object_metrics"]["accel_mm_f2"])
    assert res["violations"] == [
        "object pose missing on 10 of 40 frames (first: [10, 11, 12, 13, 14]); occluded frames need a pose too"
    ]


def test_prediction_visibility_flag_is_ignored(box_path):
    gt = make_episode(box_path)
    pred = make_episode(box_path)
    pred.obj_visible[10:20] = False  # poses are still there
    res = score_episode(gt, pred, FakeBody(), ObjectModels(3000), Config(object_samples=3000))
    assert res["object_metrics"]["coverage"] == 1.0
    assert res["violations"] == []
    assert np.isclose(res["object_metrics"]["accel_mm_f2"], res["object_metrics"]["accel_ref_mm_f2"])


def test_unscorable_predictions_stop_the_run(box_path):
    gt = make_episode(box_path)
    pred = make_episode(box_path)
    pred.transl[3] = np.nan
    with pytest.raises(ValueError, match="human parameters not finite: translation"):
        score_episode(gt, pred, FakeBody(), ObjectModels(3000), Config(object_samples=3000))
    with pytest.raises(ValueError, match="prediction has 30 frames, ground truth has 40"):
        score_episode(gt, make_episode(box_path, T=30), FakeBody(), ObjectModels(3000), Config(object_samples=3000))


def test_leaderboard_is_in_cm(box_path):
    pred = make_episode(box_path)
    pred.transl = (pred.transl + np.random.default_rng(0).normal(scale=0.01, size=pred.transl.shape)).astype(np.float32)
    res = score_episode(make_episode(box_path), pred, FakeBody(), ObjectModels(3000), Config(object_samples=3000))
    assert set(res["leaderboard"]) == set(LEADERBOARD)
    for key, (section, local) in LEADERBOARD.items():
        assert np.isclose(res["leaderboard"][key], res[section][local] / 10, equal_nan=True)
    assert res["leaderboard"]["cd_h_cm"] > 0.1


def test_config_deviations():
    assert Config().deviations() == []
    assert len(Config(align="se3", stride=3).deviations()) == 2


def test_mesh_rules(tmp_path, box_path):
    objects = ObjectModels(3000)
    ref = tmp_path / "ref"
    (ref / "mesh" / "box").mkdir(parents=True)
    assert check_mesh("box", box_path, objects(box_path), ref) == []

    shutil.copy(box_path, ref / "mesh" / "box" / "box.glb")
    [problem] = check_mesh("box", box_path, objects(box_path), ref)
    assert "copy of reference asset box.glb" in problem

    mm = box_mesh(tmp_path / "mm.glb", 1000.0)
    [problem] = check_mesh("box", mm, objects(mm), tmp_path / "empty")
    assert "expected metres" in problem


def write_root(root, episodes: dict[int, Episode], mesh_path):
    """A minimal Tier 1 layout holding ``episodes``."""
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "mesh" / "box").mkdir(parents=True)
    shutil.copy(mesh_path, root / "mesh" / "box" / "box.glb")
    info = {"data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", "chunks_size": 1000}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    rows = [{"episode_index": e, "object": "box", "mesh": "mesh/box/box.glb"} for e in episodes]
    (root / "meta" / "episodes_metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for e, ep in episodes.items():
        pd.DataFrame({
            "observation.human.pose": list(ep.pose.reshape(len(ep), -1)),
            "observation.human.translation": list(ep.transl),
            "observation.human.identity_coeffs": list(ep.identity),
            "observation.human.scale_params": list(ep.scale),
            "observation.human.bone_length_flexibles": list(ep.bone_flex),
            "observation.object.pose": list(matrix_to_pose7(ep.obj_T).astype(np.float32)),
            "observation.object.visible": ep.obj_visible,
        }).to_parquet(root / "data" / "chunk-000" / f"episode_{e:06d}.parquet")


def test_score_applies_the_submission_rules(tmp_path, box_path, monkeypatch):
    monkeypatch.setattr("v2hoi.body.SomaBody", lambda device=None: FakeBody())
    gt_root, pred_root = tmp_path / "gt", tmp_path / "pred"
    write_root(gt_root, {0: make_episode(box_path), 1: make_episode(box_path)}, box_path)
    hidden = make_episode(box_path)
    hidden.obj_visible[5:9] = False  # flagged, but the poses are kept
    missing = make_episode(box_path)
    missing.obj_T[5:9] = np.nan  # written as zero quaternions
    write_root(pred_root, {0: hidden, 1: missing}, box_mesh(tmp_path / "own.glb", 1.02))

    cfg = Config(object_samples=3000)
    report = score(gt_root, pred_root, cfg=cfg, log=lambda *_: None)
    sub = report["submission"]
    assert sub["official_settings"] and sub["deviations"] == []
    assert not sub["valid"]
    assert sub["violations"] == [
        "episode 1: object pose missing on 4 of 40 frames (first: [5, 6, 7, 8]); occluded frames need a pose too"
    ]
    assert report["per_episode"][0]["object_metrics"]["coverage"] == 1.0
    assert set(report["mean"]["leaderboard"]) == set(LEADERBOARD)

    # the reference's own meshes are Track 2 assets
    report = score(gt_root, pred_root, [0], cfg, pred_mesh_dir=gt_root / "mesh", log=lambda *_: None)
    assert report["submission"]["deviations"] == ["scored 1 of 2 episodes"]
    [problem] = report["submission"]["violations"]
    assert "Track 2 assets are not allowed" in problem

    # every reference episode is required unless a subset is asked for
    (pred_root / "meta" / "episodes_metadata.jsonl").write_text(
        json.dumps({"episode_index": 0, "object": "box", "mesh": "mesh/box/box.glb"})
    )
    with pytest.raises(ValueError, match="pass --episodes"):
        score(gt_root, pred_root, cfg=cfg, log=lambda *_: None)


def test_missing_object_metrics_do_not_improve_aggregate(box_path):
    gt = make_episode(box_path, T=5)
    complete = score_episode(gt, gt, FakeBody(), ObjectModels(1000), Config(object_samples=1000))
    missing = make_episode(box_path, T=5)
    missing.obj_visible[:] = False
    missing.obj_T[:] = np.nan
    incomplete = score_episode(gt, missing, FakeBody(), ObjectModels(1000), Config(object_samples=1000))
    mean = aggregate({0: complete, 1: incomplete})
    assert incomplete["object_metrics"]["coverage"] == 0.0
    assert np.isnan(mean["object_metrics"]["chamfer_mm"])
    assert np.isnan(mean["contact"]["penetration_err_mm"])


def test_default_score_requires_every_gt_episode(monkeypatch, tmp_path):
    import v2hoi.score as scorer

    gt, pred = tmp_path / "gt", tmp_path / "pred"
    monkeypatch.setattr(scorer, "list_episodes", lambda root: [0, 1] if root == gt else [0])
    with pytest.raises(ValueError, match=r"prediction root has no episodes \[1\]"):
        score(gt, pred)


def test_missing_frames_are_scored_but_invalid(monkeypatch, tmp_path, box_path):
    import v2hoi.body as body_module
    import v2hoi.score as scorer

    gt_root, pred_root = tmp_path / "gt", tmp_path / "pred"
    gt = make_episode(box_path, T=5)
    pred = make_episode(box_path, T=5)
    pred.obj_T[0] = np.nan
    monkeypatch.setattr(scorer, "list_episodes", lambda root: [0])
    monkeypatch.setattr(scorer, "load_episode", lambda root, index, *_, **__: gt if root == gt_root else pred)
    monkeypatch.setattr(body_module, "SomaBody", lambda device=None: FakeBody())

    report = score(gt_root, pred_root, cfg=Config(object_samples=1000), log=lambda _: None)
    assert not report["submission"]["valid"]
    assert report["mean"]["object_metrics"]["coverage"] == 0.8


def test_sim3_penetration_uses_ground_truth_scale(tmp_path, box_path):
    scaled_path = tmp_path / "scaled_box.glb"
    trimesh.creation.box(extents=[0.4, 0.2, 0.6]).export(scaled_path)
    gt = make_episode(box_path, T=5)
    pred = make_episode(scaled_path, T=5)
    pred.transl *= 2
    pred.obj_T[:, :3, 3] *= 2

    class ScaledBody(FakeBody):
        def __call__(self, ep):
            joints, verts = super().__call__(ep)
            if ep is pred:
                joints += joints - ep.transl[:, None]
                verts += verts - ep.transl[:, None]
            return joints, verts

    body = ScaledBody()
    body.vert_offsets[0] = [0.3, 0.0, 0.0]  # a point inside the box
    res = score_episode(gt, pred, body, ObjectModels(3000), Config(align="sim3", object_samples=3000))
    assert np.isclose(res["align"]["sim3_scale"], 0.5, atol=1e-6)
    assert res["contact"]["penetration_gt_mm"] > 10
    assert res["contact"]["penetration_err_mm"] < 1


@pytest.mark.skipif(not (TIER1_ROOT / "meta" / "info.json").is_file(), reason="Tier 1 not downloaded")
def test_tier1_ground_truth_against_itself():
    pytest.importorskip("soma")
    from v2hoi.body import SomaBody

    gt = load_episode(TIER1_ROOT, 7)
    res = score_episode(gt, load_episode(TIER1_ROOT, 7), SomaBody(), ObjectModels(5000),
                        Config(stride=50, object_samples=5000))
    assert res["human"]["chamfer_mm"] < 0.01
    assert res["human"]["accel_err_mm_f2"] < 0.01
    assert np.isclose(res["human"]["accel_mm_f2"], res["human"]["accel_ref_mm_f2"], rtol=1e-3)
    assert res["object_metrics"]["chamfer_mm"] < 0.01
    assert res["contact"]["penetration_err_mm"] < 0.01
    assert abs(res["align"]["scale"] - 1.0) < 1e-4
