"""Artifacts round-trip, reject what their consumers cannot use, and resolve through upstream runs."""
import json

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from v2hoi.contracts import (
    CONTRACT_VERSION, Camera, ContractError, Human, Masks, Motion, ObjectAsset, RefinedHuman, Run,
)
from v2hoi.stages.human import lock_identity
from v2hoi.stages.inputs import merge_intrinsics
from v2hoi.stages.objects import merge_scale


def make_human(T=5) -> Human:
    rng = np.random.default_rng(0)
    return Human(**{
        name: rng.normal(size=(T, *shape)).astype(np.float32)
        for name, shape in [
            ("pose", (77, 3)), ("transl", (3,)), ("identity", (45,)), ("scale", (68,)), ("bone_flex", (6,)),
            ("mhr_global_rot", (3,)), ("mhr_body_pose", (133,)), ("mhr_hand_pose", (108,)),
            ("mhr_scale", (28,)), ("mhr_shape", (45,)), ("mhr_transl", (3,)),
        ]
    })


def make_motion(T=5) -> Motion:
    T_cam_obj = np.tile(np.eye(4), (T, 1, 1))
    T_cam_obj[:, :3, :3] = Rotation.from_rotvec(np.outer(np.arange(T), [0.0, 0.1, 0.0])).as_matrix()
    T_cam_obj[:, :3, 3] = [0.1, 0.2, 2.0]
    return Motion(T_cam_obj=T_cam_obj, confidence=np.ones(T))


def test_human_round_trip_and_validation(tmp_path):
    human = make_human()
    human.validate(5)
    human.save(tmp_path / "human.npz")
    back = Human.load(tmp_path / "human.npz")
    assert np.allclose(back.pose, human.pose) and np.allclose(back.mhr_body_pose, human.mhr_body_pose)

    with pytest.raises(ContractError, match=r"Human.pose has shape \(5, 77, 3\), expected \(6, 77, 3\)"):
        human.validate(6)
    human.transl[2] = np.nan
    with pytest.raises(ContractError, match="non-finite"):
        human.validate(5)


def test_contract_version_is_checked(tmp_path):
    path = tmp_path / "camera.json"
    Camera(64, 48, 50.0, 50.0, 32.0, 24.0).save(path)
    data = json.loads(path.read_text())
    data["contract"] = CONTRACT_VERSION + 1
    path.write_text(json.dumps(data))
    with pytest.raises(ContractError, match="contract version"):
        Camera.load(path)


def test_camera_validation():
    Camera(64, 48, 50.0, 50.0, 32.0, 24.0).validate()
    with pytest.raises(ContractError, match="outside the image"):
        Camera(64, 48, 50.0, 50.0, 100.0, 24.0).validate()


def test_masks_pack_and_unpack(tmp_path):
    rng = np.random.default_rng(0)
    human = rng.random((3, 5, 13)) > 0.5  # width not a multiple of 8
    obj = rng.random((3, 5, 13)) > 0.5
    masks = Masks.pack(human, obj)
    masks.validate(3, 5, 13)
    masks.save(tmp_path / "masks.npz")
    back = Masks.load(tmp_path / "masks.npz")
    assert np.array_equal(back.frame("human", 1), human[1])
    assert np.array_equal(back.frame("obj", 2), obj[2])
    with pytest.raises(ContractError):
        back.validate(3, 5, 16)


def test_motion_needs_proper_rotations():
    motion = make_motion()
    motion.validate(5)
    motion.T_cam_obj[2, :3, :3] *= 2.0
    with pytest.raises(ContractError, match="proper rotations"):
        motion.validate(5)


def test_object_mesh_must_be_metric(tmp_path):
    asset = ObjectAsset(name="box", scale=1.0)
    trimesh.creation.box(extents=[0.2, 0.1, 0.3]).export(tmp_path / "m.glb")
    asset.validate(tmp_path / "m.glb")
    trimesh.creation.box(extents=[200.0, 100.0, 300.0]).export(tmp_path / "mm.glb")
    with pytest.raises(ContractError, match="expected metres"):
        asset.validate(tmp_path / "mm.glb")


def test_run_reads_fall_back_to_upstream(tmp_path):
    base = Run.start(tmp_path / "base", "tier1", stages=["human"])
    base.save(make_human(), episode=7)
    child = Run.start(tmp_path / "child", "tier1", base, stages=["motion"])
    child.save(make_motion(), episode=7)

    assert child.find(Human, episode=7) == base.path(Human, episode=7)
    assert child.find(Motion, episode=7) == child.path(Motion, episode=7)
    assert not child.has(RefinedHuman, episode=7)
    with pytest.raises(ContractError, match="human/000009/human.npz not found"):
        child.load(Human, episode=9)

    reopened = Run.open(tmp_path / "child")
    assert reopened.upstream.root == base.root
    with pytest.raises(ContractError, match="is on tier1, not track1"):
        Run.start(tmp_path / "child", "track1")
    Run.start(tmp_path / "child", "tier1", stages=["export"])
    history = json.loads((tmp_path / "child" / "run.json").read_text())["history"]
    assert [h["stages"] for h in history] == [["motion"], ["export"]]


def test_merge_rules():
    cams = [Camera(64, 48, f, f, 32.0, 24.0, "front") for f in (50.0, 52.0, 60.0)]
    assert merge_intrinsics(cams).fx == 52.0
    with pytest.raises(ContractError):
        merge_intrinsics([Camera(64, 48, 50, 50, 32, 24), Camera(32, 48, 50, 50, 16, 24)])
    assert merge_scale([0.9, 1.1, 1.0]) == 1.0
    assert lock_identity(np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])).tolist() == [2.0, 3.0]
