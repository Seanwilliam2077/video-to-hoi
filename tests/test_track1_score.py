"""Independent analytic fixtures only; no challenge data or model assets."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from v2hoi import score
from v2hoi.score import DecodedHuman, NativeMHR, Provenance, Reconstruction, Roles, score_dataset, score_episode


VERTICES = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.2, 0., 0.], [3., 0., 0.]])
JOINTS = np.column_stack((np.arange(24) * .03, np.arange(24) % 3 * .05, np.arange(24) % 5 * .02))
ROLES = Roles([0, 1, 2, 3], list(range(6)), list(range(22)), [4, 5], "independent-analytic-fixture-roles")
PROVENANCE = Provenance("independent_synthetic", "Geometry and motion defined analytically in this test file")


def analytic_decoder(native):
    """Synthetic adapter tests the interface, not real MHR parameter semantics."""
    rotation = Rotation.from_rotvec(native.pose[:, 4:7]).as_matrix()
    vertices = np.einsum("tij,nj->tni", rotation, VERTICES) * native.pose[:, 3, None, None] + native.pose[:, None, :3]
    joints = np.einsum("tij,nj->tni", rotation, JOINTS) * native.pose[:, 3, None, None] + native.pose[:, None, :3]
    joints[:, 22:, 0] += native.pose[:, 10, None]
    vertices[:, 4:, 0] += native.pose[:, 10, None]
    return DecodedHuman(vertices, joints)


def reconstruction(frames=5, start=0, object_id="analytic-object"):
    pose = np.zeros((frames, 136)); pose[:, 3] = 1
    return Reconstruction(np.arange(start, start + frames), NativeMHR(pose, np.zeros(68), np.zeros(45)),
                          VERTICES[:4] * .2, np.tile(np.eye(4), (frames, 1, 1)), 1., object_id, PROVENANCE,
                          np.ones(frames, dtype=bool))


def evaluate(pred, ref, **kwargs):
    return score_episode(pred, ref, analytic_decoder, ROLES, expected_frame_indices=ref.frame_indices, **kwargs)


def transform_scene(reconstruction, scale=2., angle=.5, translation=(1., 2., 3.)):
    rotation = Rotation.from_rotvec([0, 0, angle]).as_matrix()
    translation = np.asarray(translation)
    human_pose = reconstruction.human.pose.copy()
    human_pose[:, :3] = scale * (human_pose[:, :3] @ rotation.T) + translation
    human_pose[:, 3] *= scale
    human_pose[:, 4:7] = [0, 0, angle]
    object_poses = reconstruction.object_poses.copy()
    object_poses[:, :3, :3] = rotation @ object_poses[:, :3, :3]
    object_poses[:, :3, 3] = scale * (object_poses[:, :3, 3] @ rotation.T) + translation
    return replace(reconstruction, human=replace(reconstruction.human, pose=human_pose),
                   object_poses=object_poses, object_scale=reconstruction.object_scale * scale)


def test_one_human_similarity_applies_to_both_entire_trajectories():
    reference = reconstruction()
    reference.human.pose[:, 0] = np.arange(5) ** 2 * .01
    reference.object_poses[:, 0, 3] = np.arange(5) ** 2 * .02
    prediction = transform_scene(reference)
    report = evaluate(prediction, reference)
    assert report["alignment"]["scale"] == pytest.approx(.5)
    assert report["alignment"]["shared_by_human_and_object"] is True
    for name, value in report["metrics"].items():
        if name != "pen_cm":
            assert value == pytest.approx(0, abs=1e-10)
    assert report["metrics"]["pen_cm"] is None
    assert report["official_equivalence"] is False


def test_alignment_stays_at_first_reference_when_scoring_starts_later():
    reference = reconstruction(frames=6, start=20)
    prediction = reconstruction(frames=6, start=20)
    prediction.human.pose[0, 0] = 1.
    prediction = replace(prediction, object_points=np.zeros((1, 3)))
    reference = replace(reference, object_points=np.zeros((1, 3)))
    report = evaluate(prediction, reference, scored_frame_indices=[22, 23, 24, 25])
    assert report["alignment"]["reference_frame_id"] == 20
    assert report["alignment"]["translation"] == pytest.approx([-1, 0, 0])
    assert report["metrics"]["cd_o_cm"] == pytest.approx(200)


def test_later_common_drift_is_not_fitted_away():
    reference, prediction = reconstruction(), reconstruction()
    prediction.human.pose[1:, 0] = np.arange(1, 5) * .1
    prediction.object_poses[1:, 0, 3] = np.arange(1, 5) * .1
    result = evaluate(prediction, reference)["metrics"]
    assert result["cd_h_cm"] > 0
    assert result["cd_o_cm"] > 0


def test_object_world_pose_and_relative_scale_affect_chamfer():
    reference = reconstruction()
    moved = reconstruction(); moved.object_poses[:, 0, 3] = 2.
    assert evaluate(moved, reference)["metrics"]["cd_o_cm"] > 100
    scaled = replace(reference, object_scale=3.)
    assert evaluate(scaled, reference)["metrics"]["cd_o_cm"] > 0
    assert evaluate(scaled, reference)["metrics"]["cd_h_cm"] == pytest.approx(0, abs=1e-12)


def test_frozen_motion_loses_to_accelerating_reference_no_fps_multiplier():
    reference, frozen = reconstruction(), reconstruction()
    reference.human.pose[:, 0] = np.arange(5) ** 2 * .01
    reference.object_poses[:, 0, 3] = np.arange(5) ** 2 * .02
    values = evaluate(frozen, reference)["metrics"]
    assert values["acc_h_cm_frame2"] == pytest.approx(2.)
    assert values["acc_o_cm_frame2"] == pytest.approx(4.)
    assert evaluate(reference, reference)["metrics"]["acc_h_cm_frame2"] == pytest.approx(0, abs=1e-12)


def test_body22_does_not_include_finger_motion():
    reference, prediction = reconstruction(), reconstruction()
    prediction.human.pose[:, 10] = np.arange(5) ** 2
    values = evaluate(prediction, reference)["metrics"]
    assert values["acc_h_cm_frame2"] == pytest.approx(0, abs=1e-12)
    assert values["cd_h_cm"] > 0


def test_object_acceleration_uses_origin_translation_not_rotating_centroid():
    reference, prediction = reconstruction(), reconstruction()
    reference.object_poses[:, :3, :3] = Rotation.from_rotvec(np.column_stack((np.zeros(5), np.zeros(5), np.arange(5) ** 2 * .1))).as_matrix()
    values = evaluate(prediction, reference)["metrics"]
    assert values["acc_o_cm_frame2"] == pytest.approx(0, abs=1e-12)
    assert values["cd_o_cm"] > 0


def test_hand_penetration_means_all_hands_and_scales_once_without_reference_subtraction():
    sphere_sdf = lambda points, workers=-1: np.linalg.norm(points, axis=1) - 1.
    reference = reconstruction()
    unaligned = evaluate(reference, reference, signed_distance=sphere_sdf)
    assert unaligned["metrics"]["pen_cm"] == pytest.approx(40.)  # depths .8 and 0, mean .4 m
    aligned = evaluate(transform_scene(reference), reference, signed_distance=sphere_sdf)
    assert aligned["metrics"]["pen_cm"] == pytest.approx(40.)


def test_hidden_invalid_pose_is_rejected_before_decoder():
    reference, prediction = reconstruction(), reconstruction()
    prediction.object_visible[2] = False
    prediction.object_poses[2, 0, 3] = np.nan
    def forbidden_decoder(_):
        pytest.fail("decoder must not run for malformed complete input")
    with pytest.raises(ValueError, match="all object poses"):
        score_episode(prediction, reference, forbidden_decoder, ROLES, expected_frame_indices=reference.frame_indices)


def test_hidden_frames_still_contribute_to_world_chamfer():
    reference, prediction = reconstruction(), reconstruction()
    prediction.object_visible[2] = False
    prediction.object_poses[2, 0, 3] = 3.
    assert evaluate(prediction, reference)["metrics"]["cd_o_cm"] > 0


@pytest.mark.parametrize("field,value", [("pose", np.zeros((5, 135))), ("scales", np.zeros(28)),
                                         ("scales", np.zeros((5, 68))), ("shape", np.zeros((5, 45)))])
def test_native_dimensions_and_static_identity_are_required(field, value):
    reference = reconstruction()
    prediction = replace(reference, human=replace(reference.human, **{field: value}))
    with pytest.raises(ValueError, match="native"):
        evaluate(prediction, reference)


def test_missing_frame_or_nonrigid_hidden_pose_fails():
    reference = reconstruction()
    with pytest.raises(ValueError, match="every expected source frame"):
        evaluate(reconstruction(frames=4), reference)
    invalid = reconstruction(); invalid.object_poses[2, 0, 0] = 2
    invalid.object_visible[2] = False
    with pytest.raises(ValueError, match="proper rigid"):
        evaluate(invalid, reference)


def test_degenerate_first_reference_cannot_use_later_frame():
    reference = reconstruction()
    def degenerate(native):
        result = analytic_decoder(native)
        result.vertices[0] = 0
        return result
    with pytest.raises(ValueError, match="degenerate"):
        score_episode(reference, reference, degenerate, ROLES, expected_frame_indices=reference.frame_indices)


def test_no_guessing_body_roles_and_no_nonfinite_decoder_geometry():
    reference = reconstruction()
    with pytest.raises(ValueError, match="exactly 22"):
        score_episode(reference, reference, analytic_decoder, replace(ROLES, body_joints22=list(range(21))), expected_frame_indices=reference.frame_indices)
    def bad_decoder(native):
        result = analytic_decoder(native); result.vertices[-1, 0, 0] = np.nan
        return result
    with pytest.raises(ValueError, match="nonfinite"):
        score_episode(reference, reference, bad_decoder, ROLES, expected_frame_indices=reference.frame_indices)


def test_unsigned_frame_overflow_is_rejected():
    reference = reconstruction(frames=3)
    overflow = np.array([2**63, 2**63 + 1, 2**63 + 2], dtype=np.uint64)
    with pytest.raises(ValueError, match="integer frame range"):
        score_episode(reference, reference, analytic_decoder, ROLES, expected_frame_indices=overflow)


def test_dataset_requires_every_declared_episode_and_object_and_macro_averages():
    a = replace(reconstruction(frames=3), object_points=np.zeros((1, 3)))
    b = replace(reconstruction(frames=9, object_id="second-object"), object_points=np.zeros((1, 3)))
    moved_a = replace(a, object_poses=a.object_poses.copy()); moved_a.object_poses[:, 0, 3] = .1
    moved_b = replace(b, object_poses=b.object_poses.copy()); moved_b.object_poses[:, 0, 3] = .3
    kwargs = dict(expected_frames={"a": a.frame_indices, "b": b.frame_indices}, required_objects=[a.object_id, b.object_id])
    result = score_dataset({"a": moved_a, "b": moved_b}, {"a": a, "b": b}, analytic_decoder, ROLES, **kwargs)
    assert result["mean"]["cd_o_cm"] == pytest.approx(40.)  # equal episodes, not weighted by length
    assert result["mean"]["pen_cm"] is None
    assert "composite" not in result["mean"]
    with pytest.raises(ValueError, match="every declared episode"):
        score_dataset({"a": a}, {"a": a, "b": b}, analytic_decoder, ROLES, **kwargs)
    with pytest.raises(ValueError, match="every required object"):
        score_dataset({"a": a, "b": b}, {"a": a, "b": b}, analytic_decoder, ROLES,
                      expected_frames=kwargs["expected_frames"], required_objects=["absent-object"])


def cli_fixture(tmp_path, monkeypatch):
    item = reconstruction()
    artifact = tmp_path / "native.npz"
    np.savez(artifact, pose=item.human.pose, scales=item.human.scales, shape=item.human.shape,
             frame_indices=item.frame_indices, object_poses=item.object_poses, object_scale=np.array(1.), object_visible=item.object_visible)
    trimesh.creation.icosphere(subdivisions=0).export(tmp_path / "mesh.ply")
    manifest = {"schema_version": 2, "provenance": asdict(PROVENANCE), "expected_frames": {"0": {"start": 0, "count": 5}},
                "required_objects": [item.object_id], "episodes": {"0": {"object_id": item.object_id, "artifact": "native.npz", "mesh": "mesh.ply", "sample_count": 32}}}
    for name in ("pred.json", "reference.json"):
        (tmp_path / name).write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "roles.json").write_text(json.dumps(asdict(ROLES)), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "analytic_native_fixture", SimpleNamespace(decode=analytic_decoder))
    args = ["--pred", str(tmp_path / "pred.json"), "--reference", str(tmp_path / "reference.json"),
            "--decoder", "analytic_native_fixture:decode", "--roles", str(tmp_path / "roles.json"), "--out", str(tmp_path / "report.json")]
    return manifest, args


def test_cli_samples_real_mesh_and_computes_all_five_diagnostics(tmp_path, monkeypatch):
    _, args = cli_fixture(tmp_path, monkeypatch)
    score.main(args)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["official_equivalence"] is False
    assert report["mean"]["cd_o_cm"] == pytest.approx(0, abs=1e-12)
    assert report["mean"]["pen_cm"] > 0
    assert len(report["mean"]) == 5
    assert "mesh_sha256=" in report["episodes"]["0"]["object_sampling"]["prediction"]


@pytest.mark.parametrize("legacy", [["--gt", "anything"], ["--align", "first-object"], ["--stride", "2"], ["--strict"]])
def test_legacy_cli_flags_are_rejected(tmp_path, monkeypatch, legacy):
    _, args = cli_fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as error:
        score.main(args + legacy)
    assert error.value.code == 2


def test_reference_is_required_and_no_default_is_opened():
    with pytest.raises(SystemExit) as error:
        score.main([])
    assert error.value.code == 2


@pytest.mark.parametrize("mutation", ["prohibited_path", "prohibited_provenance", "fake_track1", "fake_backend", "fake_boolean",
                                      "mixed_episode_provenance", "duplicate_key", "nonfinite_json"])
def test_bad_manifest_fails_before_arrays_or_decoder(tmp_path, monkeypatch, mutation):
    manifest, args = cli_fixture(tmp_path, monkeypatch)
    if mutation == "prohibited_path":
        manifest["episodes"]["0"]["artifact"] = "track_2/forbidden.npz"
    elif mutation == "prohibited_provenance":
        manifest["provenance"] = {"kind": "track2", "evidence": "disallowed declaration"}
    elif mutation == "fake_track1":
        manifest["provenance"]["kind"] = "track1"; manifest["output_kind"] = "synthetic_smoke"
    elif mutation == "fake_backend":
        manifest["provenance"]["kind"] = "track1"; manifest["backends"] = {"human": "fake"}
    elif mutation == "fake_boolean":
        manifest["provenance"]["kind"] = "track1"; manifest["contains_fake_outputs"] = True
    elif mutation == "mixed_episode_provenance":
        manifest["episodes"]["0"]["provenance"] = {"kind": "track1", "evidence": "contradictory kind"}
    payload = json.dumps(manifest)
    if mutation == "duplicate_key":
        payload = payload.replace('"schema_version": 2', '"schema_version": 2, "schema_version": 2')
    elif mutation == "nonfinite_json":
        payload = payload.replace('"count": 5', '"count": NaN')
    (tmp_path / "pred.json").write_text(payload, encoding="utf-8")
    def forbidden(*args, **kwargs):
        pytest.fail("invalid provenance/metadata must fail before arrays or decoder imports")
    monkeypatch.setattr(score.np, "load", forbidden)
    monkeypatch.setattr(score.importlib, "import_module", forbidden)
    with pytest.raises(SystemExit) as error:
        score.main(args)
    assert error.value.code == 2


def test_cli_does_not_overwrite_native_input(tmp_path, monkeypatch):
    _, args = cli_fixture(tmp_path, monkeypatch)
    original = (tmp_path / "native.npz").read_bytes()
    args[-1] = str(tmp_path / "native.npz")
    with pytest.raises(SystemExit):
        score.main(args)
    assert (tmp_path / "native.npz").read_bytes() == original


def test_cli_rejects_mesh_format_with_external_sidecars(tmp_path, monkeypatch):
    manifest, args = cli_fixture(tmp_path, monkeypatch)
    (tmp_path / "mesh.obj").write_text("mtllib unknown.mtl\n", encoding="utf-8")
    manifest["episodes"]["0"]["mesh"] = "mesh.obj"
    (tmp_path / "pred.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit):
        score.main(args)


def test_dataset_does_not_mix_synthetic_and_track1_reports():
    synthetic = reconstruction()
    declared_track1 = replace(reconstruction(object_id="other"), provenance=Provenance("track1", "caller supplied reference"))
    episodes = {"a": synthetic, "b": declared_track1}
    with pytest.raises(ValueError, match="cannot mix"):
        score_dataset(episodes, episodes, analytic_decoder, ROLES,
                      expected_frames={key: value.frame_indices for key, value in episodes.items()},
                      required_objects=["analytic-object", "other"])


def test_cli_subprocess_native_glb_reports_five_known_metrics(tmp_path, monkeypatch):
    manifest, args = cli_fixture(tmp_path, monkeypatch)
    trimesh.creation.box(extents=[2., 2., 2.]).export(tmp_path / "box.glb")
    manifest["episodes"]["0"]["mesh"] = "box.glb"
    for name in ("pred.json", "reference.json"):
        (tmp_path / name).write_text(json.dumps(manifest), encoding="utf-8")
    decoder_source = (
        "import numpy as np\nfrom v2hoi.score import DecodedHuman\n"
        "def decode(native):\n"
        f"    vertices = np.array({VERTICES.tolist()!r}, dtype=float)\n"
        f"    joints = np.array({JOINTS.tolist()!r}, dtype=float)\n"
        "    return DecodedHuman(vertices[None] + native.pose[:, None, :3], joints[None] + native.pose[:, None, :3])\n"
    )
    (tmp_path / "independent_fixture_decoder.py").write_text(decoder_source, encoding="utf-8")
    args[args.index("--decoder") + 1] = "independent_fixture_decoder:decode"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path(__file__).resolve().parents[1] / "src"), environment.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-m", "v2hoi.score", *args], cwd=tmp_path, env=environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["official_equivalence"] is False
    assert report["mean"] == pytest.approx({"cd_h_cm": 0., "cd_o_cm": 0., "acc_h_cm_frame2": 0.,
                                           "acc_o_cm_frame2": 0., "pen_cm": 40.}, abs=1e-9)
