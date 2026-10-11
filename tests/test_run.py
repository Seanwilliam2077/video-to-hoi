"""Track 1 metadata and synthetic format smoke tests; no reference data or scorer."""

import json
import sys

import numpy as np
import pytest

from v2hoi.clips import load_clips
from v2hoi.contracts import ContractError, Depth, DepthScale, Human, Masks, ObjectAsset, RefinedMotion
from v2hoi.dataset import list_episodes, load_episode
from v2hoi.download import patterns
from v2hoi.run import main, parse_backends, run_pipeline


EPISODES = {3: ("iron", 12), 4: ("iron", 10), 8: ("bowl", 9)}
QUIET = {"log": lambda *_: None}


@pytest.mark.parametrize("field,value", [(None,None),("scale_factor",-1),
    ("mask_sha256","other-video"),("faces_preserved",False),
    ("fine_candidates",3),("revision","unreviewed")])
def test_legacy_scale_search_requires_preserved_budget_and_matching_inputs(tmp_path, field, value):
    from v2hoi.native import sha256
    from v2hoi.mesh_budget import POLICY, verify_budgeted_preparation
    from v2hoi.legacy_scale import METHOD, REVISION
    def artifact(name, payload):
        path = tmp_path/name
        path.write_bytes(payload)
        return {"path":name,"sha256":sha256(path)}
    original = artifact("original.fixture", b"synthetic dense mesh")
    budgeted = artifact("budget.fixture", b"synthetic fixed faces")
    scaled = artifact("scaled.fixture", b"synthetic uniformly scaled faces")
    code = artifact("source.fixture", b"independent source fixture")
    budget = {"policy":POLICY,"method":"v2dlb.mesh_budget.budget_mesh",
              "face_budget":4096,"vertex_budget":4096,"appearance_preserved":True,
              "full_mesh":original,"output_mesh":budgeted,"official_code":code}
    budget_receipt = artifact("budget.json", json.dumps(budget).encode())
    search = {"status":"COMPLETED","provenance":"track1","method":METHOD,"revision":REVISION,
              "input_mesh":budgeted,"output_mesh":scaled,"scale_factor":1.5,
              "coarse_candidates":30,"fine_candidates":11,"registration_iterations":5,"surface_samples":10000,
              "video_sha256":"video-fixture","mask_sha256":"mask-fixture",
              "uniform_scale_only":True,"appearance_preserved":True,"faces_preserved":True}
    for key in ("official_code","chamfer_code","io_adapter","fine_search","source_preparation"):
        search[key] = code
    if field:
        search[field] = value
    prepared = {"input_mesh_policy":POLICY,"mesh":scaled,"mesh_budget":{"receipt":budget_receipt},
                "mesh_scale_search":{"receipt":artifact("search.json",json.dumps(search).encode())},
                "video_sha256":"video-fixture","masks":{"sha256":"mask-fixture"}}
    if field:
        with pytest.raises(ValueError, match="legacy scale"):
            verify_budgeted_preparation(prepared,tmp_path,original["sha256"],allowed=True)
    else:
        assert verify_budgeted_preparation(prepared,tmp_path,original["sha256"],allowed=True)
        evidence_directory = tmp_path/"evidence"
        evidence_directory.mkdir()
        for key in ("full_mesh","output_mesh","official_code"):
            budget[key] = {**budget[key],"path":"../"+budget[key]["path"]}
        prepared["mesh_budget"]["receipt"] = artifact("evidence/budget.json",json.dumps(budget).encode())
        assert verify_budgeted_preparation(prepared,tmp_path,original["sha256"],allowed=True)
        (tmp_path/"budget.fixture").write_bytes(b"tampered source geometry")
        with pytest.raises(ValueError, match="changed"):
            verify_budgeted_preparation(prepared,tmp_path,original["sha256"],allowed=True)


def test_legacy_vertex_color_writer_does_not_require_uv_texture_objects():
    """Independent I/O regression fixture; no models or challenge assets."""
    from v2hoi.legacy_scale import adapted_search_source
    fixture = '''
def estimate_metric_scale(hy_file, mesh_out, best_scale):
    debug_dir = "shared-debug"
    mesh_hy_orig: Meshes = load_objs_as_meshes([hy_file])[0]
    tex: TexturesUV = mesh_hy_orig.textures
    save_obj(mesh_out, mesh_hy_orig.verts_packed(), texture_map=tex.maps_padded()[0])
    return best_scale * 3, debug_dir
'''
    saved = []
    context = {"_save_scaled_mesh":lambda *args:saved.append(args)}
    exec(adapted_search_source(fixture,"isolated-debug"), context)
    assert context["estimate_metric_scale"]("source","scaled",2.) == (6.,"isolated-debug")
    assert saved == [("source","scaled",2.)]


def test_pre_inference_mesh_budget_requires_opt_in_and_preserved_evidence(tmp_path):
    """Independent synthetic bytes; never read challenge assets or real models."""
    from v2hoi.native import sha256
    from v2hoi.mesh_budget import POLICY, verify_budgeted_preparation
    def artifact(name, payload):
        path = tmp_path / name
        path.write_bytes(payload)
        return {"path": name, "sha256": sha256(path)}
    full = artifact("full.fixture", b"synthetic original surface")
    reduced = artifact("reduced.fixture", b"synthetic budget surface")
    code = artifact("source.fixture", b"synthetic source evidence")
    receipt = {"policy": POLICY, "method": "v2dlb.mesh_budget.budget_mesh",
        "face_budget": 4096, "vertex_budget": 4096, "appearance_preserved": True,
        "full_mesh": full, "output_mesh": reduced, "official_code": code}
    evidence = artifact("receipt.json", json.dumps(receipt).encode())
    prepared = {"input_mesh_policy": POLICY, "mesh": reduced, "mesh_budget": {"receipt": evidence}}
    with pytest.raises(ValueError, match="requires --allow-budgeted-mesh"):
        verify_budgeted_preparation(prepared, tmp_path, full["sha256"], allowed=False)
    assert verify_budgeted_preparation(prepared, tmp_path, full["sha256"], allowed=True)
    with pytest.raises(ValueError, match="does not match"):
        verify_budgeted_preparation(prepared, tmp_path, "wrong-generation", allowed=True)
    (tmp_path / full["path"]).write_bytes(b"altered preserved source")
    with pytest.raises(ValueError, match="changed"):
        verify_budgeted_preparation(prepared, tmp_path, full["sha256"], allowed=True)
    (tmp_path / full["path"]).write_bytes(b"synthetic original surface")
    (tmp_path / reduced["path"]).write_bytes(b"altered inference surface")
    with pytest.raises(ValueError, match="changed"):
        verify_budgeted_preparation(prepared, tmp_path, full["sha256"], allowed=True)
    assert not verify_budgeted_preparation({"mesh": full}, tmp_path, full["sha256"], allowed=False)
    with pytest.raises(ValueError, match="does not match"):
        verify_budgeted_preparation({"mesh": reduced}, tmp_path, full["sha256"], allowed=False)


@pytest.mark.parametrize("bad_field,bad_value", [(None,None), ("status","REJECTED"),
    ("scale_factor",-1), ("scale_factor",float("nan")), ("appearance_preserved",False),
    ("training_frame_ids",[1,2,3]), ("holdout_frame_ids",[1]), ("video_sha256","changed-video")])
def test_calibrated_mesh_requires_independent_valid_evidence(tmp_path, bad_field, bad_value):
    from v2hoi.native import sha256
    from v2hoi.mesh_budget import verify_budgeted_preparation
    def artifact(name, payload):
        path = tmp_path/name
        path.write_bytes(payload)
        return {"path":name,"sha256":sha256(path)}
    original = artifact("original.fixture", b"independent synthetic original")
    scaled = artifact("scaled.fixture", b"independent synthetic scaled")
    code = artifact("code.fixture", b"independent synthetic solver evidence")
    record = {"status":"PASS","provenance":"track1","method":"NVIDIA scale_mesh_srt.optimize_srt(mode=str)",
        "input_mesh":original,"output_mesh":scaled,"official_code":code,"scale_factor":2.,
        "video_sha256":"synthetic-video","mask_sha256":"synthetic-mask",
        "uniform_scale_only":True,"appearance_preserved":True,"training_frame_ids":[0,2,4],
        "holdout_frame_ids":[1,3,5],"holdout_projected_vertex_iou":{"mean_after":.8,"minimum_mean":.4}}
    if bad_field:
        record[bad_field] = bad_value
    receipt = artifact("calibration.json", json.dumps(record).encode())
    prepared = {"mesh":scaled,"mesh_calibration":{"receipt":receipt},
        "video_sha256":"synthetic-video","masks":{"sha256":"synthetic-mask"}}
    if bad_field:
        with pytest.raises(ValueError, match="calibration"):
            verify_budgeted_preparation(prepared, tmp_path, original["sha256"], allowed=False)
    else:
        assert not verify_budgeted_preparation(prepared, tmp_path, original["sha256"], allowed=False)


def test_motion_audit_measures_known_rotation_and_translation_without_smoothing():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("motion_audit_fixture", Path(__file__).parents[1]/"tools/windows/audit_object_motion.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    poses = np.repeat(np.eye(4)[None],3,axis=0)
    poses[1,:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]]
    poses[2,:3,:3] = [[-1,0,0],[0,-1,0],[0,0,1]]
    poses[1:,0,3] = 1.
    before = poses.copy()
    measured = audit.motion_statistics(poses,[0,1,2],30)
    assert measured["translation_increment_m"] == {"median":.5,"p95":.95,"max":1.}
    assert measured["rotation_increment_deg"]["max"] == pytest.approx(90.)
    assert measured["largest_translation_changes"][0]["to_frame"] == 1
    assert np.array_equal(poses,before)


@pytest.fixture
def dataset_root(tmp_path):
    """Hand-authored metadata for tiny Track 1 shaped clips; no video is read."""
    root = tmp_path / "track_1"
    (root / "meta").mkdir(parents=True)
    info = {
        "robot_type": "video_only_object_tracking",
        "fps": 30,
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"observation.images.exo_camera": {"shape": [48, 64, 3]}},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "episodes_metadata.jsonl").write_text("".join(
        json.dumps({"episode_index": e, "object": name, "object_prompt": f"a {name}.",
                    "camera": "front", "video_key": "observation.images.exo_camera"}) + "\n"
        for e, (name, _) in EPISODES.items()
    ), encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text("".join(
        json.dumps({"episode_index": e, "length": n}) + "\n"
        for e, (_, n) in EPISODES.items()
    ), encoding="utf-8")
    return root


def test_clips_come_from_track1_metadata(dataset_root):
    clips = load_clips("track1", root=dataset_root)
    assert [(c.episode, c.object_name, c.n_frames) for c in clips] == [
        (3, "iron", 12), (4, "iron", 10), (8, "bowl", 9)
    ]
    assert clips[0].prompt == "a iron." and clips[0].camera == "front"
    assert (clips[0].width, clips[0].height) == (64, 48)
    assert clips[0].video.name == "episode_000003.mp4"
    with pytest.raises(ValueError, match=r"has no episodes \[5\]"):
        load_clips("track1", [5], root=dataset_root)


def test_synthetic_pipeline_writes_internal_export(tmp_path, dataset_root):
    run = run_pipeline(tmp_path / "runs" / "demo", "track1", dataset_root=dataset_root, **QUIET)

    run.load(Masks, episode=3).validate(12, 48, 64)
    run.load(Depth, episode=3).validate(12, 48, 64)
    run.load(DepthScale, episode=3).validate()
    assert run.load(ObjectAsset, name="iron").episodes == [3, 4]
    run.load(RefinedMotion, episode=8).validate(9)

    export = run.root / "export"
    assert list_episodes(export) == [3, 4, 8]
    episode = load_episode(export, 4, mask_hidden=False)
    assert len(episode) == 10 and np.isfinite(episode.obj_T).all()
    assert np.allclose(episode.transl, run.load(Human, episode=4).transl)
    with np.load(export / "mhr" / "episode_000004.npz") as mhr:
        assert mhr["mhr_body_pose"].shape == (10, 133)
    info = json.loads((export / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["source_dataset"] == "track1"


def test_stage_reuse_requires_fresh_refinement(tmp_path, dataset_root):
    runs = tmp_path / "runs"
    run_pipeline(runs / "base", "track1", dataset_root=dataset_root, **QUIET)

    with pytest.raises(ContractError, match="refine"):
        run_pipeline(runs / "stale", "track1", stages=["motion", "export"],
                     upstream=runs / "base", dataset_root=dataset_root, **QUIET)

    child = run_pipeline(runs / "try", "track1", stages=["motion", "refine", "export"],
                         upstream=runs / "base", dataset_root=dataset_root, **QUIET)
    assert sorted(p.name for p in child.root.iterdir()) == ["export", "motion", "refine", "run.json"]
    assert list_episodes(child.root / "export") == [3, 4, 8]


@pytest.mark.parametrize("dataset", ["tier1", "tier2", "track2"])
def test_legacy_dataset_names_are_rejected(tmp_path, dataset_root, dataset):
    with pytest.raises(ValueError, match="Track 1"):
        run_pipeline(tmp_path / "run", dataset, dataset_root=dataset_root, **QUIET)
    with pytest.raises(ValueError, match="unknown dataset"):
        load_clips(dataset, root=dataset_root)


def test_track2_path_is_rejected_before_metadata_read(tmp_path):
    with pytest.raises(ValueError, match="Track 2 root is prohibited"):
        load_clips("track1", root=tmp_path / "track_2" / "track_1")


@pytest.mark.parametrize("backend", ["human=tier2", "motion=tier2", "objects=reference"])
def test_legacy_backends_are_rejected_by_parser(backend):
    with pytest.raises(ValueError, match="Track 2 assets"):
        parse_backends([backend])


@pytest.mark.parametrize("stage,backend", [("human", "tier2"), ("motion", "tier2"),
                                           ("objects", "reference")])
def test_legacy_backends_are_rejected_by_runner(tmp_path, dataset_root, stage, backend):
    with pytest.raises(ContractError, match="Track 2 assets"):
        run_pipeline(tmp_path / "run", "track1", stages=[stage], backends={stage: backend},
                     dataset_root=dataset_root, **QUIET)


def test_backend_choice_and_unknown_name(tmp_path, dataset_root):
    assert parse_backends(["motion=foundationpose"])["motion"] == "foundationpose"
    assert parse_backends([])["refine"] == "fake"
    with pytest.raises(ValueError, match="<stage>=<name>"):
        parse_backends(["tracking=fp"])
    with pytest.raises(ValueError, match="stage human has no backend 'sam3d'"):
        run_pipeline(tmp_path / "run", "track1", stages=["human"], backends={"human": "sam3d"},
                     dataset_root=dataset_root, **QUIET)


def test_score_cli_is_rejected_without_reading_data(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["v2hoi.run", "--run-id", "demo", "--score"])
    with pytest.raises(SystemExit):
        main()
    assert "--score" in capsys.readouterr().err


def test_download_patterns_are_track1_only():
    for include_videos in (False, True):
        selected = patterns(include_videos)
        assert selected and all(path.startswith("track_1/") for path in selected)
        assert ("track_1/videos/**" in selected) is include_videos
