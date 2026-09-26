"""The fake pipeline end to end on a tiny dataset: every stage, upstream runs, Tier 2 development
input, and export into the scorer."""
import json

import numpy as np
import pytest

from test_score import FakeBody, box_mesh, make_episode, write_root
from v2hoi.clips import TRACK1_ROOT, load_clips
from v2hoi.contracts import ContractError, Depth, DepthScale, Human, Masks, Motion, ObjectAsset, RefinedMotion
from v2hoi.dataset import TIER1_ROOT, list_episodes, load_episode
from v2hoi.run import parse_backends, run_pipeline
from v2hoi.score import LEADERBOARD, Config, score
from v2hoi.stages.human import Tier2Human
from v2hoi.stages.motion import Tier2Motion
from v2hoi.stages.objects import ReferenceObjects

EPISODES = {3: ("iron", 12), 4: ("iron", 10), 8: ("bowl", 9)}
QUIET = {"log": lambda *_: None}


@pytest.fixture
def dataset_root(tmp_path):
    """A Track 1 style root: metadata only, 64×48 video."""
    root = tmp_path / "track1"
    (root / "meta").mkdir(parents=True)
    info = {
        "fps": 30, "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"observation.images.exo_camera": {"shape": [48, 64, 3]}},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "episodes_metadata.jsonl").write_text("".join(
        json.dumps({"episode_index": e, "object": name, "object_prompt": f"a {name}.", "camera": "front",
                    "video_key": "observation.images.exo_camera"}) + "\n"
        for e, (name, _) in EPISODES.items()
    ))
    (root / "meta" / "episodes.jsonl").write_text("".join(
        json.dumps({"episode_index": e, "length": n}) + "\n" for e, (_, n) in EPISODES.items()
    ))
    return root


def test_clips_come_from_metadata(dataset_root):
    clips = load_clips("track1", root=dataset_root)
    assert [(c.episode, c.object_name, c.n_frames) for c in clips] == [(3, "iron", 12), (4, "iron", 10), (8, "bowl", 9)]
    assert clips[0].prompt == "a iron." and clips[0].camera == "front"
    assert (clips[0].width, clips[0].height) == (64, 48)
    assert clips[0].video.name == "episode_000003.mp4"
    with pytest.raises(ValueError, match=r"has no episodes \[5\]"):
        load_clips("track1", [5], root=dataset_root)


def test_fake_pipeline_exports_what_the_scorer_reads(tmp_path, dataset_root, monkeypatch):
    run = run_pipeline(tmp_path / "runs" / "demo", "track1", dataset_root=dataset_root, **QUIET)

    run.load(Masks, episode=3).validate(12, 48, 64)
    run.load(Depth, episode=3).validate(12, 48, 64)
    run.load(DepthScale, episode=3).validate()
    assert run.load(ObjectAsset, name="iron").episodes == [3, 4]  # one mesh per object
    run.load(RefinedMotion, episode=8).validate(9)

    export = run.root / "export"
    assert list_episodes(export) == [3, 4, 8]
    ep = load_episode(export, 4, mask_hidden=False)
    assert len(ep) == 10 and np.isfinite(ep.obj_T).all()
    assert np.allclose(ep.transl, run.load(Human, episode=4).transl)
    with np.load(export / "mhr" / "episode_000004.npz") as mhr:
        assert mhr["mhr_body_pose"].shape == (10, 133)

    # the export scores against itself: zero error, but its meshes are "reference assets"
    monkeypatch.setattr("v2hoi.body.SomaBody", lambda device=None: FakeBody())
    report = score(export, export, cfg=Config(object_samples=500), **QUIET)
    assert set(report["mean"]["leaderboard"]) == set(LEADERBOARD)
    assert report["mean"]["leaderboard"]["cd_h_cm"] < 1e-4
    assert report["mean"]["leaderboard"]["cd_o_cm"] < 1e-4
    assert report["submission"]["official_settings"]
    assert all("Track 2 assets" in v for v in report["submission"]["violations"])


def test_a_stage_runs_against_an_upstream_snapshot(tmp_path, dataset_root):
    runs = tmp_path / "runs"
    run_pipeline(runs / "base", "track1", dataset_root=dataset_root, **QUIET)

    # new tracking with the upstream's old refined trajectory would be silently ignored
    with pytest.raises(ContractError, match="motion/000003/motion.npz is newer than refine/000003/motion.npz"):
        run_pipeline(runs / "stale", "track1", stages=["motion", "export"], upstream=runs / "base",
                     dataset_root=dataset_root, **QUIET)

    child = run_pipeline(runs / "try", "track1", stages=["motion", "refine", "export"], upstream=runs / "base",
                         dataset_root=dataset_root, **QUIET)
    assert sorted(p.name for p in child.root.iterdir()) == ["export", "motion", "refine", "run.json"]
    assert list_episodes(child.root / "export") == [3, 4, 8]
    with pytest.raises(ValueError, match="is on track1, not tier1"):
        run_pipeline(runs / "other", "tier1", upstream=runs / "base", dataset_root=dataset_root, **QUIET)


@pytest.fixture
def references(tmp_path, monkeypatch):
    """Stand-ins for the Tier 1 meshes and the Tier 2 trajectories of the tiny dataset."""
    tier1 = tmp_path / "tier1_ref"
    for name in ("iron", "bowl"):
        (tier1 / "mesh" / name).mkdir(parents=True)
        box_mesh(tier1 / "mesh" / name / f"{name}.glb")
    tier2 = tmp_path / "tier2"
    noisy = {e: make_episode(tier1 / "mesh" / name / f"{name}.glb", T=n) for e, (name, n) in EPISODES.items()}
    for ep in noisy.values():  # the SOMA-X parameter sizes of the real Tier 2
        ep.identity = np.zeros((len(ep), 45), np.float32)
        ep.scale = np.zeros((len(ep), 68), np.float32)
        ep.bone_flex = np.zeros((len(ep), 6), np.float32)
    noisy[3].obj_T[5] = np.nan  # a dropped frame
    write_root(tier2, noisy, tier1 / "mesh" / "iron" / "iron.glb")
    monkeypatch.setattr(ReferenceObjects, "root", tier1)
    monkeypatch.setattr(Tier2Human, "root", tier2)
    monkeypatch.setattr(Tier2Motion, "root", tier2)
    return noisy


def test_tier2_input_for_the_refine_stage(tmp_path, dataset_root, references):
    backends = {"human": "tier2", "motion": "tier2", "objects": "reference"}
    run = run_pipeline(tmp_path / "t2", "tier1", backends=backends, dataset_root=dataset_root, **QUIET)

    motion = run.load(Motion, episode=3)
    assert motion.confidence[5] == 0 and motion.confidence[4] == 1
    assert np.allclose(motion.T_cam_obj[5], motion.T_cam_obj[4], atol=1e-6)
    ep = load_episode(run.root / "export", 3, mask_hidden=False)
    assert np.allclose(ep.transl, references[3].transl)

    with pytest.raises(ContractError, match="only Tier 1 development runs"):
        run_pipeline(tmp_path / "t1", "track1", backends=backends, dataset_root=dataset_root, **QUIET)


def test_backend_choice():
    assert parse_backends(["motion=foundationpose"])["motion"] == "foundationpose"
    assert parse_backends([])["refine"] == "fake"
    with pytest.raises(ValueError, match="<stage>=<name>"):
        parse_backends(["tracking=fp"])


def test_unknown_backend_names_the_choices(tmp_path, dataset_root):
    with pytest.raises(ValueError, match=r"stage human has no backend 'sam3d'; available: \['fake', 'tier2'\]"):
        run_pipeline(tmp_path / "r", "track1", stages=["human"], backends={"human": "sam3d"},
                     dataset_root=dataset_root, **QUIET)


@pytest.mark.skipif(not (TRACK1_ROOT / "meta" / "info.json").is_file(), reason="Track 1 not downloaded")
def test_track1_clips():
    clips = load_clips("track1")
    assert len(clips) == 30
    assert {c.camera for c in clips} == {
        "back_stereo_camera_left", "front_stereo_camera_left", "left_stereo_camera_left", "right_stereo_camera_left",
    }
    assert all(c.prompt and (c.width, c.height) == (1536, 1152) for c in clips)


@pytest.mark.skipif(not (TIER1_ROOT / "meta" / "info.json").is_file(), reason="Tier 1 not downloaded")
def test_tier1_clips():
    clips = load_clips("tier1", [7, 9])
    assert [(c.object_name, c.n_frames, c.camera) for c in clips] == [("iron", 896, None), ("big_red_bowl", 870, None)]
    assert clips[0].prompt == "An metallic iron with blue accents."
