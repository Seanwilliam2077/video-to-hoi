"""The fake pipeline end to end on a tiny dataset: every stage, upstream runs, export into the scorer."""
import json

import numpy as np
import pytest

from test_score import FakeBody
from v2hoi.clips import TRACK1_ROOT, load_clips
from v2hoi.contracts import ContractError, Human, Masks, Motion, ObjectAsset
from v2hoi.dataset import TIER1_ROOT, list_episodes, load_episode
from v2hoi.run import parse_backends, run_pipeline
from v2hoi.score import LEADERBOARD, Config, score
from v2hoi.stages.objects import ReferenceObjects

EPISODES = {3: ("iron", 12), 4: ("iron", 10), 8: ("bowl", 9)}


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
    run = run_pipeline(tmp_path / "runs" / "demo", "track1", dataset_root=dataset_root, log=lambda *_: None)

    masks = run.load(Masks, episode=3)
    masks.validate(12, 48, 64)
    assert run.load(ObjectAsset, name="iron").episodes == [3, 4]  # one mesh per object
    run.load(Motion, episode=8).validate(9)

    export = run.root / "export"
    assert list_episodes(export) == [3, 4, 8]
    ep = load_episode(export, 4, mask_hidden=False)
    assert len(ep) == 10 and np.isfinite(ep.obj_T).all()
    assert np.allclose(ep.transl, run.load(Human, episode=4).transl)
    with np.load(export / "mhr" / "episode_000004.npz") as mhr:
        assert mhr["mhr_body_pose"].shape == (10, 133)

    # the export scores against itself: zero error, but its meshes are "reference assets"
    monkeypatch.setattr("v2hoi.body.SomaBody", lambda device=None: FakeBody())
    report = score(export, export, cfg=Config(object_samples=500), log=lambda *_: None)
    assert set(report["mean"]["leaderboard"]) == set(LEADERBOARD)
    assert report["mean"]["leaderboard"]["cd_h_cm"] < 1e-4
    assert report["mean"]["leaderboard"]["cd_o_cm"] < 1e-4
    assert report["submission"]["official_settings"]
    assert all("Track 2 assets" in v for v in report["submission"]["violations"])


def test_a_stage_runs_against_an_upstream_snapshot(tmp_path, dataset_root):
    runs = tmp_path / "runs"
    run_pipeline(runs / "base", "track1", dataset_root=dataset_root, log=lambda *_: None)
    child = run_pipeline(runs / "try", "track1", stages=["motion", "export"], upstream=runs / "base",
                         dataset_root=dataset_root, log=lambda *_: None)

    assert sorted(p.name for p in child.root.iterdir()) == ["export", "motion", "run.json"]
    assert list_episodes(child.root / "export") == [3, 4, 8]
    with pytest.raises(ValueError, match="is on track1, not tier1"):
        run_pipeline(runs / "other", "tier1", upstream=runs / "base", dataset_root=dataset_root,
                     log=lambda *_: None)


def test_reference_meshes_are_for_tier1_only(tmp_path, dataset_root, monkeypatch):
    kwargs = dict(stages=["objects"], backends={"objects": "reference"}, dataset_root=dataset_root,
                  log=lambda *_: None)
    with pytest.raises(ContractError, match="Track 2 assets"):
        run_pipeline(tmp_path / "t1", "track1", **kwargs)

    ref = tmp_path / "tier1"
    for name in ("iron", "bowl"):
        (ref / "mesh" / name).mkdir(parents=True)
        (ref / "mesh" / name / f"{name}.glb").write_bytes(name.encode())
    monkeypatch.setattr(ReferenceObjects, "root", ref)
    run = run_pipeline(tmp_path / "dev", "tier1", **kwargs)
    assert run.mesh("iron").read_bytes() == b"iron"


def test_backend_choice():
    assert parse_backends(["motion=foundationpose"])["motion"] == "foundationpose"
    assert parse_backends([])["export"] == "tier1"
    with pytest.raises(ValueError, match="<stage>=<name>"):
        parse_backends(["tracking=fp"])


def test_unknown_backend_names_the_choices(tmp_path, dataset_root):
    with pytest.raises(ValueError, match=r"stage human has no backend 'sam3d'; available: \['fake'\]"):
        run_pipeline(tmp_path / "r", "track1", stages=["human"], backends={"human": "sam3d"},
                     dataset_root=dataset_root, log=lambda *_: None)


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
