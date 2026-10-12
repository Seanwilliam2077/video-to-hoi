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
