"""Synthetic checks for target selection and immutable complete mask outputs."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("prepare_body_masks", TOOLS / "prepare_body_masks.py")
masks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(masks)
sys.path.pop(0)


def test_target_point_excludes_more_confident_background_person():
    background = {"bbox_xyxy": [0, 0, 10, 10], "score": .999}
    target = {"bbox_xyxy": [20, 20, 40, 60], "score": .81}
    assert masks.select_candidate([background, target], [30, 30]) == target


def test_no_matching_target_is_not_replaced_with_an_unrelated_person():
    with pytest.raises(ValueError, match="no person detection"):
        masks.select_candidate([{"bbox_xyxy": [0, 0, 10, 10], "score": .99}], [20, 20])


def test_bbox_maxima_are_exclusive_and_quality_is_observation_not_confidence():
    image = np.zeros((8, 10), dtype=bool)
    image[2:5, 3:7] = True
    observed = masks.mask_observation(image)
    assert observed["bbox_xyxy"] == [3, 2, 7, 5]
    assert observed["area_pixels"] == 12
    assert observed["valid"] is True
    assert observed["confidence"] is None


@pytest.mark.parametrize("fill,flag", [(False, "empty_mask"), (True, "full_image_mask")])
def test_empty_and_full_image_masks_are_never_valid(fill, flag):
    item = masks.mask_observation(np.full((8, 8), fill, dtype=bool))
    assert not item["valid"]
    assert flag in item["review_flags"]


def test_disjoint_temporal_masks_raise_review_flag():
    first = np.zeros((10, 10), dtype=bool)
    second = first.copy()
    first[:3, :3] = True
    second[7:, 7:] = True
    item = masks.mask_observation(second, first)
    assert item["adjacent_mask_iou"] == 0
    assert "adjacent_mask_iou_below_0.3" in item["review_flags"]


def complete_output(tmp_path):
    (tmp_path / "human_masks").mkdir()
    for i in range(2):
        (tmp_path / f"human_masks/{i:06d}.png").write_bytes(bytes([i]))
    masks.write_json(tmp_path / "observations.json", {"frames": []})
    masks.write_json(tmp_path / "masks-provenance.json", {
        "checkpoint_sha256": masks.SAM_CHECKPOINT_SHA256,
        "code_revision": masks.SAM_REVISION, "prompt_entry_sha256": "prompt"})
    document = {"schema_version": "body_masks.v1", "episode_index": 0, "video_sha256": "video",
                "files": [masks.file_entry(tmp_path / f"human_masks/{i:06d}.png", tmp_path) for i in range(2)],
                "artifacts": [masks.file_entry(tmp_path / p, tmp_path)
                              for p in ["observations.json", "masks-provenance.json"]]}
    masks.write_json(tmp_path / "masks-manifest.json", document)
    return {"episode_index": 0, "frame_count": 2}


def test_resume_rechecks_every_byte_and_missing_final_frame(tmp_path):
    episode = complete_output(tmp_path)
    masks.verify_existing(tmp_path, episode, "video", "prompt")
    (tmp_path / "human_masks/000001.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="byte check"):
        masks.verify_existing(tmp_path, episode, "video", "prompt")
    (tmp_path / "human_masks/000001.png").unlink()
    with pytest.raises(ValueError, match="directory"):
        masks.verify_existing(tmp_path, episode, "video", "prompt")


def test_resume_rejects_a_different_source_or_prompt(tmp_path):
    episode = complete_output(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        masks.verify_existing(tmp_path, episode, "other-video", "prompt")
    with pytest.raises(ValueError, match="prompt identity"):
        masks.verify_existing(tmp_path, episode, "video", "changed-prompt")


@pytest.mark.parametrize("indices", ["-1", "30", "0,0", "1,31"])
def test_episode_selection_rejects_invalid_or_repeated_indices(indices):
    with pytest.raises(ValueError):
        masks.parse_episodes(indices)


def test_pending_prompt_review_blocks_inference(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text("{}")
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"schema_version": "body_mask_prompts.v1",
                                  "raw_manifest_sha256": masks.sha256(raw),
                                  "visual_review": {"status": "pending"}}))
    with pytest.raises(ValueError, match="explicit visual review"):
        masks.load_prompts(prompts, raw, [0])


@pytest.mark.parametrize("count", [360, 634, 790, 877])
def test_qa_always_includes_exact_gate_midpoint_and_endpoints(count):
    ids = masks.qa_frame_ids(count, [])
    assert {0, count // 2, count - 1} <= set(ids)
    assert len(ids) >= 9
    assert ids == sorted(set(ids))


def test_qa_preserves_suspicious_frames_in_addition_to_uniform_samples():
    ids = masks.qa_frame_ids(634, [{"frame_id": 123, "review_flags": ["empty_mask"]},
                                  {"frame_id": 124, "review_flags": []}])
    assert 123 in ids
    assert {0, 317, 633} <= set(ids)
