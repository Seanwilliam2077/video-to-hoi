"""Independent synthetic crop/receipt fixtures; no challenge assets."""
import hashlib
import json

import numpy as np
import pytest

from v2hoi.hand_crop_inputs import load_verified_crop_boxes


def fixture(tmp_path, boxes=None, **changes):
    if boxes is None:
        boxes = np.tile(np.array([10, 20, 74, 84]), (3, 2, 1))
    path = tmp_path / "boxes.npy"
    np.save(path, boxes)
    receipt = {"track": 1, "episode": 16, "frames": 3,
               "source_video_sha256": "a" * 64, "all30_execution_authorized": False,
               "crop_boxes_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "policy": {"source": "independent synthetic fixture"}}
    receipt.update(changes)
    record = tmp_path / "receipt.json"
    record.write_text(json.dumps(receipt), encoding="utf-8")
    return path, record


def load(paths, **changes):
    args = {"source_video_sha256": "a" * 64, "episode": 16,
            "frames": 3, "image_size_wh": [100, 100]}
    args.update(changes)
    return load_verified_crop_boxes(*paths, **args)


def test_shared_boxes_preserve_exact_original_frame_and_side_mapping(tmp_path):
    boxes = np.tile([10, 20, 74, 84], (3, 2, 1))
    boxes[1, 0] = [12, 21, 76, 85]
    actual, receipt = load(fixture(tmp_path, boxes))
    np.testing.assert_array_equal(actual, boxes)
    assert receipt["policy"]["source"] == "independent synthetic fixture"


@pytest.mark.parametrize("changes", [
    {"track": 2}, {"episode": 11}, {"frames": 2},
    {"source_video_sha256": "b" * 64}, {"all30_execution_authorized": True},
    {"crop_boxes_sha256": "0" * 64}, {"side_order": ["right", "left"]},
])
def test_wrong_provenance_coverage_or_side_is_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        load(fixture(tmp_path, **changes))


@pytest.mark.parametrize("boxes", [
    np.zeros((3, 2, 4), dtype=int),
    np.tile([-1, 20, 63, 84], (3, 2, 1)),
    np.tile([10, 40, 74, 104], (3, 2, 1)),
    np.tile([10, 20, 74, 83], (3, 2, 1)),
    np.tile([10., 20., 74., 84.], (3, 2, 1)),
    np.tile([10, 20, 74, 84], (2, 2, 1)),
])
def test_invalid_geometry_or_timeline_cannot_reach_observer(tmp_path, boxes):
    with pytest.raises(ValueError):
        load(fixture(tmp_path, boxes))


def test_receipt_does_not_allow_changed_array_bytes(tmp_path):
    paths = fixture(tmp_path)
    np.save(paths[0], np.tile([11, 20, 75, 84], (3, 2, 1)))
    with pytest.raises(ValueError, match="identity"):
        load(paths)
