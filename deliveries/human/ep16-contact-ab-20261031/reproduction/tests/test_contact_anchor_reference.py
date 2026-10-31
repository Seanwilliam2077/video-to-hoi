"""Synthetic native metadata; no challenge assets, videos or model weights."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def validator():
    path = Path(__file__).resolve().parents[1] / "tools/refine_human_hands.py"
    spec = importlib.util.spec_from_file_location("contact_anchor_metadata_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_contact_anchor_reference


def fixture():
    camera = np.array([[800., 0, 700], [0, 800, 500], [0, 0, 1]], np.float32)
    baseline = {"pose": np.zeros((8, 136), np.float32), "shape": np.zeros(45, np.float32),
                "scales": np.zeros(68, np.float32), "frame_indices": np.arange(8), "fps": np.asarray(30.)}
    reference = copy.deepcopy(baseline)
    reference["pose"][:, 68] = 0.1
    reference["report"] = {"track": 1, "episode": 16, "source_video_sha256": "synthetic-video",
                           "mhr_model_sha256": "synthetic-decoder", "camera_intrinsics": camera.tolist(),
                           "parameter_format": "native_mhr_pose136_scales68_shape45"}
    manifest = {"track": 1, "episode": 16, "video_frames": 8, "video_sha256": "synthetic-video", "video_fps": 30.}
    return reference, baseline, manifest, camera


def test_valid_reference_is_read_only_and_keeps_original_seed(validator):
    reference, baseline, manifest, camera = fixture()
    before_reference, before_baseline = copy.deepcopy(reference), copy.deepcopy(baseline)
    report = validator(reference, baseline, manifest, "synthetic-decoder", camera)
    assert report["same_native_decoder"] and report["exact_camera_intrinsics_match"]
    assert report["optimization_initialization_changed"] is False
    np.testing.assert_array_equal(reference["pose"], before_reference["pose"])
    np.testing.assert_array_equal(baseline["pose"], before_baseline["pose"])
    assert not np.array_equal(reference["pose"], baseline["pose"])


@pytest.mark.parametrize("key,value", [("track", 2), ("episode", 11), ("source_video_sha256", "different")])
def test_reference_must_identify_the_same_permitted_source(validator, key, value):
    reference, baseline, manifest, camera = fixture()
    reference["report"][key] = value
    with pytest.raises(ValueError, match="same Track 1 video"):
        validator(reference, baseline, manifest, "synthetic-decoder", camera)


@pytest.mark.parametrize("name", ["shape", "scales"])
def test_reference_cannot_change_identity_or_skeleton_scale(validator, name):
    reference, baseline, manifest, camera = fixture()
    reference[name][0] = 0.01
    with pytest.raises(ValueError, match="changed fixed"):
        validator(reference, baseline, manifest, "synthetic-decoder", camera)


@pytest.mark.parametrize("control", [0, 3])
def test_reference_cannot_realign_root_translation_or_rotation(validator, control):
    reference, baseline, manifest, camera = fixture()
    reference["pose"][4, control] = 0.02
    with pytest.raises(ValueError, match="root controls"):
        validator(reference, baseline, manifest, "synthetic-decoder", camera)


@pytest.mark.parametrize("change", ["shift", "missing"])
def test_reference_requires_complete_original_frame_ids(validator, change):
    reference, baseline, manifest, camera = fixture()
    if change == "shift":
        reference["frame_indices"] += 1
    else:
        del reference["frame_indices"]
    with pytest.raises(ValueError, match="complete original frame IDs"):
        validator(reference, baseline, manifest, "synthetic-decoder", camera)


def test_reference_decoder_identity_is_required(validator):
    reference, baseline, manifest, camera = fixture()
    with pytest.raises(ValueError, match="different native decoder"):
        validator(reference, baseline, manifest, "another-decoder", camera)


@pytest.mark.parametrize("change", ["missing", "different"])
def test_reference_camera_report_must_exactly_match(validator, change):
    reference, baseline, manifest, camera = fixture()
    if change == "missing":
        del reference["report"]["camera_intrinsics"]
    else:
        reference["report"]["camera_intrinsics"][0][0] = 801.
    with pytest.raises(ValueError, match="camera differs"):
        validator(reference, baseline, manifest, "synthetic-decoder", camera)


def test_reference_native_representation_tag_is_required(validator):
    reference, baseline, manifest, camera = fixture()
    reference["report"]["parameter_format"] = "unverified-array-layout"
    with pytest.raises(ValueError, match="native MHR"):
        validator(reference, baseline, manifest, "synthetic-decoder", camera)


def test_reference_original_frame_rate_is_required(validator):
    reference, baseline, manifest, camera = fixture()
    reference["fps"] = np.asarray(24.)
    with pytest.raises(ValueError, match="frame rate"):
        validator(reference, baseline, manifest, "synthetic-decoder", camera)
