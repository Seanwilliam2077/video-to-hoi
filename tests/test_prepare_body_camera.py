"""Independent synthetic checks for camera coordinate and consistency gates."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("prepare_body_camera", Path(__file__).parents[1] / "tools" / "prepare_body_camera.py")
camera = importlib.util.module_from_spec(spec)
spec.loader.exec_module(camera)


def test_sampling_includes_both_endpoints_and_every_temporal_interval():
    ids = camera.sampled_frame_ids(1000, 13)
    assert len(set(ids)) == 13
    assert ids[0] == 0 and ids[-1] == 999
    assert max(b - a for a, b in zip(ids, ids[1:])) <= 84
    assert min(b - a for a, b in zip(ids, ids[1:])) >= 83


@pytest.mark.parametrize("count,samples", [(8, 9), (100, 8), (100, 16), (False, 9)])
def test_invalid_sample_count_is_rejected(count, samples):
    with pytest.raises(ValueError):
        camera.sampled_frame_ids(count, samples)


def test_uv_conversion_preserves_pixel_center_projection():
    matrix = [[0.8, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]]
    result = camera.original_pixel_intrinsics(matrix, 1000, 800)
    assert result == {"fx": 800.0, "fy": 800.0, "cx": 499.5, "cy": 399.5, "width": 1000, "height": 800}
    # x/z = 0.2 -> UV u=.66 -> integer-centered pixel u*W-.5.
    assert result["fx"] * 0.2 + result["cx"] == pytest.approx(659.5)


@pytest.mark.parametrize("matrix", [
    [[math.nan, 0, 0.5], [0, 1, 0.5], [0, 0, 1]],
    [[-1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]],
    [[1, 0.1, 0.5], [0, 1, 0.5], [0, 0, 1]],
    [[1, 0, 1.5], [0, 1, 0.5], [0, 0, 1]],
    [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 0]],
])
def test_invalid_normalized_camera_does_not_become_ready(matrix):
    with pytest.raises(ValueError):
        camera.original_pixel_intrinsics(matrix, 1000, 800)


def estimates(focals):
    return [{"fx": f, "fy": f, "cx": 499.5, "cy": 399.5, "width": 1000, "height": 800} for f in focals]


def test_stable_estimates_preserve_nonzero_spread():
    intrinsics, diagnostics, blockers = camera.aggregate_estimates(estimates([790, 792, 796, 798, 800, 803, 806, 808, 810]))
    assert intrinsics["fx"] == 800
    assert diagnostics["fx"]["minimum"] == 790
    assert diagnostics["fx"]["median_absolute_deviation"] > 0
    assert blockers == []


def test_single_large_outlier_blocks_even_when_median_stable():
    intrinsics, diagnostics, blockers = camera.aggregate_estimates(estimates([800] * 12 + [1600]))
    assert intrinsics["fx"] == 800
    assert diagnostics["fx"]["maximum_relative_deviation"] == 1.0
    assert any("maximum deviation" in blocker for blocker in blockers)


def test_broad_spread_blocks_without_extreme_outlier():
    _, diagnostics, blockers = camera.aggregate_estimates(estimates([650, 675, 700, 750, 800, 850, 900, 925, 950]))
    assert diagnostics["fx"]["maximum_relative_deviation"] < 0.4
    assert any("p10-p90" in blocker for blocker in blockers)


def test_different_image_grids_are_never_pooled():
    rows = estimates([800] * 9)
    rows[-1]["width"] = 500
    with pytest.raises(ValueError, match="different pixel grids"):
        camera.aggregate_estimates(rows)


def receipt_fixture(tmp_path):
    module = tmp_path / "runtime.py"
    module.write_text("value = 1\n", encoding="utf-8")
    receipt = {"repository": "microsoft/MoGe", "revision": camera.SOURCE_PIN,
               "files": [{"path": "runtime.py", "bytes": module.stat().st_size,
                          "sha256": camera.sha256(module)}]}
    (tmp_path / "source-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return module


def test_filtered_source_receipt_verifies_runtime_bytes(tmp_path):
    receipt_fixture(tmp_path)
    result = camera.verify_checkout(tmp_path, camera.SOURCE_PIN)
    assert result["verified_file_count"] == 1
    assert result["revision"] == camera.SOURCE_PIN


def test_changed_source_receipt_file_is_rejected(tmp_path):
    module = receipt_fixture(tmp_path)
    module.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="verification failed"):
        camera.verify_checkout(tmp_path, camera.SOURCE_PIN)


def test_unlisted_python_module_is_rejected(tmp_path):
    receipt_fixture(tmp_path)
    (tmp_path / "extra.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unlisted"):
        camera.verify_checkout(tmp_path, camera.SOURCE_PIN)


def test_portable_source_identity_excludes_producer_path():
    source = {"path": "C:/Users/private-person/project/source", "revision": camera.SOURCE_PIN,
              "verification": "source_receipt_file_sha256", "source_receipt_sha256": "a" * 64}
    portable = camera.portable_source_identity(source, "microsoft/MoGe")
    assert "path" not in portable
    assert portable["repository"] == "microsoft/MoGe"
    assert portable["revision"] == camera.SOURCE_PIN
    assert portable["source_receipt_sha256"] == "a" * 64
    assert source["path"] == "C:/Users/private-person/project/source"
