"""Renderer evidence isolation with independent synthetic images and a CV stub.

CI intentionally needs neither OpenCV nor model inference. The actual renderer
functions run against a small raster/text stub; production OpenCV is checked
separately on the real saved frame by the episode reviewer.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


class RasterCV:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 0

    def __init__(self):
        self.labels = []

    def line(self, image, first, last, color, *args):
        length = max(abs(last[0] - first[0]), abs(last[1] - first[1])) + 1
        for x, y in np.rint(np.linspace(first, last, length)).astype(int):
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image[y, x] = color

    def circle(self, image, point, radius, color, *args):
        x, y = point
        for row in range(max(0, y-radius), min(image.shape[0], y+radius+1)):
            for column in range(max(0, x-radius), min(image.shape[1], x+radius+1)):
                if (row-y)**2 + (column-x)**2 <= radius**2:
                    image[row, column] = color

    def getTextSize(self, text, font, scale, thickness):
        return (round(len(text) * 16 * scale), round(16 * scale)), 2

    def putText(self, image, text, origin, font, scale, color, *args):
        width = self.getTextSize(text, font, scale, 1)[0][0]
        assert origin[0] + width <= image.shape[1] - 6
        self.labels.append(text)
        image[max(0, origin[1]-2):origin[1], origin[0]:origin[0]+width] = color


@pytest.fixture
def renderer(monkeypatch):
    cv = RasterCV()
    with monkeypatch.context() as isolated:
        isolated.setitem(sys.modules, "cv2", cv)
        isolated.syspath_prepend(str(ROOT / "tools"))
        spec = importlib.util.spec_from_file_location("side_isolated_hand_renderer", ROOT / "tools/compare_human_hands.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, cv


def test_overlapping_crops_do_not_copy_right_hand_observations_into_empty_left_row(renderer):
    module, _ = renderer
    source = np.full((32, 32, 3), 10, dtype=np.uint8)
    points = np.full((2, 21, 2), 16., dtype=np.float32)
    weights = np.zeros((2, 21), dtype=np.float32)
    weights[1] = 1
    original_points, original_weights = points.copy(), weights.copy()
    left, left_count = module.observations_for_side(source, points, weights, 0)
    right, right_count = module.observations_for_side(source, points, weights, 1)
    assert left_count == 0
    assert right_count == 21
    assert np.array_equal(left, source)
    assert not np.array_equal(right, source)
    assert np.array_equal(points, original_points)
    assert np.array_equal(weights, original_weights)
    assert np.all(source == 10)


def test_two_line_header_stays_inside_256_pixel_cell_with_side_and_role(renderer):
    module, cv = renderer
    canvas = np.zeros((80, 768, 3), dtype=np.uint8)
    module.panel_header(canvas, 256, 20, 256, "Left", "MediaPipe hand candidate with a deliberately long display label", 0)
    assert np.any(canvas[20:60, 256:512])
    assert not np.any(canvas[:, :256])
    assert not np.any(canvas[:, 512:])
    assert cv.labels[0].startswith("Left hand | 0/21")
    assert cv.labels[1].startswith("MediaPipe hand candidate")


def test_long_episode_snapshots_cover_all_time_ranges_and_both_endpoints(renderer):
    module, _ = renderer
    assert sorted(module.snapshot_frame_ids(877, 11)) == list(range(0, 877, 73))
    assert module.snapshot_frame_ids(5, 0) == set(range(5))
    assert {110, 285, 359} <= module.snapshot_frame_ids(360, 16)
