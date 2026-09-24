from pathlib import Path

import numpy as np

from v2hoi.pipeline import DEMO_CLIPS, DemoBackend, camera_to_world, run


def test_one_scale_per_object_and_locked_camera(tmp_path: Path):
    results = run(list(DEMO_CLIPS), DemoBackend(), tmp_path)
    by_object: dict[str, set[float]] = {}
    for result in results:
        by_object.setdefault(result.clip.object_name, set()).add(result.scale)
        saved = np.load(tmp_path / "episodes" / f"{result.clip.episode_index:06d}" / "identity.npy")
        assert saved.shape == (45,)
        assert result.intrinsics["width"] == 1536
    assert len(by_object["black_pan"]) == 1
    assert len(by_object["foam_grass_block"]) == 1
    pan_scale = json_scale(tmp_path, "black_pan")
    assert all(item.scale == pan_scale for item in results if item.clip.object_name == "black_pan")


def test_invisible_frame_stays_zero(tmp_path: Path):
    results = run(list(DEMO_CLIPS), DemoBackend(), tmp_path)
    pose = results[0].obj_pose7
    assert np.all(pose[1] == 0)
    assert pose[0, 2] != 0


def test_world_up_is_plus_y():
    T = camera_to_world(np.array([0.0, 0.0, 1.0]))
    up = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(up, [0, 1, 0], atol=1e-6)


def json_scale(root: Path, name: str) -> float:
    import json

    return json.loads((root / "objects" / name / "scale.json").read_text(encoding="utf-8"))["scale"]
