"""Metadata paths must stay under the selected dataset or mesh directory."""

import json

import numpy as np
import pandas as pd
import pytest

from v2hoi import dataset


def _root(tmp_path, *, data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", **meta):
    root = tmp_path / "pred"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"data_path": data_path}), encoding="utf-8")
    row = {"episode_index": 0, "object": "box", **meta}
    (root / "meta" / "episodes_metadata.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def fake_parquet(monkeypatch):
    frame = pd.DataFrame({
        "observation.object.visible": [True],
        "observation.object.pose": [np.array([0, 0, 0, 1, 0, 0, 0])],
        "observation.human.pose": [np.zeros(77 * 3)],
        "observation.human.translation": [np.zeros(3)],
        "observation.human.identity_coeffs": [np.zeros(45)],
        "observation.human.scale_params": [np.zeros(68)],
        "observation.human.bone_length_flexibles": [np.zeros(6)],
    })
    monkeypatch.setattr(dataset.pd, "read_parquet", lambda path: frame)


@pytest.mark.parametrize("outside", ["../outside.parquet", "absolute"])
def test_data_path_cannot_escape_root(tmp_path, outside):
    path = str(tmp_path / "outside.parquet") if outside == "absolute" else outside
    root = _root(tmp_path, data_path=path)
    with pytest.raises(ValueError, match="data_path"):
        dataset._parquet_path(root, 0)


@pytest.mark.parametrize("outside", ["../outside.glb", "absolute"])
def test_mesh_path_cannot_escape_root(tmp_path, fake_parquet, outside):
    path = str(tmp_path / "outside.glb") if outside == "absolute" else outside
    root = _root(tmp_path, mesh=path)
    with pytest.raises(ValueError, match="mesh"):
        dataset.load_episode(root, 0)


@pytest.mark.parametrize("outside", ["../outside.json", "absolute"])
def test_ground_plane_path_cannot_escape_root(tmp_path, fake_parquet, outside):
    path = str(tmp_path / "outside.json") if outside == "absolute" else outside
    root = _root(tmp_path, ground_plane=path)
    with pytest.raises(ValueError, match="ground_plane"):
        dataset.load_episode(root, 0)


@pytest.mark.parametrize("outside", ["../box", "absolute"])
def test_object_name_cannot_be_a_path(tmp_path, fake_parquet, outside):
    name = str(tmp_path / "box") if outside == "absolute" else outside
    root = _root(tmp_path, object=name)
    with pytest.raises(ValueError, match="object name"):
        dataset.load_episode(root, 0)


def test_explicit_mesh_dir_and_valid_ground_plane(tmp_path, fake_parquet):
    # An explicit mesh directory can be outside the prediction root. The
    # metadata mesh path is ignored when that override is supplied.
    root = _root(tmp_path, mesh="../ignored.glb", ground_plane="ground_plane/box.json")
    plane_path = root / "ground_plane" / "box.json"
    plane_path.parent.mkdir()
    plane_path.write_text(json.dumps({"plane": [0, 1, 0, -1]}), encoding="utf-8")
    mesh_dir = tmp_path / "shared_meshes"

    ep = dataset.load_episode(root, 0, mesh_dir=mesh_dir)

    assert ep.mesh_path == (mesh_dir / "box" / "box.glb").resolve()
    assert np.array_equal(ep.ground_plane, [0, 1, 0, -1])
    assert dataset._parquet_path(root, 0) == (root / "data/chunk-000/episode_000000.parquet").resolve()
