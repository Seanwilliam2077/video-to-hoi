"""Run the reviewed official Track 1 packer and verify its serialized arrays.

Without --commit, the output is an explicitly marked format-check draft using
the organizer's sample placeholder. It must not be uploaded as a submission.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--kit", type=Path, default=Path("third_party/official-track1-kit"))
    p.add_argument("--commit")
    a = p.parse_args()
    kit = a.kit.resolve()
    sys.path.insert(0, str(kit))
    from v2dlb import mhr_submission as ms
    spec = importlib.util.spec_from_file_location("official_packer", kit / "tools/pack_reconstruction.py")
    packer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packer)
    sample = pd.read_parquet(kit / "data/track_1_sample_submission.parquet")
    sample = sample[sample.row_id.str.startswith(f"t1_{a.episode:06d}_")].copy()
    if sample.empty:
        raise ValueError("Episode is absent from the official Track 1 row manifest")
    sample_path = a.episodes / "official-episode-rows.parquet"
    sample[["row_id"]].to_parquet(sample_path, index=False)
    output = a.episodes / ("track1-episode.parquet" if a.commit else "FORMAT_CHECK_ONLY.parquet")
    commit = a.commit or str(sample.code_commit_url.iloc[0])
    packer.pack_track1(SimpleNamespace(episodes=a.episodes, sample=sample_path, commit=commit, out=output))
    actual = pd.read_parquet(output)
    if list(actual.columns) != ["row_id", "x", "y", "z", "code_commit_url"] or actual.row_id.tolist() != sample.row_id.tolist():
        raise ValueError("Serialized columns/row identities differ from the official sample")
    parts = actual.row_id.str.split("_", expand=True)
    actual["frame_index"] = parts[2].astype(int)
    actual["array_name"] = parts[3].astype(int).map(dict(enumerate(ms.array_order())))
    actual["point_index"] = parts[4].astype(int)
    arrays = ms._to_arrays(actual, ["x", "y", "z"])
    frames = sorted(set(actual.frame_index) - {ms.EPISODE_FRAME})
    source = np.load(a.episodes / f"episode_{a.episode:06d}.npz", allow_pickle=False)
    for dest, key, count in (("mhr_pose", "pose", 136), ("mhr_scales", "scales", 68), ("mhr_shape", "shape", 45)):
        unpacked = ms._unpack(arrays[dest], count, dest)
        expected = source[key][frames] if key == "pose" else source[key]
        if not np.array_equal(unpacked, expected):
            raise ValueError(f"Packed {key} values changed")
    if not np.array_equal(arrays["object_rotation"], source["object_rotation"][frames]):
        raise ValueError("Packed object rotation changed")
    if not np.array_equal(arrays["object_translation"][:,0], source["object_translation"][frames]):
        raise ValueError("Packed object translation changed")
    mesh = trimesh.load(a.episodes / f"episode_{a.episode:06d}_object.glb", force="mesh", process=False)
    packed_vertices = arrays["object_mesh_vertices"]
    packed_faces = arrays["object_mesh_faces"].astype(np.int64)
    triangles = packed_vertices[packed_faces]
    active = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1) > 0
    surface_vertices = packed_vertices[np.unique(packed_faces[active])]
    error = max(cKDTree(mesh.vertices).query(surface_vertices)[0].max(), cKDTree(surface_vertices).query(mesh.vertices)[0].max())
    if error > 1e-6:
        raise ValueError(f"Official mesh budgeting changed geometry by {error} m")
    scale = ms._object_scale(arrays["object_scale"])
    if scale != float(source["object_scale"]):
        raise ValueError("Packed object scale changed")
    report = {"format_roundtrip": "PASS", "episode": a.episode, "rows": len(actual),
        "video_frames_exported": len(source["pose"]), "official_scored_frames": len(frames),
        "native_parameters_bit_exact": True, "object_mesh_error_m": float(error),
        "ready_for_submission": False, "placeholder_commit": a.commit is None,
        "scope": "One episode only; all 30 episodes and a published generating commit are required for submission"}
    (a.episodes / "official-format-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
