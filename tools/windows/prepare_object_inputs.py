"""Prepare lossless SAM3D inputs using the official human-anchored depth rule."""
import argparse
import json
from pathlib import Path
import cv2
import h5py
import joblib
import numpy as np
from PIL import Image
import torch
import nvdiffrast.torch as dr
from v2d.common.datatypes import DepthImage
from prep.align_utils import compute_scale_robust_log
from prep.align_depth_to_mhr_wild import _render_human_depth
from v2hoi.native import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--raw-depth", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, required=True)
    p.add_argument("--human-mesh", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    seq = a.video.name.removesuffix(".0.color.mp4")
    with h5py.File(a.masks) as f:
        if f.attrs["source_video_sha256"] != sha256(a.video) or f.attrs.get("fill_hole_area") != 8:
            raise ValueError("Require official SAM2 masks for this exact video")
        human_mask = f[f"{seq}/000000-k0.person_mask.png"][()].astype(bool)
        object_mask = f[f"{seq}/000000-k0.obj_rend_mask.png"][()].astype(bool)
    intrinsics = joblib.load(a.intrinsics)
    k = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                  [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]], dtype=np.float32)
    raw = np.load(a.raw_depth, mmap_mode="r")[0].astype(np.float32) / 1000
    human = torch.load(a.human_mesh, map_location="cpu", weights_only=False)
    vertices = human["pred_vertices"].numpy().astype(np.float32)
    faces = human["faces"].to(device="cuda", dtype=torch.int32)
    depth = _render_human_depth(vertices, k, raw.shape, dr.RasterizeCudaContext(), {"faces": faces})[0]
    valid = human_mask & np.isfinite(raw) & np.isfinite(depth) & (raw > .001) & (depth > .001)
    if valid.sum() < 100:
        raise ValueError("Insufficient corresponding human/depth pixels")
    scale = float(compute_scale_robust_log(raw, depth, valid))
    aligned = raw.copy()
    aligned[aligned > .001] *= scale
    DepthImage(aligned).save(str(a.output / "depth.png"))
    video = cv2.VideoCapture(str(a.video)); ok, bgr = video.read(); video.release()
    if not ok:
        raise ValueError("Missing original frame zero")
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(a.output / "rgb.png")
    Image.fromarray(object_mask.astype(np.uint8) * 255).save(a.output / "mask.png")
    (a.output / "intrinsics.json").write_text(json.dumps({"fx": intrinsics["fx"], "fy": intrinsics["fy"],
        "cx": intrinsics["cx"], "cy": intrinsics["cy"], "width": raw.shape[1], "height": raw.shape[0]}, indent=2))
    report = {"provenance": "track1", "bootstrap_frame_id": 0, "video_sha256": sha256(a.video),
        "scale": scale, "valid_pixels": int(valid.sum()), "human_mesh_sha256": sha256(a.human_mesh),
        "mask_sha256": sha256(a.masks), "depth_rule": "Official compute_scale_robust_log + _render_human_depth",
        "rgb_encoding": "Lossless PNG of original decoded frame", "object_geometry": "Not generated at this stage",
        "files": {name: sha256(a.output / name) for name in ("rgb.png", "mask.png", "depth.png", "intrinsics.json")}}
    (a.output / "inputs.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
