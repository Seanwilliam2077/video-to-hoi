"""Reconstruct a convex object surface from visible Track 1 depth samples.

Unseen surfaces are convex closure, not measured geometry. This deliberately
simple reconstruction is restricted to convex objects; inspect its result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import joblib
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--intrinsics", type=Path, required=True)
    p.add_argument("--rgb", type=Path, required=True)
    p.add_argument("--sequence", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--metric-scale", type=float, default=1.0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    K = joblib.load(a.intrinsics)
    depth = np.load(a.depth, mmap_mode="r")[a.frame].astype(np.float32) / 1000 * a.metric_scale
    with h5py.File(a.masks) as f:
        mask = f[f"{a.sequence}/{a.frame:06d}-k0.obj_rend_mask.png"][()].astype(np.uint8)
    # Remove mixed foreground/background pixels before backprojection.
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
    y, x = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    z = depth[y, x]
    valid = (z >= np.quantile(z, .01)) & (z <= np.quantile(z, .99))
    x, y, z = x[valid], y[valid], z[valid]
    points = np.column_stack(((x - K["cx"]) / K["fx"] * z, (y - K["cy"]) / K["fy"] * z, z))
    if len(points) < 100:
        raise ValueError("Insufficient visible object depth")
    center = np.median(points, axis=0)
    local = points - center
    hull = trimesh.convex.convex_hull(local)
    image = cv2.cvtColor(cv2.imread(str(a.rgb)), cv2.COLOR_BGR2RGB)
    _, nearest = cKDTree(local).query(hull.vertices)
    hull.visual.vertex_colors = image[y[nearest], x[nearest]]
    if not hull.is_watertight or hull.volume <= 0:
        raise ValueError("Invalid convex reconstruction")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    hull.export(a.output)
    report = {
        "schema": "v2hoi.visible-depth-convex-mesh.v1", "sequence": a.sequence,
        "original_frame_id": a.frame, "source": "Track 1 RGB + SAM2 mask + MoGe2 depth",
        "method": "Convex hull of eroded visible depth; hidden surfaces are convex closure",
        "known_limitations": ["Unobserved backside is not recovered", "Monocular depth shape uncertainty"],
        "metric_scale_from_human": a.metric_scale, "camera_center_m": center.tolist(),
        "vertices": len(hull.vertices), "faces": len(hull.faces),
        "extents_m": hull.extents.tolist(), "volume_m3": hull.volume,
        "mesh_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
        "inputs": {key: str(getattr(a, key).resolve()) for key in ("depth", "masks", "intrinsics", "rgb")},
    }
    a.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
