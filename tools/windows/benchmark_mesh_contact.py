"""Time the unchanged contact-distance kernel on dense and budgeted surfaces.

Uses seeded independent synthetic query points near a Track 1 generated object.
This is a kernel benchmark, not video inference or reconstruction accuracy.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import numpy as np
import torch
import trimesh
from v2hoi.native import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepared = json.loads(a.preparation.read_text())
    receipt = json.loads((a.preparation.parent / prepared["mesh_budget"]["receipt"]["path"]).read_text())
    from v2hoi.mesh_budget import verify_budgeted_preparation
    verify_budgeted_preparation(prepared, a.preparation.parent, receipt["full_mesh"]["sha256"], allowed=True)
    source = Path("third_party/CARI4D_MHR_Win/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py")
    code = source.read_text()
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "_point_triangle_surface_distances")
    function = ast.get_source_segment(code, node)
    namespace = {"Any": Any, "_torch": lambda: torch}
    exec(compile(function, str(source), "exec"), namespace)
    kernel = namespace[node.name]
    full = trimesh.load(a.preparation.parent / receipt["full_mesh"]["path"], force="mesh", process=False)
    generator = np.random.default_rng(4096)
    query = generator.uniform(full.bounds[0] - .01, full.bounds[1] + .01, (360, 2, 256, 3)).astype(np.float32)
    results, outputs = {}, {}
    for name, artifact in (("dense", receipt["full_mesh"]), ("budget4096", receipt["output_mesh"])):
        path = a.preparation.parent / artifact["path"]
        mesh = trimesh.load(path, force="mesh", process=False)
        vertices = torch.as_tensor(np.asarray(mesh.vertices).copy(), dtype=torch.float32, device="cuda")
        faces = torch.as_tensor(np.asarray(mesh.faces).copy(), dtype=torch.long, device="cuda")
        times = []
        for iteration in range(4):
            points = torch.tensor(query, device="cuda", requires_grad=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            distance = kernel(points, vertices, faces)
            distance.min(dim=-1).values.square().mean().backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            if not torch.isfinite(distance).all() or not torch.isfinite(points.grad).all():
                raise ValueError("Contact distance or query gradients are nonfinite")
            if iteration:
                times.append(elapsed)
        outputs[name] = distance.detach().cpu().numpy()
        results[name] = {"faces": len(mesh.faces), "vertices": len(mesh.vertices),
            "sha256": sha256(path), "forward_backward_seconds": times, "median_seconds": float(np.median(times))}
        print(f"{name}: {results[name]['median_seconds']:.4f} seconds", flush=True)
        del vertices, faces, points, distance
        torch.cuda.empty_cache()
    report = {"scope": "Unchanged distance/min/square/backward kernel with seeded synthetic queries; not end-to-end speedup or accuracy",
        "provenance": "Track 1 generated surfaces; independent synthetic query points",
        "queries": list(query.shape), "function_sha256": hashlib.sha256(function.encode()).hexdigest(),
        "source_sha256": sha256(source), "results": results,
        "kernel_speedup": results["dense"]["median_seconds"] / results["budget4096"]["median_seconds"],
        "synthetic_distance_change_mean_m": float(np.abs(outputs["dense"]-outputs["budget4096"]).mean())}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
