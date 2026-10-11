"""Compare original and checkpointed rendering losses/gradients on synthetic cubes."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from learning.training.mhr_opt_refineout import MHRParityPostOptimizer
from v2hoi.native import sha256
import inspect


def main():
    model = object.__new__(MHRParityPostOptimizer)
    model.cfg = SimpleNamespace(w_silhouette=1)
    model.glctx = dr.RasterizeCudaContext()
    mesh = trimesh.creation.box(extents=(.4, .4, .4))
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
    model.object_silhouette_tensors = {"faces": torch.tensor(mesh.faces, dtype=torch.int32, device="cuda"),
        "pos": vertices, "vertex_color": (vertices + .2) / .4}
    n = 25  # Includes a partial final batch; independently constructed geometry.
    model.object_mask = torch.zeros(n, 64, 64, device="cuda")
    model.object_mask[:, 20:42, 20:42] = 1
    model.human_mask = torch.zeros_like(model.object_mask)
    model.human_mask[1::2, :, :28] = 1
    model.postopt_K_rois = torch.tensor([[75., 0, 32.], [0, 75., 32.], [0, 0, 1.]], device="cuda")[None].repeat(n, 1, 1)
    world = vertices[None].repeat(n, 1, 1)
    world[:, :, 2] += 1.5
    world[:, :, 0] += torch.linspace(-.1, .1, n, device="cuda")[:, None]
    reference_vertices = world.clone().requires_grad_()
    chunked_vertices = world.clone().requires_grad_()
    indices = torch.arange(n, device="cuda")
    reference = model._silhouette_loss_unchunked(reference_vertices, indices)
    reference.backward()
    actual = model._silhouette_loss(chunked_vertices, indices)
    actual.backward()
    assert torch.allclose(reference, actual, atol=1e-4, rtol=1e-6)
    assert torch.allclose(reference_vertices.grad, chunked_vertices.grad, atol=1e-4, rtol=1e-6)
    assert reference_vertices.grad.abs().sum() > 0 and torch.isfinite(chunked_vertices.grad).all()
    result = {"module_sha256": sha256(Path(inspect.getfile(MHRParityPostOptimizer))),
        "checker_sha256": sha256(Path(__file__)),
        "frames": n, "batch": 16, "loss_difference": float(abs(reference-actual).detach()),
        "gradient_max_difference": float(abs(reference_vertices.grad-chunked_vertices.grad).max()),
        "loss_bit_exact": torch.equal(reference, actual),
        "gradient_bit_exact": torch.equal(reference_vertices.grad, chunked_vertices.grad),
        "result": "PASS", "scope": "Independent synthetic cubes/colors/masks; includes occlusion and a partial final batch. No challenge reference data."}
    # Also preserve the actual constant-white upstream silhouette behavior.
    # Its renderer has no antialias operation, so that case has zero vertex
    # gradients; do not silently add an algorithmic renderer change.
    model.object_silhouette_tensors["vertex_color"] = torch.ones_like(vertices)
    white_reference = world.clone().requires_grad_()
    white_chunked = world.clone().requires_grad_()
    a = model._silhouette_loss_unchunked(white_reference, indices)
    b = model._silhouette_loss(white_chunked, indices)
    a.backward(); b.backward()
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-6)
    assert torch.equal(white_reference.grad, white_chunked.grad)
    result["constant_white_gradient_sum"] = float(white_reference.grad.abs().sum())
    path = Path("artifacts/windows-migration/silhouette-batching.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
