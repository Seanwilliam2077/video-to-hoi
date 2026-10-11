"""Check the relocated Windows environment without model downloads or inference."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = root / ".venv"
    assert not environment.is_junction() and not environment.is_symlink()
    assert Path(sys.prefix).resolve() == environment.resolve(), sys.prefix

    import torch
    from pytorch3d.ops import knn_points
    from pytorch3d.renderer.mesh import rasterize_meshes
    from pytorch3d.structures import Meshes
    import nvdiffrast.torch as dr

    assert torch.cuda.is_available()
    device = "cuda"
    checks = []
    x = torch.randn(32, 32, device=device, requires_grad=True)
    (x @ x.T).square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    checks.append("torch CUDA forward/backward")

    from sam2 import _C
    from sam2.utils.misc import fill_holes_in_mask_scores
    masks = torch.ones(1, 1, 32, 32, device=device)
    masks[:, :, 8, 8] = -1
    masks[:, :, 16:19, 16:19] = -1
    filled = fill_holes_in_mask_scores(masks, 8)
    assert filled[0, 0, 8, 8] > 0 and (filled[0, 0, 16:19, 16:19] < 0).all()
    checks.append("Original SAM2 CUDA hole filling at default area 8")

    p = torch.randn(1, 32, 3, device=device, requires_grad=True)
    q = torch.randn(1, 40, 3, device=device)
    knn_points(p, q, K=3).dists.sum().backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    checks.append("PyTorch3D CUDA KNN forward/backward")

    v = torch.tensor([[-.8, -.8, 1.], [.8, -.8, 1.], [0., .8, 1.]],
                     device=device, requires_grad=True)
    faces = torch.tensor([[0, 1, 2]], device=device)
    pixels, z, _, _ = rasterize_meshes(Meshes([v], [faces]), image_size=32,
                                      faces_per_pixel=1, bin_size=0)
    assert (pixels >= 0).any()
    z[pixels >= 0].sum().backward()
    assert torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0
    checks.append("PyTorch3D CUDA rasterizer forward/backward")

    pos = torch.tensor([[[-.8, -.8, 0., 1.], [.8, -.8, 0., 1.], [0., .8, 0., 1.]]],
                       device=device, requires_grad=True)
    tri = faces.to(torch.int32)
    colors = torch.eye(3, device=device)[None].requires_grad_()
    rast, _ = dr.rasterize(dr.RasterizeCudaContext(), pos, tri, resolution=[32, 32])
    im, _ = dr.interpolate(colors, rast, tri)
    dr.antialias(im, rast, pos, tri).sum().backward()
    assert (rast[..., 3] > 0).any()
    assert torch.isfinite(pos.grad).all() and torch.isfinite(colors.grad).all()
    assert colors.grad.abs().sum() > 0
    checks.append("nvdiffrast CUDA forward/backward")

    imported_sources = {}
    for module in ["v2d.cari4d.lib", "v2d.sam3d_body.lib", "v2d.moge.lib"]:
        spec = importlib.util.find_spec(module)
        assert spec is not None
        paths = list(spec.submodule_search_locations or [])
        assert paths and all(Path(p).resolve().is_relative_to(root) for p in paths)
        imported_sources[module] = paths
    torch.cuda.synchronize()
    report = {"python": sys.version, "environment": str(environment),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "checks": checks, "editable_sources": imported_sources,
              "full_model_inference": False}
    output = root / "artifacts/windows-migration/runtime-check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
