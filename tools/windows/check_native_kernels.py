"""Independent synthetic penetration-gradient and depth-warp batching checks."""
import json
import os
from pathlib import Path
import tempfile
import copy
import numpy as np
import torch
import trimesh
from lib_mhr.sdf import object_inside_human_penetration_loss
from prep.mhr_depth_h5 import MHRDepthH5Writer


def main():
    mesh = trimesh.creation.box()
    vertices = torch.tensor(mesh.vertices[None], device="cuda", dtype=torch.float32, requires_grad=True)
    points = torch.tensor([[[.2, 0., 0.], [.8, 0., 0.]]], device="cuda", requires_grad=True)
    loss = object_inside_human_penetration_loss(vertices, mesh.faces, points)
    loss.backward()
    assert abs(loss.item() - .15) < 1e-5
    assert torch.allclose(points.grad, points.new_tensor([[[-.5, 0., 0.], [0., 0., 0.]]]), atol=1e-5)
    assert torch.isfinite(vertices.grad).all() and vertices.grad.abs().sum() > 0
    with tempfile.TemporaryDirectory() as directory:
        with MHRDepthH5Writer(Path(directory) / "depth.h5", {"camera": ["000000"]}):
            pass
    from learning.foundationpose_data import FoundationPoseBatchData, FoundationPoseScoreDataProcessor
    synthetic = FoundationPoseBatchData(
        rgbAs=torch.zeros(9, 3, 16, 16, device="cuda"),
        xyz_mapAs=torch.ones(9, 3, 16, 16, device="cuda"),
        depthBs=torch.ones(1, 1, 16, 16, device="cuda"),
        poseA=torch.eye(4, device="cuda")[None].repeat(9, 1, 1),
        tf_to_crops=torch.diag(torch.tensor([.25, .25, 1.], device="cuda"))[None].repeat(9, 1, 1),
        Ks=torch.tensor([[50., 0., 32.], [0., 50., 32.], [0., 0., 1.]], device="cuda")[None].repeat(9, 1, 1),
        mesh_diameters=torch.ones(9, device="cuda"))
    processor = FoundationPoseScoreDataProcessor({"normalize_xyz": True})
    original_depth_batch = os.environ.get("FOUNDATIONPOSE_DEPTH_BATCH_SIZE")
    try:
        os.environ["FOUNDATIONPOSE_DEPTH_BATCH_SIZE"] = "128"
        full_depth = processor.transform_depth_to_xyzmap(copy.deepcopy(synthetic), 64, 64).xyz_mapBs
        os.environ["FOUNDATIONPOSE_DEPTH_BATCH_SIZE"] = "3"
        chunked_depth = processor.transform_depth_to_xyzmap(copy.deepcopy(synthetic), 64, 64).xyz_mapBs
        assert torch.equal(full_depth, chunked_depth)
    finally:
        if original_depth_batch is None:
            os.environ.pop("FOUNDATIONPOSE_DEPTH_BATCH_SIZE", None)
        else:
            os.environ["FOUNDATIONPOSE_DEPTH_BATCH_SIZE"] = original_depth_batch
    report = {"penetration_value_and_gradients": "PASS", "h5_lock": "PASS",
              "depth_warp_chunk_bit_exact": True,
              "penetration_equivalence_to_kaolin": "Not established; owner-permitted exception",
              "score_policy": "Unmodified upstream full forward; no scorer feature chunks",
              "scope": "Independent synthetic cube/points and depth warp; no challenge reference data"}
    path = Path("artifacts/windows-migration/native-kernels.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
