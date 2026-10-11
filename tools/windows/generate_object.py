"""Generate the object with NVIDIA's SAM3D entry; preserve the generated surface."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import trimesh
from v2hoi.native import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=Path("weights/sam3d"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-offload", action="store_true",
                   help="Move unchanged models between CPU and CUDA at stage boundaries")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    # Use the upstream SAM3D/MoGe API versions in this process only. The CARI4D
    # process keeps its separate pinned MoGe2/utils3d imports in the same venv.
    for name in ("sam-3d-objects", "moge-sam3d", "utils3d-sam3d"):
        sys.path.insert(0, str(root / "third_party" / name))
    os.environ["LIDRA_SKIP_INIT"] = "1"
    os.environ["ATTN_BACKEND"] = "sdpa"
    os.environ["V2HOI_SAM3D_CPU_OFFLOAD"] = "1" if a.cpu_offload else "0"
    from v2d.sam3d.lib.image_to_mesh import image_to_mesh
    record = json.loads((a.inputs / "inputs.json").read_text())
    if record["provenance"] != "track1":
        raise ValueError("Only permitted Track 1 inputs may generate challenge objects")
    for name, digest in record["files"].items():
        if sha256(a.inputs / name) != digest:
            raise ValueError(f"Object-generation input changed: {name}")
    if (a.output / "generation.json").exists():
        raise FileExistsError("Use a new output to preserve an existing generated candidate")
    a.output.mkdir(parents=True, exist_ok=True)
    raw = a.output / "sam3d-raw.glb"
    transform = a.output / "sam3d-transform.json"
    image_to_mesh(str(a.inputs / "rgb.png"), str(a.inputs / "mask.png"), str(raw),
        str(transform), str(a.output / "sam3d-intrinsics.json"), str(a.weights.resolve()),
        depth_path=str(a.inputs / "depth.png"),
        depth_intrinsics_path=str(a.inputs / "intrinsics.json"),
        depth_mask_path=str(a.inputs / "mask.png"))
    pose = json.loads(transform.read_text())
    scale = np.asarray(pose["scale"], dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Official SAM3D returned an invalid scene scale")
    scene = trimesh.load(raw, force="scene", process=False)
    # Bake only the official returned scale into the canonical mesh. Its rigid
    # pose is estimated by FoundationPose; never convexify/simplify the surface.
    matrix = np.eye(4); matrix[:3, :3] = np.diag(scale)
    scene.apply_transform(matrix)
    mesh = a.output / "object.glb"
    scene.export(mesh)
    merged = scene.dump(concatenate=True)
    reread = trimesh.load(mesh, force="scene", process=False).dump(concatenate=True)
    if not np.isfinite(merged.vertices).all() or not np.allclose(merged.vertices, reread.vertices, atol=1e-6):
        raise ValueError("Generated mesh changed during serialization")
    result = {"provenance": "track1", "entry": "v2d.sam3d.lib.image_to_mesh.image_to_mesh",
        "cpu_offload": a.cpu_offload, "wrapper_sha256": sha256(Path(__file__)),
        "offload_adapter_sha256": sha256(root / "src/v2hoi/sam3d_offload.py") if a.cpu_offload else None,
        "input_record": record, "input_record_sha256": sha256(a.inputs / "inputs.json"),
        "raw_mesh_sha256": sha256(raw), "mesh_sha256": sha256(mesh),
        "returned_transform": pose, "baked_scale": scale.tolist(),
        "vertices": len(merged.vertices), "faces": len(merged.faces),
        "nvidia_entry_defaults": {"seed": None, "stage1_only": False,
            "with_mesh_postprocess": False, "with_texture_baking": False,
            "with_layout_postprocess": False, "use_vertex_color": True,
            "stage1_inference_steps": None},
        "surface_replacement": False, "sampler_step_override": False,
        "note": "Original generated surface with the official returned metric scale; no depth hull or proxy mesh"}
    (a.output / "generation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
