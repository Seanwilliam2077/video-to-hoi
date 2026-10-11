"""Create an explicit 4096-face inference variant; preserve full preparation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import trimesh
from v2hoi.native import sha256
from v2hoi.mesh_budget import POLICY, DEVIATION, verify_budgeted_preparation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New variant directory, never the original preparation")
    p.add_argument("--kit", type=Path, default=Path("third_party/official-track1-kit"))
    a = p.parse_args()
    source = a.preparation.resolve()
    prepared = json.loads(source.read_text())
    if (prepared.get("provenance") != "track1"
            or prepared.get("input_mesh_policy", "full-generated-surface") != "full-generated-surface"
            or prepared["mesh_generation"]["entry"] != "v2d.sam3d.lib.image_to_mesh.image_to_mesh"
            or prepared["segmentation"]["entry"] != "v2d.sam2.lib.video_to_masks.video_to_masks"):
        raise ValueError("Use an original generated Track 1 preparation, not a historical or already budgeted variant")
    for section, key in (("mesh_generation", "receipt"), ("segmentation", "report")):
        artifact = prepared[section][key]
        if sha256(source.parent / artifact["path"]) != artifact["sha256"]:
            raise ValueError("Preparation evidence changed")
    generation = json.loads((source.parent / prepared["mesh_generation"]["receipt"]["path"]).read_text())
    full_mesh = (source.parent / prepared["mesh"]["path"]).resolve()
    masks = (source.parent / prepared["masks"]["path"]).resolve()
    if (sha256(full_mesh) != generation["mesh_sha256"]
            or generation["mesh_sha256"] != prepared["mesh"]["sha256"]
            or sha256(masks) != prepared["masks"]["sha256"]):
        raise ValueError("Original generated geometry or masks changed")
    kit = a.kit.resolve()
    code = kit / "v2dlb/mesh_budget.py"
    manifest = json.loads(Path("deployment/windows/official-kit.json").read_text())
    expected = next(item["sha256"] for item in manifest["members"] if item["path"] == "v2dlb/mesh_budget.py")
    if sha256(code) != expected:
        raise ValueError("Organizer mesh budgeting source differs from the reviewed kit")
    sys.path.insert(0, str(kit))
    from v2dlb.mesh_budget import budget_mesh
    destination = a.output.resolve()
    if destination.exists():
        raise FileExistsError("Use a new directory to preserve previous variants")
    destination.mkdir(parents=True)
    started = time.perf_counter()
    vertices, faces = budget_mesh(full_mesh, faces=4096, vertices=4096)
    triangles = vertices[faces]
    active = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) > 0
    mesh = trimesh.Trimesh(vertices, faces[active], process=False)
    mesh.remove_unreferenced_vertices()
    output_mesh = destination / "object-budget4096.glb"
    mesh.export(output_mesh)
    loaded = trimesh.load(output_mesh, force="mesh", process=False)
    if (not np.array_equal(loaded.faces, mesh.faces)
            or not np.array_equal(loaded.vertices, np.asarray(mesh.vertices, dtype=np.float32))):
        raise ValueError("Budgeted GLB changed vertices or faces on readback")
    original = trimesh.load(full_mesh, force="mesh", process=False)
    def artifact(path):
        return {"path": Path(os.path.relpath(path, destination)).as_posix(), "sha256": sha256(path)}
    # Rebase existing artifact paths without copying any masks, models or full mesh.
    for item in (prepared["masks"], prepared["mesh_generation"]["receipt"], prepared["segmentation"]["report"]):
        item["path"] = artifact((source.parent / item["path"]).resolve())["path"]
    receipt = {"policy": POLICY, "method": "v2dlb.mesh_budget.budget_mesh",
        "face_budget": 4096, "vertex_budget": 4096, "full_mesh": artifact(full_mesh),
        "output_mesh": artifact(output_mesh), "official_code": artifact(code),
        "source_preparation": artifact(source), "elapsed_seconds": time.perf_counter()-started,
        "full_faces": len(original.faces), "full_vertices": len(original.vertices),
        "active_faces": len(mesh.faces), "active_vertices": len(mesh.vertices),
        "padding_removed": "Only zero-area padding faces and unreferenced vertices",
        "coordinates": "Canonical coordinates preserved; no recentering or rescaling",
        "limitations": [DEVIATION, "No measured end-to-end speedup or accuracy equivalence"]}
    receipt_path = destination / "mesh-budget.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    prepared.update(mesh=artifact(output_mesh), input_mesh_policy=POLICY,
        mesh_budget={"receipt": artifact(receipt_path)},
        limitations=prepared.get("limitations", []) + [DEVIATION])
    verify_budgeted_preparation(prepared, destination, generation["mesh_sha256"], allowed=True)
    (destination / "preparation.json").write_text(json.dumps(prepared, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
