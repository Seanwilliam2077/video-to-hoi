"""Validate explicitly selected pre-inference organizer mesh budgeting."""
import json
from pathlib import Path


POLICY = "official-budget4096-before-inference"
DEVIATION = "Organizer 4096-face budget applied before inference; tracking/contact/penetration use a simplified surface"


def verify_budgeted_preparation(prepared, directory: Path, generated_hash: str, *, allowed: bool):
    from v2hoi.native import sha256
    policy = prepared.get("input_mesh_policy", "full-generated-surface")
    if policy == "full-generated-surface":
        if generated_hash != prepared["mesh"]["sha256"]:
            raise ValueError("Generated surface does not match the native input")
        return False
    if policy != POLICY:
        raise ValueError("Unknown input mesh policy")
    if not allowed:
        raise ValueError("Pre-inference mesh decimation requires --allow-budgeted-mesh")
    def read_artifact(artifact):
        path = (directory / artifact["path"]).resolve()
        if sha256(path) != artifact["sha256"]:
            raise ValueError("Mesh budget evidence or preserved full surface changed")
        return path
    receipt = json.loads(read_artifact(prepared["mesh_budget"]["receipt"]).read_text())
    read_artifact(receipt["full_mesh"])
    read_artifact(receipt["official_code"])
    if (receipt["policy"] != POLICY or receipt["face_budget"] != 4096
            or receipt["vertex_budget"] != 4096
            or receipt["full_mesh"]["sha256"] != generated_hash
            or receipt["output_mesh"]["sha256"] != prepared["mesh"]["sha256"]
            or receipt["method"] != "v2dlb.mesh_budget.budget_mesh"):
        raise ValueError("Mesh budget receipt does not match the actual generated input")
    read_artifact(receipt["output_mesh"])
    return True
