"""Validate explicitly selected pre-inference organizer mesh budgeting."""
import json
from pathlib import Path


POLICY = "official-budget4096-before-inference"
DEVIATION = "Organizer 4096-face budget applied before inference; tracking/contact/penetration use a simplified surface"


def verify_budgeted_preparation(prepared, directory: Path, generated_hash: str, *, allowed: bool):
    from v2hoi.native import sha256
    import math
    def check_artifact(artifact, base):
        path = (base / artifact["path"]).resolve()
        if sha256(path) != artifact["sha256"]:
            raise ValueError("Mesh budget evidence or preserved full surface changed")
        return path
    calibration = prepared.get("mesh_calibration")
    if calibration:
        receipt_path = check_artifact(calibration["receipt"], directory)
        record = json.loads(receipt_path.read_text())
        for key in ("input_mesh", "output_mesh", "official_code"):
            check_artifact(record[key], receipt_path.parent)
        if (record["status"] != "PASS" or record["provenance"] != "track1"
                or record["method"] not in {"NVIDIA scale_mesh_srt.optimize_srt(mode=s)", "NVIDIA scale_mesh_srt.optimize_srt(mode=str)"}
                or record["input_mesh"]["sha256"] != generated_hash
                or not math.isfinite(record["scale_factor"]) or record["scale_factor"] <= 0
                or record["video_sha256"] != prepared["video_sha256"]
                or record["mask_sha256"] != prepared["masks"]["sha256"]
                or not record["uniform_scale_only"] or not record["appearance_preserved"]):
            raise ValueError("Invalid mesh calibration evidence")
        if (len(record["training_frame_ids"]) < 3 or len(record["holdout_frame_ids"]) < 3
                or set(record["training_frame_ids"]) & set(record["holdout_frame_ids"])
                or record["holdout_projected_vertex_iou"]["mean_after"] < record["holdout_projected_vertex_iou"]["minimum_mean"]):
            raise ValueError("Mesh calibration lacks independent accepted holdout observations")
        generated_hash = record["output_mesh"]["sha256"]
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
        return check_artifact(artifact, directory)
    budget_path = read_artifact(prepared["mesh_budget"]["receipt"])
    receipt = json.loads(budget_path.read_text())
    check_artifact(receipt["full_mesh"], budget_path.parent)
    check_artifact(receipt["official_code"], budget_path.parent)
    input_hash = prepared["mesh"]["sha256"]
    if prepared.get("mesh_scale_search"):
        from v2hoi.legacy_scale import METHOD, REVISION
        search_path = read_artifact(prepared["mesh_scale_search"]["receipt"])
        search = json.loads(search_path.read_text())
        for key in ("input_mesh", "output_mesh", "official_code", "chamfer_code", "io_adapter", "fine_search", "source_preparation"):
            check_artifact(search[key], search_path.parent)
        if (search["status"] != "COMPLETED" or search["provenance"] != "track1"
                or search["method"] != METHOD or search["revision"] != REVISION
                or search["input_mesh"]["sha256"] != receipt["output_mesh"]["sha256"]
                or search["output_mesh"]["sha256"] != input_hash
                or not math.isfinite(search["scale_factor"]) or search["scale_factor"] <= 0
                or search["coarse_candidates"] != 30 or search["fine_candidates"] != 11
                or search["registration_iterations"] != 5 or search["surface_samples"] != 10000
                or search["video_sha256"] != prepared["video_sha256"]
                or search["mask_sha256"] != prepared["masks"]["sha256"]
                or not all(search[k] for k in ("uniform_scale_only", "appearance_preserved", "faces_preserved"))):
            raise ValueError("Invalid legacy scale search evidence")
        input_hash = search["input_mesh"]["sha256"]
    if (receipt["policy"] != POLICY or receipt["face_budget"] != 4096
            or receipt["vertex_budget"] != 4096
            or receipt["full_mesh"]["sha256"] != generated_hash
            or receipt["output_mesh"]["sha256"] != input_hash
            or receipt["method"] != "v2dlb.mesh_budget.budget_mesh"
            or receipt.get("appearance_preserved") is not True):
        raise ValueError("Mesh budget receipt does not match the actual generated input")
    check_artifact(receipt["output_mesh"], budget_path.parent)
    return True
