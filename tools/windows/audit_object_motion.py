"""Locate temporal discontinuities across a completed Track 1 native pipeline.

Prediction-only increments are diagnostics, not reference-relative ACC scores.
No smoothing, replacement trajectories or reconstruction edits are performed.
"""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from v2hoi.native import sha256


def motion_statistics(poses, frame_ids, fps):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (len(frame_ids), 4, 4) or not np.isfinite(poses).all():
        raise ValueError("Nonfinite, incomplete or malformed object trajectory")
    translation = poses[:, :3, 3]
    dt = np.linalg.norm(np.diff(translation, axis=0), axis=1)
    rotations = poses[:, :3, :3]
    relative = rotations[:-1].transpose(0, 2, 1) @ rotations[1:]
    angles = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))
    def stats(values):
        return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "max": float(values.max())}
    def extremes(values):
        return [{"from_frame": frame_ids[int(i)], "to_frame": frame_ids[int(i)+1],
            "time_seconds": frame_ids[int(i)+1]/fps, "value": float(values[i])}
            for i in np.argsort(values)[-8:][::-1]]
    return {"translation_increment_m": stats(dt), "rotation_increment_deg": stats(angles),
        "largest_translation_changes": extremes(dt), "largest_rotation_changes": extremes(angles)}


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--without-export", action="store_true", help="Inference diagnostics only; no submission conversion required")
    a = p.parse_args()
    identity = json.loads((a.run / "native-inputs.json").read_text())
    if identity["provenance"] != "track1":
        raise ValueError("Only permitted Track 1 reconstructions may be audited")
    ids = identity["original_frame_ids"]
    if ids != list(range(identity["expected_frames"])):
        raise ValueError("Original-frame identities are not consecutive and complete")
    seq = f"episode_{identity['episode']:06d}"
    root = a.run / "native" / seq
    before_path = root / "inference/coconet.pth"
    after_path = root / "inference/refined.pth"
    before = torch.load(before_path, map_location="cpu", weights_only=False)
    after = torch.load(after_path, map_location="cpu", weights_only=False)
    fp_path = root / "foundationpose" / f"{seq}.pkl"
    with fp_path.open("rb") as handle:
        fp = pickle.load(handle)
    exported = None
    if not a.without_export:
        with np.load(a.run / "export" / f"{seq}.npz", allow_pickle=False) as data:
            exported = np.repeat(np.eye(4, dtype=np.float32)[None], len(ids), axis=0)
            exported[:, :3, :3] = data["object_rotation"]
            exported[:, :3, 3] = data["object_translation"]
            if data["frame_ids"].tolist() != ids:
                raise ValueError("Exported timeline changed")
    stages = {"foundationpose_camera": np.asarray(fp["fp_poses_cam"])[:, 0],
        "network_input": before["in"]["pose_abs"], "network_initial": before["pr_initial"]["pose_abs"],
        "network_output": before["pr"]["pose_abs"], "postopt": after["pr"]["pose_abs"]}
    if exported is not None:
        stages["export"] = exported
    mask_iou = np.asarray(fp["fp_full_image_iou"], dtype=np.float64)
    def overlap_stats(values):
        valid = values[np.isfinite(values)]
        return {"observations":int(len(valid)),"mean":float(valid.mean()),
                "median":float(np.median(valid)),"min":float(valid.min()),"p95":float(np.percentile(valid,95))}
    report = {"episode": identity["episode"], "provenance": "track1", "frames": len(ids),
        "scope": "Per-stage prediction increments only; no reference data, quality score or automatic jump classification",
        "fps": 30, "stages": {name: motion_statistics(value, ids, 30) for name, value in stages.items()},
        "postopt_export_pose_bit_exact": None if exported is None else bool(np.array_equal(stages["postopt"], exported)),
        "submission_export_checked": exported is not None,
        "postopt_alias_bit_exact": bool(np.array_equal(stages["postopt"], after["pr"]["pose_abs_postopt"])),
        "tracking": {key: fp["metadata"][key] for key in ("registration_call_count", "tracking_call_count",
            "reliable_frame_count", "object_pose_fallback_count", "first_usable_frame_gt_rotation_oracle")},
        "foundationpose_mask_fit": {"scope":"Original FP rendered visible-mask IoU against permitted input masks; not ground-truth accuracy",
            "all_frames":overlap_stats(mask_iou),"excluding_search_frame_zero":overlap_stats(mask_iou[1:]),
            "segments":{f"{start}-{min(start+119,len(ids)-1)}":overlap_stats(mask_iou[start:start+120]) for start in range(0,len(ids),120)}},
        "sources": {"before": sha256(before_path), "after": sha256(after_path), "fp": sha256(fp_path)}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    for name, values in report["stages"].items():
        print(name, "translation", values["translation_increment_m"], "rotation", values["rotation_increment_deg"])
    print("Postopt/export exactly equal:", report["postopt_export_pose_bit_exact"])


if __name__ == "__main__":
    main()
