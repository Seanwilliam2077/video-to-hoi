"""Freeze one Track 1 object-depth hypothesis before a controlled hand A/B."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v2hoi.hand_contact import closest_mesh_points, convex_mesh_planes
from v2hoi.object_depth_gauge import apply_projective_gauge, far_side_rigid_shift, fit_projective_gauge, projection


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve attempts: output directory must not exist")
    policy = json.loads(args.policy.read_text(encoding="utf-8-sig"))
    if policy.get("track") != 1 or policy.get("episode") != 16 or policy.get("frames") != 360:
        raise ValueError("This controlled policy must identify Track 1 EP16 all360")
    bound = policy["inputs"]
    before = {name: sha(item["path"]) for name, item in bound.items()}
    if any(before[name] != item["sha256"] for name, item in bound.items()):
        raise ValueError("A frozen input digest changed")
    t0 = time.monotonic()
    candidate = dict(np.load(bound["object"]["path"], allow_pickle=False))
    observations = np.load(bound["observations"]["path"], allow_pickle=False)
    hands = np.load(bound["hand_joints"]["path"], allow_pickle=False)
    vertices = np.load(bound["human_vertices"]["path"], mmap_mode="r", allow_pickle=False)
    skin = np.load(bound["skin_indices"]["path"], allow_pickle=False)
    K = candidate["camera_intrinsics"]
    if not np.array_equal(candidate["frame_indices"], np.arange(360)) or not np.array_equal(observations["frame_indices"], np.arange(360)):
        raise ValueError("Retain exact original IDs0..359")
    if str(observations["source_video_sha256"]) != policy["source_video_sha256"]:
        raise ValueError("Observer source video mismatch")
    if hands.shape != (360, 2, 21, 3) or vertices.shape != (360, 18439, 3):
        raise ValueError("Current native geometry dimensions differ")
    if not np.array_equal(np.load(bound["camera"]["path"], allow_pickle=False), K):
        raise ValueError("Human and object camera differ")
    normal, offset = convex_mesh_planes(candidate["vertices"], candidate["faces"])
    tips = np.asarray([4, 8, 12, 16, 20])
    settings = policy["selection"]
    first, last = settings["frames_inclusive"]
    stride = settings["excluded_original_id_stride"]
    conf = observations["confidence"]
    source_xy = observations["joints2d"]
    rows, skipped, train_frames = [], Counter(), []
    for frame in range(first, last + 1):
        # Holdouts are excluded before consulting any coordinates/support flags.
        if frame % stride == 0:
            skipped["original_stride_holdout"] += 1
            continue
        if not bool(candidate["pose_usable"][frame]):
            skipped["object_not_usable"] += 1
            continue
        supported = conf[frame, 1] >= settings["minimum_confidence"]
        if not supported[0] or supported[1:].sum() < settings["minimum_nonwrist_landmarks"]:
            skipped["right_hand_not_supported"] += 1
            continue
        train_frames.append(frame)
        transform = candidate["object_to_camera"][frame]
        R, t = transform[:3, :3], transform[:3, 3]
        points = np.asarray(vertices[frame, skin["right"]], float)
        patches = {}
        for tip in tips:
            if not supported[tip]:
                skipped["unsupported_tip"] += 1
                continue
            point = hands[frame, 1, tip]
            h = point @ K.T
            discrepancy = float(np.linalg.norm(h[:2] / h[2] - source_xy[frame, 1, tip]))
            if discrepancy > settings["maximum_reprojection_px"]:
                skipped["tip_reprojection"] += 1
                continue
            ids = np.flatnonzero(np.linalg.norm(points - point, axis=1) <= settings["native_skin_patch_radius_m"])
            if not len(ids):
                skipped["empty_skin_patch"] += 1
                continue
            patches[int(tip)] = (ids, discrepancy)
        if not patches:
            continue
        union = np.unique(np.concatenate([item[0] for item in patches.values()]))
        local = (points[union] - t) @ R
        closest, distances = closest_mesh_points(local, candidate["vertices"], candidate["faces"], chunk_size=16)
        lookup = {int(index): row for row, index in enumerate(union)}
        for tip, (ids, discrepancy) in patches.items():
            available = np.asarray([lookup[int(index)] for index in ids])
            index = int(available[distances[available].argmin()])
            distance = float(distances[index])
            if distance > settings["maximum_surface_distance_m"]:
                skipped["skin_too_far"] += 1
                continue
            surface_local = closest[index]
            surface_camera = surface_local @ R.T + t
            active = np.abs(normal @ surface_local - offset) <= 2e-6
            if not active.any():
                skipped["no_surface_plane"] += 1
                continue
            front = -(normal[active] @ R.T) @ surface_camera / np.linalg.norm(surface_camera)
            facing = float(front.max())
            if facing < settings["minimum_front_facing_cosine"]:
                skipped["surface_not_camera_facing"] += 1
                continue
            vertex_index = int(skin["right"][union[index]])
            rows.append({"original_frame_id": frame, "side": "right", "tip_landmark": tip,
                         "native_skin_vertex": vertex_index, "native_tip_camera_m": hands[frame, 1, tip].tolist(),
                         "native_skin_camera_m": points[union[index]].tolist(),
                         "fixed_surface_camera_m": surface_camera.tolist(),
                         "fixed_surface_object_m": surface_local.tolist(),
                         "observed_tip_pixel": source_xy[frame, 1, tip].tolist(),
                         "image_weight": float(conf[frame, 1, tip]), "tip_reprojection_px": discrepancy,
                         "initial_surface_distance_m": distance, "camera_facing_cosine": facing})
    if len(rows) < settings["minimum_selected_points"] or len({r["original_frame_id"] for r in rows}) < settings["minimum_selected_frames"]:
        raise ValueError(f"Abstain: insufficient visible-right contact hypotheses ({len(rows)} points)")
    fit = fit_projective_gauge(np.asarray([r["native_skin_camera_m"] for r in rows]),
                               np.asarray([r["fixed_surface_camera_m"] for r in rows]),
                               np.asarray([r["original_frame_id"] for r in rows]), **policy["fit"])
    final = apply_projective_gauge(candidate, fit["factor"])
    final["depth_gauge_factor"] = np.asarray(fit["factor"], np.float64)
    final["data_provenance"] = np.asarray("Track1 EP16 own RGB/masks; fixed current-MP right-skin contact-supported monocular depth hypothesis, not calibrated physical depth or official reference")
    maximum_projection_error = 0.0
    feasibility = []
    for frame in range(360):
        maximum_projection_error = max(maximum_projection_error, float(np.abs(
            projection(candidate["vertices"], candidate["object_to_camera"][frame], K) -
            projection(final["vertices"], final["object_to_camera"][frame], K)).max()))
        if first <= frame <= last:
            transform = final["object_to_camera"][frame]
            n = normal @ transform[:3, :3].T
            b = offset * fit["factor"] + n @ transform[:3, 3]
            row = far_side_rigid_shift(vertices[frame, skin["left"]], hands[frame, 0, 0], n, b)
            row.update(original_frame_id=frame, object_pose_usable=bool(final["pose_usable"][frame]),
                       maximum_projection_change_pixels=row.pop("normalized_projection_max_change") * float(K[0, 0]))
            feasibility.append(row)
    after = {name: sha(item["path"]) for name, item in bound.items()}
    if before != after or maximum_projection_error > 1e-7:
        raise ValueError("Protected input or exact projection invariant changed")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "contact_ready.npz", **final)
    report = {"schema_version": "ep16_shared_projective_depth_gauge.v1", "recorded_at": datetime.now(timezone.utc).isoformat(),
              "track": 1, "episode": 16, "frame_count": 360, "policy_path": str(args.policy), "policy_sha256": sha(args.policy),
              "source_sha256": sha(Path(__file__)), "module_sha256": sha(Path(__file__).resolve().parents[1] / "src/v2hoi/object_depth_gauge.py"),
              "inputs_before_and_after": before, "factor_fit": fit, "selection": rows,
              "eligible_training_frames_before_tip_selection": train_frames,
              "selected_frames": sorted({r["original_frame_id"] for r in rows}),
              "points_per_digit": dict(Counter(str(r["tip_landmark"]) for r in rows)), "exclusions": dict(skipped),
              "camera_intrinsics_exact": True, "rotations_exact": bool(np.array_equal(candidate["object_to_camera"][:, :3, :3], final["object_to_camera"][:, :3, :3])),
              "original_frame_ids_exact": True, "all360_object_projection_max_abs_px": maximum_projection_error,
              "mesh_extents_before_m": np.ptp(candidate["vertices"], axis=0).tolist(),
              "mesh_extents_after_m": np.ptp(final["vertices"], axis=0).tolist(),
              "output_sha256": sha(args.output / "contact_ready.npz"),
              "left_fixed_articulation_far_side_feasibility": feasibility,
              "left_feasibility_is_selection_input": False,
              "limits": ["Right skin/near-surface contact is an initializer assumption, not labelled true contact.",
                         "Image projection is gauge-invariant; 2D IoU and reprojection cannot establish relative depth.",
                         "Left rigid translation is a fixed-articulation diagnostic, not a native IK solution or verified hidden pose.",
                         "A/B must share this immutable object; no per-arm/post-result depth fitting.",
                         "No forbidden reference, Track2 content, official score or acceptance is used."],
              "quality_accepted": False, "physical_depth_or_scale_accepted": False,
              "seconds": time.monotonic() - t0}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "factor": fit["factor"], "data_only_factor": fit["data_only_factor"],
                      "points": len(rows), "frames": len(report["selected_frames"]), "projection_error_px": maximum_projection_error,
                      "output_sha256": report["output_sha256"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
