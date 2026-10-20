"""Fit actual MHR finger controls to independently detected Track 1 image hands."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v2hoi.hand_refinement import HandFitConfig, NativeMHRHands, refine_hands
from v2hoi.human_result import load_human_result, native_vertices_to_track1, write_human_result


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review-arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "weights/human/mhr_model.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model-batch", type=int, default=8)
    parser.add_argument("--holdout-stride", type=int, default=5)
    parser.add_argument("--unobserved-policy", choices=("preserve", "bounded_temporal_prior"), default="preserve")
    parser.add_argument("--acceleration-weight", type=float, default=0.025)
    parser.add_argument("--pose-prior-weight", type=float, default=0.03)
    parser.add_argument("--max-delta-radians", type=float, default=0.65)
    parser.add_argument("--relative-weight", type=float, default=1.0)
    parser.add_argument("--optimize-wrist-chain", action="store_true")
    parser.add_argument("--arm-delta-limit-radians", type=float, default=0.10)
    parser.add_argument("--wrist-max-displacement-metres", type=float, default=0.04)
    parser.add_argument("--arm-update-temporal-weight", type=float, default=0.08)
    parser.add_argument("--finger-update-temporal-weight", type=float, default=0.08)
    parser.add_argument("--spatial-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--motion-phases", type=Path)
    parser.add_argument("--contact-object", type=Path)
    parser.add_argument("--contact-phases", type=Path)
    parser.add_argument("--contact-anchor-weight", type=float, default=0.01)
    parser.add_argument("--contact-penetration-weight", type=float, default=0.005)
    parser.add_argument("--contact-scale", type=float, default=1.0)
    parser.add_argument("--contact-scale-mode", choices=("projection_preserving", "fixed_pose_geometry"), default="projection_preserving")
    parser.add_argument("--skip-mesh-export", action="store_true", help="Development probe only; does not create a complete review")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists: use a fresh directory to preserve prior experiments")
    if args.threads < 1 or args.model_batch < 1:
        parser.error("threads and model-batch must be positive")
    if bool(args.contact_object) != bool(args.contact_phases):
        parser.error("contact-object and contact-phases must be supplied together")
    if args.spatial_acceleration_weight > 0 and not args.motion_phases:
        parser.error("A positive spatial-acceleration-weight requires explicit motion-phases")
    if not np.isfinite(args.contact_scale) or args.contact_scale <= 0:
        parser.error("contact-scale must be finite and positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("track") != 1:
        parser.error("Only verified Track 1 input manifests are allowed")
    baseline = load_human_result(args.baseline)
    baseline_report = baseline.get("report", {})
    if baseline_report.get("video_sha256", baseline_report.get("source_video_sha256")) != manifest["video_sha256"]:
        parser.error("Baseline provenance does not identify the same Track 1 source video")
    if len(baseline["pose"]) != manifest["video_frames"] or not np.array_equal(baseline["frame_indices"], np.arange(manifest["video_frames"])):
        parser.error("Baseline must cover all consecutive original video frames")
    with np.load(args.observations, allow_pickle=False) as archive:
        observations = {name: archive[name] for name in archive.files}
    observation_report_path = args.observations.parent / "report.json"
    observation_report = json.loads(observation_report_path.read_text(encoding="utf-8"))
    if observation_report.get("source_video_sha256") != manifest["video_sha256"]:
        parser.error("Independent observation report does not identify the same Track 1 video")
    if observation_report.get("frames") != manifest["video_frames"]:
        parser.error("Independent observation report has different video coverage")
    intrinsics = np.load(args.review_arrays / "camera_intrinsics.npy", allow_pickle=False)
    if not np.allclose(intrinsics, manifest["camera_intrinsics"], atol=1e-5, rtol=0):
        parser.error("Review intrinsics differ from the video manifest")
    import torch
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    source_paths = [Path(__file__), ROOT / "src/v2hoi/hand_refinement.py"]
    if args.contact_object or args.motion_phases:
        source_paths.append(ROOT / "src/v2hoi/hand_contact.py")
    source_files = {str(path.relative_to(ROOT)).replace("\\", "/"): path.read_bytes() for path in source_paths}
    config = HandFitConfig(iterations=args.iterations, holdout_stride=args.holdout_stride,
                           unobserved_policy=args.unobserved_policy, acceleration_weight=args.acceleration_weight,
                           pose_prior_weight=args.pose_prior_weight, max_delta_radians=args.max_delta_radians,
                           relative_weight=args.relative_weight, optimize_wrist_chain=args.optimize_wrist_chain,
                           arm_delta_limit_radians=args.arm_delta_limit_radians,
                           wrist_max_displacement_metres=args.wrist_max_displacement_metres,
                           arm_update_temporal_weight=args.arm_update_temporal_weight,
                           finger_update_temporal_weight=args.finger_update_temporal_weight,
                           spatial_acceleration_weight=args.spatial_acceleration_weight)
    config.validate()
    hands = NativeMHRHands(str(args.model), device=args.device)
    positions = sorted(set([0, len(baseline["pose"]) // 3, 2 * len(baseline["pose"]) // 3, len(baseline["pose"]) - 1]))
    parity = hands.verify_full_forward(baseline, positions)
    print(json.dumps({"native_hand_mapping_parity": parity}), flush=True)
    contact = None
    contact_provenance = None
    motion_prior_weights = None
    motion_phase_provenance = None
    if args.motion_phases:
        from v2hoi.hand_contact import phase_weights
        phase_report = json.loads(args.motion_phases.read_text(encoding="utf-8"))
        if phase_report.get("video_sha256") != manifest["video_sha256"]:
            parser.error("Motion phase annotations must identify this Track 1 source video")
        motion_prior_weights = phase_weights(phase_report, len(baseline["pose"]))
        motion_phase_provenance = {"sha256": digest(args.motion_phases), "annotation": phase_report,
                                   "interpretation": "Phase-conditioned motion prior, not object contact or inferred true fingertip observations"}
    if args.contact_object:
        from v2hoi.hand_contact import WeakContactConstraint
        object_report_path = args.contact_object.parent / "report.json"
        object_report = json.loads(object_report_path.read_text(encoding="utf-8"))
        phase_report = json.loads(args.contact_phases.read_text(encoding="utf-8"))
        if object_report.get("source_video_sha256", object_report.get("video_sha256")) != manifest["video_sha256"] or phase_report.get("video_sha256") != manifest["video_sha256"]:
            parser.error("Object candidate and phase annotations must identify this Track 1 source video")
        with np.load(args.contact_object, allow_pickle=False) as archive:
            object_candidate = {name: archive[name] for name in archive.files}
        if not np.allclose(object_candidate["camera_intrinsics"], intrinsics, atol=1e-5, rtol=0):
            parser.error("Object hypothesis camera differs from the Human initializer")
        object_candidate["vertices"] = object_candidate["vertices"] * args.contact_scale
        if args.contact_scale_mode == "projection_preserving":
            object_candidate["object_to_camera"] = object_candidate["object_to_camera"].copy()
            object_candidate["object_to_camera"][:, :3, 3] *= args.contact_scale
        contact_provenance = {
            "object_sha256": digest(args.contact_object), "object_report_sha256": digest(object_report_path),
            "phases_sha256": digest(args.contact_phases), "object_report": object_report,
            "phases": phase_report, "scale_factor": args.contact_scale, "scale_mode": args.contact_scale_mode,
            "scale_interpretation": ("Local mesh and camera translation scaled together; projection preserved, human/object relative depth changes"
                                     if args.contact_scale_mode == "projection_preserving" else
                                     "Local mesh scaled at fixed pose; geometry stress test changes projected silhouette"),
            "uncertainty": "The sensitivity range is chosen, not a calibrated confidence interval; shape/backside and metric scale remain hypotheses",
        }
        with torch.no_grad():
            baseline_hands = hands(torch.as_tensor(baseline["pose"], device=args.device),
                                   torch.as_tensor(baseline["scales"], device=args.device)).cpu().numpy()
        try:
            contact = WeakContactConstraint(object_candidate, phase_report, baseline_hands, baseline["frame_indices"],
                                            device=args.device, anchor_weight=args.contact_anchor_weight,
                                            penetration_weight=args.contact_penetration_weight)
        except ValueError as error:
            args.output.mkdir(parents=True)
            snapshot = args.output / "source_snapshot"
            snapshot.mkdir()
            for name, content in source_files.items():
                (snapshot / Path(name).name).write_bytes(content)
            rejected = {"status": "contact_abstained_before_optimization", "reason": str(error),
                        "track": 1, "episode": manifest["episode"], "frames": len(baseline["pose"]),
                        "source_video_sha256": manifest["video_sha256"], "baseline_sha256": digest(args.baseline),
                        "contact_provenance": contact_provenance, "no_new_human_result_written": True,
                        "source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in source_files.items()}}
            (args.output / "contact-rejection.json").write_text(json.dumps(rejected, indent=2, allow_nan=False), encoding="utf-8")
            parser.exit(2, f"Contact abstained: {error}; evidence saved to {args.output}\n")
    fitted = refine_hands(baseline, observations, intrinsics, hands, config=config, device=args.device,
                          log=lambda message: print(message, flush=True), contact_constraint=contact,
                          motion_prior_weights=motion_prior_weights)
    report = fitted["report"]
    report.update({
        "track": 1, "episode": manifest["episode"], "complete_video": True,
        "source_video_sha256": manifest["video_sha256"], "baseline_sha256": digest(args.baseline),
        "observations_sha256": digest(args.observations), "mhr_model_sha256": digest(args.model),
        "observation_report_sha256": digest(observation_report_path),
        "independent_observation_provenance": observation_report,
        "observation_source": str(args.observations.resolve()),
        "observation_semantics": "Independent MediaPipe image landmarks on SAM-wrist-seeded crops; reliability weights are not per-joint probabilities or measured visibility",
        "native_hand_mapping_parity": parity,
        "camera_intrinsics": intrinsics.tolist(), "intrinsics_status": manifest["intrinsics_status"],
        "source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in source_files.items()},
        "hand_only_development_probe": bool(args.skip_mesh_export),
        "contact_provenance": contact_provenance,
        "motion_phase_provenance": motion_phase_provenance,
        "torch_num_threads": torch.get_num_threads(),
    })
    args.output.mkdir(parents=True)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for name, content in source_files.items():
        (snapshot / Path(name).name).write_bytes(content)
    np.save(args.output / "hand_projected.npy", fitted["hand_projected"])
    np.save(args.output / "hand_joints_m.npy", fitted["hand_joints_m"])
    np.save(args.output / "active_hands.npy", fitted["active_hands"])
    np.save(args.output / "optimization_weights.npy", fitted["optimization_weights"])
    if contact is not None:
        np.save(args.output / "contact_phase_weights.npy", contact.weights)
        np.save(args.output / "contact_anchor_weights.npy", contact.anchor_weights)
        np.save(args.output / "contact_anchor_targets_m.npy", contact.anchor_targets)
    if not args.skip_mesh_export:
        vertices, joints = [], []
        with torch.no_grad(), torch.autocast(device_type=torch.device(args.device).type, enabled=False):
            shape = torch.as_tensor(fitted["shape"], device=args.device)
            scales = torch.as_tensor(fitted["scales"], device=args.device)
            for start in range(0, len(fitted["pose"]), args.model_batch):
                pose = torch.as_tensor(fitted["pose"][start:start + args.model_batch], device=args.device)
                params = torch.cat([pose, scales[None].expand(len(pose), -1)], dim=1)
                v, j = hands.model(shape[None].expand(len(pose), -1).contiguous(), params, pose.new_zeros(len(pose), 72), True)
                vertices.append(native_vertices_to_track1(v.cpu().numpy()))
                joints.append(native_vertices_to_track1(j[..., :3].cpu().numpy()))
                if start % (args.model_batch * 10) == 0:
                    print(f"Exported native geometry {min(start + len(pose), len(fitted['pose']))}/{len(fitted['pose'])}", flush=True)
        review = args.output / "review_arrays"
        review.mkdir()
        np.save(review / "vertices_m.npy", np.concatenate(vertices))
        np.save(review / "joints_3d.npy", np.concatenate(joints))
        for name in ("frame_indices", "faces", "camera_intrinsics", "fps"):
            np.save(review / f"{name}.npy", np.load(args.review_arrays / f"{name}.npy", allow_pickle=False))
        baseline_vertices = np.load(args.review_arrays / "vertices_m.npy", mmap_mode="r", allow_pickle=False)
        distances = np.linalg.norm(np.concatenate(vertices) - baseline_vertices, axis=-1).mean(-1) * 1000
        report["mesh_change_from_baseline_mm"] = {"mean": float(distances.mean()), "max_frame_mean": float(distances.max()),
                                                  "interpretation": "Change from baseline prediction, not error against true geometry"}
        if contact is not None:
            from v2hoi.hand_contact import native_hand_vertex_indices, native_mesh_penetration_proxy
            selection = native_hand_vertex_indices(hands.model)
            print("Measuring actual native hand-skin penetration against hypothetical convex object", flush=True)
            before_penetration = native_mesh_penetration_proxy(baseline_vertices, selection, contact.candidate, contact.weights)
            after_penetration = native_mesh_penetration_proxy(np.concatenate(vertices), selection, contact.candidate, contact.weights)
            report["contact"]["actual_hand_skin_penetration_before"] = before_penetration["report"]
            report["contact"]["actual_hand_skin_penetration_after"] = after_penetration["report"]
            for prefix, measurement in (("before", before_penetration), ("after", after_penetration)):
                for name in ("per_frame_penetrating_vertices", "per_frame_mean_depth_m", "per_frame_max_depth_m"):
                    np.save(args.output / f"contact_mesh_{prefix}_{name}.npy", measurement[name])
    if contact is not None:
        guards = {}
        for side in ("left", "right"):
            before = report["before_heldout"][side]
            after = report["after_heldout"][side]
            sufficient = before["observed_frames"] >= 5
            guards[side] = {"enough_heldout_detections": sufficient,
                            "before_absolute_pixels": before["weighted_reprojection_pixels"],
                            "after_absolute_pixels": after["weighted_reprojection_pixels"],
                            "passed": bool(after["weighted_reprojection_pixels"] <= before["weighted_reprojection_pixels"] * 1.05 + 0.25) if sufficient else None}
        report["contact"]["heldout_image_guards"] = guards
        rejection_reasons = []
        if any(value["passed"] is False for value in guards.values()):
            rejection_reasons.append("heldout_image_regression")
        if report["contact"]["after"]["weighted_anchor_distance_mm"] > report["contact"]["before"]["weighted_anchor_distance_mm"] + 1:
            rejection_reasons.append("anchor_detachment_regression")
        if "actual_hand_skin_penetration_before" in report["contact"]:
            for side in ("left", "right"):
                before = report["contact"]["actual_hand_skin_penetration_before"][side]
                after = report["contact"]["actual_hand_skin_penetration_after"][side]
                if before["admitted_frames"] and (after["mean_depth_over_hand_vertices_mm"] > before["mean_depth_over_hand_vertices_mm"] + 0.5 or after["max_vertex_depth_mm"] > before["max_vertex_depth_mm"] + 2):
                    rejection_reasons.append(f"{side}_native_hand_skin_penetration_regression")
        report["contact"]["rejection_reasons"] = rejection_reasons
        report["contact"]["candidate_status"] = ("rejected_by_quality_guard" if rejection_reasons
                                                  else "experimental_passed_available_guards_not_confirmed_contact")
    report["total_seconds"] = round(time.monotonic() - started, 3)
    summary = write_human_result(args.output / f"episode_{manifest['episode']:06d}_human.npz", fitted, report)
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "frames": len(fitted["pose"]), "seconds": report["total_seconds"],
                      "initial_objective": report["initial_objective"], "best_objective": report["best_objective"]}), flush=True)


if __name__ == "__main__":
    main()
