"""Fit native MHR finger controls to source-bound Track 1 hand targets."""
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
from v2hoi.hand_refinement import HandFitConfig, NativeMHRHands, control_prior_support_weights, refine_hands
from v2hoi.human_result import load_human_result, native_vertices_to_track1, validate_human_result, write_human_result


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def target_provenance(observation_report: dict, observations: dict) -> dict:
    """Classify metadata without mistaking hybrid model targets for image data."""
    raw_kind = np.asarray(observations.get("observation_kind", ""))
    if raw_kind.ndim != 0:
        raise ValueError("Observation kind must be a scalar provenance tag")
    kind = str(raw_kind.item())
    generated = (kind == "egohand_geometric_proposals.v1" or
                 observation_report.get("schema_version") in {
                     "egohand_geometric_proposals.v1", "refined_egohand_proposals.v1"})
    if generated:
        return {
            "method": "source_bound_model_target_native_mhr_sequence_fit",
            "generated_model_proposal_provenance": observation_report,
            "observation_semantics": (
                "Generated EgoHand geometry projected into original pixels; retargeting may share "
                "MediaPipe wrist/support inputs. Admission is not independent image evidence, "
                "measured visibility or reconstruction truth"),
        }
    if observation_report.get("observer") == "MediaPipe HandLandmarker independent 2D observations":
        return {
            "independent_observation_provenance": observation_report,
            "observation_semantics": (
                "Independent MediaPipe image landmarks on source-bound crops; reliability weights "
                "are not per-joint probabilities or measured visibility"),
        }
    return {
        "unclassified_target_provenance": observation_report,
        "observation_semantics": "Source-bound hand targets of unverified observer type; no image-independence or visibility claim",
    }


def validate_contact_anchor_reference(reference: dict, baseline: dict, manifest: dict,
                                      model_sha256: str, intrinsics: np.ndarray) -> dict:
    """Validate a read-only anchor selector without changing optimization input."""
    count = manifest["video_frames"]
    validate_human_result(reference, expected_frames=count)
    validate_human_result(baseline, expected_frames=count)
    report = reference.get("report", {})
    if (manifest.get("track") != 1 or report.get("track") != 1 or
            report.get("episode") != manifest["episode"] or
            report.get("source_video_sha256", report.get("video_sha256")) != manifest["video_sha256"]):
        raise ValueError("Contact anchor reference must identify the same Track 1 video/episode")
    expected = np.arange(count)
    if not np.array_equal(reference.get("frame_indices"), expected) or not np.array_equal(baseline.get("frame_indices"), expected):
        raise ValueError("Contact anchor reference requires the same complete original frame IDs")
    if report.get("parameter_format") != "native_mhr_pose136_scales68_shape45":
        raise ValueError("Contact anchor reference must be native MHR parameters")
    if report.get("mhr_model_sha256") != model_sha256:
        raise ValueError("Contact anchor reference uses a different native decoder")
    for name in ("shape", "scales"):
        if not np.array_equal(reference[name], baseline[name]):
            raise ValueError(f"Contact anchor reference changed fixed {name}")
    if not np.array_equal(reference["pose"][:, :6], baseline["pose"][:, :6]):
        raise ValueError("Contact anchor reference changed root controls")
    reference_k = np.asarray(report.get("camera_intrinsics"), dtype=np.float64)
    expected_k = np.asarray(intrinsics, dtype=np.float64)
    if reference_k.shape != (3, 3) or not np.array_equal(reference_k, expected_k):
        raise ValueError("Contact anchor reference camera differs from the fixed human camera")
    if (not np.isclose(float(reference.get("fps", 0)), manifest["video_fps"], atol=1e-6, rtol=0) or
            not np.isclose(float(baseline.get("fps", 0)), manifest["video_fps"], atol=1e-6, rtol=0)):
        raise ValueError("Contact anchor reference frame rate differs from the source")
    return {"frames": count, "same_source_video_and_original_ids": True,
            "same_native_decoder": True, "fixed_identity_unchanged": True,
            "fixed_skeleton_scales_unchanged": True, "root_controls_unchanged": True,
            "exact_camera_intrinsics_match": True,
            "optimization_initialization_changed": False,
            "interpretation": "Geometry for selecting fixed contact anchors only; not hidden contact truth or new image observations"}


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
    parser.add_argument("--learning-rate", type=float, default=0.025,
                        help="Adam learning rate; small native-geometry budget probes may require a lower value")
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
    parser.add_argument("--support-transition-frames", type=int, default=0,
                        help="Opt-in bounded correction collar around image support; adds no image targets")
    parser.add_argument("--correction-velocity-budget-weight", type=float, default=0.0,
                        help="Opt-in existing-budget loss on additional correction motion; independent checks still required")
    parser.add_argument("--correction-budget-safety-fraction", type=float, default=1.0,
                        help="Internal training-budget fraction in (0,1]; tightens priors without changing independent checks")
    parser.add_argument("--spatial-acceleration-weight", type=float, default=0.0)
    parser.add_argument("--motion-phases", type=Path)
    parser.add_argument("--control-prior-support", type=Path,
                        help="Source-reviewed prior-only bounded control eligibility; adds no image targets")
    parser.add_argument("--control-prior-arm-delta-limit-radians", type=float)
    parser.add_argument("--control-prior-wrist-max-displacement-metres", type=float)
    parser.add_argument("--control-prior-wrist-ray-weight", type=float, default=0.0)
    parser.add_argument("--control-prior-wrist-max-projection-pixels", type=float)
    parser.add_argument("--surface-contact-config", type=Path,
                        help="Pinned experimental selected-native-skin and tangential-free surface-prior configuration")
    parser.add_argument("--contact-object", type=Path)
    parser.add_argument("--contact-phases", type=Path)
    parser.add_argument("--contact-anchor-reference", type=Path,
                        help="Optional same-source native MHR geometry for anchor selection only; initialization and priors remain baseline")
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
    if args.contact_anchor_reference and not args.contact_object:
        parser.error("contact-anchor-reference requires contact-object and contact-phases")
    if args.surface_contact_config and args.contact_object:
        parser.error("Choose either surface contact or the legacy fixed-anchor contact intervention")
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
        parser.error("Hand-target report does not identify the same Track 1 video")
    if observation_report.get("frames") != manifest["video_frames"]:
        parser.error("Hand-target report has different video coverage")
    intrinsics = np.load(args.review_arrays / "camera_intrinsics.npy", allow_pickle=False)
    if not np.allclose(intrinsics, manifest["camera_intrinsics"], atol=1e-5, rtol=0):
        parser.error("Review intrinsics differ from the video manifest")
    import torch
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    source_paths = [Path(__file__), ROOT / "src/v2hoi/hand_refinement.py"]
    if args.contact_object or args.motion_phases or args.surface_contact_config:
        source_paths.append(ROOT / "src/v2hoi/hand_contact.py")
    if args.surface_contact_config:
        source_paths.append(ROOT / "src/v2hoi/hand_surface_contact.py")
    source_files = {str(path.relative_to(ROOT)).replace("\\", "/"): path.read_bytes() for path in source_paths}
    config = HandFitConfig(iterations=args.iterations, learning_rate=args.learning_rate, holdout_stride=args.holdout_stride,
                           unobserved_policy=args.unobserved_policy, acceleration_weight=args.acceleration_weight,
                           pose_prior_weight=args.pose_prior_weight, max_delta_radians=args.max_delta_radians,
                           relative_weight=args.relative_weight, optimize_wrist_chain=args.optimize_wrist_chain,
                           arm_delta_limit_radians=args.arm_delta_limit_radians,
                           wrist_max_displacement_metres=args.wrist_max_displacement_metres,
                           arm_update_temporal_weight=args.arm_update_temporal_weight,
                           finger_update_temporal_weight=args.finger_update_temporal_weight,
                           support_transition_frames=args.support_transition_frames,
                           correction_velocity_budget_weight=args.correction_velocity_budget_weight,
                           correction_budget_safety_fraction=args.correction_budget_safety_fraction,
                           spatial_acceleration_weight=args.spatial_acceleration_weight,
                           control_prior_arm_delta_limit_radians=args.control_prior_arm_delta_limit_radians,
                           control_prior_wrist_max_displacement_metres=args.control_prior_wrist_max_displacement_metres,
                           control_prior_wrist_ray_weight=args.control_prior_wrist_ray_weight,
                           control_prior_wrist_max_projection_pixels=args.control_prior_wrist_max_projection_pixels)
    config.validate()
    control_prior_support = None
    control_prior_provenance = None
    if args.control_prior_support:
        annotation = json.loads(args.control_prior_support.read_text(encoding="utf-8-sig"))
        try:
            control_prior_support = control_prior_support_weights(
                annotation, track=manifest["track"], episode=manifest["episode"],
                frame_count=manifest["video_frames"], video_sha256=manifest["video_sha256"])
            receipt = Path(annotation["source_review_receipt"])
            if digest(receipt) != annotation["source_review_receipt_sha256"]:
                raise ValueError("Control-prior source-review receipt bytes changed")
        except (ValueError, OSError) as error:
            parser.error(str(error))
        control_prior_provenance = {
            "path": str(args.control_prior_support.resolve()),
            "sha256": digest(args.control_prior_support), "annotation": annotation,
            "source_review_receipt_sha256": digest(receipt),
            "interpretation": "Parameter eligibility only; no image coordinates, contact targets or observed hidden motion"}
    hands = NativeMHRHands(str(args.model), device=args.device)
    positions = sorted(set([0, len(baseline["pose"]) // 3, 2 * len(baseline["pose"]) // 3, len(baseline["pose"]) - 1]))
    parity = hands.verify_full_forward(baseline, positions)
    print(json.dumps({"native_hand_mapping_parity": parity}), flush=True)
    contact = None
    contact_provenance = None
    surface_contact = None
    surface_contact_provenance = None
    motion_prior_weights = None
    motion_phase_provenance = None
    if args.surface_contact_config:
        from v2hoi.hand_surface_contact import NativeHandSurfaceContact, NativeSelectedHandSurface
        specification = json.loads(args.surface_contact_config.read_text(encoding="utf-8-sig"))
        if (specification.get("method") != "native_hand_surface_contact.v1" or
                specification.get("track") != 1 or specification.get("episode") != manifest["episode"] or
                specification.get("frame_count") != len(baseline["pose"]) or
                specification.get("video_sha256") != manifest["video_sha256"] or
                specification.get("baseline_sha256") != digest(args.baseline) or
                specification.get("mhr_model_sha256") != digest(args.model)):
            parser.error("Surface-contact configuration must bind the exact source, initialization and native decoder")
        object_path = Path(specification["object_path"])
        phases_path = Path(specification["phases_path"])
        object_report_path = object_path.parent / "report.json"
        if (digest(object_path) != specification["object_sha256"] or
                digest(phases_path) != specification["phases_sha256"] or
                digest(object_report_path) != specification["object_report_sha256"]):
            parser.error("Surface-contact object/phase bytes differ from the frozen configuration")
        object_report = json.loads(object_report_path.read_text(encoding="utf-8-sig"))
        phases = json.loads(phases_path.read_text(encoding="utf-8-sig"))
        if (object_report.get("track") != 1 or object_report.get("episode") != manifest["episode"] or
                object_report.get("source_video_sha256", object_report.get("video_sha256")) != manifest["video_sha256"] or
                phases.get("track") != 1 or phases.get("episode") != manifest["episode"] or
                phases.get("video_sha256") != manifest["video_sha256"]):
            parser.error("Surface-contact object and phases must identify the same Track 1 source")
        with np.load(object_path, allow_pickle=False) as archive:
            object_candidate = {name: archive[name] for name in archive.files}
        if not np.array_equal(np.asarray(object_candidate["camera_intrinsics"], dtype=np.float32), intrinsics):
            parser.error("Surface-contact object must use the identical fixed human camera")
        selected_surface = NativeSelectedHandSurface(hands, baseline["shape"], baseline["scales"], count_per_hand=128)
        surface_parity = selected_surface.verify_full_forward(
            baseline["pose"], positions=positions, gradient_positions=positions[:3])
        with torch.no_grad():
            baseline_hands = hands(torch.as_tensor(baseline["pose"], device=args.device),
                                   torch.as_tensor(baseline["scales"], device=args.device)).cpu().numpy()
        surface_contact = NativeHandSurfaceContact(object_candidate, phases, selected_surface,
                                                   baseline["pose"], baseline_hands,
                                                   baseline["frame_indices"], specification)
        surface_contact_provenance = {"configuration_path": str(args.surface_contact_config.resolve()),
                                      "configuration_sha256": digest(args.surface_contact_config),
                                      "path": str(args.surface_contact_config.resolve()),
                                      "sha256": digest(args.surface_contact_config),
                                      "object_sha256": specification["object_sha256"],
                                      "phases_sha256": specification["phases_sha256"],
                                      "baseline_sha256": specification["baseline_sha256"],
                                      "specification": specification, "native_surface_parity": surface_parity,
                                      "before": surface_contact.before,
                                      "interpretation": "Frozen native-skin geometric priors; no network training or hidden-contact truth"}
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
        anchor_reference = None
        contact_provenance["anchor_reference"] = None
        if args.contact_anchor_reference:
            anchor_reference = load_human_result(args.contact_anchor_reference)
            try:
                reference_validation = validate_contact_anchor_reference(anchor_reference, baseline, manifest,
                                                                          digest(args.model), intrinsics)
            except ValueError as error:
                parser.error(str(error))
            contact_provenance["anchor_reference"] = {
                "path": str(args.contact_anchor_reference.resolve()),
                "sha256": digest(args.contact_anchor_reference), "validation": reference_validation,
                "role": "Read-only native-MHR geometry used only to select fixed contact anchors; original SAM initialization and priors unchanged",
                "same_optimizer_initialization": True,
                "native_hand_mapping_parity": hands.verify_full_forward(anchor_reference, positions),
            }
        with torch.no_grad():
            baseline_hands = hands(torch.as_tensor(baseline["pose"], device=args.device),
                                   torch.as_tensor(baseline["scales"], device=args.device)).cpu().numpy()
            anchor_selection_hands = (baseline_hands if anchor_reference is None else
                                      hands(torch.as_tensor(anchor_reference["pose"], device=args.device),
                                            torch.as_tensor(anchor_reference["scales"], device=args.device)).cpu().numpy())
        try:
            contact = WeakContactConstraint(object_candidate, phase_report, anchor_selection_hands, baseline["frame_indices"],
                                            device=args.device, anchor_weight=args.contact_anchor_weight,
                                            penetration_weight=args.contact_penetration_weight)
            if anchor_reference is not None:
                contact_provenance["anchor_reference"]["anchor_selection_geometry_diagnostics"] = contact.before
                # Preserve before/after diagnostic semantics for the actual optimizer seed.
                # This does not enter the contact loss or change parameter availability.
                contact.before = contact.diagnostics(baseline_hands)
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
                          motion_prior_weights=motion_prior_weights, control_prior_support=control_prior_support,
                          surface_contact_constraint=surface_contact)
    report = fitted["report"]
    report.update({
        "track": 1, "episode": manifest["episode"], "complete_video": True,
        "source_video_sha256": manifest["video_sha256"], "baseline_sha256": digest(args.baseline),
        "observations_sha256": digest(args.observations), "mhr_model_sha256": digest(args.model),
        "observation_report_sha256": digest(observation_report_path),
        **target_provenance(observation_report, observations),
        "observation_source": str(args.observations.resolve()),
        "native_hand_mapping_parity": parity,
        "camera_intrinsics": intrinsics.tolist(), "intrinsics_status": manifest["intrinsics_status"],
        "source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in source_files.items()},
        "hand_only_development_probe": bool(args.skip_mesh_export),
        "contact_provenance": contact_provenance,
        "motion_phase_provenance": motion_phase_provenance,
        "control_prior_provenance": control_prior_provenance,
        "surface_contact_provenance": surface_contact_provenance,
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
    np.save(args.output / "original_active_hands.npy", fitted["original_active_hands"])
    np.save(args.output / "support_transition_envelope.npy", fitted["support_transition_envelope"])
    np.save(args.output / "image_support_transition_envelope.npy", fitted["image_support_transition_envelope"])
    np.save(args.output / "control_prior_support.npy", fitted["control_prior_support"])
    np.save(args.output / "arm_control_envelope.npy", fitted["arm_control_envelope"])
    np.save(args.output / "arm_control_delta_limits.npy", fitted["arm_control_delta_limits"])
    np.save(args.output / "wrist_displacement_limits_m.npy", fitted["wrist_displacement_limits_m"])
    np.save(args.output / "optimization_weights.npy", fitted["optimization_weights"])
    if surface_contact is not None:
        np.save(args.output / "surface_contact_phase_weights.npy", surface_contact.weights)
        np.savez(args.output / "surface_contact_selection.npz",
                 hand_vertex_ids=np.stack(surface_contact.surface.hand_vertex_ids),
                 patch_vertex_ids=np.stack(surface_contact.patch_native_vertex_ids),
                 left_native_vertices=surface_contact.surface.hand_vertex_ids[0],
                 right_native_vertices=surface_contact.surface.hand_vertex_ids[1],
                 left_patch_vertices=surface_contact.patch_native_vertex_ids[0],
                 right_patch_vertices=surface_contact.patch_native_vertex_ids[1])
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
