"""Recheck completed new-pilot artifact integrity without reconstruction acceptance."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


TASK = Path(__file__).resolve().parent
RESULTS = Path("J:/video-to-hoi/runs/egohand-refined-pilots-20261031")
ATTEMPT = "02"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(item):
    path = Path(item["path"])
    actual = sha(path)
    if actual != item["sha256"]:
        raise ValueError(f"Artifact changed: {path}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def projection_consistency(item, spec):
    manifest = read(item["manifest"])
    K = np.asarray(manifest["camera_intrinsics"], dtype=np.float64)
    boxes = np.load(spec["crop_boxes"]["path"], allow_pickle=False)
    source = Path(spec["prediction_root"])
    receipt = read(source / "prediction-receipt.json")
    output = []
    for variant, field in (("raw", "raw_outputs"), ("smoothed", "smoothed_outputs")):
        for row in receipt[field]:
            binding = verify(row)
            side = ("left", "right").index(row["side"])
            with np.load(row["path"], allow_pickle=False) as data:
                ids = data["frame_names"].astype(str).tolist()
                if ids != [f"{i:06d}" for i in range(item["frames"])]:
                    raise ValueError("Upstream prediction original IDs differ")
                camera = data["joints"].astype(np.float64) + data["cam_t"].astype(np.float64)[:, None]
                predicted = camera @ K.T
                if not np.isfinite(predicted).all() or np.any(predicted[..., 2] <= 0):
                    raise ValueError("Invalid upstream camera geometry")
                predicted = predicted[..., :2] / predicted[..., 2:]
                b = boxes[:, side]
                center = (b[:, :2] + b[:, 2:]) / 2
                size = (b[:, 2] - b[:, 0])[:, None, None]
                upstream = data["pred_keypoints_2d"].astype(np.float64) * size + center[:, None]
                difference = np.linalg.norm(predicted - upstream, axis=-1)
                maximum = float(difference.max())
                if maximum >= .001:
                    raise ValueError("Source-K/crop projection convention mismatch")
                output.append({"variant": variant, "side": row["side"], "frames": item["frames"],
                    "npz": binding, "mean_difference_pixels": float(difference.mean()),
                    "maximum_difference_pixels": maximum, "coordinate_consistent_not_accuracy_claim": True})
    return output


def updates_by_hand(item, evaluation):
    with np.load(item["baseline"]["path"], allow_pickle=False) as source:
        baseline = source["pose"].astype(np.float64)
    result = {}
    for role, metadata in evaluation["native_artifact_bindings"].items():
        if role == "baseline":
            continue
        path = Path(metadata["directory"]) / f"episode_{item['episode']:06d}_human.npz"
        with np.load(path, allow_pickle=False) as source:
            delta = source["pose"].astype(np.float64) - baseline
        result[role] = {}
        for side, finger, arm in (("left", slice(95, 122), list(range(43, 50))),
                                  ("right", slice(68, 95), list(range(33, 40)))):
            fingers = np.abs(delta[:, finger])
            arms = np.abs(delta[:, arm])
            maximum = np.maximum(fingers.max(-1), arms.max(-1))
            changed = maximum > 1e-7
            result[role][side] = {"changed_original_frame_count": int(changed.sum()),
                "zero_update_hand": bool(not changed.any()), "maximum_finger_control_delta_radians": float(fingers.max()),
                "p95_absolute_finger_control_delta_radians": float(np.percentile(fingers, 95)),
                "maximum_arm_control_delta_radians": float(arms.max()),
                "interpretation": "Magnitude of the retained control correction, not actual hand motion or accuracy; an unchanged hand does not demonstrate an improvement"}
    return result


def fresh_chunk_provenance(item, spec, plan):
    root = Path(spec["prediction_root"])
    receipt = read(root / "prediction-receipt.json")
    fields = ("hand_poses", "identity_coeffs", "global_scale", "vertices", "joints", "cam_t", "scale_params",
              "pred_keypoints_2d", "pred_projection_valid", "pred_kp2d_crop_presence", "pred_kp2d_sigma")
    merged, seen = {}, {"left": [], "right": []}
    for row in receipt["raw_outputs"]:
        verify(row)
        with np.load(row["path"], allow_pickle=False) as data:
            merged[row["side"]] = {name: data[name].copy() for name in fields}
    checked = []
    for chunk in receipt["chunks"]:
        for row in chunk["outputs"]:
            path = Path(row["path"]).resolve()
            if not path.is_relative_to((root / "chunks").resolve()):
                raise ValueError("Fresh neural chunk escaped its isolated attempt")
            file_binding = verify(row)
            with np.load(path, allow_pickle=False) as data:
                ids = data["frame_names"].astype(str).tolist()
                if not set(ids) <= set(chunk["original_frame_ids"]):
                    raise ValueError("Fresh chunk original IDs differ from its recorded forward call")
                indices = list(map(int, ids))
                for name in fields:
                    if not np.array_equal(merged[row["side"]][name][indices], data[name]):
                        raise ValueError("Merged RAW recomputed rows differ from fresh chunk output")
                seen[row["side"]].extend(indices)
            checked.append({"side": row["side"], "original_frame_ids": indices, "file": file_binding,
                            "all_exported_fields_exactly_equal_to_merged_raw": True})
    for side in ("left", "right"):
        if len(seen[side]) != len(set(seen[side])) or sorted(seen[side]) != plan["neural_recomputed_original_frame_ids_per_side"][side]:
            raise ValueError("Fresh chunk row IDs must exactly equal the unmatched source-input complement")
    return {"checked_chunks": checked, "fresh_neural_output_views": sum(map(len, seen.values())),
            "new_view_ids_exactly_match_cache_complement": True,
            "interpretation": "Fresh chunk/merged-array/source lineage; not reconstruction accuracy"}


def metadata_errata_bridge(path, expected_hash):
    manifest_binding = verify({"path": str(path), "sha256": expected_hash})
    manifest = read(path)
    if manifest.get("track") != 1 or manifest.get("numeric_parameter_or_geometry_change") is not False:
        raise ValueError("Metadata correction must preserve numeric reconstruction")
    verify({"path": manifest["helper"], "sha256": manifest["helper_sha256"]})
    verify({"path": manifest["completion"], "sha256": manifest["completion_sha256"]})
    allowed = {"method", "observation_semantics", "independent_observation_provenance",
               "generated_model_proposal_provenance", "metadata_correction"}
    output = []
    for case in manifest["episodes"]:
        if case["episode"] not in (11, 16):
            raise ValueError("Unexpected metadata-correction episode")
        canonical = verify({"path": case["canonical_npz"], "sha256": case["canonical_npz_sha256"]})
        original = verify({"path": case["original_npz"], "sha256": case["original_npz_sha256"]})
        corrected_report = verify({"path": case["canonical_report"], "sha256": case["canonical_report_sha256"]})
        receipt = verify({"path": case["correction_receipt"], "sha256": case["correction_receipt_sha256"]})
        before, after = read(case["original_report"]), read(case["canonical_report"])
        changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
        if not changed <= allowed or before["source_sha256"] != after["source_sha256"]:
            raise ValueError("Canonical report changed non-provenance fields or fictionalized executed source")
        if after["method"] != "source_bound_model_target_native_mhr_sequence_fit" or after.get("independent_observation_provenance") is not None:
            raise ValueError("Canonical generated-target semantics remain ambiguous")
        if after["generated_model_proposal_provenance"] != before["independent_observation_provenance"]:
            raise ValueError("Canonical provenance must retain the original target receipt without changing evidence")
        arrays = []
        with zipfile.ZipFile(case["original_npz"]) as original_zip, zipfile.ZipFile(case["canonical_npz"]) as corrected_zip:
            if set(original_zip.namelist()) != set(corrected_zip.namelist()):
                raise ValueError("Canonical NPZ added or deleted arrays")
            for name in original_zip.namelist():
                if name == "report.npy":
                    continue
                first, second = original_zip.read(name), corrected_zip.read(name)
                if first != second:
                    raise ValueError("Canonical NPZ changed a numeric/identity/frame member")
                arrays.append({"member": name, "exact_member_bytes_unchanged": True,
                               "sha256": hashlib.sha256(first).hexdigest()})
        with np.load(case["original_npz"], allow_pickle=False) as first, np.load(case["canonical_npz"], allow_pickle=False) as second:
            old_embedded = json.loads(str(first["report"].item()))
            new_embedded = json.loads(str(second["report"].item()))
            changes = {key for key in set(old_embedded) | set(new_embedded) if old_embedded.get(key) != new_embedded.get(key)}
            if not changes <= allowed:
                raise ValueError("Canonical NPZ embedded report changed non-provenance content")
            for name in first.files:
                if name != "report" and (first[name].dtype != second[name].dtype or first[name].shape != second[name].shape or first[name].tobytes() != second[name].tobytes()):
                    raise ValueError("Canonical array dtype/shape/payload changed")
        output.append({"episode": case["episode"], "original": original, "canonical": canonical,
                       "canonical_report": corrected_report, "correction_receipt": receipt,
                       "changed_external_report_keys": sorted(changed), "numeric_member_bridge": arrays,
                       "evaluation_and_video_remain_bound_to_executed_original": True})
    return {"manifest": manifest_binding, "episodes": output,
            "numeric_arrays_and_executed_source_unchanged": True, "geometry_or_fit_rerun": False}


def episode(cfg, item, spec):
    number = item["episode"]
    root = RESULTS / f"episode_{number:06d}"
    completion_path = root / f"postprocessing-completion{ATTEMPT}.json"
    completion = read(completion_path)
    if completion["status"] != "complete_numeric_and_video_evidence_awaiting_visual_review":
        raise ValueError("Pilot postprocessing is incomplete")
    stages = []
    for stage in (f"fit-mp{ATTEMPT}", f"adapt-raw{ATTEMPT}", f"adapt-smoothed{ATTEMPT}", f"fit-egohand{ATTEMPT}", f"evaluation{ATTEMPT}", f"render{ATTEMPT}"):
        path = TASK / f"ep{number}-{stage}.finished.json"
        receipt = read(path)
        if receipt["returncode"] != 0:
            raise ValueError("A completed stage failed")
        stages.append({"stage": stage, "receipt": {"path": str(path), "sha256": sha(path)},
                       "returncode": 0, "wall_seconds": receipt["wall_seconds"],
                       "input_bindings_reverified": [verify(value) for value in receipt["input_bindings"]]})
    originals = [verify(item["baseline"]), verify({"path": item["video"], "sha256": item["video_sha256"]}),
                 verify({"path": item["manifest"], "sha256": item["manifest_sha256"]}),
                 verify({"path": item["original_observations"], "sha256": item["original_observations_sha256"]})]
    for role in ("legacy_mediapipe", "legacy_egohand"):
        originals += [{"role": role, **verify(spec[role])},
                      verify({"path": str(Path(spec[role]["path"]).parent / "report.json"), "sha256": spec[role]["report_sha256"]})]
    crops = {name: verify(spec[name]) for name in ("crop_boxes", "crop_receipt", "crop_anchors", "review_windows")}
    with np.load(spec["crop_anchors"]["path"], allow_pickle=False) as anchors:
        if not np.array_equal(anchors["frame_indices"], np.arange(item["frames"])) or np.any(anchors["training_anchor_weights"][::5] > 0):
            raise ValueError("Crop anchor stride-five exclusion changed")
    evaluation_path = root / f"evaluation{ATTEMPT}.json"
    evaluation = read(evaluation_path)
    if evaluation["frames"] != item["frames"] or not all(value["passed"] and value["frames_freshly_decoded"] == item["frames"] for value in evaluation["native_decoder"].values()):
        raise ValueError("Full native decoder parity incomplete")
    native = {}
    for role, metadata in evaluation["native_artifact_bindings"].items():
        native[role] = [verify({"path": str(Path(metadata["directory"]) / relative), "sha256": row["sha256"]})
                        for relative, row in metadata["files"].items()]
    summary_path = root / f"review{ATTEMPT}/summary.json"
    summary = read(summary_path)
    if summary["frames"] != item["frames"] or summary["source_video_sha256"] != item["video_sha256"]:
        raise ValueError("Video source binding differs")
    videos = []
    for row in summary["videos"]:
        expected = item["frames"]
        if row["file"].startswith("critical/"):
            first, last = map(int, Path(row["file"]).stem.split("-")[-2:])
            expected = last - first + 1
        if row["decoded_frames"] != expected:
            raise ValueError("Video full-decode frame receipt differs")
        videos.append({**row, **verify({"path": str(root / f"review{ATTEMPT}" / row["file"]), "sha256": row["sha256"]})})
    roi = verify({"path": str(root / f"review{ATTEMPT}" / summary["display_roi"]["file"]), "sha256": summary["display_roi"]["sha256"]})
    prediction_verification_path = TASK / f"ep{number}-prediction-verification{ATTEMPT}.json"
    prediction_verification = read(prediction_verification_path)
    prediction_files = [verify(row) for row in prediction_verification["files"]]
    return {"episode": number, "frames": item["frames"], "completed_stages": stages,
        "original_artifacts_unchanged": originals, "frozen_crop_and_window_artifacts": crops,
        "crop_anchor_holdout_exclusion_verified": True, "native_variants_reverified": native,
        "retained_native_corrections_by_hand": updates_by_hand(item, evaluation),
        "native_evaluation": {"path": str(evaluation_path), "sha256": sha(evaluation_path)},
        "retarget_provenance_verification": {"path": str(TASK / f"ep{number}-retarget-verification{ATTEMPT}.json"), "sha256": sha(TASK / f"ep{number}-retarget-verification{ATTEMPT}.json")},
        "video_summary": {"path": str(summary_path), "sha256": sha(summary_path)},
        "fully_decoded_video_receipts_and_reverified_file_bytes": videos,
        "full_video_decode_repeated_by_this_audit": False, "shared_display_rois": roi,
        "prediction_provenance_verification": {"path": str(prediction_verification_path), "sha256": sha(prediction_verification_path)},
        "prediction_files_reverified": prediction_files,
        "exact_raw_cache_plan": prediction_verification["raw_cache_plan"],
        "actual_reused_raw_array_equality_checks": prediction_verification["reused_raw_value_checks"],
        "fresh_recomputed_chunk_provenance": fresh_chunk_provenance(item, spec, prediction_verification["raw_cache_plan"]),
        "full_clip_smoothing_recomputed_source_and_receipt_evidence": True,
        "upstream_full_projection_consistency": projection_consistency(item, spec),
        "quality_accepted": False, "contact_quality_accepted": False, "official_score_available": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--episode", type=int, choices=(11, 16), action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-errata", type=Path)
    parser.add_argument("--metadata-errata-sha256")
    args = parser.parse_args()
    if args.output.exists() or len(set(args.episode)) != len(args.episode):
        raise ValueError("Preserve earlier audits and do not repeat episodes")
    if bool(args.metadata_errata) != bool(args.metadata_errata_sha256):
        raise ValueError("Errata manifest requires its immutable SHA binding")
    verify({"path": str(args.config), "sha256": args.config_sha256})
    cfg = read(args.config)
    if cfg["track"] != 1 or cfg["all30_execution_authorized"] is not False:
        raise ValueError("Wrong task scope")
    for name in ("runtime_snapshot", "input_manifest", "protocol", "renderer", "inference_runner", "raw_cache_contract", "raw_cache_proof", "raw_cache_sparse_export_proof"):
        verify(cfg[name])
    runtime = Path(cfg["runtime_root"])
    snapshot = read(cfg["runtime_snapshot"]["path"])
    frozen = [verify({"path": str(runtime / row["path"]), "sha256": row["sha256"]}) for row in snapshot["files"]]
    sources = {entry["episode"]: entry for entry in read(cfg["input_manifest"]["path"])["episodes"]}
    specs = {entry["episode"]: entry for entry in cfg["episodes"]}
    episodes = [episode(cfg, sources[number], specs[number]) for number in args.episode]
    cache_total = sum(row["exact_raw_cache_plan"]["cache_hits"] for row in episodes)
    new_total = sum(row["exact_raw_cache_plan"]["neural_recomputed_views"] for row in episodes)
    if set(args.episode) == {11, 16} and (cache_total, new_total) != (2160, 314):
        raise ValueError("Final total RAW reuse/recomputation differs from reviewed2160+314")
    result = {"recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(), "track": 1, "attempt": ATTEMPT,
        "authorized_episodes": args.episode, "helper_sha256": sha(__file__), "config": verify({"path": str(args.config), "sha256": args.config_sha256}),
        "all30_execution_authorized": False, "automatic_promotion": False,
        "frozen_runtime_files_reverified": frozen,
        "episodes": episodes, "original_hand_views": cache_total + new_total,
        "exact_prior_raw_cache_views": cache_total, "new_neural_hand_views": new_total,
        "all_original_hand_views_neurally_recomputed": False,
        "executed_refiner_metadata_erratum": {
            "present": True,
            "actual_numeric_reports_and_NPZ_remain_immutable": True,
            "affected_executed_fields": ["independent_observation_provenance", "observation_semantics"],
            "correct_interpretation": "Refined EgoHand targets are generated geometric proposals sharing training-only MP wrist/scale/image-support inputs; they are not independent image detections.",
            "cause": "The executed frozen refiner recognized only the earlier report schema family for provenance wording; numerical inputs, target bindings and optimization are unchanged.",
            "canonical_correction_manifest": (metadata_errata_bridge(args.metadata_errata, args.metadata_errata_sha256) if args.metadata_errata else None)},
        "scope": "Source/coverage/representation/provenance/file integrity; no hidden truth or reconstruction acceptance"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "episodes": args.episode}))


if __name__ == "__main__":
    main()
