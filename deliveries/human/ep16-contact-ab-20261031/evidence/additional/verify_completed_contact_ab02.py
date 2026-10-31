"""Independently recheck completed EP16 numeric, source and video bindings."""
from pathlib import Path
import datetime as dt
import hashlib
import json

import numpy as np

TASK = Path(__file__).resolve().parent
RESULTS = Path("J:/video-to-hoi/runs/ep16-contact-ab-20261031")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def check(row):
    path = Path(row["path"])
    assert path.is_file() and digest(path) == row["sha256"], str(path)
    if "bytes" in row:
        assert path.stat().st_size == row["bytes"]


def main():
    destination = TASK / "protocol-review01/combined-integrity02.json"
    assert not destination.exists()
    snapshots = []
    for folder in ("runtime01", "runtime02", "evaluation-runtime01", "evaluation-runtime02", "render-runtime02", "closeup-runtime02"):
        root = TASK / folder
        snap = read(root / "snapshot.json")
        for row in snap["files"]:
            check({**row, "path": str(root / row["path"])})
        snapshots.append({"root": str(root), "manifest": binding(root / "snapshot.json"),
                          "all_declared_files_rehashed": True, "files": len(snap["files"])})
    source_pin = read(TASK / "object-runtime-source-pin01.json")
    for relative, sha in source_pin["files"].items():
        assert digest(TASK / "object-runtime01" / relative) == sha
    assert digest(TASK / "run_object_pipeline01.py") == source_pin["runner_sha256"]
    obj_done = read(TASK / "object-pipeline-completed01.json")
    assert obj_done["status"] == "completed_with_environment_only_export_recovery"
    assert digest(Path(obj_done["contact_ready"])) == obj_done["contact_ready_sha256"]
    object_root = Path(obj_done["contact_ready"]).parent
    assert digest(object_root / "geometry.ply") == obj_done["geometry_ply_sha256"]
    assert digest(object_root / "report.json") == obj_done["object_report_sha256"]
    assert digest(object_root / "validation.json") == obj_done["validation_sha256"]
    assert digest(TASK / "object-runtime-source-pin01.json") == obj_done["source_pin_sha256"]
    object_stages = []
    for name in ("object-initial01", "object-temporal01", "object-validation01", "object-contact-export-recovery01", "object-samples-recovery01"):
        path = TASK / (name + ".finished.json")
        assert read(path)["returncode"] == 0
        object_stages.append(binding(path))
    original_failure = TASK / "object-contact-export01.finished.json"
    assert read(original_failure)["returncode"] != 0
    fits, evaluations = [], []
    for version in (1, 2):
        plan_path = TASK / f"ab-plan{version:02d}.json"
        plan = read(plan_path)
        assert plan["track"] == 1 and plan["episode"] == 16 and plan["frames"] == 360
        for row in plan["input_bindings"] + plan["fitter_source_bindings"]:
            check(row)
        for arm in ("A", "B"):
            finished = TASK / f"ab-plan{version:02d}-{arm}.finished.json"
            receipt = read(finished)
            assert receipt["returncode"] == 0 and receipt["status"] == "fit_completed"
            assert receipt["plan_sha256"] == digest(plan_path)
            assert receipt["frozen_inputs_and_sources_unchanged"]
            fits.append({"revision": version, "arm": arm, "receipt": binding(finished),
                         "native": binding(Path(plan["outputs"][arm]) / "episode_000016_human.npz")})
        inputs = TASK / f"evaluation-inputs{version:02d}.json"
        spec = read(inputs)
        evaluated_path = RESULTS / f"evaluation{version:02d}.json"
        evaluated = read(evaluated_path)
        completed_path = TASK / f"evaluation{version:02d}.finished.json"
        completed = read(completed_path)
        assert completed["returncode"] == 0 and completed["input_and_runtime_source_bytes_unchanged"]
        assert completed["output_sha256"] == digest(evaluated_path)
        assert evaluated["frames"] == 360 and evaluated["input_manifest_sha256"] == digest(inputs)
        assert evaluated["controlled_pair"] == ["a", "b"]
        assert all(evaluated["native_decoder"][r]["passed"] and evaluated["native_decoder"][r]["frames_freshly_decoded"] == 360 for r in ("baseline", "a", "b"))
        assert all(evaluated["historical_no_contact_numeric_parity"].values())
        assert evaluated["matched_controls"]["exact_active_hand_masks_identical"]
        assert evaluated["matched_controls"]["exact_image_weights_identical"]
        for key in ("source_video", "manifest", "baseline", "a", "b", "historical_no_contact", "legacy_reference", "fresh_reference", "object", "object_report", "phases", "criteria", "protocol"):
            check(spec[key])
        for role in ("baseline", "a", "b"):
            for name, sha in evaluated["native_review_array_sha256"][role].items():
                assert digest(Path(spec[role]["path"]).parent / "review_arrays" / (name + ".npy")) == sha
        if version == 2:
            check(spec["contact_anchor_reference"])
            assert evaluated["anchor_reference_contract"]["same_identity_scales_root_frames_camera_decoder_verified"]
        evaluations.append({"revision": version, "receipt": binding(completed_path),
                            "input_manifest": binding(inputs), "evaluation": binding(evaluated_path),
                            "all360_native_parity_and_review_array_hashes_rechecked": True})
    latest = read(TASK / "evaluation-inputs02.json")
    camera = np.asarray(read(latest["manifest"]["path"])["camera_intrinsics"], np.float32)
    with np.load(latest["object"]["path"], allow_pickle=False) as archive:
        assert np.array_equal(archive["frame_indices"], np.arange(360))
        assert np.array_equal(archive["camera_intrinsics"], camera)
        assert archive["object_to_camera"].shape == (360, 4, 4)
        assert np.isfinite(archive["object_to_camera"]).all()
        object_pose_gate_count = int(archive["pose_usable"].sum())
    videos = []
    for stem, folder in (("render02", "review02"), ("closeup-render02", "review-closeup02")):
        finished_path = TASK / (stem + ".finished.json")
        receipt = read(finished_path)
        assert receipt["returncode"] == 0 and receipt["frozen_sources_unchanged"]
        root = RESULTS / folder
        summary = read(root / "summary.json")
        assert summary["frames"] == 360 and len(summary["videos"]) == 3
        assert summary["input_manifest_sha256"] == digest(TASK / "evaluation-inputs02.json")
        assert summary["evaluation_sha256"] == digest(RESULTS / "evaluation02.json")
        assert sorted(r["frames_fully_decoded"] for r in summary["videos"]) == [71, 71, 360]
        for row in summary["videos"]:
            path = root / row["file"]
            assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
        for row in summary["actually_saved_static_samples"]:
            assert digest(root / row["file"]) == row["sha256"]
        videos.append({"review": folder, "receipt": binding(finished_path), "video_summary": binding(root / "summary.json"),
                       "videos": summary["videos"], "sample_images_rehashed": len(summary["actually_saved_static_samples"])})
    clarification_path = TASK / "review-index-clarification02.json"
    for row in read(clarification_path)["outputs"]:
        assert digest(Path(row["directory"]) / "index.html") == row["current_index_sha256"]
        assert digest(row["original_backup"]) == row["original_index_sha256"]
        assert row["all_video_bytes_unchanged"]
    static_path = TASK / "protocol-review01/final-static-review02.json"
    static = read(static_path)
    assert static["primary_closeup_images_actually_viewed"] == 20 and static["context_images_actually_viewed"] == 10
    for row in static["actual_image_views"]:
        check(row)
    numeric_path = TASK / "protocol-review01/evaluation02-independent-review01.json"
    numeric = read(numeric_path)
    assert numeric["contact_arrays_reproduced_exactly_from_frozen_manual_phases_object_gate_and_actual_recorded_anchors"]
    for name, sha in numeric["contact_arrays_sha256"].items():
        assert digest(Path(latest["b"]["path"]).parent / (name + ".npy")) == sha
    repo = Path("C:/Users/arthasxu/.codex/worktrees/human-all30/video-to-hoi")
    status_sha = digest(repo / "docs/status.html")
    assert status_sha == read(TASK / "task-start.json")["preexisting_unrelated_status_sha256"]
    result = {"recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(), "schema_version": "ep16_contact_ab_completed_integrity.v1",
        "scope": "Read-only completed source/artifact/stage/hash consistency; not physical accuracy or automatic quality acceptance",
        "track": 1, "authorized_episodes": [16], "frames": 360,
        "source_video_sha256": read(TASK / "ab-plan02.json")["source_video_sha256"],
        "source_snapshots": snapshots, "object_runtime_source_files_rehashed": len(source_pin["files"]),
        "object_completion": binding(TASK / "object-pipeline-completed01.json"), "object_successful_stages": object_stages,
        "preserved_original_object_export_failure": binding(original_failure), "object_pose_usable_frames": object_pose_gate_count,
        "actual_native_fits": fits, "completed_independent_evaluations": evaluations, "reviews": videos,
        "videos_fully_decoded_and_current_bytes_rehashed": 6,
        "html_only_clarification": binding(clarification_path),
        "final_static_review": binding(static_path), "independent_numeric_review": binding(numeric_path),
        "contact_arrays_exact_reconstruction_and_current_hashes_rechecked": True,
        "same_camera_complete_original_ids_verified": True,
        "preexisting_unrelated_status_sha256": status_sha,
        "quality_accepted": False, "contact_quality_accepted": False, "scale_accepted": False,
        "official_score_available": False, "contact_model_training_started": False,
        "episode11_execution_started": False, "all30_execution_authorized": False,
        "main_merge_or_deployment_performed": False}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"integrity": str(destination), "sha256": digest(destination), "videos": 6,
                      "fits": 4, "native_evaluations": 2, "status_page_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
