"""Retarget complete EP11/16 EgoHand outputs to native-depth image-supported proposals."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v2hoi.egohand_adapter import adapt_egohand_proposals
from v2hoi.hand_refinement import HAND_JOINTS
from v2hoi.refined_egohand import retarget_egohand_proposals

MEDIAPIPE_MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("left", "right", "scale-reference-left", "scale-reference-right",
                 "fresh-observations", "manifest", "baseline-review-arrays",
                 "crop-policy-receipt", "prediction-receipt", "output"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--variant", choices=("raw", "smoothed"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior attempts; output must not exist")
    if not re.fullmatch("[0-9a-f]{40}", args.upstream_commit):
        parser.error("Require a complete pinned upstream Git SHA")
    manifest = read(args.manifest)
    if manifest.get("track") != 1 or manifest.get("episode") not in (11, 16):
        parser.error("Only the two authorized Track 1 pilot episodes are allowed")
    if "track_1" not in Path(manifest["video"]).parts or digest(manifest["video"]) != manifest["video_sha256"]:
        parser.error("Explicit original Track 1 source identity differs")
    count = manifest["video_frames"]
    receipt = read(args.prediction_receipt)
    if (receipt.get("source_video_sha256") != manifest["video_sha256"]
            or receipt.get("frames") != count or receipt.get("complete_source_coverage") is not True
            or receipt.get("status") != "completed" or receipt.get("episode") != manifest["episode"]
            or receipt.get("upstream_commit") != args.upstream_commit):
        parser.error("Prediction receipt must bind the complete same video")
    crop_receipt = read(args.crop_policy_receipt)
    for key, expected in {"track": 1, "episode": manifest["episode"], "frames": count,
                          "source_video_sha256": manifest["video_sha256"],
                          "all30_execution_authorized": False}.items():
        if crop_receipt.get(key) != expected:
            parser.error(f"Crop policy binding differs: {key}")
    crop_path = args.crop_policy_receipt.parent / "crop_boxes.npy"
    if digest(crop_path) != crop_receipt["crop_boxes_sha256"]:
        parser.error("Frozen crop bytes changed")
    if receipt.get("input_bindings", {}).get("crop_boxes_sha256") != crop_receipt["crop_boxes_sha256"]:
        parser.error("Neural run did not use these frozen crop bytes")
    observations = load(args.fresh_observations)
    if str(observations["source_video_sha256"].item()) != manifest["video_sha256"]:
        parser.error("Fresh detector source identity differs")
    if not np.array_equal(observations["frame_indices"], np.arange(count)):
        parser.error("Fresh detector must retain complete original frame IDs")
    if not np.array_equal(observations["image_size"], [manifest["video_width"], manifest["video_height"]]):
        parser.error("Fresh detector image size differs")
    if not np.isclose(float(observations["fps"]), manifest["video_fps"], rtol=0, atol=1e-6):
        parser.error("Fresh detector frame rate differs")
    if not np.array_equal(observations["crop_boxes"], np.load(crop_path, allow_pickle=False)):
        parser.error("Fresh detector and neural run used different crop bounds")
    observer_report_path = args.fresh_observations.parent / "report.json"
    observer = read(observer_report_path)
    observer_checks = {
        "observer": "MediaPipe HandLandmarker independent 2D observations",
        "episode": manifest["episode"], "frames": count,
        "source_video_sha256": manifest["video_sha256"],
        "side_order": ["left", "right"],
        "image_size_wh": [manifest["video_width"], manifest["video_height"]],
        "model_sha256": MEDIAPIPE_MODEL_SHA256,
        "mirror_check": True, "require_handedness": False,
        "crop_boxes_sha256": crop_receipt["crop_boxes_sha256"],
        "crop_policy_receipt_sha256": digest(args.crop_policy_receipt),
        "all30_execution_authorized": False,
    }
    for key, expected in observer_checks.items():
        if observer.get(key) != expected:
            parser.error(f"Fresh detector report binding differs: {key}")
    if not np.isclose(float(observer["fps"]), manifest["video_fps"], rtol=0, atol=1e-6):
        parser.error("Fresh detector report frame rate differs")
    selected_records = receipt.get("raw_outputs" if args.variant == "raw" else "smoothed_outputs", [])
    raw_records = receipt.get("raw_outputs", [])
    for label, path, records in (("left", args.left, selected_records), ("right", args.right, selected_records),
                                 ("left", args.scale_reference_left, raw_records),
                                 ("right", args.scale_reference_right, raw_records)):
        bound = [x for x in records if x.get("side") == label and Path(x["path"]).resolve() == path.resolve()]
        if len(bound) != 1 or bound[0]["sha256"] != digest(path):
            parser.error("Proposal or raw scale-reference file is not bound by the complete inference receipt")
    frame_path = args.baseline_review_arrays / "frame_indices.npy"
    camera_path = args.baseline_review_arrays / "camera_intrinsics.npy"
    joint_path = args.baseline_review_arrays / "joints_3d.npy"
    indices = np.load(frame_path, allow_pickle=False)
    camera = np.load(camera_path, allow_pickle=False)
    if not np.array_equal(indices, observations["frame_indices"]) or not np.allclose(camera, manifest["camera_intrinsics"], rtol=0, atol=1e-5):
        parser.error("Native baseline frame mapping or common camera differs")
    native_report_path = args.baseline_review_arrays.parent / "report.json"
    native_report = read(native_report_path)
    if native_report.get("source_video_sha256", native_report.get("video_sha256")) != manifest["video_sha256"]:
        parser.error("Native baseline identifies a different video")
    joints = np.load(joint_path, mmap_mode="r", allow_pickle=False)
    if joints.shape != (count, 127, 3):
        parser.error("Require complete native baseline127-joint export")
    common = dict(frame_indices=observations["frame_indices"], intrinsics=camera,
                  image_size=observations["image_size"], crop_boxes=observations["crop_boxes"])
    proposal, initial_report = adapt_egohand_proposals({"left": load(args.left), "right": load(args.right)},
                                                     variant=args.variant, **common)
    reference, _ = adapt_egohand_proposals({"left": load(args.scale_reference_left), "right": load(args.scale_reference_right)},
                                          variant="raw", **common)
    arrays, report = retarget_egohand_proposals(proposal, observations, joints[:, HAND_JOINTS],
                                               intrinsics=camera, scale_reference=reference)
    arrays.update(fps=observations["fps"].copy(), source_video_sha256=observations["source_video_sha256"].copy())
    paths = {"left": args.left, "right": args.right, "scale_reference_left": args.scale_reference_left,
             "scale_reference_right": args.scale_reference_right, "fresh_observations": args.fresh_observations,
             "manifest": args.manifest, "crop_policy_receipt": args.crop_policy_receipt, "crop_boxes": crop_path,
             "prediction_receipt": args.prediction_receipt, "native_joints": joint_path,
             "native_camera": camera_path, "native_frame_indices": frame_path, "native_report": native_report_path,
             "fresh_observer_report": observer_report_path}
    source = [Path(__file__), ROOT / "src/v2hoi/refined_egohand.py", ROOT / "src/v2hoi/egohand_adapter.py",
              ROOT / "src/v2hoi/hand_refinement.py"]
    report.update(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  track=1, episode=manifest["episode"], variant=args.variant,
                  source_video_sha256=manifest["video_sha256"], upstream_commit=args.upstream_commit,
                  original_geometry_adapter=initial_report, all30_execution_authorized=False,
                  fresh_observer_metadata=observer_checks,
                  camera_quality="Shared Track 1 estimate; native metric depth and scale are not calibrated truth.",
                  input_sha256={name: digest(path) for name, path in paths.items()},
                  source_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in source})
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "observations.npz", **arrays)
    report["output_sha256"] = digest(args.output / "observations.npz")
    with (args.output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for path in source:
        (snapshot / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"episode": manifest["episode"], "variant": args.variant, "frames": count,
                      "fixed_scale_anchor_frames": report["fixed_scale_anchor_frames"],
                      "fixed_native_to_ego_palm_scalar": report["fixed_native_to_ego_palm_scalar"],
                      "new_image_target_frames": report["new_image_target_frames"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
