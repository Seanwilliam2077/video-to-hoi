"""Freeze ROUTER02 shared hand crops while preserving the original/tight trials."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v2hoi.hand_crop_router import route_hand_crops

MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("old-observations", "trial-observations", "trial-crop-policy-receipt", "manifest", "output"):
        parser.add_argument("--"+flag, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior attempts; output must not exist")
    manifest = read(args.manifest)
    if manifest.get("track") != 1 or manifest.get("episode") not in (11, 16):
        parser.error("Only the two authorized Track 1 pilot episodes are allowed")
    if "track_1" not in Path(manifest["video"]).parts or digest(manifest["video"]) != manifest["video_sha256"]:
        parser.error("Require the unchanged explicit Track 1 source video")
    count = manifest["video_frames"]
    tight_receipt = read(args.trial_crop_policy_receipt)
    for key, expected in {"track":1, "episode":manifest["episode"], "frames":count,
                          "source_video_sha256":manifest["video_sha256"],
                          "all30_execution_authorized":False}.items():
        if tight_receipt.get(key) != expected:
            parser.error(f"Tight crop receipt binding differs: {key}")
    tight_path = args.trial_crop_policy_receipt.parent / "crop_boxes.npy"
    if digest(tight_path) != tight_receipt["crop_boxes_sha256"]:
        parser.error("Immutable tight crop bytes changed")
    observations, reports = {}, {}
    for label, path in (("original_wide", args.old_observations), ("tight_trial01", args.trial_observations)):
        obs = load(path)
        if (str(obs["source_video_sha256"].item()) != manifest["video_sha256"]
                or not np.array_equal(obs["frame_indices"], np.arange(count))
                or not np.array_equal(obs["image_size"], [manifest["video_width"],manifest["video_height"]])
                or not np.isclose(float(obs["fps"]), manifest["video_fps"], rtol=0, atol=1e-6)):
            parser.error(f"Observer source/timeline differs: {label}")
        report = read(path.parent / "report.json")
        for key, expected in {"observer":"MediaPipe HandLandmarker independent 2D observations",
                              "episode":manifest["episode"], "frames":count,
                              "source_video_sha256":manifest["video_sha256"],
                              "side_order":["left","right"], "model_sha256":MODEL_SHA256,
                              "mirror_check":True, "require_handedness":False}.items():
            if report.get(key) != expected:
                parser.error(f"Observer report binding differs: {label}/{key}")
        observations[label], reports[label] = obs, report
    trial = observations["tight_trial01"]
    if not np.array_equal(trial["crop_boxes"], np.load(tight_path, allow_pickle=False)):
        parser.error("Trial detector did not use the frozen tight crop bytes")
    if (reports["tight_trial01"].get("crop_boxes_sha256") != tight_receipt["crop_boxes_sha256"]
            or reports["tight_trial01"].get("crop_policy_receipt_sha256") != digest(args.trial_crop_policy_receipt)):
        parser.error("Trial detector report does not bind the tight crop receipt")
    if reports["original_wide"]["mediapipe_version"] != reports["tight_trial01"]["mediapipe_version"]:
        parser.error("Observer versions changed between the crop trials")
    boxes, arrays, report = route_hand_crops(observations["original_wide"], trial)
    args.output.mkdir(parents=True)
    np.save(args.output / "crop_boxes.npy", boxes, allow_pickle=False)
    np.savez_compressed(args.output / "crop_anchors.npz", **arrays)
    source = [Path(__file__), ROOT/"src/v2hoi/hand_crop_router.py", ROOT/"src/v2hoi/refined_egohand.py",
              ROOT/"src/v2hoi/egohand_adapter.py"]
    paths = {"original_wide_observations":args.old_observations, "tight_trial_observations":args.trial_observations,
             "original_observer_report":args.old_observations.parent/"report.json",
             "tight_observer_report":args.trial_observations.parent/"report.json",
             "source_manifest":args.manifest, "tight_crop_policy_receipt":args.trial_crop_policy_receipt,
             "tight_crop_boxes":tight_path}
    report.update(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  track=1, episode=manifest["episode"], source_video_sha256=manifest["video_sha256"],
                  image_size_wh=trial["image_size"].tolist(), side_order=["left","right"],
                  crop_boxes_sha256=digest(args.output/"crop_boxes.npy"),
                  crop_anchors_sha256=digest(args.output/"crop_anchors.npz"),
                  input_sha256={name:digest(path) for name,path in paths.items()},
                  source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in source},
                  tight_trial_policy=tight_receipt["policy"], all30_execution_authorized=False)
    with (args.output/"crop-policy-receipt.json").open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    snapshot=args.output/"source_snapshot"
    snapshot.mkdir()
    for path in source:
        (snapshot/path.name).write_bytes(path.read_bytes())
    print(json.dumps({"episode":manifest["episode"],"frames":count,
                      "route_tight_frames":report["route_tight_frames"],
                      "new_mixed_pair_duplicate_frames":len(report["new_mixed_pair_duplicate_original_frame_ids"]),
                      "output":str(args.output)}))


if __name__=="__main__":
    main()
