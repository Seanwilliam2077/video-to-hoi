"""Freeze shared training-only detector crops for authorized Track 1 EP11/16."""
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
from v2hoi.refined_egohand import build_refined_crops


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous attempts; output must not exist")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("track") != 1 or manifest.get("episode") not in (11, 16):
        parser.error("Only the two authorized Track 1 pilots are allowed")
    if "track_1" not in Path(manifest["video"]).parts or digest(manifest["video"]) != manifest["video_sha256"]:
        parser.error("Require the same explicit Track 1 source video")
    with np.load(args.observations, allow_pickle=False) as archive:
        observations = {key: archive[key] for key in archive.files}
    if str(observations["source_video_sha256"].item()) != manifest["video_sha256"]:
        parser.error("Observation source identity differs")
    if not np.array_equal(observations["frame_indices"], np.arange(manifest["video_frames"])):
        parser.error("Require all original source frame IDs")
    if not np.array_equal(observations["image_size"], [manifest["video_width"], manifest["video_height"]]):
        parser.error("Observation source image size differs")
    if not np.isclose(float(observations["fps"]), manifest["video_fps"], rtol=0, atol=1e-6):
        parser.error("Observation frame rate differs")
    boxes, arrays, report = build_refined_crops(observations)
    args.output.mkdir(parents=True)
    box_path = args.output / "crop_boxes.npy"
    np.save(box_path, boxes, allow_pickle=False)
    np.savez_compressed(args.output / "crop_anchors.npz", **arrays)
    source = [Path(__file__), ROOT / "src/v2hoi/refined_egohand.py", ROOT / "src/v2hoi/egohand_adapter.py"]
    report.update(recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  track=1, episode=manifest["episode"], source_video_sha256=manifest["video_sha256"],
                  image_size_wh=observations["image_size"].tolist(), side_order=["left", "right"],
                  all30_execution_authorized=False, crop_boxes_sha256=digest(box_path),
                  crop_anchors_sha256=digest(args.output / "crop_anchors.npz"),
                  input_sha256={"observations": digest(args.observations), "manifest": digest(args.manifest)},
                  source_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in source})
    with (args.output / "crop-policy-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for path in source:
        (snapshot / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"episode": manifest["episode"], "frames": report["frames"],
                      "training_anchor_frames": arrays["training_anchor_weights"].sum(0).tolist(),
                      "changed_hand_frames": report["changed_hand_frames"],
                      "revised_side_median": report["revised_side_median"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
