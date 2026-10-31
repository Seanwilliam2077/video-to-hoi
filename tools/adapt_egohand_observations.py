"""Bind EgoHand camera geometry to the two authorized Track 1 hand pilots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v2hoi.egohand_adapter import adapt_egohand_proposals


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--frozen-observations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--prediction-receipt", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--variant", choices=("raw", "smoothed"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not exist; preserve previous attempts")
    if not re.fullmatch("[0-9a-f]{40}", args.upstream_commit):
        parser.error("upstream-commit must be a complete pinned Git SHA")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("track") != 1 or manifest.get("episode") not in (11, 16):
        parser.error("Only Track 1 episodes 11 and 16 are authorized for these pilots")
    receipt = json.loads(args.prediction_receipt.read_text(encoding="utf-8"))
    if receipt.get("source_video_sha256") != manifest["video_sha256"] or receipt.get("frames") != manifest["video_frames"]:
        parser.error("Prediction receipt must identify the same complete source video")
    frozen = load_npz(args.frozen_observations)
    if frozen["source_video_sha256"].item() != manifest["video_sha256"]:
        parser.error("Frozen crops identify a different source video")
    if frozen["side_order"].tolist() != ["left", "right"]:
        parser.error("Frozen crop order must be anatomical left/right")
    if not np.array_equal(frozen["frame_indices"], np.arange(manifest["video_frames"])):
        parser.error("Frozen observations must retain every original video frame")
    if not np.array_equal(frozen["image_size"], [manifest["video_width"], manifest["video_height"]]):
        parser.error("Frozen crop image size differs from the source manifest")
    if not np.isclose(float(frozen["fps"]), manifest["video_fps"], atol=1e-6, rtol=0):
        parser.error("Frozen observations have a different frame rate")
    intrinsics = np.load(args.intrinsics, allow_pickle=False)
    if not np.allclose(intrinsics, manifest["camera_intrinsics"], atol=1e-5, rtol=0):
        parser.error("Intrinsics differ from the common estimated camera")
    arrays, report = adapt_egohand_proposals(
        {"left": load_npz(args.left), "right": load_npz(args.right)},
        frame_indices=frozen["frame_indices"], intrinsics=intrinsics,
        image_size=frozen["image_size"], crop_boxes=frozen["crop_boxes"], variant=args.variant,
    )
    arrays.update(fps=frozen["fps"].copy(), source_video_sha256=frozen["source_video_sha256"].copy())
    report.update(track=1, episode=manifest["episode"], source_video_sha256=manifest["video_sha256"],
                  upstream_commit=args.upstream_commit, prediction_receipt=receipt,
                  camera_quality="Same Track 1 estimate as MediaPipe/MHR; not measured calibration.",
                  all30_execution_authorized=False,
                  input_sha256={name: digest(path) for name, path in {
                      "left": args.left, "right": args.right, "frozen_observations": args.frozen_observations,
                      "manifest": args.manifest, "intrinsics": args.intrinsics,
                      "prediction_receipt": args.prediction_receipt,
                  }.items()})
    source_paths = [Path(__file__), ROOT / "src/v2hoi/egohand_adapter.py"]
    report["source_sha256"] = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in source_paths}
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "observations.npz", **arrays)
    report["output_sha256"] = digest(args.output / "observations.npz")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for path in source_paths:
        (snapshot / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"episode": report["episode"], "frames": report["frames"], "variant": args.variant,
                      "hands": report["hands"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
