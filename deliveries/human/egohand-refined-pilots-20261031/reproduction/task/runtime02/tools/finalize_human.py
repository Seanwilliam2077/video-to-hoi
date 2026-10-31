"""Fit shared identity/scales and export the Human portion of a Track 1 result.

This consumes real, provenance-recorded SAM Body outputs. It does not generate
object geometry or claim a complete challenge submission or reference score.
"""
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
from v2hoi.human_result import fit_shared_identity, native_mhr_forward, write_human_result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "weights/human/mhr_model.pt")
    parser.add_argument("--converter", type=Path, default=ROOT / "third_party/track1-mhr-converter/mesh_to_mhr_params.py")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--method", choices=("fit", "median"), default="fit",
                        help="median is an explicit initial baseline without pose refitting")
    parser.add_argument("--allow-partial", action="store_true", help="Only for labelled smoke outputs")
    parser.add_argument("--model-batch", type=int, default=1)
    parser.add_argument("--keyframes", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--pose-workers", type=int, default=1,
                        help="Independent CPU pose refits; more than one requires --threads 1")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU fitting defaults to 1: the workstation's threaded dense solve can stall")
    args = parser.parse_args()
    if args.threads is None:
        args.threads = 1 if args.device == "cpu" else 8
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.pose_workers < 1:
        parser.error("--pose-workers must be positive")
    if args.pose_workers > 1:
        if args.device != "cpu":
            parser.error("Parallel pose refits are supported only on CPU")
        if args.threads != 1:
            parser.error("Parallel pose refits require --threads 1")
        if args.method != "fit":
            parser.error("--pose-workers applies only to --method fit")
    if args.output.exists():
        parser.error("Output already exists; use a fresh directory to preserve evidence")
    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("track") != 1:
        parser.error("Only a Track 1 inference run is allowed")
    raw = args.run / "raw"
    arrays = {name: np.load(raw / f"{name}.npy", mmap_mode="r", allow_pickle=False)
              for name in ("frame_indices", "pose", "scales", "shape", "pred_vertices")}
    indices = arrays["frame_indices"]
    complete = np.array_equal(indices, np.arange(manifest["video_frames"]))
    if not complete and not args.allow_partial:
        parser.error("Inference does not cover the complete video; finish inference or explicitly label a partial smoke")
    import torch
    torch.set_num_threads(args.threads)
    if args.pose_workers > 1:
        # Set once before model evaluation, never from an individual worker.
        torch.set_num_interop_threads(1)
    started = time.monotonic()
    if args.method == "fit":
        expected = "c799ad612fca19620563fcb93bf61e5a4adad0a04251482358746b5f27f8a52e"
        if digest(args.converter) != expected:
            parser.error("Official converter is not the reviewed version")
        result = fit_shared_identity(
            arrays["pred_vertices"], arrays["pose"], arrays["scales"], arrays["shape"],
            args.model, args.converter, device=args.device, model_batch=args.model_batch,
            keyframes=args.keyframes, pose_iterations=args.iterations, identity_iterations=args.iterations,
            pose_workers=args.pose_workers,
        )
    else:
        result = {"pose": np.asarray(arrays["pose"]).copy(),
                  "scales": np.median(arrays["scales"], axis=0),
                  "shape": np.median(arrays["shape"], axis=0),
                  "report": {"method": "fixed_median_initial_baseline_no_pose_refit"}}
    result.update(frame_indices=np.array(indices), fps=np.asarray(manifest["video_fps"]))
    forward = native_mhr_forward(result, args.model, device=args.device, batch_size=args.model_batch)
    errors = np.linalg.norm(forward["vertices_m"] - arrays["pred_vertices"], axis=-1).mean(axis=1) * 1000
    result["per_frame_vertex_error_mm"] = errors
    report = result["report"]
    report.update({
        "track": 1, "episode": manifest["episode"], "source_manifest_sha256": digest(manifest_path),
        "video_sha256": manifest["video_sha256"], "mhr_model_sha256": digest(args.model),
        "complete_video": bool(complete), "expected_frames": manifest["video_frames"],
        "frame_indices": indices.tolist(), "camera_intrinsics": manifest["camera_intrinsics"],
        "intrinsics_status": manifest["intrinsics_status"], "bbox_method": manifest["bbox_method"],
        "saved_forward_error_mm": {"mean": float(errors.mean()), "max_frame_mean": float(errors.max())},
        "error_interpretation": "Difference from the raw model prediction, not hidden-reference accuracy",
        "total_seconds": round(time.monotonic() - started, 3),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    })
    args.output.mkdir(parents=True)
    parameter_path = args.output / f"episode_{manifest['episode']:06d}_human.npz"
    summary = write_human_result(parameter_path, result, report)
    review = args.output / "review_arrays"
    review.mkdir()
    np.save(review / "vertices_m.npy", forward["vertices_m"])
    np.save(review / "joints_3d.npy", forward["joints_m"])
    for name in ("frame_indices", "faces", "camera_intrinsics", "fps"):
        np.save(review / f"{name}.npy", np.load(raw / f"{name}.npy", allow_pickle=False))
    # SAM70 skeleton edges do not index the native MHR127 joints. Do not copy them.
    (args.output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"human_parameters": str(parameter_path), "review_arrays": str(review),
                      "complete_video": complete, "frames": len(indices),
                      "mean_change_from_raw_mm": float(errors.mean())}, indent=2))


if __name__ == "__main__":
    main()
