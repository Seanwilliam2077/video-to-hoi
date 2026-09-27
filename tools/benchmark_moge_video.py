"""Run pinned MoGe v2/v3 on every video frame; record raw outputs, not a quality score.

Requires a local checkpoint and a separately installed MoGe source pinned to
74fbce054ebed49800de42d0ad0e83495065719a. This adapter is unverified on GPU.
"""

from __future__ import annotations

import argparse, hashlib, json, math
import os, platform, sys, time
from pathlib import Path
SOURCE_PIN = "74fbce054ebed49800de42d0ad0e83495065719a"
CHUNK = 1024 * 1024

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()

def run(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "schema_version": 1, "status": "failed", "version": args.version,
        "moge_source_pin_expected": SOURCE_PIN, "video": str(args.video),
        "checkpoint": str(args.checkpoint), "expected_frames": args.expected_frames,
        "decoded_frames": 0, "frames": [],
    }
    torch = None
    capture = None
    try:
        if args.expected_frames <= 0:
            raise ValueError("--expected-frames must be positive")
        video, checkpoint = args.video.resolve(), args.checkpoint.resolve()
        if not video.is_file() or not checkpoint.is_file():
            raise FileNotFoundError("video and local checkpoint must both be files")
        report.update(video=str(video), checkpoint=str(checkpoint),
                      video_sha256=sha256(video), checkpoint_sha256=sha256(checkpoint))
        # MoGe falls back to Hugging Face if a checkpoint path disappears.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import cv2
        import numpy as np
        import torch as torch_module
        torch = torch_module
        if args.version == "v2":
            from moge.model.v2 import MoGeModel
        else:
            from moge.model.v3 import MoGeModel
        report["environment"] = {"python": sys.version.split()[0], "platform": platform.platform(),
                                 "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                                 "opencv": cv2.__version__, "numpy": np.__version__,
                                 "moge_module": sys.modules[MoGeModel.__module__].__file__}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU fallback is disabled")
        report["environment"].update(cuda_device=torch.cuda.get_device_name(0),
                                      cuda_capability=list(torch.cuda.get_device_capability(0)))
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        model = MoGeModel.from_pretrained(str(checkpoint)).to("cuda").eval()
        torch.cuda.synchronize()
        report["model_load_seconds"] = time.perf_counter() - load_started
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError("cannot open video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"invalid video FPS: {fps}")
        report["fps"] = fps
        frame_dir = output / "frames"
        frame_dir.mkdir()
        totals = {key: 0 for key in ("pixels", "valid_pixels", "nonfinite_pixels",
                                    "nonfinite_valid_pixels", "nonpositive_valid_pixels", "nonfinite_intrinsics")}
        inference_seconds = 0.0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            index = report["decoded_frames"]
            height, width = bgr.shape[:2]
            if not height or not width:
                raise ValueError(f"empty decoded frame {index}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(rgb).to(device="cuda", dtype=torch.float32).permute(2, 0, 1) / 255.0
            options = {"resolution_level": 9}
            if args.version == "v3":
                options["refine_steps"] = 3
            infer_started = time.perf_counter()
            with torch.inference_mode():
                prediction = model.infer(image, **options)
            depth = prediction["depth"].detach().float().cpu().numpy()
            mask = prediction["mask"].detach().cpu().numpy().astype(bool)
            intrinsics = prediction["intrinsics"].detach().float().cpu().numpy()
            inference_seconds += time.perf_counter() - infer_started
            if depth.shape != (height, width) or mask.shape != depth.shape or intrinsics.shape != (3, 3):
                raise ValueError(f"unexpected output shape at frame {index}")
            finite = np.isfinite(depth)
            metrics = {"index": index, "valid_pixels": int(mask.sum()), "nonfinite_pixels": int((~finite).sum()),
                       "nonfinite_valid_pixels": int((mask & ~finite).sum()),
                       "nonpositive_valid_pixels": int((mask & finite & (depth <= 0)).sum()),
                       "nonfinite_intrinsics": int((~np.isfinite(intrinsics)).sum())}
            totals["pixels"] += height * width
            for key in totals.keys() - {"pixels"}:
                totals[key] += metrics[key]
            target = frame_dir / f"{index:06d}.npz"
            temporary = frame_dir / f".{index:06d}.npz.tmp"
            try:
                with temporary.open("xb") as stream:
                    np.savez_compressed(stream, depth=depth.astype(np.float32), mask=mask,
                                        intrinsics=intrinsics.astype(np.float32))
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            report["frames"].append(metrics)
            report["decoded_frames"] += 1
            if report["decoded_frames"] > args.expected_frames:
                raise ValueError("decoded more frames than --expected-frames")
        if report["decoded_frames"] != args.expected_frames:
            raise ValueError("decoded frame count differs from --expected-frames")
        report.update(status="complete", raw_metrics={**totals, "valid_fraction": totals["valid_pixels"] / totals["pixels"]},
                      inference_seconds=inference_seconds)
        return report
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if capture is not None:
            capture.release()
        report["elapsed_seconds_including_model_load"] = time.perf_counter() - started
        if torch is not None and torch.cuda.is_available():
            report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        with (output / "execution.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("video", "checkpoint", "output"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--version", required=True, choices=("v2", "v3"))
    parser.add_argument("--expected-frames", required=True, type=int)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        parser.exit(2, f"MoGe benchmark failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
