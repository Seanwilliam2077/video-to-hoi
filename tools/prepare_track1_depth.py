"""Save pinned MoGe2 depth for every original Track 1 frame.

Uses the same checkpoint, source pin and per-frame inference settings as
tools/prepare_body_camera.py. Depth is stored as float16; the valid mask is the
model's own valid region. This is monocular evidence, not metric scale.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

from prepare_body_camera import (
    DATA_REVISION,
    MODEL_BYTES,
    MODEL_REVISION,
    MODEL_SHA256,
    SOURCE_PIN,
    UTILS3D_PIN,
    episode_inputs,
    load_raw_inputs,
    portable_source_identity,
    sha256,
    verify_checkout,
)


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_model(args, moge_source, utils_source):
    import torch

    checkpoint = args.checkpoint.resolve(strict=True)
    if checkpoint.stat().st_size != MODEL_BYTES or sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint is not the pinned MoGe2 ViT-L model")
    if not 1200 <= args.num_tokens <= 3600:
        raise ValueError("num-tokens must be in the pinned model's supported 1200-3600 range")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path[:0] = [moge_source["path"], utils_source["path"]]
    from moge.model.v2 import MoGeModel
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    weights = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model = MoGeModel(**weights["model_config"])
    model.load_state_dict(weights["model"], strict=True)
    del weights
    model.enable_pytorch_native_sdpa()
    return model.eval().to(args.device)


def episode_done(directory: Path, frame_count: int) -> bool:
    report_path = directory / "depth-report.json"
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report.get("status") == "complete" and report.get("frames_written") == frame_count


def save_episode(model, row, video, timeline, args, provenance) -> dict:
    import av
    import numpy as np
    import torch

    index = row["episode_index"]
    count, height, width = row["frame_count"], row["height"], row["width"]
    directory = args.output / f"episode_{index:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    if episode_done(directory, count):
        print(f"EP{index:02d}: existing depth verified by report; skipped", flush=True)
        return {"episode_index": index, "status": "complete", "frames_written": count, "skipped": True}

    depth_path = directory / "depth.f16"
    valid_path = directory / "valid.u8"
    progress_path = directory / "progress.json"
    expected_depth = count * height * width * np.dtype(np.float16).itemsize
    expected_valid = count * height * width
    start_frame = 0
    if progress_path.is_file() and depth_path.is_file() and valid_path.is_file() and depth_path.stat().st_size == expected_depth and valid_path.stat().st_size == expected_valid:
        start_frame = int(json.loads(progress_path.read_text(encoding="utf-8"))["frames_written"])
    else:
        if depth_path.exists() or valid_path.exists() or progress_path.exists():
            raise ValueError(f"partial depth files have an unexpected size: {directory}")
        with depth_path.open("wb") as stream:
            stream.truncate(expected_depth)
        with valid_path.open("wb") as stream:
            stream.truncate(expected_valid)
    depth_map = np.memmap(depth_path, dtype=np.float16, mode="r+", shape=(count, height, width))
    valid_map = np.memmap(valid_path, dtype=np.uint8, mode="r+", shape=(count, height, width))
    flagged = []
    started = time.perf_counter()
    decoded = 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for frame_id, frame in enumerate(container.decode(stream)):
            decoded += 1
            if frame_id >= count:
                raise ValueError("decoded more frames than verified timeline")
            if (frame.width != width or frame.height != height or frame.pts != timeline["pts"][frame_id]
                    or Fraction(frame.time_base) != Fraction(timeline["time_base"]["numerator"], timeline["time_base"]["denominator"])):
                raise ValueError(f"decoded dimensions/PTS differ at source frame {frame_id}")
            if frame_id < start_frame:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            image = torch.from_numpy(rgb).permute(2, 0, 1).to(args.device, dtype=torch.float32) / 255.0
            with torch.inference_mode():
                prediction = model.infer(image, num_tokens=args.num_tokens, use_fp16=args.device == "cuda", fov_x=None)
            mask = prediction["mask"].detach().cpu().numpy().astype(bool)
            values = prediction["depth"].detach().float().cpu().numpy()
            if mask.shape != (height, width) or values.shape != mask.shape:
                raise ValueError(f"unexpected depth shape at frame {frame_id}")
            finite = np.isfinite(values) & (values > 0) & (values < np.float16(np.finfo(np.float16).max))
            valid = mask & finite
            if int((mask & ~finite).sum()) or not valid.any():
                flagged.append(frame_id)
            stored = np.zeros((height, width), dtype=np.float16)
            stored[valid] = values[valid].astype(np.float16)
            depth_map[frame_id] = stored
            valid_map[frame_id] = valid.astype(np.uint8)
            if (frame_id + 1) % 25 == 0 or frame_id + 1 == count:
                depth_map.flush()
                valid_map.flush()
                write_json(progress_path, {"frames_written": frame_id + 1, "flagged_frames": flagged})
                elapsed = time.perf_counter() - started
                print(f"EP{index:02d}: {frame_id + 1}/{count} depth frames, {elapsed:.1f}s", flush=True)
            del prediction, image, mask, values, rgb
    if decoded != count:
        raise ValueError("decoded frame count differs from the verified timeline")
    depth_map.flush()
    valid_map.flush()
    del depth_map, valid_map
    report = {
        "schema_version": "track1_moge2_depth.v1",
        "status": "complete",
        "episode_index": index,
        "frames_written": count,
        "height": height,
        "width": width,
        "storage": "float16 depth.f16 row-major [frame,height,width]; valid.u8 is 0 or 1",
        "depth_sha256": sha256(depth_path),
        "valid_sha256": sha256(valid_path),
        "depth_bytes": depth_path.stat().st_size,
        "valid_bytes": valid_path.stat().st_size,
        "flagged_frames": flagged,
        "video_sha256": timeline["video_sha256"],
        "timeline_sha256": row["timeline_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
        "provenance": provenance,
        "scale": "monocular MoGe2 depth, not a verified metric scene scale",
    }
    write_json(directory / "depth-report.json", report)
    print(f"EP{index:02d}: depth complete, {len(flagged)} flagged frames", flush=True)
    return {"episode_index": index, "status": "complete", "frames_written": count, "flagged_frames": len(flagged)}


def run(args) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = load_raw_inputs(args.manifest.resolve(), args.dataset_root.resolve())
    moge_source = verify_checkout(args.moge_source, SOURCE_PIN)
    utils_source = verify_checkout(args.utils3d_source, UTILS3D_PIN)
    import torch
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = load_model(args, moge_source, utils_source)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": DATA_REVISION,
        "moge_source": portable_source_identity(moge_source, "microsoft/MoGe"),
        "utils3d_source": portable_source_identity(utils_source, "EasternJournalist/utils3d"),
        "checkpoint": {"repository": "Ruicheng/moge-2-vitl-normal", "revision": MODEL_REVISION,
                       "sha256": MODEL_SHA256, "size_bytes": MODEL_BYTES},
        "adapter_sha256": sha256(Path(__file__)),
        "inference": {"num_tokens": args.num_tokens, "batch_size": 1, "fov_x": None,
                      "precision": "fp16 inference, float16 stored depth" if args.device == "cuda" else "fp32"},
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                        "device": args.device,
                        "packages": sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())},
    }
    if args.device == "cuda":
        provenance["environment"].update(cuda_device=torch.cuda.get_device_name(),
                                         cuda_capability=list(torch.cuda.get_device_capability()))
    summary = {"schema_version": "track1_moge2_depth_batch.v1", "status": "failed", "episodes": []}
    started = time.perf_counter()
    try:
        for index in args.episodes:
            row = next(item for item in manifest["episodes"] if item["episode_index"] == index)
            video, timeline = episode_inputs(manifest, row, args.manifest.resolve(), args.dataset_root.resolve())
            summary["episodes"].append(save_episode(model, row, video, timeline, args, provenance))
        summary["status"] = "complete" if all(item["status"] == "complete" for item in summary["episodes"]) else "requires_attention"
        if args.device == "cuda":
            summary["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "depth-batch.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--moge-source", type=Path, required=True)
    parser.add_argument("--utils3d-source", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(30)))
    parser.add_argument("--num-tokens", type=int, default=1600)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    try:
        summary = run(args)
    except Exception as error:
        parser.exit(2, f"Depth preparation failed: {error}\n")
    print(json.dumps({"status": summary["status"], "episodes": len(summary["episodes"])}))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
