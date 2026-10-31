"""Run real SAM 3D Body on one verified Track 1 video, with resumable frames.

This produces raw per-frame native MHR predictions and review arrays. Run the
separate shared-identity fitter before treating the Human parameter file as a
submission-shaped result. Object reconstruction is not provided by this tool.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from v2hoi.human_result import sam_prediction_to_native
from v2hoi.sam_body_inputs import (decoded_frame_sha256, fingerprint, load_body_inputs,
                                 validate_resume_frame)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def process_memory() -> dict:
    """Read this process's peak resident/commit use without an extra package."""
    if os.name != "nt":
        import resource
        return {"peak_working_set_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)}
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "quota_peak_paged_pool", "quota_paged_pool",
                "quota_peak_nonpaged_pool", "quota_nonpaged_pool", "pagefile", "peak_pagefile", "private_usage",
            )
        ]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return {}
    return {"peak_working_set_bytes": counters.peak_working_set, "working_set_bytes": counters.working_set,
            "peak_committed_bytes": counters.peak_pagefile, "private_committed_bytes": counters.private_usage}


def save_frame(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(destination, **value)
    temporary.replace(path)


def call_with_cuda_cache_policy(cuda, *, device: str, policy: str, operation):
    """Scope allocator-cache retention to one standalone inference call.

    The upstream policy is unchanged by default. Retention suppresses only
    empty_cache calls, never model operations, and restores the original API
    even when inference fails. This worker runs inference calls sequentially.
    """
    if policy not in ("upstream", "retain"):
        raise ValueError(f"Unknown CUDA cache policy: {policy}")
    if device != "cuda" or policy == "upstream":
        return operation()
    original = cuda.empty_cache
    try:
        cuda.empty_cache = lambda: None
        return operation()
    finally:
        cuda.empty_cache = original


def verify_runtime_source(source: Path) -> None:
    manifest = json.loads((source / "installed_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source manifest path: {relative}")
        local_path = source / relative
        if not local_path.is_file() or digest(local_path) != record["installed_sha256"]:
            raise ValueError(f"Runtime source no longer matches installed manifest: {local_path}")


def load_model(checkpoint: Path, mhr: Path, source: Path, device: str):
    """Use the upstream architecture/loader with explicit CPU precision policy."""
    sys.path.insert(0, str(source))
    os.environ["MOMENTUM_ENABLED"] = "0"  # Use the checkpoint's TorchScript MHR.
    import torch
    from sam_3d_body import SAM3DBodyEstimator
    from sam_3d_body.models.meta_arch import SAM3DBody
    from sam_3d_body.utils.checkpoint import load_state_dict
    from sam_3d_body.utils.config import get_config

    config_path = checkpoint.parent / "model_config.yaml"
    config = get_config(str(config_path))
    config.defrost()
    config.MODEL.MHR_HEAD.MHR_MODEL_PATH = str(mhr)
    if device == "cpu":
        config.TRAIN.USE_FP16 = False
    config.freeze()
    model = SAM3DBody(config)
    try:
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as error:
        if "mmap" not in str(error):
            raise
        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = checkpoint_data.get("state_dict", checkpoint_data)
    # Upstream allows missing keys so the separate TorchScript asset can supply
    # geometry. Fail if that permissiveness would leave learned SAM weights
    # uninitialized. DINO's patch mask_token is unused by the external
    # segmentation prompt: the backbone is called without patch masks. The
    # separate learned prompt_encoder.mask_downscaling weights must be present.
    expected = model.state_dict()
    allowed_missing = {"backbone.encoder.mask_token", "head_pose.hand_pose_comps_ori", "head_pose_hand.hand_pose_comps_ori"}
    missing = set(expected) - set(weights)
    unexplained = sorted(name for name in missing if name not in allowed_missing and not name.startswith(("head_pose.mhr.", "head_pose_hand.mhr.")))
    unexpected = sorted(set(weights) - set(expected))
    bad_shapes = sorted(name for name in set(weights) & set(expected) if weights[name].shape != expected[name].shape)
    if unexplained or unexpected or bad_shapes:
        raise ValueError(f"Checkpoint architecture mismatch: missing={unexplained}, unexpected={unexpected}, wrong_shapes={bad_shapes}")
    checkpoint_audit = {"loaded_tensor_count": len(weights), "separately_loaded_geometry_tensors": sum(name.startswith(("head_pose.mhr.", "head_pose_hand.mhr.")) for name in missing),
                        "unused_initialized_tensors": sorted(missing & allowed_missing), "learned_weight_mismatches": 0,
                        "learned_mask_prompt_tensors": sum(name.startswith("prompt_encoder.mask_downscaling.") for name in weights)}
    del expected
    load_state_dict(model, weights, strict=False)
    del weights, checkpoint_data
    gc.collect()
    model = model.to(device).eval()
    model.runtime_checkpoint_audit = checkpoint_audit
    return model, SAM3DBodyEstimator(model, config), config


def skeleton_edges(source: Path) -> np.ndarray:
    import runpy
    metadata = runpy.run_path(str(source / "sam_3d_body" / "metadata" / "mhr70.py"))["pose_info"]
    names = {record["name"]: int(idx) for idx, record in metadata["keypoint_info"].items()}
    return np.asarray([[names[a], names[b]] for record in metadata["skeleton_info"].values() for a, b in [record["link"]]], dtype=np.int64)


def combine_frames(output: Path, faces: np.ndarray, edges: np.ndarray, intrinsics: np.ndarray, fps: float,
                   configuration_sha256: str | None = None, total: int | None = None) -> dict:
    """Write standard .npy arrays incrementally to avoid stacking meshes in RAM."""
    files = sorted((output / "frames").glob("*.npz"))
    if not files:
        raise ValueError("No completed inference frames")
    combined = output / "raw"
    combined.mkdir(exist_ok=True)
    with np.load(files[0], allow_pickle=False) as first:
        fields = {name: (first[name].shape, first[name].dtype) for name in (
            "pose", "scales", "shape", "mhr_model_params", "pred_vertices",
            "pred_keypoints_2d", "pred_keypoints_3d", "pred_cam_t", "frame_index",
            "inference_seconds",
        )}
    maps = {}
    for name, (shape, dtype) in fields.items():
        target_name = "frame_indices" if name == "frame_index" else name
        maps[name] = np.lib.format.open_memmap(combined / f"{target_name}.npy", mode="w+", dtype=dtype, shape=(len(files), *shape))
    for row, path in enumerate(files):
        with np.load(path, allow_pickle=False) as item:
            if path.stem != f"{int(item['frame_index']):06d}" or (total is not None and not 0 <= int(item["frame_index"]) < total):
                raise ValueError(f"Saved frame filename/index is invalid: {path}")
            if configuration_sha256 is not None:
                if str(item["input_configuration_sha256"].item()) != configuration_sha256:
                    raise ValueError(f"Saved frame configuration differs: {path}")
            for name, array in maps.items():
                if not np.isfinite(item[name]).all() or item[name].shape != fields[name][0]:
                    raise ValueError(f"Invalid saved field {name}: {path}")
                array[row] = item[name]
    for array in maps.values():
        array.flush()
    np.save(combined / "faces.npy", faces)
    np.save(combined / "skeleton_edges.npy", edges)
    np.save(combined / "camera_intrinsics.npy", intrinsics)
    np.save(combined / "fps.npy", np.asarray(fps))
    return {"completed_frames": len(files), "raw_arrays": str(combined), "mesh_coordinates": "OpenCV camera metres, including pred_cam_t exactly once"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-url", required=True, help="Verified official URL for the selected Track 1 video")
    parser.add_argument("--episode", type=int, required=True, choices=range(30))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mhr-path", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "third_party" / "sam3d-body-runtime")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-cache-policy", choices=("upstream", "retain"), default="upstream",
                        help="Retain skips upstream empty_cache calls only during each inference call; model math is unchanged")
    profiles = parser.add_mutually_exclusive_group(required=True)
    profiles.add_argument("--bbox", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                          help="Explicit legacy unmasked fixed-box profile; requires an assumed --hfov")
    profiles.add_argument("--body-input-dir", type=Path, help="Verified per-episode masks and six-field camera package")
    parser.add_argument("--receiver-checks", type=Path, help="Receiver's successful all-30 body_handoff_checks.v1 report")
    parser.add_argument("--hfov", type=float, help="Legacy assumed horizontal FOV in degrees; not measured calibration")
    parser.add_argument("--inference-type", choices=("full", "body"), default="full")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 processes all frames after --start")
    parser.add_argument("--frames", type=int, nargs="+", help="Sorted unique source frame IDs; mutually exclusive with --start/--limit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    parsed = urlparse(args.source_url)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or "/datasets/nvidia/video_to_data_challenge/" not in parsed.path or "/track_1/" not in parsed.path:
        parser.error("--source-url must identify an official NVIDIA Track 1 video on Hugging Face")
    expected_video_suffix = f"/track_1/videos/chunk-000/observation.images.exo_camera/episode_{args.episode:06d}.mp4"
    if not parsed.path.endswith(expected_video_suffix):
        parser.error("--source-url must identify this episode's original Track 1 exo-camera video")
    if args.body_input_dir is not None and (args.receiver_checks is None or args.hfov is not None):
        parser.error("--body-input-dir requires --receiver-checks and forbids assumed --hfov")
    if args.body_input_dir is None and (args.receiver_checks is not None or args.hfov is None):
        parser.error("Legacy --bbox requires explicit --hfov and cannot use --receiver-checks")
    if (args.hfov is not None and not 0 < args.hfov < 180) or args.start < 0 or args.limit < 0 or args.threads < 1:
        parser.error("Invalid FOV, start, limit, or threads")
    if args.frames is not None and (args.start or args.limit or args.frames != sorted(set(args.frames)) or min(args.frames) < 0):
        parser.error("--frames requires sorted unique nonnegative IDs and cannot be combined with --start/--limit")
    video, checkpoint, mhr, source, output = [path.resolve() for path in (args.video, args.checkpoint, args.mhr_path, args.source, args.output)]
    for path in (video, checkpoint, mhr, checkpoint.parent / "model_config.yaml", source / "installed_manifest.json"):
        if not path.is_file():
            parser.error(f"Required file missing: {path}")
    verify_runtime_source(source)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        parser.error(f"Cannot open video {video}")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total = float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1 or args.start >= total or fps <= 0:
        parser.error("Video metadata or start frame is invalid")
    if args.frames is not None and max(args.frames) >= total:
        parser.error("Requested source frame exceeds video length")
    video_sha = digest(video)
    body_inputs = None
    if args.body_input_dir is not None:
        body_inputs = load_body_inputs(args.body_input_dir, args.receiver_checks, video=video,
                                      episode=args.episode, width=width, height=height,
                                      frame_count=total, fps=fps, video_sha256=video_sha)
        intrinsics = body_inputs.intrinsics
        input_provenance = body_inputs.provenance
    else:
        x1, y1, x2, y2 = args.bbox
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            parser.error("Bounding box must lie within the video frame")
        focal = width / (2 * math.tan(math.radians(args.hfov) / 2))
        intrinsics = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32)
        input_provenance = {"profile": "legacy_unmasked_fixed_bbox_assumed_hfov.v1", "bbox_xyxy": args.bbox,
                            "bbox_method": "manual fixed whole-clip bounding box", "hfov_degrees": args.hfov,
                            "intrinsics_status": "assumed shared pinhole camera, not measured calibration"}
    environment = {"python": sys.version, "executable": str(Path(sys.executable).resolve()), "opencv": cv2.__version__}
    for name in ("torch", "torchvision", "numpy", "Pillow", "timm", "roma"):
        environment[name] = importlib.metadata.version(name)
    provenance = {
        "schema_version": 2, "track": 1, "episode": args.episode,
        "video": str(video), "video_url": args.source_url, "video_sha256": video_sha,
        "video_width": width, "video_height": height, "video_fps": fps, "video_frames": total,
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "mhr_model": str(mhr), "mhr_model_sha256": digest(mhr),
        "config_sha256": digest(checkpoint.parent / "model_config.yaml"),
        "runtime_source_manifest_sha256": digest(source / "installed_manifest.json"),
        **input_provenance,
        "camera_intrinsics": intrinsics.tolist(), "inference_type": args.inference_type,
        "device": args.device, "cpu_float32": args.device == "cpu", "threads": args.threads,
        "cuda_cache_policy": args.cuda_cache_policy,
        "runtime_environment": environment,
        "consumer_code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in
                                 (Path(__file__).resolve(), ROOT / "src/v2hoi/sam_body_inputs.py", ROOT / "src/v2hoi/human_result.py")},
        "scope": "raw Human predictions; per-frame identity and scale still require shared refitting",
    }
    configuration_sha256 = fingerprint(provenance)
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        if not args.resume:
            parser.error("Output already has a manifest; use --resume with the same configuration or a new output directory")
        if json.loads(manifest.read_text()) != provenance:
            parser.error("Resume configuration or file hashes differ from the saved manifest")
    elif list((output / "frames").glob("*.npz")):
        parser.error("Existing frames have no provenance manifest")
    else:
        write_json(manifest, provenance)
    import torch
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("PyTorch CUDA is unavailable")
    start = min(args.frames) if args.frames is not None else args.start
    stop = max(args.frames) + 1 if args.frames is not None else min(total, args.start + args.limit) if args.limit else total
    requested_indices = args.frames if args.frames is not None else list(range(start, stop))
    selected = set(requested_indices)
    state = {"status": "loading_model", "requested_start": start, "requested_stop_exclusive": stop,
             "requested_frame_indices": requested_indices, "last_completed_frame": None}
    write_json(output / "state.json", state)
    print(f"Loading real SAM 3D Body on {args.device}; model/checkpoint hashes recorded.", flush=True)
    load_started = time.monotonic()
    try:
        model, estimator, _ = load_model(checkpoint, mhr, source, args.device)
    except BaseException as error:
        state.update(status="model_load_failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "state.json", state)
        capture.release()
        raise
    faces = estimator.faces.astype(np.int64)
    edges = skeleton_edges(source)
    state["model_load_seconds"] = time.monotonic() - load_started
    state["checkpoint_audit"] = model.runtime_checkpoint_audit
    state.update(process_memory())
    print(f"Model ready in {state['model_load_seconds']:.1f}s; video has {total} frames.", flush=True)
    # Sequential decoding from source frame zero preserves original IDs even
    # when a codec's random seek lands on a nearby keyframe.
    state["status"] = "running"
    write_json(output / "state.json", state)
    try:
        for frame_index in range(stop):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Video decode failed at frame {frame_index}")
            if frame_index not in selected:
                continue
            if bgr.dtype != np.uint8 or bgr.shape != (height, width, 3):
                raise ValueError(f"Decoded frame grid changed at frame {frame_index}")
            pixel_sha256 = decoded_frame_sha256(bgr)
            if body_inputs is not None:
                masks, bboxes, mask_sha256 = body_inputs.frame(frame_index)
            else:
                masks, bboxes, mask_sha256 = None, np.asarray(args.bbox, dtype=np.float32).reshape(1, 4), ""
            frame_path = output / "frames" / f"{frame_index:06d}.npz"
            if frame_path.exists() and args.resume:
                with np.load(frame_path, allow_pickle=False) as existing:
                    validate_resume_frame(existing, frame_index, configuration_sha256, pixel_sha256, mask_sha256)
                continue
            started = time.monotonic()
            with torch.inference_mode():
                predictions = call_with_cuda_cache_policy(
                    torch.cuda, device=args.device, policy=args.cuda_cache_policy,
                    operation=lambda: estimator.process_one_image(
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                        bboxes=bboxes, masks=masks, use_mask=body_inputs is not None,
                        cam_int=torch.from_numpy(intrinsics).unsqueeze(0).to(args.device),
                        inference_type=args.inference_type,
                    ),
                )
            if len(predictions) != 1:
                raise RuntimeError(f"Expected exactly one person at frame {frame_index}")
            item = {name: value.detach().cpu().numpy() if torch.is_tensor(value) else value for name, value in predictions[0].items()}
            native = sam_prediction_to_native(item)
            elapsed = time.monotonic() - started
            placed_keypoints = np.asarray(item["pred_keypoints_3d"] + item["pred_cam_t"], dtype=np.float32)
            record = {
                "frame_index": np.asarray(frame_index, dtype=np.int64),
                "inference_seconds": np.asarray(elapsed),
                "pose": native["pose"], "scales": native["scales"], "shape": native["shape"],
                "mhr_model_params": np.concatenate([native["pose"], native["scales"]]),
                "sam_root_local_model_params": np.asarray(item["mhr_model_params"], dtype=np.float32),
                "pred_vertices": native["vertices_m"],
                "pred_keypoints_2d": np.asarray(item["pred_keypoints_2d"], dtype=np.float32),
                "pred_keypoints_3d": placed_keypoints,
                "pred_cam_t": np.asarray(item["pred_cam_t"], dtype=np.float32),
                "input_bbox_xyxy": bboxes[0],
                "input_configuration_sha256": np.asarray(configuration_sha256),
                "decoded_bgr_sha256": np.asarray(pixel_sha256),
                "input_mask_sha256": np.asarray(mask_sha256),
            }
            if not all(np.isfinite(array).all() for array in record.values() if array.dtype.kind in "fiu"):
                raise ValueError(f"Non-finite prediction at frame {frame_index}")
            save_frame(frame_path, record)
            state.update(last_completed_frame=frame_index, last_inference_seconds=elapsed)
            state.update(process_memory())
            write_json(output / "state.json", state)
            print(f"Frame {frame_index}/{total - 1}: {elapsed:.2f}s, saved native MHR + camera-space mesh.", flush=True)
        if stop == total and capture.read()[0]:
            raise ValueError("Video decoded more frames than verified metadata")
        state.update(combine_frames(output, faces, edges, intrinsics, fps, configuration_sha256, total))
        state["status"] = "requested_range_complete"
        state["all_video_frames_complete"] = state["completed_frames"] == total and all((output / "frames" / f"{i:06d}.npz").exists() for i in range(total))
        if args.device == "cuda":
            state["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            state["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(output / "state.json", state)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "state.json", state)
        raise
    finally:
        capture.release()


if __name__ == "__main__":
    main()
