"""Estimate each Track 1 camera from pinned MoGe2 predictions on its own pixels.

Requires local, unmodified pinned MoGe/utils3d source snapshots and a verified
local checkpoint. No downloads, guessed FoV, cross-episode pooling, or silent
resolution fallback. Batch size is one. Outputs are estimates, not calibration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

SOURCE_PIN = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
UTILS3D_PIN = "3fab839f0be9931dac7c8488eb0e1600c236e183"
MODEL_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"
MODEL_SHA256 = "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01"
MODEL_BYTES = 1323815904
DATA_REVISION = "5f68335f3acc802033d1e80728c1633197521de8"
MAX_RELATIVE_P10_P90_SPREAD = 0.25
MAX_RELATIVE_DEVIATION = 0.40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, document: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sampled_frame_ids(frame_count: int, samples: int) -> list[int]:
    if type(frame_count) is not int or frame_count < samples or not 9 <= samples <= 15:
        raise ValueError("require 9-15 samples and at least that many original frames")
    # Integer arithmetic avoids platform-dependent rounding at halfway values.
    return [(i * (frame_count - 1) + (samples - 1) // 2) // (samples - 1)
            for i in range(samples)]


def original_pixel_intrinsics(matrix: list, width: int, height: int) -> dict:
    """Convert normalized UV to original integer pixel centers (utils3d default)."""
    if (len(matrix) != 3 or any(len(row) != 3 for row in matrix)
            or any(not math.isfinite(float(v)) for row in matrix for v in row)):
        raise ValueError("intrinsics must be a finite 3 x 3 matrix")
    if (width <= 0 or height <= 0 or matrix[0][0] <= 0 or matrix[1][1] <= 0
            or abs(matrix[0][1]) > 1e-6 or abs(matrix[1][0]) > 1e-6
            or any(abs(a - b) > 1e-6 for a, b in zip(matrix[2], [0, 0, 1]))):
        raise ValueError("invalid pinhole intrinsics or original image dimensions")
    result = {"fx": float(matrix[0][0] * width), "fy": float(matrix[1][1] * height),
              "cx": float(matrix[0][2] * width - 0.5),
              "cy": float(matrix[1][2] * height - 0.5),
              "width": width, "height": height}
    if not (0 <= result["cx"] < width and 0 <= result["cy"] < height):
        raise ValueError("principal point outside original image")
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    lower = math.floor(location)
    upper = math.ceil(location)
    return ordered[lower] * (upper - location) + ordered[upper] * (location - lower) if lower != upper else ordered[lower]


def aggregate_estimates(estimates: list[dict]) -> tuple[dict, dict, list[str]]:
    """Median of every sample; report disagreement rather than removing outliers."""
    if len(estimates) < 9:
        raise ValueError("at least nine independent frame estimates are required")
    widths = {row["width"] for row in estimates}
    heights = {row["height"] for row in estimates}
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError("cannot aggregate different pixel grids")
    result = {key: float(statistics.median(row[key] for row in estimates))
              for key in ("fx", "fy", "cx", "cy")}
    result.update(width=estimates[0]["width"], height=estimates[0]["height"])
    diagnostics, blockers = {}, []
    for key in ("fx", "fy"):
        values = [float(row[key]) for row in estimates]
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("all focal estimates must be finite and positive")
        middle = result[key]
        p10, p90 = percentile(values, 0.1), percentile(values, 0.9)
        spread = (p90 - p10) / middle
        deviation = max(abs(v - middle) for v in values) / middle
        diagnostics[key] = {"median": middle, "minimum": min(values), "maximum": max(values),
                            "p10": p10, "p90": p90,
                            "median_absolute_deviation": statistics.median(abs(v - middle) for v in values),
                            "relative_p10_p90_spread": spread,
                            "maximum_relative_deviation": deviation}
        if spread > MAX_RELATIVE_P10_P90_SPREAD:
            blockers.append(f"{key}: p10-p90 spread / median exceeds {MAX_RELATIVE_P10_P90_SPREAD}")
        if deviation > MAX_RELATIVE_DEVIATION:
            blockers.append(f"{key}: maximum deviation / median exceeds {MAX_RELATIVE_DEVIATION}")
    return result, diagnostics, blockers


def verify_checkout(path: Path, revision: str) -> dict:
    path = path.resolve(strict=True)
    receipt_path = path / "source-receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_repository = {SOURCE_PIN: "microsoft/MoGe", UTILS3D_PIN: "EasternJournalist/utils3d"}[revision]
        if receipt.get("revision") != revision or receipt.get("repository") != expected_repository:
            raise ValueError("source receipt repository/revision does not match pinned runtime")
        listed = set()
        for entry in receipt.get("files", []):
            relative = entry.get("path")
            if (not isinstance(relative, str) or "\\" in relative or ":" in relative
                    or relative.startswith("/") or any(part in {"", ".", ".."} for part in relative.split("/"))
                    or relative in listed):
                raise ValueError("source receipt contains an unsafe/duplicate path")
            target = path / relative
            if (not target.resolve(strict=True).is_relative_to(path) or target.stat().st_size != entry["bytes"]
                    or sha256(target) != entry["sha256"]):
                raise ValueError(f"source receipt file verification failed: {relative}")
            listed.add(relative)
        python_files = {item.relative_to(path).as_posix() for item in path.rglob("*.py")}
        if not python_files or not python_files <= listed:
            raise ValueError("source snapshot contains unlisted Python runtime files")
        return {"revision": revision, "path": str(path), "verification": "source_receipt_file_sha256",
                "source_receipt_sha256": sha256(receipt_path), "verified_file_count": len(listed)}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    actual = git("rev-parse", "HEAD")
    if actual != revision or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"source must be an unmodified checkout at {revision}: {path}")
    tracked = set(git("ls-files").splitlines())
    if any(item.relative_to(path).as_posix() not in tracked for item in path.rglob("*.py")):
        raise ValueError("source checkout contains untracked Python runtime files")
    return {"revision": actual, "path": str(path), "verification": "clean_pinned_git_checkout"}


def portable_source_identity(source: dict, repository: str) -> dict:
    """Keep producer filesystem locations out of transferable result metadata."""
    return {"repository": repository, **{key: value for key, value in source.items() if key != "path"}}


def load_raw_inputs(manifest_path: Path, dataset_root: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "track1_raw_preparation.v1"
            or manifest.get("repository") != "nvidia/video_to_data_challenge"
            or manifest.get("revision") != DATA_REVISION or manifest.get("track") != "track_1"
            or manifest.get("raw_data_ready") is not True
            or sorted(row["episode_index"] for row in manifest.get("episodes", [])) != list(range(30))):
        raise ValueError("requires the verified complete 30-episode Track 1 manifest")
    for path in (manifest_path, dataset_root):
        if any(part.lower() in {"track_2", "track2"} for part in path.parts):
            raise ValueError("prohibited data path")
    return manifest


def episode_inputs(manifest: dict, row: dict, manifest_path: Path, dataset_root: Path) -> tuple[Path, dict]:
    index = row["episode_index"]
    expected_path = f"videos/chunk-000/observation.images.exo_camera/episode_{index:06d}.mp4"
    expected_timeline = f"timelines/episode_{index:06d}.json"
    if row["video_path"] != expected_path or row["timeline_path"] != expected_timeline:
        raise ValueError("unexpected source path")
    video = dataset_root / expected_path
    timeline_path = manifest_path.parent / expected_timeline
    if sha256(video) != manifest["source_files"][expected_path]["sha256"]:
        raise ValueError(f"source video hash mismatch: episode {index}")
    if sha256(timeline_path) != row["timeline_sha256"]:
        raise ValueError(f"timeline hash mismatch: episode {index}")
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if (timeline["episode_index"] != index or timeline["video_sha256"] != sha256(video)
            or timeline["frame_count"] != row["frame_count"]
            or timeline["frame_ids"] != list(range(row["frame_count"]))
            or timeline["width"] != row["width"] or timeline["height"] != row["height"]):
        raise ValueError(f"timeline/source identity mismatch: episode {index}")
    return video, timeline


def estimate_episode(model, row: dict, video: Path, timeline: dict,
                     args: argparse.Namespace, provenance: dict) -> dict:
    import av
    import numpy as np
    import torch

    index = row["episode_index"]
    directory = args.output / f"episode_{index:06d}"
    directory.mkdir(exist_ok=True)
    if any((directory / name).exists() for name in ("camera-estimation.json", "intrinsics.json", "candidate-intrinsics.json")):
        raise FileExistsError(f"camera result already exists: {directory}")
    ids = sampled_frame_ids(row["frame_count"], args.samples)
    selected = set(ids)
    report = {"schema_version": "body_camera_estimation.v1", "status": "failed",
              "episode_index": index, "video_sha256": timeline["video_sha256"],
              "timeline_sha256": row["timeline_sha256"], "source_frame_ids": ids,
              "method": "MoGe2 ViT-L per-frame point-map focal recovery; episode-local median",
              "code_revision": SOURCE_PIN, "checkpoint_sha256": MODEL_SHA256,
              "aggregation": "componentwise median of all sampled original-pixel estimates; no outlier removal",
              "coordinate_transform": "K_original = [[width,0,-0.5],[0,height,-0.5],[0,0,1]] @ K_normalized; integer pixel centers",
              "transforms": {"original_width": row["width"], "original_height": row["height"],
                             "external_crop": None, "external_resize": None,
                             "rgb_range": "uint8 / 255 to float32 [0,1]",
                             "uv_to_original_pixel_matrix": [[row["width"], 0, -0.5], [0, row["height"], -0.5], [0, 0, 1]],
                             "backbone_resizing": "MoGe DINOv2 whole-image interpolation to rounded token grid * 14; unchanged field of view"},
              "assumptions": ["static intrinsics within this clip", "pinhole camera without distortion estimation",
                              "MoGe fixes normalized principal point at (0.5,0.5); it is not independently estimated"],
              "quality_policy": {"maximum_relative_p10_p90_spread": MAX_RELATIVE_P10_P90_SPREAD,
                                 "maximum_relative_deviation": MAX_RELATIVE_DEVIATION,
                                 "thresholds_are": "engineering consistency screens, not calibrated confidence intervals"},
              "provenance": provenance, "raw_estimates": [], "blockers": []}
    started = time.perf_counter()
    try:
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            count = 0
            for frame_id, frame in enumerate(container.decode(stream)):
                count += 1
                if frame_id >= row["frame_count"]:
                    raise ValueError("decoded more frames than verified timeline")
                if (frame.width != row["width"] or frame.height != row["height"]
                        or frame.pts != timeline["pts"][frame_id]
                        or Fraction(frame.time_base) != Fraction(timeline["time_base"]["numerator"], timeline["time_base"]["denominator"])):
                    raise ValueError(f"decoded dimensions/PTS differ at source frame {frame_id}")
                if frame_id not in selected:
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                image = torch.from_numpy(rgb).permute(2, 0, 1).to(args.device, dtype=torch.float32) / 255.0
                inference_start = time.perf_counter()
                with torch.inference_mode():
                    prediction = model.infer(image, num_tokens=args.num_tokens,
                                             use_fp16=args.device == "cuda", fov_x=None)
                matrix = prediction["intrinsics"].detach().float().cpu().numpy()
                mask = prediction["mask"].detach().cpu().numpy().astype(bool)
                depth = prediction["depth"].detach().float().cpu().numpy()
                raw = original_pixel_intrinsics(matrix.tolist(), row["width"], row["height"])
                if mask.shape != rgb.shape[:2] or depth.shape != mask.shape:
                    raise ValueError("MoGe returned incorrect dense output dimensions")
                invalid = int((mask & (~np.isfinite(depth) | (depth <= 0))).sum())
                if invalid or not mask.any():
                    raise ValueError("MoGe returned invalid valid-region geometry or no valid pixels")
                raw.update(frame_id=frame_id, normalized_intrinsics=matrix.tolist(),
                           pts=int(frame.pts), time_base=timeline["time_base"],
                           rgb_sha256=hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
                           valid_geometry_fraction=float(mask.mean()),
                           inference_seconds=time.perf_counter() - inference_start)
                report["raw_estimates"].append(raw)
                print(json.dumps({"episode_index": index, "frame_id": frame_id,
                                  "fx": raw["fx"], "samples_completed": len(report["raw_estimates"])}), flush=True)
                del prediction, image, matrix, mask, depth, rgb
            if count != row["frame_count"] or len(report["raw_estimates"]) != len(ids):
                raise ValueError("decoded/sample coverage differs from verified original timeline")
        intrinsics, diagnostics, blockers = aggregate_estimates(report["raw_estimates"])
        report.update(status="camera_estimate_inconsistent" if blockers else "camera_estimate_consistent",
                      focal_diagnostics=diagnostics, blockers=blockers,
                      uncertainty_review="Sample spread is temporal estimator disagreement, not absolute calibration error. "
                      "A consistent result remains a monocular estimate; metric accuracy, distortion and principal point are unverified. "
                      + ("Substantial inconsistency blocks automatic Body handoff." if blockers else "All predeclared episode-local consistency screens passed."))
        write_json(directory / ("candidate-intrinsics.json" if blockers else "intrinsics.json"), intrinsics)
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["blockers"].append(str(error))
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "camera-estimation.json", report)
    return report


def run(args: argparse.Namespace) -> dict:
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "camera-batch.json").exists():
        raise FileExistsError("camera-batch.json already exists; use a new attempt directory")
    summary = {"schema_version": "body_camera_batch.v1", "status": "failed", "episodes": []}
    started = time.perf_counter()
    try:
        if not 1200 <= args.num_tokens <= 3600:
            raise ValueError("num-tokens must be in the pinned model's supported 1200-3600 range")
        if not 9 <= args.samples <= 15:
            raise ValueError("samples must be 9-15")
        if len(set(args.episodes)) != len(args.episodes) or not set(args.episodes) <= set(range(30)):
            raise ValueError("episode indices must be unique and in 0-29")
        manifest_path, dataset_root = args.manifest.resolve(), args.dataset_root.resolve()
        manifest = load_raw_inputs(manifest_path, dataset_root)
        moge_source = verify_checkout(args.moge_source, SOURCE_PIN)
        utils_source = verify_checkout(args.utils3d_source, UTILS3D_PIN)
        checkpoint = args.checkpoint.resolve(strict=True)
        if checkpoint.stat().st_size != MODEL_BYTES or sha256(checkpoint) != MODEL_SHA256:
            raise ValueError("checkpoint is not the pinned MoGe2 ViT-L model")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path[:0] = [moge_source["path"], utils_source["path"]]
        import torch
        import av
        import numpy as np
        import utils3d
        from moge.model.v2 import MoGeModel

        if not Path(sys.modules[MoGeModel.__module__].__file__).resolve().is_relative_to(Path(moge_source["path"])):
            raise ValueError("loaded MoGe module does not belong to verified source checkout")
        if not Path(utils3d.__file__).resolve().is_relative_to(Path(utils_source["path"])):
            raise ValueError("loaded utils3d does not belong to verified source checkout")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; select --device cpu explicitly if required")
        torch.manual_seed(0)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                       "torch_cuda": torch.version.cuda, "numpy": np.__version__, "av": av.__version__,
                       "device": args.device, "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())}
        if args.device == "cuda":
            environment.update(cuda_device=torch.cuda.get_device_name(), cuda_capability=list(torch.cuda.get_device_capability()))
        provenance = {"created_at": datetime.now(timezone.utc).isoformat(),
                      "raw_manifest_sha256": sha256(manifest_path), "dataset_revision": DATA_REVISION,
                      "moge_source": portable_source_identity(moge_source, "microsoft/MoGe"),
                      "utils3d_source": portable_source_identity(utils_source, "EasternJournalist/utils3d"),
                      "checkpoint": {"repository": "Ruicheng/moge-2-vitl-normal", "revision": MODEL_REVISION,
                                     "sha256": MODEL_SHA256, "size_bytes": MODEL_BYTES},
                      "adapter_sha256": sha256(Path(__file__)), "environment": environment,
                      "inference": {"samples": args.samples, "num_tokens": args.num_tokens, "batch_size": 1,
                                    "precision": "fp16 autocast with fp32 geometry recovery" if args.device == "cuda" else "fp32",
                                    "supplied_fov": None, "seed": 0, "native_sdpa": True}}
        summary["provenance"] = provenance
        # Strict loading closes the upstream from_pretrained(strict=False) gap.
        weights = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        model = MoGeModel(**weights["model_config"])
        model.load_state_dict(weights["model"], strict=True)
        del weights
        model.enable_pytorch_native_sdpa()
        model = model.eval().to(args.device)
        for index in args.episodes:
            row = next(row for row in manifest["episodes"] if row["episode_index"] == index)
            video, timeline = episode_inputs(manifest, row, manifest_path, dataset_root)
            report = estimate_episode(model, row, video, timeline, args, provenance)
            summary["episodes"].append({"episode_index": index, "status": report["status"], "blockers": report["blockers"],
                                        "report_sha256": sha256(args.output / f"episode_{index:06d}" / "camera-estimation.json")})
        summary["status"] = "complete" if all(row["status"] == "camera_estimate_consistent" for row in summary["episodes"]) else "requires_attention"
        if args.device == "cuda":
            summary["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            summary["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    except Exception as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output / "camera-batch.json", summary)
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
    parser.add_argument("--samples", type=int, default=13)
    parser.add_argument("--num-tokens", type=int, default=1600)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    try:
        summary = run(args)
    except Exception as error:
        parser.exit(2, f"Camera preparation failed: {error}\n")
    print(json.dumps({"status": summary["status"], "episodes": len(summary["episodes"])}))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
