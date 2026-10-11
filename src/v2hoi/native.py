"""Run the real coupled CARI4D MHR backend without the v1 fake-stage contract.

Prepared masks and mesh must have a reviewed Track 1 preparation manifest.
Native outputs remain authoritative; export them with v2hoi.native_export.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from v2hoi.clips import load_clips


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_environment(source: Path, weights: Path) -> dict[str, str]:
    modules = source / "reconstruction/modules"
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.pathsep.join(str(modules / part) for part in (
            "v2d_cari4d/lib/cari4d", "v2d_sam3d_body/lib", "v2d_foundation_pose/lib/FoundationPose")),
        "HF_HOME": str(weights / "hf_home"), "TORCH_HOME": str(weights / "sam3d_body/torch_home"),
        "SAM3D_BODY_ROOT": str(modules / "v2d_sam3d_body/lib"),
        "DINOV3_SOURCE": str(weights / "sam3d_body/torch_home/hub/facebookresearch_dinov3_main"),
        "CARI4D_DINOV2_SOURCE": str(source.parent / "dinov2"),
        "MHR_ASSETS_ROOT": str(weights / "sam3d_body"),
        "FOUNDATIONPOSE_WEIGHTS_DIR": str(weights / "foundationpose/nvlabs_pytorch"),
        "FOUNDATIONPOSE_RENDER_BATCH_SIZE": "16", "FOUNDATIONPOSE_REFINE_BATCH_SIZE": "16",
        "FOUNDATIONPOSE_SCORE_FEATURE_BATCH_SIZE": "16",
        "FOUNDATIONPOSE_DEPTH_BATCH_SIZE": "16",
        "V2HOI_CUDA_MEMORY_FRACTION": env.get("V2HOI_CUDA_MEMORY_FRACTION", "0.75"),
        "PYTORCH_ALLOC_CONF": env.get("PYTORCH_ALLOC_CONF", "garbage_collection_threshold:0.8"),
        "HDF5_USE_FILE_LOCKING": "FALSE", "PYTHONUNBUFFERED": "1", "OPENCV_IO_ENABLE_OPENEXR": "1",
    })
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path("third_party/CARI4D_MHR_Win"))
    p.add_argument("--weights", type=Path, default=Path("weights/cari4d"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--postopt-steps", type=int, default=300)
    p.add_argument("--postopt-batch-size", type=int, default=0)
    p.add_argument("--cuda-memory-fraction", type=float, default=.75)
    a = p.parse_args()
    if not 0 < a.cuda_memory_fraction <= 1:
        p.error("--cuda-memory-fraction must be between 0 (exclusive) and 1")
    clip = load_clips("track1", [a.episode], a.dataset_root)[0]
    prepared = json.loads(a.preparation.read_text())
    if prepared.get("provenance") != "track1" or prepared.get("episode") != a.episode:
        raise ValueError("Preparation must identify this Track 1 episode")
    video_hash = sha256(clip.video)
    if video_hash != prepared["video_sha256"]:
        raise ValueError("Preparation video differs from the Track 1 metadata input")
    inputs = {}
    for name in ("masks", "mesh"):
        path = (a.preparation.parent / prepared[name]["path"]).resolve()
        if sha256(path) != prepared[name]["sha256"]:
            raise ValueError(f"Preparation {name} content changed")
        inputs[name] = path
    a.output.mkdir(parents=True, exist_ok=True)
    video = a.output.resolve() / "inputs" / f"episode_{a.episode:06d}.0.color.mp4"
    video.parent.mkdir(exist_ok=True)
    if video.exists():
        if sha256(video) != video_hash:
            raise ValueError("Run input video changed")
    else:
        shutil.copy2(clip.video, video)
    source, weights = a.source.resolve(), a.weights.resolve()
    # Hash the actual imported sources, including untracked compatibility files.
    source_hash = hashlib.sha256()
    for path in sorted((source / "reconstruction/modules").rglob("*.py")):
        if ".venv" not in path.parts:
            source_hash.update(path.relative_to(source).as_posix().encode())
            source_hash.update(path.read_bytes())
    command = [sys.executable, "-m", "v2d.cari4d.lib.run_inference",
        "--video_path", str(video), "--mask_h5_path", str(inputs["masks"]),
        "--object_mesh_path", str(inputs["mesh"]), "--weights_path", str(weights),
        "--output_dir", str(a.output.resolve() / "native"), "--skip_weight_download",
        "--expected_frames", str(clip.n_frames), "--moge_batch_size", "1", "--sam3d_chunk_size", "1",
        "--sam3d_refit_batch_size", "16", "--depth_alignment_render_batch_size", "1",
        "--depth_alignment_encoding_workers", "8", "--depth_alignment_write_batch_size", "16",
        "--coconet_render_batch_size", "1", "--coconet_crop_workers", "1", "--render_batch_size", "1",
        "--postopt_num_steps", str(a.postopt_steps), "--postopt_batch_size", str(a.postopt_batch_size)]
    env = native_environment(source, weights)
    env["V2HOI_CUDA_MEMORY_FRACTION"] = str(a.cuda_memory_fraction)
    identity = {"schema": "v2hoi.native-run.v1", "provenance": "track1", "episode": a.episode,
        "video_sha256": video_hash, "expected_frames": clip.n_frames,
        "original_frame_ids": list(range(clip.n_frames)), "preparation": prepared,
        "source_sha256": source_hash.hexdigest(), "command": command,
        "cuda_budget_wrapper_sha256": sha256(Path(__file__).with_name("cuda_budget.py")),
        "entry_sha256": sha256(Path(__file__)),
        "runtime_environment": {k: env[k] for k in ("V2HOI_CUDA_MEMORY_FRACTION", "PYTORCH_ALLOC_CONF",
            "FOUNDATIONPOSE_RENDER_BATCH_SIZE", "FOUNDATIONPOSE_REFINE_BATCH_SIZE", "FOUNDATIONPOSE_SCORE_FEATURE_BATCH_SIZE",
            "FOUNDATIONPOSE_DEPTH_BATCH_SIZE")}}
    manifest = a.output / "native-inputs.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Inputs/source/options changed; use a new run directory to preserve the previous run")
    manifest.write_text(json.dumps(identity, indent=2))
    samples = []
    with (a.output / "native.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                query = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                if query.returncode == 0:
                    for gpu, row in enumerate(query.stdout.strip().splitlines()):
                        used, total, utilization = (int(x.strip()) for x in row.split(","))
                        samples.append({"time": time.time(), "gpu": gpu, "used_mib": used,
                                        "total_mib": total, "utilization_percent": utilization})
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
    (a.output / "gpu-memory.json").write_text(json.dumps({"scope": "Whole-device samples, including desktop and other applications; 2 second interval",
        "peak_observed_mib": max((x["used_mib"] for x in samples), default=None), "samples": samples}, indent=2))
    (a.output / "native-status.json").write_text(json.dumps({"exit_code": process.returncode, "complete": process.returncode == 0}, indent=2))
    if process.returncode:
        raise SystemExit(f"Native backend failed ({process.returncode}); see {a.output / 'native.log'}")


if __name__ == "__main__":
    main()
