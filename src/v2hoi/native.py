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
        "FOUNDATIONPOSE_RENDER_BATCH_SIZE": "64", "FOUNDATIONPOSE_REFINE_BATCH_SIZE": "128",
        "FOUNDATIONPOSE_DEPTH_BATCH_SIZE": "32",
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
    p.add_argument("--postopt-learning-rate", type=float, default=1e-3)
    p.add_argument("--postopt-batch-size", type=int, default=0)
    p.add_argument("--object-refinement", choices=("none", "observations"), default="observations",
                   help="Run the experimental generic object observation stage after native inference")
    p.add_argument("--cuda-memory-fraction", type=float, default=.75)
    p.add_argument("--allow-experimental-preparation", action="store_true",
                   help="Diagnostic runs only: explicitly allow prior non-official preparation")
    p.add_argument("--allow-budgeted-mesh", action="store_true",
                   help="Explicitly select organizer 4096-face decimation before inference; changes the surface")
    p.add_argument("--allow-legacy-scale-search", action="store_true",
                   help="Explicit legacy scale-preprocessing comparison; original MHR stages remain unchanged")
    p.add_argument("--allow-unvalidated-object-scale", action="store_true",
                   help="Historical diagnostics only: generated scale has not passed independent calibration")
    p.add_argument("--optimize-object-rotation", action="store_true",
                   help="Enable the original NVIDIA rotation optimization option (default is frozen)")
    p.add_argument("--moge-batch-size", type=int, default=4)
    p.add_argument("--sam3d-chunk-size", type=int, default=4)
    p.add_argument("--sam3d-refit-batch-size", type=int, default=64)
    p.add_argument("--depth-render-batch-size", type=int, default=4)
    p.add_argument("--coconet-render-batch-size", type=int, default=4)
    p.add_argument("--render-batch-size", type=int, default=4)
    p.add_argument("--foundationpose-render-batch-size", type=int, default=64)
    p.add_argument("--foundationpose-refine-batch-size", type=int, default=128)
    p.add_argument("--foundationpose-depth-batch-size", type=int, default=32)
    a = p.parse_args()
    if not 0 < a.cuda_memory_fraction <= 1:
        p.error("--cuda-memory-fraction must be between 0 (exclusive) and 1")
    import math
    if not math.isfinite(a.postopt_learning_rate) or a.postopt_learning_rate <= 0:
        p.error("--postopt-learning-rate must be finite and positive")
    clip = load_clips("track1", [a.episode], a.dataset_root)[0]
    prepared = json.loads(a.preparation.read_text())
    if prepared.get("provenance") != "track1" or prepared.get("episode") != a.episode:
        raise ValueError("Preparation must identify this Track 1 episode")
    if a.object_refinement=='observations' and prepared.get('mesh_generation',{}).get('entry')!='blender.bpy.observed-profile':
        raise ValueError('The owner-selected mainline requires a bpy object asset; --object-refinement none is an explicit legacy comparison')
    faithful_preparation = (
        prepared.get("mesh_generation", {}).get("entry") == "v2d.sam3d.lib.image_to_mesh.image_to_mesh"
        and prepared.get("segmentation", {}).get("entry") == "v2d.sam2.lib.video_to_masks.video_to_masks"
        and prepared.get("segmentation", {}).get("fill_hole_area") == 8
        and prepared.get("segmentation", {}).get("jpeg_frame_cache") is False)
    if not faithful_preparation and not a.allow_experimental_preparation:
        raise ValueError("Official baseline requires generated SAM3D object geometry and official SAM2 preparation; "
                         "the previous convex/JPEG preparation is an experiment and is rejected by default")
    batch_names = ("moge_batch_size", "sam3d_chunk_size", "sam3d_refit_batch_size", "depth_render_batch_size",
                   "coconet_render_batch_size", "render_batch_size", "foundationpose_render_batch_size",
                   "foundationpose_refine_batch_size", "foundationpose_depth_batch_size")
    if any(getattr(a, name) < 1 for name in batch_names):
        p.error("All runtime batch sizes must be positive")
    video_hash = sha256(clip.video)
    if video_hash != prepared["video_sha256"]:
        raise ValueError("Preparation video differs from the Track 1 metadata input")
    inputs = {}
    for name in ("masks", "mesh"):
        path = (a.preparation.parent / prepared[name]["path"]).resolve()
        if sha256(path) != prepared[name]["sha256"]:
            raise ValueError(f"Preparation {name} content changed")
        inputs[name] = path
    budgeted_mesh = False
    if faithful_preparation:
        evidence = {}
        for stage, key in (("segmentation", "report"), ("mesh_generation", "receipt")):
            artifact = prepared[stage][key]
            path = (a.preparation.parent / artifact["path"]).resolve()
            if sha256(path) != artifact["sha256"]:
                raise ValueError(f"Official preparation evidence changed: {stage}")
            evidence[stage] = json.loads(path.read_text())
        segmentation, generation = evidence["segmentation"], evidence["mesh_generation"]
        if (segmentation["mask_sha256"] != prepared["masks"]["sha256"]
                or segmentation["frames"] != clip.n_frames
                or generation["input_record"]["video_sha256"] != video_hash
                or generation["input_record"]["mask_sha256"] != prepared["masks"]["sha256"]
                or generation["surface_replacement"] is not False
                or generation["sampler_step_override"] is not False):
            raise ValueError("Official preparation receipts do not match the actual inputs")
        from v2hoi.mesh_budget import verify_budgeted_preparation
        if "mesh_scale_search" in prepared and not a.allow_legacy_scale_search:
            raise ValueError("Legacy scale search requires --allow-legacy-scale-search")
        budgeted_mesh = verify_budgeted_preparation(prepared, a.preparation.parent,
            generation["mesh_sha256"], allowed=a.allow_budgeted_mesh)
        if "mesh_calibration" not in prepared and not a.allow_unvalidated_object_scale:
            raise ValueError("Generated object requires validated metric mesh calibration before tracking; predicted scale alone is insufficient")
    elif "input_mesh_policy" in prepared:
        raise ValueError("Mesh budgeting requires verified official generation and segmentation")
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
        "--expected_frames", str(clip.n_frames), "--moge_batch_size", str(a.moge_batch_size), "--sam3d_chunk_size", str(a.sam3d_chunk_size),
        "--sam3d_refit_batch_size", str(a.sam3d_refit_batch_size), "--depth_alignment_render_batch_size", str(a.depth_render_batch_size),
        "--depth_alignment_encoding_workers", "8", "--depth_alignment_write_batch_size", "16",
        "--coconet_render_batch_size", str(a.coconet_render_batch_size), "--coconet_crop_workers", "1", "--render_batch_size", str(a.render_batch_size),
        "--postopt_num_steps", str(a.postopt_steps), "--postopt_batch_size", str(a.postopt_batch_size)]
    if a.optimize_object_rotation:
        command.append("--optimize_object_rotation")
    if a.postopt_learning_rate != 1e-3:
        command += ["--postopt_lr", str(a.postopt_learning_rate)]
    env = native_environment(source, weights)
    env["V2HOI_CUDA_MEMORY_FRACTION"] = str(a.cuda_memory_fraction)
    env["FOUNDATIONPOSE_RENDER_BATCH_SIZE"] = str(a.foundationpose_render_batch_size)
    env["FOUNDATIONPOSE_REFINE_BATCH_SIZE"] = str(a.foundationpose_refine_batch_size)
    env["FOUNDATIONPOSE_DEPTH_BATCH_SIZE"] = str(a.foundationpose_depth_batch_size)
    identity = {"schema": "v2hoi.native-run.v1", "provenance": "track1", "episode": a.episode,
        "preparation_manifest":str(a.preparation.resolve()),
        "video_sha256": video_hash, "expected_frames": clip.n_frames,
        "original_frame_ids": list(range(clip.n_frames)), "preparation": prepared,
        "source_sha256": source_hash.hexdigest(), "command": command,
        "official_preparation": faithful_preparation and not budgeted_mesh and "mesh_calibration" in prepared,
        "algorithm_deviations": ["Kaolin penetration fallback is permitted but unproven equivalent"] +
            ([] if faithful_preparation else ["Explicitly selected experimental preparation"]) +
            (["Organizer 4096-face budget applied before inference; changes tracking/contact/penetration surface"] if budgeted_mesh else []),
        "object_scale_calibrated": "mesh_calibration" in prepared,
        "legacy_scale_search": "mesh_scale_search" in prepared,
        "object_rotation_optimization": a.optimize_object_rotation,
        "postopt_learning_rate": a.postopt_learning_rate,
        "cuda_budget_wrapper_sha256": sha256(Path(__file__).with_name("cuda_budget.py")),
        "entry_sha256": sha256(Path(__file__)),
        "runtime_environment": {k: env[k] for k in ("V2HOI_CUDA_MEMORY_FRACTION", "PYTORCH_ALLOC_CONF",
            "FOUNDATIONPOSE_RENDER_BATCH_SIZE", "FOUNDATIONPOSE_REFINE_BATCH_SIZE",
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
    if a.object_refinement == "observations":
        refinement_command = [sys.executable, "-m", "v2hoi.object_refine", "--runs", str(a.output.resolve()),
            "--dataset-root", str(a.dataset_root.resolve()), "--output", str(a.output.resolve()/"object-refinement")]
        with (a.output/"object-refinement.log").open("w", encoding="utf-8") as log:
            subprocess.run(refinement_command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        pipeline_receipt={"native_status":"complete","object_refinement":"observations",
            "object_refinement_manifest":str((a.output/"object-refinement/batch.json").resolve()),
            "object_refinement_manifest_sha256":sha256(a.output/"object-refinement/batch.json"),
            "candidate_selected_for_submission":False,"submission_export_requested":False}
        (a.output/"pipeline-status.json").write_text(json.dumps(pipeline_receipt,indent=2))


if __name__ == "__main__":
    main()
