"""Run the pinned legacy NVIDIA scale candidate search on a controlled mesh.

Only Track 1 inputs from the specified completed MHR run are used. The legacy
search uses the SAME current FoundationPose implementation and model weights.
It replaces the scale of the control's exact 4096-face surface, never its shape.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import trimesh
from v2hoi.legacy_scale import METHOD, REVISION, adapted_search_source
from v2hoi.native import native_environment, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--control-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--legacy-source", type=Path, default=Path("D:/Projects/CARI4D_Win"))
    p.add_argument("--wait-pid", type=int)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse the original search's saved coarse/fine candidates after an I/O interruption")
    a = p.parse_args()
    if not a.worker:
        if a.wait_pid:
            import psutil
            try:
                owner = psutil.Process(a.wait_pid)
                command = owner.cmdline()
                if not any("finish_variant_run.py" in v or "v2hoi.native_export" in v for v in command) or Path(command[command.index("--run")+1]).resolve() != a.control_run.resolve():
                    raise ValueError("Wait PID is not the specified control's exporter")
                print("Waiting for the control's existing GPU/export work", flush=True)
                while owner.is_running():
                    try:
                        owner.wait(timeout=5)
                        break
                    except psutil.TimeoutExpired:
                        pass
            except psutil.NoSuchProcess:
                pass
            identity = json.loads((a.control_run/"native-inputs.json").read_text())
            if not (a.control_run / "export" / f"episode_{identity['episode']:06d}.npz").is_file():
                raise ValueError("Control export failed; do not overlap GPU work")
        identity = json.loads((a.control_run/"native-inputs.json").read_text())
        command = identity["command"]
        weights = Path(command[command.index("--weights_path")+1]).resolve()
        environment = native_environment(Path("third_party/CARI4D_MHR_Win").resolve(), weights)
        environment.update(FOUNDATIONPOSE_RENDER_BATCH_SIZE="256", FOUNDATIONPOSE_REFINE_BATCH_SIZE="512", FOUNDATIONPOSE_DEPTH_BATCH_SIZE="32")
        report = a.output.parent / (a.output.name + ("-resume-memory.json" if a.resume else "-memory.json"))
        subprocess.run([sys.executable, "-m", "v2hoi.cuda_budget", "--fraction", ".75", "--report", str(report), "--",
                        str(Path(__file__)), *sys.argv[1:], "--worker"], env=environment, check=True)
        return

    prepared = json.loads(a.preparation.read_text())
    identity = json.loads((a.control_run/"native-inputs.json").read_text())
    if (prepared["provenance"] != "track1" or identity["provenance"] != "track1"
            or prepared != identity["preparation"] or "mesh_scale_search" in prepared):
        raise ValueError("Use the matching Track 1 control and its original preparation")
    from v2hoi.mesh_budget import verify_budgeted_preparation
    generation_path = (a.preparation.parent / prepared["mesh_generation"]["receipt"]["path"]).resolve()
    generation = json.loads(generation_path.read_text())
    verify_budgeted_preparation(prepared, a.preparation.parent, generation["mesh_sha256"], allowed=True)
    if not json.loads((a.control_run/"native-status.json").read_text())["complete"]:
        raise ValueError("Control's real MHR stages are incomplete")
    output = a.output.resolve()
    if output.exists() and not a.resume:
        raise FileExistsError("Preserve candidates; use a new output directory")
    output.mkdir(parents=True, exist_ok=a.resume)
    original_mesh = (a.preparation.parent / prepared["mesh"]["path"]).resolve()
    mesh = trimesh.load(original_mesh, force="mesh", process=False)
    if len(mesh.faces) != 4096 or mesh.visual.kind != "vertex":
        raise ValueError("Expected the control's colored 4096-face surface")
    calibration_path = (a.preparation.parent / prepared["mesh_calibration"]["receipt"]["path"]).resolve()
    calibration = json.loads(calibration_path.read_text())
    scales = np.asarray(generation["baked_scale"])
    if not np.allclose(scales, scales[0]):
        raise ValueError("Nonuniform generation scale is unsupported")
    divisor = float(scales[0] * calibration["scale_factor"])
    normalized = mesh.copy()
    normalized.vertices /= divisor
    normalized_path = output / "legacy_rgba.obj"
    if not normalized_path.exists():
        normalized.export(normalized_path)
    loaded = trimesh.load(normalized_path, force="mesh", process=False)
    if not np.array_equal(loaded.faces, normalized.faces) or not np.allclose(loaded.vertices, normalized.vertices, atol=1e-7, rtol=0):
        raise ValueError("Legacy OBJ adapter changed the normalized surface")
    source_directory = Path("third_party/cari4d-legacy-scale")
    source_directory.mkdir(exist_ok=True)
    sources = {}
    for name in ("estimate_scale.py", "chamfer_dist_np.py"):
        payload = subprocess.check_output(["git", "-C", str(a.legacy_source), "show", f"{REVISION}:tools/{name}"])
        saved = source_directory / name
        if saved.exists() and saved.read_bytes() != payload:
            raise ValueError("Preserved legacy source changed")
        saved.write_bytes(payload)
        sources[name] = {"path": str(saved.resolve()), "sha256": sha256(saved)}
    source = (source_directory/"estimate_scale.py").read_text()
    adapted = adapted_search_source(source, str(output/"debug"))
    adapter_file = output/"estimate_scale_io_adapter.py"
    if adapter_file.exists() and adapter_file.read_text() != adapted:
        backup = output/"estimate_scale_io_adapter-before-resume.py"
        if backup.exists():
            raise ValueError("Preserve every I/O adapter attempt; refusing another overwrite")
        backup.write_bytes(adapter_file.read_bytes())
    adapter_file.write_text(adapted)
    import torch
    import cv2
    import nvdiffrast.torch as dr
    from tqdm import tqdm
    import Utils
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    from prep.mhr_depth_h5 import read_metric_depth
    from prep.mhr_export_utils import read_rgb, read_mask, camera_calibration, load_edex
    module_spec = importlib.util.spec_from_file_location("legacy_scale_chamfer", source_directory/"chamfer_dist_np.py")
    chamfer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(chamfer)
    seq = f"episode_{identity['episode']:06d}"
    native = a.control_run / "native" / seq
    export = native / "export" / seq
    depth_path = native / "depth/moge2/aligned.h5"
    depth_m = read_metric_depth(depth_path, "aligned", "front_stereo_camera_left", "000000")
    depth_mm = np.rint(depth_m*1000).astype(np.uint16)
    color = read_rgb(export, 0, "000000")
    mask = read_mask(export, "object", 0, "000000")
    camera_k, _ = camera_calibration(load_edex(export), 0)
    def save_scaled_mesh(path, destination, scale):
        saved = trimesh.load(path, force="mesh", process=False)
        saved.vertices *= scale
        saved.export(destination)
    namespace = dict(torch=torch, os=os, json=json, trimesh=trimesh, np=np, tqdm=tqdm,
                     cv2=cv2, osp=os.path, FoundationPose=FoundationPose, Utils=Utils,
                     chamfer_distance=chamfer.chamfer_distance, _save_scaled_mesh=save_scaled_mesh)
    exec(compile(adapted, str(adapter_file), "exec"), namespace)
    results = output/"search/legacy_rgba/legacy_fp-res-refine.json"
    cached = results.is_file()
    scorer, refiner = (None,None) if cached else (ScorePredictor(), PoseRefinePredictor())
    started = time.perf_counter()
    namespace["estimate_metric_scale"](scorer, refiner, dr.RasterizeCudaContext(), str(output/"search"),
        str(normalized_path), color, depth_mm, mask.astype(np.uint8)*255, np.asarray(camera_k))
    selected = json.loads(results.read_text())
    factor = float(selected["best_scale"] / divisor)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Search selected an invalid scale")
    scaled = mesh.copy()
    scaled.vertices *= factor
    final_mesh = output/"object-legacy-scale4096.glb"
    scaled.export(final_mesh)
    restored = trimesh.load(final_mesh, force="mesh", process=False)
    if not np.array_equal(restored.faces, mesh.faces) or not np.array_equal(restored.visual.vertex_colors, mesh.visual.vertex_colors) or not np.allclose(restored.vertices, mesh.vertices*factor, atol=1e-6, rtol=0):
        raise ValueError("Search output changed geometry beyond uniform scale or lost color")
    def artifact(path):
        return {"path": Path(os.path.relpath(path, output)).as_posix(), "sha256": sha256(path)}
    record = {"status":"COMPLETED", "provenance":"track1", "method":METHOD, "revision":REVISION,
        "scope":"Controlled legacy search preprocessing experiment, not an unmodified MHR default",
        "input_mesh":artifact(original_mesh), "output_mesh":artifact(final_mesh), "source_preparation":artifact(a.preparation.resolve()),
        "official_code":artifact(source_directory/"estimate_scale.py"), "chamfer_code":artifact(source_directory/"chamfer_dist_np.py"),
        "io_adapter":artifact(adapter_file), "wrapper":artifact(Path(__file__).resolve()),
        "normalization_divisor":divisor, "selected_normalized_scale":selected["best_scale"], "scale_factor":factor,
        "coarse_candidates":30, "fine_candidates":11, "registration_iterations":5, "surface_samples":10000,
        "selection":"Original bidirectional unsquared Chamfer distance divided by rendered mask IoU",
        "frame_id":0, "depth":artifact(depth_path), "video_sha256":prepared["video_sha256"], "mask_sha256":prepared["masks"]["sha256"],
        "uniform_scale_only":True, "appearance_preserved":True, "faces_preserved":True,
        "extents_before_m":mesh.extents.tolist(), "extents_after_m":scaled.extents.tolist(),
        "fine_search":artifact(results), "elapsed_seconds":time.perf_counter()-started,
        "reused_completed_candidate_search":cached,
        "limitations":["Estimated human-anchored MoGe2 depth reused; no ground truth or true-size guarantee",
                       "The current MHR FoundationPose implementation and weights are retained",
                       "Single generation-frame search; subsequent whole-video tracking is evaluated separately"]}
    receipt = output/"legacy-scale-search.json"
    receipt.write_text(json.dumps(record, indent=2))
    for key in ("masks",):
        prepared[key]["path"] = artifact((a.preparation.parent/prepared[key]["path"]).resolve())["path"]
    for key, child in (("mesh_generation","receipt"),("segmentation","report"),("mesh_calibration","receipt"),("mesh_budget","receipt")):
        item = prepared[key][child]
        item["path"] = artifact((a.preparation.parent/item["path"]).resolve())["path"]
    prepared.update(mesh=artifact(final_mesh), mesh_scale_search={"receipt":artifact(receipt)},
                    limitations=prepared.get("limitations", [])+record["limitations"])
    verify_budgeted_preparation(prepared, output, generation["mesh_sha256"], allowed=True)
    (output/"preparation.json").write_text(json.dumps(prepared, indent=2))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
