"""Audit completed Track 1 native coverage, bootstrap scale and run evidence."""
import argparse
import json
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
from v2hoi.native import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    a = p.parse_args()
    inputs = json.loads((a.run / "native-inputs.json").read_text())
    status = json.loads((a.run / "native-status.json").read_text())
    prepared = json.loads(a.preparation.read_text())
    if not status["complete"] or inputs["provenance"] != "track1" or inputs["preparation"] != prepared:
        raise ValueError("Incomplete run or mismatched Track 1 preparation")
    for name in ("masks", "mesh", "human_bootstrap", "depth_alignment"):
        if name not in prepared:
            continue
        if sha256((a.preparation.parent / prepared[name]["path"]).resolve()) != prepared[name]["sha256"]:
            raise ValueError(f"Preparation asset changed: {name}")
    seq = f"episode_{inputs['episode']:06d}"
    native = a.run / "native" / seq
    bundle = torch.load(native / "inference/refined.pth", map_location="cpu", weights_only=False)
    ids = [int(x["src_frame"]) for x in bundle["frame_meta"]]
    if ids != list(range(inputs["expected_frames"])) or ids != inputs["original_frame_ids"]:
        raise ValueError("Missing, duplicate or reordered original frames")
    if bundle["metadata"]["ground_truth_used"] is not False or bundle.get("gt"):
        raise ValueError("Unexpected reference data")
    for k, v in bundle["pr"].items():
        if isinstance(v, (np.ndarray, torch.Tensor)) and not np.isfinite(np.asarray(v)).all():
            raise ValueError(f"Nonfinite prediction: {k}")
    camera = "front_stereo_camera_left"
    with h5py.File(native / "depth/moge2/aligned.h5") as actual:
        right = actual[f"alignment/{camera}/scale"][:]
    generated = "mesh_generation" in prepared
    if generated:
        receipt = prepared["mesh_generation"]["receipt"]
        receipt_path = a.preparation.parent / receipt["path"]
        if sha256(receipt_path) != receipt["sha256"]:
            raise ValueError("Object generation evidence changed")
        generation = json.loads(receipt_path.read_text())
        from v2hoi.mesh_budget import verify_budgeted_preparation
        budgeted = verify_budgeted_preparation(prepared, a.preparation.parent,
            generation["mesh_sha256"], allowed=True)
        bootstrap_scale = generation["input_record"]["scale"]
        max_difference = float(abs(right[0] - bootstrap_scale))
        raw_identical = None
        geometry_limit = "Official generative object; inferred monocular shape and scale"
        if budgeted:
            geometry_limit += "; explicit organizer 4096-face decimation before inference"
    else:
        with h5py.File(a.preparation.parent / prepared["depth_alignment"]["path"]) as bootstrap:
            left = bootstrap[f"alignment/{camera}/scale"][:]
        max_difference = float(abs(left-right).max())
        if max_difference > 1e-5 or abs(right[0] - prepared["mesh_method"]["metric_scale_from_human"]) > 1e-5:
            raise ValueError("Native depth scale differs from the object's bootstrap scale")
        raw = json.loads((native / "depth/moge2/depth_report.json").read_text())["rawDepth"]["sha256"]
        raw_identical = raw == sha256(Path(prepared["mesh_method"]["inputs"]["depth"]))
        if not raw_identical:
            raise ValueError("Bootstrap and native passes used different raw depth")
        geometry_limit = "Historical convex visible-depth object experiment"
    rendered = native / "inference" / f"{seq}_step200000_reconstruction.mp4"
    video = cv2.VideoCapture(str(rendered))
    fps = video.get(cv2.CAP_PROP_FPS)
    count = 0
    while video.read()[0]:
        count += 1
    video.release()
    if count != len(ids):
        raise ValueError("Native rendered video is incomplete")
    budget = [json.loads(x.read_text()) for x in sorted((native / ".stages").glob("*.memory.json"))]
    if len(budget) != 8 or not all(x["completed"] for x in budget):
        raise ValueError("Missing stage memory evidence")
    memory = json.loads((a.run / "gpu-memory.json").read_text())
    report = {"real_native_run": "PASS", "provenance": "track1", "episode": inputs["episode"],
              "original_frames": len(ids), "rendered_frames": count, "fps": fps,
              "bootstrap_scale_max_difference": max_difference, "raw_depth_identical": raw_identical,
              "bootstrap_scale_comparison": "First-frame estimate versus native pass" if generated else "Full cached bootstrap pass",
              "official_preparation": inputs.get("official_preparation", False),
              "whole_gpu_peak_sample_mib": memory["peak_observed_mib"],
              "max_torch_peak_allocated_mib": max(x["peak_allocated_bytes"] for x in budget) / 2**20,
              "max_torch_peak_reserved_mib": max(x["peak_reserved_bytes"] for x in budget) / 2**20,
              "refined_bundle_sha256": sha256(native / "inference/refined.pth"),
              "native_video_sha256": sha256(rendered),
              "limitations": ["Single episode only", geometry_limit,
                  "Kaolin penetration fallback is permitted but unproven equivalent",
                  "No reference data or official score", "Bootstrap frame zero is not an asserted hidden reference frame"]}
    (a.run / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
