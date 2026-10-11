"""Calibrate one constant mesh scale with the original NVIDIA SRT solver.

The input adapter uses Track 1 pilot object-to-camera poses as relative views
in object coordinates, observed masks and human-anchored estimated depth. This
is not the upstream stereo/SfM scan workflow and is recorded explicitly.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import hashlib
import h5py
import numpy as np
import trimesh
from v2hoi.native import sha256

REVISION = "193382a184308f6800331f3db829f76dc4294cfe"
SOURCE = "reconstruction/modules/v2d_hoi_object_reconstruction/lib/scale_mesh_srt.py"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--pilot-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-pilot-depth-gap", type=float, default=.15,
                   help="Adapter view-quality gate in metres; reject already drifted pilot poses")
    p.add_argument("--fit-mode", choices=("s", "str"), default="str",
                   help="Original SRT option; rigid fit is nuisance registration, only scale is applied to the output mesh")
    p.add_argument("--use-depth-loss",action="store_true",
                   help="Original optional depth loss; upstream SRT capture workflow defaults to silhouette fitting")
    a = p.parse_args()
    prepared = json.loads(a.preparation.read_text())
    identity = json.loads((a.pilot_run / "native-inputs.json").read_text())
    if (prepared["provenance"] != "track1" or identity["provenance"] != "track1"
            or prepared != identity["preparation"] or "mesh_calibration" in prepared):
        raise ValueError("Use the original Track 1 preparation and its matching pilot")
    from v2hoi.mesh_budget import verify_budgeted_preparation
    generation = json.loads((a.preparation.parent / prepared["mesh_generation"]["receipt"]["path"]).read_text())
    verify_budgeted_preparation(prepared, a.preparation.parent, generation["mesh_sha256"], allowed=False)
    original_mesh = (a.preparation.parent / prepared["mesh"]["path"]).resolve()
    mask_path = (a.preparation.parent / prepared["masks"]["path"]).resolve()
    if sha256(mask_path) != prepared["masks"]["sha256"]:
        raise ValueError("Pilot masks changed")
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve all previous candidates; use a new output")
    output.mkdir(parents=True)
    code = subprocess.check_output(["git", "-C", "third_party/CARI4D_MHR_Win", "show", f"{REVISION}:{SOURCE}"])
    code_path = Path("third_party/nvidia-srt/scale_mesh_srt.py").resolve()
    code_path.parent.mkdir(exist_ok=True)
    if code_path.exists() and code_path.read_bytes() != code:
        raise ValueError("Pinned SRT source changed")
    if not code_path.exists():
        code_path.write_bytes(code)
    spec = importlib.util.spec_from_file_location("pinned_nvidia_srt", code_path)
    srt = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = srt
    spec.loader.exec_module(srt)
    os.environ["V2D_SRT_CKDTREE_WORKERS"] = "1"
    from prep.mhr_depth_h5 import read_metric_depth
    seq = f"episode_{prepared['episode']:06d}"
    native = a.pilot_run / "native" / seq
    fp_path = native / "foundationpose" / f"{seq}.pkl"
    with fp_path.open("rb") as handle:
        tracking = pickle.load(handle)
    if tracking["metadata"]["gt_object_pose_available"]:
        raise ValueError("Pilot must not contain reference pose data")
    export = native / "export" / seq
    meta = json.loads((export / "wild_export.json").read_text())
    k = np.asarray(meta["intrinsics"])
    intrinsics = srt._Intrinsics(k[0,0], k[1,1], k[0,2], k[1,2], meta["width"], meta["height"])
    mesh_path = export / "object_mesh/output_aligned.glb"
    aligned = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(aligned.vertices, dtype=np.float64)
    train_ids = np.linspace(0, prepared["original_frames"]-1, 25).round().astype(int).tolist()
    holdout_ids = [int((left+right)//2) for left,right in zip(train_ids[:-1], train_ids[1:])]
    depth_path = native / "depth/moge2/aligned.h5"
    rejected_views = []
    def views(ids):
        result = []
        with h5py.File(mask_path) as masks:
            for frame_id in ids:
                mask = masks[f"{seq}/{frame_id:06d}-k0.obj_rend_mask.png"][()].astype(np.uint8)*255
                ys,xs = np.nonzero(mask)
                if len(xs) < 50:
                    continue
                depth = read_metric_depth(depth_path, "aligned", "front_stereo_camera_left", f"{frame_id:06d}")
                observed = depth[mask.astype(bool)]
                observed = observed[np.isfinite(observed) & (observed > .01)]
                pose = np.asarray(tracking["fp_poses_cam"][frame_id,0], dtype=np.float64)
                gap = float(abs(pose[2,3]-np.median(observed))) if len(observed) else float("inf")
                if gap > a.max_pilot_depth_gap:
                    rejected_views.append({"frame":frame_id, "pilot_depth_gap_m":gap})
                    continue
                result.append(srt.FrameView(str(frame_id), intrinsics,
                    pose,
                    mask, srt._compute_dt(mask), srt._mask_boundary(mask),
                    float(max(np.ptp(xs), np.ptp(ys))), depth if a.use_depth_loss else None))
        return result
    training, holdout = views(train_ids), views(holdout_ids)
    if min(len(training), len(holdout)) < 3:
        raise ValueError("Insufficient separate fitting and holdout observations")
    center = np.zeros(3)
    orient = np.eye(3)
    kwargs = dict(orient=orient, rotvec=np.zeros(3), translation=np.zeros(3), mesh_center=center)
    before_train = srt.evaluate_srt(vertices, training, scale=1., **kwargs)
    before_holdout = srt.evaluate_srt(vertices, holdout, scale=1., **kwargs)
    initial = srt._estimate_initial_scale(training, float(aligned.extents.max()), center)
    fitted = srt.optimize_srt(vertices, training, orient, center, np.zeros(3), initial, mode=a.fit_mode)
    factor = float(fitted["scale"])
    fit_kwargs = dict(kwargs, rotvec=np.asarray(fitted["rotvec"]), translation=np.asarray(fitted["translation"]))
    after_train = srt.evaluate_srt(vertices, training, scale=factor, **fit_kwargs)
    after_holdout = srt.evaluate_srt(vertices, holdout, scale=factor, **fit_kwargs)
    def ious(scale, rotation, translation):
        result = []
        for view in holdout:
            from scipy.spatial.transform import Rotation
            positioned = (Rotation.from_rotvec(rotation).as_matrix() @ (vertices*scale).T).T + translation
            camera = (view.world_to_cam[:3,:3] @ positioned.T).T + view.world_to_cam[:3,3]
            u,v,valid = srt._project(camera, view.intrinsics)
            result.append(1-srt._raster_iou(u[valid],v[valid],view.mask_u8))
        return result
    before_ious = ious(1., np.zeros(3), np.zeros(3))
    after_ious = ious(factor, fit_kwargs["rotvec"], fit_kwargs["translation"])
    accepted = (np.isfinite(factor) and factor > 0 and after_train < before_train
        and after_holdout < before_holdout and np.mean(after_ious) >= tracking["metadata"]["reacquisition_iou_min"]
        and np.mean(after_ious) > np.mean(before_ious))
    mesh_output = output / "object-calibrated.glb"
    srt.export_scaled_glb(original_mesh, mesh_output, orientation_matrix=orient,
        rotation_matrix=orient, scale=factor, mesh_center=center, translation=np.zeros(3))
    original = trimesh.load(original_mesh, force="mesh", process=False)
    scaled = trimesh.load(mesh_output, force="mesh", process=False)
    if (not np.allclose(scaled.vertices, original.vertices*factor, atol=1e-6, rtol=0)
            or not np.array_equal(scaled.faces, original.faces)
            or not np.array_equal(scaled.visual.vertex_colors, original.visual.vertex_colors)):
        raise ValueError("Calibration changed geometry beyond uniform scaling or lost original colors")
    def artifact(path):
        return {"path": Path(os.path.relpath(path, output)).as_posix(), "sha256": sha256(path)}
    record = {"status": "PASS" if accepted else "REJECTED", "provenance": "track1",
        "method": f"NVIDIA scale_mesh_srt.optimize_srt(mode={a.fit_mode})", "scale_factor": factor,
        "input_mesh": artifact(original_mesh), "output_mesh": artifact(mesh_output),
        "official_code": artifact(code_path), "official_revision": REVISION,
        "official_code_sha256": hashlib.sha256(code).hexdigest(), "pilot_poses": artifact(fp_path),
        "depth": artifact(depth_path), "video_sha256": prepared["video_sha256"],
        "mask_sha256": prepared["masks"]["sha256"],
        "adapter": "Estimated pilot object-to-camera transforms used as relative views in object coordinates; no stereo/SfM or reference poses",
        "training_frame_ids": [int(v.frame_name) for v in training], "holdout_frame_ids": [int(v.frame_name) for v in holdout],
        "loss": {"train_before": before_train, "train_after": after_train,
            "holdout_before": before_holdout, "holdout_after": after_holdout}, "solver": fitted,
        "holdout_projected_vertex_iou": {"before":before_ious,"after":after_ious,
            "mean_after":float(np.mean(after_ious)), "minimum_mean":tracking["metadata"]["reacquisition_iou_min"]},
        "view_quality_adapter": {"max_pilot_depth_gap_m":a.max_pilot_depth_gap,"rejected":rejected_views},
        "use_depth_loss":a.use_depth_loss,
        "uniform_scale_only": True, "appearance_preserved": True,
        "nuisance_registration": "Rigid parameters fitted for scale estimation only; new tracking estimates scene poses independently",
        "extents_before_m": original.extents.tolist(), "extents_after_m": scaled.extents.tolist(),
        "limitations": ["Original solver with an explicit monocular relative-view adapter, not the provided-mesh release test",
            "Depth and poses are estimates; held-out image fit does not establish true physical dimensions"]}
    receipt = output / "mesh-calibration.json"
    receipt.write_text(json.dumps(record, indent=2))
    if not accepted:
        raise ValueError("Scale fit did not improve both fitting and held-out observations; candidate preserved but not accepted")
    for item in (prepared["masks"], prepared["segmentation"]["report"], prepared["mesh_generation"]["receipt"]):
        item["path"] = artifact((a.preparation.parent/item["path"]).resolve())["path"]
    prepared.update(mesh=artifact(mesh_output), mesh_calibration={"receipt":artifact(receipt)},
        limitations=prepared.get("limitations", [])+record["limitations"])
    verify_budgeted_preparation(prepared, output, generation["mesh_sha256"], allowed=False)
    (output / "preparation.json").write_text(json.dumps(prepared, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
