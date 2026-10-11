"""Convert the official CARI4D wrapper through its real MHR head to native 204.

This module never concatenates the team's legacy v1 human arrays. It verifies
geometry against the original decoder before serializing native parameters.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def fit_shared_identity(native, shapes, model_path, converter_path, *, log=print):
    """Warm-start the official LM solvers from decoded backend parameters.

    This is a measured, lossy fit when the backend changes identity over time;
    the exact wrapper-to-native conversion and this fit have separate reports.
    """
    import importlib.util
    import time
    import torch
    spec = importlib.util.spec_from_file_location("official_mhr_converter", converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    start = time.monotonic()
    mhr = converter.MHR(str(model_path), "cuda", chunk=16, precision="float32")
    pose = torch.as_tensor(native[:, :136], device="cuda", dtype=torch.float64)
    identity = torch.as_tensor(np.concatenate((native[:, 136:], shapes), axis=1), device="cuda", dtype=torch.float64)
    if not torch.isfinite(pose).all() or not torch.isfinite(identity).all():
        raise ValueError("Nonfinite source geometry cannot be exported")
    target = mhr.verts(pose, identity)
    shared = identity.median(dim=0, keepdim=True).values
    before = (mhr.verts(pose, shared) - target).reshape(len(pose), -1, 3).norm(dim=-1)
    keys = np.unique(np.linspace(0, len(pose) - 1, min(4, len(pose))).round().astype(int)).tolist()
    log(f"Fitting shared identity on original frames {keys}; initial mean vertex error {before.mean().item():.3f} mm", flush=True)
    key_pose, shared = converter.lm_joint(mhr, target[keys], pose[keys], shared, iters=15, prior=0)
    fitted = pose.clone()
    fitted[keys] = key_pose
    for begin in range(0, len(pose), 4):
        end = min(begin + 4, len(pose))
        fitted[begin:end], _ = converter.lm_pose(mhr, target[begin:end], fitted[begin:end], shared, iters=15)
        if begin % 24 == 0:
            log(f"Shared identity pose fit: {end}/{len(pose)}", flush=True)
    # Verify the actual serialized float32 values using the official forward path.
    fitted = fitted.float().double()
    shared = shared.float().double()
    after = (mhr.verts(fitted, shared) - target).reshape(len(pose), -1, 3).norm(dim=-1).cpu().numpy()
    report = {"method": "Official LM joint identity + per-frame pose, warm-started from exact native parameters",
        "keyframes": keys, "frames": len(pose), "before_mean_vertex_error_mm": float(before.mean()),
        "mean_vertex_error_mm": float(after.mean()), "p95_vertex_error_mm": float(np.quantile(after, .95)),
        "max_vertex_error_mm": float(after.max()), "worst_frame_mean_mm": float(after.mean(1).max()),
        "per_frame_mean_mm": after.mean(1).tolist(), "seconds": time.monotonic() - start,
        "note": "Fit error relative to the backend reconstruction, not a challenge ground-truth score"}
    return fitted.cpu().numpy().astype(np.float32), shared[0, :68].cpu().numpy().astype(np.float32), shared[0, 68:].cpu().numpy().astype(np.float32), report


def encode_native(layer, params, batch_size=8):
    import torch
    native, shapes, faces, errors = [], [], [], []
    n = len(params["mhr_trans"])
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            chunk = {k: v[start:start + batch_size] for k, v in params.items()}
            backend = layer.backend
            context = backend._vertices_context(chunk, detach_fixed=False)
            trans, body, shape, scale = backend._mutable_vertices_inputs(chunk, context)
            verts, model_params = context.head.mhr_forward(
                global_trans=trans * context.flip, global_rot=context.global_rot,
                body_pose_params=body, hand_pose_params=context.hand, scale_params=scale,
                shape_params=shape, expr_params=context.face, return_keypoints=False,
                return_joint_coords=False, return_model_params=True, return_joint_rotations=False,
            )
            original = layer.mhr_forward_vertices(chunk)
            # The native model emits centimetres; the head already divided by 100.
            error = (verts * context.flip - original).norm(dim=-1)
            errors.append(error.cpu().numpy())
            native.append(model_params.cpu().numpy())
            shapes.append(shape.cpu().numpy())
            faces.append(context.face.cpu().numpy())
    error = np.concatenate(errors)
    if error.max() > 1e-4:
        raise ValueError(f"Native MHR conversion changed geometry by {error.max():.6g} m")
    return np.concatenate(native), np.concatenate(shapes), np.concatenate(faces), {
        "max_vertex_error_m": float(error.max()), "mean_vertex_error_m": float(error.mean()),
        "tolerance_m": 1e-4, "frames": n,
        "mapping": "MHRHead.mhr_forward return_model_params, native translation 10 units/metre, CV axis flip",
        "decoder": layer.decoder_identity(),
    }


def export_run(run: Path, assets: Path, output: Path, kit: Path):
    import torch
    import trimesh
    from lib_mhr import MHRLayer, MHR_PARAM_DIMS
    from lib_mhr.object_pose_frame import resolve_object_pose_frame
    from v2hoi.native import sha256
    identity = json.loads((run / "native-inputs.json").read_text())
    status = json.loads((run / "native-status.json").read_text())
    if identity.get("provenance") != "track1" or not status.get("complete"):
        raise ValueError("Only a completed real Track 1 run can be exported")
    sequence = f"episode_{identity['episode']:06d}"
    native_root = run / "native" / sequence
    bundle_path = native_root / "inference/refined.pth"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    frame_ids = [int(row["src_frame"]) for row in bundle["frame_meta"]]
    if frame_ids != identity["original_frame_ids"] or len(frame_ids) != identity["expected_frames"]:
        raise ValueError("Native output does not cover every original video frame")
    if bundle["metadata"].get("ground_truth_used") is not False or bundle.get("gt"):
        raise ValueError("Unexpected reference data in real reconstruction")
    params = {k: np.asarray(bundle["pr"][k], dtype=np.float32) for k in MHR_PARAM_DIMS}
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=assets, device="cuda")
    native, shapes, face, mapping = encode_native(layer, params)
    if abs(face).max() > 1e-7:
        raise ValueError("Official format has no expression field; nonzero expressions require a measured fit")
    output.mkdir(parents=True, exist_ok=True)
    converter = kit / "tools/track1/mesh_to_mhr_params.py"
    fit_identity = {"bundle_sha256": sha256(bundle_path), "converter_sha256": sha256(converter),
        "exporter_sha256": sha256(Path(__file__)),
        "model_sha256": sha256(assets / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")}
    fit_path = output / f"{sequence}_human_fit.npz"
    fit_report_path = output / f"{sequence}_human_fit.json"
    if fit_path.exists() and fit_report_path.exists():
        fit_report = json.loads(fit_report_path.read_text())
        if fit_report["inputs"] != fit_identity:
            raise ValueError("Cached identity fit has different inputs")
        with np.load(fit_path, allow_pickle=False) as cached:
            pose, scales, shape = cached["pose"], cached["scales"], cached["shape"]
    else:
        del layer
        torch.cuda.empty_cache()
        pose, scales, shape, fit_report = fit_shared_identity(
            native, shapes, assets / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt", converter)
        fit_report["inputs"] = fit_identity
        np.savez(fit_path, pose=pose, scales=scales, shape=shape)
        fit_report_path.write_text(json.dumps(fit_report, indent=2))
    poses = np.asarray(bundle["pr"]["pose_abs"], dtype=np.float32)
    if poses.shape != (len(frame_ids), 4, 4) or not np.isfinite(poses).all():
        raise ValueError("Invalid object trajectory")
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0,2,1), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError("Object poses are not proper rigid transforms")
    mesh_source = Path(bundle["metadata"]["object_mesh"])
    mesh = trimesh.load(mesh_source, force="mesh", process=False)
    frame = resolve_object_pose_frame(bundle["metadata"])
    mesh.apply_transform(frame.mesh_to_training)
    mesh_path = output / f"{sequence}_object.glb"
    mesh.export(mesh_path)
    readback = trimesh.load(mesh_path, force="mesh", process=False)
    if mesh.vertices.shape != readback.vertices.shape or not np.allclose(mesh.vertices, readback.vertices, atol=1e-6):
        raise ValueError("Object mesh changed during export")
    np.savez(output / f"{sequence}.npz", pose=pose, scales=scales, shape=shape,
        object_rotation=rotations, object_translation=poses[:, :3, 3], object_scale=np.float32(1),
        frame_ids=np.asarray(frame_ids, dtype=np.int64))
    report = {"schema": "v2hoi.native-export.v1", "provenance": "track1", "episode": identity["episode"],
        "frames": len(frame_ids), "mapping": mapping, "shared_identity_fit": fit_report,
        "mesh_sha256": sha256(mesh_path), "object_scale": 1,
        "object_mesh_transform": frame.mesh_to_training.tolist(),
        "complete_submission": False, "scope": "One complete video; official format validation still required"}
    (output / "export-report.json").write_text(json.dumps(report, indent=2))
    print(f"Exported {len(frame_ids)} native MHR and object poses to {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--initialization", type=Path)
    group.add_argument("--run", type=Path)
    p.add_argument("--kit", type=Path, default=Path("third_party/official-track1-kit"))
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.run:
        export_run(a.run, a.assets, a.output, a.kit)
        return
    from lib_mhr import MHRLayer, load_mhr_result, MHR_PARAM_DIMS
    source = load_mhr_result(a.initialization)
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=a.assets, device="cuda")
    native, shape, face, report = encode_native(layer, {k: np.asarray(source[k]) for k in MHR_PARAM_DIMS})
    report.update({"native_shape": list(native.shape), "shape_max_temporal_range": float(np.ptp(shape, axis=0).max()),
        "scale_max_temporal_range": float(np.ptp(native[:,136:], axis=0).max()), "face_max_abs": float(abs(face).max())})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
