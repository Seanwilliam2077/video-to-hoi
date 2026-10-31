"""Native MHR Human results for V2D Track 1 (not complete HOI submissions).

The schema and coordinate conversion follow the public Track 1 submission kit's
``tools/track1/mesh_to_mhr_params.py``. No challenge reference data is required.
Torch is optional until fitting or model evaluation is requested.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

NUM_VERTICES = 18439
NUM_POSE = 136
NUM_SCALES = 68
NUM_SHAPE = 45
MHR_TO_TRACK1 = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array


def native_vertices_to_track1(vertices_cm: np.ndarray) -> np.ndarray:
    """MHR native centimetres, y-up/z-back -> Track 1 metres, y-down/z-forward."""
    vertices = _finite_array(vertices_cm, "vertices_cm")
    if vertices.ndim < 2 or vertices.shape[-1] != 3:
        raise ValueError("vertices_cm must end in a 3D coordinate axis")
    return vertices * MHR_TO_TRACK1 / 100.0


def track1_translation_to_native(translation_m: np.ndarray) -> np.ndarray:
    """Convert a Track 1 translation *offset* to MHR pose[0:3] units.

    This converts an offset, not an absolute root joint position: the neutral
    body's root is not necessarily at the coordinate origin. The public
    converter establishes 1 native translation parameter unit = 100 mm.
    """
    translation = _finite_array(translation_m, "translation_m")
    if translation.ndim < 1 or translation.shape[-1] != 3:
        raise ValueError("translation_m must end in a 3D coordinate axis")
    return translation * MHR_TO_TRACK1 * 10.0


def sam_prediction_to_native(prediction: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Place a single SAM 3D Body prediction in its camera coordinate frame.

    The NVIDIA SAM fork exposes the assembled 204 native parameters after hand
    refinement, with root translation zero. ``pred_cam_t`` is a separate offset
    in metres; ``pred_vertices`` is already axis-flipped and in metres. Apply
    that camera offset once to both representations. Torch tensors must first
    be detached and copied to CPU by the inference adapter.
    """
    params = _finite_array(prediction["mhr_model_params"], "mhr_model_params")
    # The estimator names this ``shape_params``; the underlying head uses ``shape``.
    shape_key = "shape_params" if "shape_params" in prediction else "shape"
    shape = _finite_array(prediction[shape_key], shape_key)
    translation = _finite_array(prediction["pred_cam_t"], "pred_cam_t")
    vertices = _finite_array(prediction["pred_vertices"], "pred_vertices")
    for name, array, expected in (
        ("mhr_model_params", params, (NUM_POSE + NUM_SCALES,)),
        ("shape", shape, (NUM_SHAPE,)), ("pred_cam_t", translation, (3,)),
        ("pred_vertices", vertices, (NUM_VERTICES, 3)),
    ):
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    placed = np.array(params, dtype=np.float32, copy=True)
    placed[:3] += track1_translation_to_native(translation)
    return {
        "pose": placed[:NUM_POSE],
        "scales": placed[NUM_POSE:],
        "shape": np.array(shape, dtype=np.float32, copy=True),
        "vertices_m": np.asarray(vertices + translation, dtype=np.float32),
    }


def validate_human_result(
    result: Mapping[str, Any], expected_frames: int | None = None,
) -> dict[str, Any]:
    """Reject malformed native parameters; return a small structural summary.

    This is a structural check, not a score or an assertion of reconstruction
    accuracy. Identity and skeleton scales must be stored once for the clip.
    """
    required = {"pose", "scales", "shape"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"missing Human arrays: {', '.join(sorted(missing))}")
    pose = _finite_array(result["pose"], "pose")
    if pose.ndim != 2 or pose.shape[1] != NUM_POSE or len(pose) == 0:
        raise ValueError(f"pose must have shape [T>0, {NUM_POSE}], got {pose.shape}")
    frames = len(pose)
    if expected_frames is not None and frames != expected_frames:
        raise ValueError(f"pose has {frames} frames, expected {expected_frames}")
    for name, count in (("scales", NUM_SCALES), ("shape", NUM_SHAPE)):
        array = _finite_array(result[name], name)
        if array.shape != (count,):
            raise ValueError(f"{name} must have one shared shape [{count}], got {array.shape}")
    if "frame_indices" in result:
        indices = _finite_array(result["frame_indices"], "frame_indices")
        if indices.shape != (frames,) or indices.dtype.kind not in "iu":
            raise ValueError("frame_indices must be an integer array with one index per pose")
        if np.any(indices < 0) or np.any(indices[1:] <= indices[:-1]):
            raise ValueError("frame_indices must be nonnegative and strictly increasing")
    if "fps" in result:
        fps = _finite_array(result["fps"], "fps")
        if fps.shape != () or float(fps) <= 0:
            raise ValueError("fps must be a positive scalar")
    for name in ("per_frame_vertex_error_mm", "before_refit_vertex_error_mm"):
        if name in result:
            error = _finite_array(result[name], name)
            if error.shape != (frames,) or np.any(error < 0):
                raise ValueError(f"{name} must be nonnegative with shape [{frames}]")
    return {
        "frames": frames,
        "pose_shape": list(pose.shape),
        "scales_shape": [NUM_SCALES],
        "shape_shape": [NUM_SHAPE],
        "identity_and_scales_shared": True,
        "artifact_scope": "human_only_not_complete_submission",
    }


def write_human_result(
    path: str | Path, result: Mapping[str, Any], report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save non-pickled arrays and an explicit Human-only JSON report.

    Cast and validate again before writing, so float32 overflow cannot sneak
    into a file that passed the input check.
    """
    summary = validate_human_result(result)
    arrays = {name: np.asarray(result[name], dtype=np.float32) for name in ("pose", "scales", "shape")}
    for name in ("per_frame_vertex_error_mm", "before_refit_vertex_error_mm", "fps"):
        if name in result:
            arrays[name] = np.asarray(result[name], dtype=np.float32)
    if "frame_indices" in result:
        arrays["frame_indices"] = np.asarray(result["frame_indices"], dtype=np.int64)
    validate_human_result(arrays)
    metadata = dict(report if report is not None else result.get("report", {}))
    metadata.update(summary)
    metadata.update({
        "parameter_format": "native_mhr_pose136_scales68_shape45",
        "root_translation_unit_m": 0.1,
        "vertices_frame": "Track 1: x-right, y-down, z-forward, metres",
        "model_expression": "zero",
        "model_pose_correctives": True,
        "official_scores_available": False,
    })
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **arrays, report=json.dumps(metadata, allow_nan=False, sort_keys=True))
    return metadata


def load_human_result(path: str | Path) -> dict[str, Any]:
    """Load only ordinary arrays, checking the schema before returning them."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    validate_human_result(result)
    if "report" in result:
        result["report"] = json.loads(str(result["report"]))
    return result


def native_mhr_forward(
    result: Mapping[str, Any], model_path: str | Path, *,
    device: str = "cpu", batch_size: int = 1,
) -> dict[str, np.ndarray]:
    """Evaluate the exported native Human arrays in the Track 1 metric frame.

    Returns ``vertices_m[T,18439,3]`` and all ``joints_m[T,127,3]`` in metres,
    with zero face expression and pose correctives enabled. These are model
    predictions, not reference data. Parameters are evaluated as float32, just
    as they are saved. Batch one bounds memory and also works on CPU.
    """
    import torch

    validate_human_result(result)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    model = torch.jit.load(str(model_path), map_location=device).float().eval()
    pose = np.asarray(result["pose"], dtype=np.float32)
    shape = torch.as_tensor(np.array(result["shape"], dtype=np.float32, copy=True), device=device)
    scales = torch.as_tensor(np.array(result["scales"], dtype=np.float32, copy=True), device=device)
    vertices, joints = [], []
    with torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for start in range(0, len(pose), batch_size):
            frame_pose = torch.as_tensor(np.array(pose[start:start + batch_size], copy=True, order="C"), device=device)
            count = len(frame_pose)
            params = torch.cat([frame_pose, scales[None].expand(count, -1)], dim=1)
            expression = torch.zeros(count, 72, dtype=torch.float32, device=device)
            v, skeleton = model(shape[None].expand(count, -1).contiguous(), params, expression, True)
            vertices.append(native_vertices_to_track1(v.cpu().numpy()))
            joints.append(native_vertices_to_track1(skeleton[..., :3].cpu().numpy()))
    output = {"vertices_m": np.concatenate(vertices), "joints_m": np.concatenate(joints)}
    if output["vertices_m"].shape != (len(pose), NUM_VERTICES, 3):
        raise ValueError("model did not produce the expected MHR lod1 vertex topology")
    if output["joints_m"].shape != (len(pose), 127, 3):
        raise ValueError("model did not produce the expected 127 MHR joints")
    return output


def _load_converter(path: str | Path):
    """Load the user's saved official Track 1 source, not any scorer or data."""
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location("v2hoi_track1_mesh_converter", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import Track 1 converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity_sequence(value: np.ndarray, frames: int, count: int, name: str) -> np.ndarray:
    array = _finite_array(value, name)
    if array.shape == (count,):
        return np.broadcast_to(array, (frames, count)).copy()
    if array.shape != (frames, count):
        raise ValueError(f"{name} must have shape [{count}] or [{frames}, {count}], got {array.shape}")
    return array


def _log_progress(message: str) -> None:
    print(message, flush=True)


def _refit_pose_frames(
    kit, mhr, target, pose, z, *, pose_iterations: int, pose_workers: int, log,
):
    """Refit independent frames against frozen identity, preserving frame order.

    The reviewed MHR forward reads shared model tensors and allocates its own
    scratch tensors. LM clones each initial pose. Neither target, pose nor z is
    modified while workers run. Torch execution contexts are thread local, so
    every worker establishes them explicitly.
    """
    from concurrent.futures import ThreadPoolExecutor
    import torch

    frames = len(pose)

    def refit(frame):
        with torch.no_grad(), torch.autocast(device_type=mhr.device.type, enabled=False):
            fitted, _ = kit.lm_pose(
                mhr, target[frame:frame + 1], pose[frame:frame + 1], z,
                iters=pose_iterations, frames_per_batch=1,
            )
        if not torch.isfinite(fitted).all():
            raise ValueError(f"pose refit produced non-finite parameters at frame {frame}")
        return fitted

    def collect(results):
        out_pose = []
        for count, fitted in enumerate(results, start=1):
            out_pose.append(fitted)
            if count % 10 == 0 or count == frames:
                log(f"Refitted pose {count}/{frames}")
        return out_pose

    if pose_workers == 1:
        return collect(map(refit, range(frames)))
    # map yields in input order even when later frames finish first.
    with ThreadPoolExecutor(max_workers=pose_workers) as pool:
        return collect(pool.map(refit, range(frames)))


def fit_shared_identity(
    vertices_m: np.ndarray,
    initial_pose: np.ndarray,
    initial_scales: np.ndarray,
    initial_shape: np.ndarray,
    model_path: str | Path,
    converter_path: str | Path,
    *,
    device: str = "cuda",
    model_batch: int = 8,
    keyframes: int = 5,
    pose_iterations: int = 8,
    identity_iterations: int = 8,
    pose_workers: int = 1,
    return_vertices: bool = False,
    log=_log_progress,
) -> dict[str, Any]:
    """Fit one native identity and skeleton scale, then refit each real pose.

    Input meshes are the *predicted* MHR lod1 meshes in Track 1 metres, never
    reference answers. The initial pose must already be in native MHR units
    and orientation. The SAM split body/hand parameters cannot be concatenated
    into this pose without applying SAM's MHR parameter mapping.

    The public kit's finite-difference LM solver runs the model in float32 and
    solver arithmetic in float64. One frame per Jacobian and small model
    batches bound memory on a 10 GB GPU. A median is only the starting identity:
    a shared-identity fit on evenly spaced frames and independent pose refits
    follow it. No temporal smoothing or missing-frame copying is performed.
    Diagnostics measure change to input predictions, not official accuracy.
    Optional CPU workers parallelize independent pose solves only; they require
    Torch intra-op threads set to one before calling this function. The shared
    identity solve remains sequential, with identical initializers and settings.
    """
    import time
    import torch

    started = time.monotonic()
    vertices = _finite_array(vertices_m, "vertices_m")
    if vertices.ndim != 3 or vertices.shape[1:] != (NUM_VERTICES, 3) or len(vertices) == 0:
        raise ValueError(f"vertices_m must have shape [T>0, {NUM_VERTICES}, 3]")
    frames = len(vertices)
    pose_array = _finite_array(initial_pose, "initial_pose")
    if pose_array.shape != (frames, NUM_POSE):
        raise ValueError(f"initial_pose must have shape [{frames}, {NUM_POSE}]")
    scales = _identity_sequence(initial_scales, frames, NUM_SCALES, "initial_scales")
    shape = _identity_sequence(initial_shape, frames, NUM_SHAPE, "initial_shape")
    for name, value in (("model_batch", model_batch), ("keyframes", keyframes),
                        ("pose_iterations", pose_iterations), ("identity_iterations", identity_iterations),
                        ("pose_workers", pose_workers)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if pose_workers > 1:
        if torch.device(device).type != "cpu":
            raise ValueError("parallel pose refits are supported only on CPU")
        if torch.get_num_threads() != 1:
            raise ValueError("parallel pose refits require torch.set_num_threads(1) before fitting")
    kit = _load_converter(converter_path)
    mhr = kit.MHR(str(model_path), device, chunk=model_batch, precision="float32")
    td = {"dtype": mhr.dtype, "device": mhr.device}
    target = torch.as_tensor(np.ascontiguousarray(vertices * 1000.0), **td).reshape(frames, -1)
    pose = torch.as_tensor(np.array(pose_array, copy=True, order="C"), **td)
    identities = np.concatenate([scales, shape], axis=1)
    z_original = torch.as_tensor(np.ascontiguousarray(identities), **td)
    z_initial = torch.as_tensor(np.median(identities, axis=0)[None], **td)

    def residuals(p, z, target_values):
        with torch.no_grad(), torch.autocast(device_type=mhr.device.type, enabled=False):
            return (mhr.verts(p, z) - target_values).reshape(len(p), NUM_VERTICES, 3)

    original_error = residuals(pose, z_original, target).norm(dim=-1).mean(1).cpu().numpy()
    baseline_residual = residuals(pose, z_initial, target)
    baseline_error = baseline_residual.norm(dim=-1).mean(1).cpu().numpy()
    del baseline_residual, z_original
    indices = np.unique(np.linspace(0, frames - 1, min(keyframes, frames)).round().astype(int))
    key = indices.tolist()
    log(f"Fitting one identity and skeleton scale on frames {key}")
    with torch.no_grad(), torch.autocast(device_type=mhr.device.type, enabled=False):
        fitted_key_pose, z = kit.lm_joint(
            mhr, target[key], pose[key], z_initial, iters=identity_iterations, frames_per_chunk=1,
        )
    # A numerical solver failure must not silently replace the usable initializer.
    if not torch.isfinite(z).all() or not torch.isfinite(fitted_key_pose).all():
        raise ValueError("shared-identity fit produced non-finite parameters")
    key_before = residuals(pose[key], z_initial, target[key]).square().sum()
    key_after = residuals(fitted_key_pose, z, target[key]).square().sum()
    accepted_identity = bool(key_after <= key_before)
    if accepted_identity:
        pose = pose.clone()
        pose[key] = fitted_key_pose
    else:
        z = z_initial
    out_pose = _refit_pose_frames(
        kit, mhr, target, pose, z, pose_iterations=pose_iterations,
        pose_workers=pose_workers, log=log,
    )
    # Evaluate the actual float32 arrays that will be exported, not solver state.
    pose32 = torch.cat(out_pose).float().double()
    z32 = z.float().double()
    with torch.no_grad(), torch.autocast(device_type=mhr.device.type, enabled=False):
        fitted_mm = mhr.verts(pose32, z32).reshape(frames, NUM_VERTICES, 3)
    error = (fitted_mm - target.reshape(frames, NUM_VERTICES, 3)).norm(dim=-1)
    frame_error = error.mean(1).cpu().numpy().astype(np.float32)
    report = {
        "method": "shared_identity_keyframe_LM_then_per_frame_pose_LM",
        "identity_keyframes": key,
        "shared_identity_step_accepted": accepted_identity,
        "initial_native_reconstruction_error_mm": {
            "mean": float(original_error.mean()), "max_frame_mean": float(original_error.max()),
        },
        "fixed_median_before_refit_error_mm": {
            "mean": float(baseline_error.mean()), "max_frame_mean": float(baseline_error.max()),
        },
        "vertex_error_mm": {
            "mean": float(frame_error.mean()), "max_frame_mean": float(frame_error.max()),
            "worst_frame_position": int(frame_error.argmax()),
        },
        "error_interpretation": "Distance to input predictions; not error against reference or an official score",
        "device": device, "model_precision": "float32", "solver_precision": "float64",
        "model_batch": model_batch, "pose_iterations": pose_iterations,
        "pose_workers": pose_workers,
        "identity_iterations": identity_iterations, "seconds": round(time.monotonic() - started, 3),
    }
    result = {
        "pose": pose32.cpu().numpy().astype(np.float32),
        "scales": z32[0, :NUM_SCALES].cpu().numpy().astype(np.float32),
        "shape": z32[0, NUM_SCALES:].cpu().numpy().astype(np.float32),
        "per_frame_vertex_error_mm": frame_error,
        "before_refit_vertex_error_mm": baseline_error.astype(np.float32),
        "report": report,
    }
    validate_human_result(result, expected_frames=frames)
    if return_vertices:
        result["vertices_m"] = fitted_mm.cpu().numpy().astype(np.float32) / 1000.0
    return result
