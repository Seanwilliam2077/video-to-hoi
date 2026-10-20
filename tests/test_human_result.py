"""Independent synthetic native-MHR fixtures, with no challenge data or models."""
import json

import numpy as np
import pytest

from v2hoi.human_result import (
    load_human_result,
    native_vertices_to_track1,
    sam_prediction_to_native,
    track1_translation_to_native,
    validate_human_result,
    write_human_result,
)


def synthetic_result(frames=4):
    rng = np.random.default_rng(8301)
    return {
        "pose": rng.normal(size=(frames, 136)).astype(np.float32),
        "scales": rng.normal(size=68).astype(np.float32),
        "shape": rng.normal(size=45).astype(np.float32),
        "frame_indices": np.arange(frames, dtype=np.int64) * 3,
        "fps": np.array(30.0),
    }


def test_human_only_roundtrip_preserves_native_poses_and_shared_identity(tmp_path):
    original = synthetic_result()
    report = write_human_result(tmp_path / "human.npz", original, {"input_sha256": "synthetic"})
    restored = load_human_result(tmp_path / "human.npz")
    for key, value in original.items():
        np.testing.assert_array_equal(restored[key], value)
    assert report["artifact_scope"] == "human_only_not_complete_submission"
    assert report["identity_and_scales_shared"] is True
    assert report["root_translation_unit_m"] == 0.1
    assert report["official_scores_available"] is False
    assert restored["report"]["input_sha256"] == "synthetic"
    # This must remain loadable without enabling pickle for metadata.
    with np.load(tmp_path / "human.npz", allow_pickle=False) as archive:
        assert json.loads(str(archive["report"]))["frames"] == 4


@pytest.mark.parametrize("key,width", [("scales", 68), ("shape", 45)])
def test_rejects_identity_stored_per_frame_even_when_constant(key, width):
    result = synthetic_result()
    result[key] = np.zeros((4, width), dtype=np.float32)
    with pytest.raises(ValueError, match="one shared"):
        validate_human_result(result)


@pytest.mark.parametrize("key,width", [("pose", 133), ("scales", 28), ("shape", 10)])
def test_rejects_legacy_or_split_parameter_sizes(key, width):
    result = synthetic_result()
    result[key] = np.zeros((4, width) if key == "pose" else width)
    with pytest.raises(ValueError, match="shape"):
        validate_human_result(result)


@pytest.mark.parametrize("key", ["pose", "scales", "shape"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_parameters_fail(key, invalid):
    result = synthetic_result()
    result[key].flat[0] = invalid
    with pytest.raises(ValueError, match="finite"):
        validate_human_result(result)


@pytest.mark.parametrize("indices", [[0, 1, 1, 3], [0, 3, 2, 4], [-1, 0, 1, 2], [0, 1]])
def test_frame_indices_cannot_silently_change_video_correspondence(indices):
    result = synthetic_result()
    result["frame_indices"] = np.asarray(indices, dtype=np.int64)
    with pytest.raises(ValueError, match="frame_indices"):
        validate_human_result(result)


def test_checks_expected_video_frame_count_and_rejects_empty_result():
    result = synthetic_result()
    with pytest.raises(ValueError, match="expected 5"):
        validate_human_result(result, expected_frames=5)
    result["pose"] = np.zeros((0, 136))
    with pytest.raises(ValueError, match="T>0"):
        validate_human_result(result)


def test_optional_residual_must_cover_every_frame_and_be_nonnegative():
    result = synthetic_result()
    result["per_frame_vertex_error_mm"] = np.array([0.0, 1.0, 2.0, -1.0])
    with pytest.raises(ValueError, match="nonnegative"):
        validate_human_result(result)
    result["per_frame_vertex_error_mm"] = np.zeros(3)
    with pytest.raises(ValueError, match="shape"):
        validate_human_result(result)


def test_native_vertices_use_centimetres_and_track1_axis_flip():
    native_cm = np.array([[100.0, 200.0, -300.0], [-5.0, 0.0, 40.0]])
    result = native_vertices_to_track1(native_cm)
    np.testing.assert_allclose(result, [[1.0, -2.0, 3.0], [-0.05, 0.0, -0.4]])
    np.testing.assert_allclose(native_cm[0], [100.0, 200.0, -300.0])


def test_native_root_translation_has_different_units_from_native_vertices():
    # The official converter's translation parameter is 100 mm, not 10 mm.
    offsets_m = np.array([[0.1, 0.2, 0.3], [-1.0, 0.0, -0.1]])
    np.testing.assert_allclose(track1_translation_to_native(offsets_m), [[1, -2, -3], [-10, 0, 1]])


def test_sam_camera_translation_is_applied_once_without_changing_pose_or_identity():
    rng = np.random.default_rng(701)
    params = rng.normal(size=204).astype(np.float32)
    params[:3] = 0.0
    shape = rng.normal(size=45).astype(np.float32)
    vertices = rng.normal(size=(18439, 3)).astype(np.float32)
    prediction = {
        "mhr_model_params": params, "shape_params": shape,
        "pred_cam_t": np.array([0.5, -1.0, 3.0]), "pred_vertices": vertices,
    }
    mapped = sam_prediction_to_native(prediction)
    np.testing.assert_allclose(mapped["pose"][:3], [5.0, 10.0, -30.0])
    np.testing.assert_array_equal(mapped["pose"][3:], params[3:136])
    np.testing.assert_array_equal(mapped["scales"], params[136:])
    np.testing.assert_array_equal(mapped["shape"], shape)
    np.testing.assert_allclose(mapped["vertices_m"], vertices + prediction["pred_cam_t"], atol=5e-7)
    np.testing.assert_array_equal(params[:3], np.zeros(3))


def test_write_rejects_float32_overflow_before_creating_artifact(tmp_path):
    result = synthetic_result()
    result["pose"] = result["pose"].astype(np.float64)
    result["pose"][0, 0] = 1e300
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        write_human_result(tmp_path / "invalid.npz", result)
    assert not (tmp_path / "invalid.npz").exists()


def test_missing_arrays_and_nonnumeric_parameters_are_rejected():
    result = synthetic_result()
    del result["shape"]
    with pytest.raises(ValueError, match="missing Human arrays"):
        validate_human_result(result)
    result = synthetic_result()
    result["shape"] = np.array(["0"] * 45)
    with pytest.raises(ValueError, match="finite real numbers"):
        validate_human_result(result)


@pytest.fixture
def synthetic_torch_contexts(monkeypatch):
    """Exercise worker contexts without installing Torch or loading any model."""
    from contextlib import contextmanager
    import sys
    import threading
    from types import SimpleNamespace

    state = threading.local()

    @contextmanager
    def no_grad():
        previous = getattr(state, "no_grad", False)
        state.no_grad = True
        try:
            yield
        finally:
            state.no_grad = previous

    @contextmanager
    def autocast(*, device_type, enabled):
        assert device_type == "cpu"
        assert enabled is False
        previous = getattr(state, "autocast_disabled", False)
        state.autocast_disabled = True
        try:
            yield
        finally:
            state.autocast_disabled = previous

    stub = SimpleNamespace(
        no_grad=no_grad, autocast=autocast, isfinite=np.isfinite,
        device=lambda name: SimpleNamespace(type=str(name).split(":")[0]),
        get_num_threads=lambda: 1,
    )
    monkeypatch.setitem(sys.modules, "torch", stub)
    return state, stub


def test_parallel_refits_overlap_preserve_order_and_match_serial(synthetic_torch_contexts):
    import threading
    from types import SimpleNamespace
    from v2hoi.human_result import _refit_pose_frames

    state, _ = synthetic_torch_contexts
    rng = np.random.default_rng(6512)
    pose = rng.normal(size=(5, 136))
    target = np.column_stack([np.arange(5), rng.normal(size=(5, 2))])
    z = rng.normal(size=(1, 113))
    original = [array.copy() for array in (pose, target, z)]
    mhr = SimpleNamespace(device=SimpleNamespace(type="cpu"))

    def run(workers):
        lock = threading.Lock()
        barrier = threading.Barrier(2)
        later_finished = threading.Event()
        stats = {"active": 0, "peak_active": 0, "thread_ids": set(), "completed": []}

        def lm_pose(model, frame_target, frame_pose, identity, *, iters, frames_per_batch):
            assert model is mhr
            assert identity is z
            assert iters == 2 and frames_per_batch == 1
            assert state.no_grad and state.autocast_disabled
            frame = int(frame_target[0, 0])
            with lock:
                stats["active"] += 1
                stats["peak_active"] = max(stats["peak_active"], stats["active"])
                stats["thread_ids"].add(threading.get_ident())
            if workers > 1 and frame in (0, 1):
                barrier.wait(timeout=5)
                if frame == 0:
                    assert later_finished.wait(timeout=5)
            fitted = frame_pose.copy() + frame_target.sum() + identity.sum()
            with lock:
                stats["completed"].append(frame)
                stats["active"] -= 1
            if workers > 1 and frame == 1:
                later_finished.set()
            return fitted, np.zeros(1)

        logs = []
        output = _refit_pose_frames(
            SimpleNamespace(lm_pose=lm_pose), mhr, target, pose, z,
            pose_iterations=2, pose_workers=workers, log=logs.append,
        )
        assert logs == ["Refitted pose 5/5"]
        return np.concatenate(output), stats

    serial, serial_stats = run(1)
    parallel, parallel_stats = run(2)
    np.testing.assert_array_equal(parallel, serial)
    assert serial_stats["completed"] == list(range(5))
    assert parallel_stats["completed"].index(1) < parallel_stats["completed"].index(0)
    assert parallel_stats["peak_active"] >= 2
    assert len(parallel_stats["thread_ids"]) == 2
    for actual, expected in zip((pose, target, z), original):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("failure", ["nonfinite", "solver_exception"])
def test_refit_worker_failure_cannot_export_a_partial_or_repeated_pose(
    synthetic_torch_contexts, workers, failure,
):
    from types import SimpleNamespace
    from v2hoi.human_result import _refit_pose_frames

    pose = np.arange(3 * 136, dtype=np.float64).reshape(3, 136)

    def lm_pose(model, target, initial, z, **kwargs):
        if target[0, 0] == 1:
            if failure == "solver_exception":
                raise RuntimeError("Synthetic solver failure")
            return np.full_like(initial, np.nan), np.zeros(1)
        return initial.copy(), np.zeros(1)

    exception = RuntimeError if failure == "solver_exception" else ValueError
    message = "Synthetic solver failure" if failure == "solver_exception" else "non-finite parameters at frame 1"
    with pytest.raises(exception, match=message):
        _refit_pose_frames(
            SimpleNamespace(lm_pose=lm_pose), SimpleNamespace(device=SimpleNamespace(type="cpu")),
            np.arange(3)[:, None], pose, np.zeros((1, 113)),
            pose_iterations=2, pose_workers=workers, log=lambda message: None,
        )


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"])
def test_rejects_invalid_pose_worker_count_before_loading_model(synthetic_torch_contexts, workers, monkeypatch):
    import v2hoi.human_result as human

    monkeypatch.setattr(human, "_load_converter", lambda path: pytest.fail("Must reject configuration before loading"))
    with pytest.raises(ValueError, match="pose_workers must be a positive integer"):
        human.fit_shared_identity(
            np.zeros((1, 18439, 3)), np.zeros((1, 136)), np.zeros(68), np.zeros(45),
            "unused-model", "unused-converter", device="cpu", pose_workers=workers,
        )


@pytest.mark.parametrize("device,threads,message", [
    ("cuda", 1, "supported only on CPU"),
    ("cuda:1", 1, "supported only on CPU"),
    ("cpu", 2, "set_num_threads"),
])
def test_parallel_configuration_rejects_gpu_and_nested_intraop_threads(
    synthetic_torch_contexts, device, threads, message, monkeypatch,
):
    import v2hoi.human_result as human

    _, stub = synthetic_torch_contexts
    stub.get_num_threads = lambda: threads
    monkeypatch.setattr(human, "_load_converter", lambda path: pytest.fail("Must reject configuration before loading"))
    with pytest.raises(ValueError, match=message):
        human.fit_shared_identity(
            np.zeros((1, 18439, 3)), np.zeros((1, 136)), np.zeros(68), np.zeros(45),
            "unused-model", "unused-converter", device=device, pose_workers=2,
        )
