"""Independent synthetic masks/receipt/resume tests; no challenge assets."""
import json
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from v2hoi.sam_body_inputs import (ARTIFACTS, PROFILE, SOURCE_REVISION, camera_matrix,
                                  decoded_frame_sha256, fingerprint, load_body_inputs,
                                  sha256, validate_resume_frame)


def cache_policy_helper():
    """Exercise allocator policy without importing an optional GPU runtime."""
    source = Path(__file__).resolve().parents[1] / "tools/run_sam3d_body.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "call_with_cuda_cache_policy")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("fails", [False, True])
def test_retain_cache_policy_restores_api_after_success_and_failure(fails):
    calls = []
    original = lambda: calls.append("upstream purge")
    cuda = SimpleNamespace(empty_cache=original)

    def operation():
        cuda.empty_cache()
        if fails:
            raise RuntimeError("inference failed")
        return "prediction"

    if fails:
        with pytest.raises(RuntimeError, match="inference failed"):
            cache_policy_helper()(cuda, device="cuda", policy="retain", operation=operation)
    else:
        assert cache_policy_helper()(cuda, device="cuda", policy="retain", operation=operation) == "prediction"
    assert calls == []
    assert cuda.empty_cache is original
    cuda.empty_cache()
    assert calls == ["upstream purge"]


@pytest.mark.parametrize("device,policy", [("cuda", "upstream"), ("cpu", "retain")])
def test_cache_policy_preserves_upstream_or_cpu_behavior(device, policy):
    calls = []
    original = lambda: calls.append("unchanged")
    cuda = SimpleNamespace(empty_cache=original)
    cache_policy_helper()(cuda, device=device, policy=policy, operation=cuda.empty_cache)
    assert calls == ["unchanged"]
    assert cuda.empty_cache is original


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def bind(root, receipt):
    identity = json.loads((root / "input.json").read_text())
    identity["artifacts"] = {name: sha256(root / name) for name in ARTIFACTS}
    identity["source"]["timeline_sha256"] = sha256(root / "timeline.json")
    write(root / "input.json", identity)
    review = json.loads((root / "review.json").read_text())
    review["reviewed_artifacts"] = {name: sha256(root / name) for name in ARTIFACTS | {"input.json"}}
    write(root / "review.json", review)
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (root / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths))
    report = json.loads(receipt.read_text())
    report["episodes"][16].update(input_sha256=sha256(root / "input.json"), review_sha256=sha256(root / "review.json"))
    write(receipt, report)


@pytest.fixture
def synthetic(tmp_path):
    root = tmp_path / "episode_000016"
    root.mkdir()
    (root / "human_masks").mkdir()
    video = tmp_path / "independent-synthetic-video.bin"
    video.write_bytes(b"Independent synthetic video identity fixture; not decodable media")
    video_sha = sha256(video)
    masks, observations = [], []
    for index in range(3):
        pixels = np.zeros((12, 16), dtype=np.uint8)
        pixels[2:9, 3 + index:9 + index] = 255 if index != 1 else 1
        path = root / f"human_masks/{index:06d}.png"
        Image.fromarray(pixels).save(path)
        masks.append({"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
        observations.append({"frame_id": index, "valid": True, "area_pixels": 42, "bbox_xyxy": [3 + index, 2, 9 + index, 9]})
    intrinsics = {"fx": 16.25, "fy": 17.125, "cx": 7.5, "cy": 5.5, "width": 16, "height": 12}
    write(root / "intrinsics.json", intrinsics)
    write(root / "timeline.json", {"schema_version": "track1_original_timeline.v1", "episode_index": 16,
          "video_sha256": video_sha, "first_source_frame_id": 0, "width": 16, "height": 12,
          "frame_count": 3, "frame_ids": [0, 1, 2], "pts": [0, 512, 1024],
          "time_base": "1/15360", "timestamps_seconds": [0, 1 / 30, 2 / 30]})
    write(root / "masks-manifest.json", {"schema_version": "body_masks.v1", "episode_index": 16,
          "video_sha256": video_sha, "files": masks})
    write(root / "observations.json", {"schema_version": "body_observations.v1", "episode_index": 16,
          "target_id": "synthetic-human", "frames": observations})
    write(root / "camera-estimation.json", {"schema_version": "body_camera_estimation.v1", "episode_index": 16,
          "video_sha256": video_sha, "status": "camera_estimate_consistent", "method": "independent synthetic fixture"})
    write(root / "input.json", {"schema_version": "body_input.v1", "profile": PROFILE, "episode_index": 16,
          "target_id": "synthetic-human", "producer": {"method": "independent synthetic fixture"},
          "source": {"repository": "nvidia/video_to_data_challenge", "revision": SOURCE_REVISION,
          "video_path": "videos/chunk-000/observation.images.exo_camera/episode_000016.mp4", "video_sha256": video_sha}})
    write(root / "review.json", {"status": "approved", "unresolved_issues": []})
    receipt = tmp_path / "receiver.json"
    rows = [{"episode_index": i, "body_inputs_verified": True, "inputs_structurally_verified": True,
             "visual_review_verified": True, "errors": [], "video_sha256": video_sha, "frame_count": 3} for i in range(30)]
    write(receipt, {"schema_version": "body_handoff_checks.v1", "profile": PROFILE, "passed": True,
          "all_30_body_inputs_verified": True, "requested_episodes": list(range(30)), "episodes": rows})
    bind(root, receipt)
    return root, receipt, video


def load(synthetic, **changes):
    root, receipt, video = synthetic
    args = dict(video=video, episode=16, width=16, height=12, frame_count=3, fps=30)
    args.update(changes)
    return load_body_inputs(root, receipt, **args)


def test_mask_normalization_exact_bbox_and_pixel_center_camera(synthetic):
    adapter = load(synthetic)
    assert adapter.intrinsics.dtype == np.float32
    np.testing.assert_array_equal(adapter.intrinsics, [[16.25, 0, 7.5], [0, 17.125, 5.5], [0, 0, 1]])
    for index in range(3):
        masks, bbox, digest = adapter.frame(index)
        assert masks.shape == (1, 12, 16, 1) and masks.dtype == np.uint8
        assert set(np.unique(masks)) == {0, 255}
        assert np.count_nonzero(masks) == 42
        np.testing.assert_array_equal(bbox, [[3 + index, 2, 9 + index, 9]])
        assert digest == adapter.mask_files[index]["sha256"]


@pytest.mark.parametrize("change", ["subset", "not_passed", "duplicate_episode", "failed_other_episode", "stale_input", "stale_review", "wrong_video"])
def test_receiver_must_bind_a_passing_all30_run_to_consumed_bytes(synthetic, change):
    root, receipt, video = synthetic
    report = json.loads(receipt.read_text())
    if change == "subset":
        report["all_30_body_inputs_verified"] = False
    elif change == "not_passed":
        report["passed"] = False
    elif change == "duplicate_episode":
        report["episodes"][3]["episode_index"] = 4
    elif change == "failed_other_episode":
        report["episodes"][3]["body_inputs_verified"] = False
    elif change == "stale_input":
        report["episodes"][16]["input_sha256"] = "a" * 64
    elif change == "stale_review":
        report["episodes"][16]["review_sha256"] = "a" * 64
    elif change == "wrong_video":
        video.write_bytes(b"different synthetic input")
    write(receipt, report)
    with pytest.raises(ValueError):
        load(synthetic)


@pytest.mark.parametrize("kind", ["nonbinary", "empty", "full", "one_pixel_width", "wrong_box", "wrong_frame", "extra_mask", "stale_manifest"])
def test_bad_mask_pixels_or_observations_fail_even_after_storage_rebinding(synthetic, kind):
    root, receipt, _ = synthetic
    path = root / "human_masks/000001.png"
    pixels = np.asarray(Image.open(path)).copy()
    if kind == "nonbinary":
        pixels[3, 4] = 99
    elif kind == "empty":
        pixels[:] = 0
    elif kind == "full":
        pixels[:] = 255
    elif kind == "one_pixel_width":
        pixels[:] = 0
        pixels[2:9, 4] = 255
    elif kind in ("wrong_box", "wrong_frame"):
        observations = json.loads((root / "observations.json").read_text())
        observations["frames"][1]["bbox_xyxy" if kind == "wrong_box" else "frame_id"] = [4, 2, 9, 9] if kind == "wrong_box" else 2
        write(root / "observations.json", observations)
    elif kind == "extra_mask":
        Image.fromarray(pixels).save(root / "human_masks/000003.png")
    Image.fromarray(pixels).save(path)
    masks = json.loads((root / "masks-manifest.json").read_text())
    if kind != "stale_manifest":
        masks["files"][1].update(sha256=sha256(path), bytes=path.stat().st_size)
        write(root / "masks-manifest.json", masks)
    else:
        path.write_bytes(path.read_bytes() + b"changed bytes")
    bind(root, receipt)
    with pytest.raises(ValueError):
        load(synthetic)


def test_mask_mutation_between_preflight_and_consumption_is_rejected(synthetic):
    adapter = load(synthetic)
    path = synthetic[0] / "human_masks/000001.png"
    path.write_bytes(path.read_bytes() + b"changed after preflight")
    with pytest.raises(ValueError, match="Mask bytes changed"):
        adapter.frame(1)


@pytest.mark.parametrize("change", [{"width": 8}, {"frame_count": 2}, {"fps": 25}, {"episode": 15}])
def test_video_dimensions_timeline_fps_and_episode_are_not_silently_remapped(synthetic, change):
    with pytest.raises(ValueError):
        load(synthetic, **change)


@pytest.mark.parametrize("key,value", [("fx", 0), ("fy", float("inf")), ("cx", 16), ("cy", -1), ("width", True)])
def test_camera_rejects_invalid_grid_without_guessing(key, value):
    fields = {"fx": 16, "fy": 16, "cx": 7.5, "cy": 5.5, "width": 16, "height": 12}
    fields[key] = value
    with pytest.raises(ValueError):
        camera_matrix(fields, 16, 12)


def resume_record():
    record = {name: np.zeros(shape, dtype=np.float32) for name, shape in (
        ("pose", (136,)), ("scales", (68,)), ("shape", (45,)), ("pred_vertices", (18439, 3)),
        ("pred_keypoints_2d", (70, 2)), ("pred_keypoints_3d", (70, 3)), ("pred_cam_t", (3,)))}
    record.update(frame_index=np.asarray(5), input_configuration_sha256=np.asarray("a" * 64),
                  decoded_bgr_sha256=np.asarray("b" * 64), input_mask_sha256=np.asarray("c" * 64))
    return record


@pytest.mark.parametrize("field,value", [("frame_index", 6), ("input_configuration_sha256", "d" * 64),
    ("decoded_bgr_sha256", "d" * 64), ("input_mask_sha256", "d" * 64), ("pred_cam_t", [0, float("nan"), 0])])
def test_resume_rejects_wrong_source_configuration_masks_and_corruption(field, value):
    record = resume_record()
    validate_resume_frame(record, 5, "a" * 64, "b" * 64, "c" * 64)
    record[field] = np.asarray(value)
    with pytest.raises(ValueError):
        validate_resume_frame(record, 5, "a" * 64, "b" * 64, "c" * 64)


def test_pixel_hash_tracks_actual_pixels_and_layout_not_memory_strides():
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    assert decoded_frame_sha256(image) == decoded_frame_sha256(image[:, ::-1][:, ::-1])
    assert decoded_frame_sha256(image) != decoded_frame_sha256(image.reshape(8, 6, 3))
    modified = image.copy()
    modified[2, 3, 1] += 1
    assert decoded_frame_sha256(image) != decoded_frame_sha256(modified)
    assert fingerprint({"mask": "a", "camera": "b"}) == fingerprint({"camera": "b", "mask": "a"})
