"""Independent synthetic geometry fixtures; no challenge assets or model decoder."""
import numpy as np
import pytest

from v2hoi.v2.artifacts import ContractError
from v2hoi.v2.geometry import (
    rebase_object_canonical,
    transform_image_points,
    transform_intrinsics,
    validate_image_transform,
    validate_native_human,
    validate_object_trajectory,
    world_object_vertices,
)


def _native():
    # Native scale coefficients are decoder-defined, not positive metric sizes.
    return dict(pose=np.zeros((3, 136)), shape=np.zeros(45), scales=np.zeros(68),
                frame_ids=np.arange(3), expected_frame_ids=np.arange(3))


def _object():
    return dict(vertices=np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
                rotations=np.array([np.eye(3), [[0, -1, 0], [1, 0, 0], [0, 0, 1]]]),
                translations=np.array([[0, 0, 1], [3, 4, 5]]),
                frame_ids=np.arange(2), expected_frame_ids=np.arange(2),
                scale_mode="explicit_uniform", scale=0.5)


def _geometry(data):
    return {key: value for key, value in data.items() if key not in ("frame_ids", "expected_frame_ids")}


def test_native_shapes_do_not_claim_decoder_or_convert_legacy():
    validate_native_human(**_native())
    for key, value in (("pose", np.zeros((3, 204))), ("pose", np.zeros((3, 77, 3))),
                       ("shape", np.zeros((3, 45))), ("scales", np.zeros((3, 68))),
                       ("scales", np.zeros(28))):
        with pytest.raises(ContractError, match="shape"):
            validate_native_human(**(_native() | {key: value}))


@pytest.mark.parametrize("frame_ids", [[0, 2], [0, 1, 1], [2, 1, 0], [1, 2, 3],
                                       [0.0, 1.0, 2.0], [False, True, True]])
def test_native_rejects_missing_hidden_frames_and_local_or_noninteger_maps(frame_ids):
    with pytest.raises(ContractError, match="frame"):
        validate_native_human(**(_native() | {"frame_ids": frame_ids}))


def test_native_expected_coverage_cannot_be_a_selected_scoring_subset():
    data = _native() | {"frame_ids": [0, 2], "expected_frame_ids": [0, 2], "pose": np.zeros((2, 136))}
    with pytest.raises(ContractError, match="untrimmed"):
        validate_native_human(**data)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_native_rejects_nonfinite_parameter_values(invalid):
    for key in ("pose", "shape", "scales"):
        data = _native()
        data[key].flat[0] = invalid
        with pytest.raises(ContractError, match="finite"):
            validate_native_human(**data)


@pytest.mark.parametrize("invalid", [np.zeros((3, 136), dtype=object),
                                     np.zeros((3, 136), dtype=complex), [[1], [1, 2]]])
def test_native_rejects_object_complex_and_ragged_payloads(invalid):
    with pytest.raises(ContractError, match="numeric"):
        validate_native_human(**(_native() | {"pose": invalid}))


def test_known_world_vertices_scale_once_and_keep_translation_in_metres():
    data = _object()
    validate_object_trajectory(**data)
    world = world_object_vertices(**_geometry(data))
    expected = np.array([[[0.5, 0, 1], [0, 1, 1], [0, 0, 2.5]],
                         [[3, 4.5, 5], [2, 4, 5], [3, 4, 6.5]]])
    np.testing.assert_allclose(world, expected, atol=0, rtol=0)
    # Independently equivalent baked metric vertices never apply scale again.
    baked = data | {"vertices": data["vertices"] * 0.5, "scale_mode": "baked_metric", "scale": 1}
    np.testing.assert_allclose(world_object_vertices(**_geometry(baked)), expected, atol=0, rtol=0)
    with pytest.raises(ContractError, match="provenance scale"):
        validate_object_trajectory(**(baked | {"scale": 0.5}))


@pytest.mark.parametrize("scale", [0, -1, [0.5, 0.5], [0.5], True, float("inf"), float("nan")])
def test_object_scale_must_be_one_positive_finite_constant(scale):
    with pytest.raises(ContractError):
        validate_object_trajectory(**(_object() | {"scale": scale}))


@pytest.mark.parametrize("rotation", [np.diag([1, 1, -1]), np.diag([1.01, 1, 1]),
                                      [[1, 0.001, 0], [0, 1, 0], [0, 0, 1]], np.zeros((3, 3))])
def test_object_rejects_reflected_scaled_sheared_and_zero_rotations(rotation):
    data = _object()
    data["rotations"][1] = rotation
    with pytest.raises(ContractError, match=r"SO\(3\)"):
        validate_object_trajectory(**data)


def test_object_coverage_and_arrays_must_agree():
    data = _object()
    with pytest.raises(ContractError, match="original frames"):
        validate_object_trajectory(**(data | {"frame_ids": [0]}))
    with pytest.raises(ContractError, match="rotations.*shape"):
        validate_object_trajectory(**(data | {"frame_ids": [0, 1, 2], "expected_frame_ids": [0, 1, 2]}))
    with pytest.raises(ContractError, match="translations.*shape"):
        validate_object_trajectory(**(data | {"translations": [[0, 0, 1]]}))
    with pytest.raises(ContractError, match="vertices"):
        validate_object_trajectory(**(data | {"vertices": []}))


def test_object_nonfinite_completion_and_overflow_fail():
    data = _object()
    translations = data["translations"].astype(float)
    translations[1, 0] = np.nan
    with pytest.raises(ContractError, match="finite"):
        validate_object_trajectory(**(data | {"translations": translations}))
    with pytest.raises(ContractError, match="overflow"):
        world_object_vertices(**_geometry(data | {"vertices": np.full((3, 3), 1e308), "scale": 100}))


@pytest.mark.parametrize("scale_mode,scale", [("explicit_uniform", 0.5), ("baked_metric", 1.0)])
def test_canonical_rebase_preserves_world_geometry_with_scaled_origin(scale_mode, scale):
    data = _object() | {"scale_mode": scale_mode, "scale": scale}
    original = {name: value.copy() for name, value in data.items() if isinstance(value, np.ndarray)}
    transform = np.array([[0, 0, 1, 3], [0, 1, 0, -2], [-1, 0, 0, 4], [0, 0, 0, 1]])
    rebased = rebase_object_canonical(**_geometry(data), T_new_from_old=transform)
    np.testing.assert_allclose(rebased.vertices, [[3, -2, 3], [3, 0, 4], [6, -2, 4]], atol=0, rtol=0)
    actual = world_object_vertices(rebased.vertices, rebased.rotations, rebased.translations,
                                   scale_mode=rebased.scale_mode, scale=rebased.scale)
    np.testing.assert_allclose(actual, world_object_vertices(**_geometry(data)), atol=1e-12, rtol=0)
    assert rebased.requires_revalidation is True
    for name, value in original.items():
        np.testing.assert_array_equal(data[name], value)
    # A mesh-only canonical edit changes the scene; the paired pose change is necessary.
    assert not np.allclose(world_object_vertices(**_geometry(data | {"vertices": rebased.vertices})), actual)


@pytest.mark.parametrize("transform", [np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1]),
                                       np.diag([1, 1, 1, 2]), np.eye(3)])
def test_rebase_rejects_nonrigid_or_wrong_shape_transforms(transform):
    with pytest.raises(ContractError):
        rebase_object_canonical(**_geometry(_object()), T_new_from_old=transform)


def test_crop_resize_pad_coordinates_roundtrip_and_intrinsics_agree():
    # Explicit pixel-centre convention: crop origin (100, 50), scale by 0.5,
    # then pad by (7, 11). The 0.5-centre resize correction is recorded here.
    transform = np.array([[0.5, 0, -43.25], [0, 0.5, -14.25], [0, 0, 1]])
    validate_image_transform(transform, (640, 480), (256, 256))
    source = np.array([[100, 50], [102, 54], [320, 240]])
    processed = transform_image_points(source, transform)
    np.testing.assert_allclose(processed, [[6.75, 10.75], [7.75, 12.75], [116.75, 105.75]])
    np.testing.assert_allclose(transform_image_points(processed, np.linalg.inv(transform)), source)
    intrinsics = np.array([[400, 0, 320], [0, 420, 240], [0, 0, 1]])
    processed_intrinsics = transform_intrinsics(intrinsics, transform)
    np.testing.assert_allclose(processed_intrinsics, [[200, 0, 116.75], [0, 210, 105.75], [0, 0, 1]])
    camera_points = np.array([[0.1, -0.3, 2], [0.6, 0.9, 3]])
    source_projection = camera_points @ intrinsics.T
    source_pixels = source_projection[:, :2] / source_projection[:, 2:]
    target_projection = camera_points @ processed_intrinsics.T
    np.testing.assert_allclose(transform_image_points(source_pixels, transform),
                               target_projection[:, :2] / target_projection[:, 2:])


@pytest.mark.parametrize("size", [(0, 480), (640, -1), (640.0, 480.0), (True, False), (640,), [640, 480, 3]])
def test_image_dimensions_are_positive_explicit_integers(size):
    with pytest.raises(ContractError, match="positive integer"):
        validate_image_transform(np.eye(3), size, (128, 128))
    with pytest.raises(ContractError, match="positive integer"):
        validate_image_transform(np.eye(3), (640, 480), size)


@pytest.mark.parametrize("transform", [np.diag([0, 1, 1]),
                                       [[1, 0, 0], [0, 1, 0], [0.1, 0, 1]],
                                       np.full((3, 3), np.nan)])
def test_image_map_rejects_singular_projective_and_nonfinite_transforms(transform):
    with pytest.raises(ContractError):
        validate_image_transform(transform, (640, 480), (128, 128))


@pytest.mark.parametrize("intrinsics", [np.diag([-1, 1, 1]), np.diag([1, 0, 1]),
                                       np.diag([1, 1, 2]), np.eye(4)])
def test_intrinsics_reject_invalid_source_projection(intrinsics):
    with pytest.raises(ContractError):
        transform_intrinsics(intrinsics, np.eye(3))
