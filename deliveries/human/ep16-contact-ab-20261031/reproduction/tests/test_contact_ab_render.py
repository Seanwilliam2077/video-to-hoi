"""Independent synthetic triangles only, without challenge or model assets."""
import numpy as np
import pytest

from v2hoi.contact_review import fixed_union_roi, project, rasterize_meshes


def triangle(z):
    return np.asarray([[0., 0., 1.], [7., 0., 1.], [0., 7., 1.]]) * z


def test_occlusion_is_shared_between_entities_and_independent_of_submission_order():
    face = np.asarray([[0, 1, 2]])
    near = (triangle(1), face, [0, 255, 0], 1)
    far = (triangle(3), face, [0, 0, 255], 2)
    first = rasterize_meshes([near, far], np.eye(3), [0, 0, 8, 8], 8)
    other = rasterize_meshes([far, near], np.eye(3), [0, 0, 8, 8], 8)
    assert np.array_equal(first[2], other[2])
    assert first[2][1, 1] == 1 and first[1][1, 1] == 1
    assert first[2][7, 7] == 0 and np.isinf(first[1][7, 7])


def test_depth_uses_inverse_z_barycentrics():
    screen = triangle(1)
    depths = np.asarray([1., 2., 4.])
    xyz = screen * depths[:, None]
    _, z, _ = rasterize_meshes([(xyz, np.asarray([[0, 1, 2]]), [255]*3, 1)], np.eye(3), [0, 0, 8, 8], 8)
    w1 = w2 = 1.5 / 7
    expected = 1 / ((1-w1-w2) + w1/2 + w2/4)
    assert z[1, 1] == pytest.approx(expected)


def test_fixed_union_contains_every_method_frame_even_outside_source_border():
    points = [np.asarray([[-10, 5], [80, 30]]), np.asarray([[20, -15], [100, 95]])]
    roi = fixed_union_roi(points)
    assert np.array_equal(roi[2:]-roi[:2], np.repeat(roi[2]-roi[0], 2))
    for p in points:
        assert np.all(p >= roi[:2]) and np.all(p <= roi[2:])


def test_invalid_geometry_and_topology_cannot_silently_render():
    with pytest.raises(ValueError, match="positive depth"):
        project(np.asarray([[0, 0, -1]]), np.eye(3))
    with pytest.raises(ValueError, match="valid triangles"):
        rasterize_meshes([(triangle(1), np.asarray([[0, 1, 3]]), [255]*3, 1)], np.eye(3), [0, 0, 8, 8], 8)


def test_bounded_process_renderer_matches_serial_on_independent_synthetic_geometry(tmp_path):
    pytest.importorskip("cv2")
    from concurrent.futures import ProcessPoolExecutor
    from tools.render_contact_ab import initialize_renderer, render_layers

    face = np.asarray([[0, 1, 2]])
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        review = root / "review_arrays"
        review.mkdir(parents=True)
        np.save(review / "vertices_m.npy", np.stack([triangle(1), triangle(1.5)]))
        np.save(review / "faces.npy", face)
    obj = tmp_path / "object.npz"
    transforms = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    np.savez_compressed(obj, vertices=triangle(3), faces=face, object_to_camera=transforms)
    arguments = ([str(r) for r in roots], str(obj), np.eye(3), np.asarray([0, 0, 8, 8]), 8)
    items = [(frame, np.full((8, 8, 3), frame*15, np.uint8)) for frame in range(2)]
    initialize_renderer(*arguments)
    serial = [render_layers(item)[0] for item in items]
    with ProcessPoolExecutor(max_workers=2, initializer=initialize_renderer, initargs=arguments) as pool:
        parallel = list(pool.map(render_layers, items))
    for original, result in zip(serial, parallel):
        assert all(np.array_equal(a, b) for a, b in zip(original, result[0]))
