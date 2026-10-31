"""Depth-tested display of estimated geometry, without contact-quality claims."""
from __future__ import annotations

import numpy as np


def project(points, camera):
    points = np.asarray(points, dtype=np.float64)
    camera = np.asarray(camera, dtype=np.float64)
    if points.shape[-1] != 3 or camera.shape != (3, 3):
        raise ValueError("Expected camera-space XYZ and a 3x3 camera")
    if not np.isfinite(points).all() or np.any(points[..., 2] <= 0) or not np.isfinite(camera).all():
        raise ValueError("Display geometry needs finite positive depth")
    homogeneous = points @ camera.T
    return homogeneous[..., :2] / homogeneous[..., 2:]


def fixed_union_roi(point_sets, *, padding=1.12):
    """One square over every frame/method; display never refits model geometry."""
    if not point_sets or not np.isfinite(padding) or padding < 1:
        raise ValueError("A finite padded display union is required")
    values = [np.asarray(p, dtype=float).reshape(-1, 2) for p in point_sets]
    if any(len(p) == 0 or not np.isfinite(p).all() for p in values):
        raise ValueError("Display points must be finite and nonempty")
    lower = np.min([p.min(0) for p in values], axis=0)
    upper = np.max([p.max(0) for p in values], axis=0)
    side = max(40, int(np.ceil((upper - lower).max() * padding)) + 2)
    if side > 4096:
        raise ValueError("Display union exceeds the supported source extent")
    first = np.floor((lower + upper - side) / 2).astype(np.int64)
    return np.r_[first, first + side]


def rasterize_meshes(meshes, camera, roi, size=384):
    """Perspective-correct shared Z-buffer for full native and object triangles.

    Meshes are (camera_XYZ, integer_faces, BGR_albedo, positive_entity_id).
    Background ID is zero. No real table/scene depth or physical truth is added.
    Faces crossing the camera plane are omitted; callers validate full geometry.
    """
    roi = np.asarray(roi, dtype=float)
    if roi.shape != (4,) or not np.isfinite(roi).all() or np.any(roi[2:] <= roi[:2]):
        raise ValueError("Invalid display ROI")
    if isinstance(size, bool) or not isinstance(size, int) or size < 8:
        raise ValueError("Display size must be an integer >=8")
    image = np.zeros((size, size, 3), np.uint8)
    depth = np.full((size, size), np.inf, np.float64)
    entity = np.zeros((size, size), np.int16)
    scale = size / (roi[2:] - roi[:2])
    for vertices, faces, albedo, identity in meshes:
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces)
        color = np.asarray(albedo, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("Mesh vertices must be finite [V,3]")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu" or np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("Mesh faces must index valid triangles")
        if color.shape != (3,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 255)):
            raise ValueError("Display albedo must be BGR in [0,255]")
        if isinstance(identity, bool) or not isinstance(identity, int) or not 1 <= identity <= 32767:
            raise ValueError("Positive int16 entity identity required")
        valid = (vertices[:, 2] > 1e-6)[faces].all(1)
        selected = faces[valid]
        if not len(selected):
            continue
        xy = np.zeros((len(vertices), 2), dtype=float)
        positive = vertices[:, 2] > 1e-6
        xy[positive] = (project(vertices[positive], camera) - roi[:2]) * scale
        polygons = xy[selected]
        low = np.maximum(0, np.floor(polygons.min(1)).astype(int))
        high = np.minimum(size - 1, np.ceil(polygons.max(1)).astype(int))
        visible = (high >= low).all(1)
        triangles = vertices[selected]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        length = np.linalg.norm(normals, axis=1)
        shade = .45 + .55 * np.abs(normals[:, 2]) / np.maximum(length, 1e-12)
        colors = np.clip(shade[:, None] * color, 0, 255).astype(np.uint8)
        for index in np.flatnonzero(visible & (length > 1e-12)):
            a, b, c = polygons[index]
            ab, ac = b - a, c - a
            denominator = ab[0] * ac[1] - ac[0] * ab[1]
            if abs(denominator) < 1e-10:
                continue
            x0, y0 = low[index]
            x1, y1 = high[index] + 1
            yy, xx = np.ogrid[y0:y1, x0:x1]
            rx, ry = xx + .5 - a[0], yy + .5 - a[1]
            w1 = (rx * ac[1] - ry * ac[0]) / denominator
            w2 = (ab[0] * ry - ab[1] * rx) / denominator
            w0 = 1 - w1 - w2
            inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
            inverse = w0 / triangles[index, 0, 2] + w1 / triangles[index, 1, 2] + w2 / triangles[index, 2, 2]
            z = np.divide(1, inverse, out=np.full_like(inverse, np.inf), where=inverse > 0)
            patch = depth[y0:y1, x0:x1]
            keep = inside & (z < patch)
            patch[keep] = z[keep]
            image[y0:y1, x0:x1][keep] = colors[index]
            entity[y0:y1, x0:x1][keep] = identity
    return image, depth, entity
