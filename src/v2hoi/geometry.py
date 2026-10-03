"""Poses, rigid alignment, surface sampling and signed-distance diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Broad unit sanity bound for Track 1 object meshes (metres), not a score.
MESH_EXTENT_M = (0.02, 3.0)


def pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """(T, 7) [x, y, z, qw, qx, qy, qz] -> (T, 4, 4). Zero quaternions give NaN."""
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(-1, 7)
    quat_xyzw = pose7[:, [4, 5, 6, 3]]
    norm = np.linalg.norm(quat_xyzw, axis=1)
    ok = norm > 1e-8
    out = np.full((len(pose7), 4, 4), np.nan)
    out[ok] = np.eye(4)
    if ok.any():
        out[ok, :3, :3] = Rotation.from_quat(quat_xyzw[ok] / norm[ok, None]).as_matrix()
        out[ok, :3, 3] = pose7[ok, :3]
    return out


def matrix_to_pose7(T: np.ndarray) -> np.ndarray:
    """(T, 4, 4) -> (T, 7) [x, y, z, qw, qx, qy, qz]. NaN matrices give zeros."""
    T = np.asarray(T, dtype=np.float64).reshape(-1, 4, 4)
    ok = np.isfinite(T).all(axis=(1, 2))
    out = np.zeros((len(T), 7))
    if ok.any():
        quat_xyzw = Rotation.from_matrix(T[ok, :3, :3]).as_quat()
        out[ok, :3] = T[ok, :3, 3]
        out[ok, 3:] = quat_xyzw[:, [3, 0, 1, 2]]
    return out


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares (s, R, t) with dst ~= s * R @ src + t. src, dst: (N, 3)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2] = -1.0
    R = U @ np.diag(D) @ Vt
    s = float((S * D).sum() / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


@dataclass(frozen=True)
class Similarity:
    """x -> s * R @ x + t."""

    s: float
    R: np.ndarray
    t: np.ndarray

    @classmethod
    def identity(cls) -> "Similarity":
        return cls(1.0, np.eye(3), np.zeros(3))

    @classmethod
    def fit(cls, src: np.ndarray, dst: np.ndarray, with_scale: bool) -> "Similarity":
        return cls(*umeyama(src, dst, with_scale))

    def points(self, x: np.ndarray) -> np.ndarray:
        return self.s * (np.asarray(x, dtype=np.float64) @ self.R.T) + self.t

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.linalg.norm(Rotation.from_matrix(self.R).as_rotvec())))


def load_mesh(path) -> trimesh.Trimesh:
    """Load a mesh the way the official MV evaluation does (scene flattened, no processing)."""
    return trimesh.load(path, process=False, force="mesh")


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Uniform surface samples and the normals of the faces they lie on."""
    points, face_idx = trimesh.sample.sample_surface(mesh, count, seed=seed)
    return np.asarray(points), np.asarray(mesh.face_normals[face_idx])


class SurfaceSDF:
    """Approximate signed distance from dense surface samples and face normals.

    Negative inside. The sign is a vote over the k nearest samples, so it
    tolerates noisy normals on scanned meshes; it assumes outward-facing
    normals. Points outside the sample bounding box get +inf.
    """

    def __init__(self, mesh: trimesh.Trimesh, count: int = 50_000, k: int = 8, seed: int = 0):
        self.points, self.normals = sample_surface(mesh, count, seed)
        self.tree = cKDTree(self.points)
        self.lo = self.points.min(0)
        self.hi = self.points.max(0)
        self.k = k

    def __call__(self, x: np.ndarray, workers: int = -1) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.full(len(x), np.inf)
        near = np.all((x > self.lo) & (x < self.hi), axis=1)
        if not near.any():
            return out
        xn = x[near]
        dist, idx = self.tree.query(xn, k=self.k, workers=workers)
        offsets = xn[:, None, :] - self.points[idx]
        vote = np.sign(np.einsum("nkj,nkj->nk", offsets, self.normals[idx])).sum(1)
        out[near] = np.where(vote < 0, -dist[:, 0], dist[:, 0])
        return out


class ExactTriangleSDF:
    """CPU triangle-distance diagnostic with absolute winding-number sign.

    Negative inside when abs(winding) > 0.5. Uses triangle surfaces rather than
    nearest vertices or sampled normals. Degenerate padding faces are ignored;
    winding on open/self-intersecting meshes retains its usual ambiguity. This
    implementation has not been numerically certified against the official
    GPU evaluator. Bounded chunks avoid a full points-by-faces allocation.
    """

    def __init__(self, mesh: trimesh.Trimesh, point_chunk_size: int = 16, triangle_chunk_size: int = 4096):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError("mesh vertices must be nonempty and finite (V, 3)")
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or np.any(faces < 0) or np.any(faces >= len(vertices))):
            raise ValueError("mesh faces must be valid integer triangles")
        triangles = vertices[faces]
        area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        self.triangles = triangles[area2 > 0]
        if not len(self.triangles):
            raise ValueError("mesh must contain nondegenerate triangles")
        if point_chunk_size <= 0 or triangle_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")
        self.point_chunk_size = point_chunk_size
        self.triangle_chunk_size = triangle_chunk_size

    def __call__(self, x: np.ndarray, workers: int = -1) -> np.ndarray:
        # workers is accepted for the shared metric interface; this path is serial.
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
            raise ValueError("query points must be finite (N, 3)")
        result = np.empty(len(x), dtype=np.float64)
        for start in range(0, len(x), self.point_chunk_size):
            points = x[start:start + self.point_chunk_size]
            distance2 = np.full(len(points), np.inf)
            solid_angle = np.zeros(len(points))
            for face_start in range(0, len(self.triangles), self.triangle_chunk_size):
                triangles = self.triangles[face_start:face_start + self.triangle_chunk_size]
                p, f = len(points), len(triangles)
                batch_triangles = np.broadcast_to(triangles, (p, f, 3, 3)).reshape(-1, 3, 3)
                batch_points = np.broadcast_to(points[:, None], (p, f, 3)).reshape(-1, 3)
                closest = trimesh.triangles.closest_point(batch_triangles, batch_points)
                squared = np.sum((closest - batch_points) ** 2, axis=1).reshape(p, f)
                distance2 = np.minimum(distance2, squared.min(axis=1))
                offsets = triangles[None] - points[:, None, None]
                a, b, c = offsets[:, :, 0], offsets[:, :, 1], offsets[:, :, 2]
                la, lb, lc = np.linalg.norm(offsets, axis=-1).transpose(2, 0, 1)
                numerator = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
                denominator = (la * lb * lc + np.einsum("pfi,pfi->pf", a, b) * lc
                               + np.einsum("pfi,pfi->pf", b, c) * la
                               + np.einsum("pfi,pfi->pf", c, a) * lb)
                solid_angle += (2 * np.arctan2(numerator, denominator)).sum(axis=1)
            distance = np.sqrt(distance2)
            inside = np.abs(solid_angle) > 2 * np.pi
            result[start:start + len(points)] = np.where(inside, -distance, distance)
        return result
