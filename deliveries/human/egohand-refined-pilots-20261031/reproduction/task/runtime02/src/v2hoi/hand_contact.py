"""Weak, explicitly hypothetical hand/object contact constraints.

Only manually admitted grasp phases attract a few already near-surface tips.
The object is a Track 1 RGB-derived convex geometry/pose hypothesis, not a
reference answer. Native hand-skin diagnostics measure this candidate geometry;
they are not the challenge penetration score.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

TIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)


def convex_mesh_planes(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a closed convex mesh and return outward unit supporting planes."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.isfinite(v).all():
        raise ValueError("object vertices must be finite [V>=4,3] metres")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) < 4 or f.dtype.kind not in "iu" or np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("object faces must be valid integer triangles")
    tri = v[f]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normal, axis=1)
    if np.any(length < 1e-12):
        raise ValueError("object mesh contains degenerate triangles")
    normal /= length[:, None]
    # Face winding may be reversed; convex centroid is unambiguously inside.
    normal *= np.where(np.einsum("ij,ij->i", normal, tri.mean(1) - v.mean(0)) < 0, -1, 1)[:, None]
    offset = np.einsum("ij,ij->i", normal, tri[:, 0])
    tolerance = max(1e-6, float(np.linalg.norm(np.ptp(v, axis=0))) * 1e-5)
    for start in range(0, len(v), 128):
        if np.max(v[start:start + 128] @ normal.T - offset) > tolerance:
            raise ValueError("object mesh is not convex; supporting-plane penetration is invalid")
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, count = np.unique(edges, axis=0, return_counts=True)
    if np.any(count != 2):
        raise ValueError("object mesh must be closed and manifold")
    return normal.astype(np.float32), offset.astype(np.float32)


def closest_mesh_points(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, *, chunk_size=32) -> tuple[np.ndarray, np.ndarray]:
    """Exact closest points on triangles, using plane interiors and three edges."""
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("closest-point queries must be finite [N,3]")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    squared = np.einsum("ij,ij->i", normal, normal)
    if np.any(squared <= 1e-24):
        raise ValueError("closest-point mesh has degenerate triangles")
    normal /= np.sqrt(squared)[:, None]
    d00, d01, d11 = (ab * ab).sum(1), (ab * ac).sum(1), (ac * ac).sum(1)
    determinant = d00 * d11 - d01**2
    out, distances = [], []
    for start in range(0, len(p), chunk_size):
        current = p[start:start + chunk_size, None]
        signed = np.einsum("nfi,fi->nf", current - a, normal)
        q = current - signed[..., None] * normal
        relative = q - a
        d20, d21 = np.einsum("nfi,fi->nf", relative, ab), np.einsum("nfi,fi->nf", relative, ac)
        u, w = (d11 * d20 - d01 * d21) / determinant, (d00 * d21 - d01 * d20) / determinant
        inside = (u >= 0) & (w >= 0) & (u + w <= 1)
        best_distance = np.where(inside, ((q - current) ** 2).sum(-1), np.inf)
        best = q.copy()
        for first, last in ((a, b), (b, c), (c, a)):
            edge = last - first
            fraction = (np.einsum("nfi,fi->nf", current - first, edge) / (edge * edge).sum(-1)).clip(0, 1)
            candidate = first + fraction[..., None] * edge
            distance = ((candidate - current) ** 2).sum(-1)
            better = distance < best_distance
            best = np.where(better[..., None], candidate, best)
            best_distance = np.minimum(best_distance, distance)
        nearest = best_distance.argmin(axis=1)
        row = np.arange(len(current))
        out.append(best[row, nearest])
        distances.append(np.sqrt(best_distance[row, nearest]))
    return (np.concatenate(out).astype(np.float32), np.concatenate(distances).astype(np.float32)) if len(p) else (np.empty((0, 3), np.float32), np.empty(0, np.float32))


def validate_object_candidate(candidate: Mapping[str, Any], frame_indices: np.ndarray) -> dict[str, np.ndarray]:
    indices = np.asarray(candidate["frame_indices"])
    if indices.dtype.kind not in "iu" or not np.array_equal(indices, frame_indices):
        raise ValueError("object frame indices must exactly match the Human result")
    transform = np.asarray(candidate["object_to_camera"], dtype=np.float32)
    if transform.shape != (len(indices), 4, 4) or not np.isfinite(transform).all():
        raise ValueError("object_to_camera must be finite [T,4,4] in metres")
    if not np.allclose(transform[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("object homogeneous transforms have invalid last rows")
    rotation = transform[:, :3, :3]
    if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4):
        raise ValueError("object rotations must be proper orthonormal matrices")
    vertices = np.asarray(candidate["vertices"], dtype=np.float32)
    faces = np.asarray(candidate["faces"])
    normal, offset = convex_mesh_planes(vertices, faces)
    pose_usable = np.asarray(candidate.get("pose_usable", np.ones(len(indices), dtype=np.uint8)))
    if pose_usable.shape != (len(indices),) or not np.isfinite(pose_usable).all() or not np.isin(pose_usable, [0, 1]).all():
        raise ValueError("object pose_usable must be an aligned binary reliability gate")
    return {"frame_indices": indices.copy(), "object_to_camera": transform.copy(), "vertices": vertices.copy(),
            "faces": faces.astype(np.int64, copy=True), "normals": normal, "offsets": offset,
            "pose_usable": pose_usable.astype(bool)}


def phase_weights(phases: Mapping[str, Any], frames: int, *, ramp_frames=8) -> np.ndarray:
    """Use only explicit positive anchor phases; all transitions remain zero."""
    if phases.get("track") != 1 or phases.get("frame_count") != frames:
        raise ValueError("contact phase annotations must identify the same Track 1 frame count")
    weights = np.zeros((frames, 2), dtype=np.float32)
    covered = np.zeros(frames, dtype=bool)
    for interval in phases["phase_intervals_inclusive"]:
        first, last = interval["first"], interval["last"]
        if isinstance(first, bool) or isinstance(last, bool) or not isinstance(first, int) or not isinstance(last, int) or not 0 <= first <= last < frames:
            raise ValueError("contact phase interval lies outside the original video")
        if covered[first:last + 1].any():
            raise ValueError("contact phase intervals must not overlap")
        covered[first:last + 1] = True
        weight = float(interval.get("anchor_weight", 0))
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("contact phase weights must lie in [0,1]")
        if weight == 0:
            continue
        sides = interval.get("eligible_sides", [])
        if not sides or any(side not in ("left", "right") for side in sides):
            raise ValueError("positive phases need explicit anatomical eligible sides")
        position = np.arange(last - first + 1)
        ramp = np.minimum(1, np.minimum(position + 1, len(position) - position) / max(1, ramp_frames))
        for name in sides:
            side = ("left", "right").index(name)
            weights[first:last + 1, side] = np.maximum(weights[first:last + 1, side], weight * ramp)
    return weights


class WeakContactConstraint:
    """Fixed local anchors with a tiny bounded effect, never asserted contact truth."""
    def __init__(self, candidate: Mapping[str, Any], phases: Mapping[str, Any], baseline_hands: np.ndarray,
                 frame_indices: np.ndarray, *, device="cpu", maximum_anchor_distance_m=0.04,
                 maximum_anchors_per_hand=2, anchor_weight=0.01, penetration_weight=0.005,
                 maximum_object_step_m=0.06, maximum_object_rotation_step_rad=0.5):
        import torch

        self.candidate = validate_object_candidate(candidate, frame_indices)
        self.weights = phase_weights(phases, len(frame_indices)) * self.candidate["pose_usable"][:, None]
        baseline = np.asarray(baseline_hands, dtype=np.float32)
        if baseline.shape != (len(frame_indices), 2, 21, 3) or not np.isfinite(baseline).all():
            raise ValueError("baseline native hands must be finite [T,2,21,3]")
        self.transform = self.candidate["object_to_camera"]
        carry = (self.weights > 0).any(1)
        adjacent = carry[1:] & carry[:-1]
        step = np.linalg.norm(np.diff(self.transform[:, :3, 3], axis=0), axis=-1)
        rotation_delta = self.transform[1:, :3, :3] @ self.transform[:-1, :3, :3].transpose(0, 2, 1)
        angular_step = np.arccos(np.clip((np.trace(rotation_delta, axis1=1, axis2=2) - 1) / 2, -1, 1))
        self.maximum_object_step = float(step[adjacent].max()) if adjacent.any() else 0
        self.maximum_object_rotation_step = float(angular_step[adjacent].max()) if adjacent.any() else 0
        if self.maximum_object_step > maximum_object_step_m or self.maximum_object_rotation_step > maximum_object_rotation_step_rad:
            raise ValueError("Object trajectory jumps in an admitted grasp phase; fixed local contacts must abstain")
        self.anchor_weight, self.penetration_weight = float(anchor_weight), float(penetration_weight)
        if min(self.anchor_weight, self.penetration_weight) < 0 or not np.isfinite([self.anchor_weight, self.penetration_weight]).all():
            raise ValueError("contact loss weights must be finite and nonnegative")
        self.anchor_targets = np.zeros((len(frame_indices), 2, 5, 3), dtype=np.float32)
        self.anchor_weights = np.zeros((len(frame_indices), 2, 5), dtype=np.float32)
        local = self.to_object_numpy(baseline[:, :, TIPS])
        admitted = []
        # Separate each admitted interval: no anchor crosses a release boundary.
        for interval in phases["phase_intervals_inclusive"]:
            if interval.get("anchor_weight", 0) <= 0:
                continue
            first, last = interval["first"], interval["last"]
            for name in interval["eligible_sides"]:
                side = ("left", "right").index(name)
                sequence = local[first:last + 1, side][self.weights[first:last + 1, side] > 0]
                if len(sequence) < 3:
                    continue
                _, distance = closest_mesh_points(sequence.reshape(-1, 3), self.candidate["vertices"], self.candidate["faces"])
                distance = distance.reshape(len(sequence), 5)
                median_distance = np.median(distance, axis=0)
                fraction = (distance <= maximum_anchor_distance_m).mean(axis=0)
                eligible = np.flatnonzero((median_distance <= maximum_anchor_distance_m) & (fraction >= 0.5))
                chosen = sorted(eligible, key=lambda tip: median_distance[tip])[:maximum_anchors_per_hand]
                for tip in chosen:
                    median = np.median(sequence[:, tip], axis=0)[None]
                    anchor, _ = closest_mesh_points(median, self.candidate["vertices"], self.candidate["faces"])
                    self.anchor_targets[first:last + 1, side, tip] = anchor[0]
                    self.anchor_weights[first:last + 1, side, tip] = self.weights[first:last + 1, side]
                    admitted.append({"side": name, "landmark": int(TIPS[tip]), "phase_first": first, "phase_last": last,
                                     "anchor_object_local_metres": anchor[0].tolist(), "baseline_median_surface_distance_m": float(median_distance[tip]),
                                     "baseline_near_surface_fraction": float(fraction[tip])})
        self.admitted = admitted
        if not admitted:
            raise ValueError("No baseline fingertip is consistently near the hypothetical object surface; contact must abstain")
        self.rotation_t = torch.as_tensor(self.transform[:, :3, :3].copy(), device=device)
        self.translation_t = torch.as_tensor(self.transform[:, :3, 3].copy(), device=device)
        self.normals_t = torch.as_tensor(self.candidate["normals"], device=device)
        self.offsets_t = torch.as_tensor(self.candidate["offsets"], device=device)
        self.targets_t = torch.as_tensor(self.anchor_targets, device=device)
        self.weights_t = torch.as_tensor(self.anchor_weights, device=device)
        self.phase_t = torch.as_tensor(self.weights, device=device)
        self.before = self.diagnostics(baseline)

    def to_object_numpy(self, points):
        translation = self.transform[:, :3, 3]
        rotation = self.transform[:, :3, :3]
        return np.einsum("tski,tij->tskj", points - translation[:, None, None], rotation)

    def loss(self, hands):
        import torch

        tips = hands[:, :, TIPS.tolist()]
        local = torch.einsum("tski,tij->tskj", tips - self.translation_t[:, None, None], self.rotation_t)
        distance = torch.linalg.vector_norm(local - self.targets_t, dim=-1) / 0.02
        robust = torch.sqrt(1 + distance.square()) - 1
        anchor = (robust * self.weights_t).sum() / self.weights_t.sum().clamp_min(1)
        depths = []
        # Supporting planes give exact interior clearance for a convex object.
        # This is not Euclidean distance outside the mesh, nor volumetric PEN.
        for start in range(0, len(hands), 32):
            p = local[start:start + 32]
            signed = (p @ self.normals_t.T - self.offsets_t).max(dim=-1).values
            depths.append(torch.relu(-signed - 0.003))
        depth = torch.cat(depths)
        phase = self.phase_t[:, :, None]
        penetration = ((depth / 0.01).square() * phase).sum() / (phase.sum() * 5).clamp_min(1)
        return {"contact_anchor": anchor, "contact_tip_penetration_proxy": penetration,
                "contact_total": self.anchor_weight * anchor + self.penetration_weight * penetration}

    def diagnostics(self, hands: np.ndarray) -> dict[str, Any]:
        local = self.to_object_numpy(np.asarray(hands)[:, :, TIPS])
        distance = np.linalg.norm(local - self.anchor_targets, axis=-1)
        total_weight = self.anchor_weights.sum()
        depths = []
        for start in range(0, len(local), 32):
            signed = (local[start:start + 32] @ self.candidate["normals"].T - self.candidate["offsets"]).max(axis=-1)
            depths.append(np.maximum(-signed - 0.003, 0))
        depth = np.concatenate(depths)
        active = self.weights > 0
        return {"weighted_anchor_distance_mm": float((distance * self.anchor_weights).sum() / max(float(total_weight), 1e-8) * 1000),
                "tip_penetration_mean_mm_in_admitted_phases": float(depth[active].mean() * 1000) if active.any() else None,
                "tip_penetration_max_mm_in_admitted_phases": float(depth[active].max() * 1000) if active.any() else None,
                "admitted_anchors": self.admitted, "attraction_weight_outside_admitted_phases": 0,
                "max_object_step_m_in_admitted_phases": self.maximum_object_step,
                "max_object_rotation_step_rad_in_admitted_phases": self.maximum_object_rotation_step,
                "object_pose_usable_frames": int(self.candidate["pose_usable"].sum()),
                "usable_contact_phase_frames_per_hand": (self.weights > 0).sum(0).tolist(),
                "interpretation": "Contact agreement with an uncertain RGB-derived object hypothesis, not true contact or official PEN"}


def native_hand_vertex_indices(model, threshold=0.5) -> list[np.ndarray]:
    """Select real native hand-skin vertices from generic MHR LBS weights."""
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("hand-skin weight threshold must lie in (0,1]")
    names = list(model.character_torch.skeleton.joint_names)
    skin = model.character_torch.linear_blend_skinning
    joint = skin.skin_indices_flattened.detach().cpu().numpy()
    vertex = skin.vert_indices_flattened.detach().cpu().numpy()
    weight = skin.skin_weights_flattened.detach().cpu().numpy()
    out = []
    for name, first, last in (("l_wrist", 78, 101), ("r_wrist", 42, 65)):
        if names[first] != name or not names[last - 1].endswith("thumb_null"):
            raise ValueError("native MHR hand-skin joint range differs from the verified rig")
        assigned = np.where((joint >= first) & (joint < last), weight, 0)
        hand_weight = np.bincount(vertex, weights=assigned, minlength=18439)
        out.append(np.flatnonzero(hand_weight >= threshold))
    return out


def native_mesh_penetration_proxy(vertices_camera: np.ndarray, hand_vertices: list[np.ndarray],
                                  candidate: Mapping[str, np.ndarray], phase: np.ndarray) -> dict[str, Any]:
    """Penetration depth of actual native hand-skin vertices into convex mesh.

    Supporting-plane depths inside a closed convex object are exact point to
    boundary clearances. This samples vertices, not intersection volume or the
    organizer's normalized penetration metric.
    """
    v = np.asarray(vertices_camera)
    transform = candidate["object_to_camera"]
    normal, offset = candidate["normals"], candidate["offsets"]
    lower, upper = candidate["vertices"].min(0), candidate["vertices"].max(0)
    if v.ndim != 3 or v.shape[0] != len(transform) or v.shape[1:] != (18439, 3):
        raise ValueError("actual native vertices must have shape [T,18439,3]")
    if not np.isfinite(v).all() or np.asarray(phase).shape != (len(v), 2) or not np.isfinite(phase).all() or np.any((phase < 0) | (phase > 1)):
        raise ValueError("native vertices and phase weights must be finite and aligned")
    if len(hand_vertices) != 2 or any(np.asarray(indices).dtype.kind not in "iu" or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= 18439) for indices in hand_vertices):
        raise ValueError("hand skin indices must be two nonempty valid integer selections")
    count, mean, maximum = np.zeros((len(v), 2), np.int64), np.zeros((len(v), 2)), np.zeros((len(v), 2))
    for frame in range(len(v)):
        for side in range(2):
            if phase[frame, side] <= 0:
                continue
            local = (v[frame, hand_vertices[side]] - transform[frame, :3, 3]) @ transform[frame, :3, :3]
            possible = local[(local >= lower).all(1) & (local <= upper).all(1)]
            depth_parts = []
            for start in range(0, len(possible), 512):
                signed = (possible[start:start + 512] @ normal.T - offset).max(1)
                depth_parts.append(np.maximum(-signed, 0))
            depth = np.concatenate(depth_parts) if depth_parts else np.zeros(0)
            count[frame, side] = int((depth > 0).sum())
            mean[frame, side] = depth.sum() / max(1, len(hand_vertices[side]))
            maximum[frame, side] = depth.max() if len(depth) else 0
    report = {"interpretation": "Actual native hand-skin vertex depth inside the uncertain reconstructed convex object; not volume, hidden-reference error or official PEN",
              "hand_vertex_counts": [len(indices) for indices in hand_vertices]}
    for side, name in enumerate(("left", "right")):
        mask = phase[:, side] > 0
        report[name] = {"admitted_frames": int(mask.sum()), "mean_depth_over_hand_vertices_mm": float(mean[mask, side].mean() * 1000) if mask.any() else None,
                        "max_vertex_depth_mm": float(maximum[mask, side].max() * 1000) if mask.any() else None,
                        "mean_penetrating_vertex_fraction": float(count[mask, side].mean() / max(1, len(hand_vertices[side]))) if mask.any() else None}
    return {"report": report, "per_frame_penetrating_vertices": count, "per_frame_mean_depth_m": mean, "per_frame_max_depth_m": maximum}
