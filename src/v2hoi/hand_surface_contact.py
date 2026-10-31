"""Exact selected native skin and explicitly hypothetical sliding contact priors.

The native decoder stays frozen. These constraints are conditional on an
RGB-derived convex object; no hidden pose, contact or physical scale is asserted.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from .hand_contact import native_hand_vertex_indices, phase_weights, validate_object_candidate


def distributed_vertex_selection(points: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    """Deterministic farthest-point coverage in the fixed native rest mesh."""
    points = np.asarray(points)
    candidates = np.asarray(candidates, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Rest points must be finite [V,3]")
    if (candidates.ndim != 1 or len(np.unique(candidates)) != len(candidates) or
            np.any(candidates < 0) or np.any(candidates >= len(points)) or
            isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(candidates)):
        raise ValueError("Require unique valid candidate vertices and a bounded integer sample count")
    candidates = np.sort(candidates)
    positions = points[candidates].astype(np.float64)
    first = int(np.argmax(((positions - positions.mean(0)) ** 2).sum(-1)))
    picked = [first]
    minimum = ((positions - positions[first]) ** 2).sum(-1)
    minimum[first] = -1
    for _ in range(1, count):
        index = int(np.argmax(minimum))
        picked.append(index)
        minimum = np.minimum(minimum, ((positions - positions[index]) ** 2).sum(-1))
        minimum[picked] = -1
    return np.sort(candidates[picked])


class NativeSelectedHandSurface:
    """Exact frozen-MHR correctives/LBS evaluated only for selected skin rows.

    Fixed identity rest positions are cached. The native sparse corrective
    feature layer is reused and only the final linear output rows are selected.
    A private copy of the native LBS module remaps selected vertex indices;
    neither the original model nor its parameters/buffers are modified.
    """
    def __init__(self, native_hands, shape: np.ndarray, scales: np.ndarray, *, count_per_hand=128):
        import torch

        self.model = native_hands.model
        self.character = self.model.character_torch
        self.device = native_hands.device
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.scales = torch.as_tensor(np.array(scales, dtype=np.float32, copy=True), device=self.device)
        self.shape = torch.as_tensor(np.array(shape, dtype=np.float32, copy=True), device=self.device)
        with torch.no_grad():
            rest = self.character.blend_shape(self.shape[None])[0]
        candidates = native_hand_vertex_indices(self.model)
        rest_numpy = rest.detach().cpu().numpy()
        self.hand_vertex_ids = [distributed_vertex_selection(rest_numpy, group, count_per_hand) for group in candidates]
        selected = np.sort(np.concatenate(self.hand_vertex_ids))
        if len(np.unique(selected)) != 2 * count_per_hand:
            raise ValueError("Native hand skin selections must be disjoint")
        self.vertex_ids = selected
        self.hand_output_indices = [np.searchsorted(selected, group) for group in self.hand_vertex_ids]
        ids = torch.as_tensor(selected, device=self.device)
        self.ids_t = ids
        output_rows = (ids[:, None] * 3 + torch.arange(3, device=self.device)[None]).reshape(-1)
        self.correctives = self.model.pose_correctives_model
        self.first = getattr(self.correctives.pose_dirs_predictor, "0")
        last = getattr(self.correctives.pose_dirs_predictor, "2")
        self.output_weight = last.weight[output_rows].detach().contiguous()
        self.output_bias = None if last.bias is None else last.bias[output_rows].detach().contiguous()
        skin = self.character.linear_blend_skinning
        native_ids = skin.vert_indices_flattened.detach().cpu().numpy()
        keep = np.isin(native_ids, selected)
        keep_t = torch.as_tensor(keep, device=self.device)
        self.skin = copy.deepcopy(skin)
        self.skin.num_vertices = len(selected)
        self.skin.skin_indices_flattened = skin.skin_indices_flattened[keep_t].clone()
        self.skin.skin_weights_flattened = skin.skin_weights_flattened[keep_t].clone()
        self.skin.vert_indices_flattened = torch.as_tensor(np.searchsorted(selected, native_ids[keep]), device=self.device)
        self.rest = rest[ids].detach().clone()
        self.count_per_hand = count_per_hand

    def __call__(self, pose):
        import torch

        parameters = torch.cat([pose, self.scales[None].expand(len(pose), -1), pose.new_zeros(len(pose), 45)], 1)
        joint_parameters = self.character.model_parameters_to_joint_parameters(parameters)
        state = self.character.joint_parameters_to_skeleton_state(joint_parameters)
        features = self.correctives._pose_features_from_joint_params(joint_parameters)
        hidden = torch.relu(self.first(features))
        offsets = torch.nn.functional.linear(hidden, self.output_weight, self.output_bias).reshape(len(pose), len(self.vertex_ids), 3)
        vertices = self.skin(state, self.rest[None] + offsets)
        return vertices * vertices.new_tensor([0.01, -0.01, -0.01])

    def verify_full_forward(self, pose: np.ndarray, positions=(0, 75, 120, 210, 250, 287, 359), *, gradient_positions=(75, 120, 250)):
        import torch

        errors = []
        with torch.no_grad():
            for start in range(0, len(positions), 4):
                sample = torch.as_tensor(np.array(pose)[list(positions)[start:start + 4]], device=self.device)
                selected = self(sample)
                params = torch.cat([sample, self.scales[None].expand(len(sample), -1)], 1)
                full, _ = self.model(self.shape[None].expand(len(sample), -1).contiguous(), params, sample.new_zeros(len(sample), 72), True)
                expected = full[:, self.ids_t] * full.new_tensor([0.01, -0.01, -0.01])
                errors.append(float((selected - expected).abs().max()))
        sample = torch.as_tensor(np.array(pose)[list(gradient_positions)], device=self.device).clone().requires_grad_(True)
        selected = self(sample)
        selected_gradient = torch.autograd.grad(selected.square().mean(), sample)[0]
        params = torch.cat([sample, self.scales[None].expand(len(sample), -1)], 1)
        full, _ = self.model(self.shape[None].expand(len(sample), -1).contiguous(), params, sample.new_zeros(len(sample), 72), True)
        expected = full[:, self.ids_t] * full.new_tensor([0.01, -0.01, -0.01])
        full_gradient = torch.autograd.grad(expected.square().mean(), sample)[0]
        gradient_error = float((selected_gradient - full_gradient).abs().max())
        if max(errors) > 1e-6 or gradient_error > 1e-6 or not bool(torch.isfinite(selected_gradient).all()):
            raise ValueError("Selected native hand skin does not match full native positions/gradients")
        return {"sample_positions": list(positions), "gradient_positions": list(gradient_positions),
                "max_absolute_vertex_difference_metres": max(errors),
                "max_absolute_control_gradient_difference": gradient_error,
                "finite_gradients": True, "selected_vertices_per_hand": self.count_per_hand,
                "native_hand_vertex_ids": [group.tolist() for group in self.hand_vertex_ids],
                "sampling": "Deterministic farthest-point selection in the fixed native identity rest mesh"}


def convex_far_ray_depth(points, rotation, translation, normals, offsets):
    """Convex ray far-depth and intersection validity, differentiable on a face.

    A ray is fixed by each query's current camera projection. No near/far target
    is invented if that ray misses the reconstructed object.
    """
    import torch

    ray = points / points[..., 2:].clamp_min(1e-6)
    directions = torch.einsum("tpi,tij->tpj", ray, rotation)
    origin = torch.einsum("ti,tij->tj", -translation, rotation)
    denominator = directions @ normals.T
    numerator = offsets[None, :] - origin @ normals.T
    safe = torch.where(denominator.abs() > 1e-8, denominator, torch.ones_like(denominator))
    ratio = numerator[:, None, :] / safe
    lower = torch.where(denominator < -1e-8, ratio, ratio.new_full((), -float("inf"))).amax(-1)
    upper = torch.where(denominator > 1e-8, ratio, ratio.new_full((), float("inf"))).amin(-1)
    parallel_valid = ~((denominator.abs() <= 1e-8) & (numerator[:, None, :] < 0)).any(-1)
    valid = parallel_valid & torch.isfinite(upper) & (upper >= lower.clamp_min(0)) & (points[..., 2] > 0)
    return torch.where(valid, upper, torch.zeros_like(upper)), valid


class NativeHandSurfaceContact:
    """Frozen sampled-skin collision, sliding surface attraction and palm order."""
    def __init__(self, candidate: Mapping[str, Any], phases: Mapping[str, Any],
                 surface: NativeSelectedHandSurface, baseline_pose: np.ndarray,
                 baseline_hands: np.ndarray, frame_indices: np.ndarray,
                 configuration: Mapping[str, Any]):
        import torch

        self.surface = surface
        self.configuration = dict(configuration)
        self.order_ray_mode = self.configuration.get("order_ray_mode", "current_query")
        if self.order_ray_mode not in ("current_query", "fixed_shared_seed"):
            raise ValueError("Surface order ray mode must be current_query or fixed_shared_seed")
        self.candidate = validate_object_candidate(candidate, frame_indices)
        self.weights = phase_weights(phases, len(frame_indices)) * self.candidate["pose_usable"][:, None]
        if not self.weights.any():
            raise ValueError("Surface contact needs an admitted reliable manual grasp interval")
        for name in ("penetration_weight", "surface_weight", "far_order_weight"):
            value = self.configuration[name]
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError("Surface contact weights must be finite and nonnegative")
        if self.configuration.get("method") != "native_hand_surface_contact.v1":
            raise ValueError("Surface contact requires an explicit method schema")
        if (self.configuration.get("contact_patch_vertices_per_hand") != 16 or
                self.configuration.get("sample_vertices_per_hand") != 128 or surface.count_per_hand != 128):
            raise ValueError("This frozen experimental design uses 128 sampled skin / 16 patch vertices per hand")
        self.device = surface.device
        self.rotation = torch.as_tensor(self.candidate["object_to_camera"][:, :3, :3].copy(), device=self.device)
        self.translation = torch.as_tensor(self.candidate["object_to_camera"][:, :3, 3].copy(), device=self.device)
        self.normals = torch.as_tensor(self.candidate["normals"].copy(), device=self.device)
        self.offsets = torch.as_tensor(self.candidate["offsets"].copy(), device=self.device)
        self.phase = torch.as_tensor(self.weights, device=self.device)
        seed_hands = torch.as_tensor(np.array(baseline_hands, dtype=np.float32, copy=True), device=self.device)
        seed_left = seed_hands[:, 0]
        seed_query = torch.stack([seed_left[:, 0], seed_left[:, [5, 9, 13, 17]].mean(1)], 1)
        with torch.no_grad():
            self.reference_far, self.reference_ray_valid = convex_far_ray_depth(
                seed_query, self.rotation, self.translation, self.normals, self.offsets)
        self.active_ids = np.flatnonzero((self.weights > 0).any(1))
        self.active_t = torch.as_tensor(self.active_ids, device=self.device)
        self.hand_indices = [torch.as_tensor(group, device=self.device) for group in surface.hand_output_indices]
        self.patch_output_indices = []
        self.patch_native_vertex_ids = []
        baseline = torch.as_tensor(np.array(baseline_pose, dtype=np.float32, copy=True), device=self.device)
        with torch.no_grad():
            signed = []
            for start in range(0, len(self.active_ids), 8):
                ids = self.active_t[start:start + 8]
                vertices = self.surface(baseline[ids])
                local = torch.einsum("tvi,tij->tvj", vertices - self.translation[ids, None], self.rotation[ids])
                signed.append((local @ self.normals.T - self.offsets).amax(-1).cpu().numpy())
            signed = np.concatenate(signed)
        for side in range(2):
            training = ((np.asarray(frame_indices)[self.active_ids] % 5) != 0) & (self.weights[self.active_ids, side] > 0)
            if not training.any():
                raise ValueError("Surface patch selection requires training-only admitted frames per hand")
            group = surface.hand_output_indices[side]
            ranking = np.median(np.abs(signed[training][:, group]), axis=0)
            chosen = np.argsort(ranking, kind="stable")[:16]
            self.patch_output_indices.append(torch.as_tensor(group[chosen], device=self.device))
            self.patch_native_vertex_ids.append(surface.hand_vertex_ids[side][chosen].copy())
        self.before = self.diagnostics(np.asarray(baseline_pose), np.asarray(baseline_hands))

    def _terms(self, pose, joints):
        import torch

        signed_chunks = []
        order_chunks, order_valid_chunks, current_valid_chunks, reference_valid_chunks = [], [], [], []
        for start in range(0, len(self.active_ids), 8):
            ids = self.active_t[start:start + 8]
            vertices = self.surface(pose[ids])
            local = torch.einsum("tvi,tij->tvj", vertices - self.translation[ids, None], self.rotation[ids])
            signed_chunks.append((local @ self.normals.T - self.offsets).amax(-1))
            if joints is not None:
                left = joints[ids, 0]
                query = torch.stack([left[:, 0], left[:, [5, 9, 13, 17]].mean(1)], 1)
                far, valid = convex_far_ray_depth(query, self.rotation[ids], self.translation[ids], self.normals, self.offsets)
                current_valid_chunks.append(valid)
                reference_valid_chunks.append(self.reference_ray_valid[ids])
                if self.order_ray_mode == "fixed_shared_seed":
                    far, valid = self.reference_far[ids], self.reference_ray_valid[ids]
                order_chunks.append(torch.relu(far + .003 - query[..., 2]) * valid.to(query.dtype))
                order_valid_chunks.append(valid)
        signed = torch.cat(signed_chunks)
        phase = self.phase[self.active_t]
        penetration_parts, surface_parts = [], []
        for side, group in enumerate(self.hand_indices):
            weights = phase[:, side]
            inside = torch.relu(-signed[:, group] - .003) / .01
            mean = (inside.square().mean(1) * weights).sum() / weights.sum().clamp_min(1)
            worst = (inside.square().amax(1) * weights).amax()
            penetration_parts.append(mean + .25 * worst)
            distance = signed[:, self.patch_output_indices[side]].abs() / .02
            robust = torch.sqrt(1 + distance.square()) - 1
            surface_parts.append((robust.mean(1) * weights).sum() / weights.sum().clamp_min(1))
        penetration = torch.stack(penetration_parts).mean()
        attraction = torch.stack(surface_parts).mean()
        order = pose.new_zeros(())
        order_valid_fraction = pose.new_zeros(())
        current_valid_fraction = pose.new_zeros(())
        reference_valid_fraction = pose.new_zeros(())
        if joints is not None:
            gaps = torch.cat(order_chunks)
            valid = torch.cat(order_valid_chunks)
            weights = phase[:, 0, None] * valid.to(phase.dtype)
            order = ((gaps / .02).square() * weights).sum() / weights.sum().clamp_min(1)
            order_valid_fraction = valid.to(phase.dtype).mean()
            current_valid_fraction = torch.cat(current_valid_chunks).to(phase.dtype).mean()
            reference_valid_fraction = torch.cat(reference_valid_chunks).to(phase.dtype).mean()
        total = (self.configuration["penetration_weight"] * penetration +
                 self.configuration["surface_weight"] * attraction +
                 self.configuration["far_order_weight"] * order)
        return {"surface_skin_penetration": penetration, "surface_sliding_attraction": attraction,
                "surface_left_wrist_palm_order": order, "surface_order_ray_intersection_fraction": order_valid_fraction,
                "surface_order_current_ray_intersection_fraction": current_valid_fraction,
                "surface_order_reference_ray_intersection_fraction": reference_valid_fraction,
                "surface_contact_total": total}

    def loss(self, pose, joints):
        return self._terms(pose, joints)

    def diagnostics(self, pose, joints):
        import torch

        value = torch.as_tensor(np.array(pose, dtype=np.float32, copy=True), device=self.device)
        hand = None if joints is None else torch.as_tensor(np.array(joints, dtype=np.float32, copy=True), device=self.device)
        with torch.no_grad():
            terms = {key: float(item) for key, item in self._terms(value, hand).items()}
        return {"method": "native_hand_surface_contact.v1", "configuration": self.configuration,
                "terms": terms, "native_hand_vertex_ids": [group.tolist() for group in self.surface.hand_vertex_ids],
                "native_patch_vertex_ids": [group.tolist() for group in self.patch_native_vertex_ids],
                "phase_weights": self.weights.tolist(), "skin_vertices_per_hand": 128,
                "surface_patch_vertices_per_hand": 16,
                "order_ray_mode": self.order_ray_mode,
                "patch_selection": "Frozen native skin IDs with lowest median absolute convex clearance in the shared seed, using only stride-5-excluded admitted reliable manual grasp frames",
                "surface_attraction_semantics": "No fixed object-local 3D target. Absolute maximum supporting-plane signed clearance permits tangential sliding. It is exact interior clearance but an outside Euclidean distance lower bound.",
                "penetration_semantics": "Actual selected native hand skin; relu(-signed_clearance-0.003m)^2/(0.01m)^2; per hand phase-weighted point mean plus0.25 maximum phase-weighted per-frame point maximum; average two hands.",
                "order_semantics": "Left wrist and mean MCP landmarks[5,9,13,17] only; convex ray far intersection+0.003m, relu(depth_gap/0.02m)^2; phase-weighted valid-ray mean. fixed_shared_seed freezes ray depths/validity at the common initializer so moving rays off-object cannot remove this prior. Current and reference ray coverage remain separate diagnostics. No ordering constraint on all fingers.",
                "interpretation": "Conditional geometric priors against an uncertain Track 1 object estimate, not observed hidden pose/contact, true scale or official scores"}
