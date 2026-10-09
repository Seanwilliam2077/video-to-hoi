"""Topology-aware interaction refinement of saved native MHR predictions.

Track 1 episode 16 only. A dynamic reference interaction tetrahedralization
defines a shared Laplacian; it is not a fused material body or a collision
guarantee. Both refinement variants share observations, weights and budget.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

from real_trial import ROOT, tensor, project, project_np, local, sdf, decode, observations, measure
from synthetic_trial import CORNERS, EDGES, box_sdf
import roma

TEXTURE_FIT = '--texture' in sys.argv
OUT = ROOT / ('results/interaction-v3' if TEXTURE_FIT else 'results/interaction-v2')


def fps_indices(points, count):
    chosen = [int(np.argmin(np.linalg.norm(points - points.mean(0), axis=-1)))]
    distance = np.full(len(points), np.inf)
    for _ in range(min(count, len(points)) - 1):
        distance = np.minimum(distance, np.linalg.norm(points - points[chosen[-1]], axis=-1))
        chosen.append(int(np.argmax(distance)))
    return np.array(chosen)


def surface_samples(half):
    points = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for sign in [-1, 1]:
            for a in np.linspace(-1, 1, 5):
                for b in np.linspace(-1, 1, 5):
                    p = np.zeros(3)
                    p[axis] = sign
                    p[other] = [a, b]
                    points.append(p * half)
    return np.unique(np.round(points, 9), axis=0)


def interaction_graph(hand, objects):
    """Edges represent geometric neighborhoods, not asserted physical contacts."""
    vertices = np.concatenate([hand, np.broadcast_to(objects, (len(hand), *objects.shape))], axis=1)
    matrices, counts = [], []
    perturbation = np.random.default_rng(20261009).normal(0, 1e-8, vertices.shape[1:])
    for points in vertices:
        tetra = Delaunay(points + perturbation)
        pairs = np.concatenate([tetra.simplices[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
        distances = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=-1)
        # Remote tetrahedral edges do not reliably describe local interaction.
        pairs, distances = pairs[distances < .22], distances[distances < .22]
        weights = np.zeros((len(points), len(points)), np.float32)
        values = np.exp(-distances / .07)
        weights[pairs[:, 0], pairs[:, 1]] = values
        weights[pairs[:, 1], pairs[:, 0]] = values
        weights /= np.maximum(weights.sum(1, keepdims=True), 1e-8)
        matrices.append(np.eye(len(points), dtype=np.float32) - weights)
        counts.append(int(np.sum((pairs[:, 0] < hand.shape[1]) & (pairs[:, 1] >= hand.shape[1]))))
    return np.array(matrices), vertices, counts


def validation_tracks(images, k, r, half):
    """Texture tracks withheld from refinement; approximate source-side anchors."""
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    mask = np.zeros(gray[60].shape, np.uint8)
    # Visually inspected texture regions away from hands and the six fitted corners.
    cv2.fillPoly(mask, [np.array([[720, 628], [745, 615], [738, 643], [719, 655]], np.int32)], 255)
    seeds = cv2.goodFeaturesToTrack(gray[60], 20, .02, 5, mask=mask)
    assert seeds is not None and len(seeds) >= 5
    seeds = seeds[:, 0]
    rr, center = r['object_rotation'][60], r['object_translation'][60]
    rays = np.c_[seeds, np.ones(len(seeds))] @ np.linalg.inv(k).T
    directions = rays @ rr
    origin = -center @ rr
    safe = np.where(np.abs(directions) < 1e-9, 1e-9, directions)
    near = np.minimum((-half - origin) / safe, (half - origin) / safe).max(1)
    far = np.maximum((-half - origin) / safe, (half - origin) / safe).min(1)
    keep = (near > 0) & (far > near)
    anchors = origin + near[keep, None] * directions[keep]
    seeds = seeds[keep]
    assert len(seeds) >= 5
    tracks = np.zeros((120, len(seeds), 2), np.float32)
    valid = np.zeros((120, len(seeds)), bool)
    tracks[60], valid[60] = seeds, True
    for direction in [1, -1]:
        previous = 60
        for t in range(60 + direction, 120 if direction > 0 else -1, direction):
            p0 = tracks[previous, :, None]
            p1, status, error = cv2.calcOpticalFlowPyrLK(gray[previous], gray[t], p0, None, winSize=(21, 21), maxLevel=3)
            p2, back, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[previous], p1, None, winSize=(21, 21), maxLevel=3)
            tracks[t] = p1[:, 0]
            valid[t] = valid[previous] & (status[:, 0] > 0) & (back[:, 0] > 0) & (error[:, 0] < 25) & (np.linalg.norm(p2[:, 0] - p0[:, 0], axis=1) < 1)
            previous = t
    assert valid.sum() >= 300
    np.savez_compressed(OUT / 'validation-tracks.npz', anchors=anchors, image_points=tracks, valid=valid, seed_frame=150)
    overlay = images[60].copy()
    for point in seeds.astype(int):
        cv2.circle(overlay, tuple(point), 3, (0, 0, 255), -1)
    cv2.imwrite(str(OUT / 'validation-seeds.jpg'), overlay[550:700, 640:800])
    return anchors, tracks, valid


def optimize(label, initial, inputs, graph, hand_ids, contact_ids, collision_ids, model, base_metrics):
    canon, obs, confidence, half, k = inputs
    lap, reference_points, objects = graph
    base = tensor(initial['vertices'])
    native = tensor(initial['mhr_model_params'])
    shape, expr, cam = [tensor(initial[key]) for key in ['shape_params', 'expr_params', 'cam_t']]
    meta = json.loads((ROOT / 'results/real/metrics.json').read_text())
    controls = np.array(meta['optimized_native_control_ids'])
    scales = tensor(meta['bounded_native_control_scales'])
    basis = torch.zeros((len(controls), native.shape[1]), device='cuda')
    basis[torch.arange(len(controls)), tensor(controls).long()] = scales
    spline = tensor(CubicSpline(np.linspace(0, 119, 16), np.eye(16), bc_type='natural')(np.arange(120)))
    # Incremental rotation updates avoid smoothing absolute rotation-vector charts.
    center0, rotation0 = tensor(initial['object_translation']), tensor(initial['object_rotation'])
    dc = torch.nn.Parameter(torch.zeros((16, 3), device='cuda'))
    dr = torch.nn.Parameter(torch.zeros((16, 3), device='cuda'))
    dq = torch.nn.Parameter(torch.zeros((16, len(controls)), device='cuda'))
    optim = torch.optim.Adam([dc, dr, dq], lr=.003)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 300, eta_min=.00003)
    canon, obs, confidence, half, k, lap, reference_points, objects = map(tensor, [canon, obs, confidence, half, k, lap, reference_points, objects])
    target_lap = lap @ reference_points
    hand0 = base[:, hand_ids]
    # Contact eligibility is fixed from the initializer; solver cannot disable it.
    initial_local = local(base[:, contact_ids], rotation0, center0)
    contact_confidence = torch.exp(-sdf(initial_local, half).abs() / .05).clamp(min=.15).detach()
    sign = tensor([1, -1, -1])
    history = []
    best, best_score = None, float('inf')
    image_gate = base_metrics.get('fitting_gate_px', base_metrics['tracked_corner_reprojection_mean_px'] * 1.05)
    for iteration in range(301):
        optim.zero_grad()
        center = center0 + spline @ dc
        rotation = roma.rotvec_to_rotmat(spline @ dr) @ rotation0
        q = spline @ dq
        params = native + q @ basis
        verts, _ = model(shape, params, expr)
        verts = verts / 100 * sign + cam[:, None]
        projected = project(canon[None] @ rotation.transpose(1, 2) + center[:, None], k)
        image_error = torch.linalg.vector_norm(projected - obs, dim=-1)
        image = ((image_error / 3).square() * confidence).sum() / confidence.sum()
        contact_local = local(verts[:, contact_ids], rotation, center)
        # Zero signed distance permits tangential sliding; no persistent point pair.
        contact = ((sdf(contact_local, half) / .01).square() * contact_confidence).mean()
        collision_local = local(verts[:, collision_ids], rotation, center)
        inside = torch.relu(.0005 - sdf(collision_local, half)) / .01
        collision = inside.square().mean() + .1 * inside.max(dim=1).values.square().mean()
        joint = torch.cat([local(verts[:, hand_ids], rotation, center), objects.expand(120, -1, -1)], dim=1)
        graph_loss = ((lap @ joint - target_lap) / .025).square().mean()
        human = ((verts[:, collision_ids] - base[:, collision_ids]) / .02).square().mean()
        human_image = ((project(verts[:, hand_ids], k) - project(hand0, k)) / 3).square().mean()
        temporal = ((center[2:] - 2 * center[1:-1] + center[:-2]) / .005).square().mean()
        hand = verts[:, hand_ids]
        temporal += ((hand[2:] - 2 * hand[1:-1] + hand[:-2]) / .005).square().mean()
        control = q.square().mean()
        rotation_control = (spline @ dr).square().mean()
        graph_weight = .5 if label == 'interaction_mesh' else 0.
        loss = image + .5 * human + .25 * human_image + .2 * contact + collision + graph_weight * graph_loss + .04 * temporal + .05 * control + .1 * rotation_control
        with torch.no_grad():
            reprojection = float((image_error * confidence).sum() / confidence.sum())
            # Select using fitting evidence and loss only; withheld tracks are never used.
            if reprojection <= image_gate and float(loss) < best_score:
                best_score = float(loss)
                best = {'native': params.detach().cpu().numpy(), 'center': center.detach().cpu().numpy(), 'rotation': rotation.detach().cpu().numpy(), 'iteration': iteration, 'fit_reprojection_px': reprojection}
        if iteration % 50 == 0 or iteration == 300:
            row = {'iteration': iteration, 'loss': float(loss.detach()), 'reprojection_px': reprojection, 'graph_loss': float(graph_loss.detach()), 'contact_loss': float(contact.detach()), 'collision_loss': float(collision.detach())}
            history.append(row)
            print(label, row, flush=True)
        if iteration == 300:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([dc, dr, dq], 50)
        optim.step()
        schedule.step()
        with torch.no_grad():
            dq.clamp_(-.8, .8)
            dc.clamp_(-.35 if TEXTURE_FIT else -.06, .35 if TEXTURE_FIT else .06)
            dr.clamp_(-.7 if TEXTURE_FIT else -.4, .7 if TEXTURE_FIT else .4)
    assert best is not None
    best['vertices'] = decode(model, initial['shape_params'], best['native'], initial['expr_params'], initial['cam_t'])
    (OUT / f'{label}-optimization.json').write_text(json.dumps({'selected_iteration': best['iteration'], 'fit_gate_px': image_gate, 'history': history}, indent=2))
    return best


def diagnostics(r, validation, k, contact_ids):
    center, rotation, vertices = r['center'], r['rotation'], r['vertices']
    anchors, tracks, valid = validation
    predicted = project_np(anchors[None] @ rotation.transpose(0, 2, 1) + center[:, None], k)
    error = np.linalg.norm(predicted - tracks, axis=-1)
    velocity_error = np.linalg.norm(np.diff(predicted, axis=0) - np.diff(tracks, axis=0), axis=-1)
    pair_valid = valid[1:] & valid[:-1]
    relative = np.einsum('tpi,tij->tpj', vertices[:, contact_ids] - center[:, None], rotation)
    rotation_step = Rotation.from_matrix(rotation[1:] @ rotation[:-1].transpose(0, 2, 1)).as_rotvec()
    return {
        'withheld_texture_reprojection_mean_px': float(error[valid].mean()),
        'withheld_texture_velocity_error_mean_px_per_frame': float(velocity_error[pair_valid].mean()),
        'object_center_max_separation_mm': float(np.linalg.norm(center[:, None] - center[None], axis=-1).max() * 1000),
        'object_rotation_max_from_first_frame_deg': float(np.rad2deg(Rotation.from_matrix(rotation @ rotation[0].T).magnitude()).max()),
        'object_rotation_increment_change_mean_deg_per_frame2': float(np.linalg.norm(np.diff(rotation_step, axis=0), axis=-1).mean() * 180 / np.pi),
        'selected_hand_second_difference_rms_mm_per_frame2': float(np.sqrt(np.sum(np.diff(vertices[:, contact_ids], n=2, axis=0) ** 2, axis=-1).mean()) * 1000),
        'hand_motion_in_object_frame_rms_mm_per_frame': float(np.sqrt(np.sum(np.diff(relative, axis=0) ** 2, axis=-1).mean()) * 1000),
    }


def synthetic_graph_checks():
    half = np.array([.1, .085, .1])
    objects = surface_samples(half)
    hand = np.array([[-.12, -.04, -.04], [-.12, .04, -.04], [-.12, .04, .04], [-.12, -.04, .04]])
    lap, reference, counts = interaction_graph(hand[None], objects)
    moved = reference + np.array([.3, -.2, .1])
    assert np.allclose(lap @ moved, lap @ reference, atol=1e-7)
    slide_a = np.array([[.1, 0, 0]])
    slide_b = np.array([[.1, .03, 0]])
    assert np.max(np.abs(box_sdf(np.concatenate([slide_a, slide_b]), half))) < 1e-9
    assert counts[0] > 0
    assert np.max(np.abs(lap.sum(-1))) < 1e-6
    return ['joint graph has cross-object edges', 'Laplacian translation invariance', 'zero-distance contact permits surface sliding', 'row-normalized graph']


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    checks = synthetic_graph_checks()
    initial = dict(np.load(ROOT / 'results/real/matched_independent.npz'))
    old = dict(np.load(ROOT / 'results/real/attachment_collision.npz'))
    meta = json.loads((ROOT / 'results/real/metrics.json').read_text())
    contacts = np.array(meta['fixed_contact_vertex_ids'])
    half = initial['object_half_extents']
    k = np.load(ROOT / 'baseline/intrinsics.npy')
    images = [cv2.imread(str(ROOT / 'data/frames' / f'{i:06d}.jpg')) for i in range(90, 210)]
    baseline150 = dict(np.load(ROOT / 'baseline/000150.npz'))
    keys = baseline150['pred_keypoints_3d'] + baseline150['pred_cam_t']
    ids = []
    for indices in [[41, 28, 32, 36, 40], [62, 49, 53, 57, 61]]:
        palm = keys[indices].mean(0)
        candidates = np.flatnonzero(np.linalg.norm(initial['vertices'][60] - palm, axis=-1) < .105)
        ids.extend(candidates[fps_indices(initial['vertices'][60, candidates], 24)])
    hand_ids = np.unique(np.r_[ids, contacts])
    near = np.flatnonzero(np.linalg.norm(initial['vertices'][60] - initial['object_translation'][60], axis=-1) < .35)
    collision_ids = np.unique(np.r_[near, hand_ids])
    data = [dict(np.load(ROOT / 'baseline' / f'{i:06d}.npz')) for i in range(90, 210)]
    depth = np.median(np.stack([d['pred_keypoints_3d'][[41, 62], 2] + d['pred_cam_t'][2] for d in data]))
    canon, obs, confidence, fitted_half, _, _ = observations(images, k, depth, OUT)
    assert np.allclose(half, fitted_half)
    validation = validation_tracks(images, k, initial, half)
    original_inputs = (canon.copy(), obs.copy(), confidence.copy())
    fit_metrics = dict(meta['metrics']['matched_independent'])
    if TEXTURE_FIT:
        # Fixed index split. This is a development validation set: the earlier
        # v2 diagnostics informed the observation change, so it is not unseen.
        anchors, texture_tracks, texture_valid = validation
        training = np.arange(len(anchors)) % 2 == 0
        validation = (anchors[~training], texture_tracks[:, ~training], texture_valid[:, ~training])
        canon = np.concatenate([canon, anchors[training]])
        obs = np.concatenate([obs, texture_tracks[:, training]], axis=1)
        confidence = np.concatenate([confidence * .1, texture_valid[:, training].astype(float)], axis=1)
        frozen_points = canon @ initial['object_rotation'][60].T + initial['object_translation'][60]
        frozen_error = np.linalg.norm(project_np(frozen_points, k)[None] - obs, axis=-1)
        frozen_fit = float((frozen_error * confidence).sum() / confidence.sum())
        fit_metrics['fitting_gate_px'] = .9 * frozen_fit
        np.savez_compressed(OUT / 'texture-split.npz', training_indices=np.flatnonzero(training), validation_indices=np.flatnonzero(~training), fitting_gate_px=fit_metrics['fitting_gate_px'])
        print('Texture training/validation:', int(training.sum()), int((~training).sum()), 'fit gate vs frozen:', fit_metrics['fitting_gate_px'], flush=True)
    hand_relative = np.einsum('tpi,tij->tpj', initial['vertices'][:, hand_ids] - initial['object_translation'][:, None], initial['object_rotation'])
    objects = surface_samples(half)
    lap, reference, counts = interaction_graph(hand_relative, objects)
    np.savez_compressed(OUT / 'interaction-structure.npz', laplacian=lap, reference_vertices=reference, hand_vertex_ids=hand_ids, object_surface_samples=objects, cross_edge_counts=counts, original_frame_ids=np.arange(90, 210))
    model = torch.jit.load(str(ROOT / 'assets/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt')).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    def convert(d):
        return {'center': d['object_translation'], 'rotation': d['object_rotation'], 'vertices': d['vertices'], 'native': d['mhr_model_params']}
    results = {'independent': convert(initial), 'old_fixed_contact': convert(old)}
    frozen = convert(initial)
    frozen['center'] = np.repeat(initial['object_translation'][60:61], 120, axis=0)
    frozen['rotation'] = np.repeat(initial['object_rotation'][60:61], 120, axis=0)
    results['frozen_object_control'] = frozen
    for label in ['sliding_without_graph', 'interaction_mesh']:
        results[label] = optimize(label, initial, (canon, obs, confidence, half, k), (lap, reference, objects), hand_ids, contacts, collision_ids, model, fit_metrics)
    faces = np.load(ROOT / 'baseline/faces.npy')
    metrics = {}
    for name, r in results.items():
        values = measure(r['vertices'], r['rotation'], r['center'], half, np.array(meta['fixed_object_attachment_points']), contacts, faces, *original_inputs, k)
        fit_projection = project_np(canon[None] @ r['rotation'].transpose(0, 2, 1) + r['center'][:, None], k)
        values['fitting_observation_reprojection_weighted_mean_px'] = float((np.linalg.norm(fit_projection - obs, axis=-1) * confidence).sum() / confidence.sum())
        values.update(diagnostics(r, validation, k, contacts))
        metrics[name] = values
        if name in ['sliding_without_graph', 'interaction_mesh']:
            np.savez_compressed(OUT / f'{name}.npz', original_frame_ids=np.arange(90, 210), mhr_model_params=r['native'], shape_params=initial['shape_params'], expr_params=initial['expr_params'], cam_t=initial['cam_t'], object_rotation=r['rotation'], object_translation=r['center'], object_half_extents=half, vertices=r['vertices'])
    report = {'scope': 'Track 1 episode 16 frames 90-209; prediction diagnostics, no official score', 'methods': metrics, 'checks': checks, 'interaction_vertices': reference.shape[1], 'hand_samples': len(hand_ids), 'cross_edges_min_max': [min(counts), max(counts)], 'graph_reference': 'per-frame Delaunay neighborhoods of initial estimated hand/object geometry in object coordinates; uncertain, not ground truth', 'contact_model': 'initializer-gated surface distance; tangent motion allowed; no hard sticking or inferred friction force', 'validation': {'source': '20 or fewer independently tracked texture features seeded in a visually inspected face polygon at frame150, excluded from optimization', 'point_count': len(validation[0]), 'valid_observations': int(validation[2].sum()), 'limitations': '3D texture anchors derived from shared initializer at seed frame; optical flow may drift; not independent 3D truth'}, 'selection': 'minimum training objective satisfying mean fitting-corner reprojection <= 1.05 * matched independent; withheld texture points unused', 'reference_relative_ACC_available': False, 'continuous_collision_guarantee': False, 'seconds': time.monotonic() - started}
    if TEXTURE_FIT:
        report['validation']['source'] = 'Odd-index texture tracks excluded from v3 fitting; even indices used for fitting. The v2 review informed this development experiment.'
        report['selection'] = 'Minimum training objective with weighted fitting reprojection at least 10% better than frozen control; validation split unused for parameter or checkpoint selection.'
        report['observation_change'] = 'Six uncertain corner tracks weighted 0.1; seven texture tracks weighted 1; object correction bounds enlarged equally for both v3 variants.'
    (OUT / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({name: {k: v for k, v in m.items() if not isinstance(v, list)} for name, m in metrics.items()}, indent=2), flush=True)
    print('Completed', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
