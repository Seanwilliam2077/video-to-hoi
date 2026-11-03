"""Unattended bpy-mesh reconstruction from source observations and long tracks.

FoundationPose supplies keyframe hypotheses only. CoTracker3 source tracks,
silhouette/depth observations and differentiable physical constraints jointly
fit all original-frame poses. No diffusion observations or old object poses.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from v2hoi.native import native_environment, sha256
from v2hoi.temporal_object import automatic_roi, select_keyframes, select_pose_path, automatic_contact_schedule, motion_diagnostics, tracking_windows, bpy_asset_paths, initialize_scale, spatially_spread_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=Path('third_party/CARI4D_MHR_Win/data/v2d-challenge/track_1'))
    parser.add_argument('--code', type=Path, default=Path('third_party/cotracker3'))
    parser.add_argument('--checkpoint', type=Path, default=Path('weights/cotracker3/scaled_offline.pth'))
    parser.add_argument('--blender', type=Path, default=Path('C:/Program Files/Blender Foundation/Blender 4.2/blender.exe'))
    parser.add_argument('--keyframes', type=int, default=12)
    parser.add_argument('--steps', type=int, default=700)
    parser.add_argument('--tracking-window', type=int, default=128)
    parser.add_argument('--points-per-keyframe', type=int, default=12)
    parser.add_argument('--refinement', choices=('legacy', 'bundle'), default='bundle')
    parser.add_argument('--initialization', choices=('foundationpose', 'mask-proposals'), default='foundationpose',
        help='Mask translation proposals are experimental and can worsen rotation or human/object depth agreement')
    parser.add_argument('--contact-profile', type=Path,
        help='Explicit optional source-receipted sparse contact assistance; not unattended mode')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.keyframes < 1 or args.steps < 1 or args.tracking_window < 8 or args.points_per_keyframe < 4:
        parser.error('Require positive keyframes/steps, window >= 8, and at least four points per keyframe')
    if args.output.exists():
        raise FileExistsError('Preserve candidates; choose a new output directory')
    if not args.checkpoint.exists() or not (args.code/'source-receipt.json').exists():
        raise FileNotFoundError('Run .venv/Scripts/python.exe tools/windows/prepare_temporal_object.py first')
    if not args.worker:
        env = native_environment(Path('third_party/CARI4D_MHR_Win').resolve(), Path('weights/cari4d').resolve())
        env['PYTHONPATH'] = str(args.code.resolve())+os.pathsep+env['PYTHONPATH']
        env['PATH'] = 'C:/Tools/ffmpeg/bin'+os.pathsep+env['PATH']
        env['IMAGEIO_FFMPEG_EXE'] = 'C:/Tools/ffmpeg/bin/ffmpeg.exe'
        env['HF_HUB_OFFLINE'] = '1'
        subprocess.run([sys.executable, '-m', 'v2hoi.temporal_object_runtime', *sys.argv[1:], '--worker'], env=env, check=True)
        subprocess.run([sys.executable, 'tools/windows/verify_physics_run.py', str(args.output)], check=True)
        subprocess.run([sys.executable, 'tools/windows/review_temporal_object.py', str(args.output)], check=True)
        return
    run(args)


def acquire(clip, root, output):
    import av
    import cv2
    from prep.mhr_export_utils import read_mask
    from v2hoi.physics_dynamic_runtime import observations
    export = root/'export'/f'episode_{clip.episode:06d}'
    small, quality = [], []
    with av.open(str(clip.video)) as reader:
        for i, decoded in enumerate(reader.decode(video=0)):
            obj = read_mask(export, 'object', 0, f'{i:06d}').astype('uint8')
            human = read_mask(export, 'human', 0, f'{i:06d}').astype('uint8')
            visible = obj&~human
            reduced = cv2.resize(visible, (384, round(384*clip.height/clip.width)), interpolation=cv2.INTER_NEAREST)
            small.append(reduced)
            rgb = cv2.resize(decoded.to_ndarray(format='rgb24'), reduced.shape[::-1])
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            sharp = float(laplacian[reduced.astype(bool)].var()) if reduced.sum() > 16 else 0.
            quality.append(float(reduced.sum()/max(1, cv2.resize(obj, reduced.shape[::-1], interpolation=cv2.INTER_NEAREST).sum())
                                 *np.log1p(sharp)*np.sqrt(reduced.sum())))
    roi = np.array(automatic_roi(np.asarray(small)), dtype=float)*clip.width/384
    roi = np.rint(roi).astype(int).tolist()
    spec = {'observation_roi': roi, 'motion_anchors': []}
    data = observations(clip, root, spec, output)
    return data, np.asarray(quality), roi


def keyframe_candidates(mesh, images, masks, humans, depths, camera, ids, normal, offset, fps, track_data, output,
                        mask_proposals=False, contact=None):
    import torch
    import Utils
    from prep.run_foundationpose_mhr_export import _pose_estimator
    from v2hoi.object_refine_runtime import ObservationRenderer
    from v2hoi.pose_review import orientation_hypotheses
    from v2hoi.physics_fp_runtime import grounded_pose
    from v2hoi.temporal_object import mask_translation_hypotheses
    Utils.set_seed(0)
    logging.getLogger().setLevel(logging.WARNING)
    estimator = _pose_estimator(mesh.copy(), output/'foundationpose-debug')
    transform = estimator.get_tf_to_centered_mesh().cpu().numpy()
    renderer = ObservationRenderer(mesh)
    candidates, costs, rows = [], [], []
    center = mesh.bounds.mean(0)
    for i in ids:
        depth = depths[i].copy()
        depth[humans[i].astype(bool)] = 0.
        visible = masks[i].astype(bool)&~humans[i].astype(bool)
        if (visible&(depth > .001)).sum() < 30:
            raise ValueError(f'Insufficient visible metric initialization at original frame {i}')
        with torch.inference_mode():
            best = estimator.register(K=camera, rgb=images[i], depth=depth, ob_mask=visible, iteration=5)
        ranked = estimator.poses.detach().cpu().numpy()[:24]@transform
        # Quarter turns are hypotheses, never declared exact object symmetries.
        variants = orientation_hypotheses(best, center)
        hypotheses = np.concatenate([ranked, variants])
        # A grounded alternative is allowed, not forced on a carried object.
        if normal is not None:
            supported = np.array([grounded_pose(p, normal, offset) for p in hypotheses[:8]])
            hypotheses = np.concatenate([hypotheses, supported])
        # Dark surfaces can produce metre-scale monocular depth errors. Keep
        # FP proposals, and add image-derived translations at the same rotations.
        # Full perspective silhouette/depth costs and long tracks choose between
        # both sets; mask moments alone never become the final object trajectory.
        aligned = hypotheses.copy()
        yy, xx = np.where(visible)
        observed_center = np.array([xx.mean(), yy.mean()])
        care = ~humans[i].astype(bool)
        for _ in range(2 if mask_proposals else 0):
            areas, centroids = [], []
            for start in range(0, len(aligned), 16):
                batch = aligned[start:start+16]
                with torch.inference_mode():
                    _, alpha, _ = renderer.render(torch.tensor(batch[:, :3, :3], dtype=torch.float32, device='cuda'),
                        torch.tensor(batch[:, :3, 3], dtype=torch.float32, device='cuda'), 1., torch.zeros(3, device='cuda'),
                        np.repeat(camera[None], len(batch), axis=0), masks.shape[1:])
                for predicted in (alpha[..., 0].cpu().numpy() > .5)&care:
                    py, px = np.where(predicted)
                    areas.append(len(px))
                    centroids.append([px.mean(), py.mean()] if len(px) else [np.nan, np.nan])
            aligned = mask_translation_hypotheses(aligned, center, camera, visible.sum(),
                observed_center, areas, centroids)
        if mask_proposals:
            hypotheses = np.concatenate([hypotheses, aligned])
        scores = []
        for start in range(0, len(hypotheses), 16):
            batch = hypotheses[start:start+16]
            with torch.inference_mode():
                _, alpha, z = renderer.render(torch.tensor(batch[:, :3, :3], dtype=torch.float32, device='cuda'),
                    torch.tensor(batch[:, :3, 3], dtype=torch.float32, device='cuda'), 1., torch.zeros(3, device='cuda'),
                    np.repeat(camera[None], len(batch), axis=0), masks.shape[1:])
            predicted = alpha[..., 0].cpu().numpy() > .5
            care = ~humans[i].astype(bool)
            overlap = (predicted&visible).sum((1, 2))/np.maximum(1, ((predicted|visible)&care).sum((1, 2)))
            predicted_depth = z[..., 0].cpu().numpy()
            errors = []
            for j in range(len(batch)):
                supported = predicted[j]&visible&(depth > .001)
                errors.append(np.median(np.minimum(np.abs(predicted_depth[j][supported]-depth[supported]), .2))/.2 if supported.sum() >= 20 else 1.)
            scores.extend(1.-overlap+.15*np.array(errors))
        costs.append(np.asarray(scores)); candidates.append(hypotheses)
        rows.append({'original_frame_id': int(i), 'candidates': len(hypotheses), 'best_observation_cost': float(min(scores)),
            'source_mask_translation_proposals': len(aligned) if mask_proposals else 0, 'authored_translation': False})
        print(f'TEMPORAL_KEYFRAME frame={i} candidates={len(hypotheses)} cost={min(scores):.3f}', flush=True)
    del estimator, renderer
    torch.cuda.empty_cache()
    candidates, costs = np.asarray(candidates), np.asarray(costs)
    # A temporal bin is not a reliable anchor when every candidate disagrees
    # with its visible image. Bridge it from supported neighbours, retaining its
    # source point queries and every original output frame for joint refinement.
    admitted = costs.min(1) <= .7
    image_costs = costs.copy()
    if contact is not None:
        from v2hoi.sparse_contact import candidate_contact_cost
        for index, frame in enumerate(ids):
            costs[index] += candidate_contact_cost(candidates[index], mesh.vertices, contact, frame)
    if admitted.sum() < min(2, len(ids)):
        raise ValueError('Insufficient image-supported keyframes for temporal initialization')
    active = np.flatnonzero(admitted)
    edges = cross_frame_costs(mesh, candidates[active], camera, ids[active], track_data)
    path = select_pose_path(candidates[active], costs[active], ids[active], float(np.linalg.norm(mesh.extents)), fps, edges)
    from v2hoi.image_motion import interpolate_poses
    selected = interpolate_poses(ids[active], candidates[active, path], int(ids[-1])+1)[ids]
    chosen = np.full(len(ids), -1, dtype=int)
    chosen[active] = path
    for row, accepted in zip(rows, admitted):
        row.update(admitted_to_pose_graph=bool(accepted),
            initialization_source='observed_candidate' if accepted else 'neighbour_interpolation_before_joint_fit')
    np.savez(output/'keyframes.npz', candidates=candidates, costs=costs, selected_branches=chosen,
             selected_poses=selected, original_frame_ids=ids, cross_frame_observation_costs=edges,
             pose_graph_original_frame_ids=ids[active], pose_graph_candidate_indices=active,
             image_observation_costs=image_costs)
    return selected, rows


def cross_frame_costs(mesh, candidates, camera, ids, track_data):
    """Observed long tracks choose branches before mesh attachments are frozen."""
    from v2hoi.image_motion import surface_points
    sampled, tracks, visible, query_frames, query_pixels, _ = track_data
    count = candidates.shape[1]
    edges = np.zeros((len(ids)-1, count, count))
    for edge, (left, right) in enumerate(zip(ids[:-1], ids[1:])):
        terms = []
        shared = np.flatnonzero(visible[np.searchsorted(sampled, left)]&visible[np.searchsorted(sampled, right)])
        if len(shared) > 24:
            shared = shared[spatially_spread_indices(tracks[np.searchsorted(sampled, left), shared], 24)]
        for source, target, reverse in ((left, right, False), (right, left, True)):
            q = shared
            if len(q) < 4:
                continue
            source_index, target_index = np.searchsorted(ids, [source, target])
            observations = tracks[np.searchsorted(sampled, target), q]
            table = np.zeros((count, count))
            for a in range(count):
                local = surface_points(tracks[np.searchsorted(sampled, source), q], camera,
                                       candidates[source_index, a], mesh.vertices, mesh.faces)
                valid = np.isfinite(local).all(1)
                if valid.sum() < 4:
                    table[a] = .5
                    continue
                xyz = local[valid]@candidates[target_index, :, :3, :3].transpose(0, 2, 1)+candidates[target_index, :, None, :3, 3]
                homogeneous = xyz@camera.T
                predicted = homogeneous[..., :2]/np.maximum(homogeneous[..., 2:], .1)
                error = np.linalg.norm(predicted-observations[valid], axis=-1)/5.
                table[a] = .12*np.log1p(error**2).mean(1)
            terms.append(table.T if reverse else table)
        if terms:
            edges[edge] = np.mean(terms, axis=0)
    return edges


def long_tracks(images, masks, humans, ids, checkpoint, output, window=128, points_per_keyframe=8):
    import cv2
    import torch
    from cotracker.predictor import CoTrackerPredictor
    # Source frames are sampled at 15 Hz for memory; estimates/observations are
    # mapped back to every original frame without claiming interpolated visibility.
    sampled = np.unique(np.r_[np.arange(0, len(images), 2), ids, len(images)-1])
    queries = []
    for i in ids:
        support = cv2.erode((masks[i].astype(bool)&~humans[i].astype(bool)).astype('uint8'), np.ones((3, 3), 'uint8'))
        gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
        points = cv2.goodFeaturesToTrack(gray, maxCorners=max(96, points_per_keyframe*8), qualityLevel=.01, minDistance=4, mask=support)
        if points is None:
            continue
        points = points[:, 0]
        points = points[spatially_spread_indices(points, points_per_keyframe)]
        for point in points:
            queries.append([int(np.searchsorted(sampled, i)), *point])
    if len(queries) < 8:
        raise ValueError('No adequate automatically sampled source object points')
    model = CoTrackerPredictor(checkpoint=str(checkpoint.resolve()), offline=True).to('cuda').eval()
    torch.cuda.reset_peak_memory_stats()
    queries = np.asarray(queries)
    tracks = np.zeros((len(sampled), len(queries), 2), dtype='float32')
    visibility = np.zeros((len(sampled), len(queries)), dtype=bool)
    ambiguity = np.zeros_like(visibility)
    hops = np.full_like(visibility, 10000, dtype=int)
    windows = tracking_windows(len(sampled), window)
    calls, disagreements = 0, 0
    for start, stop in windows+windows[::-1]:
        local_queries, global_ids, query_hops = [], [], []
        for q, (source_t, x, y) in enumerate(queries):
            if start <= source_t < stop:
                local_queries.append([source_t-start, x, y]); global_ids.append(q); query_hops.append(0)
            else:
                known = np.flatnonzero(visibility[start:stop, q])+start
                if len(known):
                    seed = int(known[np.argmin(hops[known, q]*1000+np.abs(known-source_t))])
                    local_queries.append([seed-start, *tracks[seed, q]]); global_ids.append(q); query_hops.append(hops[seed, q]+1)
        if not local_queries:
            continue
        video = torch.tensor(images[sampled[start:stop]], device='cuda', dtype=torch.float32).permute(0, 3, 1, 2)[None]
        for first in range(0, len(local_queries), 96):
            batch_ids = np.asarray(global_ids[first:first+96])
            batch_hops = np.asarray(query_hops[first:first+96])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                predicted, seen = model(video, queries=torch.tensor(local_queries[first:first+96], dtype=torch.float32, device='cuda')[None], backward_tracking=True)
            predicted = predicted[0].float().cpu().numpy(); seen = seen[0].cpu().numpy().astype(bool)
            calls += 1
            for j, frame in enumerate(sampled[start:stop]):
                xy = np.rint(predicted[j]).astype(int)
                inside = (xy[:, 0] >= 0)&(xy[:, 0] < images.shape[2])&(xy[:, 1] >= 0)&(xy[:, 1] < images.shape[1])
                xy[:, 0] = np.clip(xy[:, 0], 0, images.shape[2]-1); xy[:, 1] = np.clip(xy[:, 1], 0, images.shape[1]-1)
                visible_now = seen[j]&inside&masks[frame, xy[:, 1], xy[:, 0]].astype(bool)&~humans[frame, xy[:, 1], xy[:, 0]].astype(bool)
                old_visible = visibility[start+j, batch_ids]
                old_hops = hops[start+j, batch_ids]
                conflict = old_visible&visible_now&(np.linalg.norm(tracks[start+j, batch_ids]-predicted[j], axis=1) > 4.)&(old_hops == batch_hops)
                disagreements += int(conflict.sum())
                better = batch_hops < old_hops
                accept = visible_now&(~old_visible|better)&~conflict&(~ambiguity[start+j, batch_ids]|better)
                tracks[start+j, batch_ids[accept]] = predicted[j, accept]
                visibility[start+j, batch_ids[accept]] = True
                hops[start+j, batch_ids[accept]] = batch_hops[accept]
                ambiguity[start+j, batch_ids[accept]] = False
                visibility[start+j, batch_ids[conflict]] = False
                ambiguity[start+j, batch_ids[conflict]] = True
        del video
        torch.cuda.empty_cache()
        print(f'TEMPORAL_TRACK_WINDOW start={start} stop={stop} queries={len(local_queries)} calls={calls}', flush=True)
    for j, frame in enumerate(sampled):
        xy = np.rint(tracks[j]).astype(int)
        inside = (xy[:, 0] >= 0)&(xy[:, 0] < images.shape[2])&(xy[:, 1] >= 0)&(xy[:, 1] < images.shape[1])
        xy[:, 0] = np.clip(xy[:, 0], 0, images.shape[2]-1); xy[:, 1] = np.clip(xy[:, 1], 0, images.shape[1]-1)
        visibility[j] &= inside&masks[frame, xy[:, 1], xy[:, 0]].astype(bool)&~humans[frame, xy[:, 1], xy[:, 0]].astype(bool)
    for q, (source_t, x, y) in enumerate(queries):
        tracks[int(source_t), q] = [x, y]
        visibility[int(source_t), q] = True
    query_frames = sampled[queries[:, 0].astype(int)]
    np.savez_compressed(output/'long-tracks.npz', tracks=tracks, visibility=visibility,
        sampled_original_frame_ids=sampled, query_original_frame_ids=query_frames, query_pixels=queries[:, 1:])
    report = {'model': 'CoTracker3 scaled_offline', 'source_queries': len(queries), 'sampled_frames': len(sampled),
        'tracked_visible_observations': int(visibility.sum()), 'manual_queries': False,
        'inference_window_frames': window, 'maximum_queries_per_call': 96, 'model_calls': calls,
        'overlap_conflicts_rejected': disagreements, 'bidirectional_overlapping_windows': True,
        'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated(),
        'scope': 'Visibility-filtered source RGB measurements, not invented hidden point coordinates'}
    del model
    torch.cuda.empty_cache()
    print(f'TEMPORAL_LONG_TRACKS {report}', flush=True)
    return sampled, tracks, visibility, query_frames, queries[:, 1:], report


def fit_joint(mesh, initial, ids, selected, images, masks, humans, depths, camera, k, affine,
              keypoints, normal, offset, track_data, steps, fps, output):
    import torch
    import torch.nn.functional as F
    from pytorch3d.transforms import so3_exp_map
    from v2hoi.image_motion import surface_points
    from v2hoi.object_refine_runtime import ObservationRenderer
    sampled, tracks, visibility, query_frames, query_pixels, _ = track_data
    vertex0 = np.asarray(mesh.vertices)
    index = {int(frame): j for j, frame in enumerate(ids)}
    local = np.full((len(query_pixels), 3), np.nan)
    for i in ids:
        q = np.flatnonzero(query_frames == i)
        if len(q):
            local[q] = surface_points(query_pixels[q], camera, selected[index[int(i)]], mesh.vertices, mesh.faces)
    attached = np.isfinite(local).all(1)
    visibility &= attached[None]
    local[~attached] = 0.
    renderer = ObservationRenderer(mesh)
    device = 'cuda'
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
    r0, t0 = tensor(initial[:, :3, :3]), tensor(initial[:, :3, 3])
    rotation_delta = torch.zeros((len(initial), 3), device=device, requires_grad=True)
    translation = t0.clone().requires_grad_(True)
    log_shape = torch.zeros(3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([rotation_delta, translation, log_shape], lr=.007)
    truth, care, measured = tensor(masks), 1.-tensor(humans), tensor(depths)
    local_t, track_t, valid_t = tensor(local), tensor(tracks), tensor(visibility)
    vertices = tensor(vertex0)
    # Both hands remain candidates; the closest supported surface is inferred.
    wrists = keypoints[:, [9, 10]]
    support, grasp, grip_local, _ = automatic_contact_schedule(initial, vertex0, wrists, normal, offset, fps)
    support_t, grasp_t, grips = tensor(support), tensor(grasp), tensor(grip_local)
    wrist_t = tensor(wrists)
    # Capsules are approximate collision priors, not decoded human surface SDFs.
    bones = [(5, 6, .07), (5, 11, .07), (6, 12, .07), (11, 12, .07),
             (11, 13, .055), (13, 15, .045), (12, 14, .055), (14, 16, .045)]
    joints = tensor(keypoints)
    n = tensor(normal) if normal is not None else None
    history = []
    def decode():
        return so3_exp_map(rotation_delta)@r0, translation, torch.exp(torch.tanh(log_shape)*.45)
    for step in range(steps):
        optimizer.zero_grad()
        r, t, shape = decode()
        batch = np.arange((step*16)%len(initial), min((step*16)%len(initial)+16, len(initial)))
        _, alpha, rendered_depth = renderer.render(r[batch].contiguous(), t[batch].contiguous(), shape,
            torch.zeros(3, device=device), np.repeat(camera[None], len(batch), axis=0), masks.shape[1:])
        observed, predicted = truth[batch]*care[batch], alpha[..., 0]*care[batch]
        silhouette = (1.-(2*(observed*predicted).sum((1, 2))+1)/((observed+predicted).sum((1, 2))+1)).mean()
        depth_valid = observed*(alpha[..., 0].detach() > .5)*(measured[batch] > .001)
        depth_loss = (F.smooth_l1_loss((rendered_depth[..., 0]-measured[batch])/.08,
            torch.zeros_like(measured[batch]), reduction='none')*depth_valid).sum()/depth_valid.sum().clamp_min(1.)
        points = (local_t*shape)@r[sampled].transpose(1, 2)+t[sampled, None]
        projected = points@tensor(camera).T
        errors = (projected[..., :2]/projected[..., 2:].clamp_min(.1)-track_t)/2.
        point_loss = (torch.log1p(errors.square().sum(-1))*valid_t).sum()/valid_t.sum().clamp_min(1.)
        world = (vertices*shape)@r.transpose(1, 2)+t[:, None]
        acceleration = world[2:]-2*world[1:-1]+world[:-2]
        temporal = F.smooth_l1_loss(acceleration/.004, torch.zeros_like(acceleration))
        physical = silhouette*0.
        if n is not None:
            clearance = offset-world@n
            physical = torch.relu(-clearance).square().mean()*200.
            physical += (support_t*clearance.min(1).values.square()).mean()*20.
            velocity = (world[1:].mean(1)-world[:-1].mean(1))*fps
            tangent = velocity-(velocity@n)[:, None]*n
            physical += ((support_t[1:]*support_t[:-1])[:, None]*tangent.square()).mean()*.03
        grip_world = torch.einsum('nhj,nkj->nhk', grips*shape, r)+t[:, None]
        hand_loss = (grasp_t[..., None]*F.smooth_l1_loss((grip_world-wrist_t)/.1,
            torch.zeros_like(grip_world), reduction='none')).mean()
        collision = silhouette*0.
        for a, b, radius in bones:
            segment = joints[:, b]-joints[:, a]
            u = ((world-joints[:, None, a])*segment[:, None]).sum(-1)/segment.square().sum(-1)[:, None].clamp_min(1e-6)
            closest = joints[:, None, a]+u.clamp(0, 1)[..., None]*segment[:, None]
            distance = torch.linalg.vector_norm(world-closest, dim=-1)
            collision += torch.relu(radius-distance).square().mean()*10.
        loss = silhouette+.08*depth_loss+.12*point_loss+.12*temporal+physical+.02*hand_loss+collision+.03*log_shape.square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([rotation_delta, translation, log_shape], 10.)
        optimizer.step()
        history.append(float(loss.detach()))
        if step % 100 == 0 or step == steps-1:
            print(f'TEMPORAL_JOINT step={step}/{steps} loss={history[-1]:.4f} shape={shape.detach().cpu().numpy().round(3)}', flush=True)
    r, t, shape = decode()
    result = np.repeat(np.eye(4)[None], len(initial), axis=0)
    result[:, :3, :3], result[:, :3, 3] = r.detach().cpu().numpy(), t.detach().cpu().numpy()
    shape = shape.detach().cpu().numpy()
    np.savez(output/'joint-fit.npz', initial=initial, poses=result, global_shape_factors=shape, support_probability=support,
             grasp_probability=grasp, attached_track_points=local, track_attachments_valid=attached, objective=history)
    return result, shape, {'steps': steps, 'source_track_attachments': int(attached.sum()),
        'manual_contact_or_support_intervals': False, 'floor_enabled': normal is not None,
        'physical_constraints': ['nonpenetration of estimated floor', 'soft support and sliding', 'soft hand surface contact', 'approximate human capsule nonpenetration'],
        'physics_engine': 'Analytic differentiable constraints inside joint image/track optimization; no forward engine overwrites poses',
        'full_differentiable_dynamics': False, 'initial_objective': history[0], 'final_objective': history[-1]}


def run(args):
    import cv2
    import torch
    import trimesh
    from lib_mhr import MHRLayer, MHR_PARAM_DIMS
    from prep.mhr_depth_h5 import read_metric_depth
    from prep.mhr_export_utils import camera_calibration, load_edex
    from v2hoi.clips import load_clips
    from v2hoi.hoi_preview import render_full_hoi
    from v2hoi.image_motion import interpolate_poses
    from v2hoi.object_refine import require_timeline
    from v2hoi.physics_motion_runtime import fit_floor, audit_poses
    for path in (args.run, args.dataset_root, args.spec, args.output, args.code, args.checkpoint):
        if 'track_2' in {p.lower() for p in path.resolve().parts}:
            raise ValueError('Track 2 is prohibited')
    spec = json.loads(args.spec.read_text())
    if any(key in spec for key in ('events', 'observation_roi', 'landmarks', 'motion_anchors')):
        raise ValueError('Temporal mode rejects authored ROI, contact events, corners and motion anchors')
    clip = load_clips('track1', [args.episode], args.dataset_root)[0]
    identity = json.loads((args.run/'native-inputs.json').read_text())
    if (spec.get('provenance') != 'track1' or spec['episode'] != clip.episode or spec['video_sha256'] != sha256(clip.video)
            or identity.get('provenance') != 'track1' or identity['video_sha256'] != spec['video_sha256']):
        raise ValueError('Source video provenance mismatch')
    root = args.run/'native'/f'episode_{clip.episode:06d}'
    export = root/'export'/f'episode_{clip.episode:06d}'
    source = root/'inference/refined.pth'
    bundle = torch.load(source, map_location='cpu', weights_only=False)
    if bundle.get('gt') or bundle['metadata'].get('ground_truth_used') is not False:
        raise ValueError('Reference-bearing input is prohibited')
    require_timeline([int(row['src_frame']) for row in bundle['frame_meta']], clip.n_frames)
    k, extrinsic = camera_calibration(load_edex(export), 0)
    np.testing.assert_allclose(extrinsic, np.eye(4), atol=1e-6)
    receipts = json.loads((args.code/'source-receipt.json').read_text())
    model_receipt = json.loads(args.checkpoint.with_name('download-receipt.json').read_text())
    if sha256(args.checkpoint) != model_receipt['files'][args.checkpoint.name]['sha256']:
        raise ValueError('CoTracker checkpoint changed')
    for name, row in receipts['files'].items():
        if sha256(args.code/name) != row['sha256']:
            raise ValueError(f'CoTracker code changed: {name}')
    builder = Path('tools/windows/build_bpy_box_asset.py').resolve()
    asset_paths = bpy_asset_paths(spec, args.spec)
    contact_profile, diagnosis, contact_inputs = None, None, []
    if args.contact_profile:
        if asset_paths is None or args.refinement != 'bundle':
            raise ValueError('Sparse contact requires a receipted bpy asset and bundle refinement')
        from v2hoi.sparse_contact import load_contact_profile
        contact_profile, diagnosis, contact_inputs = load_contact_profile(args.contact_profile, clip, asset_paths[0])
    watched = [source, args.run/'native-inputs.json', args.spec, clip.video, builder, Path(__file__),
        Path(__file__).with_name('temporal_object.py'), Path(__file__).with_name('physics_fp_runtime.py'),
        Path(__file__).with_name('temporal_bundle_runtime.py'),
        Path(__file__).with_name('physics_dynamic_runtime.py'), Path(__file__).with_name('physics_motion_runtime.py'),
        Path(__file__).with_name('image_motion.py'), Path(__file__).with_name('object_refine_runtime.py'),
        args.checkpoint, args.checkpoint.with_name('download-receipt.json'), args.code/'source-receipt.json',
        root/'depth/moge2/aligned.h5', export/'wild_export.json', export/'object_masks/front_stereo_camera_left.h5',
        export/'human_masks/front_stereo_camera_left.h5']
    if contact_profile is not None:
        watched.extend([*contact_inputs, Path(__file__).with_name('sparse_contact.py')])
        import importlib.util
        watched.extend([Path(importlib.util.find_spec('lib_mhr.hand_surface_contact').origin),
            Path(os.environ['MHR_ASSETS_ROOT'])/'checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt'])
    watched.extend(sorted(Path(os.environ['FOUNDATIONPOSE_WEIGHTS_DIR']).rglob('*.pth')))
    if asset_paths is not None:
        watched.extend(asset_paths)
    hashes = {str(path.resolve()): sha256(path) for path in watched}
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/'spec.json').write_bytes(args.spec.read_bytes())
    data, quality, roi = acquire(clip, root, output)
    images, masks, humans, affine, first_depth, excluded, bbox, observation_receipts = data
    camera = k.copy(); camera[:2] = affine[:, :2]@k[:2]; camera[:2, 2] += affine[:, 2]
    depths = []
    for i in range(clip.n_frames):
        depth = read_metric_depth(root/'depth/moge2/aligned.h5', 'aligned', 'front_stereo_camera_left', f'{i:06d}')
        if depth.shape != (clip.height, clip.width):
            depth = cv2.resize(depth, (clip.width, clip.height), interpolation=cv2.INTER_NEAREST)
        depth = cv2.warpAffine(depth, affine, masks.shape[1:][::-1], flags=cv2.INTER_NEAREST)
        depth = np.nan_to_num(depth, nan=0., posinf=0., neginf=0.); depth[(depth < .001)|(depth > 8.)] = 0.
        depths.append(depth)
    depths = np.asarray(depths)
    try:
        normal, offset, floor_report = fit_floor(first_depth, excluded, k, bbox)
    except ValueError as error:
        normal, offset, floor_report = None, None, {'enabled': False, 'reason': str(error)}
    if asset_paths is None:
        with (output/'blender.log').open('w') as log:
            subprocess.run([str(args.blender), '--background', '--factory-startup', '--python-exit-code', '1', '--python', str(builder),
                '--', '--spec', str(args.spec.resolve()), '--output', str(output/'object.glb')], stdout=log, stderr=subprocess.STDOUT, check=True)
        geometry = output/'object.npz'
    else:
        geometry = asset_paths[0]
    if geometry.suffix.lower() == '.npz':
        with np.load(geometry) as arrays:
            mesh = trimesh.Trimesh(vertices=arrays['vertices'], faces=arrays['faces'], vertex_colors=arrays['colors'], process=False)
    else:
        mesh = trimesh.load(geometry, force='mesh', process=False)
        if mesh.visual.kind != 'vertex':
            mesh.visual = mesh.visual.to_color() if hasattr(mesh.visual, 'to_color') else trimesh.visual.ColorVisuals(
                mesh, vertex_colors=mesh.visual.vertex_colors)
    if len(mesh.faces) > 4096 or not mesh.is_watertight or not np.isfinite(mesh.vertices).all():
        raise ValueError('Require a finite, closed low-poly bpy asset with at most 4096 faces')
    contact_parts = None
    if contact_profile is not None:
        from v2hoi.sparse_contact import part_vertex_indices
        contact_parts = part_vertex_indices(mesh.vertices, contact_profile)
    mesh.vertices -= np.r_[mesh.bounds.mean(0)[:2], mesh.bounds[0, 2]]
    scale, shape_report = initialize_scale(depths[0], masks[0].astype(bool)&~humans[0].astype(bool), camera,
        mesh.extents, normal, offset)
    mesh.vertices *= scale
    ids = select_keyframes(quality, args.keyframes)
    ids = np.array([i for i in ids if ((masks[i].astype(bool)&~humans[i].astype(bool))&(depths[i] > .001)).sum() >= 30])
    if not len(ids):
        raise ValueError('No observed keyframe supports pose initialization')
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'], device='cuda')
    keypoints, hand_points = [], []
    hand_spec = None
    if contact_profile is not None:
        from lib_mhr.hand_surface_contact import build_mhr_hand_surface_spec
        hand_spec = build_mhr_hand_surface_spec(layer.backend.mhr_model_path, layer.mesh_faces(device='cpu').numpy(), sample_count=64)
        sampled_hand_vertices = np.take_along_axis(hand_spec.vertex_indices, hand_spec.sample_local_indices, axis=1)
    with torch.inference_mode():
        for i in range(0, clip.n_frames, 16):
            decoded = layer.mhr_forward({key: torch.as_tensor(np.asarray(bundle['pr'][key][i:i+16]), device='cuda') for key in MHR_PARAM_DIMS})
            keypoints.append(decoded.coco17.cpu().numpy())
            if hand_spec is not None:
                hand_points.append(decoded.vertices[:, sampled_hand_vertices].cpu().numpy())
    keypoints = np.concatenate(keypoints)
    contact = None
    if contact_profile is not None:
        from v2hoi.sparse_contact import semantic_contact_schedule, fit_local_support_plane
        hands = np.concatenate(hand_points)
        schedule = semantic_contact_schedule(diagnosis, hands, camera, masks, humans)
        plane, support_report = (None, {'enabled': False}) if normal is None else fit_local_support_plane(
            depths[0], masks[0], humans[0], camera, normal)
        contact = {'profile': contact_profile, 'diagnosis': diagnosis, 'parts': contact_parts,
            'hands': hands, 'hand_centers': hands.mean(2), 'schedule': schedule, 'downward': normal,
            'support_plane': plane, 'support_report': support_report}
        contact.update(camera=camera, human_masks=humans,
            gray_images=np.asarray([cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images], dtype='float32')/255.,
            appearance_parts=[])
        for name in set(schedule['part_names'])-{None}:
            indices = contact_parts[name]
            if np.asarray(mesh.visual.vertex_colors)[indices, :3].mean()/255. > .15:
                continue
            points = np.asarray(mesh.vertices)[indices]
            axis = int(np.ptp(points, axis=0).argmax())
            centerline = np.repeat(points.mean(0)[None], 7, axis=0)
            centerline[:, axis] = np.linspace(*np.quantile(points[:, axis], [.15, .85]), 7)
            contact['appearance_parts'].append((centerline, .25))
        np.savez(output/'contact-evidence.npz', human_hand_vertices=hands, hand_vertex_ids=sampled_hand_vertices,
            hand_centers=contact['hand_centers'], contact_weights=schedule['weights'], sparse_prior=schedule['prior'],
            image_evidence=schedule['image_evidence'], opening_weights=schedule['opening_weights'],
            support_weights=schedule['support_weights'], original_frame_ids=np.arange(clip.n_frames))
        (output/'contact-diagnosis.json').write_text(json.dumps(diagnosis, indent=2))
    track_data = long_tracks(images, masks, humans, ids, args.checkpoint, output, args.tracking_window, args.points_per_keyframe)
    selected, initialization_report = keyframe_candidates(mesh, images, masks, humans, depths, camera, ids, normal, offset, clip.fps, track_data, output,
        mask_proposals=args.initialization == 'mask-proposals', contact=contact)
    initial = interpolate_poses(ids, selected, clip.n_frames)
    if args.refinement == 'bundle':
        from v2hoi.temporal_bundle_runtime import fit_surface_bundle
        poses, shape, fit_report = fit_surface_bundle(mesh, initial, ids, selected, masks, humans, depths, camera,
            keypoints, normal, offset, track_data, args.steps, clip.fps, output, contact=contact)
    else:
        poses, shape, fit_report = fit_joint(mesh, initial, ids, selected, images, masks, humans, depths, camera, k, affine,
            keypoints, normal, offset, track_data, args.steps, clip.fps, output)
    initial_iou = audit_poses(mesh, initial, k, masks, humans, affine)
    mesh.vertices *= shape
    mesh.export(output/'object.glb')
    final_iou = audit_poses(mesh, poses, k, masks, humans, affine)
    candidate = copy.deepcopy(bundle)
    candidate['pr']['pose_abs'] = poses.astype('float32')
    for name in ('pose_abs_postopt', 'pose_abs_1st_delta', 'contact_logits'):
        candidate['pr'].pop(name, None)
    candidate['metadata'].update(object_mesh=str(output/'object.glb'), object_pose_storage_frame='bpy_support_origin',
        object_mesh_to_pose_transform=np.eye(4).tolist(), object_pose_storage_to_training_transform=np.eye(4).tolist(),
        object_mesh_to_training_transform=np.eye(4).tolist(), physics_motion='cotracker-keyframe-joint-physical-v1')
    candidate['postopt'] = {'mode': 'source_long_tracks_joint_physical_constraints', 'source_contact_statistics_invalidated': True}
    torch.save(candidate, output/'refined.pth')
    np.savez(output/'scene.npz', original_frame_ids=np.arange(clip.n_frames), object_poses=poses,
        human_coco17=keypoints, intrinsics=k, floor_normal=np.zeros(3) if normal is None else normal,
        floor_offset=0. if offset is None else offset, floor_enabled=normal is not None,
        object_vertices=np.asarray(mesh.vertices), object_faces=np.asarray(mesh.faces))
    np.savez(output/'visual-diagnostics.npz', initialization_iou=initial_iou, joint_iou=final_iou)
    preview = render_full_hoi(args.run, clip, output/'refined.pth', output, before_path=source)
    if hashes != {str(path.resolve()): sha256(path) for path in watched}:
        raise ValueError('Inputs changed during reconstruction')
    report = {'schema': 'v2hoi.temporal-object.v1', 'provenance': 'track1', 'reference_used': False,
        'episode': clip.episode, 'frames': clip.n_frames, 'inputs': hashes, 'training': False,
        'submission_conversion': False, 'human_parameters_unchanged': True, 'camera_identity_unchanged': True,
        'authored_image_correspondences': 0, 'authored_roi_or_motion_events': False, 'automatic_roi': roi,
        'keyframe_original_ids': ids.tolist(), 'keyframe_initialization': initialization_report,
        'experimental_mask_translation_proposals': args.initialization == 'mask-proposals',
        'sparse_contact_assistance': {'enabled': contact is not None,
            'authored_original_frame_ids': diagnosis['frame_ids'] if diagnosis else [],
            'authored_poses_or_corners': False,
            'hand_surface': '64 native MHR hand vertices per side; centroid proxy, not fitted finger grasp' if contact else None,
            'support_surface': contact['support_report'] if contact else None},
        'long_tracks': track_data[-1], 'shape_initialization': shape_report, 'global_shape_factors': shape.tolist(),
        'floor': floor_report, 'joint_optimization': fit_report, 'full_hoi': preview,
        'motion': motion_diagnostics(poses), 'observations': observation_receipts,
        'visual_diagnostics': {'initialization_iou': float(initial_iou.mean()), 'joint_iou': float(final_iou.mean()),
            'scope': 'Estimated visible-mask agreement, not reference pose accuracy or official scores'},
        'limitations': ['Bpy geometry is agent-authored once; source native human/masks/depth remain uncertain',
            'Initial frame must show enough object depth; scale is uncertain under occlusion or incorrect depth',
            'Unreliable floor evidence disables floor/support losses; table surfaces are not separately reconstructed',
            'Track-to-mesh attachments depend on uncertain initialization and may be wrong',
            'Contact phases and body capsules are soft inferred hypotheses, not reference contact labels',
            'Analytic differentiable physical constraints, not a full differentiable dynamics engine',
            'Complete occlusion is underdetermined; full coverage is not verified hidden motion']}
    names = ('refined.pth', 'scene.npz', 'object.glb', 'full-hoi.mp4', 'full-hoi-comparison.mp4',
             'keyframes.npz', 'long-tracks.npz', 'joint-fit.npz', 'visual-diagnostics.npz')
    report['outputs'] = {name: sha256(output/name) for name in names}
    if contact is not None:
        for name in ('contact-evidence.npz', 'contact-diagnosis.json', 'contact-fit.npz'):
            report['outputs'][name] = sha256(output/name)
    (output/'result.json').write_text(json.dumps(report, indent=2))
    print(f'TEMPORAL_COMPLETE {json.dumps(report["visual_diagnostics"])}', flush=True)


if __name__ == '__main__':
    main()
