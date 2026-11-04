"""Training-free moving-object pilot with source corners and actuated grasp.

Contact intervals and sparse corners are agent annotations from source RGB.
Newton executes a finite-stiffness grasp proxy, not a frictional finger model.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from v2hoi.native import sha256
from v2hoi.physics_motion import plane_frame
from v2hoi.physics_motion_runtime import fit_floor, fit_supported_pose, visual_refine, audit_poses


def calibrate_multiview(spec, k, scales, first_pose, normal, offset):
    """One set of dimensions across all observed orientations, not per-frame scale."""
    import cv2
    from scipy.optimize import least_squares
    local = np.asarray(spec['landmarks']['points_local'], dtype=float)
    rows = [{'frame': 0, 'indices': list(range(len(local))), 'pixels': spec['landmarks']['pixels']}, *spec['motion_anchors']]
    initial = [*np.log(scales)]
    previous = first_pose
    for row in rows:
        xyz = local[row['indices']]*scales
        uv = np.asarray(row['pixels'], dtype=float)
        if row['frame'] == 0:
            pose = first_pose
        else:
            method = cv2.SOLVEPNP_IPPE if np.ptp(xyz[:, 2]) < 1e-6 else cv2.SOLVEPNP_SQPNP
            success, rs, ts, _ = cv2.solvePnPGeneric(xyz, uv, k, None, flags=method)
            candidates = []
            for rv, tv in zip(rs, ts):
                r = cv2.Rodrigues(rv)[0]
                angle = Rotation.from_matrix(r@previous[:3, :3].T).magnitude()
                p = (xyz@r.T+tv.ravel())@k.T
                error = np.median(np.linalg.norm(p[:, :2]/p[:, 2:]-uv, axis=1))
                candidates.append((error+.5*angle, r, tv.ravel()))
            _, r, t = min(candidates, key=lambda x: x[0])
            pose = np.eye(4); pose[:3, :3] = r; pose[:3, 3] = t
        initial.extend(Rotation.from_matrix(pose[:3, :3]).as_rotvec())
        initial.extend(pose[:3, 3])
        previous = pose
    initial = np.asarray(initial)
    def residual(parameters):
        factors = np.exp(parameters[:3])
        result = []
        for index, row in enumerate(rows):
            motion = parameters[3+6*index:9+6*index]
            rotation = Rotation.from_rotvec(motion[:3]).as_matrix()
            xyz = (local[row['indices']]*factors)@rotation.T+motion[3:]
            projected = xyz@k.T
            result.extend(((projected[:, :2]/projected[:, 2:]-np.asarray(row['pixels']))/4.).ravel())
            if row['frame'] in (0, rows[-1]['frame']):
                result.extend((rotation[:, 2]+normal)/.04)
                result.append((motion[3:]@normal-offset)/.02)
        result.extend((parameters[:3]-initial[:3])*.3)
        return result
    fitted = least_squares(residual, initial, loss='soft_l1', f_scale=1., max_nfev=250)
    final_scales = np.exp(fitted.x[:3])
    first = np.eye(4)
    first[:3, :3] = Rotation.from_rotvec(fitted.x[3:6]).as_matrix()
    first[:3, 3] = fitted.x[6:9]
    # Keep the first supported pose exactly on the common floor.
    frame, _ = plane_frame(normal, offset)
    yaw = np.arctan2((frame.T@first[:3, :3])[1, 0], (frame.T@first[:3, :3])[0, 0])
    first[:3, :3] = frame@Rotation.from_euler('z', yaw).as_matrix()
    first[:3, 3] += normal*(offset-first[:3, 3]@normal)
    fitted_poses = {}
    for index, row in enumerate(rows):
        motion = fitted.x[3+6*index:9+6*index]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(motion[:3]).as_matrix()
        pose[:3, 3] = motion[3:]
        fitted_poses[row['frame']] = pose
    return first, final_scales, fitted_poses, {'source_frames': [r['frame'] for r in rows],
        'initial_axis_scales': scales.tolist(), 'final_axis_scales': final_scales.tolist(),
        'robust_objective_before_after': [float(np.mean(np.abs(residual(initial)))), float(np.mean(np.abs(residual(fitted.x))))]}


def observations(clip, root, spec, output):
    import av, cv2
    from prep.mhr_export_utils import read_mask
    from prep.mhr_depth_h5 import read_metric_depth
    export = root/'export'/f'episode_{clip.episode:06d}'
    x, y, side = spec['observation_roi']
    size = 384
    affine = np.array([[size/side, 0, -x*size/side], [0, size/side, -y*size/side]], dtype='float32')
    images, masks, humans = [], [], []
    requested = {0, clip.n_frames-1, *[row['frame'] for row in spec['motion_anchors']]}
    receipts = []
    with av.open(str(clip.video)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            rgb = frame.to_ndarray(format='rgb24')
            obj = read_mask(export, 'object', 0, f'{i:06d}').astype('uint8')
            human = read_mask(export, 'human', 0, f'{i:06d}').astype('uint8')
            images.append(cv2.warpAffine(rgb, affine, (size, size)))
            masks.append(cv2.warpAffine(obj, affine, (size, size), flags=cv2.INTER_NEAREST))
            humans.append(cv2.warpAffine(human, affine, (size, size), flags=cv2.INTER_NEAREST))
            if i in requested:
                path = output/f'observation-{i:06d}.png'
                frame.to_image().save(path)
                receipts.append({'original_frame_id': i, 'path': path.name, 'sha256': sha256(path)})
            if i == 0:
                first_mask, first_human = obj, human
            if i % 90 == 0:
                print(f'DYNAMIC_OBSERVATIONS frame={i}/{clip.n_frames}', flush=True)
    if len(images) != clip.n_frames:
        raise ValueError('Original-frame coverage mismatch')
    depth = read_metric_depth(root/'depth/moge2/aligned.h5', 'aligned', 'front_stereo_camera_left', '000000')
    if depth.shape != first_mask.shape:
        depth = cv2.resize(depth, first_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
    yy, xx = np.where(first_mask)
    bbox = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
    np.savez_compressed(output/'observations.npz', object_masks=masks, human_masks=humans,
                        affine=affine, original_frame_ids=np.arange(clip.n_frames))
    return np.asarray(images), np.asarray(masks), np.asarray(humans), affine, depth, (first_mask|first_human).astype(bool), bbox, receipts


def annotated_anchors(spec, scale, k, first_pose, normal, offset, count, fitted_poses):
    import cv2
    from v2hoi.image_motion import interpolate_poses, project_points
    points = np.asarray(spec['landmarks']['points_local'])*scale
    estimates = {0: first_pose, spec['events']['initial_support_end']: first_pose}
    landmarks = {0: (points, np.asarray(spec['landmarks']['pixels']))}
    report = []
    previous = first_pose
    for row in spec['motion_anchors']:
        ids = np.asarray(row['indices'])
        xyz = points[ids].astype('float64')
        uv = np.asarray(row['pixels'], dtype='float64')
        method = cv2.SOLVEPNP_IPPE if np.ptp(xyz[:, 2]) < 1e-6 else cv2.SOLVEPNP_SQPNP
        success, rotations, translations, _ = cv2.solvePnPGeneric(xyz, uv, k, None, flags=method)
        if not success:
            raise ValueError(f'No source corner pose at original frame {row["frame"]}')
        candidates = []
        seeds = list(zip(rotations, translations))
        calibrated = fitted_poses[row['frame']]
        seeds.append((cv2.Rodrigues(calibrated[:3, :3])[0], calibrated[:3, 3, None].copy()))
        for rv, tv in seeds:
            rv, tv = cv2.solvePnPRefineLM(xyz, uv, k, None, rv.copy(), tv.copy())
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
            if (xyz@pose[:3, :3].T+pose[:3, 3])[:, 2].min() < .1:
                continue
            error = np.linalg.norm(project_points(xyz, pose, k)-uv, axis=1)
            angle = Rotation.from_matrix(pose[:3, :3]@previous[:3, :3].T).magnitude()
            candidates.append((np.mean(error)+.5*angle, pose, error))
        if not candidates:
            raise ValueError('Source landmarks have no positive-depth pose')
        _, pose, error = min(candidates, key=lambda row: row[0])
        report.append({'original_frame_id': row['frame'], 'median_error_px': float(np.median(error)),
                       'max_error_px': float(error.max())})
        if np.median(error) > 15:
            raise ValueError(f'Inconsistent authored corners: {report[-1]}')
        estimates[row['frame']] = pose
        landmarks[row['frame']] = (xyz, uv)
        previous = pose
    # Project the released upright endpoint onto the same estimated floor.
    frame, origin = plane_frame(normal, offset)
    last = estimates[count-1].copy()
    yaw = np.arctan2((frame.T@last[:3, :3])[1, 0], (frame.T@last[:3, :3])[0, 0])
    last[:3, :3] = frame@Rotation.from_euler('z', yaw).as_matrix()
    last[:3, 3] += normal*(offset-normal@last[:3, 3])
    estimates[count-1] = last
    estimates[spec['events']['final_support_start']] = last
    ids = sorted(estimates)
    return estimates, landmarks, interpolate_poses(ids, [estimates[i] for i in ids], count), report


def refine_trajectory(mesh, initial, landmarks, features, masks, humans, affine, k, keypoints, spec, scales, normal, offset, output):
    import torch
    from pytorch3d.transforms import so3_exp_map
    from v2hoi.object_refine_runtime import ObservationRenderer
    from v2hoi.physics_motion import grasp_schedule
    device = 'cuda'
    count = len(initial)
    renderer = ObservationRenderer(mesh)
    camera = k.copy()
    camera[:2] = affine[:, :2]@camera[:2]
    camera[:2, 2] += affine[:, 2]
    r0 = torch.tensor(initial[:, :3, :3], dtype=torch.float32, device=device)
    t0 = torch.tensor(initial[:, :3, 3], dtype=torch.float32, device=device)
    delta = torch.zeros((count, 6), device=device, requires_grad=True)
    free = np.ones(count, dtype='float32')
    free[:spec['events']['initial_support_end']+1] = 0.
    free[spec['events']['final_support_start']:] = 0.
    free = torch.tensor(free[:, None], device=device)
    truth = torch.tensor(masks, device=device, dtype=torch.float32)
    care = 1.-torch.tensor(humans, device=device, dtype=torch.float32)
    intrinsics = torch.tensor(k, device=device, dtype=torch.float32)
    grip = np.asarray(spec['events']['grip_point_local'])*scales
    target_grip = grip@initial[:, :3, :3].transpose(0, 2, 1)+initial[:, :3, 3]
    wrist = keypoints[:, spec['events']['primary_wrist_coco17']]
    confidence = grasp_schedule(count, spec['events']['grasp_start'], spec['events']['grasp_end'])
    selected = confidence > .9
    correction = np.median(target_grip[selected]-wrist[selected], axis=0)
    corrected_wrist = torch.tensor(wrist+correction, dtype=torch.float32, device=device)
    grip_t = torch.tensor(grip, device=device, dtype=torch.float32)
    weights = torch.tensor(confidence, device=device, dtype=torch.float32)
    # Eight independent box corners constrain local angular acceleration as well
    # as translation, without differentiating Euler angles across wraparound.
    probes = torch.tensor(np.array([[a, b, c] for a in [-.14, .14] for b in [-.12, .12] for c in [0., .48]])*scales,
                          dtype=torch.float32, device=device)
    n = torch.tensor(normal, device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam([delta], lr=.008)
    records = []
    packed = {}
    for i in set(features)|set(landmarks):
        xyz, uv = landmarks.get(i, features.get(i))
        packed[i] = (torch.tensor(xyz, device=device, dtype=torch.float32), torch.tensor(uv, device=device, dtype=torch.float32), i in landmarks)
    def decode():
        rotation = so3_exp_map(torch.tanh(delta[:, :3])*.5*free)@r0
        translation = t0+torch.tanh(delta[:, 3:])*.12*free
        return rotation, translation
    for step in range(260):
        start = (step*24) % count
        ids = np.arange(start, min(start+24, count))
        optimizer.zero_grad()
        rotations, centers = decode()
        _, alpha, _ = renderer.render(rotations[ids].contiguous(), centers[ids].contiguous(), 1.,
            torch.zeros(3, device=device), np.repeat(camera[None], len(ids), axis=0), masks.shape[1:])
        observed, predicted = truth[ids]*care[ids], alpha[..., 0]*care[ids]
        dice = 1.-(2*(observed*predicted).sum((1, 2))+1)/((observed+predicted).sum((1, 2))+1)
        image_loss = dice.mean()
        pixel_losses = []
        for i in ids:
            if i not in packed:
                continue
            xyz, uv, authored = packed[i]
            camera_xyz = xyz@rotations[i].T+centers[i]
            projected = camera_xyz@intrinsics.T
            errors = (projected[:, :2]/projected[:, 2:].clamp_min(.1)-uv)/(8. if authored else 3.)
            pixel_losses.append(torch.nn.functional.smooth_l1_loss(errors, torch.zeros_like(errors))*(.5 if authored else .1))
        pixel_loss = torch.stack(pixel_losses).mean() if pixel_losses else image_loss*0
        trajectory = probes@rotations.transpose(1, 2)+centers[:, None]
        d2 = trajectory[2:]-2*trajectory[1:-1]+trajectory[:-2]
        continuity = torch.nn.functional.smooth_l1_loss(d2/.003, torch.zeros_like(d2))
        grip_world = grip_t@rotations.transpose(1, 2)+centers
        hand_loss = (weights[:, None]*torch.nn.functional.smooth_l1_loss((grip_world-corrected_wrist)/.08,
                      torch.zeros_like(grip_world), reduction='none')).mean()
        # Floor and support intervals are constraints inside the fit, not a
        # post-render change to the camera or a per-frame scale adjustment.
        floor_clearance = offset-(trajectory@n)
        penetration = torch.relu(-floor_clearance).square().mean()*100.
        static_ids = np.r_[np.arange(spec['events']['initial_support_end']+1),
                            np.arange(spec['events']['final_support_start'], count)]
        anchor_loss = (centers[static_ids]-t0[static_ids]).square().mean()*20.
        loss = image_loss+pixel_loss+.08*continuity+.01*hand_loss+penetration+anchor_loss+.002*delta.square().mean()
        loss.backward()
        optimizer.step()
        records.append(float(loss.detach()))
        if step % 50 == 0:
            print(f'DYNAMIC_VISUAL_FIT step={step}/260 objective={records[-1]:.4f}', flush=True)
    rotations, centers = decode()
    result = np.repeat(np.eye(4)[None], count, axis=0)
    result[:, :3, :3] = rotations.detach().cpu().numpy()
    result[:, :3, 3] = centers.detach().cpu().numpy()
    # Only explicitly source-annotated resting intervals share a supported pose.
    result[:spec['events']['initial_support_end']+1] = initial[0]
    result[spec['events']['final_support_start']:] = initial[-1]
    np.savez(output/'trajectory-fit.npz', initial=initial, visual=result, objective=records,
             corrected_wrist=(wrist+correction), wrist_correction_m=correction, grasp_confidence=confidence)
    return result, wrist+correction, confidence, {'wrist_offset_m': correction.tolist(),
        'offset_scope': 'Constant registration correction from source-fitted grasp; human parameters unchanged',
        'feature_supported_frames': len(features), 'authored_anchor_frames': sorted(landmarks),
        'automatic_contact_detection': False, 'optimization_steps': len(records)}


def simulate_grasp(mesh, spec, scales, target, corrected_wrist, confidence, normal, offset, fps, output):
    import newton, warp as wp
    from v2hoi.physics_motion import grasp_wrench
    frame, origin = plane_frame(normal, offset)
    rotations = np.einsum('ij,njk->nik', frame.T, target[:, :3, :3])
    centers = (target[:, :3, 3]-origin)@frame
    grip = np.asarray(spec['events']['grip_point_local'])*scales
    visual_grip = np.einsum('nij,j->ni', rotations, grip)+centers
    human_grip = (corrected_wrist-origin)@frame
    # Human evidence is soft because the upstream wrist can disagree with RGB.
    from scipy.ndimage import gaussian_filter1d
    human_grip = gaussian_filter1d(human_grip, sigma=1.5, axis=0)
    wrist_error = np.linalg.norm(human_grip-visual_grip, axis=1)
    human_weight = .1*np.exp(-np.square(wrist_error/.1))
    grip_targets = visual_grip+human_weight[:, None]*(human_grip-visual_grip)
    builder = newton.ModelBuilder(gravity=(0., 0., -9.81))
    body = builder.add_body(xform=wp.transform(centers[0], Rotation.from_matrix(rotations[0]).as_quat()))
    cfg = newton.ModelBuilder.ShapeConfig(density=500., mu=.8, gap=.001)
    for box in spec['boxes']:
        size = np.asarray(box['size'])*scales/2
        builder.add_shape_box(body, xform=wp.transform(np.asarray(box['center'])*scales, wp.quat_identity()),
            hx=float(size[0]), hy=float(size[1]), hz=float(size[2]), cfg=cfg)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=.8, gap=.001))
    model = builder.finalize(device='cuda:0')
    solver = newton.solvers.SolverXPBD(model, iterations=12, angular_damping=1., deterministic=wp.DeterministicMode.RUN_TO_RUN)
    state, other, control = model.state(), model.state(), model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    com = model.body_com.numpy()[0]
    mass = float(model.body_mass.numpy()[0])
    inertia = model.body_inertia.numpy()[0]
    dt = 1/fps/8
    poses, counts, forces, torques = [], [], [], []
    target_com = np.einsum('nij,j->ni', rotations, com)+centers
    for frame_index in range(len(target)+30):
        i = max(0, frame_index-30)
        for sub in range(8):
            q = state.body_q.numpy()[0]
            velocity = state.body_qd.numpy()[0]
            r = Rotation.from_quat(q[3:]).as_matrix()
            current_com = q[:3]+r@com
            u = (sub+1)/8.
            previous_i = max(0, i-1)
            interpolated_confidence = confidence[previous_i]*(1-u)+confidence[i]*u
            target_grip = grip_targets[previous_i]*(1-u)+grip_targets[i]*u
            target_velocity = (grip_targets[i]-grip_targets[previous_i])*fps
            angular_delta = Rotation.from_matrix(rotations[i]@rotations[previous_i].T).as_rotvec()
            target_rotation = Rotation.from_rotvec(angular_delta*u).as_matrix()@rotations[previous_i]
            if frame_index >= 30 and interpolated_confidence > .001:
                force, torque = grasp_wrench(q[:3], r, velocity, com, grip,
                    target_grip, target_velocity, target_rotation, mass, inertia, interpolated_confidence)
            else:
                force = (target_com[i]-current_com)*mass*100.-velocity[:3]*mass*15.
                torque = inertia@(Rotation.from_matrix(r.T@rotations[i]).as_rotvec()*400.)
                torque = r@torque-velocity[3:]*mass*.1
            state.clear_forces()
            state.body_f.assign(np.r_[force, torque][None].astype('float32'))
            pipeline.collide(state, contacts)
            solver.step(state, other, control, contacts, dt)
            state, other = other, state
        if frame_index < 30:
            continue
        q = state.body_q.numpy()[0]
        pose = np.eye(4)
        pose[:3, :3] = frame@Rotation.from_quat(q[3:]).as_matrix()
        pose[:3, 3] = origin+frame@q[:3]
        poses.append(pose)
        counts.append(int(contacts.rigid_contact_count.numpy()[0]))
        forces.append(force)
        torques.append(torque)
        if i % 90 == 0:
            print(f'NEWTON_DYNAMIC frame={i}/{len(target)} contacts={counts[-1]} grasp={confidence[i]:.2f}', flush=True)
    result = np.asarray(poses)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite dynamic solver output')
    clearance = np.min(offset-(np.asarray(mesh.vertices)[None]@result[:, :3, :3].transpose(0, 2, 1)+result[:, None, :3, 3])@normal, axis=1)
    np.savez(output/'physics-diagnostics.npz', poses=result, contacts=counts, forces_n=forces,
        torques_nm=torques, grasp_confidence=confidence, support_clearance_m=clearance,
        original_frame_ids=np.arange(len(result)))
    return result, {'engine': 'Newton', 'version': newton.__version__, 'warp_version': wp.__version__,
        'solver': 'XPBD', 'device': str(model.device), 'substeps': 8, 'mass_prior_kg': mass,
        'contact_frames': int((np.asarray(counts)>0).sum()), 'minimum_floor_clearance_m': float(clearance.min()),
        'maximum_shift_from_visual_m': float(np.linalg.norm(result[:, :3, 3]-target[:, :3, 3], axis=1).max()),
        'grasp_proxy': 'Finite-stiffness point force and wrist torque; source target with uncertainty-weighted native wrist',
        'native_wrist_target_weight_mean_max': [float(human_weight.mean()), float(human_weight.max())],
        'subframe_target_interpolation': True,
        'force_norm_n_p95_max': np.quantile(np.linalg.norm(forces, axis=1), [.95, 1.]).tolist(),
        'object_transform_overwritten_each_frame': False, 'frictional_finger_contact': False}


def run_dynamic_scene(args):
    import torch, trimesh
    from lib_mhr import MHRLayer, MHR_PARAM_DIMS
    from prep.mhr_export_utils import camera_calibration, load_edex
    from v2hoi.clips import load_clips
    from v2hoi.hoi_preview import render_full_hoi
    from v2hoi.object_refine import require_timeline
    from v2hoi.image_motion import track_surface_features
    for path in (args.run, args.spec, args.dataset_root):
        if 'track_2' in {p.lower() for p in path.resolve().parts}:
            raise ValueError('Track 2 is prohibited')
    clip = load_clips('track1', [args.episode], args.dataset_root)[0]
    spec = json.loads(args.spec.read_text())
    identity = json.loads((args.run/'native-inputs.json').read_text())
    if (spec.get('provenance') != 'track1' or spec['episode'] != clip.episode or spec['video_sha256'] != sha256(clip.video)
            or identity.get('provenance') != 'track1' or identity['video_sha256'] != spec['video_sha256']):
        raise ValueError('Permitted source video/asset identity mismatch')
    root = args.run/'native'/f'episode_{clip.episode:06d}'
    export = root/'export'/f'episode_{clip.episode:06d}'
    source = root/'inference/refined.pth'
    bundle = torch.load(source, map_location='cpu', weights_only=False)
    if bundle.get('gt') or bundle['metadata'].get('ground_truth_used') is not False:
        raise ValueError('Reference-bearing bundles are prohibited')
    require_timeline([int(row['src_frame']) for row in bundle['frame_meta']], clip.n_frames)
    k, extrinsic = camera_calibration(load_edex(export), 0)
    if not np.allclose(extrinsic, np.eye(4), atol=1e-6):
        raise ValueError('Expected unchanged shared camera coordinates')
    builder = Path('tools/windows/build_bpy_box_asset.py').resolve()
    watched = [source, args.run/'native-inputs.json', args.spec, clip.video, builder,
        root/'depth/moge2/aligned.h5', export/'wild_export.json',
        export/'object_masks/front_stereo_camera_left.h5', export/'human_masks/front_stereo_camera_left.h5',
        Path(__file__), Path(__file__).with_name('physics_motion.py'), Path(__file__).with_name('physics_motion_runtime.py')]
    hashes = {str(p.resolve()): sha256(p) for p in watched}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'spec.json').write_bytes(args.spec.read_bytes())
    images, masks, humans, affine, depth, excluded, bbox, receipts = observations(clip, root, spec, output)
    normal, offset, floor_report = fit_floor(depth, excluded, k, bbox)
    initial, scales, initial_report = fit_supported_pose(spec, k, normal, offset)
    with (output/'blender.log').open('w') as log:
        subprocess.run([str(args.blender), '--background', '--factory-startup', '--python-exit-code', '1', '--python', str(builder),
            '--', '--spec', str(args.spec.resolve()), '--output', str(output/'object.glb')], stdout=log, stderr=subprocess.STDOUT, check=True)
    with np.load(output/'object.npz') as arrays:
        mesh = trimesh.Trimesh(vertices=arrays['vertices']*scales, faces=arrays['faces'], vertex_colors=arrays['colors'], process=False)
    mesh, initial, scales, shape_report = visual_refine(mesh, initial, scales, spec, normal, offset, k,
        masks[:spec['events']['initial_support_end']], humans[:spec['events']['initial_support_end']], affine, output)
    initial, calibrated_scales, fitted_poses, multiview_report = calibrate_multiview(spec, k, scales, initial, normal, offset)
    mesh.vertices *= calibrated_scales/scales
    scales = calibrated_scales
    (output/'multiview-shape.json').write_text(json.dumps(multiview_report, indent=2))
    mesh.export(output/'object.glb')
    anchors, landmarks, interpolated, anchor_report = annotated_anchors(spec, scales, k, initial, normal, offset, clip.n_frames, fitted_poses)
    (output/'anchors.json').write_text(json.dumps(anchor_report, indent=2))
    roi_k = affine[:, :2]@k[:2]
    camera = k.copy()
    camera[:2] = roi_k
    camera[:2, 2] += affine[:, 2]
    boxes = np.repeat(np.asarray(spec['observation_roi'])[None], clip.n_frames, axis=0)
    tracked, features, feature_report = track_surface_features(images.astype('float32')/255., masks, humans,
        np.repeat(camera[None], clip.n_frames, axis=0), boxes, k, mesh, anchors, interpolated, clip.fps)
    print(f'DYNAMIC_FEATURES supported={len(features)}/{clip.n_frames}', flush=True)
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'], device='cuda')
    keypoints = []
    with torch.inference_mode():
        for i in range(0, clip.n_frames, 16):
            decoded = layer.mhr_forward({key: torch.as_tensor(np.asarray(bundle['pr'][key][i:i+16]), device='cuda') for key in MHR_PARAM_DIMS})
            keypoints.append(decoded.coco17.cpu().numpy())
    keypoints = np.concatenate(keypoints)
    visual, wrist, confidence, fit_report = refine_trajectory(mesh, interpolated, landmarks, features, masks, humans, affine,
        k, keypoints, spec, scales, normal, offset, output)
    poses, physics_report = simulate_grasp(mesh, spec, scales, visual, wrist, confidence, normal, offset, clip.fps, output)
    visual_iou = audit_poses(mesh, visual, k, masks, humans, affine)
    physical_iou = audit_poses(mesh, poses, k, masks, humans, affine)
    candidate = copy.deepcopy(bundle)
    candidate['pr']['pose_abs'] = poses.astype('float32')
    for key in ('pose_abs_postopt', 'pose_abs_1st_delta', 'contact_logits'):
        candidate['pr'].pop(key, None)
    candidate['metadata'].update(object_mesh=str(output/'object.glb'), object_pose_storage_frame='bpy_support_origin',
        object_mesh_to_pose_transform=np.eye(4).tolist(), object_pose_storage_to_training_transform=np.eye(4).tolist(),
        object_mesh_to_training_transform=np.eye(4).tolist(), physics_motion='source-corners-actuated-grasp-newton-v1')
    candidate['postopt'] = {'mode': 'visual_physical_dynamic_object', 'source_contact_statistics_invalidated': True}
    torch.save(candidate, output/'refined.pth')
    np.savez(output/'scene.npz', original_frame_ids=np.arange(clip.n_frames), object_poses=poses, visual_poses=visual,
        floor_normal=normal, floor_offset=offset, intrinsics=k, human_coco17=keypoints,
        object_vertices=np.asarray(mesh.vertices), object_faces=np.asarray(mesh.faces))
    from lib_mhr.object_pose_frame import resolve_object_pose_frame
    old_mesh = trimesh.load(bundle['metadata']['object_mesh'], force='mesh', process=False)
    object_frame = resolve_object_pose_frame(bundle['metadata'])
    old_mesh.apply_transform(object_frame.mesh_to_training)
    old_poses = np.asarray(bundle['pr']['pose_abs'])@np.linalg.inv(object_frame.storage_to_training)
    old_iou = audit_poses(old_mesh, old_poses, k, masks, humans, affine)
    np.savez(output/'visual-diagnostics.npz', old_iou=old_iou, visual_only_iou=visual_iou, physics_iou=physical_iou)
    preview = render_full_hoi(args.run, clip, output/'refined.pth', output, before_path=source)
    if hashes != {str(p.resolve()): sha256(p) for p in watched}:
        raise ValueError('Input or runtime changed during this run')
    report = {'schema': 'v2hoi.visual-physical-dynamic-object.v1', 'provenance': 'track1', 'episode': clip.episode,
        'status': 'pilot_complete_requires_review', 'frames': clip.n_frames, 'original_frame_ids': list(range(clip.n_frames)),
        'inputs': hashes, 'upstream_inputs_unchanged': True, 'observations': receipts,
        'training': False, 'reference_used': False, 'submission_conversion': False, 'human_parameters_unchanged': True,
        'camera_identity_unchanged': True, 'foundationpose_calls': 0, 'foundationpose_used_for_fit': False,
        'floor': {**floor_report, 'normal': normal.tolist(), 'offset_m': offset}, 'initial_fit': initial_report,
        'shape_fit': shape_report, 'multiview_shape': multiview_report, 'motion_anchors': anchor_report, 'feature_tracking': feature_report,
        'trajectory_fit': fit_report, 'physics': physics_report, 'full_hoi': preview,
        'visual_diagnostics': {'old_mesh_and_motion_iou': float(old_iou.mean()),
            'visual_only_iou': float(visual_iou.mean()), 'physical_iou': float(physical_iou.mean()),
            'scope': 'Estimated input segmentation agreement, not reference pose accuracy'},
        'limitations': ['Sparse source corners and contact events are agent-authored; no automatic event detector',
            'Simplified wrist-actuated grasp; no frictional fingers, arm force limits or full human collision',
            'Original human pose errors persist; wrist registration is a soft proxy',
            'Monocular floor, scale and hidden geometry remain uncertain',
            'An executed rollout is not evidence of improved reconstruction; inspect source alignment and failures']}
    report['outputs'] = {n: sha256(output/n) for n in ('refined.pth', 'object.glb', 'scene.npz', 'full-hoi.mp4', 'full-hoi-comparison.mp4')}
    (output/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'output': str(output), 'visual': report['visual_diagnostics'], 'physics': physics_report}, indent=2), flush=True)
