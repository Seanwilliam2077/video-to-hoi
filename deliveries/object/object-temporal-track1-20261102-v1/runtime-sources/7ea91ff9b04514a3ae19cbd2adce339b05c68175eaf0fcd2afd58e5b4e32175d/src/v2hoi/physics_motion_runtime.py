"""Native Windows pilot: observed supported pose, Newton contacts, complete HOI.

The static-support branch is evidence-gated. It is not a general dynamic
inverse-physics solver, and its assumed masses/loads are not measured forces.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from v2hoi.native import sha256
from v2hoi.physics_motion import plane_frame, plane_pose, static_evidence


def fit_floor(depth, excluded, intrinsics, box):
    """Robustly fit visible floor depth near the object; no reference geometry."""
    h, w = depth.shape
    x0, y0, x1, y1 = box
    y, x = np.mgrid[max(y1 - 50, int(h * .70)):min(h, y1 + 180):5,
                    max(0, x0 - 250):min(w, x1 + 250):5]
    z = depth[y, x]
    keep = ~excluded[y, x] & np.isfinite(z) & (z > .1)
    x, y, z = x[keep], y[keep], z[keep]
    cloud = np.column_stack(((x - intrinsics[0, 2]) * z / intrinsics[0, 0],
                             (y - intrinsics[1, 2]) * z / intrinsics[1, 1], z))
    if len(cloud) < 100:
        raise ValueError('Insufficient visible floor depth')
    rng = np.random.default_rng(42)
    best = np.zeros(len(cloud), dtype=bool)
    for _ in range(400):
        points = cloud[rng.choice(len(cloud), 3, replace=False)]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        normal *= np.sign(normal[1])
        if normal[1] < .6:
            continue
        inliers = abs(cloud @ normal - points[0] @ normal) < .02
        if inliers.sum() > best.sum():
            best = inliers
    if best.mean() < .5:
        raise ValueError('No dominant visible floor plane; manual review required')
    points = cloud[best]
    _, _, axes = np.linalg.svd(points - points.mean(0), full_matrices=False)
    normal = axes[-1] * np.sign(axes[-1, 1])
    offset = float(normal @ points.mean(0))
    return normal, offset, {'samples': len(cloud), 'inlier_fraction': float(best.mean()),
                           'median_residual_m': float(np.median(abs(points @ normal - offset))),
                           'source': 'First original-frame human-aligned monocular depth; uncertain scale'}


def observe(clip, root, output):
    import av
    import cv2
    from prep.mhr_export_utils import read_mask
    from prep.mhr_depth_h5 import read_metric_depth
    export = root / 'export' / f'episode_{clip.episode:06d}'
    masks, humans, rgbs = [], [], []
    # Fixed crops preserve image-space movement; no per-frame bbox recentering.
    with av.open(str(clip.video)) as container:
        first = next(container.decode(video=0)).to_ndarray(format='rgb24')
    first_mask = read_mask(export, 'object', 0, '000000').astype(bool)
    y, x = np.where(first_mask)
    if len(x) < 100:
        raise ValueError('The supported-object pilot needs visible first-frame geometry')
    box = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
    x0, y0 = max(0, box[0]-60), max(0, box[1]-60)
    x1, y1 = min(clip.width, box[2]+60), min(clip.height, box[3]+60)
    affine = np.array([[256/(x1-x0), 0, -x0*256/(x1-x0)], [0, 320/(y1-y0), -y0*320/(y1-y0)]], dtype='float32')
    reference = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    feature_mask = cv2.erode(first_mask.astype('uint8'), np.ones((3, 3), 'uint8'))
    points = cv2.goodFeaturesToTrack(reference, maxCorners=240, qualityLevel=.01, minDistance=3, mask=feature_mask)
    if points is None or len(points) < 12:
        raise ValueError('No reliable long-baseline object feature support')
    displacements, supports = [], []
    # Caches use only permitted source RGB/masks. Empty observations stay empty.
    with av.open(str(clip.video)) as container:
        for i, decoded in enumerate(container.decode(video=0)):
            rgb = decoded.to_ndarray(format='rgb24')
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            mask = read_mask(export, 'object', 0, f'{i:06d}').astype(bool)
            human = read_mask(export, 'human', 0, f'{i:06d}').astype(bool)
            moved, valid, error = cv2.calcOpticalFlowPyrLK(reference, gray, points, None, winSize=(15, 15), maxLevel=3)
            back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, reference, moved, None, winSize=(15, 15), maxLevel=3)
            xy = np.rint(moved[:, 0]).astype(int)
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < clip.width) & (xy[:, 1] >= 0) & (xy[:, 1] < clip.height)
            xy[:, 0] = np.clip(xy[:, 0], 0, clip.width-1)
            xy[:, 1] = np.clip(xy[:, 1], 0, clip.height-1)
            valid = (inside & valid[:, 0].astype(bool) & backward[:, 0].astype(bool)
                     & (np.linalg.norm(back[:, 0]-points[:, 0], axis=1) < .7)
                     & (error[:, 0] < 15.) & mask[xy[:, 1], xy[:, 0]] & ~human[xy[:, 1], xy[:, 0]])
            supports.append(int(valid.sum()))
            # Normalize pixels to a 768-wide image for a resolution-independent gate.
            displacements.append(float(np.median(np.linalg.norm(moved[valid, 0]-points[valid, 0], axis=1))) * 768/clip.width if valid.any() else np.nan)
            masks.append(cv2.warpAffine(mask.astype('uint8'), affine, (256, 320), flags=cv2.INTER_NEAREST))
            humans.append(cv2.warpAffine(human.astype('uint8'), affine, (256, 320), flags=cv2.INTER_NEAREST))
            if i in (0, clip.n_frames//3, clip.n_frames*2//3, clip.n_frames-1):
                rgbs.append(cv2.warpAffine(rgb, affine, (256, 320)))
            if i % 90 == 0:
                print(f'PHYSICS_OBSERVATIONS frame={i}/{clip.n_frames}', flush=True)
    if len(masks) != clip.n_frames:
        raise ValueError('Decoded source length differs from the original-frame manifest')
    diagnostic = static_evidence(displacements, supports)
    diagnostic['displacement_units'] = 'Pixels normalized to image width 768'
    np.savez_compressed(output/'image-evidence.npz', displacement=displacements, support=supports,
                        original_frame_ids=np.arange(clip.n_frames), crop_affine=affine,
                        object_masks=masks, human_masks=humans)
    cv2.imwrite(str(output/'observed-object-timeline.jpg'), cv2.cvtColor(np.hstack(rgbs), cv2.COLOR_RGB2BGR))
    (output/'image-evidence.json').write_text(json.dumps(diagnostic, indent=2))
    if not diagnostic['accepted']:
        raise ValueError(f'Static-support evidence gate failed: {diagnostic}; moving-object solver is not implemented')
    depth = read_metric_depth(root/'depth/moge2/aligned.h5', 'aligned', 'front_stereo_camera_left', '000000')
    if depth.shape != first_mask.shape:
        depth = cv2.resize(depth, first_mask.shape[::-1], interpolation=cv2.INTER_NEAREST)
    human = read_mask(export, 'human', 0, '000000').astype(bool)
    return np.asarray(masks), np.asarray(humans), affine, depth, first_mask | human, box, diagnostic


def fit_supported_pose(spec, k, normal, offset):
    """Fit a grounded upright pose and constant dimensions from source landmarks."""
    import cv2
    points = np.asarray(spec['landmarks']['points_local'], dtype=float)
    pixels = np.asarray(spec['landmarks']['pixels'], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or pixels.shape != (len(points), 2)
            or len(points) < 6 or not np.isfinite(points).all() or not np.isfinite(pixels).all()):
        raise ValueError('Require at least six finite, source-receipted landmark correspondences')
    success, rv, tv = cv2.solvePnP(points, pixels, k, None, flags=cv2.SOLVEPNP_SQPNP)
    if not success:
        raise ValueError('Source landmark initialization failed')
    rotation = cv2.Rodrigues(rv)[0]
    frame, origin = plane_frame(normal, offset)
    scale = offset / float(normal @ tv.ravel())
    if not .1 < scale < 10:
        raise ValueError('Inconsistent floor/landmark scale initialization')
    yaw = Rotation.from_matrix(frame.T @ rotation).as_euler('xyz')[2]
    translation = frame.T @ (tv.ravel()*scale - origin)
    initial = np.r_[yaw, translation[:2], np.full(3, np.log(scale))]
    def decode(parameters):
        pose, _ = plane_pose(np.r_[parameters[:3], 0.], frame, origin)
        return pose, np.exp(parameters[3:])
    def residual(parameters):
        pose, scales = decode(parameters)
        camera = (points * scales) @ pose[:3, :3].T + pose[:3, 3]
        projected = camera @ k.T
        image_error = (projected[:, :2]/projected[:, 2:] - pixels).ravel()
        # Proportions are an observational prior, never a metric reference.
        shape_prior = (parameters[3:]-np.mean(parameters[3:])) * 5.
        return np.r_[image_error, shape_prior]
    fitted = least_squares(residual, initial, loss='soft_l1', f_scale=2., max_nfev=400)
    pose, scales = decode(fitted.x)
    return pose, scales, {'source_original_frame_id': spec['landmarks']['original_frame_id'],
                         'initial_median_landmark_error_px': float(np.median(np.linalg.norm(residual(fitted.x)[:-3].reshape(-1, 2), axis=1))),
                         'floor_constrained_upright': True, 'constant_axis_scales': scales.tolist()}


def visual_refine(mesh, pose, scales, spec, normal, offset, k, masks, humans, affine, output):
    """Bounded source-mask refinement; held-out frames are never fitted."""
    import torch
    from v2hoi.object_refine_runtime import ObservationRenderer
    renderer = ObservationRenderer(mesh)
    camera = k.copy()
    camera[:2] = affine[:, :2] @ camera[:2]
    camera[0, 2] += affine[0, 2]
    camera[1, 2] += affine[1, 2]
    eligible = np.array([i for i in range(len(masks)) if i % 7 != 0 and masks[i].sum() > 100])
    ids = eligible[np.linspace(0, len(eligible)-1, 8).round().astype(int)]
    truth = torch.tensor(masks[ids], dtype=torch.float32, device='cuda')
    valid = 1.-torch.tensor(humans[ids], dtype=torch.float32, device='cuda')
    target = truth*valid
    base_r = torch.tensor(pose[:3, :3], dtype=torch.float32, device='cuda')
    base_t = torch.tensor(pose[:3, 3], dtype=torch.float32, device='cuda')
    frame, _ = plane_frame(normal, offset)
    basis = torch.tensor(frame[:, :2], dtype=torch.float32, device='cuda')
    points = torch.tensor(np.asarray(spec['landmarks']['points_local'])*scales, dtype=torch.float32, device='cuda')
    pixels = torch.tensor(spec['landmarks']['pixels'], dtype=torch.float32, device='cuda')
    intrinsics = torch.tensor(k, dtype=torch.float32, device='cuda')
    delta = torch.zeros(6, device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=.004)
    def decode():
        yaw = torch.tanh(delta[0])*.12
        c, s = torch.cos(yaw), torch.sin(yaw)
        zero, one = yaw*0., yaw*0.+1.
        rz = torch.stack((c, -s, zero, s, c, zero, zero, zero, one)).reshape(3, 3)
        rotation = base_r @ rz
        center = base_t + basis @ (torch.tanh(delta[1:3])*.06)
        factors = torch.exp(torch.tanh(delta[3:])*.16)
        return rotation, center, factors
    losses = []
    for step in range(160):
        optimizer.zero_grad()
        rotation, center, factors = decode()
        _, alpha, _ = renderer.render(rotation[None].expand(len(ids), -1, -1).contiguous(),
            center[None].expand(len(ids), -1).contiguous(), factors,
            torch.zeros(3, device='cuda'), np.repeat(camera[None], len(ids), axis=0), masks.shape[1:])
        predicted = alpha[..., 0]*valid
        dice = 1.-(2*(predicted*target).sum((1, 2))+1)/((predicted+target).sum((1, 2))+1)
        silhouette_loss = dice.mean()
        xyz = (points*factors) @ rotation.T + center
        uv = xyz @ intrinsics.T
        landmark_loss = torch.nn.functional.smooth_l1_loss((uv[:, :2]/uv[:, 2:]-pixels)/10., torch.zeros_like(pixels))
        loss = silhouette_loss + .12*landmark_loss + .02*delta.square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 40 == 0:
            print(f'PHYSICS_VISUAL_FIT step={step}/160 objective={losses[-1]:.5f}', flush=True)
    rotation, center, factors = decode()
    result = np.eye(4)
    result[:3, :3] = rotation.detach().cpu().numpy()
    result[:3, 3] = center.detach().cpu().numpy()
    factors = factors.detach().cpu().numpy()
    mesh.vertices *= factors
    scales = scales*factors
    np.savez(output/'visual-optimization.npz', objective=losses, fitted_original_frame_ids=ids,
             axis_factors=factors, initial_pose=pose, fitted_pose=result)
    return mesh, result, scales, {'fitted_original_frame_ids': ids.tolist(),
        'held_out_rule': 'Original frame ID modulo 7 equals zero, except frame 0 used for landmarks',
        'first_last_objective': [losses[0], losses[-1]], 'axis_factors': factors.tolist()}


def audit_poses(mesh, poses, k, masks, humans, affine):
    import torch
    from v2hoi.object_refine_runtime import ObservationRenderer
    renderer = ObservationRenderer(mesh)
    camera = k.copy()
    camera[:2] = affine[:, :2] @ camera[:2]
    camera[0, 2] += affine[0, 2]
    camera[1, 2] += affine[1, 2]
    values = []
    with torch.inference_mode():
        for start in range(0, len(poses), 16):
            batch = poses[start:start+16]
            _, alpha, _ = renderer.render(torch.tensor(batch[:, :3, :3], dtype=torch.float32, device='cuda'),
                torch.tensor(batch[:, :3, 3], dtype=torch.float32, device='cuda'), 1.,
                torch.zeros(3, device='cuda'), np.repeat(camera[None], len(batch), axis=0), masks.shape[1:])
            silhouette = alpha[..., 0].cpu().numpy() > .5
            valid = ~humans[start:start+len(batch)].astype(bool)
            predicted, observed = silhouette & valid, masks[start:start+len(batch)].astype(bool) & valid
            values.extend(((predicted & observed).sum((1, 2))/np.maximum((predicted | observed).sum((1, 2)), 1)).tolist())
    return np.asarray(values)


def simulate_supported(mesh, spec, pose, scales, normal, offset, keypoints, fps, output):
    """Free dynamic compound body, floor collision, and bounded visual restoring wrench.

    Seat loading is an explicitly assumed support probe, inferred from pelvis
    proximity. It is not a measured human force or full human collision model.
    """
    import newton
    import warp as wp
    frame, origin = plane_frame(normal, offset)
    world_r = frame.T @ pose[:3, :3]
    world_t = frame.T @ (pose[:3, 3]-origin)
    builder = newton.ModelBuilder(gravity=(0., 0., -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(density=500., mu=.8, restitution=0., gap=.001)
    body = builder.add_body(xform=wp.transform(world_t, Rotation.from_matrix(world_r).as_quat()), label='observed_object')
    for component in spec['boxes']:
        center = np.asarray(component['center'])*scales
        size = np.asarray(component['size'])*scales
        builder.add_shape_box(body, xform=wp.transform(center, wp.quat_identity()),
                              hx=float(size[0]/2), hy=float(size[1]/2), hz=float(size[2]/2), cfg=cfg,
                              label=component['name'])
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=.8, gap=.001))
    model = builder.finalize(device='cuda:0')
    solver = newton.solvers.SolverXPBD(model, iterations=12, angular_damping=3., deterministic=wp.DeterministicMode.RUN_TO_RUN)
    state, other = model.state(), model.state()
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    local_com = model.body_com.numpy()[0]
    target_com = world_t + world_r @ local_com
    mass = float(model.body_mass.numpy()[0])
    count = len(keypoints)
    poses, contact_counts, load_history, floor_errors = [], [], [], []
    dt = 1./fps/8
    pelvis = keypoints[:, [11, 12]].mean(1)
    local_pelvis = (pelvis-pose[:3, 3]) @ pose[:3, :3]
    seat = spec['support_surface']
    half = np.asarray(seat['half_extents'])*scales[:2]
    seat_z = float(seat['height']*scales[2])
    near = ((abs(local_pelvis[:, 0]) < half[0]+.08) & (abs(local_pelvis[:, 1]) < half[1]+.12)
            & (local_pelvis[:, 2] > seat_z-.08) & (local_pelvis[:, 2] < seat_z+.30))
    from scipy.ndimage import gaussian_filter1d
    seat_confidence = gaussian_filter1d(near.astype(float), sigma=3.)
    # Warm-up is simulation, not a hidden original-frame offset.
    total = count + 30
    for index in range(total):
        i = max(0, index-30)
        assumed_load = float(seat_confidence[i]*35.*9.81) if index >= 30 else 0.
        for _ in range(8):
            q = state.body_q.numpy()[0]
            velocity = state.body_qd.numpy()[0]
            r = Rotation.from_quat(q[3:]).as_matrix()
            com = q[:3]+r@local_com
            # A soft visual drive permits collision resolution. Do not assign
            # object transforms to their observed targets on every frame.
            force = (target_com-com)*mass*30. - velocity[:3]*mass*8.
            torque = Rotation.from_matrix(world_r @ r.T).as_rotvec()*mass*.8 - velocity[3:]*mass*.15
            force[2] -= assumed_load
            # The support load acts through the seat centre, avoiding invented
            # lateral friction or torque from uncertain human joint offsets.
            application = world_t + world_r @ np.array([0., 0., seat_z])
            torque += np.cross(application-com, np.array([0., 0., -assumed_load]))
            state.clear_forces()
            state.body_f.assign(np.r_[force, torque][None].astype('float32'))
            pipeline.collide(state, contacts)
            solver.step(state, other, control, contacts, dt)
            state, other = other, state
        if index < 30:
            continue
        q = state.body_q.numpy()[0]
        result = np.eye(4)
        result[:3, :3] = frame @ Rotation.from_quat(q[3:]).as_matrix()
        result[:3, 3] = origin + frame @ q[:3]
        poses.append(result)
        contact_counts.append(int(contacts.rigid_contact_count.numpy()[0]))
        load_history.append(assumed_load)
        vertices = np.asarray(mesh.vertices) @ result[:3, :3].T+result[:3, 3]
        floor_errors.append(float(np.min(offset-vertices@normal)))
        if i % 90 == 0:
            print(f'NEWTON_SIMULATION frame={i}/{count} contacts={contact_counts[-1]}', flush=True)
    poses = np.asarray(poses)
    if not np.isfinite(poses).all():
        raise ValueError('Non-finite physical trajectory')
    shift = np.linalg.norm(poses[:, :3, 3]-pose[:3, 3], axis=1)
    angles = Rotation.from_matrix(poses[:, :3, :3] @ pose[:3, :3].T).magnitude()
    if shift.max() > .03 or angles.max() > np.deg2rad(5):
        raise ValueError('Physics exceeds the observed static support tolerance; do not publish this candidate')
    np.savez(output/'physics-diagnostics.npz', poses=poses, original_frame_ids=np.arange(count),
             contacts=contact_counts, assumed_seat_load_n=load_history, support_clearance_m=floor_errors,
             seat_contact_hypothesis=seat_confidence)
    return poses, {'engine': 'Newton', 'engine_version': newton.__version__, 'warp_version': wp.__version__,
        'solver': 'XPBD', 'device': str(model.device), 'substeps_per_frame': 8, 'training': False,
        'dynamic_object': True, 'object_transform_overwritten_each_frame': False,
        'contact_frames': int((np.asarray(contact_counts)>0).sum()), 'body_mass_prior_kg': mass,
        'max_shift_from_visual_pose_m': float(shift.max()), 'max_rotation_from_visual_pose_deg': float(np.rad2deg(angles.max())),
        'minimum_floor_clearance_m': float(np.min(floor_errors)), 'maximum_floor_clearance_m': float(np.max(floor_errors)),
        'seat_hypothesis_frames': int((seat_confidence>.5).sum()),
        'load_policy': 'Up to 35 kg effective seated load: an assumed support probe, not measured force',
        'human_collision': 'Seat load proxy only; full articulated human collisions not implemented'}


def run_supported_scene(args):
    import torch
    import trimesh
    from lib_mhr import MHRLayer, MHR_PARAM_DIMS
    from lib_mhr.object_pose_frame import resolve_object_pose_frame
    from prep.mhr_export_utils import camera_calibration, load_edex
    from v2hoi.clips import load_clips
    from v2hoi.hoi_preview import render_full_hoi
    from v2hoi.object_refine import require_timeline
    clip = load_clips('track1', [args.episode], args.dataset_root)[0]
    identity_path = args.run/'native-inputs.json'
    identity_hash = sha256(identity_path)
    identity = json.loads(identity_path.read_text())
    if identity.get('provenance') != 'track1' or identity['episode'] != clip.episode or identity['video_sha256'] != sha256(clip.video):
        raise ValueError('Native input provenance/video mismatch')
    for path in (args.run, args.dataset_root, args.spec):
        if 'track_2' in {p.lower() for p in path.resolve().parts}:
            raise ValueError('Track 2 data is prohibited')
    root = args.run/'native'/f'episode_{clip.episode:06d}'
    source = root/'inference/refined.pth'
    source_hash = sha256(source)
    bundle = torch.load(source, map_location='cpu', weights_only=False)
    if bundle.get('gt') or bundle['metadata'].get('ground_truth_used') is not False:
        raise ValueError('Reference-bearing inputs are prohibited')
    require_timeline([int(row['src_frame']) for row in bundle['frame_meta']], clip.n_frames)
    export = root/'export'/f'episode_{clip.episode:06d}'
    k, extrinsic = camera_calibration(load_edex(export), 0)
    if not np.allclose(extrinsic, np.eye(4), atol=1e-6):
        raise ValueError('This adapter requires one unchanged camera-space scene')
    spec = json.loads(args.spec.read_text())
    if (spec.get('provenance') != 'track1' or spec.get('episode') != clip.episode
            or spec.get('video_sha256') != sha256(clip.video) or spec['landmarks']['original_frame_id'] != 0):
        raise ValueError('Authored geometry must identify this permitted original video')
    builder = Path('tools/windows/build_bpy_box_asset.py').resolve()
    watched = [source, identity_path, clip.video, args.spec, builder, root/'depth/moge2/aligned.h5',
               export/'wild_export.json', export/'human_masks/front_stereo_camera_left.h5',
               export/'object_masks/front_stereo_camera_left.h5', Path(bundle['metadata']['object_mesh']),
               Path(__file__), Path(__file__).with_name('physics_motion.py')]
    inputs = {str(path.resolve()): sha256(path) for path in watched}
    if inputs[str(source.resolve())] != source_hash or inputs[str(identity_path.resolve())] != identity_hash:
        raise ValueError('Shared upstream inputs changed while loading')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'spec.json').write_bytes(args.spec.read_bytes())
    masks, humans, affine, depth, excluded, box, evidence = observe(clip, root, output)
    normal, offset, floor_report = fit_floor(depth, excluded, k, box)
    pose, scales, fit_report = fit_supported_pose(spec, k, normal, offset)
    with (output/'blender.log').open('w') as log:
        subprocess.run([str(args.blender), '--background', '--factory-startup', '--python-exit-code', '1',
            '--python', str(builder), '--', '--spec', str(args.spec.resolve()), '--output', str(output/'object.glb')],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    with np.load(output/'object.npz') as arrays:
        mesh = trimesh.Trimesh(vertices=arrays['vertices']*scales, faces=arrays['faces'], vertex_colors=arrays['colors'], process=False)
    if not mesh.is_watertight or len(mesh.faces)>4096:
        raise ValueError('Invalid compound bpy surface')
    mesh, pose, scales, optimization_report = visual_refine(mesh, pose, scales, spec, normal, offset, k, masks, humans, affine, output)
    mesh.export(output/'object.glb')
    landmark_camera = (np.asarray(spec['landmarks']['points_local'])*scales) @ pose[:3, :3].T+pose[:3, 3]
    landmark_projection = landmark_camera @ k.T
    fit_report['median_landmark_error_px'] = float(np.median(np.linalg.norm(
        landmark_projection[:, :2]/landmark_projection[:, 2:]-np.asarray(spec['landmarks']['pixels']), axis=1)))
    asset = {'schema': 'v2hoi.observed-physical-asset.v1', 'provenance': 'track1', 'episode': clip.episode,
        'video_sha256': sha256(clip.video), 'entry': 'blender.bpy.observed-compound-boxes',
        'builder_sha256': inputs[str(builder.resolve())], 'spec_sha256': sha256(args.spec),
        'mesh_sha256': sha256(output/'object.glb'), 'triangles': len(mesh.faces),
        'canonical_origin': 'Support-plane centre; physical axis scales baked exactly once into final GLB',
        'fitted_axis_scales': scales.tolist(), 'reference_geometry_used': False, 'training': False,
        'metric_scale_verified': False}
    (output/'asset.json').write_text(json.dumps(asset, indent=2))
    visual_poses = np.repeat(pose[None], clip.n_frames, axis=0)
    visual_iou = audit_poses(mesh, visual_poses, k, masks, humans, affine)
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'], device='cuda')
    keypoints = []
    with torch.inference_mode():
        for i in range(0, clip.n_frames, 16):
            decoded = layer.mhr_forward({key: torch.as_tensor(np.asarray(bundle['pr'][key][i:i+16]), device='cuda') for key in MHR_PARAM_DIMS})
            keypoints.append(decoded.coco17.cpu().numpy())
    keypoints = np.concatenate(keypoints)
    del layer
    torch.cuda.empty_cache()
    poses, physics_report = simulate_supported(mesh, spec, pose, scales, normal, offset, keypoints, clip.fps, output)
    iou = audit_poses(mesh, poses, k, masks, humans, affine)
    if iou.mean() < visual_iou.mean()-.01 or iou[7::7].mean() < visual_iou[7::7].mean()-.01:
        raise ValueError('Physics degraded source-image agreement; reject the candidate')
    candidate = copy.deepcopy(bundle)
    candidate['pr']['pose_abs'] = poses.astype('float32')
    for key in ('pose_abs_postopt', 'pose_abs_1st_delta'):
        candidate['pr'].pop(key, None)
    candidate['metadata'].update(object_mesh=str(output/'object.glb'),
        object_pose_storage_frame='bpy_support_origin', object_mesh_to_pose_transform=np.eye(4).tolist(),
        object_pose_storage_to_training_transform=np.eye(4).tolist(), object_mesh_to_training_transform=np.eye(4).tolist(),
        physics_motion='observed-static-support-newton-xpbd-v1')
    # Derived caches and network-era contact statistics no longer describe the
    # candidate. They must not be mistaken for its authoritative scene state.
    candidate['postopt'] = {'mode': 'visual_physical_supported_object', 'source_contact_statistics_invalidated': True}
    candidate['pr'].pop('contact_logits', None)
    torch.save(candidate, output/'refined.pth')
    np.savez(output/'scene.npz', original_frame_ids=np.arange(clip.n_frames), object_poses=poses,
        visual_pose=pose, floor_normal=normal, floor_offset=offset, intrinsics=k,
        human_coco17=keypoints, object_vertices=np.asarray(mesh.vertices), object_faces=np.asarray(mesh.faces))
    # Diagnose old geometry only as an explicit historical comparison.
    old_mesh = trimesh.load(bundle['metadata']['object_mesh'], force='mesh', process=False)
    object_frame = resolve_object_pose_frame(bundle['metadata'])
    old_mesh.apply_transform(object_frame.mesh_to_training)
    old = np.asarray(bundle['pr']['pose_abs']) @ np.linalg.inv(object_frame.storage_to_training)
    old_iou = audit_poses(old_mesh, old, k, masks, humans, affine)
    # Isolate motion from the changed asset: replay the old relative motion on
    # the identical newly fitted mesh, sharing its initial observed alignment.
    old_relative = old @ np.linalg.inv(old[0]) @ pose
    same_mesh_old_motion_iou = audit_poses(mesh, old_relative, k, masks, humans, affine)
    np.savez(output/'visual-diagnostics.npz', original_frame_ids=np.arange(clip.n_frames),
        old_iou=old_iou, same_mesh_old_motion_iou=same_mesh_old_motion_iou,
        visual_only_iou=visual_iou, physics_iou=iou)
    before_jitter = np.linalg.norm(np.diff(old[:, :3, 3], n=2, axis=0), axis=1)
    preview = render_full_hoi(args.run, clip, output/'refined.pth', output, before_path=source)
    if inputs != {str(path.resolve()): sha256(path) for path in watched}:
        raise ValueError('An upstream input or runtime changed during the experiment')
    report = {'schema': 'v2hoi.visual-physical-supported-object.v1', 'provenance': 'track1', 'episode': clip.episode,
        'status': 'pilot_complete_requires_visual_review', 'frames': clip.n_frames,
        'original_frame_ids': list(range(clip.n_frames)), 'camera_identity_unchanged': True,
        'reference_alignment': 'No reference available or used; original source-frame identity is preserved',
        'training': False, 'reference_used': False, 'submission_conversion': False,
        'foundationpose_calls': 0, 'foundationpose_pose_used_for_fit': False,
        'human_parameters_unchanged': True, 'mesh_entry': 'blender.bpy.observed-compound-boxes',
        'inputs': inputs, 'upstream_inputs_unchanged': True, 'image_evidence': evidence,
        'floor': {**floor_report, 'normal': normal.tolist(), 'offset_m': offset},
        'visual_fit': {**fit_report, 'constant_axis_scales': scales.tolist(), 'optimization': optimization_report,
                       'visible_mask_iou_mean': float(iou.mean()),
                       'visual_only_mask_iou_mean': float(visual_iou.mean()),
                       'held_out_every_seventh_frame_iou_mean': float(iou[7::7].mean()),
                       'scope': 'Input segmentation diagnostic; no reconstruction ground truth'},
        'physics': physics_report, 'full_hoi': preview,
        'historical_comparison': {'mesh_generation': 'Existing SAM3D low-poly baseline, read-only comparison',
            'visible_mask_iou_mean': float(old_iou.mean()),
            'same_new_mesh_old_relative_motion_iou_mean': float(same_mesh_old_motion_iou.mean()),
            'translation_self_acceleration_p95_m_per_frame_squared': float(np.quantile(before_jitter, .95))},
        'limitations': ['Static-support branch only; dynamic clips fail the image-evidence gate',
            'Floor and dimensions inherit monocular depth scale uncertainty',
            'Pelvis proximity and assumed seated load are not measured contact forces',
            'Human motion is preserved with its existing errors; no articulated body collision correction',
            'Static supported motion is justified only within the reported image tolerance',
            'Smoother motion and fewer contacts do not establish better reconstruction accuracy']}
    report['outputs'] = {name: sha256(output/name) for name in ('refined.pth', 'object.glb', 'scene.npz', 'full-hoi.mp4', 'full-hoi-comparison.mp4')}
    (output/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'output': str(output), 'visual_fit': report['visual_fit'], 'physics': physics_report}, indent=2), flush=True)
