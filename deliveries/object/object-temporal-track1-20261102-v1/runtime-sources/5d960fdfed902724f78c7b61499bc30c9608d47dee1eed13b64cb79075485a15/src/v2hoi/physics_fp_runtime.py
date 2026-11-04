"""FoundationPose initialization without authored image corners or pose seeds.

The authored bpy shape, ROI and contact/support intervals remain explicit inputs.
Monocular depth supplies uncertain metric scale; no reference geometry is used.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from v2hoi.native import sha256


def require_unannotated_spec(spec):
    """Do not silently consume source-corner calibration in this comparison."""
    if spec.get('landmarks') or spec.get('motion_anchors'):
        raise ValueError('FoundationPose comparison prohibits authored image landmarks and motion anchors')
    if spec.get('provenance') != 'track1':
        raise ValueError('Require permitted Track 1 observations')


def depth_height_scale(depth, mask, k, normal, offset, template_height):
    """One shared scale from the visible supported object's depth above the floor."""
    y, x = np.where(np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > .1))
    if len(x) < 50 or template_height <= 0:
        raise ValueError('Insufficient supported-object depth for automatic scale')
    z = depth[y, x]
    xyz = np.column_stack(((x-k[0, 2])*z/k[0, 0], (y-k[1, 2])*z/k[1, 1], z))
    heights = offset-xyz@normal
    height = float(np.quantile(heights, .95))
    scale = height/template_height
    if not .25 < scale < 4.:
        raise ValueError(f'Unreliable depth-derived object scale: {scale}')
    return scale, {'method': '95th percentile visible object height above estimated floor',
                   'observed_height_m': height, 'template_height_m': template_height,
                   'uniform_scale': scale, 'metric_reference': False}


def centered_tracking_pose(pose, mesh_to_centered):
    """FoundationPose pose_last uses centered mesh coordinates, not bpy origin."""
    return np.asarray(pose)@np.linalg.inv(mesh_to_centered)


def grounded_pose(pose, normal, offset):
    """Use the annotated upright support state, without fixing object yaw."""
    from v2hoi.physics_motion import plane_frame
    frame, _ = plane_frame(normal, offset)
    local = frame.T@pose[:3, :3]
    yaw = np.arctan2(local[1, 0], local[0, 0])
    result = pose.copy()
    result[:3, :3] = frame@Rotation.from_euler('z', yaw).as_matrix()
    result[:3, 3] += normal*(offset-result[:3, 3]@normal)
    return result


def track_foundationpose(clip, root, mesh, images, masks, humans, affine, k, normal, offset, spec, output, completion=None):
    import cv2
    import torch
    import Utils
    from prep.mhr_depth_h5 import read_metric_depth
    from prep.run_foundationpose_mhr_export import _pose_estimator
    from v2hoi.object_refine_runtime import ObservationRenderer
    camera = k.copy()
    camera[:2] = affine[:, :2]@k[:2]
    camera[:2, 2] += affine[:, 2]
    renderer = ObservationRenderer(mesh)
    Utils.set_seed(0)
    logging.getLogger().setLevel(logging.WARNING)
    estimator = _pose_estimator(mesh.copy(), output/'foundationpose-debug')
    transform = estimator.get_tf_to_centered_mesh().cpu().numpy()
    depth_path = root/'depth/moge2/aligned.h5'
    poses, raw, records, anchors = [], [], [], {}
    calls = {'register': 0, 'track': 0}

    def measure(pose, index):
        with torch.inference_mode():
            _, alpha, _ = renderer.render(torch.tensor(pose[None, :3, :3], dtype=torch.float32, device='cuda'),
                torch.tensor(pose[None, :3, 3], dtype=torch.float32, device='cuda'), 1.,
                torch.zeros(3, device='cuda'), camera[None], masks.shape[1:])
        valid = ~humans[index].astype(bool)
        a, b = (alpha[0, ..., 0].cpu().numpy() > .5)&valid, masks[index].astype(bool)&valid
        return float((a&b).sum()/max(1, (a|b).sum()))

    for i, rgb in enumerate(images):
        depth = read_metric_depth(depth_path, 'aligned', 'front_stereo_camera_left', f'{i:06d}')
        if depth.shape != (clip.height, clip.width):
            depth = cv2.resize(depth, (clip.width, clip.height), interpolation=cv2.INTER_NEAREST)
        depth = cv2.warpAffine(depth.astype('float32'), affine, masks.shape[1:][::-1], flags=cv2.INTER_NEAREST)
        depth = np.nan_to_num(depth, nan=0., posinf=0., neginf=0.)
        depth[(depth < .001)|(depth > 8.)] = 0.
        visible = masks[i].astype(bool)&~humans[i].astype(bool)
        if completion is not None:
            from v2hoi.amodal_observations import appearance_proposal
            rgb, depth = appearance_proposal(rgb, depth, visible, humans[i], completion['rgb'][i], completion['primary'][i])
        registration = i == 0
        with torch.inference_mode():
            if registration:
                if (visible&(depth > .001)).sum() < 50:
                    raise ValueError('No usable first-frame object depth')
                proposed = estimator.register(K=camera, rgb=rgb, depth=depth, ob_mask=visible, iteration=5)
                calls['register'] += 1
            else:
                proposed = estimator.track_one(rgb=rgb, depth=depth, K=camera, iteration=4, extra={})
                calls['track'] += 1
        if not np.isfinite(proposed).all():
            raise ValueError(f'Non-finite FoundationPose output at original frame {i}')
        raw.append(proposed.copy())
        iou = measure(proposed, i)
        # Inspect ranked registration hypotheses using visible masks and the
        # existing support-state prior. No source corner or old trajectory seed.
        if i > 0 and i % 45 == 0 and iou < .45 and visible.sum() >= 100:
            with torch.inference_mode():
                registered = estimator.register(K=camera, rgb=rgb, depth=depth, ob_mask=visible, iteration=5)
            calls['register'] += 1
            candidate_iou = measure(registered, i)
            if candidate_iou > iou+.05:
                proposed, iou, registration = registered, candidate_iou, True
        if registration and (i <= spec['events']['initial_support_end'] or i >= spec['events']['final_support_start']):
            hypotheses = (estimator.poses.detach().cpu().numpy()[:32]@transform)
            supported = []
            for candidate in hypotheses:
                up = float(candidate[:3, 2]@(-normal))
                if up > .75:
                    candidate = grounded_pose(candidate, normal, offset)
                    supported.append((measure(candidate, i), candidate))
            if supported:
                iou, proposed = max(supported, key=lambda row: row[0])
        if i <= spec['events']['initial_support_end']:
            proposed = poses[0].copy() if poses else grounded_pose(proposed, normal, offset)
        elif i >= spec['events']['final_support_start']:
            proposed = poses[spec['events']['final_support_start']].copy() if i > spec['events']['final_support_start'] else grounded_pose(proposed, normal, offset)
        estimator.pose_last = torch.tensor(centered_tracking_pose(proposed, transform), dtype=torch.float32, device='cuda')
        poses.append(proposed.copy())
        iou = measure(proposed, i)
        accepted_anchor = iou >= .4 and (i % 45 == 0 or i == clip.n_frames-1 or registration)
        if accepted_anchor:
            anchors[i] = proposed.copy()
        records.append({'original_frame_id': i, 'visible_mask_iou': iou, 'registered': registration,
                        'surface_attachment_anchor': accepted_anchor})
        if i % 30 == 0:
            np.savez(output/'foundationpose-progress.npz', raw_poses=raw, poses=poses, original_frame_ids=np.arange(i+1))
            print(f'PHYSICS_FOUNDATIONPOSE frame={i}/{clip.n_frames} iou={iou:.3f} calls={calls}', flush=True)
    del estimator, renderer
    torch.cuda.empty_cache()
    poses = np.asarray(poses)
    np.savez(output/'foundationpose.npz', raw_poses=raw, poses=poses, original_frame_ids=np.arange(clip.n_frames))
    report = {'calls': calls, 'frames': records, 'authored_image_correspondences': 0,
              'amodal_rgb_used': completion is not None,
              'generated_metric_depth_used': False,
              'human_depth_invalidated': completion is not None,
              'source_corner_scale_calibration': False, 'native_object_pose_seeds': False,
              'support_intervals_agent_authored': True, 'candidate_anchor_threshold_iou': .4,
              'recovery_policy': 'Try registration every 45 frames below 0.45 visible-mask IoU; accept improvement over 0.05',
              'anchor_scope': 'Pose-backed feature attachment hypotheses, not independent source landmark labels'}
    (output/'foundationpose.json').write_text(json.dumps(report, indent=2))
    return poses, anchors, report


def run_fp_scene(args):
    import torch
    import trimesh
    from lib_mhr import MHRLayer, MHR_PARAM_DIMS
    from prep.mhr_export_utils import camera_calibration, load_edex
    from v2hoi.clips import load_clips
    from v2hoi.hoi_preview import render_full_hoi
    from v2hoi.image_motion import track_surface_features
    from v2hoi.object_refine import require_timeline
    from v2hoi.physics_dynamic_runtime import observations, refine_trajectory, simulate_grasp
    from v2hoi.physics_motion_runtime import fit_floor, audit_poses
    for path in (args.run, args.spec, args.dataset_root, args.output):
        if 'track_2' in {p.lower() for p in path.resolve().parts}:
            raise ValueError('Track 2 is prohibited')
    clip = load_clips('track1', [args.episode], args.dataset_root)[0]
    spec = json.loads(args.spec.read_text())
    require_unannotated_spec(spec)
    identity = json.loads((args.run/'native-inputs.json').read_text())
    if (spec['episode'] != clip.episode or spec['video_sha256'] != sha256(clip.video)
            or identity.get('provenance') != 'track1' or identity['video_sha256'] != spec['video_sha256']):
        raise ValueError('Permitted source video identity mismatch')
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
    fp_root = Path('third_party/CARI4D_MHR_Win/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose')
    watched = [source, args.run/'native-inputs.json', args.spec, clip.video, builder,
        root/'depth/moge2/aligned.h5', export/'wild_export.json',
        export/'object_masks/front_stereo_camera_left.h5', export/'human_masks/front_stereo_camera_left.h5',
        Path(__file__), Path(__file__).with_name('physics_motion.py'),
        Path(__file__).with_name('physics_dynamic_runtime.py'), Path(__file__).with_name('physics_motion_runtime.py'),
        Path(__file__).with_name('image_motion.py'), Path(__file__).with_name('object_refine_runtime.py'),
        fp_root/'estimater.py']
    weights = Path(os.environ['FOUNDATIONPOSE_WEIGHTS_DIR'])
    watched.extend(sorted(weights.rglob('*.pth')))
    amodal_cache = getattr(args, 'amodal_cache', None)
    completion = None
    if amodal_cache is not None:
        cache_report = json.loads((amodal_cache/'result.json').read_text())
        if (cache_report.get('provenance') != 'track1' or cache_report.get('reference_used') is not False
                or cache_report['video_sha256'] != spec['video_sha256'] or cache_report['frames'] != clip.n_frames):
            raise ValueError('Completion cache source identity mismatch')
        if sha256(amodal_cache/'tracking.npz') != cache_report['outputs']['tracking.npz']:
            raise ValueError('Completion cache hash mismatch')
        with np.load(amodal_cache/'tracking.npz') as data:
            np.testing.assert_array_equal(data['original_frame_ids'], np.arange(clip.n_frames))
            completion = {key: data[key] for key in ('rgb', 'primary', 'affine')}
        watched.extend([amodal_cache/'result.json', amodal_cache/'tracking.npz', Path(__file__).with_name('amodal_observations.py')])
    if not any(p.suffix == '.pth' for p in watched if p.is_relative_to(weights)):
        raise ValueError('Missing local FoundationPose weights')
    hashes = {str(p.resolve()): sha256(p) for p in watched}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'spec.json').write_bytes(args.spec.read_bytes())
    images, masks, humans, affine, depth, excluded, bbox, receipts = observations(clip, root, dict(spec, motion_anchors=[]), output)
    if completion is not None:
        np.testing.assert_allclose(completion['affine'], affine, atol=1e-6)
        if completion['rgb'].shape != images.shape or completion['primary'].shape != masks.shape:
            raise ValueError('Completion cache camera/crop mismatch')
    normal, offset, floor_report = fit_floor(depth, excluded, k, bbox)
    visible_first = (masks[0] > 0)&~humans[0].astype(bool)
    import cv2
    small_depth = cv2.warpAffine(depth, affine, masks.shape[1:][::-1], flags=cv2.INTER_NEAREST)
    camera = k.copy(); camera[:2] = affine[:, :2]@k[:2]; camera[:2, 2] += affine[:, 2]
    with (output/'blender.log').open('w') as log:
        subprocess.run([str(args.blender), '--background', '--factory-startup', '--python-exit-code', '1', '--python', str(builder),
            '--', '--spec', str(args.spec.resolve()), '--output', str(output/'object.glb')], stdout=log, stderr=subprocess.STDOUT, check=True)
    with np.load(output/'object.npz') as arrays:
        mesh = trimesh.Trimesh(vertices=arrays['vertices'], faces=arrays['faces'], vertex_colors=arrays['colors'], process=False)
    scale, shape_report = depth_height_scale(small_depth, visible_first, camera, normal, offset, float(mesh.bounds[1, 2]-mesh.bounds[0, 2]))
    scales = np.full(3, scale)
    mesh.vertices *= scales
    mesh.export(output/'object.glb')
    print(f'PHYSICS_FP_SCALE {shape_report}', flush=True)
    initial, anchors, fp_report = track_foundationpose(clip, root, mesh, images, masks, humans, affine, k, normal, offset, spec, output, completion=completion)
    boxes = np.repeat(np.asarray(spec['observation_roi'])[None], clip.n_frames, axis=0)
    _, features, feature_report = track_surface_features(images.astype('float32')/255., masks, humans,
        np.repeat(camera[None], clip.n_frames, axis=0), boxes, k, mesh, anchors, initial, clip.fps)
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ['MHR_ASSETS_ROOT'], device='cuda')
    keypoints = []
    with torch.inference_mode():
        for i in range(0, clip.n_frames, 16):
            decoded = layer.mhr_forward({key: torch.as_tensor(np.asarray(bundle['pr'][key][i:i+16]), device='cuda') for key in MHR_PARAM_DIMS})
            keypoints.append(decoded.coco17.cpu().numpy())
    keypoints = np.concatenate(keypoints)
    visual, wrist, confidence, fit_report = refine_trajectory(mesh, initial, {}, features, masks, humans, affine,
        k, keypoints, spec, scales, normal, offset, output)
    poses, physics_report = simulate_grasp(mesh, spec, scales, visual, wrist, confidence, normal, offset, clip.fps, output)
    fp_iou = audit_poses(mesh, initial, k, masks, humans, affine)
    visual_iou = audit_poses(mesh, visual, k, masks, humans, affine)
    physical_iou = audit_poses(mesh, poses, k, masks, humans, affine)
    candidate = copy.deepcopy(bundle)
    candidate['pr']['pose_abs'] = poses.astype('float32')
    for key in ('pose_abs_postopt', 'pose_abs_1st_delta', 'contact_logits'):
        candidate['pr'].pop(key, None)
    candidate['metadata'].update(object_mesh=str(output/'object.glb'), object_pose_storage_frame='bpy_support_origin',
        object_mesh_to_pose_transform=np.eye(4).tolist(), object_pose_storage_to_training_transform=np.eye(4).tolist(),
        object_mesh_to_training_transform=np.eye(4).tolist(), physics_motion='foundationpose-actuated-grasp-newton-v1')
    candidate['postopt'] = {'mode': 'foundationpose_visual_physical_dynamic_object', 'source_contact_statistics_invalidated': True}
    if completion is not None:
        candidate['metadata']['physics_motion'] = 'contact-amodal-foundationpose-actuated-grasp-newton-v1'
        candidate['postopt']['amodal_rgb_cache'] = str(amodal_cache.resolve())
    torch.save(candidate, output/'refined.pth')
    np.savez(output/'scene.npz', original_frame_ids=np.arange(clip.n_frames), object_poses=poses, visual_poses=visual,
        floor_normal=normal, floor_offset=offset, intrinsics=k, human_coco17=keypoints,
        object_vertices=np.asarray(mesh.vertices), object_faces=np.asarray(mesh.faces))
    np.savez(output/'visual-diagnostics.npz', foundationpose_iou=fp_iou, visual_only_iou=visual_iou, physics_iou=physical_iou)
    preview = render_full_hoi(args.run, clip, output/'refined.pth', output, before_path=source)
    if hashes != {str(p.resolve()): sha256(p) for p in watched}:
        raise ValueError('Input or runtime changed during this run')
    moving = slice(spec['events']['initial_support_end']+1, spec['events']['final_support_start'])
    diagnostics = {}
    for name, values in (('foundationpose', fp_iou), ('visual_only', visual_iou), ('physical', physical_iou)):
        diagnostics[name+'_iou'] = float(values.mean())
        diagnostics[name+'_moving_iou'] = float(values[moving].mean())
    diagnostics['scope'] = 'Estimated input-mask agreement, not reference pose accuracy; geometry/scale differ from corner-calibrated pilot'
    report = {'schema': 'v2hoi.foundationpose-physical-object.v1', 'provenance': 'track1', 'episode': clip.episode,
        'status': 'pilot_complete_requires_review', 'frames': clip.n_frames, 'original_frame_ids': list(range(clip.n_frames)),
        'inputs': hashes, 'upstream_inputs_unchanged': True, 'observations': receipts,
        'training': False, 'reference_used': False, 'submission_conversion': False, 'human_parameters_unchanged': True,
        'camera_identity_unchanged': True, 'foundationpose_calls': sum(fp_report['calls'].values()), 'foundationpose_used_for_fit': True,
        'amodal_rgb_used': completion is not None, 'generated_metric_depth_used': False,
        'authored_image_correspondences': 0, 'source_corner_scale_calibration': False,
        'floor': {**floor_report, 'normal': normal.tolist(), 'offset_m': offset}, 'shape_fit': shape_report,
        'foundationpose': fp_report, 'feature_tracking': feature_report, 'trajectory_fit': fit_report,
        'physics': physics_report, 'full_hoi': preview, 'visual_diagnostics': diagnostics,
        'limitations': ['bpy shape, source ROI and grasp/support intervals remain agent-authored',
            'Automatic scale uses uncertain monocular depth; no source-corner proportion calibration',
            'Feature attachment is conditional on predicted pose and can be wrong under symmetry or occlusion',
            'Simplified actuated grasp; no frictional fingers or full human collision',
            'Original human errors persist; physics does not establish video fidelity']}
    names = ('refined.pth', 'object.glb', 'scene.npz', 'full-hoi.mp4', 'full-hoi-comparison.mp4',
             'foundationpose.npz', 'foundationpose.json', 'visual-diagnostics.npz')
    report['outputs'] = {name: sha256(output/name) for name in names}
    (output/'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'output': str(output), 'visual': diagnostics, 'physics': physics_report}, indent=2), flush=True)
