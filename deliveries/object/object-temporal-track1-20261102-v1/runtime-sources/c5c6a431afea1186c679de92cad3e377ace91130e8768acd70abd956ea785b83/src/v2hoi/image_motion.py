"""Source-image correspondences for fixed-geometry rigid motion inference.

OpenCV is imported only by the optional image tracker. Geometry helpers and
receipt validation remain usable in the independent CPU test environment.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def project_points(points, pose, intrinsics):
    camera = np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]
    homogeneous = camera @ np.asarray(intrinsics).T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def surface_points(pixels, intrinsics, pose, vertices, faces):
    """Nearest positive ray hits, in the actual canonical mesh coordinates.

    No bounding box surrogate, spatial-index dependency or hidden geometry data.
    Misses are NaNs and must never become image correspondences.
    """
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(intrinsics).T @ pose[:3, :3]
    origin = -pose[:3, :3].T @ pose[:3, 3]
    triangles = np.asarray(vertices)[np.asarray(faces)]
    a = triangles[:, 0]
    e1, e2 = triangles[:, 1] - a, triangles[:, 2] - a
    displacement = origin - a
    q = np.cross(displacement, e1)
    result = np.full((len(pixels), 3), np.nan)
    for start in range(0, len(pixels), 64):
        directions = rays[start:start + 64]
        p = np.cross(directions[:, None], e2)
        determinant = np.einsum('rtj,tj->rt', p, e1)
        inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=np.abs(determinant) > 1e-10)
        u = np.einsum('tj,rtj->rt', displacement, p) * inverse
        v = np.einsum('rj,tj->rt', directions, q) * inverse
        distance = np.einsum('tj,tj->t', e2, q)[None] * inverse
        valid = (np.abs(determinant) > 1e-10) & (u >= 0) & (v >= 0) & (u + v <= 1) & (distance > 0)
        distance = np.where(valid, distance, np.inf)
        nearest = distance.argmin(1)
        length = distance[np.arange(len(directions)), nearest]
        hit = np.isfinite(length)
        result[start + np.flatnonzero(hit)] = origin + directions[hit] * length[hit, None]
    return result


def interpolate_poses(frame_ids, poses, count):
    """Explicit estimates through unsupported intervals; not stationary truth."""
    ids = np.asarray(frame_ids)
    poses = np.asarray(poses)
    if (ids.ndim != 1 or not len(ids) or poses.shape != (len(ids), 4, 4)
            or (np.diff(ids) <= 0).any() or ids[0] < 0 or ids[-1] >= count):
        raise ValueError('Require ordered original-frame pose anchors')
    timeline = np.arange(count)
    result = np.repeat(np.eye(4)[None], count, axis=0)
    if len(ids) == 1:
        result[:] = poses[0]
    else:
        result[:, :3, :3] = Slerp(ids, Rotation.from_matrix(poses[:, :3, :3]))(
            np.clip(timeline, ids[0], ids[-1])).as_matrix()
        for axis in range(3):
            result[:, axis, 3] = np.interp(timeline, ids, poses[:, axis, 3])
    return result


def solve_landmark_anchors(landmarks, intrinsics, fallback):
    """Estimate anchor poses from RGB correspondences instead of copying a prior.

    A poor landmark fit supplies no verified feature attachment. It must remain
    an explicit missing anchor rather than becoming hundreds of false tracks.
    """
    import cv2
    anchors = {}
    diagnostics = []
    for frame, (points, pixels) in sorted(landmarks.items()):
        success, vectors, translations, _ = cv2.solvePnPGeneric(
            np.asarray(points, dtype=np.float64), np.asarray(pixels, dtype=np.float64),
            np.asarray(intrinsics, dtype=np.float64), None, flags=cv2.SOLVEPNP_SQPNP)
        if not success: continue
        candidates = []
        for rv, tv in zip(vectors, translations):
            pose = np.eye(4); pose[:3, :3] = cv2.Rodrigues(rv)[0]; pose[:3, 3] = tv.ravel()
            if (points @ pose[:3, :3].T + pose[:3, 3])[:, 2].min() <= .05: continue
            error = np.linalg.norm(project_points(points, pose, intrinsics) - pixels, axis=1)
            angle = Rotation.from_matrix(pose[:3, :3] @ fallback[frame, :3, :3].T).magnitude()
            candidates.append((float(np.median(error) + .03 * angle), pose, error))
        if not candidates: continue
        _, pose, error = min(candidates, key=lambda row: row[0])
        accepted = np.median(error) < 3 and np.max(error) < 6
        diagnostics.append({'original_frame_id': int(frame), 'median_error_px': float(np.median(error)),
            'accepted': bool(accepted)})
        if accepted: anchors[frame] = pose
    return anchors, diagnostics


def curve_observations(asset, frames):
    """Validate visible image arcs against receipted original source frames.

    Points describe only visible pixels, never an invented hidden ellipse.
    Local circles are agent-authored geometric hypotheses on the bpy asset.
    """
    result = {}
    source_ids = {row['original_frame_id'] for row in asset.get('observations', [])} if asset else set()
    for row in asset.get('curve_observations', []) if asset else []:
        frame = row['original_frame_id']
        center = np.asarray(row['center_local'], dtype=float)
        normal = np.asarray(row['normal_local'], dtype=float)
        pixels = np.asarray(row['pixels'], dtype=float)
        radius = float(row['radius'])
        if (frame not in source_ids or not 0 <= frame < frames or frame in result
                or center.shape != (3,) or normal.shape != (3,) or not np.isfinite(center).all()
                or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1.)
                or not np.isfinite(radius) or radius <= 0 or pixels.ndim != 2
                or pixels.shape[1:] != (2,) or len(pixels) < 6 or not np.isfinite(pixels).all()
                or np.linalg.matrix_rank(pixels - pixels.mean(0)) < 2):
            raise ValueError('Curves require finite visible arcs and permitted original RGB observations')
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        tangent = np.cross(normal, axis); tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        angles = np.linspace(0, 2 * np.pi, 192, endpoint=False)
        points = center + radius * (np.cos(angles)[:, None] * tangent + np.sin(angles)[:, None] * bitangent)
        result[frame] = (points, pixels)
    return result


def circle_pixel_residual(points, pixels, pose, intrinsics):
    """First-order pixel distance to a projected circle's continuous conic."""
    center = points.mean(0)
    axes = np.stack([points[0] - center, points[len(points)//4] - center], axis=1)
    h = np.asarray(intrinsics) @ np.column_stack((pose[:3, :3] @ axes,
        pose[:3, :3] @ center + pose[:3, 3]))
    inverse = np.linalg.pinv(h)
    conic = inverse.T @ np.diag([1., 1., -1.]) @ inverse
    homogeneous = np.c_[pixels, np.ones(len(pixels))]
    gradient = homogeneous @ conic
    value = np.sum(gradient * homogeneous, axis=1)
    return value / np.maximum(2 * np.linalg.norm(gradient[:, :2], axis=1), 1e-15)


def fit_curve_anchors(curves, intrinsics, fallback, semantic_axis=None, face_targets=None):
    """Multi-start visible-arc fitting, with no rotation lock to the old pose.

    Image arcs leave yaw and hidden arcs ambiguous. A weak pose preference
    resolves equivalent solutions; final silhouette/handle fitting remains
    necessary. Translation has a bounded image neighborhood for partial arcs.
    """
    from scipy.optimize import least_squares
    estimates = {}
    diagnostics = []
    alternatives = []
    candidate_costs = []
    k = np.asarray(intrinsics)
    for frame, (points, pixels) in sorted(curves.items()):
        reference = fallback[frame]
        center_local = points.mean(0)
        center = reference[:3, :3] @ center_local + reference[:3, 3]
        center_pixel = project_points(center_local[None], reference, k)[0]
        reference_rotation = Rotation.from_matrix(reference[:3, :3])
        def state(x):
            rotation = Rotation.from_rotvec(x[:3]).as_matrix()
            uv = center_pixel + x[3:5]
            z = center[2] * np.exp(x[5])
            camera_center = np.linalg.inv(k) @ np.r_[uv, 1.] * z
            pose = np.eye(4);pose[:3, :3] = rotation
            pose[:3, 3] = camera_center - rotation @ center_local
            return pose
        def residual(x):
            pose = state(x)
            arc = circle_pixel_residual(points, pixels, pose, k) / 1.5
            angle = (Rotation.from_matrix(pose[:3, :3]) * reference_rotation.inv()).as_rotvec()
            prior = np.r_[angle * .25, x[3:5] * .025, x[5] * .3]
            sign = 0. if face_targets is None else face_targets[frame]
            facing = 0. if semantic_axis is None else np.dot(pose[:3, :3] @ semantic_axis, -pose[:3, 3] / np.linalg.norm(pose[:3, 3]))
            return np.r_[arc, prior, max(0., .1 - sign * facing) * 8 if sign else 0.]
        candidates = []
        starts = Rotation.create_group('O') * reference_rotation
        for rotation in starts:
            initial = np.r_[rotation.as_rotvec(), 0., 0., 0.]
            fit = least_squares(residual, initial, bounds=(np.r_[[-np.inf]*3, -8., -8., -.2],
                np.r_[[np.inf]*3, 8., 8., .2]), loss='soft_l1', f_scale=1., max_nfev=100)
            candidates.append((np.mean(np.square(residual(fit.x))), state(fit.x)))
        alternatives.append(np.stack([row[1] for row in candidates]))
        candidate_costs.append(np.array([row[0] for row in candidates]))
    from v2hoi.object_refine import select_orientation_path
    ids = sorted(curves)
    branches = select_orientation_path(np.asarray(alternatives)[:, :, :3, :3], candidate_costs, ids,
        angular_step_deg=5., weight=.15)
    for frame, candidates, selected in zip(ids, alternatives, branches):
        pose = candidates[selected]
        points, pixels = curves[frame]
        reference_rotation = Rotation.from_matrix(fallback[frame, :3, :3])
        estimates[frame] = pose
        errors = np.abs(circle_pixel_residual(points, pixels, pose, k))
        diagnostics.append({'original_frame_id': frame, 'visible_arc_median_error_px': float(np.median(errors)),
            'rotation_change_deg': float(np.rad2deg((Rotation.from_matrix(pose[:3, :3]) * reference_rotation.inv()).magnitude()))})
    return interpolate_poses(ids, [estimates[i] for i in ids], len(fallback)), diagnostics


def track_surface_features(crops, masks, humans, roi_k, boxes, intrinsics,
                           mesh, anchors, fallback, fps=30.):
    """Bidirectional KLT plus source-descriptor relocalization and robust PnP.

    All surface identities originate in verified image/mesh anchor pairs. They
    are never reattached to an unverified tracker prediction. Human pixels,
    forward/backward failures and PnP outliers are excluded. Weak frames retain
    explicit interpolation estimates and have no fabricated feature evidence.
    """
    import cv2
    cv2.setRNGSeed(0)
    grays = [cv2.cvtColor(np.uint8(np.clip(rgb * 255, 0, 255)), cv2.COLOR_RGB2GRAY) for rgb in crops]
    support = [cv2.erode(np.uint8(m > .5), np.ones((3, 3), np.uint8)) &
               ~cv2.dilate(np.uint8(h), np.ones((3, 3), np.uint8)) for m, h in zip(masks, humans)]
    sift = cv2.SIFT_create(nfeatures=600, contrastThreshold=.012, edgeThreshold=12)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    observations = {}
    estimates = {}
    diagnostics = []
    count = len(crops)
    for anchor_id, anchor_pose in sorted(anchors.items()):
        keypoints, descriptors = sift.detectAndCompute(grays[anchor_id], support[anchor_id] * 255)
        if descriptors is None or len(keypoints) < 8:
            diagnostics.append({'anchor_frame': int(anchor_id), 'status': 'insufficient source texture'})
            continue
        pixels = np.array([point.pt for point in keypoints], dtype=np.float32)
        local = surface_points(pixels, roi_k[anchor_id], anchor_pose, mesh.vertices, mesh.faces)
        valid = np.isfinite(local).all(1)
        local, pixels, descriptors = local[valid], pixels[valid], descriptors[valid]
        if len(local) < 8:
            continue
        for direction in (-1, 1):
            last_id = anchor_id
            last_pixels = pixels.copy()
            last_indices = np.arange(len(local))
            last_pose = anchor_pose.copy()
            last_pose_frame = anchor_id
            for frame in range(anchor_id, -1 if direction < 0 else count, direction):
                found = {}
                if frame == anchor_id:
                    found = dict(enumerate(pixels))
                else:
                    if len(last_pixels):
                        forward, good, _ = cv2.calcOpticalFlowPyrLK(grays[last_id], grays[frame],
                            last_pixels[:, None], None, winSize=(21, 21), maxLevel=3,
                            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))
                        backward, back_good, _ = cv2.calcOpticalFlowPyrLK(grays[frame], grays[last_id],
                            forward, None, winSize=(21, 21), maxLevel=3)
                        good = good.ravel().astype(bool) & back_good.ravel().astype(bool)
                        good &= np.linalg.norm(backward[:, 0] - last_pixels, axis=1) < .65
                        for index, point, keep in zip(last_indices, forward[:, 0], good):
                            if keep: found[int(index)] = point
                    # Absolute source matches recover identities after occlusion;
                    # a low ratio avoids repetitive-pattern matches on the cube.
                    keys, desc = sift.detectAndCompute(grays[frame], support[frame] * 255)
                    if desc is not None and len(desc) >= 2:
                        for pair in matcher.knnMatch(descriptors, desc, k=2):
                            if len(pair) == 2 and pair[0].distance < .65 * pair[1].distance:
                                match = pair[0]
                                # Repeated texture can produce a plausible absolute
                                # match on another face. Never replace a surviving
                                # forward/backward-checked temporal identity.
                                found.setdefault(match.queryIdx, np.array(keys[match.trainIdx].pt, dtype=np.float32))
                ids, current = [], []
                for index, point in found.items():
                    x, y = np.rint(point).astype(int)
                    if 0 <= y < support[frame].shape[0] and 0 <= x < support[frame].shape[1] and support[frame][y, x]:
                        ids.append(index); current.append(point)
                ids = np.asarray(ids, dtype=int)
                current = np.asarray(current, dtype=np.float32).reshape(-1, 2)
                last_id, last_pixels, last_indices = frame, current, ids
                if len(ids) < 8: continue
                original = current * boxes[frame][2] / grays[frame].shape[0] + np.array(boxes[frame][:2])
                success, rv, tv, inliers = cv2.solvePnPRansac(local[ids], original, intrinsics, None,
                    iterationsCount=250, reprojectionError=1.6, confidence=.999, flags=cv2.SOLVEPNP_EPNP)
                if not success or inliers is None or len(inliers) < 8: continue
                inliers = inliers.ravel()
                if len(inliers) < .5 * len(ids): continue
                uv, xyz = original[inliers], local[ids[inliers]]
                # Small clusters constrain translation, but do not identify 3D orientation.
                if np.linalg.svd(uv - uv.mean(0), compute_uv=False)[-1] / np.sqrt(len(uv)) < 4: continue
                spread=np.linalg.svd(xyz-xyz.mean(0),compute_uv=False)
                planar=spread[-1]<.04*spread[0]
                gap=max(1,abs(frame-last_pose_frame))
                candidates=[]
                # Planar PnP has competing low-error branches. Refine from both
                # the independent RANSAC pose and the last accepted physical pose.
                for initial_r,initial_t in [(rv,tv),(cv2.Rodrigues(last_pose[:3,:3])[0],last_pose[:3,3,None].copy())]:
                    fit_r,fit_t=cv2.solvePnPRefineLM(xyz,uv,intrinsics,None,initial_r.copy(),initial_t.copy())
                    pose=np.eye(4);pose[:3,:3]=cv2.Rodrigues(fit_r)[0];pose[:3,3]=fit_t.ravel()
                    if (xyz@pose[:3,:3].T+pose[:3,3])[:,2].min()<=.05:continue
                    error=np.linalg.norm(project_points(xyz,pose,intrinsics)-uv,axis=1)
                    angle=Rotation.from_matrix(pose[:3,:3]@last_pose[:3,:3].T).magnitude()
                    # This broad physical bound rejects unsupported branch jumps,
                    # not real motion inferred from smooth pixel trajectories.
                    if frame!=anchor_id and (angle>np.deg2rad(720./fps)*gap
                            or abs(np.log(pose[2,3]/last_pose[2,3]))>.12*gap):continue
                    if np.median(error)>1.2:continue
                    candidates.append((float(np.median(error)+(1.5 if planar else .4)*angle),pose))
                if not candidates:continue
                _,pose=min(candidates,key=lambda row:row[0])
                last_pose = pose
                last_pose_frame = frame
                if frame not in observations or len(xyz) > len(observations[frame][0]):
                    observations[frame] = (xyz, uv)
                    estimates[frame] = pose
                last_pixels, last_indices = current[inliers], ids[inliers]
        diagnostics.append({'anchor_frame': int(anchor_id), 'surface_features': len(local)})
    ids = sorted(estimates)
    poses = interpolate_poses(ids, [estimates[i] for i in ids], count) if ids else fallback.copy()
    return poses, observations, {'anchors': diagnostics, 'supported_frame_ids': ids,
        'unsupported_frame_ids': sorted(set(range(count)) - set(ids)),
        'maximum_supported_angular_speed_deg_per_second':720.,
        'policy': 'Source-anchored features; bidirectional KLT and PnP branch continuity checks; no predicted-pose reattachment; faster ambiguous motion remains unsupported'}
