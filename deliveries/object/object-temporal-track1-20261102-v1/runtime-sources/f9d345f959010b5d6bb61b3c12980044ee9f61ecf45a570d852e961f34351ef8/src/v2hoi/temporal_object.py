"""Observation-driven helpers for unattended, fixed-mesh temporal reconstruction.

These helpers accept only source measurements or explicitly uncertain model
predictions. They do not read challenge references, old object poses or labels.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def bpy_asset_paths(spec, spec_path):
    """Resolve a receipted one-time bpy asset, excluding unknown mesh provenance."""
    from pathlib import Path
    from v2hoi.native import sha256
    asset = spec.get('bpy_asset')
    if asset is None:
        return None
    if (spec.get('boxes') or asset.get('method') != 'agent_bpy' or asset.get('units') != 'm'
            or asset.get('provenance') != 'track1' or asset.get('video_sha256') != spec.get('video_sha256')):
        raise ValueError('Require a single, source-grounded agent-bpy asset in metres')
    paths = []
    for name in ('geometry', 'authoring_script', *(['authoring_spec'] if 'authoring_spec' in asset else [])):
        row = asset[name]
        path = (Path(spec_path).resolve().parent/row['path']).resolve()
        if any(part.lower().replace('-', '_').replace(' ', '_') in ('track_2', 'track2') for part in path.parts):
            raise ValueError('Track 2 assets are prohibited')
        if sha256(path) != row['sha256']:
            raise ValueError(f'Bpy asset receipt mismatch: {name}')
        paths.append(path)
    if paths[0].suffix.lower() not in ('.npz', '.glb', '.ply', '.obj') or paths[1].suffix.lower() != '.py':
        raise ValueError('Require supported mesh geometry and its bpy Python source')
    return tuple(paths)


def initialize_scale(depth, mask, intrinsics, extents, normal=None, offset=None):
    """Uncertain metric initialization, with no upright/floor assumption for flat objects."""
    depth, mask, extents = np.asarray(depth), np.asarray(mask, dtype=bool), np.asarray(extents)
    y, x = np.where(mask&np.isfinite(depth)&(depth > .1))
    if len(x) < 50 or extents.shape != (3,) or (extents <= 0).any():
        raise ValueError('Insufficient visible depth or nondegenerate template dimensions')
    z = depth[y, x]
    k = np.asarray(intrinsics)
    cloud = np.column_stack(((x-k[0, 2])*z/k[0, 0], (y-k[1, 2])*z/k[1, 1], z))
    # Monocular depth spread is not an observed object thickness. In particular,
    # dark thin surfaces and small segmentation leaks can span metres in depth.
    # Use image angular extent at one robust distance for the fallback instead.
    rays = np.column_stack(((x-k[0, 2])/k[0, 0], (y-k[1, 2])/k[1, 1]))
    centered = rays-np.median(rays, axis=0)
    _, axes = np.linalg.eigh(centered.T@centered)
    planar = centered@axes
    span = np.quantile(planar, .95, axis=0)-np.quantile(planar, .05, axis=0)
    median_depth = float(np.median(z))
    method = 'Projected point-cloud diameter at median depth / template diameter; uncertain under foreshortening or occlusion'
    scale = np.linalg.norm(span)*median_depth/np.linalg.norm(extents)
    if normal is not None:
        heights = offset-cloud@normal
        if extents[2] > extents.max()*.4 and abs(np.quantile(heights, .05)) < .12:
            scale = np.quantile(heights, .95)/extents[2]
            method = 'Visible supported-object height above estimated floor'
    if not .1 < scale < 10.:
        raise ValueError('Unreliable source-depth object scale')
    return float(scale), {'method': method, 'uniform_scale': float(scale), 'metric_reference': False,
        'visible_depth_median_m': median_depth,
        'visible_depth_p05_p95_m': np.quantile(z, [.05, .95]).tolist(),
        'depth_spread_used_as_object_thickness': False}


def automatic_roi(masks, margin=.15):
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or not masks.any():
        raise ValueError('Require visible object masks')
    _, y, x = np.where(masks)
    side = int(np.ceil(max(x.max()-x.min()+1, y.max()-y.min()+1)*(1+2*margin)))
    return [int(np.floor((x.max()+x.min()-side)/2)),
            int(np.floor((y.max()+y.min()-side)/2)), max(side, 64)]


def mask_translation_hypotheses(poses, center, intrinsics, observed_area,
                                observed_centroid, rendered_areas, rendered_centroids):
    """Weak-perspective image hypotheses; depth estimates do not fix translation.

    These are proposals to score with full perspective rendering and tracks,
    not measurements of hidden geometry or a correction applied to final poses.
    """
    poses = np.asarray(poses, dtype=float)
    areas, centroids = np.asarray(rendered_areas), np.asarray(rendered_centroids)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or areas.shape != (len(poses),)
            or centroids.shape != (len(poses), 2) or observed_area <= 0):
        raise ValueError('Require finite image area and pose hypotheses')
    result = poses.copy()
    valid = (areas >= 8)&np.isfinite(centroids).all(1)
    canonical = np.asarray(center)
    camera_centers = poses[:, :3, :3]@canonical+poses[:, :3, 3]
    valid &= camera_centers[:, 2] > .1
    ratio = np.sqrt(areas[valid]/observed_area).clip(.4, 2.5)
    ray_shift = (np.asarray(observed_centroid)-centroids[valid])@np.linalg.inv(np.asarray(intrinsics)[:2, :2]).T
    adjusted = camera_centers[valid]*ratio[:, None]
    adjusted[:, :2] += ray_shift*adjusted[:, 2:]
    result[valid, :3, 3] = adjusted-poses[valid, :3, :3]@canonical
    return result


def select_keyframes(quality, count=12):
    """Best measured frame per temporal bin, plus explicit timeline endpoints."""
    quality = np.asarray(quality, dtype=float)
    if quality.ndim != 1 or not len(quality) or not np.isfinite(quality).all():
        raise ValueError('Require finite all-frame quality')
    selected = {0, len(quality)-1}
    for ids in np.array_split(np.arange(len(quality)), min(count, len(quality))):
        selected.add(int(ids[np.argmax(quality[ids])]))
    return np.array(sorted(selected), dtype=int)


def tracking_windows(count, window=128):
    """Bound memory by overlapping inference windows, never dropping the tail."""
    if count < 1 or window < 8:
        raise ValueError('Require a positive timeline and at least eight window frames')
    if count <= window:
        return [(0, count)]
    starts = list(range(0, count-window+1, window//2))
    if starts[-1] != count-window:
        starts.append(count-window)
    return [(start, start+window) for start in starts]


def spatially_spread_indices(pixels, count):
    """Retain strong corners while covering different visible object regions."""
    pixels = np.asarray(pixels, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or count < 1 or not np.isfinite(pixels).all():
        raise ValueError('Require finite source points and positive count')
    if len(pixels) <= count:
        return np.arange(len(pixels))
    selected = [0]
    distances = np.linalg.norm(pixels-pixels[0], axis=1)
    strength = 1./np.sqrt(1.+np.arange(len(pixels))/max(1, count))
    for _ in range(count-1):
        scores = distances*strength
        scores[selected] = -1.
        index = int(scores.argmax())
        selected.append(index)
        distances = np.minimum(distances, np.linalg.norm(pixels-pixels[index], axis=1))
    return np.asarray(selected)


def select_pose_path(candidates, costs, frame_ids, diameter, fps, observation_edges=None):
    """Viterbi over full SE(3), retaining data cost and elapsed-time identity."""
    poses, costs, ids = np.asarray(candidates), np.asarray(costs), np.asarray(frame_ids)
    if (costs.ndim != 2 or poses.shape != (*costs.shape, 4, 4) or ids.shape != (len(costs),)
            or len(ids) == 0 or (np.diff(ids) <= 0).any() or diameter <= 0 or fps <= 0
            or not np.isfinite(poses).all() or not np.isfinite(costs).all()):
        raise ValueError('Invalid full-pose candidates or original frame IDs')
    if observation_edges is not None:
        observation_edges = np.asarray(observation_edges)
        if observation_edges.shape != (len(ids)-1, costs.shape[1], costs.shape[1]) or not np.isfinite(observation_edges).all():
            raise ValueError('Invalid cross-frame observation costs')
    accumulated, parents = costs[0].copy(), []
    for i in range(1, len(ids)):
        dt = (ids[i]-ids[i-1])/fps
        traces = np.einsum('aij,bij->ab', poses[i-1, :, :3, :3], poses[i, :, :3, :3])
        angles = np.arccos(np.clip((traces-1)/2, -1, 1))
        centers = poses[i-1, :, None, :3, 3]-poses[i, None, :, :3, 3]
        # Soft, time-normalized priors permit real turns instead of freezing yaw.
        transition = .025*(angles/max(np.deg2rad(60)*dt, np.deg2rad(8)))**2
        transition += .025*(np.linalg.norm(centers, axis=-1)/max(diameter*dt, diameter*.1))**2
        if observation_edges is not None:
            transition += observation_edges[i-1]
        alternatives = accumulated[:, None]+transition
        predecessor = alternatives.argmin(0)
        parents.append(predecessor)
        accumulated = costs[i]+alternatives[predecessor, np.arange(costs.shape[1])]
    path = [int(accumulated.argmin())]
    for parent in reversed(parents):
        path.append(int(parent[path[-1]]))
    return np.array(path[::-1])


def rigid_fit(source, target):
    """Kabsch on independently corresponding points; reject degenerate evidence."""
    source, target = np.asarray(source), np.asarray(target)
    if (source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3
            or len(source) < 4 or not np.isfinite(source).all() or not np.isfinite(target).all()
            or np.linalg.matrix_rank(source-source.mean(0), tol=1e-5) < 2):
        raise ValueError('Insufficient non-collinear rigid correspondences')
    u, _, vt = np.linalg.svd((source-source.mean(0)).T@(target-target.mean(0)))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T@u.T)
    r = vt.T@correction@u.T
    pose = np.eye(4)
    pose[:3, :3] = r
    pose[:3, 3] = target.mean(0)-r@source.mean(0)
    return pose


def automatic_contact_schedule(poses, vertices, wrists, normal, offset, fps):
    """Soft geometric hypotheses, not manually supplied grasp/support events."""
    from scipy.ndimage import gaussian_filter1d
    poses, vertices, wrists = np.asarray(poses), np.asarray(vertices), np.asarray(wrists)
    world = vertices[None]@poses[:, :3, :3].transpose(0, 2, 1)+poses[:, None, :3, 3]
    clearance = np.min(offset-world@normal, axis=1) if normal is not None else np.full(len(poses), np.nan)
    speed = np.r_[0., np.linalg.norm(np.diff(world.mean(1), axis=0), axis=1)*fps]
    support = np.exp(-np.square(clearance/.035))*np.exp(-np.square(speed/.25)) if normal is not None else np.zeros(len(poses))
    distances = np.linalg.norm(wrists[:, :, None]-world[:, None], axis=-1)
    nearest = distances.argmin(-1)
    close = np.take_along_axis(distances, nearest[..., None], axis=-1)[..., 0]
    grasp = np.exp(-np.square(close/.12))*(1-.7*support[:, None])
    grasp = gaussian_filter1d(grasp, sigma=2., axis=0)
    local = vertices[nearest]
    return np.clip(support, 0., 1.), np.clip(grasp, 0., 1.), local, clearance


def motion_diagnostics(poses):
    poses = np.asarray(poses)
    shift = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    angle = np.rad2deg(Rotation.from_matrix(poses[1:, :3, :3]@poses[:-1, :3, :3].transpose(0, 2, 1)).magnitude())
    return {'maximum_translation_step_m': float(shift.max(initial=0.)),
            'maximum_rotation_step_deg': float(angle.max(initial=0.)),
            'scope': 'Self-motion only; smoothness is not video fidelity or official acceleration accuracy'}
