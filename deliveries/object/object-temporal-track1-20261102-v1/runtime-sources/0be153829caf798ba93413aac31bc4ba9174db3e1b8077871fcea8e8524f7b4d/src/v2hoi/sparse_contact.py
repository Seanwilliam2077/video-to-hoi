"""Explicit, source-receipted sparse semantic contact assistance.

Labels are uncertain agent observations, not reference poses or contact truth.
Unobserved transition times remain soft; no initial 3D distance gates contact.
"""
import json
from pathlib import Path

import numpy as np

from v2hoi.native import sha256


def load_contact_profile(path, clip, geometry):
    path = Path(path).resolve()
    profile = json.loads(path.read_text())
    diagnosis_path = (path.parent/profile['diagnosis']).resolve()
    for item in (path, diagnosis_path):
        if any(p.lower().replace('-', '_') in ('track_2', 'track2') for p in item.parts):
            raise ValueError('Track 2 contact inputs are prohibited')
    diagnosis = json.loads(diagnosis_path.read_text())
    for item in (profile, diagnosis):
        if (item.get('provenance') != 'track1' or item.get('episode') != clip.episode
                or item.get('video_sha256') != sha256(clip.video)):
            raise ValueError('Sparse contact video identity mismatch')
    if diagnosis.get('reference_used') is not False or not diagnosis.get('authored_source_observations'):
        raise ValueError('Require explicit source-observation provenance')
    if profile['geometry_sha256'] != sha256(geometry):
        raise ValueError('Sparse contact part geometry changed')
    axis = np.asarray(profile['opening_axis'], dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1.):
        raise ValueError('Require a unit canonical opening axis')
    if profile.get('shape_mode') not in ('radial_xy', 'independent'):
        raise ValueError('Unsupported structural shape parameterization')
    if 'planar_support_normals' in profile:
        normals = np.asarray(profile['planar_support_normals'], dtype=float)
        if (normals.ndim != 2 or normals.shape[1] != 3 or not 1 <= len(normals) <= 32
                or not np.isfinite(normals).all() or not np.allclose(np.linalg.norm(normals, axis=1), 1.)):
            raise ValueError('Require unit canonical planar support normals')
    rows = diagnosis['rows']
    ids = [row['original_frame_id'] for row in rows]
    if not 1 <= len(ids) <= 6 or ids != sorted(set(ids)) or min(ids) < 0 or max(ids) >= clip.n_frames:
        raise ValueError('Require at most six ordered, unique original contact frames')
    watched = [path, diagnosis_path]
    for row in rows:
        image = (diagnosis_path.parent/row['source_image']['path']).resolve()
        if not image.is_relative_to(diagnosis_path.parent) or sha256(image) != row['source_image']['sha256']:
            raise ValueError('Contact source image receipt mismatch')
        watched.append(image)
        for hand in ('left', 'right'):
            value = row['hands'][hand]
            if value['state'] not in ('no_contact', 'grasp', 'support') or not 0 <= value['confidence'] <= 1:
                raise ValueError('Invalid sparse hand contact observation')
            if value['state'] != 'no_contact' and value['object_part'] not in profile['parts']:
                raise ValueError('Contact references an unknown bpy part')
    return profile, diagnosis, watched


def part_vertex_indices(raw_vertices, profile):
    vertices = np.asarray(raw_vertices)
    parts = {}
    for name, row in profile['parts'].items():
        bounds = np.asarray(row['bounds'], dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or (bounds[1] <= bounds[0]).any():
            raise ValueError('Require finite nondegenerate part bounds')
        lower, upper = bounds
        indices = np.flatnonzero(((vertices >= lower-1e-6)&(vertices <= upper+1e-6)).all(1))
        if len(indices) < 3:
            raise ValueError(f'Bpy contact part has insufficient vertices: {name}')
        parts[name] = indices
    return parts


def semantic_contact_schedule(diagnosis, hands, camera, masks, humans):
    """Interpolate sparse confidence, modulated by projected hand/image evidence."""
    import cv2
    from scipy.ndimage import gaussian_filter1d
    count = len(masks)
    ids = np.array([row['original_frame_id'] for row in diagnosis['rows']])
    prior = np.zeros((count, 2))
    part_names = []
    for side, hand in enumerate(('left', 'right')):
        values, parts = [], set()
        for row in diagnosis['rows']:
            item = row['hands'][hand]
            values.append(item['confidence'] if item['state'] != 'no_contact' else 0.)
            if item['state'] != 'no_contact':
                parts.add(item['object_part'])
        if len(parts) > 1:
            raise ValueError('This adapter requires one contact part per hand; split phases explicitly for other cases')
        part_names.append(next(iter(parts)) if parts else None)
        prior[:, side] = np.interp(np.arange(count), ids, values)
        positives = np.flatnonzero(np.asarray(values) > 0.)
        if len(positives):
            first, last = ids[positives[[0, -1]]]
            # Outside observed positive anchors, contact timing is unknown.
            # Use localized confidence rather than a long linear contact ramp.
            timeline = np.arange(count)
            prior[timeline < first, side] = values[positives[0]]*np.exp(-.5*((timeline[timeline < first]-first)/12.)**2)
            prior[timeline > last, side] = values[positives[-1]]*np.exp(-.5*((timeline[timeline > last]-last)/12.)**2)
    projected = hands@np.asarray(camera).T
    projected = projected[..., :2]/np.maximum(projected[..., 2:], .1)
    evidence = np.zeros((count, 2))
    for i in range(count):
        visible = masks[i].astype(bool)&~humans[i].astype(bool)
        field = cv2.distanceTransform((~visible).astype('uint8'), cv2.DIST_L2, 3)
        for hand in range(2):
            xy = np.rint(projected[i, hand]).astype(int)
            inside = (xy[:, 0] >= 0)&(xy[:, 0] < field.shape[1])&(xy[:, 1] >= 0)&(xy[:, 1] < field.shape[0])
            distance = field[xy[inside, 1], xy[inside, 0]].min() if inside.any() else np.inf
            evidence[i, hand] = np.exp(-(distance/35.)**2)
    # A sparse positive label does not imply contact throughout the intervening
    # interval. Image evidence may turn it off when the hand is visibly away.
    weights = gaussian_filter1d(prior*evidence, sigma=2., axis=0)
    for index, row in zip(ids, diagnosis['rows']):
        for hand, name in enumerate(('left', 'right')):
            item = row['hands'][name]
            if item['state'] != 'no_contact':
                local = item['confidence']*np.exp(-.5*((np.arange(count)-index)/3.)**2)
                weights[:, hand] = np.maximum(weights[:, hand], local)
            else:
                weights[index, hand] = 0.
    # Positive kernels must never overwrite a later/earlier explicit negative.
    for index, row in zip(ids, diagnosis['rows']):
        for hand, name in enumerate(('left', 'right')):
            if row['hands'][name]['state'] == 'no_contact':
                weights[index, hand] = 0.
    # A resting object need not have its semantic face pointing up. For example,
    # the grass face of a block may be vertical on the table. Keep positional
    # support separate from explicitly observed semantic-axis orientation.
    support = np.zeros(count)
    opening = np.zeros(count)
    for row in diagnosis['rows']:
        state = row['support']['state']
        if state not in ('table_supported', 'probable_table_support',
                         'ground_supported', 'probable_ground_support'):
            continue
        # Support labels localize an observation, not an exact long interval.
        frame = row['original_frame_id']
        kernel = np.exp(-.5*((np.arange(count)-frame)/15.)**2)
        support = np.maximum(support, row['support']['confidence']*kernel)
        orientation = row.get('orientation', {})
        direction = orientation.get('axis_direction', orientation.get('opening'))
        if direction in ('away_from_support', 'away_from_table', 'away_from_ground'):
            opening = np.maximum(opening, min(row['support']['confidence'],
                orientation['confidence'])*kernel)
    return {'weights': np.clip(weights, 0, 1), 'prior': prior, 'image_evidence': evidence,
            'part_names': part_names, 'opening_weights': opening, 'support_weights': support}


def fit_local_support_plane(depth, mask, human, camera, downward):
    """Estimate local support from a depth annulus; never substitute floor height."""
    import cv2
    valid = np.isfinite(depth)&(depth > .1)
    visible = mask.astype(bool)&~human.astype(bool)&valid
    z0 = np.median(depth[visible])
    ring = (cv2.dilate(mask.astype('uint8'), np.ones((61, 61), 'uint8')) > 0)&~mask.astype(bool)
    ring &= ~human.astype(bool)&valid&(abs(depth-z0) < .4)
    y, x = np.where(ring)
    if len(x) < 100:
        return None, {'enabled': False, 'reason': 'Insufficient nearby support depth'}
    # Deterministic bounded sampling of actual permitted image/depth points.
    take = np.linspace(0, len(x)-1, min(len(x), 3000)).astype(int)
    x, y = x[take], y[take]; z = depth[y, x]
    points = np.column_stack(((x-camera[0, 2])*z/camera[0, 0], (y-camera[1, 2])*z/camera[1, 1], z))
    rng = np.random.default_rng(7); best = np.zeros(len(points), dtype=bool)
    for _ in range(300):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a); normal /= max(np.linalg.norm(normal), 1e-12)
        normal *= 1 if normal@downward >= 0 else -1
        if normal@downward < .9:
            continue
        keep = abs(points@normal-a@normal) < .015
        if keep.sum() > best.sum():
            best = keep
    if best.mean() < .4:
        return None, {'enabled': False, 'reason': 'No reliable nearby support plane'}
    cloud = points[best]; _, _, axes = np.linalg.svd(cloud-cloud.mean(0), full_matrices=False)
    normal = axes[-1]; normal *= 1 if normal@downward >= 0 else -1
    offset = float(normal@cloud.mean(0))
    return (normal, offset), {'enabled': True, 'normal': normal.tolist(), 'offset_m': offset,
        'inlier_fraction': float(best.mean()), 'source': 'First source depth annulus; uncertain monocular surface'}


def candidate_contact_cost(poses, vertices, contact, frame):
    """Semantic opening and part-specific hand proximity score pose proposals."""
    rotations, translations = poses[:, :3, :3], poses[:, :3, 3]
    cost = np.zeros(len(poses))
    if contact['downward'] is not None:
        direction = rotations@np.asarray(contact['profile']['opening_axis'])
        dot = direction@contact['downward']
        cost += 3.*contact['schedule']['opening_weights'][frame]*np.maximum(dot, 0.)**2
    if contact.get('support_plane') is not None and 'planar_support_normals' in contact['profile']:
        # Any authored planar support face is allowed; none is declared world up.
        normals = rotations@np.asarray(contact['profile']['planar_support_normals']).T
        dots = np.einsum('nij,i->nj', normals, contact['support_plane'][0])
        alignment = np.maximum(1.-np.square(dots).max(1), 0.)
        cost += 2.*contact['schedule']['support_weights'][frame]*alignment
    for side, part in enumerate(contact['schedule']['part_names']):
        if part is None:
            continue
        points = vertices[contact['parts'][part]]@rotations.transpose(0, 2, 1)+translations[:, None]
        distance = np.linalg.norm(points-contact['hand_centers'][frame, side], axis=-1).min(1)
        cost += .5*contact['schedule']['weights'][frame, side]*np.minimum((np.maximum(distance-.025, 0.)/.15)**2, 4.)
    # Part-specific source appearance retains handle evidence absent from the
    # upstream whole-object mask. Human-occluded samples carry no appearance cost.
    if not contact.get('appearance_parts'):
        return cost
    import cv2
    for points, upper in contact.get('appearance_parts', []):
        world = points@rotations.transpose(0, 2, 1)+translations[:, None]
        projected = world@contact['camera'].T
        xy = (projected[..., :2]/np.maximum(projected[..., 2:], .1)).astype('float32')
        gray = cv2.remap(contact['gray_images'][frame], xy[..., 0], xy[..., 1], cv2.INTER_LINEAR)
        human = cv2.remap(contact['human_masks'][frame].astype('float32'), xy[..., 0], xy[..., 1], cv2.INTER_NEAREST)
        valid = (xy[..., 0] >= 0)&(xy[..., 0] < contact['gray_images'].shape[2])&(xy[..., 1] >= 0)&(xy[..., 1] < contact['gray_images'].shape[1])&(human < .5)
        cost += 2.*(np.maximum(gray-upper, 0.)*valid).sum(1)/np.maximum(valid.sum(1), 1)
    return cost
