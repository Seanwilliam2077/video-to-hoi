"""Source-balanced pose/surface-point bundle fitting with physical constraints.

Track coordinates remain image observations. Their locations on the fixed bpy
surface are uncertain latent variables, not frozen pseudo-3D measurements.
"""
import numpy as np


def frame_balanced_mean(values, weights):
    """Normalize observations per frame, retaining low-support uncertainty."""
    counts = weights.sum(1)
    strength = (counts/6.).clamp(max=1.)
    per_frame = (values*weights).sum(1)/counts.clamp_min(1.)
    return (per_frame*strength).sum()/strength.sum().clamp_min(1.)


def point_mesh_squared_distance(points, triangles):
    """Differentiable point-to-triangle distance, including edges and corners."""
    import torch
    a, b, c = triangles.unbind(1)
    ab, ac = b-a, c-a
    p = points[:, None]-a[None]
    aa, bb, cc = (ab*ab).sum(-1), (ab*ac).sum(-1), (ac*ac).sum(-1)
    ap, cp = (p*ab[None]).sum(-1), (p*ac[None]).sum(-1)
    denom = (aa*cc-bb*bb).clamp_min(1e-16)
    u, v = (ap*cc-cp*bb)/denom, (cp*aa-ap*bb)/denom
    normal = torch.linalg.cross(ab, ac)
    plane = (p*normal[None]).sum(-1).square()/normal.square().sum(-1).clamp_min(1e-16)
    inside = (u >= 0)&(v >= 0)&(u+v <= 1)&((aa*cc-bb*bb)[None] > 1e-16)
    edges = []
    for start, end in ((a, b), (b, c), (c, a)):
        direction = end-start
        relative = points[:, None]-start[None]
        ratio = (relative*direction[None]).sum(-1)/direction.square().sum(-1).clamp_min(1e-16)
        delta = relative-ratio.clamp(0, 1)[..., None]*direction[None]
        edges.append(delta.square().sum(-1))
    boundary = torch.stack(edges).min(0).values
    return torch.where(inside, plane, boundary).min(1).values


def fit_surface_bundle(mesh, initial, ids, selected, masks, humans, depths, camera,
                       keypoints, normal, offset, track_data, steps, fps, output, contact=None):
    import cv2
    import torch
    import torch.nn.functional as F
    from pytorch3d.transforms import so3_exp_map
    from v2hoi.image_motion import surface_points, project_points
    from v2hoi.object_refine_runtime import ObservationRenderer
    from v2hoi.temporal_object import automatic_contact_schedule
    sampled, tracks, visibility, query_frames, query_pixels, _ = track_data
    visibility = visibility.copy()
    vertices = np.asarray(mesh.vertices)
    local = np.zeros((len(query_pixels), 3))
    ray_hit = np.zeros(len(local), dtype=bool)
    attachment_weight = np.ones(len(local))
    for i, pose in zip(ids, selected):
        q = np.flatnonzero(query_frames == i)
        if not len(q):
            continue
        xyz = surface_points(query_pixels[q], camera, pose, mesh.vertices, mesh.faces)
        hits = np.isfinite(xyz).all(1)
        local[q[hits]], ray_hit[q[hits]] = xyz[hits], True
        if (~hits).any():
            projected = project_points(vertices, pose, camera)
            distance = np.linalg.norm(query_pixels[q[~hits], None]-projected[None], axis=-1)
            nearest = distance.argmin(1)
            local[q[~hits]] = vertices[nearest]
            attachment_weight[q[~hits]] = np.maximum(.25, np.exp(-distance.min(1)/15.))
    renderer = ObservationRenderer(mesh)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device='cuda')
    vertex_t = tensor(vertices)
    triangles = tensor(vertices[np.asarray(mesh.faces)])
    local0 = tensor(local)
    local_delta = torch.zeros_like(local0, requires_grad=True)
    r0 = tensor(initial[:, :3, :3])
    rotation_delta = torch.zeros((len(initial), 3), device='cuda', requires_grad=True)
    translation = tensor(initial[:, :3, 3]).clone().requires_grad_(True)
    log_shape = torch.zeros(3, device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([
        {'params': [rotation_delta], 'lr': .006}, {'params': [translation], 'lr': .004},
        {'params': [log_shape], 'lr': .004}, {'params': [local_delta], 'lr': .01}])
    train = visibility.copy()
    train[np.arange(len(sampled)) % 5 == 2] = False
    for q, source in enumerate(query_frames):
        train[np.searchsorted(sampled, source), q] = visibility[np.searchsorted(sampled, source), q]
    heldout = visibility&~train
    train_t = tensor(train)*tensor(attachment_weight)[None]
    target = tensor(tracks)
    camera_t = tensor(camera)
    truth, care, measured = tensor(masks), 1.-tensor(humans), tensor(depths)
    outside = np.asarray([cv2.distanceTransform((~(mask.astype(bool)&~human.astype(bool))).astype('uint8'),
        cv2.DIST_L2, 3) for mask, human in zip(masks, humans)])
    outside_t = tensor(np.minimum(outside, 24.)/8.)
    wrists, joints = keypoints[:, [9, 10]], tensor(keypoints)
    wrist_t = tensor(wrists)
    support, grasp, _, _ = automatic_contact_schedule(initial, vertices, wrists, normal, offset, fps)
    support_t, grasp_t = tensor(support), tensor(grasp)
    contact_weights = tensor(contact['schedule']['weights']) if contact else None
    hand_centers = tensor(contact['hand_centers']) if contact else None
    hand_surfaces = tensor(contact['hands']) if contact else None
    opening_weights = tensor(contact['schedule']['opening_weights']) if contact else None
    semantic_support_weights = tensor(contact['schedule']['support_weights']) if contact else None
    table_n = tensor(contact['support_plane'][0]) if contact and contact['support_plane'] else None
    table_d = contact['support_plane'][1] if table_n is not None else None
    gray_source = tensor(contact['gray_images'])[:, None] if contact else None
    appearance_care = tensor(1.-humans)[:, None] if contact else None
    # Source motion suppresses a spurious static-support hypothesis.
    image_motion = np.zeros(len(sampled))
    for j in range(1, len(sampled)):
        common = visibility[j]&visibility[j-1]
        image_motion[j] = np.median(np.linalg.norm(tracks[j, common]-tracks[j-1, common], axis=-1))/(sampled[j]-sampled[j-1]) if common.sum() >= 4 else np.inf
    support_gate = np.interp(np.arange(len(initial)), sampled, np.exp(-np.square(image_motion/1.2)))
    support_t *= tensor(support_gate)
    n = tensor(normal) if normal is not None else None
    bones = [(5, 6, .07), (5, 11, .07), (6, 12, .07), (11, 12, .07),
             (11, 13, .055), (13, 15, .045), (12, 14, .055), (14, 16, .045)]
    history, components = [], []
    activation = 0.
    def decode():
        raw_shape = log_shape
        if contact and contact['profile'].get('shape_mode') == 'radial_xy':
            radial = log_shape[:2].mean()
            raw_shape = torch.stack([radial, radial, log_shape[2]])
        return so3_exp_map(rotation_delta)@r0, translation, torch.exp(torch.tanh(raw_shape)*(.65 if contact else .45)), local0+.12*torch.tanh(local_delta)*activation
    for step in range(steps):
        activation = 0. if step < 80 else min(1., (step-80)/120.)
        optimizer.zero_grad()
        r, t, shape, local_t = decode()
        batch = np.arange((step*16)%len(initial), min((step*16)%len(initial)+16, len(initial)))
        _, alpha, z = renderer.render(r[batch].contiguous(), t[batch].contiguous(), shape,
            torch.zeros(3, device='cuda'), np.repeat(camera[None], len(batch), axis=0), masks.shape[1:])
        observed, predicted = truth[batch]*care[batch], alpha[..., 0]*care[batch]
        silhouette = (1.-(2*(observed*predicted).sum((1, 2))+1)/((observed+predicted).sum((1, 2))+1)).mean()
        contour = ((predicted*outside_t[batch]).sum((1, 2))/predicted.sum((1, 2)).clamp_min(1.)).mean()
        valid_depth = observed*(alpha[..., 0].detach() > .5)*(measured[batch] > .001)
        depth_loss = (F.smooth_l1_loss((z[..., 0]-measured[batch])/.08, torch.zeros_like(z[..., 0]), reduction='none')*valid_depth).sum()/valid_depth.sum().clamp_min(1.)
        xyz = (local_t*shape)@r[sampled].transpose(1, 2)+t[sampled, None]
        projection = xyz@camera_t.T
        errors = (projection[..., :2]/projection[..., 2:].clamp_min(.1)-target)/3.
        robust = F.smooth_l1_loss(errors, torch.zeros_like(errors), reduction='none').sum(-1)
        point_loss = frame_balanced_mean(robust, train_t)
        surface = point_mesh_squared_distance(local_t, triangles).mean()/.007**2
        attachment_prior = ((local_t-local0)/.1).square().mean()
        world = (vertex_t*shape)@r.transpose(1, 2)+t[:, None]
        d2 = world[2:]-2*world[1:-1]+world[:-2]
        temporal = F.smooth_l1_loss(d2/.005, torch.zeros_like(d2))
        physical = silhouette*0.
        if n is not None:
            clearance = offset-world@n
            physical = torch.relu(-clearance).square().mean()*200.
            physical += (support_t*clearance.min(1).values.square()).mean()*20.
            velocity = (world[1:].mean(1)-world[:-1].mean(1))*fps
            tangent = velocity-(velocity@n)[:, None]*n
            physical += ((support_t[1:]*support_t[:-1])[:, None]*tangent.square()).mean()*.03
        hand_distance = torch.linalg.vector_norm(wrist_t[:, :, None]-world[:, None], dim=-1).min(-1).values
        hand_loss = (grasp_t*torch.relu(hand_distance-.04).square()).mean()*10.
        semantic = silhouette*0.
        contact_gaps = []
        if contact:
            for side, name in enumerate(contact['schedule']['part_names']):
                if name is None:
                    contact_gaps.append(torch.zeros(len(initial), device='cuda'))
                    continue
                part = world[:, contact['parts'][name]]
                distances = torch.linalg.vector_norm(part-hand_centers[:, side, None], dim=-1)
                gap = distances.min(1).values
                contact_gaps.append(gap)
                weights = contact_weights[:, side]
                fit_contact = F.smooth_l1_loss(torch.relu(gap-.025)/.05, torch.zeros_like(gap), reduction='none')
                semantic += .8*(weights*fit_contact).sum()/weights.sum().clamp_min(1.)
                # Actual decoded hand vertices complement the hand-centroid proxy.
                surface_gap = torch.cdist(part, hand_surfaces[:, side]).flatten(1).min(1).values
                robust_surface = F.smooth_l1_loss(torch.relu(surface_gap-.015)/.04, torch.zeros_like(surface_gap), reduction='none')
                semantic += .15*(weights*robust_surface).sum()/weights.sum().clamp_min(1.)
                assignment = torch.softmax(-distances.square()/.02**2, dim=1)
                relative = (part*assignment[..., None]).sum(1)-hand_centers[:, side]
                held = weights[1:]*weights[:-1]
                semantic += .08*(held[:, None]*F.smooth_l1_loss((relative[1:]-relative[:-1])/.01,
                    torch.zeros_like(relative[1:]), reduction='none')).mean()
            if n is not None:
                opening = r@tensor(contact['profile']['opening_axis'])
                semantic += 3.*(opening_weights*(1.+opening@n).square()).mean()
            if table_n is not None:
                table_clearance = table_d-world@table_n
                semantic += .15*torch.relu(-table_clearance).square().mean()/.04**2
                semantic += .2*(semantic_support_weights*table_clearance.min(1).values.square()).mean()/.05**2
                if 'planar_support_normals' in contact['profile']:
                    normals = r@tensor(contact['profile']['planar_support_normals']).T
                    dots = torch.einsum('nij,i->nj', normals, table_n)
                    alignment = (1.-dots.square().max(1).values).clamp_min(0.)
                    semantic += 4.*(semantic_support_weights*alignment).mean()
            for line, upper in contact.get('appearance_parts', []):
                points = (tensor(line)*shape)@r.transpose(1, 2)+t[:, None]
                projected = points@camera_t.T
                uv = projected[..., :2]/projected[..., 2:].clamp_min(.1)
                grid = uv.clone()
                grid[..., 0] = 2*grid[..., 0]/(masks.shape[2]-1)-1
                grid[..., 1] = 2*grid[..., 1]/(masks.shape[1]-1)-1
                samples = F.grid_sample(gray_source, grid[:, None], align_corners=True)[:, 0, 0]
                valid = F.grid_sample(appearance_care, grid[:, None], align_corners=True)[:, 0, 0].detach()
                valid *= (grid.abs() <= 1).all(-1)
                semantic += 2.*(torch.relu(samples-upper)*valid).sum()/valid.sum().clamp_min(1.)
            # Rotation regularization is independent of the thin object's radius.
            curvature = (r[2:]-2*r[1:-1]+r[:-2]).square().sum((1, 2))
            semantic += 4.*torch.topk(curvature, max(1, len(curvature)//20)).values.mean()
        collision = silhouette*0.
        for a, b, radius in bones:
            segment = joints[:, b]-joints[:, a]
            u = ((world-joints[:, None, a])*segment[:, None]).sum(-1)/segment.square().sum(-1)[:, None].clamp_min(1e-6)
            closest = joints[:, None, a]+u.clamp(0, 1)[..., None]*segment[:, None]
            collision += torch.relu(radius-torch.linalg.vector_norm(world-closest, dim=-1)).square().mean()*5.
        loss = silhouette+.08*contour+.04*depth_loss+.22*point_loss+.08*temporal+physical+(.0 if contact else .02)*hand_loss+semantic+collision+.03*surface+.005*attachment_prior+.015*log_shape.square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([rotation_delta, translation, log_shape, local_delta], 10.)
        optimizer.step()
        history.append(float(loss.detach()))
        if step % 100 == 0 or step == steps-1:
            row = {name: float(value.detach()) for name, value in [('silhouette', silhouette), ('points', point_loss),
                ('depth', depth_loss), ('surface', surface), ('temporal', temporal), ('physical', physical), ('semantic_contact', semantic)]}
            components.append({'step': step, **row})
            print(f'SURFACE_BUNDLE step={step}/{steps} loss={history[-1]:.4f} terms={row} shape={shape.detach().cpu().numpy().round(3)}', flush=True)
    r, t, shape, local_t = decode()
    poses = np.repeat(np.eye(4)[None], len(initial), axis=0)
    poses[:, :3, :3], poses[:, :3, 3] = r.detach().cpu().numpy(), t.detach().cpu().numpy()
    shape, local_final = shape.detach().cpu().numpy(), local_t.detach().cpu().numpy()
    contact_report = {'enabled': contact is not None}
    if contact:
        final_world = (vertices*shape)@poses[:, :3, :3].transpose(0, 2, 1)+poses[:, None, :3, 3]
        final_gaps, initial_gaps = [], []
        initial_world = vertices@initial[:, :3, :3].transpose(0, 2, 1)+initial[:, None, :3, 3]
        for side, name in enumerate(contact['schedule']['part_names']):
            if name is None:
                final_gaps.append(np.full(len(initial), np.nan)); initial_gaps.append(np.full(len(initial), np.nan))
                continue
            indices = contact['parts'][name]
            final_gaps.append(np.linalg.norm(final_world[:, indices]-contact['hand_centers'][:, side, None], axis=-1).min(1))
            initial_gaps.append(np.linalg.norm(initial_world[:, indices]-contact['hand_centers'][:, side, None], axis=-1).min(1))
        final_gaps, initial_gaps = np.stack(final_gaps, axis=1), np.stack(initial_gaps, axis=1)
        active = contact['schedule']['weights'] > .35
        contact_report.update(authored_source_frames=len(contact['diagnosis']['rows']),
            part_names=contact['schedule']['part_names'],
            initial_active_part_hand_centroid_gap_m=float(np.median(initial_gaps[active])) if active.any() else None,
            final_active_part_hand_centroid_gap_m=float(np.median(final_gaps[active])) if active.any() else None,
            scope='Part-to-native-hand centroid proxy and hand-surface proximity, not verified finger wrapping')
        np.savez(output/'contact-fit.npz', initial_part_hand_center_gap_m=initial_gaps,
            final_part_hand_center_gap_m=final_gaps, contact_weights=contact['schedule']['weights'],
            opening_axis_world=poses[:, :3, :3]@np.asarray(contact['profile']['opening_axis']))
    final_xyz = (local_final*shape)@poses[sampled, :3, :3].transpose(0, 2, 1)+poses[sampled, None, :3, 3]
    uv = final_xyz@camera.T
    errors_px = np.linalg.norm(uv[..., :2]/uv[..., 2:]-tracks, axis=-1)
    np.savez(output/'joint-fit.npz', initial=initial, poses=poses, global_shape_factors=shape,
        support_probability=support, grasp_probability=grasp, attached_track_points=local_final,
        initial_track_points=local, track_attachments_valid=np.ones(len(local), dtype=bool),
        initial_ray_hits=ray_hit, objective=history, track_training_mask=train, track_heldout_mask=heldout,
        final_reprojection_errors_px=errors_px, source_support_gate=support_gate)
    import json
    (output/'objective-components.json').write_text(json.dumps(components, indent=2))
    return poses, shape, {'steps': steps, 'source_track_attachments': len(local), 'initial_ray_hits': int(ray_hit.sum()),
        'latent_surface_attachment_refinement': True, 'frame_balanced_track_loss': True,
        'visible_training_observations': int(train.sum()), 'heldout_track_observations': int(heldout.sum()),
        'heldout_median_reprojection_px': float(np.median(errors_px[heldout])) if heldout.any() else None,
        'visible_median_reprojection_px': float(np.median(errors_px[visibility])),
        'holdout_scope': 'Track coordinates held out; image masks/depth at these frames remain in fitting',
        'manual_contact_or_support_intervals': False, 'floor_enabled': normal is not None,
        'sparse_contact_optimization': contact_report,
        'physical_constraints': ['estimated floor nonpenetration', 'source-motion-gated support/sliding',
            'soft hand proximity', 'approximate body capsule nonpenetration'],
        'physics_engine': 'Analytic differentiable constraints inside bundle optimization; no pose overwrite',
        'full_differentiable_dynamics': False, 'initial_objective': history[0], 'final_objective': history[-1]}
