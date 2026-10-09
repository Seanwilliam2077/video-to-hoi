"""Matched 2x2 contact-schedule/3D-interaction-mesh development experiment.

Track 1 episode 16, original frames 90..209 only. Contact modes are frozen
pseudo-labels inferred from the common v3 no-graph warm start, not contact GT.
Unknown deeply penetrating geometry is excluded from contact inference but
still receives the common collision penalty. No extra model training.
"""
import json
import time
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import roma
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter

from real_trial import ROOT, tensor, project, project_np, local, sdf, decode, measure, observations
from synthetic_trial import box_sdf
from interaction_trial import interaction_graph, diagnostics

OUT = ROOT / 'results/interaction-v4'
NAMES = ['separated', 'sliding', 'sticking', 'uncertain_geometry']
SETTINGS = dict(near_m=.02, distance_scale_m=.025, stick_speed_m_per_frame=.003,
                deep_penetration_m=.015, transition_cost=.8, iterations=300,
                graph_weight=.5, stick_weight=.05, contact_weight=.2)


def surface(points, half):
    result = np.clip(points, -half, half)
    inside = np.all(np.abs(points) <= half, axis=-1)
    axis = np.argmin(half - np.abs(points), axis=-1)
    for a in range(3):
        use = inside & (axis == a)
        result[..., a] = np.where(use, np.where(points[..., a] >= 0, half[a], -half[a]), result[..., a])
    return result


def infer_modes(points, half):
    """Viterbi with a switching cost; labels never depend on solver iterates."""
    distance = box_sdf(points, half)
    anchors = surface(points, half)
    speed = np.linalg.norm(np.diff(anchors, axis=0, prepend=anchors[:1]), axis=-1)
    speed = median_filter(speed, size=(3, 1), mode='nearest')
    near = np.exp(-np.abs(distance) / SETTINGS['distance_scale_m'])
    slow = np.exp(-speed / SETTINGS['stick_speed_m_per_frame'])
    costs = np.stack([-np.log(np.clip(1-near, 1e-4, 1)),
                      -np.log(np.clip(near*(1-slow), 1e-4, 1)),
                      -np.log(np.clip(near*slow, 1e-4, 1))], axis=-1)
    transition = (1-np.eye(3))*SETTINGS['transition_cost']
    states = np.zeros(distance.shape, np.int64)
    for p in range(points.shape[1]):
        score = costs[0, p].copy()
        back = np.zeros((len(points), 3), int)
        for t in range(1, len(points)):
            candidate = score[:, None] + transition
            back[t] = candidate.argmin(0)
            score = costs[t, p] + candidate.min(0)
        states[-1, p] = score.argmin()
        for t in range(len(points)-1, 0, -1):
            states[t-1, p] = back[t, states[t, p]]
    states[distance < -SETTINGS['deep_penetration_m']] = 3
    active = ((states == 1) | (states == 2)).astype(np.float32)
    sticking = (states == 2).astype(np.float32)
    return states, active, sticking, distance, speed


def synthetic_checks():
    half = np.array([.1, .1, .1])
    points = np.zeros((120, 1, 3))
    points[:, 0, 0] = .1
    points[:20, 0, 0] = .20
    points[100:, 0, 0] = .20
    points[50:80, 0, 1] = np.arange(30)*.004 - .058
    points[80:100, 0, 1] = points[79, 0, 1]
    truth = np.r_[np.zeros(20), np.full(30, 2), np.ones(30), np.full(20, 2), np.zeros(20)].astype(int)
    states, active, sticking, _, _ = infer_modes(points, half)
    interior = np.ones(120, bool)
    for t in [20, 50, 80, 100]:
        interior[max(0,t-3):t+3] = False
    accuracy = float((states[interior, 0] == truth[interior]).mean())
    assert accuracy > .95, accuracy
    assert active[:15].sum() == 0 and active[105:].sum() == 0
    assert sticking[55:75].sum() == 0
    slide = np.array([[[.1, 0, 0]], [[.1, .04, 0]]])
    assert np.allclose(box_sdf(slide, half), 0)
    static_gate = active.mean(0, keepdims=True)
    fixed_release_cost = float(((box_sdf(points[100:],half)/.01)**2*static_gate).mean())
    dynamic_release_cost = float(((box_sdf(points[100:],half)/.01)**2*active[100:]).mean())
    assert dynamic_release_cost == 0 and fixed_release_cost > 0
    return dict(interior_mode_accuracy=accuracy, fixed_gate_release_penalty=fixed_release_cost,
                dynamic_gate_release_penalty=dynamic_release_cost,
                scope='Known-truth mode inference and constraint behavior, not a reconstruction accuracy experiment')


def gated_laplacian(lap, gate, hand_relative, contact_relative, nh):
    adjacency = -lap.copy()
    for i in range(adjacency.shape[1]):
        adjacency[:, i, i] = 0
    nearest = np.linalg.norm(hand_relative[:, :, None] - contact_relative[:, None], axis=-1).argmin(-1)
    node_gate = np.take_along_axis(gate, nearest, axis=1)
    adjacency[:, :nh, nh:] *= node_gate[:, :, None]
    adjacency[:, nh:, :nh] *= node_gate[:, None, :]
    degree = adjacency.sum(-1)
    normalized = adjacency / np.maximum(degree[..., None], 1e-8)
    result = -normalized
    for i in range(result.shape[1]):
        result[:, i, i] = (degree[:, i] > 1e-8).astype(float)
    assert np.max(np.abs(result.sum(-1))) < 2e-6
    return result


def optimize(name, initial, fitting, graph, gate, stick, hand_ids, contacts, collision_ids, model, fit_gate):
    canon, obs, confidence, half, k = map(tensor, fitting)
    lap, reference, objects = map(tensor, graph)
    gate, stick = tensor(gate), tensor(stick)
    base = tensor(initial['vertices'])
    native = tensor(initial['mhr_model_params'])
    shape, expr, cam = [tensor(initial[x]) for x in ['shape_params','expr_params','cam_t']]
    meta = json.loads((ROOT/'results/real/metrics.json').read_text())
    controls = np.array(meta['optimized_native_control_ids'])
    basis = torch.zeros((len(controls), native.shape[1]), device='cuda')
    basis[torch.arange(len(controls)), tensor(controls).long()] = tensor(meta['bounded_native_control_scales'])
    spline = tensor(CubicSpline(np.linspace(0,119,16),np.eye(16),bc_type='natural')(np.arange(120)))
    dc = torch.nn.Parameter(torch.zeros((16,3),device='cuda'))
    dr = torch.nn.Parameter(torch.zeros((16,3),device='cuda'))
    dq = torch.nn.Parameter(torch.zeros((16,len(controls)),device='cuda'))
    center0, rotation0 = tensor(initial['object_translation']), tensor(initial['object_rotation'])
    optim = torch.optim.Adam([dc,dr,dq],lr=.003)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim,300,eta_min=.00003)
    target = lap @ reference
    prior_image = project(base[:,hand_ids], k)
    sign = tensor([1,-1,-1])
    graph_weight = 0. if name.endswith('_no_mesh') else SETTINGS['graph_weight']
    best, best_score, history = None, float('inf'), []
    for iteration in range(301):
        optim.zero_grad()
        center = center0+spline@dc
        rotation = roma.rotvec_to_rotmat(spline@dr)@rotation0
        params = native+(spline@dq)@basis
        verts,_ = model(shape,params,expr)
        verts = verts/100*sign+cam[:,None]
        error = torch.linalg.vector_norm(project(canon[None]@rotation.transpose(1,2)+center[:,None],k)-obs,dim=-1)
        image = ((error/3).square()*confidence).sum()/confidence.sum()
        contact_local = local(verts[:,contacts],rotation,center)
        contact_loss = ((sdf(contact_local,half)/.01).square()*gate).mean()
        pair_stick = torch.minimum(stick[1:],stick[:-1])
        relative_delta = contact_local[1:]-contact_local[:-1]
        sticking_loss = ((relative_delta/.005).square().sum(-1)*pair_stick).mean()
        collision_local = local(verts[:,collision_ids],rotation,center)
        inside = torch.relu(.0005-sdf(collision_local,half))/.01
        collision = inside.square().mean()+.1*inside.max(1).values.square().mean()
        joint = torch.cat([local(verts[:,hand_ids],rotation,center),objects.expand(120,-1,-1)],dim=1)
        graph_loss = ((lap@joint-target)/.025).square().mean()
        human = ((verts[:,collision_ids]-base[:,collision_ids])/.02).square().mean()
        human_image = ((project(verts[:,hand_ids],k)-prior_image)/3).square().mean()
        temporal = ((center[2:]-2*center[1:-1]+center[:-2])/.005).square().mean()
        hand = verts[:,hand_ids]
        temporal += ((hand[2:]-2*hand[1:-1]+hand[:-2])/.005).square().mean()
        q = spline@dq
        loss = image+.5*human+.25*human_image+SETTINGS['contact_weight']*contact_loss+collision+graph_weight*graph_loss+SETTINGS['stick_weight']*sticking_loss+.04*temporal+.05*q.square().mean()+.1*(spline@dr).square().mean()
        with torch.no_grad():
            fit = float((error*confidence).sum()/confidence.sum())
            if fit <= fit_gate and float(loss) < best_score:
                best_score = float(loss)
                best = dict(native=params.detach().cpu().numpy(),center=center.detach().cpu().numpy(),rotation=rotation.detach().cpu().numpy(),iteration=iteration,fit_reprojection_px=fit)
        if iteration%50 == 0:
            row = dict(iteration=iteration,loss=float(loss.detach()),fit_px=fit,contact=float(contact_loss.detach()),stick=float(sticking_loss.detach()),collision=float(collision.detach()),mesh=float(graph_loss.detach()))
            history.append(row)
            print(name,json.dumps(row),flush=True)
        if iteration == 300:
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([dc,dr,dq],50)
        optim.step(); schedule.step()
        with torch.no_grad():
            dc.clamp_(-.06,.06); dr.clamp_(-.2,.2); dq.clamp_(-.8,.8)
    assert best is not None, name
    best['vertices'] = decode(model,initial['shape_params'],best['native'],initial['expr_params'],initial['cam_t'])
    (OUT/f'{name}-optimization.json').write_text(json.dumps(dict(selected_iteration=best['iteration'],fit_gate_px=fit_gate,history=history),indent=2))
    return best


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol = dict(scope='Track 1 episode 16 original frames 90-209',settings=SETTINGS,
                    conditions=['static_no_mesh','static_mesh','dynamic_no_mesh','dynamic_mesh'],
                    initialization='Common v3 sliding_without_graph output; extra bounded refinement budget, not a fresh blind test',
                    state_source='Frozen geometric pseudo-labels from common warm start; not solver-controlled, not contact GT',
                    static_schedule='Temporal mean of dynamic contact/stick weights: same total exposure, no switching',
                    observations='Same v3 texture split; previous development validation reused',
                    selection='Training objective only, fit reprojection <=1.10*warm-start training error; validation not used for selection',
                    hypotheses=['Temporal contact modes reduce false binding during separation/sliding','3D interaction Laplacian adds benefit beyond contact modes'],
                    limitations=['Estimated object geometry and contact labels','Short clip may lack true release','No 3D/contact GT','Sampled collision, no continuous guarantee'])
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    checks = synthetic_checks()
    (OUT/'synthetic-checks.json').write_text(json.dumps(checks,indent=2))
    initial = dict(np.load(ROOT/'results/interaction-v3/sliding_without_graph.npz'))
    meta = json.loads((ROOT/'results/real/metrics.json').read_text())
    contacts = np.array(meta['fixed_contact_vertex_ids'])
    geometry = dict(np.load(ROOT/'results/interaction-v3/interaction-structure.npz'))
    hand_ids = geometry['hand_vertex_ids']
    objects = geometry['object_surface_samples']
    half = initial['object_half_extents']
    k = np.load(ROOT/'baseline/intrinsics.npy')
    tracks = dict(np.load(ROOT/'results/interaction-v3/validation-tracks.npz'))
    split = dict(np.load(ROOT/'results/interaction-v3/texture-split.npz'))
    images = [cv2.imread(str(ROOT/'data/frames'/f'{i:06d}.jpg')) for i in range(90,210)]
    baseline = [dict(np.load(ROOT/'baseline'/f'{i:06d}.npz')) for i in range(90,210)]
    depth = np.median(np.stack([d['pred_keypoints_3d'][[41,62],2]+d['pred_cam_t'][2] for d in baseline]))
    corner_canon, corner_obs, corner_confidence, fitted_half, _, _ = observations(images,k,depth,OUT)
    assert np.allclose(fitted_half,half)
    original_inputs = (corner_canon,corner_obs,corner_confidence)
    train, val = split['training_indices'], split['validation_indices']
    canon = np.concatenate([original_inputs[0],tracks['anchors'][train]])
    obs = np.concatenate([original_inputs[1],tracks['image_points'][:,train]],axis=1)
    confidence = np.concatenate([original_inputs[2]*.1,tracks['valid'][:,train].astype(float)],axis=1)
    validation = (tracks['anchors'][val],tracks['image_points'][:,val],tracks['valid'][:,val])
    warm_error = np.linalg.norm(project_np(canon[None]@initial['object_rotation'].transpose(0,2,1)+initial['object_translation'][:,None],k)-obs,axis=-1)
    fit_gate = float((warm_error*confidence).sum()/confidence.sum())*1.10
    contact_relative = np.einsum('tpi,tij->tpj',initial['vertices'][:,contacts]-initial['object_translation'][:,None],initial['object_rotation'])
    states, active, stick, distance, speed = infer_modes(contact_relative,half)
    assert active.sum() > 0, 'No inferred contact: cannot test interaction hypothesis'
    static_active = np.broadcast_to(active.mean(0,keepdims=True),active.shape).copy()
    static_stick = np.broadcast_to(stick.mean(0,keepdims=True),stick.shape).copy()
    assert np.allclose(static_active.sum(0),active.sum(0),atol=1e-4)
    hand_relative = np.einsum('tpi,tij->tpj',initial['vertices'][:,hand_ids]-initial['object_translation'][:,None],initial['object_rotation'])
    base_lap, reference, _ = interaction_graph(hand_relative,objects)
    near = np.flatnonzero(np.linalg.norm(initial['vertices'][60]-initial['object_translation'][60],axis=-1)<.35)
    collision_ids = np.unique(np.r_[near,hand_ids])
    np.savez_compressed(OUT/'contact-modes.npz',original_frame_ids=np.arange(90,210),states=states,active=active,sticking=stick,signed_distance=distance,surface_speed=speed,contact_vertex_ids=contacts)
    model = torch.jit.load(str(ROOT/'assets/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt')).cuda().eval()
    for p in model.parameters(): p.requires_grad_(False)
    faces = np.load(ROOT/'baseline/faces.npy')
    metrics = {}
    for schedule in ['static','dynamic']:
        gate, sticking = (static_active,static_stick) if schedule == 'static' else (active,stick)
        lap = gated_laplacian(base_lap,gate,hand_relative,contact_relative,len(hand_ids))
        np.savez_compressed(OUT/f'{schedule}-structure.npz',laplacian=lap,reference_vertices=reference,hand_vertex_ids=hand_ids,object_surface_samples=objects)
        for with_mesh in [False,True]:
            name = schedule+('_mesh' if with_mesh else '_no_mesh')
            result = optimize(name,initial,(canon,obs,confidence,half,k),(lap,reference,objects),gate,sticking,hand_ids,contacts,collision_ids,model,fit_gate)
            values = measure(result['vertices'],result['rotation'],result['center'],half,np.array(meta['fixed_object_attachment_points']),contacts,faces,*original_inputs,k)
            values.update(diagnostics(result,validation,k,contacts))
            fitted = project_np(canon[None]@result['rotation'].transpose(0,2,1)+result['center'][:,None],k)
            values['fitting_reprojection_mean_px'] = float((np.linalg.norm(fitted-obs,axis=-1)*confidence).sum()/confidence.sum())
            contact_local = np.einsum('tpi,tij->tpj',result['vertices'][:,contacts]-result['center'][:,None],result['rotation'])
            sd = box_sdf(contact_local,half)
            values['active_contact_distance_mean_mm'] = float((np.abs(sd)*active).sum()/active.sum()*1000)
            values['inferred_state_counts'] = {NAMES[s]:int((states==s).sum()) for s in range(4)}
            metrics[name] = values
            np.savez_compressed(OUT/f'{name}.npz',original_frame_ids=np.arange(90,210),vertices=result['vertices'],mhr_model_params=result['native'],shape_params=initial['shape_params'],expr_params=initial['expr_params'],cam_t=initial['cam_t'],object_translation=result['center'],object_rotation=result['rotation'],object_half_extents=half)
    report = dict(protocol=protocol,methods=metrics,synthetic_checks=checks,seconds=time.monotonic()-started,
                  contact_state_summary={NAMES[s]:int((states==s).sum()) for s in range(4)},
                  state_switches_per_point=(states[1:]!=states[:-1]).sum(0).tolist(),
                  state_provenance='Estimated geometry proxy; separation in these labels is not verified real release')
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({name:{k:v for k,v in m.items() if not isinstance(v,(list,dict))} for name,m in metrics.items()},indent=2),flush=True)
    print('completed',report['seconds'],flush=True)


if __name__ == '__main__':
    main()
