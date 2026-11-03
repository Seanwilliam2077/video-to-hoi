"""Keep every selected thin-ring anchor inside the existing image-admission gate.

The original frame-level gate can admit a frame but subsequently select one of
its unsupported branches when contact or track-edge costs dominate. This
explicit initializer variant removes those branches before the same pose graph.
The reference bundle objective, renderer and full-timeline export are unchanged.
"""
import json
from pathlib import Path
import sys

import numpy as np


def supported_pose_path(candidates, costs, image_costs, frame_ids, diameter, fps,
                        observation_edges=None):
    from v2hoi.temporal_object import select_pose_path
    poses, costs, evidence = map(np.asarray, (candidates, costs, image_costs))
    if evidence.shape != costs.shape or not np.isfinite(evidence).all():
        raise ValueError('Invalid per-candidate image evidence')
    accepted = evidence <= .7
    if not accepted.any(1).all():
        raise ValueError('A pose-graph frame has no image-supported branch')
    # Duplicate supported branches to retain the rectangular graph interface.
    # Remapping edges preserves the original costs between the surviving poses.
    mapping = np.tile(np.arange(costs.shape[1]), (len(costs), 1))
    for i in range(len(costs)):
        good = np.flatnonzero(accepted[i])
        mapping[i, ~accepted[i]] = good[np.argmin(evidence[i, good])]
    rows = np.arange(len(costs))[:, None]
    edges = None
    if observation_edges is not None:
        edges = np.asarray([edge[np.ix_(a, b)] for edge, a, b in
                            zip(observation_edges, mapping[:-1], mapping[1:])])
        if len(costs) == 1:
            edges = np.empty((0, costs.shape[1], costs.shape[1]))
    path = select_pose_path(poses[rows, mapping], costs[rows, mapping], frame_ids,
                            diameter, fps, edges)
    result = mapping[np.arange(len(path)), path]
    if not accepted[np.arange(len(path)), result].all():
        raise ValueError('Selected an image-unsupported branch')
    return result


def main(ring_hypotheses=True):
    import torch
    from v2hoi import temporal_object_runtime as runtime
    from v2hoi import temporal_ring_tolerant_runtime as tolerant
    from v2hoi.native import sha256
    from v2hoi.object_refine_runtime import ObservationRenderer
    own_hash = sha256(Path(__file__))
    original_candidates = runtime.keyframe_candidates
    selected_evidence = []

    def candidates(mesh, images, masks, humans, depths, camera, ids, normal, offset,
                   fps, track_data, output, mask_proposals=False, contact=None):
        original_path = runtime.select_pose_path

        def path(poses, costs, active_ids, diameter, rate, edges=None):
            renderer = ObservationRenderer(mesh)
            image_costs = []
            for frame, alternatives in zip(active_ids, poses):
                frame = int(frame)
                depth = depths[frame].copy()
                depth[humans[frame].astype(bool)] = 0.
                visible = masks[frame].astype(bool) & ~humans[frame].astype(bool)
                care = ~humans[frame].astype(bool)
                scores = []
                for start in range(0, len(alternatives), 16):
                    batch = alternatives[start:start+16]
                    with torch.inference_mode():
                        _, alpha, z = renderer.render(
                            torch.as_tensor(batch[:, :3, :3], dtype=torch.float32, device='cuda'),
                            torch.as_tensor(batch[:, :3, 3], dtype=torch.float32, device='cuda'),
                            1., torch.zeros(3, device='cuda'),
                            np.repeat(camera[None], len(batch), axis=0), masks.shape[1:])
                    predicted = alpha[..., 0].cpu().numpy() > .5
                    overlap = (predicted & visible).sum((1, 2)) / np.maximum(
                        1, ((predicted | visible) & care).sum((1, 2)))
                    predicted_depth = z[..., 0].cpu().numpy()
                    for j in range(len(batch)):
                        support = predicted[j] & visible & (depth > .001)
                        error = (np.median(np.minimum(np.abs(predicted_depth[j][support]
                            - depth[support]), .2))/.2 if support.sum() >= 20 else 1.)
                        scores.append(1.-overlap[j]+.15*error)
                image_costs.append(scores)
            del renderer
            image_costs = np.asarray(image_costs)
            chosen = supported_pose_path(poses, costs, image_costs, active_ids,
                                         diameter, rate, edges)
            np.savez(output/'supported-branches.npz', original_frame_ids=active_ids,
                     image_costs=image_costs, selected_branches=chosen,
                     image_admission_threshold=np.array(.7))
            selected_evidence.extend(dict(original_frame_id=int(frame),
                supported_candidates=int((scores <= .7).sum()),
                chosen_image_cost=float(scores[branch]))
                for frame, scores, branch in zip(active_ids, image_costs, chosen))
            print(f'RING_SUPPORTED_GRAPH frames={len(active_ids)} '
                  f'max_chosen_image_cost={image_costs[np.arange(len(chosen)), chosen].max():.3f}', flush=True)
            return chosen

        runtime.select_pose_path = path
        try:
            return original_candidates(mesh, images, masks, humans, depths, camera,
                ids, normal, offset, fps, track_data, output, mask_proposals, contact)
        finally:
            runtime.select_pose_path = original_path

    runtime.keyframe_candidates = candidates
    if ring_hypotheses:
        tolerant.main()
    else:
        runtime.main()
    if '--worker' in sys.argv:
        output = Path(sys.argv[sys.argv.index('--output')+1])
        if sha256(Path(__file__)) != own_hash:
            raise ValueError('Supported-branch initializer changed during inference')
        target = output/'result.json'
        report = json.loads(target.read_text())
        report['inputs'][str(Path(__file__).resolve())] = own_hash
        report['outputs']['supported-branches.npz'] = sha256(output/'supported-branches.npz')
        evidence = dict(per_candidate_image_gate=True, image_admission_threshold=.7,
            selected_image_evidence=selected_evidence, final_bundle_objective_unchanged=True)
        report['image_supported_initialization'] = evidence
        if ring_hypotheses:
            report['thin_ring_initialization'].update(evidence)
        target.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
