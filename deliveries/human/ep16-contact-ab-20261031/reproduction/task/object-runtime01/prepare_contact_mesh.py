"""Separate closed collision topology from duplicated texture UV topology."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.spatial import cKDTree
import trimesh
from fit_object_hypotheses import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path,required=True)
    args=parser.parse_args()
    source=args.candidate/'object_candidate.npz';p=np.load(source,allow_pickle=False)
    validation=json.loads((args.candidate/'validation.json').read_text())
    if not validation['weak_contact_experiment_guards_passed']:
        raise ValueError('Pose has not passed weak contact experiment guards')
    if (args.candidate/'contact_ready.npz').exists():raise FileExistsError('Preserve existing export')
    mesh=trimesh.Trimesh(vertices=p['vertices'],faces=p['faces'],process=True)
    mesh.merge_vertices(merge_tex=True,merge_norm=True,digits_vertex=7)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0:
        raise ValueError('Collision geometry must be closed with consistent outward winding')
    max_nearest=float(cKDTree(mesh.vertices).query(p['vertices'])[0].max())
    if max_nearest>1e-7:raise ValueError('Topology operation changed geometry materially')
    triangles=mesh.vertices[mesh.faces]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    offsets=(normals*triangles[:,0]).sum(axis=1)
    plane_violation=float((mesh.vertices@normals.T-offsets).max())
    if plane_violation>1e-7:raise ValueError('Geometry fails outward convex supporting planes')
    T=p['object_to_camera']
    assert np.isfinite(T).all() and np.allclose(T[:,3],np.array([0,0,0,1]))
    assert np.allclose(np.linalg.det(T[:,:3,:3]),1,atol=1e-6)
    assert p['frame_indices'].tolist()==list(range(360))
    experiment=json.loads((args.candidate/'report.json').read_text())
    diagnostics=experiment['temporal_refinement']['diagnostics']
    quality_iou=np.array([d['candidate_mask_iou'] for d in diagnostics])
    heldout=np.array([d['candidate_heldout_mean_px'] if d['candidate_heldout_mean_px'] is not None else np.nan for d in diagnostics])
    pose_usable=(quality_iou>=.9)&np.isfinite(heldout)&(heldout<=3.)
    np.savez_compressed(args.candidate/'contact_ready.npz',frame_indices=p['frame_indices'],rotation=T[:,:3,:3],translation=T[:,:3,3],object_to_camera=T,vertices=mesh.vertices,faces=mesh.faces,camera_intrinsics=p['camera_intrinsics'],fps=np.float32(30),scale_sensitivity=np.array([.8,1.,1.2]),pose_usable=pose_usable,pose_mask_iou=quality_iou,pose_heldout_mean_px=heldout,data_provenance=np.array('Track 1 episode16 own RGB/SAM2-mask parametric mesh hypothesis, camera/scale and hidden geometry unverified'))
    mesh.export(args.candidate/'geometry.ply')
    report={'source_npz_sha256':digest(source),'contact_npz_sha256':digest(args.candidate/'contact_ready.npz'),'before_vertices':len(p['vertices']),'before_faces':len(p['faces']),'after_vertices':len(mesh.vertices),'after_faces':len(mesh.faces),'geometry_max_nearest_change_m':max_nearest,'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),'convex_supporting_plane_max_violation_m':plane_violation,'volume_m3':float(mesh.volume),'extents_m':mesh.extents.tolist(),'pose_usable_frames':int(pose_usable.sum()),'pose_usable_semantics':'Binary conservative gate, not probability: per-frame conditional heldout mean<=3px and visiblemaskIoU>=.9. Missing heldout evidence gets0. Global motion/image guards also required.','notice':'Contact-ready denotes passed weak-experiment guards plus closed topology and finite poses only. It does not certify camera, scale, hidden geometry or physical contact accuracy. Scale sensitivity0.8/1/1.2 is a chosen experiment range, not a calibrated interval.'}
    (args.candidate/'contact-export-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
