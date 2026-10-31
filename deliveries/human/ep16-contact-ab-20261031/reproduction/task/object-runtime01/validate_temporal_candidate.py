"""Audit pose continuity and spatially distinct held-out image evidence."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def project(vertices,T,K):
    xyz=vertices@T[:3,:3].T+T[:3,3];h=xyz@K.T
    return h[:,:2]/h[:,2:]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path,required=True)
    args=parser.parse_args();report=json.loads((args.candidate/'report.json').read_text())
    p=np.load(args.candidate/'object_candidate.npz',allow_pickle=False)
    s=report['temporal_refinement'];source=Path(s['source_candidate'])
    initial=np.load(source/'object_candidate.npz',allow_pickle=False)
    evidence=np.load(args.candidate/'feature_evidence.npz',allow_pickle=False)
    T=p['object_to_camera'];K=p['camera_intrinsics']
    assert p['frame_indices'].tolist()==list(range(360))
    assert np.isfinite(T).all() and np.isfinite(p['vertices']).all()
    assert np.allclose(T[:,3],np.array([0,0,0,1]))
    assert np.allclose(np.linalg.det(T[:,:3,:3]),1,atol=1e-6)
    assert np.array_equal(K,initial['camera_intrinsics'])
    # SIFT can emit several orientations at one pixel. Exclude any test feature
    # lying within2 source pixels of ANY training-reference point, rather than
    # counting duplicate descriptors as independent held-out evidence.
    all_train=np.concatenate([evidence[f'xyz_{i:03d}'] for i in range(360)])
    train_reference=project(all_train,initial['object_to_camera'][0],K)
    tree=cKDTree(train_reference)
    before=[];after=[];counts=[]
    for i in range(360):
        xyz=evidence[f'test_xyz_{i:03d}'];xy=evidence[f'test_xy_{i:03d}']
        if not len(xyz): counts.append(0);continue
        reference=project(xyz,initial['object_to_camera'][0],K)
        distinct=tree.query(reference)[0]>2.
        counts.append(int(distinct.sum()))
        if distinct.any():
            before.extend(np.linalg.norm(project(xyz[distinct],initial['object_to_camera'][i],K)-xy[distinct],axis=1))
            after.extend(np.linalg.norm(project(xyz[distinct],T[i],K)-xy[distinct],axis=1))
    def stats(values):
        return {'count':len(values),'mean_px':float(np.mean(values)),'median_px':float(np.median(values)),'p90_px':float(np.percentile(values,90)),'max_px':float(np.max(values))}
    motion=s['candidate_motion']
    qualified=bool(motion['translation_step_max_m']<=.04 and motion['rotation_step_max_deg']<=8 and s['mean_candidate_mask_iou']>=s['mean_initial_mask_iou']-.025 and len(after)>=100 and np.mean(after)<=min(3.,np.mean(before)))
    result={'finite_exact_frame_camera_rotation_checks':'passed','spatially_distinct_heldout':{'protocol':'Test query IDs are excluded from objective; additionally remove points within2 source pixels of any training-reference feature to eliminate SIFT orientation duplicates. Conditional on hypothesized geometry, not true3D/scale.','before':stats(before),'after':stats(after),'frames_with_distinct_test_points':sum(c>0 for c in counts)},'guard_thresholds':{'max_translation_step_m':.04,'max_rotation_step_deg':8,'max_mean_mask_iou_loss':.025,'max_distinct_heldout_mean_px':3.,'min_distinct_heldout_points':100},'weak_contact_experiment_guards_passed':qualified,'not_certified':'Physical pose/scale, hidden geometry, contact and official challenge score remain unverified; +/-20percent is chosen sensitivity only.'}
    (args.candidate/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
