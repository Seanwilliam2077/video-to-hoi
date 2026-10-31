"""Read-only independent hull, coordinate and native-hand conflict diagnostic."""
from pathlib import Path
from datetime import datetime,timezone
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import hashlib,json,sys
import numpy as np
from scipy.spatial import ConvexHull
import torch
ROOT=Path('C:/Users/arthasxu/.codex/worktrees/human-all30/video-to-hoi')
sys.path.insert(0,str(ROOT/'src'))
from v2hoi.hand_contact import native_hand_vertex_indices,closest_mesh_points
from v2hoi.hand_refinement import NativeMHRHands,HAND_JOINTS
from v2hoi.human_result import load_human_result

T=Path(__file__).resolve().parent
R=Path('J:/video-to-hoi/runs/ep16-contact-ab-20261031')
BASE=Path('J:/video-to-hoi/runs/human-all30-20261021/episode_000016/baseline/attempt-0001')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stats(x):
    x=np.asarray(x);return {'count':len(x),'min':float(x.min()),'mean':float(x.mean()),'max':float(x.max())}

def main():
    torch.set_num_threads(1)
    inputs=json.loads((T/'evaluation-inputs01.json').read_text())
    object_path=R/'object-final01/contact_ready.npz'
    assert sha(object_path)==inputs['object']['sha256']=='95f696082ef636c7860b89ca0e2c3206b5ebebe433f02e270dc39a24954678a9'
    p=np.load(object_path,allow_pickle=False)
    K=np.load(BASE/'review_arrays/camera_intrinsics.npy',allow_pickle=False)
    assert np.array_equal(p['camera_intrinsics'],K)
    native={name:load_human_result(R/folder/'episode_000016_human.npz') for name,folder in [('a','A-no-contact01'),('b','B-manual-contact01')]}
    original=load_human_result(BASE/'episode_000016_human.npz')
    paths=[object_path,BASE/'episode_000016_human.npz',T/'render-hand-skin-indices01.npz']
    for role,folder in [('a','A-no-contact01'),('b','B-manual-contact01')]:
        path=R/folder/'episode_000016_human.npz';assert sha(path)==inputs[role]['sha256'];paths.append(path)
        assert np.array_equal(np.load(R/folder/'review_arrays/camera_intrinsics.npy',allow_pickle=False),K)
    before={str(path):sha(path) for path in paths}
    model_path=Path('J:/video-to-hoi/weights/human/mhr_model.pt')
    assert sha(model_path)=='352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'
    decoder=NativeMHRHands(str(model_path),device='cpu')
    cache=np.load(T/'render-hand-skin-indices01.npz',allow_pickle=False)
    skin=native_hand_vertex_indices(decoder.model,threshold=.5)
    assert all(np.array_equal(value,cache[name]) for value,name in zip(skin,('left','right')))
    cache_threshold_note={'actual_verified_threshold':.5,'actual_cache_counts':[len(x) for x in skin],
        'all_cache_indices_equal_rig_recomputed':True,
        'counts_if_threshold_0p1':[len(x) for x in native_hand_vertex_indices(decoder.model,threshold=.1)],
        'historical_benchmark_label_erratum':'hand-skin-display-benchmark01.json says >=.1; actual cached indices exactly match .5, not .1. This is stale metadata wording, not a geometry change.'}
    av=np.load(R/'A-no-contact01/review_arrays/vertices_m.npy',mmap_mode='r',allow_pickle=False)
    bv=np.load(R/'B-manual-contact01/review_arrays/vertices_m.npy',mmap_mode='r',allow_pickle=False)
    basev=np.load(BASE/'review_arrays/vertices_m.npy',mmap_mode='r',allow_pickle=False)
    joints=np.load(R/'A-no-contact01/hand_joints_m.npy',mmap_mode='r',allow_pickle=False)
    ids=np.asarray([90,120,210,250]);x=native['a']
    with torch.no_grad():
        pose=torch.as_tensor(np.asarray(x['pose'])[ids]);scales=torch.as_tensor(np.asarray(x['scales']))
        shape=torch.as_tensor(np.asarray(x['shape']))
        params=torch.cat([pose,scales[None].expand(len(ids),-1)],1)
        vertices,fullj=decoder.model(shape[None].expand(len(ids),-1).contiguous(),params,pose.new_zeros(len(ids),72),True)
        factor=pose.new_tensor([.01,-.01,-.01]);newv=(vertices*factor).cpu().numpy();newj=(fullj[:,HAND_JOINTS.reshape(-1),:3]*factor).reshape(len(ids),2,21,3).cpu().numpy()
    decode_checks={'frames':ids.tolist(),'full_native_vertices_vs_saved_max_absolute_m':float(np.abs(newv-av[ids]).max()),
        'full_native_hand_joints_vs_saved_max_absolute_m':float(np.abs(newj-joints[ids]).max()),
        'native_conversion':'Native centimetres/y-up/z-back multiplied by[.01,-.01,-.01] into OpenCV metres/y-down/z-forward',
        'joint_names':[decoder.model.character_torch.skeleton.joint_names[i] for i in (78,42)]}
    assert decode_checks['full_native_vertices_vs_saved_max_absolute_m']<2e-6
    assert decode_checks['full_native_hand_joints_vs_saved_max_absolute_m']<2e-6
    local_hull=ConvexHull(p['vertices']);eq=local_hull.equations
    rows=[]
    for frame in ids:
        transform=p['object_to_camera'][frame];rotation=transform[:3,:3];translation=transform[:3,3]
        world_object=p['vertices']@rotation.T+translation
        world_equations=ConvexHull(world_object).equations
        row={'frame':int(frame),'object_center_camera_m':translation.tolist(),
            'object_camera_z_minmax_m':[float(world_object[:,2].min()),float(world_object[:,2].max())],
            'object_local_bbox_m':[p['vertices'].min(0).tolist(),p['vertices'].max(0).tolist()],
            'pose_usable':bool(p['pose_usable'][frame]),'hands':{}}
        for side,name in enumerate(('left','right')):
            camera=av[frame,skin[side]];local=(camera-translation)@rotation
            alternative=(np.linalg.inv(transform)@np.c_[camera,np.ones(len(camera))].T).T[:,:3]
            local_signed=(local@eq[:,:3].T+eq[:,3]).max(1)
            world_signed=(camera@world_equations[:,:3].T+world_equations[:,3]).max(1)
            assert np.array_equal(local_signed<0,world_signed<0)
            assert np.abs(local_signed-world_signed).max()<1e-10
            inside=np.maximum(-local_signed,0)
            chosen=np.flatnonzero(local_signed<0)
            selected=chosen[np.linspace(0,len(chosen)-1,min(16,len(chosen)),dtype=int)] if len(chosen) else chosen
            if len(selected):
                _,tri_distance=closest_mesh_points(local[selected],p['vertices'],p['faces'])
                distance_error=float(np.abs(tri_distance-inside[selected]).max())
            else:distance_error=None
            handj=(joints[frame,side]-translation)@rotation
            row['hands'][name]={'skin_vertices':len(camera),'penetrating_vertices':int((inside>0).sum()),
                'penetrating_fraction':float((inside>0).mean()),'mean_depth_over_all_hand_vertices_mm':float(inside.mean()*1000),
                'max_depth_mm':float(inside.max()*1000),'local_skin_bbox_m':[local.min(0).tolist(),local.max(0).tolist()],
                'wrist_local_m':handj[0].tolist(),'palm_knuckle_centroid_local_m':handj[[5,9,13,17]].mean(0).tolist(),
                'tip_order':['thumb','index','middle','ring','pinky'],'fingertip_local_m':handj[[4,8,12,16,20]].tolist(),
                'skin_camera_z_minmax_m':[float(camera[:,2].min()),float(camera[:,2].max())],
                'wrist_camera_z_m':float(joints[frame,side,0,2]),
                'a_vs_original_SAM_skin_max_abs_m':float(np.abs(camera-basev[frame,skin[side]]).max()),
                'a_vs_b_skin_max_abs_m':float(np.abs(camera-bv[frame,skin[side]]).max()),
                'row_transform_vs_inverse_homogeneous_max_abs_m':float(np.abs(local-alternative).max()),
                'independent_world_vs_local_hull_max_signed_difference_m':float(np.abs(local_signed-world_signed).max()),
                'inside_exact_triangle_vs_support_plane_distance_max_difference_m':distance_error}
        rows.append(row)
    use=np.arange(75,251);use=use[p['pose_usable'][use]]
    preservation={'usable_carry_frames':use.tolist(),'A_B_left_finger_controls_exact':bool(np.array_equal(native['a']['pose'][use,95:122],native['b']['pose'][use,95:122])),
        'A_left_finger_controls_exact_original_SAM':bool(np.array_equal(native['a']['pose'][use,95:122],original['pose'][use,95:122])),
        'A_B_left_selected_skin_exact':bool(np.array_equal(av[use][:,skin[0]],bv[use][:,skin[0]]))}
    assert before=={str(path):sha(path) for path in paths}
    report={'schema_version':'ep16_object_left_hand_conflict_audit.v1','recorded_at':datetime.now(timezone.utc).isoformat(),
        'track':1,'episode':16,'scope':'Read-only coordinate/topology/rig validation and conflict attribution against one fixed uncertain reconstructed object; no object or hand corrections.',
        'source_sha256':sha(Path(__file__)),'protected_inputs_sha256_before_and_after':before,
        'native_decoder':decode_checks,'verified_skin_selection':cache_threshold_note,'carry_parameter_preservation':preservation,
        'rows':rows,'interpretation':[
            'Independent posed-world ConvexHull checks agree with object-local checks; the large left invasion is present in the saved coordinates rather than introduced by a transpose, handedness, native-unit or reversed-normal test bug.',
            'At the sampled carry frames, A left skin remains the original SAM prediction within decoder numerical tolerance and A/B left skin is exactly equal. B contact attraction did not create the left invasion.',
            'The left palm and fingers lie deep within the assumed cube. At frame210 its entire skin bounding box lies within the cube interior; right hand stays mainly near/outside the +localZ surface.',
            'The unobserved left pose is a prior while object hidden shape and center-depth gauge are assumptions. These checks identify a joint human-object incompatibility, not which estimate is physically wrong.',
            'Shared camera and units establish mathematical consistency only. Source RGB alone does not identify a unique hidden left articulation, object backside or exact relative depth.',
            'Do not tune object scale/pose after A/B or infer improved official penetration/true contact from these diagnostics.'],
        'object_camera_mesh_scale_or_pose_changed':False,'quality_accepted':False,'contact_or_scale_accepted':False,'official_score_available':False}
    path=T/'object-left-hand-conflict-audit01.json'
    with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'output':str(path),'sha256':sha(path),'native_decoder':decode_checks,'verified_skin_selection':cache_threshold_note,'preservation':{k:v for k,v in preservation.items() if k!='usable_carry_frames'},'per_frame':[{r['frame']:{name:{k:x[k] for k in ('penetrating_fraction','mean_depth_over_all_hand_vertices_mm','max_depth_mm')} for name,x in r['hands'].items()}} for r in rows]}),flush=True)

if __name__=='__main__':main()
