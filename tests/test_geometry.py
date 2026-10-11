import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from v2hoi.geometry import ExactTriangleSDF, Similarity, SurfaceSDF, matrix_to_pose7, pose7_to_matrix, umeyama
from v2hoi.pose_review import orientation_hypotheses, rigid_pose, rotate_about_mesh_center


def test_pose7_roundtrip_and_invisible():
    rng = np.random.default_rng(0)
    q = Rotation.random(5, random_state=1).as_quat()  # xyzw
    pose7 = np.concatenate([rng.normal(size=(5, 3)), q[:, [3, 0, 1, 2]]], axis=1)
    pose7[2] = 0.0  # invisible frame: zero quaternion
    T = pose7_to_matrix(pose7)
    assert np.isnan(T[2]).all()
    assert np.allclose(T[0, :3, :3], Rotation.from_quat(q[0]).as_matrix())
    back = matrix_to_pose7(T)
    assert np.allclose(back[2], 0.0)
    ok = [0, 1, 3, 4]
    # q and -q are the same rotation
    same = np.minimum(np.abs(back[ok, 3:] - pose7[ok, 3:]).max(1), np.abs(back[ok, 3:] + pose7[ok, 3:]).max(1))
    assert np.allclose(back[ok, :3], pose7[ok, :3]) and same.max() < 1e-9


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(100, 3))
    R = Rotation.from_rotvec([0.2, -0.5, 0.9]).as_matrix()
    dst = 1.3 * src @ R.T + [0.5, -1.0, 2.0]
    s, R_hat, t = umeyama(src, dst, with_scale=True)
    assert np.isclose(s, 1.3) and np.allclose(R_hat, R) and np.allclose(t, [0.5, -1.0, 2.0])
    sim = Similarity.fit(src, dst, with_scale=True)
    assert np.allclose(sim.points(src), dst)


def test_surface_sdf_box():
    sdf = SurfaceSDF(trimesh.creation.box(extents=[1.0, 1.0, 1.0]), count=20_000)
    d = sdf(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.45], [0.0, 0.0, 0.49], [3.0, 0.0, 0.0]]))
    assert abs(d[0] + 0.5) < 0.02
    assert abs(d[1] + 0.05) < 0.02
    assert d[2] < 0
    assert np.isinf(d[3])


def test_exact_triangle_sdf_box_winding_padding_and_pose():
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    points = np.array([[0, 0, 0], [0, 0, 0.45], [0.5, 0, 0], [1, 0, 0]], dtype=float)
    expected = [-0.5, -0.05, 0, 0.5]
    assert np.allclose(ExactTriangleSDF(mesh, 2, 5)(points), expected)
    mesh.faces = np.concatenate([mesh.faces[:, ::-1], [[0, 0, 0]]])
    assert np.allclose(ExactTriangleSDF(mesh)(points), expected)
    rotation = Rotation.from_rotvec([0.4, -0.7, 0.1]).as_matrix()
    mesh.vertices = 2 * mesh.vertices @ rotation.T + [3, 2, 1]
    moved = 2 * points @ rotation.T + [3, 2, 1]
    assert np.allclose(ExactTriangleSDF(mesh)(moved), 2 * np.array(expected), atol=1e-12)


def test_exact_triangle_sdf_rejects_empty_or_degenerate_mesh():
    with pytest.raises(ValueError):
        ExactTriangleSDF(trimesh.Trimesh())
    with pytest.raises(ValueError, match="nondegenerate"):
        ExactTriangleSDF(trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]], process=False))


def test_similarity_does_not_truncate_integer_coordinates():
    assert np.allclose(Similarity(0.5, np.eye(3), np.ones(3) * 0.25).points(np.array([[1, 0, 0]])), [[0.75, 0.25, 0.25]])


def test_orientation_review_preserves_offset_center_and_physical_size():
    # Independent asymmetric, off-origin fixture: no challenge geometry.
    mesh = trimesh.creation.box(extents=[0.2, 0.3, 0.5])
    mesh.apply_translation([0.7, -0.2, 0.4])
    center = mesh.bounds.mean(axis=0)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0.3, -0.2, 0.8]).as_matrix()
    pose[:3, 3] = [1, 2, 3]
    hypotheses = orientation_hypotheses(pose, center)
    assert hypotheses.shape == (24, 4, 4)
    assert np.allclose(hypotheses[0], pose)
    assert len(np.unique(hypotheses.round(8).reshape(24, -1), axis=0)) == 24
    target_center = pose[:3, :3] @ center + pose[:3, 3]
    original_distance = np.linalg.norm(mesh.vertices[0] - mesh.vertices[6])
    for candidate in hypotheses:
        rigid_pose(candidate)
        assert np.allclose(candidate[:3, :3] @ center + candidate[:3, 3], target_center)
        points = mesh.vertices @ candidate[:3, :3].T + candidate[:3, 3]
        assert np.isclose(np.linalg.norm(points[0] - points[6]), original_distance)
    assert not np.allclose(hypotheses[1, :3, 3], pose[:3, 3])


@pytest.mark.parametrize("bad", [np.diag([2., 1, 1, 1]), np.diag([-1., 1, 1, 1]),
                                 np.full((4, 4), np.nan), np.eye(3)])
def test_orientation_review_rejects_nonrigid_input(bad):
    with pytest.raises(ValueError):
        orientation_hypotheses(bad, np.zeros(3))


def test_orientation_review_rejects_invalid_local_rotation():
    with pytest.raises(ValueError):
        rotate_about_mesh_center(np.eye(4), np.zeros(3), np.diag([2., 1, 1]))


@pytest.mark.parametrize("kind", ["box", "ring", "pan"])
def test_shared_object_scale_keeps_world_pose_pairing_for_different_shapes(kind):
    from v2hoi.object_refine import camera_centers_to_poses
    # Independent fixtures cover a closed solid, a hole and a thin handle shape.
    if kind == "box":
        mesh = trimesh.creation.box(extents=[.2, .3, .4])
    elif kind == "ring":
        mesh = trimesh.creation.annulus(r_min=.2, r_max=.3, height=.02)
    else:
        dish = trimesh.creation.cylinder(radius=.15, height=.02)
        handle = trimesh.creation.box(extents=[.3, .03, .02])
        handle.apply_translation([.27, 0, 0])
        mesh = trimesh.util.concatenate([dish, handle])
    mesh.apply_translation([.3, -.7, .1])
    center = mesh.bounds.mean(axis=0)
    rotations = Rotation.random(7, random_state=3).as_matrix()
    cameras = np.array([[.1+i*.01, .2-i*.02, 2.+i*.03] for i in range(7)])
    scale = 1.17
    poses = camera_centers_to_poses(rotations, cameras, center, scale)
    for r, c, pose in zip(rotations, cameras, poses):
        expected = (mesh.vertices-center)*scale @ r.T + c
        restored = (mesh.vertices*scale) @ pose[:3,:3].T + pose[:3,3]
        assert np.allclose(expected, restored)
    assert np.allclose(poses[:,3], [0,0,0,1])


def test_object_observation_anchors_are_support_driven_and_cover_the_sequence():
    from v2hoi.object_refine import anchor_indices, require_timeline, RefineConfig
    assert anchor_indices([0, 80, 0, 60, 0, 90], count=2).tolist() == [1, 5]
    assert anchor_indices([0, 0, 0]).tolist() == []
    require_timeline([0,1,2], 3)
    for ids in ([0,2], [1,2,3], [0,1,1]):
        with pytest.raises(ValueError):
            require_timeline(ids, 3)
    for config in (RefineConfig(steps=0), RefineConfig(scale_min=-1), RefineConfig(scale_max=.9)):
        with pytest.raises(ValueError):
            config.validate()


def test_multi_object_scene_identity_rejects_incompatible_humans_and_cameras():
    from v2hoi.object_refine import scene_identity
    first={"mhr_trans":np.zeros((4,3),dtype=np.float32)}
    same={"mhr_trans":first["mhr_trans"].copy()}
    moved={"mhr_trans":first["mhr_trans"]+.01}
    k=np.diag([100.,100.,1.])
    assert scene_identity(first,k)==scene_identity(same,k.copy())
    assert scene_identity(first,k)!=scene_identity(moved,k)
    assert scene_identity(first,k)!=scene_identity(first,np.diag([110.,100.,1.]))


def test_single_frame_mesh_is_not_misinterpreted_as_thousands_of_frames():
    from v2hoi.pose_review import one_frame_vertices
    mesh=np.arange(30,dtype=np.float32).reshape(10,3)
    assert one_frame_vertices(mesh).shape==(1,10,3)
    assert np.array_equal(one_frame_vertices(mesh),one_frame_vertices(mesh[None]))
    for invalid in (np.zeros((2,10,3)),np.zeros((10,2)),np.full((10,3),np.nan)):
        with pytest.raises(ValueError): one_frame_vertices(invalid)


def test_observation_seed_uses_support_instead_of_episode_or_first_frame():
    from v2hoi.object_refine import seed_frame_index
    assert seed_frame_index([50,200,100])==1
    assert seed_frame_index([500,500,100])==0
    with pytest.raises(ValueError): seed_frame_index([0,1,2])


def test_observation_bbox_ignores_remote_specks_without_losing_occluded_parts():
    from v2hoi.object_refine import observation_bbox
    mask=np.zeros((120,160),dtype=bool);mask[20:50,30:70]=True
    mask[25:30,80:95]=True  # substantial disconnected handle
    mask[110,150]=True  # unrelated one-pixel island
    assert observation_bbox(mask)==(30,20,94,49)
    assert mask[110,150]  # fitting/scoring input was not modified
    with pytest.raises(ValueError):observation_bbox(np.zeros((3,3),dtype=bool))


def test_semantic_face_hints_are_observations_not_reference_poses():
    from v2hoi.object_refine import face_observation_targets
    asset={'semantic_normal':[0,0,2],'observations':[{'original_frame_id':1},{'original_frame_id':5}],
        'face_observations':[{'original_frame_id':1,'visible_side':'front'},{'original_frame_id':5,'visible_side':'back'}]}
    normal,targets=face_observation_targets(asset,7)
    assert np.array_equal(normal,[0,0,1])
    assert targets.tolist()==[1,1,.5,0,-.5,-1,-1]
    assert face_observation_targets(None,7)==(None,None)
    asset['face_observations'][1]['original_frame_id']=4
    with pytest.raises(ValueError,match='permitted original observation'):face_observation_targets(asset,7)


def test_orientation_path_prevents_ambiguous_half_turns_and_retains_real_motion():
    from v2hoi.object_refine import select_orientation_path
    moving=Rotation.from_rotvec(np.array([[0.,0.,np.deg2rad(i*2.)] for i in range(8)])).as_matrix()
    flipped=moving@Rotation.from_rotvec([0,0,np.pi]).as_matrix()
    candidates=np.stack([moving,flipped],axis=1)
    costs=np.tile([0.,.01],(8,1));costs[1::2]=[.01,0.]
    choices=select_orientation_path(candidates,costs,np.arange(8))
    assert np.all(choices==choices[0])
    selected=candidates[np.arange(8),choices]
    assert np.allclose(selected[:-1].transpose(0,2,1)@selected[1:],Rotation.from_rotvec([0,0,np.deg2rad(2)]).as_matrix())


def test_image_rotation_evidence_requires_stable_visible_observations():
    from v2hoi.object_refine import image_rotation_evidence,RefineConfig
    rng=np.random.default_rng(9)
    patch=rng.random((32,32,3));images=np.repeat(patch[None],3,axis=0)
    masks=np.ones((3,32,32),dtype=bool);occluders=np.zeros_like(masks)
    assert np.allclose(image_rotation_evidence(images,masks,occluders),1.)
    images[1]=np.rot90(images[1])
    assert np.allclose(image_rotation_evidence(images,masks,occluders),0.)
    assert np.allclose(image_rotation_evidence(images,masks,np.ones_like(masks)),0.)
    assert RefineConfig().tracking_backend=='observations' and RefineConfig().full_hoi_preview
    with pytest.raises(ValueError,match='motion initializer'):RefineConfig(tracking_backend='unknown').validate()


def test_observation_center_accounts_for_projected_off_origin_handle():
    from v2hoi.object_refine import observation_centers
    dish=trimesh.creation.cylinder(radius=.15,height=.02)
    handle=trimesh.creation.box(extents=[.3,.03,.02]); handle.apply_translation([.27,0,0])
    mesh=trimesh.util.concatenate([dish,handle]); mesh.apply_translation([.3,-.7,.1])
    rotations=Rotation.from_euler('xyz',[[30,10,40],[70,30,10]],degrees=True).as_matrix()
    k=np.array([[700,0,400],[0,650,300],[0,0,1.]])
    uv=np.array([500,200]); scale=2.4; surface=2.
    c=observation_centers(mesh.vertices,rotations,mesh.bounds.mean(0),scale,surface,uv,k)
    points=(mesh.vertices-mesh.bounds.mean(0))*scale
    camera=points[None]@rotations.transpose(0,2,1)+c[:,None]
    pix=camera@k.T; pix=pix[...,:2]/pix[...,2:]
    assert np.allclose(camera[...,2].min(1),surface)
    assert np.max(np.abs((pix.min(1)+pix.max(1))/2-uv))<.01


def test_long_range_evidence_does_not_chain_small_changes_into_a_frozen_pose():
    from v2hoi.object_refine import rotation_anchor_pairs
    rng=np.random.default_rng(29);patch=rng.random((16,16,3))*.5
    frames=np.stack([patch+i*.012 for i in range(8)])
    masks=np.ones((8,16,16),bool);occluders=np.zeros_like(masks)
    pairs=rotation_anchor_pairs(frames,masks,occluders)
    assert len(pairs) and not any(a==0 and b==7 for a,b,_ in pairs)
    still=rotation_anchor_pairs(np.repeat(patch[None],8,0),masks,occluders)
    assert any(a==0 and b==7 and w==1 for a,b,w in still)
    assert not len(rotation_anchor_pairs(frames,masks,np.ones_like(masks)))
    assert not len(rotation_anchor_pairs(np.full_like(frames,.05),masks,occluders))


def test_landmarks_require_original_rgb_observation_receipts():
    from v2hoi.object_refine import landmark_observations
    row={'original_frame_id':2,'points_local':[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],'pixels':[[1,2],[2,2],[1,3],[2,3]]}
    asset={'observations':[{'original_frame_id':2}],'landmark_observations':[row]}
    assert landmark_observations(asset,4)[2][0].shape==(4,3)
    asset['appearance_rotation_observable']='yes'
    with pytest.raises(ValueError,match='explicit boolean'):landmark_observations(asset,4)
    asset['appearance_rotation_observable']=True
    asset['observations']=[]
    with pytest.raises(ValueError,match='permitted original RGB'):landmark_observations(asset,4)


def test_native_bpy_receipt_is_discovered_and_rejects_changed_observation(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from v2hoi import object_refine
    from v2hoi.native import sha256
    video=tmp_path/'synthetic.mp4';video.write_bytes(b'independent fixture, not decoded')
    clip=SimpleNamespace(episode=1,object_name='synthetic_box',n_frames=2,video=video)
    monkeypatch.setattr(object_refine,'load_clips',lambda *args:[clip])
    run=tmp_path/'run';run.mkdir();preparation=tmp_path/'preparation';preparation.mkdir()
    mesh=preparation/'object.glb';mesh.write_bytes(b'independent geometry receipt fixture')
    observation=preparation/'observation.png';observation.write_bytes(b'independent RGB receipt fixture')
    asset={'schema':'v2hoi.observed-asset.v1','entry':'blender.bpy.observed-profile','provenance':'track1',
        'episode':1,'object_instance_id':'synthetic_box','reference_geometry_used':False,'video_sha256':sha256(video),
        'mesh':{'path':mesh.name,'sha256':sha256(mesh)},'observations':[{'path':observation.name,'sha256':sha256(observation),'original_frame_id':0}]}
    receipt=preparation/'asset.json';receipt.write_text(json.dumps(asset))
    identity={'provenance':'track1','episode':1,'original_frame_ids':[0,1],'video_sha256':sha256(video),
        'preparation_manifest':str(preparation/'preparation.json'),
        'preparation':{'mesh_generation':{'entry':'blender.bpy.observed-profile','receipt':{'path':'asset.json','sha256':sha256(receipt)}}}}
    (run/'native-inputs.json').write_text(json.dumps(identity));(run/'native-status.json').write_text('{"complete":true}')
    for name in ['inference/refined.pth','inference/coconet.pth','depth/moge2/aligned.h5','export/episode_000001/wild_export.json']:
        path=run/'native/episode_000001'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'prerequisite fixture')
    assert object_refine.plan_jobs([run],tmp_path)[0][2]['asset_manifest']==str(receipt.resolve())
    observation.write_bytes(b'changed')
    with pytest.raises(ValueError,match='observation changed'):object_refine.plan_jobs([run],tmp_path)
