"""Exact original-frame RAW EgoHand cache contract and geometry-only proof.

No smoothed predictions are reused. Internal metadata omitted by the NPZ exporter
is restored only from the byte-identical source camera, anatomical side and ID.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import numpy as np

PINNED_RUNNER = '9cbfdd91c492da8d54f661d02f3cfba567e09ec6206ff477b4f2a36eac75770b'
PINNED_SOURCE = '9dfc9f25fe0b8c69a2f4f0bc01eb669097f7d1b9'
PINNED_CHECKPOINT = 'f940ecb6e3b73d5bd9e562cff232550e1c072f6c063eb4f705951122b15d8a75'
LEGACY_RESULTS = Path('J:/video-to-hoi/runs/egohand-pilots-20261031')
TASK = Path(__file__).resolve().parent
FIELD_MAP = {'poses': 'hand_poses', 'identity_coeffs': 'identity_coeffs',
             'pred_vertices': 'vertices', 'pred_keypoints_3d': 'joints', 'transl': 'cam_t',
             'scale_params': 'scale_params', 'pred_keypoints_2d': 'pred_keypoints_2d',
             'pred_projection_valid': 'pred_projection_valid',
             'pred_kp2d_crop_presence': 'pred_kp2d_crop_presence', 'pred_kp2d_sigma': 'pred_kp2d_sigma'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def restore_records(arrays, K, side, count):
    expected = [f'{i:06d}' for i in range(count)]
    if int(arrays['num_frames']) != count or arrays['frame_names'].tolist() != expected:
        raise ValueError('Legacy RAW export must contain every exact original frame')
    for field in (*FIELD_MAP.values(), 'global_scale'):
        if field not in arrays or len(arrays[field]) != count:
            raise ValueError(f'Legacy RAW cache is missing required field: {field}')
        if not np.isfinite(arrays[field]).all():
            raise ValueError(f'Legacy RAW cache is nonfinite: {field}')
    if arrays['joints'].shape != (count, 25, 3) or arrays['cam_t'].shape != (count, 3):
        raise ValueError('Legacy RAW cache geometry layout differs')
    result = {}
    for i, frame in enumerate(expected):
        row = {key: arrays[field][i].copy() for key, field in FIELD_MAP.items()}
        row.update(global_scale=float(arrays['global_scale'][i]), hand_type=side, image_id=frame,
                   K_fullimg=np.asarray(K, dtype=np.float32).copy())
        result[frame] = row
    return result


def load_legacy_raw(entry, K):
    episode, count = entry['episode'], entry['frames']
    if episode not in (16, 11) or count != {16: 360, 11: 877}[episode]:
        raise ValueError('Legacy cache is restricted to the two authorized Track 1 videos')
    root = LEGACY_RESULTS / f'episode_{episode:06d}' / 'egohand-raw-attempt01'
    receipt_path = root / 'prediction-receipt.json'
    receipt = read(receipt_path)
    expected = {'status': 'completed', 'episode': episode, 'frames': count, 'source_frames': count,
                'source_video_sha256': entry['video_sha256'], 'complete_source_coverage': True,
                'upstream_commit': PINNED_SOURCE, 'runner_sha256': PINNED_RUNNER,
                'checkpoint_sha256': PINNED_CHECKPOINT, 'device': 'cpu', 'dtype': 'float32',
                'perspective_cropping': False, 'pixel_aspect_rectification': False, 'box_expansion': False}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f'Legacy RAW cache inference condition differs: {key}')
    old = receipt['input_bindings']
    if old['camera_sha256'] != entry['camera_sha256'] or old['fps'] != entry['fps']:
        raise ValueError('Legacy RAW cache camera or original timing differs')
    if digest(old['camera']) != old['camera_sha256']:
        raise ValueError('Legacy source camera bytes changed')
    old_K = np.asarray(read(old['camera'])['camera_matrix']['matrix_3x3'], dtype=np.float32)
    if not np.array_equal(old_K, np.asarray(K, dtype=np.float32)):
        raise ValueError('Legacy source camera matrix differs')
    cfg = receipt['model_load']['checkpoint_cfg']
    if cfg['image_size'] != 256 or cfg['backbone_input_crop_w'] is not None or cfg['camera_mode'] != 'sam3db_perspective':
        raise ValueError('Legacy trained sampling/camera contract differs')
    if digest(old['original_observations']) != old['original_observations_sha256']:
        raise ValueError('Original RAW source crop proof changed')
    with np.load(old['original_observations'], allow_pickle=False) as archive:
        old_boxes = archive['crop_boxes'].copy()
        if archive['frame_indices'].tolist() != list(range(count)) or archive['side_order'].tolist() != ['left', 'right']:
            raise ValueError('Original source crop frame/side layout differs')
    for item in old['saved_boxes']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Exact input box JSON changed')
        s = ['left', 'right'].index(item['side'])
        stored = read(item['path'])
        for i, box in enumerate(old_boxes[:, s]):
            original = np.asarray([(box[0]+box[2])/2, (box[1]+box[3])/2, box[2]-box[0]], dtype=np.float32)
            if not np.array_equal(original, np.asarray(stored[f'{i:06d}'], dtype=np.float32)):
                raise ValueError('Original observation bounds do not match executed EgoHand input boxes')
    records, outputs = {}, {}
    for item in receipt['raw_outputs']:
        if item['variant'] != 'raw' or item['frames'] != count or not item['complete_source_coverage']:
            raise ValueError('Only complete RAW outputs can supply cache records')
        if digest(item['path']) != item['sha256']:
            raise ValueError('Legacy RAW export bytes changed')
        side = item['side']
        with np.load(item['path'], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        records[side] = restore_records(arrays, K, side, count)
        outputs[side] = {'path': item['path'], 'sha256': item['sha256']}
    if set(records) != {'left', 'right'}:
        raise ValueError('Both anatomical hand caches are required')
    lineage = {'receipt': str(receipt_path), 'receipt_sha256': digest(receipt_path),
               'raw_outputs': outputs, 'source_video_sha256': receipt['source_video_sha256'],
               'camera_sha256': old['camera_sha256'], 'original_observations_sha256': old['original_observations_sha256'],
               'upstream_commit': PINNED_SOURCE, 'checkpoint_sha256': PINNED_CHECKPOINT,
               'original_runner_sha256': PINNED_RUNNER, 'sampling': 'Official affine256 bilinear RGB/ImageNet; FP32 CPU; no perspective/aspect rectification/box expansion',
               'restored_internal_metadata': ['K_fullimg from exact original K', 'hand_type from anatomical side', 'image_id from original frame ID'],
               'smoothed_cache_reused': False, 'prediction_interpolation': False}
    return records, old_boxes, lineage


def proof():
    import argparse
    from datetime import datetime, timezone
    import torch
    import warp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(TASK):
        raise ValueError('Use a fresh task-local proof receipt')
    old_inputs = read(args.asset_root / 'inputs/manifest.json')
    entry = next(value for value in old_inputs['episodes'] if value['episode'] == 16)
    K = np.asarray(read(entry['camera'])['camera_matrix']['matrix_3x3'], dtype=np.float32)
    records, old_boxes, lineage = load_legacy_raw(entry, K)
    golden_receipt = read(Path('J:/video-to-hoi/runs/egohand-refined-pilots-20261031/episode_000016/smoke-cpu-attempt01/prediction-receipt.json'))
    trial_boxes = np.load(TASK / 'crops/episode_000016/crop_boxes.npy', allow_pickle=False)
    golden_checks = []
    for item in golden_receipt['raw_outputs']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Golden RAW output bytes changed')
        side = item['side'];s = ['left', 'right'].index(side)
        with np.load(item['path'], allow_pickle=False) as a:
            for row, name in enumerate(a['frame_names'].tolist()):
                i = int(name)
                if not np.array_equal(trial_boxes[i, s], old_boxes[i, s]):
                    continue
                old = records[side][name]
                j = float(np.linalg.norm(a['joints'][row]-old['pred_keypoints_3d'], axis=-1).max())
                t = float(np.linalg.norm(a['cam_t'][row]-old['transl']))
                p = np.asarray(a['joints'][row]+a['cam_t'][row], dtype=np.float64)@K.T
                q = np.asarray(old['pred_keypoints_3d']+old['transl'], dtype=np.float64)@K.T
                pixel = float(np.linalg.norm(p[:,:2]/p[:,2:]-q[:,:2]/q[:,2:], axis=-1).max())
                assert j<=1e-4 and t<=1e-4 and pixel<=.1
                golden_checks.append({'side':side,'original_frame_id':i,'joint_norm_metres':j,'translation_norm_metres':t,'source_projection_pixels':pixel})
    assert all(any(row['side']==side and row['original_frame_id']==0 for row in golden_checks) for side in ['left','right'])
    torch.set_num_threads(12);torch.set_num_interop_threads(1)
    warp.config.kernel_cache_dir = str(TASK/'cache/warp')
    original = args.asset_root/'run_controlled_egohand.py'
    if digest(original)!=PINNED_RUNNER:raise ValueError('Original loading/export recipe changed')
    spec=importlib.util.spec_from_file_location('cache_proof_original_recipes',original);recipes=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipes)
    model,model_report=recipes.load_model('cpu')
    def forbid_neural_forward(*args,**kwargs):raise AssertionError('A cache geometry proof must never call neural forward')
    model.forward_step=forbid_neural_forward
    from inference.pipeline.predictor import EgoHandPredictor
    predictor=EgoHandPredictor(model,8,device='cpu',dtype=torch.float32,smooth=True,use_perspective_cropping=False)
    ids=[f'{i:06d}' for i in range(360)]
    boxes={name:{side:np.asarray([(b[0]+b[2])/2,(b[1]+b[3])/2,b[2]-b[0]],dtype=np.float32) for side,b in zip(['left','right'],old_boxes[i])} for i,name in enumerate(ids)}
    smoothed=copy.deepcopy(records)
    with torch.inference_mode():predictor.postprocess_results(smoothed,ids,boxes,frame_times={name:int(name)/30 for name in ids},intrinsics_per_image={name:K for name in ids})
    target=TASK/'cache-proof-ep16-reexport01'
    if target.exists():raise ValueError('Preserve previous geometry proof export')
    exports=recipes.export(smoothed,target)
    old_receipt=read(lineage['receipt']);roundtrip=[]
    for new in exports:
        old=next(x for x in old_receipt['smoothed_outputs'] if x['side']==new['side'])
        if digest(old['path'])!=old['sha256']:raise ValueError('Original smoothing output changed')
        with np.load(new['path'],allow_pickle=False) as a,np.load(old['path'],allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            assert np.array_equal(a['frame_names'],b['frame_names'])
            diffs={key:float(np.max(np.abs(a[key].astype(np.float64)-b[key].astype(np.float64)))) for key in a.files if a[key].dtype.kind in 'fci'}
            s=['left','right'].index(new['side']);pixel=float(np.linalg.norm((a['pred_keypoints_2d']-b['pred_keypoints_2d'])*((old_boxes[:,s,2]-old_boxes[:,s,0])[:,None,None]),axis=-1).max())
            assert max(diffs.get(k,0) for k in ['vertices','joints','cam_t'])<=1e-5
            assert max(diffs.get(k,0) for k in ['hand_poses','identity_coeffs','global_scale','scale_params'])<=1e-6
            assert pixel<=1e-3 and np.array_equal(a['pred_projection_valid'],b['pred_projection_valid'])
        roundtrip.append({'side':new['side'],'frames':360,'old_smoothed_sha256':old['sha256'],'recomputed_smoothed_sha256':new['sha256'],'max_abs_differences':diffs,'source_projection_norm_difference_pixels':pixel,'passed':True})
    report={'recorded_at':datetime.now(timezone.utc).isoformat(),'track':1,'episode':16,'passed':True,'neural_forward_calls':0,'raw_cache_lineage':lineage,'golden_matching_crop_checks':golden_checks,'whole_clip_official_smoothing_roundtrip':roundtrip,'model_camera_mode':model_report['camera_mode'],'proof_source_sha256':digest(__file__),'quality_assessed':False,'all30_execution_started':False}
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'passed':True,'output':str(args.output),'matching_golden_views':len(golden_checks),'neural_forward_calls':0}),flush=True)


if __name__=='__main__':proof()
