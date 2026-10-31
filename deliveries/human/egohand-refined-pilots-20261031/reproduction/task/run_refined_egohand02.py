"""Source-pinned EgoHand inference with exact RAW cache reuse for the two pilots.

The original asset root is read-only. Crop boxes are consumed exactly as frozen;
visibility admission and native-MHR retargeting belong to later stages.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'win32'
TASK = Path(__file__).resolve().parent
RESULTS = Path('J:/video-to-hoi/runs/egohand-refined-pilots-20261031').resolve()
PINNED_RUNNER_SHA256 = '9cbfdd91c492da8d54f661d02f3cfba567e09ec6206ff477b4f2a36eac75770b'
PINNED_SOURCE = '9dfc9f25fe0b8c69a2f4f0bc01eb669097f7d1b9'
PINNED_DINO = '5aa7e93ab42ae3526291f6b4e3fcd22637e03326'
PINNED_CHECKPOINT = 'f940ecb6e3b73d5bd9e562cff232550e1c072f6c063eb4f705951122b15d8a75'
COUNTS = {16: 360, 11: 877}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, *, exclusive=True):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def expected_file(path, expected):
    path = Path(path).resolve()
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f'Frozen file identity changed: {path}')
    return path


def revision(path):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()


def load_pinned_recipes(asset_root):
    original = expected_file(asset_root / 'run_controlled_egohand.py', PINNED_RUNNER_SHA256)
    expected_file(asset_root / 'inference-runtime01/run_controlled_egohand.py', PINNED_RUNNER_SHA256)
    if revision(asset_root / 'upstream-git') != PINNED_SOURCE or revision(asset_root / 'dinov3-source') != PINNED_DINO:
        raise ValueError('Public source revisions differ from the completed original pilots')
    manifest_path = asset_root / 'weights-verified-upstream-objects.json'
    manifest = read(manifest_path)
    if len(manifest['assets']) != 13:
        raise ValueError('Expected exactly the verified generic public asset allowlist')
    for entry in manifest['assets']:
        path = Path(entry['path']).resolve()
        if not entry['verified'] or not path.is_relative_to(asset_root) or path.stat().st_size != entry['bytes']:
            raise ValueError('Generic asset scope or length changed')
        expected_file(path, entry['sha256'])
    checkpoint = expected_file(asset_root / 'weights/egohand/egohand_s060000.ckpt', PINNED_CHECKPOINT)
    spec = importlib.util.spec_from_file_location('immutable_original_egohand_recipes', original)
    recipes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipes)
    return recipes, {'asset_root': str(asset_root), 'original_recipe_sha256': PINNED_RUNNER_SHA256,
                     'verified_asset_manifest': str(manifest_path), 'verified_asset_manifest_sha256': digest(manifest_path),
                     'asset_count': 13, 'checkpoint': str(checkpoint), 'checkpoint_sha256': PINNED_CHECKPOINT,
                     'upstream_commit': PINNED_SOURCE, 'dinov3_commit': PINNED_DINO}



def export_sparse_chunk(results, path):
    """Save only freshly inferred hand views; cached/empty sides are not fabricated."""
    import numpy as np
    path.mkdir(parents=True, exist_ok=False)
    output = []
    mapping = {'hand_poses': 'poses', 'identity_coeffs': 'identity_coeffs',
               'global_scale': 'global_scale', 'vertices': 'pred_vertices',
               'joints': 'pred_keypoints_3d', 'cam_t': 'transl', 'scale_params': 'scale_params',
               'pred_kp2d_crop_presence': 'pred_kp2d_crop_presence', 'pred_kp2d_sigma': 'pred_kp2d_sigma',
               'pred_keypoints_2d': 'pred_keypoints_2d', 'pred_projection_valid': 'pred_projection_valid',
               'source_bbox': 'bbox'}
    for side in ('left', 'right'):
        ids = sorted(results[side], key=int)
        if not ids:
            continue
        values = [results[side][frame] for frame in ids]
        arrays = {'num_frames': np.asarray(len(ids)), 'frame_names': np.asarray(ids)}
        for key, source in mapping.items():
            present = [value.get(source) for value in values]
            if all(value is not None for value in present):
                arrays[key] = np.stack(present)
        target = path / f'{side}_results.npz'
        np.savez(target, **arrays)
        output.append({'side': side, 'path': str(target), 'sha256': digest(target), 'frames': len(ids),
                       'fields': {key: list(value.shape) for key, value in arrays.items()},
                       'finite_camera_geometry': bool(np.isfinite(arrays['joints']).all() and np.isfinite(arrays['cam_t']).all())})
    return output


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--input-manifest-sha256', required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--crop-boxes', type=Path, required=True)
    parser.add_argument('--crop-receipt', type=Path, required=True)
    parser.add_argument('--crop-receipt-sha256', required=True)
    parser.add_argument('--episode', type=int, choices=(11, 16), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu',), required=True)
    parser.add_argument('--dtype', choices=('float32',), default='float32')
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--frame-chunk-size', type=int, default=8)
    parser.add_argument('--frame-ids', help='Original IDs for an explicitly incomplete smoke, comma-separated')
    parser.add_argument('--with-smoothing', action='store_true', required=True)
    parser.add_argument('--reuse-verified-raw-cache', action='store_true')
    parser.add_argument('--raw-cache-proof', type=Path)
    parser.add_argument('--raw-cache-proof-sha256')
    return parser.parse_args()


def main(args):
    import cv2
    import numpy as np
    import torch

    if min(args.threads, args.batch_size, args.frame_chunk_size) < 1:
        raise ValueError('Runtime sizes must be positive')
    asset_root = args.asset_root.resolve()
    if asset_root == TASK or not asset_root.is_dir():
        raise ValueError('Use the previous verified asset root read-only')
    protocol_path = expected_file(args.protocol, args.protocol_sha256)
    protocol = read(protocol_path)
    if protocol.get('track') != 1 or protocol.get('all30_execution_authorized') is not False:
        raise ValueError('Protocol must explicitly authorize Track 1 pilots only')
    input_path = expected_file(args.input_manifest, args.input_manifest_sha256)
    inputs = read(input_path)
    if inputs.get('track') != 1 or inputs.get('all30_execution_authorized') is not False:
        raise ValueError('Input package must explicitly bind Track 1 pilots only')
    if {entry['episode'] for entry in inputs['episodes']} != {11, 16}:
        raise ValueError('Only the two authorized episodes may be bound')
    entry = next(item for item in inputs['episodes'] if item['episode'] == args.episode)
    count = COUNTS[args.episode]
    if entry['frames'] != count or entry['fps'] != 30:
        raise ValueError('Original video frame count/rate changed')
    video = expected_file(entry['video'], entry['video_sha256'])
    if '/track_1/' not in video.as_posix().lower():
        raise ValueError('Only verified Track 1 source video paths are accepted')
    camera_path = expected_file(entry['camera'], entry['camera_sha256'])
    camera = read(camera_path)
    if camera['model'] != 'PINHOLE':
        raise ValueError('Main controlled variant uses unchanged estimated pinhole K and affine crops')
    K = np.asarray(camera['camera_matrix']['matrix_3x3'], dtype=np.float32)
    width, height = int(camera['image_width']), int(camera['image_height'])
    crop_receipt_path = expected_file(args.crop_receipt, args.crop_receipt_sha256)
    crop_receipt = read(crop_receipt_path)
    for key, expected in [('track', 1), ('episode', args.episode), ('frames', count),
                          ('source_video_sha256', entry['video_sha256']), ('all30_execution_authorized', False)]:
        if crop_receipt.get(key) != expected:
            raise ValueError(f'Crop receipt binding differs: {key}')
    crop_path = expected_file(args.crop_boxes, crop_receipt['crop_boxes_sha256'])
    crop_boxes = np.load(crop_path, allow_pickle=False)
    if crop_boxes.shape != (count, 2, 4) or crop_boxes.dtype.kind not in 'iu':
        raise ValueError('Require full original-frame integer xyxy crop boxes, anatomical left then right')
    size = crop_boxes[..., 2:] - crop_boxes[..., :2]
    if (not np.all(size[..., 0] == size[..., 1]) or not np.all(size > 0)
            or not np.all(crop_boxes[..., :2] >= 0) or not np.all(crop_boxes[..., 2] <= width)
            or not np.all(crop_boxes[..., 3] <= height)):
        raise ValueError('Frozen refined crops must be positive in-frame squares')
    # Only a coordinate conversion. No expansion, clipping, smoothing or crop selection here.
    bboxes = {f'{i:06d}': {side: np.asarray([(box[0] + box[2]) / 2,
                (box[1] + box[3]) / 2, box[2] - box[0]], dtype=np.float32)
                for side, box in zip(('left', 'right'), crop_boxes[i])} for i in range(count)}
    selected = set(map(int, args.frame_ids.split(','))) if args.frame_ids else set(range(count))
    if not selected or min(selected) < 0 or max(selected) >= count:
        raise ValueError('Smoke selection must retain valid original frame IDs')
    cache_records, cache_lineage = {'left': {}, 'right': {}}, None
    reuse_mask = np.zeros((count, 2), dtype=bool)
    if args.reuse_verified_raw_cache:
        if args.raw_cache_proof is None or not args.raw_cache_proof_sha256:
            raise ValueError('Exact RAW reuse requires the successful source-bound cache proof')
        proof_path = expected_file(args.raw_cache_proof, args.raw_cache_proof_sha256)
        proof = read(proof_path)
        import raw_cache02
        if (proof.get('passed') is not True or proof.get('neural_forward_calls') != 0
                or proof['proof_source_sha256'] != digest(raw_cache02.__file__)):
            raise ValueError('RAW cache proof does not match the current restored-record contract')
        cache_records, old_boxes, cache_lineage = raw_cache02.load_legacy_raw(entry, K)
        reuse_mask = np.all(crop_boxes == old_boxes, axis=-1)
        cache_lineage.update(cache_contract_source_sha256=digest(raw_cache02.__file__),
                             cache_proof=str(proof_path), cache_proof_sha256=args.raw_cache_proof_sha256)
    cache_hit_ids = {side: [i for i in sorted(selected) if reuse_mask[i, s]]
                     for s, side in enumerate(('left', 'right'))}
    recomputed_ids = {side: [i for i in sorted(selected) if not reuse_mask[i, s]]
                      for s, side in enumerate(('left', 'right'))}
    output = args.output.resolve()
    episode_root = RESULTS / f'episode_{args.episode:06d}'
    if output.exists() or not output.is_relative_to(episode_root) or output == episode_root:
        raise ValueError('Use a fresh attempt folder inside the authorized new result root')
    output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable; preserve attempt and use an explicit CPU fallback attempt')
    # Limit process-local generated cache writes to the new task directory.
    import warp
    warp.config.kernel_cache_dir = str(TASK / 'cache/warp')
    recipes, asset_binding = load_pinned_recipes(asset_root)
    receipt = {'started_at': now(), 'episode': args.episode, 'source_video_sha256': entry['video_sha256'],
               'source_frames': count, 'frames': len(selected), 'complete_source_coverage': len(selected) == count,
               'selected_original_frame_ids': sorted(selected), 'same_original_camera': True,
               'camera_estimated_not_calibrated': True, 'same_crop_bounds': True,
               'crop_variant': 'refined_image_evidence_boxes_shared_with_mediapipe',
               'perspective_cropping': False, 'pixel_aspect_rectification': False, 'box_expansion': False,
               'runner_sha256': digest(__file__), 'protocol': str(protocol_path), 'protocol_sha256': args.protocol_sha256,
               'input_manifest': str(input_path), 'input_manifest_sha256': args.input_manifest_sha256,
               'input_bindings': {**entry, 'crop_boxes': str(crop_path), 'crop_boxes_sha256': digest(crop_path),
                                  'crop_receipt': str(crop_receipt_path), 'crop_receipt_sha256': args.crop_receipt_sha256},
               **asset_binding, 'device': args.device, 'dtype': args.dtype, 'threads': args.threads,
               'batch_size': args.batch_size, 'frame_chunk_size': args.frame_chunk_size,
               'tf32_enabled': False, 'status': 'running', 'all30_execution_started': False,
               'quality_accepted': False, 'contact_quality_accepted': False,
               'raw_cache_enabled': bool(args.reuse_verified_raw_cache), 'raw_cache_lineage': cache_lineage,
               'cache_hit_original_frame_ids_per_side': cache_hit_ids,
               'neural_recomputed_original_frame_ids_per_side': recomputed_ids,
               'cache_hits_per_side': {side: len(ids) for side, ids in cache_hit_ids.items()},
               'neural_recomputed_views_per_side': {side: len(ids) for side, ids in recomputed_ids.items()},
               'all_original_frames_reconstructed': len(selected) == count,
               'all_original_hand_views_neurally_recomputed': len(selected) == count and not bool(np.any(reuse_mask)),
               'official_smoothing_recomputed_over_full_merged_raw_clip': len(selected) == count}
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        receipt['cuda_free_total_before'] = list(torch.cuda.mem_get_info())
        receipt['cuda_device'] = torch.cuda.get_device_name(0)
    write(output / 'started.json', receipt)
    begin = time.perf_counter()
    model, report = recipes.load_model(args.device)
    receipt['model_load'] = report
    receipt['model_load_seconds'] = time.perf_counter() - begin
    from inference.pipeline.predictor import EgoHandPredictor
    predictor = EgoHandPredictor(model, args.batch_size, device=args.device, dtype=torch.float32,
                                 smooth=False, use_perspective_cropping=False)
    receipt['model_transfer_seconds'] = time.perf_counter() - begin - receipt['model_load_seconds']
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError('Original source video cannot be decoded')
    results = {side: {f'{i:06d}': cache_records[side][f'{i:06d}'] for i in cache_hit_ids[side]}
               for side in ('left', 'right')}
    chunk_frames, chunk_ids, chunks = [], [], []
    inference_seconds, row = 0.0, 0

    def synchronize():
        if args.device == 'cuda':
            torch.cuda.synchronize()

    def flush():
        nonlocal inference_seconds
        if not chunk_ids:
            return
        synchronize()
        start = time.perf_counter()
        changed_boxes = {i: {side: bboxes[i][side] for s, side in enumerate(('left', 'right'))
                             if not reuse_mask[int(i), s]} for i in chunk_ids}
        values = predictor.run(chunk_frames, chunk_ids, changed_boxes, ['left', 'right'],
                               intrinsics_per_image={i: K for i in chunk_ids})
        synchronize()
        elapsed = time.perf_counter() - start
        inference_seconds += elapsed
        for side in results:
            if set(results[side]) & set(values[side]):
                raise ValueError('A fresh neural result must never overwrite a RAW cache hit')
            results[side].update(values[side])
        exported = export_sparse_chunk(values, output / 'chunks' / f'chunk-{len(chunks):04d}')
        chunks.append({'original_frame_ids': list(chunk_ids), 'inference_seconds': elapsed, 'outputs': exported})
        progress = {'last_original_frame_id': int(chunk_ids[-1]),
                    'prediction_frames_per_side': {side: len(values) for side, values in results.items()},
                    'inference_seconds': inference_seconds, 'wall_seconds': time.perf_counter() - begin,
                    'last_chunk_seconds': elapsed, 'last_chunk_frames': len(chunk_ids), 'chunks': chunks}
        write(output / 'progress.json', progress, exclusive=False)
        print(json.dumps({k: value for k, value in progress.items() if k != 'chunks'}), flush=True)
        chunk_frames.clear()
        chunk_ids.clear()

    try:
        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise ValueError('Decoded source dimensions differ from original camera')
                if row in selected and not reuse_mask[row].all():
                    chunk_frames.append(frame)
                    chunk_ids.append(f'{row:06d}')
                    if len(chunk_ids) >= args.frame_chunk_size:
                        flush()
                row += 1
            flush()
            if row != count or any(set(values) != {f'{i:06d}' for i in selected} for values in results.values()):
                raise ValueError('Original source coverage/order differs')
            actual_recomputed = {side: sorted(int(name) for name in results[side] if int(name) not in set(cache_hit_ids[side]))
                                 for side in ('left', 'right')}
            if actual_recomputed != recomputed_ids:
                raise ValueError('Neural result coverage differs from the exact unmatched crop views')
            receipt['raw_outputs'] = recipes.export(results, output / 'raw')
            receipt['inference_seconds'] = inference_seconds
            receipt['decoded_source_frames'] = row
            receipt['chunks'] = chunks
            for record in receipt['raw_outputs']:
                record.update(variant='raw', complete_source_coverage=len(selected) == count)
            receipt['wrist_relative_joint0_max_norm_metres'] = {
                side: max(float(np.linalg.norm(value['pred_keypoints_3d'][0])) for value in hand.values())
                for side, hand in results.items()}
            if max(receipt['wrist_relative_joint0_max_norm_metres'].values()) >= 1e-4:
                raise ValueError('Unexpected camera-space joints: cam_t must not be added twice')
            smoothed = copy.deepcopy(results)
            ids = [f'{i:06d}' for i in sorted(selected)]
            predictor.smooth = True
            synchronize()
            start = time.perf_counter()
            predictor.postprocess_results(smoothed, ids, {i: bboxes[i] for i in ids},
                frame_times={i: int(i) / entry['fps'] for i in ids}, intrinsics_per_image={i: K for i in ids})
            synchronize()
            receipt['smoothing_seconds'] = time.perf_counter() - start
            receipt['smoothed_outputs'] = recipes.export(smoothed, output / 'smoothed')
            for record in receipt['smoothed_outputs']:
                record.update(variant='upstream_adaptive_smoothed', complete_source_coverage=len(selected) == count)
    finally:
        cap.release()
    if args.device == 'cuda':
        receipt['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        receipt['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        receipt['cuda_free_total_after'] = list(torch.cuda.mem_get_info())
    import psutil
    memory = psutil.Process().memory_info()
    receipt['process_memory'] = {key: getattr(memory, key) for key in ('rss', 'vms', 'peak_wset') if hasattr(memory, key)}
    receipt.update(status='completed', finished_at=now(), wall_seconds=time.perf_counter() - begin)
    write(output / 'prediction-receipt.json', receipt)
    print(json.dumps({'completed': True, 'episode': args.episode, 'wall_seconds': receipt['wall_seconds']}), flush=True)


if __name__ == '__main__':
    arguments_value = arguments()
    try:
        main(arguments_value)
    except BaseException as error:
        failure = {'recorded_at': now(), 'episode': arguments_value.episode, 'device': arguments_value.device,
                   'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
                   'runner_sha256': digest(__file__), 'all30_execution_started': False,
                   'retry_policy': 'Preserve this attempt; an explicit fresh CPU/GPU attempt is required'}
        # Never mutate an existing or invalid output attempt, even on preflight failure.
        failure_root = TASK / 'runtime-failures'
        failure_root.mkdir(exist_ok=True)
        marker = failure_root / f'failure-{os.getpid()}-{time.time_ns()}.json'
        write(marker, failure)
        print(json.dumps(failure), file=sys.stderr, flush=True)
        raise
