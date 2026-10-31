"""Audit completed controlled EgoHand outputs; this does not assess pose accuracy."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, choices=(11, 16), required=True)
    args = parser.parse_args()
    result = Path('J:/video-to-hoi/runs/egohand-pilots-20261031') / f'episode_{args.episode:06d}' / 'egohand-raw-attempt01'
    receipt = json.loads((result / 'prediction-receipt.json').read_text(encoding='utf-8'))
    expected = {11: 877, 16: 360}[args.episode]
    assert receipt['status'] == 'completed' and receipt['complete_source_coverage'] is True
    assert receipt['frames'] == expected and receipt['source_frames'] == expected
    assert receipt['selected_original_frame_ids'] == list(range(expected))
    bindings = receipt['input_bindings']
    assert bindings['track'] == 1 and bindings['episode'] == args.episode
    assert sha256(bindings['camera']) == bindings['camera_sha256']
    camera = json.loads(Path(bindings['camera']).read_text(encoding='utf-8'))
    K = np.asarray(camera['camera_matrix']['matrix_3x3'], dtype=np.float64)
    boxes = {}
    for entry in bindings['saved_boxes']:
        assert sha256(entry['path']) == entry['sha256']
        boxes[entry['side']] = json.loads(Path(entry['path']).read_text(encoding='utf-8'))
    checks = []
    records = receipt['raw_outputs'] + receipt['smoothed_outputs']
    assert len(records) == 4
    for record in records:
        assert sha256(record['path']) == record['sha256']
        assert record['frames'] == expected and record['complete_source_coverage'] is True
        with np.load(record['path'], allow_pickle=False) as values:
            ids = values['frame_names'].tolist()
            assert ids == [f'{i:06d}' for i in range(expected)]
            assert int(values['num_frames']) == expected
            assert values['joints'].shape == (expected, 25, 3)
            assert values['vertices'].shape == (expected, 2859, 3)
            assert values['cam_t'].shape == (expected, 3)
            assert all(np.isfinite(values[key]).all() for key in values.files if values[key].dtype.kind in 'fci')
            relative = values['joints'].astype(np.float64)
            translated = relative + values['cam_t'][:, None, :]
            assert (translated[:, :, 2] > 0).all()
            wrist_zero = float(np.linalg.norm(relative[:, 0], axis=-1).max())
            assert wrist_zero < 1e-6
            projected = translated @ K.T
            projected = projected[:, :, :2] / projected[:, :, 2:3]
            b = np.asarray([boxes[record['side']][name] for name in ids], dtype=np.float64)
            # Official keypoints use crop-centred normalized coordinates (not 0..1).
            pixel_prediction = values['pred_keypoints_2d'].astype(np.float64) * b[:, None, 2:3] + b[:, None, :2]
            error = float(np.linalg.norm(projected - pixel_prediction, axis=-1).max())
            assert error < 1e-3, error
        checks.append({'variant': record['variant'], 'side': record['side'], 'frame_count': expected,
                       'sha256': record['sha256'], 'geometry_finite': True, 'positive_depth': True,
                       'projection_roundtrip_max_pixels': error, 'wrist_relative_zero_max': wrist_zero})
    report = {'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'episode': args.episode,
              'source_video_sha256': receipt['source_video_sha256'], 'passed': True,
              'all_original_frames_per_hand': expected, 'checks': checks,
              'native_fullbody_MHR_output': False, 'accuracy_or_contact_accepted': False,
              'all30_execution_started': False, 'audit_script_sha256': sha256(__file__)}
    output = ROOT / f'ep{args.episode}-output-audit.json'
    assert not output.exists(), 'Preserve existing audit receipt; choose a new attempt before rerunning.'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
