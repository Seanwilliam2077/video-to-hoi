"""Create a compact, source-bound summary only after both controlled pilots finish."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


TASK = Path(__file__).resolve().parent
RESULTS = Path('J:/video-to-hoi/runs/egohand-pilots-20261031')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def binding(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def guard_summary(value):
    def counts(item, wrist=False):
        visible = 'observed_wrist_failed' if wrist else 'observed_motion_failed'
        generated = visible.replace('observed', 'target')
        unknown = 'unknown_wrist_budget_failed' if wrist else 'unknown_motion_budget_failed'
        return {'passes': item['all_engineering_guards_passed'],
                'observed_or_generated_velocity_failures': item.get(visible, item.get(generated)),
                'unknown_correction_budget_failures': item[unknown]}
    return {'edges': value['edge_count'],
            'finger': {name: counts(item) for name, item in value['variants'].items()},
            'wrist': {name: counts(item, True) for name, item in value['absolute_wrist_motion']['variants'].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous summaries; output must be new')
    result = {'schema_version': 'egohand_completed_pilot_summary.v1',
              'recorded_at': datetime.now(timezone.utc).isoformat(), 'track': 1,
              'authorized_episodes': [16, 11], 'all30_execution_authorized': False,
              'automatic_promotion': False, 'official_scores_available': False,
              'heldout_interpretation': 'Within-video development validation; upstream smoothing sees the full clip.',
              'jitter_interpretation': 'Pooled 20-finger wrist-relative native-coordinate second-difference norms; prediction-only, not reference acceleration error.',
              'episodes': []}
    for ep, count in ((16, 360), (11, 877)):
        root = RESULTS / f'episode_{ep:06d}'
        inference_path = root / 'egohand-raw-attempt01/prediction-receipt.json'
        inference = read(inference_path)
        assert inference['frames'] == count and inference['complete_source_coverage']
        row = {'episode': ep, 'frames': count, 'source_video_sha256': inference['source_video_sha256'],
               'inference': binding(inference_path), 'variants': {}}
        review_path = root / 'review-raw01/summary.json'
        read(review_path)  # A renderer writes this only after all videos decode.
        row['review_receipt'] = binding(review_path)
        row['grasp_diagnostics'] = binding(root / 'grasp-motion01/summary.json')
        for variant in ('raw', 'smoothed'):
            path = root / f'evaluation-{variant}01.json'
            data = read(path)
            assert data['frames'] == count and data['original_frame_coverage_complete']
            assert all(item['passed'] for item in data['native_decoder'].values())
            frozen = data['frozen_conditions']
            assert frozen['same_camera'] and frozen['same_source_crop_bounds'] and frozen['same_fit_config']
            row['variants'][variant] = {
                'evaluation': binding(path), 'native_decoder': data['native_decoder'],
                'parameter_preservation': data['parameter_preservation'],
                'prediction_self_motion': data['prediction_self_motion'],
                'image_agreement_with_mediapipe': data['image_agreement_by_target_source']['mediapipe'],
                'motion_screens': {target: {scope: guard_summary(screens[scope]) for scope in
                    ('all_adjacent_original_frames', 'original_mediapipe_boundaries', 'current_mediapipe_boundaries', 'egohand_boundaries')}
                    for target, screens in data['motion_screens_by_target_source'].items()}}
        result['episodes'].append(row)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'episodes': 2, 'frames': 1237}))


if __name__ == '__main__':
    main()
