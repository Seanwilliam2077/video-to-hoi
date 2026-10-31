"""Compact completed native/video evidence without declaring quality acceptance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
RESULTS = Path('J:/video-to-hoi/runs/egohand-refined-pilots-20261031')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def binding(path):
    with path.open('rb') as stream:
        h = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h}


def compact_screens(screen):
    out = {}
    for scope in ('all_adjacent_original_frames', 'original_legacy_support_boundaries',
                  'support_boundaries_by_variant', 'holdout_touching_reference_edges', 'fast_reference_motion'):
        out[scope] = screen[scope]
    out['manual_source_inspection_windows'] = screen['manual_source_inspection_windows']
    return out


def stats(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {'count': 0, 'min': None, 'p05': None, 'median': None, 'p95': None, 'max': None}
    assert np.isfinite(values).all()
    return {'count': len(values), 'min': float(values.min()), 'p05': float(np.percentile(values, 5)),
            'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)), 'max': float(values.max())}


def main():
    path = TASK / 'completed-summary02.json'
    if path.exists():
        raise ValueError('Preserve existing summary')
    result = {'schema_version': 'refined_egohand_completed_summary.v1',
              'recorded_at': datetime.now(timezone.utc).isoformat(), 'track': 1,
              'authorized_episodes': [16, 11], 'all30_execution_authorized': False,
              'quality_accepted': False, 'official_score_available': False,
              'automatic_promotion': False,
              'interpretation': 'Image references are shared/method-dependent, self-motion and opening are prediction diagnostics, and hidden articulation/contact truth is unavailable.',
              'protocol': binding(TASK / 'protocol02.json'), 'episodes': []}
    for ep, count in ((16, 360), (11, 877)):
        root = RESULTS / f'episode_{ep:06d}'
        evaluation_path = root / 'evaluation02.json'
        data = read(evaluation_path)
        render = read(root / 'review02/summary.json')
        assert data['frames'] == render['frames'] == count
        assert data['original_frame_coverage_complete']
        assert all(v['passed'] for v in data['native_decoder'].values())
        opening = {}
        for role, kinds in data['native_opening_proxy_curves'].items():
            opening[role] = {}
            for kind, value in kinds.items():
                opening[role][kind] = {}
                for side, values in value['sides'].items():
                    opening[role][kind][side] = {
                        'admitted_frames': values['admitted_frames'],
                        'normalization_valid_frames': values['normalization_valid_frames'],
                        'tiny_palm_denominator_frames': values['tiny_palm_denominator_frames'],
                        **{key: stats([row[key] for row in values['rows']]) for key in (
                            'palm_width', 'thumb_index_distance', 'thumb_middle_distance',
                            'thumb_index_over_palm', 'thumb_middle_over_palm')}}
        row = {key: data[key] for key in ('episode', 'frames', 'source_video_sha256', 'mhr_model_sha256',
                'role_labels', 'controlled_pair', 'frozen_conditions', 'native_decoder',
                'parameter_preservation', 'retained_native_updates', 'admission_and_optimization_counts',
                'image_reference_distances', 'prediction_self_motion', 'support_boundaries')}
        row.update(full_evaluation=binding(evaluation_path),
                   video_receipt=binding(root / 'review02/summary.json'), videos=render['videos'],
                   inference_receipt=binding(root / 'inference-attempt02/prediction-receipt.json'),
                   retarget_report=read(root / 'observations-smoothed02/report.json'),
                   image_reference_motion_screens={role: compact_screens(value) for role, value in data['image_reference_motion_screens'].items()},
                   opening_proxy_distributions=opening)
        result['episodes'].append(row)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(path), 'source_frames': 1237, 'episodes': 2}), flush=True)


if __name__ == '__main__':
    main()
