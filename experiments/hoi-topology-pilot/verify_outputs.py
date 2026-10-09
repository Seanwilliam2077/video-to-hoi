"""Validate completed pilot artifacts without rerunning optimization."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parent
real = json.loads((root / 'results/real/metrics.json').read_text())
synthetic = json.loads((root / 'results/synthetic/metrics.json').read_text())
assert real['original_frame_ids'] == list(range(90, 210))
assert len(real['metrics']) == 4
assert not real['continuous_collision_guarantee']
assert not real['reference_relative_ACC_available']
for name, metrics in real['metrics'].items():
    assert len(metrics['per_frame_max_penetration_mm']) == 120, name
    for key, value in metrics.items():
        assert np.isfinite(value).all(), (name, key)
    if 'max_native_decode_consistency_error_mm' in metrics:
        assert metrics['max_native_decode_consistency_error_mm'] < 0.01, name
assert len(synthetic['tests']) == 4
assert synthetic['metrics']['Shared structure']['trajectory_rmse_mm'] < synthetic['metrics']['Independent']['trajectory_rmse_mm']
assert synthetic['metrics']['Shared structure']['max_penetration_mm'] < 1e-6
assert synthetic['wrong_fixed_binding_slip_control_rmse_mm'] > synthetic['metrics']['Shared structure']['trajectory_rmse_mm']
videos = {}
for relative in ['results/real/comparison.mp4', 'results/real/interaction_3d.mp4', 'results/synthetic/comparison.mp4']:
    path = root / relative
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    dimensions = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if relative == 'results/real/comparison.mp4' and count in (0, 119):
            cv2.imwrite(str(root / f'results/real/comparison-frame{count + 90}.jpg'), frame)
        count += 1
    capture.release()
    assert count == 120, (relative, count)
    assert abs(fps - 30) < 1e-6, (relative, fps)
    videos[relative] = {'decoded_frames': count, 'fps': fps, 'dimensions': dimensions, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
report = {'scope': 'Artifact consistency checks only, not reconstruction accuracy validation', 'passed': True, 'source_frames': [90, 209], 'method_count': 4, 'synthetic_checks': synthetic['tests'], 'videos': videos}
(root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
