"""Present the completed interaction pilot at a shared fixed view and scale."""
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from real_trial import ROOT, project_np
from synthetic_trial import CORNERS, EDGES

OUT = ROOT / ('results/interaction-v3' if '--texture' in sys.argv else 'results/interaction-v2')
old = dict(np.load(ROOT / 'results/real/attachment_collision.npz'))
new = dict(np.load(OUT / 'interaction_mesh.npz'))
initial = dict(np.load(ROOT / 'results/real/matched_independent.npz'))
geometry = dict(np.load(OUT / 'interaction-structure.npz'))
faces = np.load(ROOT / 'baseline/faces.npy')
k = np.load(ROOT / 'baseline/intrinsics.npy')
reference = initial['object_translation'][60]
ids = np.flatnonzero(np.linalg.norm(initial['vertices'][60] - reference, axis=-1) < .30)
triangles = faces[np.all(np.isin(faces, ids), axis=1)]
edges = np.unique(np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0)[::2]
view = Rotation.from_euler('yx', [35, -10], degrees=True).as_matrix()
points = np.concatenate([np.concatenate([r['vertices'][:, ids].reshape(-1, 3), (CORNERS * r['object_half_extents'] @ r['object_rotation'].transpose(0, 2, 1) + r['object_translation'][:, None]).reshape(-1, 3)]) for r in [old, new]])
xy = ((points - reference) @ view.T)[:, :2]
lo, hi = xy.min(0), xy.max(0)
mid = (lo + hi) / 2
scale = min(560 / (hi[0] - lo[0]), 320 / (hi[1] - lo[1]))
def novel(p):
    return np.rint((((p - reference) @ view.T)[:, :2] - mid) * scale + [320, 660]).astype(np.int32)
def label(image, value, pos, size=.6):
    cv2.putText(image, value, pos, cv2.FONT_HERSHEY_SIMPLEX, size, (35, 35, 35), 1, cv2.LINE_AA)
writers = [cv2.VideoWriter(str(OUT / name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 880)) for name, fps in [('comparison-1x.mp4', 30), ('comparison-half-speed.mp4', 15)]]
assert all(w.isOpened() for w in writers)
for t in range(120):
    source = cv2.imread(str(ROOT / 'data/frames' / f'{t + 90:06d}.jpg'))
    panels = []
    for r, title in [(old, 'Previous: fixed contact'), (new, 'New: interaction mesh')]:
        v, rr, c, half = r['vertices'][t], r['object_rotation'][t], r['object_translation'][t], r['object_half_extents']
        cube = CORNERS * half @ rr.T + c
        uv = np.rint(project_np(v, k)).astype(int)
        cuv = np.rint(project_np(cube, k)).astype(int)
        overlay = source.copy()
        for a, b in edges:
            cv2.line(overlay, tuple(uv[a]), tuple(uv[b]), (230, 155, 45), 1, cv2.LINE_AA)
        for a, b in EDGES:
            cv2.line(overlay, tuple(cuv[a]), tuple(cuv[b]), (35, 220, 60), 3, cv2.LINE_AA)
        cv2.circle(overlay, tuple(cuv[0]), 5, (20, 80, 255), -1)
        panel = np.full((880, 640, 3), 248, np.uint8)
        panel[55:435] = cv2.resize(overlay[470:750, 560:900], (640, 380))
        label(panel, title, (12, 25), .73)
        label(panel, f'frame {t+90} | original time {(t+90)/30:.2f}s', (12, 48), .55)
        label(panel, 'Original camera; same crop', (12, 460))
        label(panel, '3D view; fixed camera and common scale', (12, 490))
        puv, ncv = novel(v), novel(cube)
        for a, b in edges:
            if np.max(np.abs(np.r_[puv[a], puv[b]])) < 3000:
                cv2.line(panel, tuple(puv[a]), tuple(puv[b]), (230, 155, 45), 1, cv2.LINE_AA)
        for a, b in EDGES:
            cv2.line(panel, tuple(ncv[a]), tuple(ncv[b]), (35, 150, 45), 2, cv2.LINE_AA)
        cv2.circle(panel, tuple(ncv[0]), 5, (20, 80, 255), -1)
        axes = np.r_[np.zeros((1, 3)), np.eye(3) * .065] @ rr.T + c
        ap = novel(axes)
        for j, color in enumerate([(20, 20, 220), (40, 150, 40), (220, 80, 30)]):
            cv2.arrowedLine(panel, tuple(ap[0]), tuple(ap[j+1]), color, 2, cv2.LINE_AA, tipLength=.2)
        label(panel, 'Blue: hand mesh | Green: estimated block', (12, 850), .55)
        label(panel, 'Orange dot and axes reveal block rotation', (12, 873), .55)
        panels.append(panel)
    frame = np.concatenate(panels, axis=1)
    for writer in writers:
        writer.write(frame)
    if t in [0, 60, 119]:
        cv2.imwrite(str(OUT / f'comparison-frame{t+90}.jpg'), frame)
for writer in writers:
    writer.release()

# Visualize the actual shared graph used by the optimizer.
points = geometry['reference_vertices'][60]
lap = geometry['laplacian'][60]
nh = len(geometry['hand_vertex_ids'])
pairs = np.argwhere(np.triu((np.abs(lap) > 1e-8) | (np.abs(lap.T) > 1e-8), 1))
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection='3d')
for a, b in pairs:
    cross = (a < nh) != (b < nh)
    ax.plot(*points[[a,b]].T, color='#bb477b' if cross else '#b8bbc2', alpha=.65 if cross else .2, linewidth=.8 if cross else .4)
ax.scatter(*points[:nh].T, s=14, color='#3388ba', label='Hand material samples')
ax.scatter(*points[nh:].T, s=10, color='#d99c27', label='Object surface samples')
ax.set(xlabel='Object x (m)', ylabel='Object y (m)', zlabel='Object z (m)', title='Shared 3D interaction graph at frame 150\nMagenta edges join hand and object neighborhoods')
ax.set_box_aspect(np.ptp(points, axis=0))
ax.legend(loc='upper right', fontsize=8)
fig.text(.5, .025, 'Neighborhood edges are not physical welds. Geometry and topology come from estimates.', ha='center', fontsize=9)
fig.savefig(OUT / 'interaction-structure.png', dpi=150)
plt.close(fig)
checks = {}
for name, expected_fps in [('comparison-1x.mp4', 30), ('comparison-half-speed.mp4', 15)]:
    path = OUT / name
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert count == 120 and abs(fps - expected_fps) < 1e-6
    checks[name] = {'decoded_frames': count, 'fps': fps, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
for key in ['object_rotation', 'object_translation', 'vertices', 'mhr_model_params']:
    assert np.isfinite(new[key]).all()
assert np.array_equal(new['original_frame_ids'], np.arange(90, 210))
assert np.allclose(new['object_rotation'] @ new['object_rotation'].transpose(0,2,1), np.eye(3), atol=1e-5)
assert np.allclose(np.linalg.det(new['object_rotation']), 1, atol=1e-5)
(OUT / 'verification.json').write_text(json.dumps({'passed': True, 'scope': 'Output integrity and video checks; not accuracy proof', 'videos': checks}, indent=2))
print(json.dumps(checks, indent=2))
