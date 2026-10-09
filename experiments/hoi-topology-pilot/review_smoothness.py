"""Review saved reconstructions; no inference or optimization is performed."""
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/real'
NAMES = ['matched_independent', 'soft_joint', 'attachment_collision']
LABELS = ['Independent (matched control)', 'Surface contact', 'Fixed contact structure']
COLORS = ['#4c78a8', '#f2a541', '#d34b59']
results = {name: dict(np.load(OUT / f'{name}.npz')) for name in NAMES}
meta = json.loads((OUT / 'metrics.json').read_text())
contacts = np.array(meta['fixed_contact_vertex_ids'])
series = {}
summary = {}
for name, r in results.items():
    center, rotations, vertices = r['object_translation'], r['object_rotation'], r['vertices']
    hand = vertices[:, contacts]
    relative = np.einsum('tpi,tij->tpj', hand - center[:, None], rotations)
    angular_steps = Rotation.from_matrix(rotations[1:] @ rotations[:-1].transpose(0, 2, 1)).as_rotvec()
    series[name] = [
        np.linalg.norm(np.diff(center, n=2, axis=0), axis=-1) * 1000,
        np.linalg.norm(np.diff(angular_steps, axis=0), axis=-1) * 180 / np.pi,
        np.sqrt(np.mean(np.sum(np.diff(hand, n=2, axis=0) ** 2, axis=-1), axis=1)) * 1000,
        np.sqrt(np.mean(np.sum(np.diff(relative, axis=0) ** 2, axis=-1), axis=1)) * 1000,
    ]
    keys = ['object_translation_second_difference_mm', 'object_rotation_step_change_deg', 'selected_hand_vertices_second_difference_rms_mm', 'hand_motion_in_object_frame_step_rms_mm']
    summary[name] = {key: {'mean': float(values.mean()), 'p95': float(np.percentile(values, 95)), 'max': float(values.max())} for key, values in zip(keys, series[name])}
report = {'scope': 'Prediction-only temporal diagnostics. No reference motion or perceptual smoothness validation.', 'frames': [90, 209], 'fps': 30, 'hand_vertices': contacts.tolist(), 'definitions': {'second_difference': 'x[t+1] - 2*x[t] + x[t-1]; not divided by time squared', 'rotation_step_change': 'difference of consecutive world-frame rotation increments expressed as rotation vectors; not reference acceleration', 'hand_vertices': 'eight hypothesized contact material vertices, not all hand/body deformation', 'relative_step': 'one-frame change of hand material vertices in the moving object coordinate frame; physical sliding is not necessarily an error', 'interpretation': 'Smaller changes may reflect smoother motion, loss of real motion, or stronger smoothing. Image consistency and motion fidelity remain necessary.'}, 'metrics': summary}
(OUT / 'smoothness-review.json').write_text(json.dumps(report, indent=2) + '\n')
titles = ['Object translation: change of frame step', 'Object rotation: change of rotation step', 'Selected hand vertices: change of frame step', 'Hand material motion in object coordinates']
units = ['mm / frame^2', 'degrees / frame^2', 'mm / frame^2 (RMS)', 'mm / frame (RMS)']
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
for j, ax in enumerate(axes):
    for name, label, color in zip(NAMES, LABELS, COLORS):
        values = series[name][j]
        frame_ids = np.arange(91, 209) if j < 3 else np.arange(91, 210)
        ax.plot(frame_ids / 30, values, label=label, color=color, linewidth=1.7)
    ax.set_title(titles[j], fontsize=11, loc='left')
    ax.set_ylabel(units[j], fontsize=9)
    ax.grid(alpha=.25)
axes[0].legend(fontsize=9)
axes[-1].set_xlabel('Original video time (seconds)')
fig.suptitle('Temporal changes in saved predictions — smaller does not prove faithful reconstruction', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .965))
fig.savefig(OUT / 'smoothness-curves.png', dpi=150)
plt.close(fig)

# Fixed camera and scale, synchronized frames; no recentering on each frame.
chosen = [NAMES[0], NAMES[2]]
faces = np.load(ROOT / 'baseline/faces.npy')
reference = results[chosen[0]]['object_translation'][60]
ids = np.flatnonzero(np.linalg.norm(results[chosen[0]]['vertices'][60] - reference, axis=1) < .30)
keep = faces[np.all(np.isin(faces, ids), axis=1)]
edges = np.unique(np.sort(np.concatenate([keep[:, [0, 1]], keep[:, [1, 2]], keep[:, [2, 0]]]), axis=1), axis=0)
view = Rotation.from_euler('yx', [35, -10], degrees=True).as_matrix()
corners = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
cube_edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.count_nonzero(corners[i] != corners[j]) == 1]
world_points = np.concatenate([np.concatenate([r['vertices'][:, ids].reshape(-1, 3), (corners * r['object_half_extents'] @ r['object_rotation'].transpose(0, 2, 1) + r['object_translation'][:, None]).reshape(-1, 3)]) for r in [results[n] for n in chosen]])
xy = ((world_points - reference) @ view.T)[:, :2]
lo, hi = xy.min(axis=0), xy.max(axis=0)
mid = (lo + hi) / 2
scale = min(560 / (hi[0] - lo[0]), 320 / (hi[1] - lo[1]))
K = np.load(ROOT / 'baseline/intrinsics.npy')
def image_project(points):
    points = points @ K.T
    return np.rint(points[:, :2] / points[:, 2:]).astype(np.int32)
def novel_project(points):
    xy = ((points - reference) @ view.T)[:, :2]
    return np.rint((xy - mid) * scale + [320, 660]).astype(np.int32)
def text(panel, message, position, size=.65):
    cv2.putText(panel, message, position, cv2.FONT_HERSHEY_SIMPLEX, size, (30, 30, 30), 1, cv2.LINE_AA)
writers = [cv2.VideoWriter(str(OUT / filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 880)) for filename, fps in [('smoothness-closeup-1x.mp4', 30), ('smoothness-closeup-half-speed.mp4', 15)]]
assert all(writer.isOpened() for writer in writers)
for t in range(120):
    source = cv2.imread(str(ROOT / 'data/frames' / f'{t + 90:06d}.jpg'))
    panels = []
    for name, label in zip(chosen, [LABELS[0], LABELS[2]]):
        r = results[name]
        vertices, rotation, center = r['vertices'][t], r['object_rotation'][t], r['object_translation'][t]
        cube = corners * r['object_half_extents'] @ rotation.T + center
        panel = np.full((880, 640, 3), 248, np.uint8)
        original = source.copy()
        uv = image_project(vertices)
        cuv = image_project(cube)
        for a, b in edges[::2]:
            cv2.line(original, tuple(uv[a]), tuple(uv[b]), (230, 155, 45), 1, cv2.LINE_AA)
        for a, b in cube_edges:
            cv2.line(original, tuple(cuv[a]), tuple(cuv[b]), (35, 220, 60), 3, cv2.LINE_AA)
        panel[55:435] = cv2.resize(original[470:750, 560:900], (640, 380))
        text(panel, label, (12, 25), .7)
        text(panel, f'frame {t + 90} | source time {(t + 90)/30:.2f}s', (12, 48), .55)
        text(panel, 'Same original camera crop', (12, 458), .6)
        text(panel, '3D close-up: fixed camera and common scale', (12, 490), .6)
        puv, cuv = novel_project(vertices), novel_project(cube)
        for a, b in edges[::2]:
            if np.max(np.abs(np.r_[puv[a], puv[b]])) < 3000:
                cv2.line(panel, tuple(puv[a]), tuple(puv[b]), (230, 155, 45), 1, cv2.LINE_AA)
        for a, b in cube_edges:
            cv2.line(panel, tuple(cuv[a]), tuple(cuv[b]), (35, 150, 45), 2, cv2.LINE_AA)
        for point in novel_project(vertices[contacts]):
            cv2.circle(panel, tuple(point), 4, (90, 70, 220), -1)
        text(panel, 'Blue: hand/arm mesh | Green: fitted block', (12, 851), .55)
        text(panel, 'Red dots: assumed material contact vertices', (12, 873), .55)
        panels.append(panel)
    frame = np.concatenate(panels, axis=1)
    for writer in writers:
        writer.write(frame)
    if t == 60:
        cv2.imwrite(str(OUT / 'smoothness-closeup-frame150.jpg'), frame)
for writer in writers:
    writer.release()
for filename, expected_fps in [('smoothness-closeup-1x.mp4', 30), ('smoothness-closeup-half-speed.mp4', 15)]:
    capture = cv2.VideoCapture(str(OUT / filename))
    assert abs(capture.get(cv2.CAP_PROP_FPS) - expected_fps) < 1e-6
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    assert count == 120, count
print(json.dumps(report, indent=2))
