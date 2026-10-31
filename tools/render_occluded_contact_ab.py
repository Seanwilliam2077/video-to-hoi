"""Render source-bound old/new EP16 humans and a transparent-object inspection.

The source comparison uses each result's explicitly bound object hypothesis;
only the two new A/B columns share the new object. The second display uses a
fixed synthetic object-local camera, never a recovered or reference view.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from v2hoi.contact_review import rasterize_meshes

SAMPLES = [0, 40, 60, 70, 75, 90, 110, 120, 180, 210, 240, 250, 251, 260, 280, 285, 287, 290, 308, 359]
WINDOWS = [(40, 110), (230, 300)]
_WORKER = None


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def bound(item):
    path = Path(item['path'])
    if digest(path) != item['sha256']:
        raise ValueError('Changed display input: ' + str(path))
    return path


def selected_hand_faces(faces, indices, vertex_count=18439):
    """Retain actual triangles only; selection clips the skin at each wrist."""
    faces = np.asarray(faces)
    selected = np.asarray(indices)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError('Expected native integer triangles')
    if selected.ndim != 1 or selected.dtype.kind not in 'iu' or not len(selected):
        raise ValueError('Expected explicit nonempty native skin vertex indices')
    if np.any(selected < 0) or np.any(selected >= vertex_count) or np.any(faces < 0) or np.any(faces >= vertex_count):
        raise ValueError('Native vertex index outside topology')
    admitted = np.zeros(vertex_count, dtype=bool)
    admitted[selected] = True
    return faces[admitted[faces].all(axis=1)]


def validate_columns(plan):
    columns = plan['columns']
    if [row['role'] for row in columns] != ['old_a', 'a', 'b']:
        raise ValueError('Show preserved old A, new A and new B in fixed order')
    if columns[1]['object'] != columns[2]['object']:
        raise ValueError('The new controlled A/B must display the exact same object')
    if columns[0]['native'] == columns[1]['native'] or columns[1]['native'] == columns[2]['native']:
        raise ValueError('Each displayed native output needs an explicit separate identity')
    return columns


def validate_evaluated_objects(columns, evaluated_inputs):
    """Reject a self-consistent display plan that silently changes geometry."""
    for column in columns:
        expected = evaluated_inputs['old_object' if column['role'] == 'old_a' else 'object']
        actual = column['object']
        if (actual['sha256'] != expected['sha256'] or
                Path(actual['path']).resolve() != Path(expected['path']).resolve()):
            raise ValueError('Displayed object differs from independently evaluated object')


def initialize_renderer(columns, camera, roi, size, skin_path):
    global _WORKER
    import cv2
    cv2.setNumThreads(1)
    with np.load(skin_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if {'left', 'right'}.issubset(keys):
            side_indices = [archive['left'], archive['right']]
            indices = np.concatenate(side_indices)
        else:
            raise ValueError('Native skin index file must declare anatomical left/right')
    variants = []
    for row in columns:
        directory = Path(row['native']['path']).parent / 'review_arrays'
        vertices = np.load(directory / 'vertices_m.npy', mmap_mode='r', allow_pickle=False)
        faces = np.load(directory / 'faces.npy', allow_pickle=False)
        with np.load(row['object']['path'], allow_pickle=False) as archive:
            obj = {key: archive[key] for key in ('vertices', 'faces', 'object_to_camera')}
        posed = np.einsum('vi,tji->tvj', obj['vertices'], obj['object_to_camera'][:, :3, :3]) + obj['object_to_camera'][:, None, :3, 3]
        variants.append((vertices, faces, selected_hand_faces(faces, indices), obj, posed))
    angle = np.deg2rad(60.)
    inspect_rotation = np.asarray([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    inspect_camera = np.asarray([[size * 1.56, 0, (size - 1) / 2], [0, size * 1.56, (size - 1) / 2], [0, 0, 1.]])
    _WORKER = variants, camera, roi, size, inspect_rotation, inspect_camera, side_indices


def render_frame(job):
    import cv2
    frame, source = job
    variants, camera, roi, size, inspect_rotation, inspect_camera, side_indices = _WORKER
    original, diagnostic = [], []
    started = time.monotonic()
    for vertices, faces, hand_faces, obj, posed in variants:
        human = np.asarray(vertices[frame])
        colors, _, identity = rasterize_meshes([
            (human, faces, [230, 195, 60], 1),
            (posed[frame], obj['faces'], [70, 170, 245], 2)], camera, roi, size)
        display = source.copy()
        visible = identity > 0
        display[visible] = np.clip(.70 * colors[visible] + .30 * source[visible], 0, 255).astype(np.uint8)
        original.append(display)
        transform = obj['object_to_camera'][frame]
        local_hand = (human - transform[:3, 3]) @ transform[:3, :3]
        local_hand = local_hand @ inspect_rotation.T + [0., 0., .90]
        local_object = obj['vertices'] @ inspect_rotation.T + [0., 0., .90]
        inspection_roi = np.asarray([0., 0., float(size), float(size)])
        object_color, _, object_id = rasterize_meshes([
            (local_object, obj['faces'], [70, 170, 245], 2)], inspect_camera, inspection_roi, size)
        hand_color, _, hand_id = rasterize_meshes([
            (local_hand, hand_faces, [230, 195, 60], 1)], inspect_camera, inspection_roi, size)
        transparent = np.full((size, size, 3), 32, dtype=np.uint8)
        object_pixels = object_id > 0
        transparent[object_pixels] = np.clip(.24 * object_color[object_pixels] + .76 * transparent[object_pixels], 0, 255)
        hand_pixels = hand_id > 0
        transparent[hand_pixels] = np.clip(.90 * hand_color[hand_pixels] + .10 * transparent[hand_pixels], 0, 255)
        for side, label in enumerate(('L', 'R')):
            center = local_hand[side_indices[side]].mean(axis=0) @ inspect_camera.T
            if center[2] > 0:
                x, y = np.rint(center[:2] / center[2]).astype(int)
                if 0 <= x < size and 20 <= y < size:
                    cv2.putText(transparent, label, (max(3, x - 5), y - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1, cv2.LINE_AA)
        diagnostic.append(transparent)
    return original, diagnostic, time.monotonic() - started, os.getpid()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=384)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.output.exists() or args.size < 256 or args.size % 2 or not 1 <= args.workers <= 8:
        raise ValueError('Require fresh output, even panel size>=256 and1..8 workers')
    plan = read(args.plan)
    if plan.get('track') != 1 or plan.get('episode') != 16 or plan.get('frames') != 360:
        raise ValueError('Only the authorized complete Track1 EP16 review is allowed')
    columns = validate_columns(plan)
    evaluated = read(bound(plan['evaluation']))
    validate_evaluated_objects(columns, evaluated['input_bindings'])
    manifest_path = bound(plan['input_manifest'])
    if evaluated['input_manifest_sha256'] != digest(manifest_path) or evaluated['frames'] != 360:
        raise ValueError('Independent evaluation binds a different experiment')
    video = bound(plan['source_video'])
    if digest(video) != evaluated['source_video_sha256']:
        raise ValueError('Original RGB differs from independent evaluation')
    camera = np.asarray(plan['camera_intrinsics'], dtype=np.float32)
    roi = np.asarray(plan['display_roi'], dtype=np.int64)
    if roi.shape != (4,) or roi[2] - roi[0] != roi[3] - roi[1] or roi[2] <= roi[0]:
        raise ValueError('Use one explicit square original-pixel ROI for every source/frame/column')
    sources = [plan['evaluation'], plan['input_manifest'], plan['source_video'], plan['hand_skin_indices']]
    for column in columns:
        role = column['role']
        parity = evaluated['native_decoder'][role]
        if not parity['passed'] or parity['frames_freshly_decoded'] != 360:
            raise ValueError('Every displayed human needs all-frame independent native parity')
        native = bound(column['native'])
        if parity['parameters_sha256'] != digest(native):
            raise ValueError('Displayed native parameters were not independently evaluated')
        sources.extend([column['native'], column['object']])
        object_path = bound(column['object'])
        with np.load(object_path, allow_pickle=False) as archive:
            if (not np.array_equal(archive['frame_indices'], np.arange(360)) or
                    not np.array_equal(np.asarray(archive['camera_intrinsics'], dtype=np.float32), camera)):
                raise ValueError('Displayed object frame IDs or camera differ')
        directory = native.parent / 'review_arrays'
        for key in ('vertices_m', 'joints_3d', 'faces', 'frame_indices', 'camera_intrinsics'):
            path = directory / (key + '.npy')
            value = digest(path)
            if evaluated['native_review_array_sha256'][role][key] != value:
                raise ValueError('Displayed native arrays differ from independent evaluation')
            sources.append({'path': str(path), 'sha256': value})
        if not np.array_equal(np.load(directory / 'frame_indices.npy', allow_pickle=False), np.arange(360)):
            raise ValueError('Original video frame identity changed')
        if not np.array_equal(np.load(directory / 'camera_intrinsics.npy', allow_pickle=False), camera):
            raise ValueError('Displayed human uses a different camera')
    skin_path = bound(plan['hand_skin_indices'])
    selection = evaluated['native_hand_skin_selection']
    if (selection['sha256'] != digest(skin_path) or
            Path(selection['path']).resolve() != skin_path.resolve()):
        raise ValueError('Display skin selection differs from independently evaluated native selection')
    before = {'plan': digest(args.plan), 'renderer': digest(__file__),
              'rasterizer': digest(ROOT / 'src/v2hoi/contact_review.py'),
              'video_helpers': digest(ROOT / 'tools/render_contact_ab.py'),
              'video_writer': digest(ROOT / 'tools/review_human.py')}
    spec = importlib.util.spec_from_file_location('occluded_media_writer', ROOT / 'tools/render_contact_ab.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    writer_spec = importlib.util.spec_from_file_location('occluded_native_video_writer', ROOT / 'tools/review_human.py')
    video_writer = importlib.util.module_from_spec(writer_spec)
    writer_spec.loader.exec_module(video_writer)
    import cv2
    cv2.setNumThreads(1)
    args.output.mkdir()
    (args.output / 'frames').mkdir()
    expected = {}
    for kind in ('rgb', 'transparent'):
        expected[kind + '-comparison.mp4'] = (360, 30.)
        expected.update({f'{kind}-slow-{first:06d}-{last:06d}.mp4': (last - first + 1, 10.) for first, last in WINDOWS})
    writers = {name: video_writer.VideoWriter(args.output / name, (args.size * (4 if name.startswith('rgb') else 3), args.size + 78), fps, cv2)
               for name, (_, fps) in expected.items()}
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    init = (columns, camera, roi, args.size, str(skin_path))
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_renderer, initargs=init) if args.workers > 1 else None
    if pool is None:
        initialize_renderer(*init)
    cap = cv2.VideoCapture(str(video))
    scale = args.size / (roi[2] - roi[0])
    warp = np.asarray([[scale, 0, -roi[0] * scale], [0, scale, -roi[1] * scale]], dtype=float)
    started, samples, seconds, pids = time.monotonic(), [], [], set()

    def pending_frames():
        pending = deque()
        for frame in range(360):
            ok, image = cap.read()
            if not ok:
                raise ValueError('Original video ended before frame' + str(frame))
            source = cv2.warpAffine(image, warp, (args.size, args.size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            future = pool.submit(render_frame, (frame, source)) if pool else None
            pending.append((frame, source, future))
            if len(pending) >= args.workers * 2:
                yield pending.popleft()
        while pending:
            yield pending.popleft()

    try:
        for frame, source, future in pending_frames():
            overlays, transparent, elapsed, pid = future.result() if future else render_frame((frame, source))
            seconds.append(elapsed)
            pids.add(pid)
            for kind, panels in (('rgb', [source, *overlays]), ('transparent', transparent)):
                canvas = np.full((args.size + 78, args.size * len(panels), 3), 20, np.uint8)
                canvas[48:48 + args.size] = np.concatenate(panels, axis=1)
                labels = (['Source RGB'] if kind == 'rgb' else []) + ['OLD A / old depth', 'NEW A / shared new depth', 'NEW B / same new depth + contact']
                for index, label in enumerate(labels):
                    cv2.putText(canvas, label, (index * args.size + 7, 39), cv2.FONT_HERSHEY_SIMPLEX, .40, (235, 235, 235), 1, cv2.LINE_AA)
                cv2.putText(canvas, f'EP16 original {frame:06d} | {frame / 30:.3f}s | ' + ('fixed RGB crop' if kind == 'rgb' else 'synthetic object-local 60deg view'),
                            (8, 16), cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 235, 235), 1, cv2.LINE_AA)
                footer = ('Estimated geometry only; old/new depth differs, NEW A/B object is identical' if kind == 'rgb' else
                          'L: anatomical left / R: right; transparent object, clipped wrists; synthetic visibility only')
                cv2.putText(canvas, footer, (8, args.size + 68), cv2.FONT_HERSHEY_SIMPLEX, .39, (205, 215, 225), 1, cv2.LINE_AA)
                writers[kind + '-comparison.mp4'].write(canvas)
                for first, last in WINDOWS:
                    if first <= frame <= last:
                        writers[f'{kind}-slow-{first:06d}-{last:06d}.mp4'].write(canvas)
                if frame in SAMPLES:
                    image_path = args.output / 'frames' / f'{kind}-{frame:06d}.jpg'
                    if not cv2.imwrite(str(image_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 93]):
                        raise ValueError('Cannot save comparison image')
                    samples.append({'kind': kind, 'original_frame_id': frame, 'file': image_path.relative_to(args.output).as_posix(), 'sha256': digest(image_path)})
            if (frame + 1) % 30 == 0:
                print(f'Rendered {frame + 1}/360 original frames', flush=True)
        if cap.read()[0]:
            raise ValueError('Original video has more than360 frames')
    finally:
        cap.release()
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        for writer in writers.values():
            writer.close()
    videos = helpers.validate_videos(args.output, expected, cv2)
    for item in sources:
        bound(item)
    after = {'plan': digest(args.plan), 'renderer': digest(__file__),
             'rasterizer': digest(ROOT / 'src/v2hoi/contact_review.py'),
             'video_helpers': digest(ROOT / 'tools/render_contact_ab.py'),
             'video_writer': digest(ROOT / 'tools/review_human.py')}
    if before != after:
        raise ValueError('Rendering sources changed')
    summary = {'schema_version': 'ep16_occluded_contact_video.v1', 'recorded_at': datetime.now(timezone.utc).isoformat(),
               'track': 1, 'episode': 16, 'frames': 360, 'fps': 30, 'columns': columns, 'source_bindings': sources,
               'input_manifest_sha256': digest(manifest_path), 'evaluation_sha256': plan['evaluation']['sha256'],
               'sources_before_and_after': before, 'display_roi_original_pixels': roi.tolist(), 'videos': videos,
               'actually_saved_static_samples': samples, 'actual_worker_pids': sorted(pids),
               'mean_frame_render_seconds': float(np.mean(seconds)), 'total_seconds': time.monotonic() - started,
               'transparent_view': 'Display-only fixed object-local60degree camera,0.90m display translation, actual selected native hand faces, hands rendered through transparent object; L/R label anatomical sides; no shape/pose edits',
               'limitations': 'Estimated hand and object geometry only; physical depth, hidden pose, real visibility and official scores are not established',
               'quality_accepted': False, 'contact_model_training_started': False, 'all30_execution_started': False}
    with (args.output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    sections = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>EP16 occluded-hand comparison</title>',
                '<style>body{font:16px system-ui;background:#15211e;color:#e7eee9;max-width:1400px;margin:32px auto;padding:0 18px;line-height:1.7}video{width:100%}a{color:#a9dfc9}</style>',
                '<h1>EP16: occluded-left and shared-depth controlled A/B</h1>',
                '<p>RGB columns: original video / preserved oldA with old object / newA with new object / newB with exactly the same new object. The old result is context, not a same-condition A/B arm.</p>',
                '<p>Transparent-object display: native hand skin shown through the object in a synthetic object-local60degree view. L is anatomical left, R is right. Wrists are clipped by the hand-skin selection; this is a geometry inspection, not a real camera or visibility prediction.</p>',
                '<p>No calibrated physical scale, true hidden finger labels, official score or automatic quality acceptance is claimed. <a href="summary.json">Source and media receipts</a></p>']
    for item in videos:
        sections.append('<h2>' + item['file'] + '</h2><video controls preload="metadata" src="' + item['file'] + '"></video>')
    (args.output / 'index.html').write_text(''.join(sections) + '</html>', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'videos_fully_decoded': len(videos), 'total_seconds': summary['total_seconds']}), flush=True)


if __name__ == '__main__':
    main()
