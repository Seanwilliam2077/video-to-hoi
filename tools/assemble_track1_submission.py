"""Assemble the official Track 1 episode files from the Human and Object deliveries.

fetch-human reads only parameters/selectedhuman.npz from each published Human
archive through HTTP byte ranges; zip CRC checks the member, but the whole
archive hash cannot be checked without downloading it.

assemble pairs each episode's selected native Human parameters with the Object
handoff's baked-scale mesh and per-frame pose, writing the kit's
episode_XXXXXX.npz and episode_XXXXXX_object.glb. It refuses incomplete
timelines and records a shared-scene screen: the Object scene's own Human hips
must stay a near-constant offset from the native Human root. pack runs the
kit's packer, then verify reloads every Parquet row against those files.
None of this is a score, reference alignment or quality acceptance.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from v2hoi.human_result import MHR_TO_TRACK1, load_human_result  # noqa: E402

EPISODES = range(30)
PARAMETERS = 'parameters/selectedhuman.npz'
OBJECT_SCHEMA = 'v2hoi.object-handoff.v1'
SAMPLE = 'data/track_1_sample_submission.parquet'
COMMIT_URL = re.compile(r'^https://github\.com/[\w.-]+/[\w.-]+/commit/[0-9a-f]{40}$')
# Shared-scene screen limits, in metres. The native MHR root lies about 0.9 m
# below the pelvis; all 30 current pairings stay well inside these bounds.
SCREEN = {'residual_median': 0.05, 'residual_p95': 0.25, 'offset_horizontal': 0.15, 'offset_vertical': (-1.1, -0.7)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, fetched in buffered byte ranges."""

    def __init__(self, url, block=1 << 20):
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60) as response:
            self.url, self.size = response.geturl(), int(response.headers['Content-Length'])
        self.position, self.block, self.start, self.buffer = 0, block, 0, b''

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence] + offset
        return self.position

    def read(self, size=-1):
        end = self.size if size is None or size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b''
        if not self.start <= self.position or end > self.start + len(self.buffer):
            stop = min(self.size, max(end, self.position + self.block))
            request = urllib.request.Request(self.url, headers={'Range': f'bytes={self.position}-{stop - 1}'})
            with urllib.request.urlopen(request, timeout=300) as response:
                require(response.status == 206, 'Server ignored the byte range request')
                self.start, self.buffer = self.position, response.read()
            require(len(self.buffer) == stop - self.position, 'Short byte range response')
        data = self.buffer[self.position - self.start:end - self.start]
        self.position = end
        return data


def fetch_human(catalog_path, output, episodes):
    catalog = json.loads(Path(catalog_path).read_text(encoding='utf-8'))
    entries = {entry['episode_id']: entry for entry in catalog['entries']}
    for episode in episodes:
        entry = entries[f'episode_{episode:06d}']
        require(entry['selected_candidate'] == 'baseline', f'Episode {episode}: unexpected Human selection')
        target = Path(output) / entry['episode_id']
        with zipfile.ZipFile(RangeFile(entry['archive']['url'])) as archive:
            names = [name for name in archive.namelist() if name.endswith('/' + PARAMETERS) or name == PARAMETERS]
            require(len(names) == 1, f'Episode {episode}: expected one {PARAMETERS}, found {names}')
            info = archive.getinfo(names[0])
            data = archive.read(info)  # zipfile verifies the member CRC-32
        depth = urllib.request.urlopen(entry['evidence']['depth_scale']['url'], timeout=60).read()
        require(sha256_bytes(depth) == entry['evidence']['depth_scale']['sha256'], f'Episode {episode}: depth-scale hash mismatch')
        target.mkdir(parents=True, exist_ok=True)
        (target / 'selectedhuman.npz').write_bytes(data)
        (target / 'depth-scale.json').write_bytes(depth)
        write_json(target / 'source.json', {
            'record_id': entry['record']['id'], 'archive_url': entry['archive']['url'],
            'archive_sha256_declared': entry['archive']['sha256'], 'archive_sha256_verified': False,
            'member': info.filename, 'member_crc32': f'{info.CRC:08x}', 'member_bytes': info.file_size,
            'sha256': sha256_bytes(data), 'mhr_model_sha256': entry['mhr_model_sha256'],
            'frame_count': entry['frame_count']})
        print(f'episode {episode}: {info.filename} ({info.file_size:,} bytes)', flush=True)


def load_object_handoff(root):
    manifest = json.loads((Path(root) / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest.get('schema') == OBJECT_SCHEMA and manifest.get('provenance') == 'track1',
            'Require the Track 1 Object handoff manifest')
    rows = {row['episode']: row for row in manifest['episodes']}
    require(sorted(rows) == list(EPISODES), 'Object handoff must hold all 30 Track 1 episodes')
    return manifest, rows, {item['path']: item for item in manifest['files']}


def object_file(root, files, relative):
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, f'Unsafe handoff path: {relative}')
    path, item = Path(root) / relative, files.get(relative)
    require(item is not None and path.is_file() and path.stat().st_size == item['bytes'] and sha256(path) == item['sha256'],
            f'Object handoff file differs from its manifest: {relative}')
    return path


def check_object(scene, motion, frames, mesh_path):
    import trimesh

    ids = np.arange(frames)
    require(np.array_equal(scene['original_frame_ids'], ids) and np.array_equal(motion['frame_indices'], ids),
            'Object timeline is not original frames 0..T-1')
    rotation = np.asarray(motion['object_rotation'], dtype=np.float64)
    translation = np.asarray(motion['object_translation'], dtype=np.float64)
    require(rotation.shape == (frames, 3, 3) and translation.shape == (frames, 3), 'Incomplete object trajectory')
    require(np.isfinite(rotation).all() and np.isfinite(translation).all(), 'Non-finite object trajectory')
    require(np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-5)
            and np.allclose(np.linalg.det(rotation), 1.0, atol=1e-5), 'Object rotations are not proper')
    poses = np.asarray(scene['object_poses'])
    require(np.array_equal(poses[:, :3, :3], rotation) and np.array_equal(poses[:, :3, 3], translation),
            'Object scene and motion disagree')
    scale = float(np.asarray(motion['object_scale']).reshape(()))
    require(scale == 1.0, 'Object geometry must carry its fitted scale exactly once (object_scale 1)')
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    require(np.allclose(mesh.vertices, scene['object_vertices'], atol=2e-6, rtol=1e-6)
            and np.array_equal(mesh.faces, scene['object_faces']), 'Object mesh differs from its scene geometry')
    return rotation, translation, scale


def scene_screen(pose, coco17):
    """Hip midpoint of the Object scene's own Human against the native Human root.

    Both are Track 1 camera-frame metres. A shared frame and depth leave a nearly
    constant root-to-pelvis offset; drift, rescaling or another person do not.
    """
    root = np.asarray(pose[:, :3], dtype=np.float64) * 0.1 * MHR_TO_TRACK1
    offset = np.asarray(coco17, dtype=np.float64)[:, [11, 12]].mean(axis=1) - root
    median = np.median(offset, axis=0)
    residual = np.linalg.norm(offset - median, axis=1)
    result = {'offset_median_m': [round(float(v), 4) for v in median],
              'residual_median_m': round(float(np.median(residual)), 4),
              'residual_p95_m': round(float(np.percentile(residual, 95)), 4),
              'residual_max_m': round(float(residual.max()), 4)}
    low, high = SCREEN['offset_vertical']
    result['passed'] = bool(result['residual_median_m'] <= SCREEN['residual_median']
                            and result['residual_p95_m'] <= SCREEN['residual_p95']
                            and max(abs(median[0]), abs(median[2])) <= SCREEN['offset_horizontal']
                            and low <= median[1] <= high)
    return result


def smooth_tracks(pose, rotation, translation, sigma):
    """Gaussian temporal smoothing over original frames, repeating the edge frames.

    Native pose angles are unwrapped and object quaternions sign-aligned first,
    so the filter never averages across a 2*pi or q/-q jump.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation

    def smooth(values):
        return gaussian_filter1d(values, sigma, axis=0, mode='nearest')

    pose = np.asarray(pose, dtype=np.float64)
    smoothed_pose = np.concatenate([smooth(pose[:, :3]), smooth(np.unwrap(pose[:, 3:], axis=0))], axis=1)
    quaternion = Rotation.from_matrix(rotation).as_quat()
    for frame in range(1, len(quaternion)):
        if quaternion[frame] @ quaternion[frame - 1] < 0:
            quaternion[frame] = -quaternion[frame]
    smoothed_rotation = Rotation.from_quat(smooth(quaternion)).as_matrix()
    return smoothed_pose.astype(np.float32), smoothed_rotation, smooth(np.asarray(translation, dtype=np.float64))


def git_state():
    def run(*args):
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {'commit': run('rev-parse', 'HEAD'), 'uncommitted_changes': bool(run('status', '--porcelain'))}


def assemble(human_root, object_root, output, episodes=EPISODES, allow_screen_failures=False, temporal_sigma=0.0):
    output = Path(output)
    require(temporal_sigma >= 0, 'temporal_sigma must be nonnegative')
    require(not output.exists() or not any(output.iterdir()), f'Output must be a new or empty directory: {output}')
    manifest, rows, files = load_object_handoff(object_root)
    prepared = []
    for episode in episodes:
        name, row = f'episode_{episode:06d}', rows[episode]
        frames = row['frames']
        human_path = Path(human_root) / name / 'selectedhuman.npz'
        source = json.loads((human_path.parent / 'source.json').read_text(encoding='utf-8'))
        require(sha256(human_path) == source['sha256'], f'{name}: Human parameters differ from their fetch record')
        human = load_human_result(human_path)
        require(len(human['pose']) == frames, f'{name}: Human has {len(human["pose"])} frames, Object has {frames}')
        if 'frame_indices' in human:
            require(np.array_equal(human['frame_indices'], np.arange(frames)), f'{name}: Human frames are not 0..T-1')
        paths = {key: object_file(object_root, files, row['artifacts'][key])
                 for key in ('scene', 'motion', 'mesh', 'human_context_contract')}
        with np.load(paths['scene'], allow_pickle=False) as scene, np.load(paths['motion'], allow_pickle=False) as motion:
            try:
                rotation, translation, scale = check_object(scene, motion, frames, paths['mesh'])
            except ValueError as error:
                raise ValueError(f'{name}: {error}') from None
            screen = scene_screen(human['pose'], scene['human_coco17'])
        context = json.loads(paths['human_context_contract'].read_text(encoding='utf-8'))
        record = {
            'episode': episode, 'frames': frames, 'object': row.get('object'),
            'human_record': source['record_id'], 'human_parameters_sha256': source['sha256'],
            'object_mesh_sha256': files[row['artifacts']['mesh']]['sha256'],
            'object_motion_sha256': files[row['artifacts']['motion']]['sha256'],
            'same_mhr_model': source.get('mhr_model_sha256') == context.get('decoder', {}).get('mhr_model_sha256'),
            'shared_scene_screen': screen}
        pose = human['pose']
        if temporal_sigma:
            smoothed = smooth_tracks(pose, rotation, translation, temporal_sigma)
            root_shift = np.linalg.norm((smoothed[0][:, :3] - pose[:, :3]) * 0.1, axis=1)
            object_shift = np.linalg.norm(smoothed[2] - translation, axis=1)
            record['temporal_smoothing_shift_m'] = {
                'human_root_median': round(float(np.median(root_shift)), 4), 'human_root_max': round(float(root_shift.max()), 4),
                'object_median': round(float(np.median(object_shift)), 4), 'object_max': round(float(object_shift.max()), 4)}
            pose, rotation, translation = smoothed
        prepared.append((name, pose, human['scales'], human['shape'], rotation, translation, scale, paths['mesh'], record))
    failed = [entry[-1]['episode'] for entry in prepared if not entry[-1]['shared_scene_screen']['passed']]
    require(allow_screen_failures or not failed, f'Shared-scene screen failed for episodes {failed}; inspect before submitting')
    output.mkdir(parents=True, exist_ok=True)
    for name, pose, scales, shape, rotation, translation, scale, mesh, record in prepared:
        np.savez(output / f'{name}.npz', pose=pose, scales=scales, shape=shape,
                 object_rotation=rotation, object_translation=translation, object_scale=np.float64(scale))
        shutil.copyfile(mesh, output / f'{name}_object.glb')
        record.update(episode_npz_sha256=sha256(output / f'{name}.npz'), episode_mesh_sha256=sha256(output / f'{name}_object.glb'))
    report = {
        'schema': 'v2hoi.track1-submission-assembly.v1', 'assembler': git_state(),
        'object_delivery': manifest['delivery_id'], 'object_manifest_sha256': sha256(Path(object_root) / 'manifest.json'),
        'human_iteration': 'sam-mediapipe-20261021 (selected baseline)', 'screen_limits_m': SCREEN,
        'temporal_smoothing': {'sigma_frames': temporal_sigma, 'method': (
            'Gaussian over original frames, nearest edges: native pose (angles unwrapped), object translation and '
            'sign-aligned object quaternions; the screen uses the unsmoothed Human') if temporal_sigma else 'none'},
        'screen_failures': failed, 'episodes': [entry[-1] for entry in prepared],
        'scope': 'Official episode-file format and structural pairing only; no score, reference alignment or quality acceptance'}
    write_json(output / 'assembly.json', report)
    return report


def packed(values, count):
    flat = np.zeros((len(values), 3 * -(-count // 3)))
    flat[:, :count] = np.asarray(values, dtype=np.float64).reshape(len(values), count)
    return flat.reshape(len(values), -1, 3)


def verify(kit, episodes_dir, submission, commit=None):
    """Reload a packed Parquet and compare every row with the episode files."""
    import pandas as pd
    import trimesh

    kit = Path(kit).resolve()
    sys.path.insert(0, str(kit))
    from v2dlb import mhr_submission as ms

    sample = pd.read_parquet(kit / SAMPLE, columns=['row_id'])['row_id'].astype(str)
    table = pd.read_parquet(submission)
    require(list(table.columns) == ['row_id', 'x', 'y', 'z', 'code_commit_url'], 'Unexpected submission columns')
    require(table['row_id'].astype(str).tolist() == sample.tolist(), 'Submission rows differ from the official sample')
    urls = table['code_commit_url'].unique()
    require(len(urls) == 1 and COMMIT_URL.match(str(urls[0])) and commit in (None, urls[0]),
            'Submission must carry exactly one full GitHub commit URL')
    values = table[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    require(np.isfinite(values).all(), 'Submission contains non-finite values')
    parts = table['row_id'].str.split('_', expand=True)
    episode, frame, array, point = (parts[i].astype(np.int64).to_numpy() for i in range(1, 5))
    index = {name: number for number, name in enumerate(ms.array_order())}
    episodes = []
    for number in np.unique(episode):
        name = f'episode_{number:06d}'
        with np.load(Path(episodes_dir) / f'{name}.npz', allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        rows = episode == number
        scored = np.unique(frame[rows & (frame != ms.EPISODE_FRAME)])
        require(scored.max() < len(arrays['pose']), f'{name}: scored frame beyond the reconstruction')
        expected = {
            'mhr_pose': lambda f, p: packed(arrays['pose'], 136)[f, p],
            'mhr_scales': lambda f, p: packed(arrays['scales'][None], 68)[0, p],
            'mhr_shape': lambda f, p: packed(arrays['shape'][None], 45)[0, p],
            'object_rotation': lambda f, p: arrays['object_rotation'][f, p],
            'object_translation': lambda f, p: arrays['object_translation'][f],
            'object_scale': lambda f, p: np.array([[float(arrays['object_scale']), 0.0, 0.0]])[p],
        }
        error = 0.0
        for key, value in expected.items():
            selected = rows & (array == index[key])
            require(selected.any(), f'{name}: no {key} rows')
            if key == 'object_translation':
                require((point[selected] == 0).all(), f'{name}: unexpected translation row')
            error = max(error, float(np.abs(values[selected] - value(frame[selected], point[selected])).max()))
        require(error <= 1e-9, f'{name}: packed parameters differ from the episode file by {error}')
        vertices = values[rows & (array == index['object_mesh_vertices'])]
        faces = values[rows & (array == index['object_mesh_faces'])]
        require(np.array_equal(faces, np.round(faces)) and faces.min() >= 0 and faces.max() < len(vertices),
                f'{name}: invalid packed mesh faces')
        real = faces[(faces[:, 0] != faces[:, 1]) | (faces[:, 1] != faces[:, 2])].astype(np.int64)
        original = trimesh.load(Path(episodes_dir) / f'{name}_object.glb', force='mesh', process=False)
        used = vertices[np.unique(real)]
        extent = np.ptp(used, axis=0) / np.ptp(original.vertices, axis=0)
        require(len(real) and np.all(np.abs(extent - 1) <= 0.02), f'{name}: packed mesh extent changed: {extent}')
        episodes.append({'episode': int(number), 'scored_frames': int(len(scored)), 'first_scored_frame': int(scored[0]),
                         'packed_faces': int(len(real)), 'max_parameter_error': error,
                         'mesh_extent_ratio': [round(float(v), 4) for v in extent]})
    return {'passed': True, 'submission': str(submission), 'submission_sha256': sha256(submission), 'rows': int(len(table)),
            'code_commit_url': str(urls[0]), 'episodes': episodes,
            'scope': 'Row set, parameter round trip and mesh validity; no score or upload'}


def pack(kit, episodes_dir, commit, out):
    require(COMMIT_URL.match(commit), 'commit must be https://github.com/<owner>/<repo>/commit/<40-hex sha>')
    with urllib.request.urlopen(urllib.request.Request(commit, method='HEAD'), timeout=60) as response:
        require(response.status == 200, f'Commit page is not publicly readable: {commit}')
    kit, out = Path(kit).resolve(), Path(out).resolve()
    subprocess.run([sys.executable, 'eval_reconstruction.py', '--episodes', str(Path(episodes_dir).resolve()),
                    '--sample', SAMPLE, '--commit', commit, '--out', str(out)], cwd=kit, check=True)
    report = verify(kit, episodes_dir, out, commit)
    write_json(out.with_suffix('.verify.json'), report)
    return report


def episode_list(text):
    if text == 'all':
        return list(EPISODES)
    selected = set()
    for part in text.split(','):
        first, _, last = part.partition('-')
        selected.update(range(int(first), int(last or first) + 1))
    require(selected <= set(EPISODES), 'Track 1 episodes are 0..29')
    return sorted(selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('fetch-human', help='read selected native Human parameters from the published archives')
    fetch.add_argument('--catalog', type=Path, default=ROOT / 'tracking/human-catalog.json')
    fetch.add_argument('--out', type=Path, required=True)
    fetch.add_argument('--episodes', type=episode_list, default=list(EPISODES), help="'all' or e.g. 0-3,16")
    build = commands.add_parser('assemble', help='write the kit episode NPZ and mesh files')
    build.add_argument('--human', type=Path, required=True, help='fetch-human output')
    build.add_argument('--objects', type=Path, required=True, help='extracted Object handoff root')
    build.add_argument('--out', type=Path, required=True, help='new or empty episodes directory')
    build.add_argument('--allow-screen-failures', action='store_true')
    build.add_argument('--temporal-sigma', type=float, default=0.0,
                       help='Gaussian smoothing width in frames for Human pose and Object pose (default: none)')
    official = commands.add_parser('pack', help='run the kit packer, then verify the Parquet')
    check = commands.add_parser('verify', help='reload a packed Parquet against the episode files')
    for command in (official, check):
        command.add_argument('--kit', type=Path, required=True, help='extracted v2d_submission_kit root')
        command.add_argument('--episodes', type=Path, required=True)
    official.add_argument('--commit', required=True, help='https://github.com/<owner>/<repo>/commit/<sha>')
    official.add_argument('--out', type=Path, required=True)
    check.add_argument('--submission', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'fetch-human':
        fetch_human(args.catalog, args.out, args.episodes)
    elif args.command == 'assemble':
        report = assemble(args.human, args.objects, args.out, allow_screen_failures=args.allow_screen_failures,
                          temporal_sigma=args.temporal_sigma)
        print(json.dumps({'episodes': len(report['episodes']), 'screen_failures': report['screen_failures'],
                          'report': str(args.out / 'assembly.json')}, indent=2))
    else:
        report = (pack(args.kit, args.episodes, args.commit, args.out) if args.command == 'pack'
                  else verify(args.kit, args.episodes, args.submission))
        print(json.dumps({key: report[key] for key in ('passed', 'rows', 'submission_sha256', 'code_commit_url')}, indent=2))


if __name__ == '__main__':
    main()
