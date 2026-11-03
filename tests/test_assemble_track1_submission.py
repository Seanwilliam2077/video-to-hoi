"""Independent synthetic Human/Object handoffs only; no challenge media, kit or model outputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
SPEC = importlib.util.spec_from_file_location('assemble_track1_submission', ROOT / 'tools/assemble_track1_submission.py')
assembly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembly)

from v2hoi.human_result import MHR_TO_TRACK1, write_human_result  # noqa: E402

FRAMES = 6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def human_pose(frames=FRAMES):
    pose = np.zeros((frames, 136), dtype=np.float32)
    pose[:, :3] = np.array([0.4, -9.0, -30.0], dtype=np.float32) + np.linspace(0, 1, frames, dtype=np.float32)[:, None]
    return pose


def build(tmp_path, episodes=(0, 1), human_frames=FRAMES, depth_scale=1.0):
    objects, human = tmp_path / 'objects', tmp_path / 'human'
    rows, files = [], []
    box = tmp_path / 'box.glb'
    trimesh.creation.box(extents=(0.2, 0.1, 0.3)).export(box)
    mesh = trimesh.load(box, force='mesh', process=False)
    for episode in range(30):
        name = f'episode_{episode:06d}'
        base = f'episodes/{name}'
        rows.append({'episode': episode, 'object': 'box', 'frames': FRAMES, 'fps': 30.0, 'artifacts': {
            'scene': f'{base}/scene.npz', 'motion': f'{base}/motion.npz', 'mesh': f'{base}/object.glb',
            'human_context_contract': f'{base}/human-context.json'}})
        if episode not in episodes:
            continue
        folder = objects / base
        folder.mkdir(parents=True)
        angle = np.linspace(0, 0.5, FRAMES)
        rotation = np.stack([[[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]] for a in angle])
        translation = np.column_stack([np.linspace(0, 0.2, FRAMES), np.full(FRAMES, 0.1), np.full(FRAMES, 3.0)])
        poses = np.tile(np.eye(4), (FRAMES, 1, 1))
        poses[:, :3, :3], poses[:, :3, 3] = rotation, translation
        root = human_pose()[:, :3].astype(np.float64) * 0.1 * MHR_TO_TRACK1
        hips = (root + [0.0, -0.9, 0.0]) * depth_scale
        coco = np.zeros((FRAMES, 17, 3), dtype=np.float32)
        coco[:, 11], coco[:, 12] = hips + [0.1, 0, 0], hips - [0.1, 0, 0]
        np.savez(folder / 'scene.npz', original_frame_ids=np.arange(FRAMES), object_poses=poses, human_coco17=coco,
                 object_vertices=mesh.vertices, object_faces=mesh.faces)
        np.savez(folder / 'motion.npz', frame_indices=np.arange(FRAMES), object_rotation=rotation,
                 object_translation=translation, object_scale=np.float64(1.0))
        (folder / 'object.glb').write_bytes(box.read_bytes())
        (folder / 'human-context.json').write_text(json.dumps({'decoder': {'mhr_model_sha256': 'm' * 64}}), encoding='utf-8')
        for key in ('scene', 'motion', 'mesh', 'human_context_contract'):
            path = objects / rows[-1]['artifacts'][key]
            files.append({'path': rows[-1]['artifacts'][key], 'bytes': path.stat().st_size, 'sha256': digest(path)})
        target = human / name / 'selectedhuman.npz'
        write_human_result(target, {'pose': human_pose(human_frames), 'scales': np.zeros(68), 'shape': np.zeros(45),
                                    'frame_indices': np.arange(human_frames)})
        (target.parent / 'source.json').write_text(json.dumps({
            'record_id': f'fixture-ep{episode:02d}', 'sha256': digest(target), 'mhr_model_sha256': 'm' * 64}), encoding='utf-8')
    (objects / 'manifest.json').write_text(json.dumps({
        'schema': assembly.OBJECT_SCHEMA, 'provenance': 'track1', 'delivery_id': 'fixture',
        'episodes': rows, 'files': files}), encoding='utf-8')
    return human, objects


def test_assemble_writes_kit_episode_files(tmp_path):
    human, objects = build(tmp_path)
    report = assembly.assemble(human, objects, tmp_path / 'out', episodes=[0, 1])
    with np.load(tmp_path / 'out/episode_000001.npz') as data, np.load(objects / 'episodes/episode_000001/motion.npz') as motion:
        assert sorted(data.files) == ['object_rotation', 'object_scale', 'object_translation', 'pose', 'scales', 'shape']
        np.testing.assert_array_equal(data['pose'], human_pose())
        np.testing.assert_array_equal(data['object_rotation'], motion['object_rotation'])
        np.testing.assert_array_equal(data['object_translation'], motion['object_translation'])
        assert float(data['object_scale']) == 1.0 and data['scales'].shape == (68,) and data['shape'].shape == (45,)
    assert (tmp_path / 'out/episode_000001_object.glb').read_bytes() == (objects / 'episodes/episode_000001/object.glb').read_bytes()
    screen = report['episodes'][0]['shared_scene_screen']
    assert screen['passed'] and screen['offset_median_m'] == [0.0, -0.9, 0.0] and screen['residual_max_m'] == 0.0
    assert report['screen_failures'] == [] and report['episodes'][0]['same_mhr_model']
    assert json.loads((tmp_path / 'out/assembly.json').read_text())['schema'] == 'v2hoi.track1-submission-assembly.v1'


def test_assemble_rejects_incomplete_human_timeline(tmp_path):
    human, objects = build(tmp_path, human_frames=FRAMES - 1)
    with pytest.raises(ValueError, match='Human has 5 frames, Object has 6'):
        assembly.assemble(human, objects, tmp_path / 'out', episodes=[0])
    assert not (tmp_path / 'out').exists()


def test_assemble_rejects_tampered_object_file(tmp_path):
    human, objects = build(tmp_path)
    (objects / 'episodes/episode_000000/object.glb').write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs from its manifest'):
        assembly.assemble(human, objects, tmp_path / 'out', episodes=[0])


def test_scene_scale_mismatch_fails_screen_unless_allowed(tmp_path):
    human, objects = build(tmp_path, depth_scale=1.3)
    with pytest.raises(ValueError, match=r'screen failed for episodes \[0, 1\]'):
        assembly.assemble(human, objects, tmp_path / 'out', episodes=[0, 1])
    report = assembly.assemble(human, objects, tmp_path / 'out', episodes=[0, 1], allow_screen_failures=True)
    assert report['screen_failures'] == [0, 1]


def test_assemble_requires_an_empty_output(tmp_path):
    human, objects = build(tmp_path)
    (tmp_path / 'out').mkdir()
    (tmp_path / 'out/stale.npz').write_bytes(b'')
    with pytest.raises(ValueError, match='new or empty directory'):
        assembly.assemble(human, objects, tmp_path / 'out', episodes=[0])


def test_episode_list():
    assert assembly.episode_list('0-2,16') == [0, 1, 2, 16]
    assert assembly.episode_list('all') == list(range(30))
    with pytest.raises(ValueError):
        assembly.episode_list('29-30')
