"""Package two completed local review bundles without publishing or selecting them."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

TASK = Path(__file__).resolve().parent
RESULTS = Path('J:/video-to-hoi/runs/egohand-pilots-20261031').resolve()
DESTINATION = (TASK / 'local-delivery01').resolve()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if DESTINATION.exists() or not DESTINATION.is_relative_to(TASK):
        raise ValueError('Use a fresh task-local delivery directory')
    inputs = read(TASK / 'inputs/manifest.json')
    assert inputs['track'] == 1 and inputs['all30_execution_authorized'] is False
    assert {item['episode'] for item in inputs['episodes']} == {11, 16}
    DESTINATION.mkdir()
    outputs = []
    for episode, frames in ((16, 360), (11, 877)):
        root = RESULTS / f'episode_{episode:06d}'
        item = next(item for item in inputs['episodes'] if item['episode'] == episode)
        review = root / 'review-raw01'
        render = read(review / 'summary.json')
        assert render['frames'] == frames and render['all_frames_and_shared_crops']
        files = {}

        def add(source, member):
            source = Path(source).resolve()
            assert source.is_file() and not source.is_symlink()
            if member in files or Path(member).is_absolute() or '..' in Path(member).parts:
                raise ValueError('Unsafe or duplicate archive member')
            files[member] = source

        for entry in render['videos']:
            source = review / entry['file']
            assert digest(source) == entry['sha256'] and source.stat().st_size == entry['bytes']
            add(source, entry['file'])
        for source in sorted((review / 'frames').glob('*.jpg')):
            add(source, source.relative_to(review).as_posix())
        add(review / 'index.html', 'index.html')
        add(review / 'summary.json', 'summary.json')
        add(root / 'grasp-motion01/summary.json', 'checks/grasp-motion-summary.json')
        add(root / 'grasp-motion01/curves.npz', 'checks/grasp-motion-curves.npz')
        add(root / 'egohand-raw-attempt01/prediction-receipt.json', 'checks/inference-receipt.json')
        for variant in ('raw', 'smoothed'):
            evaluation = read(root / f'evaluation-{variant}01.json')
            assert evaluation['frames'] == frames and evaluation['original_frame_coverage_complete']
            assert all(row['passed'] for row in evaluation['native_decoder'].values())
            assert read(TASK / f'ep{episode:02d}-fit-{variant}01.finished.json')['returncode'] == 0
            add(root / f'evaluation-{variant}01.json', f'checks/evaluation-{variant}.json')
            native = root / f'candidate-{variant}01'
            add(native / f'episode_{episode:06d}_human.npz', f'parameters/egohand-{variant}.npz')
            add(native / 'report.json', f'checks/fit-{variant}.json')
            for name in ('hand_joints_m.npy', 'hand_projected.npy', 'active_hands.npy',
                         'optimization_weights.npy', 'original_active_hands.npy', 'support_transition_envelope.npy'):
                add(native / name, f'parameters/egohand-{variant}/{name}')
            add(root / f'observations-{variant}01/observations.npz', f'proposals/{variant}/observations.npz')
            add(root / f'observations-{variant}01/report.json', f'checks/adapter-{variant}.json')
            for side in ('left', 'right'):
                add(root / f'egohand-raw-attempt01/{variant}/{side}_results.npz',
                    f'proposals/{variant}/{side}_results.npz')
        add(Path(item['baseline']['path']), 'parameters/sam-baseline.npz')
        add(Path(item['mediapipe_candidate']), 'parameters/current-mediapipe.npz')
        add(Path(item['original_observations']), 'proposals/mediapipe-observations.npz')
        inventory = [{'member': member, 'bytes': source.stat().st_size, 'sha256': digest(source)}
                     for member, source in sorted(files.items())]
        readme = (
            f'EP{episode:02d}: controlled EgoHand versus current bounded MediaPipe\n\n'
            'Extract the entire archive, then open index.html in a browser. No model server is required for videos.\n'
            'The raw-smoothed video columns are Source RGB / current MediaPipe / raw EgoHand fitted to MHR / smoothed EgoHand fitted to MHR.\n'
            'Top row: anatomical left hand. Bottom row: anatomical right hand. Original frame IDs are printed.\n'
            f'All {frames} original frames at 30 fps are retained. Critical clips replay every included frame at 10 fps (one-third speed).\n'
            'Proposal videos separate detector image evidence from generated specialist geometry. Neither is ground truth.\n'
            'Full native reconstruction is not embedded in the videos: parameter NPZs require the exact permitted MHR model to decode 3D.\n'
            'Identity and skeleton scale are fixed. Object-depth occlusion is not composited; overlap is not measured penetration.\n'
            'Prediction smoothness is not grasp correctness. Missing image evidence is not hidden ground truth.\n'
            'These are experimental candidates. No dashboard result is replaced, official score reported, contact/scale accepted or all-30 run authorized.\n'
            'inventory.json binds payload files; it excludes itself and this explanatory README.\n'
        )
        archive = DESTINATION / f'egohand-pilots-20261031-ep{episode:02d}-review.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for member, source in sorted(files.items()):
                output.write(source, member)
            output.writestr('inventory.json', json.dumps(inventory, indent=2) + '\n')
            output.writestr('README.txt', readme)
        with zipfile.ZipFile(archive) as check:
            assert check.testzip() is None
            assert len(check.namelist()) == len(set(check.namelist())) == len(inventory) + 2
            for row in inventory:
                with check.open(row['member']) as stream:
                    assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']
        outputs.append({'episode': episode, 'path': str(archive), 'bytes': archive.stat().st_size,
                        'sha256': digest(archive), 'payload_files': len(inventory),
                        'all_zip_members_rehashed': True, 'published': False})
    receipt = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'track': 1,
               'authorized_episodes': [16, 11], 'all30_execution_authorized': False,
               'automatic_promotion': False, 'outputs': outputs}
    path = DESTINATION / 'packaging-receipt.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
