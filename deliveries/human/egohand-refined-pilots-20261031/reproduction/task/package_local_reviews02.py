"""Package completed refined EP16/EP11 reviews without publication or promotion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

TASK = Path(__file__).resolve().parent
RESULTS = Path('J:/video-to-hoi/runs/egohand-refined-pilots-20261031')
DEST = TASK / 'local-delivery02'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if DEST.exists():
        raise ValueError('Preserve previous delivery attempts')
    inputs = read(TASK / 'inputs-manifest02.json')
    assert inputs['track'] == 1 and inputs['all30_execution_authorized'] is False
    assert {e['episode'] for e in inputs['episodes']} == {11, 16}
    errata_path = TASK / 'metadata-errata-manifest02.json'
    errata = read(errata_path)
    assert errata['all_original_bytes_preserved_rechecked']
    assert errata['all_nonreport_npy_members_identical_rechecked']
    assert errata['numeric_parameter_or_geometry_change'] is False
    # Validate complete stage receipts before creating any output archive.
    for ep in (16, 11):
        assert read(RESULTS / f'episode_{ep:06d}/postprocessing-completion02.json')['frames'] == (360 if ep == 16 else 877)
        row = next(x for x in errata['episodes'] if x['episode'] == ep)
        correction = read(Path(row['correction_receipt']))
        assert digest(Path(row['correction_receipt'])) == row['correction_receipt_sha256']
        assert correction['passed'] and correction['all_original_bytes_preserved']
        native = read(RESULTS / f'episode_{ep:06d}/evaluation02.json')['input_bindings']['native_variants']['refined_egohand']
        assert Path(native['path']).resolve() == Path(row['original_npz']).resolve()
        assert native['sha256'] == row['original_npz_sha256'] == digest(Path(row['original_npz']))
        for kind in ('npz', 'report'):
            assert digest(Path(row[f'canonical_{kind}'])) == row[f'canonical_{kind}_sha256'] == correction[f'canonical_{kind}_sha256']
        # Independently prove that the canonical metadata bridge changes no numeric NPY member.
        with zipfile.ZipFile(row['original_npz']) as old, zipfile.ZipFile(row['canonical_npz']) as new:
            assert set(old.namelist()) == set(new.namelist())
            assert all(old.read(n) == new.read(n) for n in old.namelist() if n != 'report.npy')
    DEST.mkdir()
    outputs = []
    for ep, count in ((16, 360), (11, 877)):
        root = RESULTS / f'episode_{ep:06d}'
        review = root / 'review02'
        render = read(review / 'summary.json')
        evaluation = read(root / 'evaluation02.json')
        assert evaluation['frames'] == render['frames'] == count
        assert all(v['passed'] for v in evaluation['native_decoder'].values())
        item = next(e for e in inputs['episodes'] if e['episode'] == ep)
        files = {}

        def add(source, member):
            source = Path(source).resolve()
            assert source.is_file() and not source.is_symlink()
            assert not Path(member).is_absolute() and '..' not in Path(member).parts and member not in files
            files[member] = source

        for row in render['videos']:
            source = review / row['file']
            assert source.stat().st_size == row['bytes'] and digest(source) == row['sha256']
            add(source, row['file'])
        for source in sorted((review / 'frames').glob('*.jpg')):
            add(source, source.relative_to(review).as_posix())
        for name in ('index.html', 'summary.json', 'display-rois.npy'):
            add(review / name, name)
        add(root / 'evaluation02.json', 'checks/full-native-evaluation.json')
        add(errata_path, 'checks/metadata-errata-manifest02.json')
        add(root / 'postprocessing-completion02.json', 'checks/postprocessing-completion.json')
        add(root / 'inference-attempt02/prediction-receipt.json', 'checks/inference-receipt.json')
        for name in ('inputs-manifest02.json', 'protocol02.json', 'mp-router02-observation-audit.json'):
            add(TASK / name, 'checks/' + name)
        for name in ('crop_boxes.npy', 'crop_anchors.npz', 'crop-policy-receipt.json'):
            add(TASK / f'crops02/episode_{ep:06d}' / name, 'crops/' + name)
        native_roles = evaluation['input_bindings']['native_variants']
        for role, entry in native_roles.items():
            path = Path(entry['path'])
            assert digest(path) == entry['sha256']
            report = path.parent / 'report.json'
            if role == 'refined_egohand':
                correction = TASK / f'metadata-corrections02/episode_{ep:06d}'
                add(path, f'parameters/as-executed-{role}.npz')
                add(report, f'checks/as-executed-native-{role}.json')
                add(correction / f'episode_{ep:06d}_human.npz', f'parameters/{role}.npz')
                add(correction / 'report.json', f'checks/native-{role}.json')
                add(correction / 'correction-receipt.json', 'checks/generated-target-metadata-correction.json')
            else:
                add(path, f'parameters/{role}.npz')
                if report.exists():
                    add(report, f'checks/native-{role}.json')
        for kind, native in (('fresh_mediapipe', root / 'candidate-mp02'),
                             ('refined_egohand', root / 'candidate-egohand02')):
            for name in ('hand_joints_m.npy', 'hand_projected.npy', 'active_hands.npy',
                         'optimization_weights.npy', 'original_active_hands.npy', 'support_transition_envelope.npy'):
                add(native / name, f'parameters/{kind}/{name}')
        for role, folder in (('fresh_mediapipe', 'fresh-mp02'), ('refined_raw', 'observations-raw02'),
                             ('refined_smoothed', 'observations-smoothed02')):
            add(root / folder / 'observations.npz', f'proposals/{role}.npz')
            add(root / folder / 'report.json', f'checks/observer-{role}.json')
        add(Path(item['original_observations']), 'proposals/legacy_mediapipe.npz')
        for variant in ('raw', 'smoothed'):
            for side in ('left', 'right'):
                add(root / f'inference-attempt02/{variant}/{side}_results.npz', f'specialist/{variant}/{side}_results.npz')
        inventory = [{'member': member, 'bytes': path.stat().st_size, 'sha256': digest(path)}
                     for member, path in sorted(files.items())]
        text = (
            f'EP{ep:02d}: refined EgoHand versus same-new-crop MediaPipe\n\n'
            'Extract the entire archive and open index.html. Videos need no model server.\n'
            'hands-comparison.mp4 columns: Source RGB / preserved bounded MP / fresh same-new-crop MP / refined EgoHand native MHR.\n'
            'old-new-egohand-comparison.mp4 columns: Source RGB / prior filtered EgoHand native MHR / refined EgoHand native MHR.\n'
            'Top row: anatomical left hand. Bottom row: anatomical right. Original IDs are printed.\n'
            f'All {count} original frames at30fps are retained; critical clips replay every included frame at10fps(one-third speed).\n'
            'Every method shares the same display ROI on each original frame. This differs from the neural input crop.\n'
            'Proposal green: admitted generated target; orange: unadmitted generated estimate, including protocol-heldout, missing or gated support. Orange does not prove occlusion or an explicit fitted prior. Neither color is ground truth or measured visibility.\n'
            'Fresh MP is a shared input to Ego wrist/gate construction, so agreement with it is method-dependent.\n'
            'Exact-input RAW caches supply unchanged views; changed crops are re-inferred. Whole-clip smoothing is recomputed.\n'
            'Native parameters require the exact permitted MHR decoder to recreate3D; standalone specialist outputs are not an official native submission.\n'
            'The refined native file has a provenance-label correction only: all non-report NPY members are byte-identical to the preserved as-executed file. The correction receipt binds both versions. Evaluation/video evidence uses the original numeric arrays.\n'
            'No object-depth composition, true hidden-hand/contact/scale acceptance, official scores, result promotion or all-30 run is implied.\n'
            'inventory.json binds payload files, excluding itself and this README.\n'
        )
        archive = DEST / f'egohand-refined-pilots-20261031-ep{ep:02d}-review.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for member, path in sorted(files.items()):
                output.write(path, member)
            output.writestr('inventory.json', json.dumps(inventory, indent=2) + '\n')
            output.writestr('README.txt', text)
        with zipfile.ZipFile(archive) as check:
            assert check.testzip() is None
            assert len(check.namelist()) == len(set(check.namelist())) == len(inventory) + 2
            for row in inventory:
                with check.open(row['member']) as stream:
                    assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']
        outputs.append({'episode': ep, 'path': str(archive), 'bytes': archive.stat().st_size,
                        'sha256': digest(archive), 'payload_files': len(inventory),
                        'all_zip_members_rehashed': True, 'published': False})
    receipt = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'track': 1,
               'packager_source_sha256': digest(Path(__file__)),
               'metadata_errata_manifest_sha256': digest(errata_path),
               'canonical_numeric_npy_members_reverified': True,
               'authorized_episodes': [16, 11], 'all30_execution_authorized': False,
               'automatic_promotion': False, 'outputs': outputs}
    with (DEST / 'packaging-receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
