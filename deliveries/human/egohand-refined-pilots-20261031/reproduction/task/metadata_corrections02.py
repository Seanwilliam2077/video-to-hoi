"""Create metadata-only canonical copies of completed EgoHand02 native results.

Executed files, source hashes, parameters, evaluations, review arrays and videos
are never replaced. Every non-report NPY ZIP member is copied byte-for-byte.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

sys.dont_write_bytecode = True
TASK = Path(__file__).resolve().parent
RESULTS = Path('J:/video-to-hoi/runs/egohand-refined-pilots-20261031').resolve()
REPO = Path('C:/Users/arthasxu/.codex/worktrees/human-all30/video-to-hoi').resolve()
DESTINATION = TASK / 'metadata-corrections02'
COUNTS = {16: 360, 11: 877}
PROVENANCE_KEYS = {'independent_observation_provenance', 'generated_model_proposal_provenance',
                   'unclassified_target_provenance'}
ALLOWED_REPORT_KEYS = PROVENANCE_KEYS | {'method', 'observation_semantics', 'metadata_correction'}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bytes_digest(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classifier_snapshot():
    source = REPO / 'tools/refine_human_hands.py'
    content = source.read_bytes()
    text = content.decode('utf-8-sig')
    module = ast.parse(text)
    matches = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'target_provenance']
    if len(matches) != 1 or matches[0].decorator_list:
        raise ValueError('Require the explicit current metadata-only provenance classifier')
    node = matches[0]
    function_text = '\n'.join(text.splitlines()[node.lineno-1:node.end_lineno]) + '\n'
    namespace = {'np': np}
    function_ast = ast.Module(body=[node], type_ignores=[])
    exec(compile(function_ast, str(source), 'exec'), namespace)
    snapshot = DESTINATION / 'classifier-source-reference.py'
    with snapshot.open('xb') as stream:
        stream.write(content)
    provenance = {'source_path': str(source), 'source_sha256_at_capture': bytes_digest(content),
                  'snapshot': str(snapshot), 'snapshot_sha256': digest(snapshot),
                  'function': 'target_provenance', 'first_line': node.lineno, 'last_line': node.end_lineno,
                  'function_source_sha256': bytes_digest(function_text.encode('utf-8')),
                  'interpretation': 'Classification reference only; not the source executed by the original fitting run'}
    return namespace['target_provenance'], provenance


def flag_value(command, name):
    return command[command.index(name)+1]


def correct_episode(episode, classifier, classifier_provenance, loader, loader_path):
    root = RESULTS / f'episode_{episode:06d}'
    fit_path = TASK / f'ep{episode}-fit-egohand02.finished.json'
    fit = read(fit_path)
    if fit.get('returncode') != 0 or fit.get('episode') != episode:
        raise ValueError('Only a successfully finished fit can receive a canonical metadata copy')
    candidate = root / 'candidate-egohand02'
    source_npz = candidate / f'episode_{episode:06d}_human.npz'
    source_report = candidate / 'report.json'
    observations = root / 'observations-smoothed02/observations.npz'
    observation_report = observations.parent / 'report.json'
    command = fit['command']
    if Path(flag_value(command, '--output')).resolve() != candidate.resolve():
        raise ValueError('Successful fit receipt names a different native output')
    if Path(flag_value(command, '--observations')).resolve() != observations.resolve():
        raise ValueError('Successful fit did not use the declared refined smoothed targets')
    executed_refiner = Path(command[2]).resolve()
    sources = [source_npz, source_report, observations, observation_report, fit_path, executed_refiner]
    old_hashes = {str(path): digest(path) for path in sources}
    original_report = read(source_report)
    obs_report = read(observation_report)
    if (original_report['track'] != 1 or original_report['episode'] != episode or
            obs_report['track'] != 1 or obs_report['episode'] != episode or
            obs_report['schema_version'] != 'refined_egohand_proposals.v1' or
            obs_report['source_video_sha256'] != original_report['source_video_sha256']):
        raise ValueError('Metadata correction is not bound to this exact Track 1 clip and generated proposal family')
    if original_report['observations_sha256'] != old_hashes[str(observations)] or original_report['observation_report_sha256'] != old_hashes[str(observation_report)]:
        raise ValueError('As-executed target bindings do not match the immutable observation package')
    if original_report['source_sha256']['tools/refine_human_hands.py'] != old_hashes[str(executed_refiner)]:
        raise ValueError('Executed refiner bytes changed; never rewrite the historical source provenance')
    with np.load(observations, allow_pickle=False) as archive:
        kind = archive['observation_kind'].copy()
    if kind.shape != () or str(kind.item()) != 'egohand_geometric_proposals.v1':
        raise ValueError('Actual NPZ observation-kind tag is not the declared generated target family')
    classified = classifier(copy.deepcopy(obs_report), {'observation_kind': kind})
    if (classified.get('method') != 'source_bound_model_target_native_mhr_sequence_fit' or
            'generated_model_proposal_provenance' not in classified or
            'independent_observation_provenance' in classified):
        raise ValueError('Current classifier did not recognize the actual generated/refined EgoHand targets')
    with np.load(source_npz, allow_pickle=False) as archive:
        original_arrays = {key: archive[key] for key in archive.files}
    if original_arrays['report'].shape != () or original_arrays['report'].dtype.kind not in 'US':
        raise ValueError('Expected one scalar embedded JSON report')
    if json.loads(str(original_arrays['report'].item())) != original_report:
        raise ValueError('As-executed standalone and embedded reports differ')
    original_loaded = loader.load_human_result(source_npz)
    original_schema = loader.validate_human_result(original_loaded, expected_frames=COUNTS[episode])
    updated = copy.deepcopy(original_report)
    for key in PROVENANCE_KEYS:
        updated.pop(key, None)
    updated.update(classified)
    correction = {'schema_version': 'human_metadata_correction.v1', 'recorded_at': now(),
        'reason': 'The frozen02 refiner misclassified generated/refined EgoHand targets as independent MediaPipe. Actual numerical target inputs and fitting configuration were correct and are unchanged.',
        'scope': 'Only top-level target method/provenance/observation semantics and this explicit erratum metadata are changed. All other report fields, including executed source hashes and historical diagnostic labels, are preserved.',
        'historical_metric_label_notice': 'Generic nested diagnostic labels remain as executed; their wording does not turn generated model targets into independent image observations, true visibility, ground truth or official scores.',
        'original_npz': str(source_npz), 'original_npz_sha256': old_hashes[str(source_npz)],
        'original_report': str(source_report), 'original_report_sha256': old_hashes[str(source_report)],
        'observations': str(observations), 'observations_sha256': old_hashes[str(observations)],
        'observation_report': str(observation_report), 'observation_report_sha256': old_hashes[str(observation_report)],
        'successful_fit_receipt': str(fit_path), 'successful_fit_receipt_sha256': old_hashes[str(fit_path)],
        'actual_observation_kind': str(kind.item()), 'actual_observation_report_schema': obs_report['schema_version'],
        'correction_helper': str(Path(__file__).resolve()), 'correction_helper_sha256': digest(__file__),
        'classifier': classifier_provenance, 'executed_source_sha256_preserved': copy.deepcopy(original_report['source_sha256']),
        'geometry_or_fit_recomputed': False, 'numeric_arrays_changed': False,
        'evaluation_and_video_relationship': 'Existing evaluations/review arrays/videos remain as executed against the original NPZ; exact unchanged parameter array bytes bridge them to this canonical metadata copy.',
        'quality_or_contact_accepted': False, 'official_scores_computed': False}
    updated['metadata_correction'] = correction
    unchanged_original = {key:value for key,value in original_report.items() if key not in ALLOWED_REPORT_KEYS}
    unchanged_updated = {key:value for key,value in updated.items() if key not in ALLOWED_REPORT_KEYS}
    if unchanged_original != unchanged_updated or updated['source_sha256'] != original_report['source_sha256']:
        raise AssertionError('An executed numerical/configuration/source report field changed')
    target_root = DESTINATION / f'episode_{episode:06d}'
    if target_root.exists():
        raise ValueError('Preserve the previous metadata correction attempt')
    target_root.mkdir()
    target_npz = target_root / source_npz.name
    target_report = target_root / 'report.json'
    report_array = np.asarray(json.dumps(updated, allow_nan=False, sort_keys=True))
    buffer = io.BytesIO()
    np.save(buffer, report_array, allow_pickle=False)
    corrected_member = buffer.getvalue()
    with zipfile.ZipFile(source_npz, 'r') as archive, zipfile.ZipFile(target_npz, 'x') as output:
        expected_members = {key + '.npy' for key in original_arrays}
        if set(archive.namelist()) != expected_members or len(archive.namelist()) != len(expected_members):
            raise ValueError('Unexpected or duplicate NPZ members')
        for info in archive.infolist():
            payload = corrected_member if info.filename == 'report.npy' else archive.read(info.filename)
            output.writestr(copy.copy(info), payload)
    write(target_report, updated)
    array_checks = []
    with np.load(target_npz, allow_pickle=False) as archive:
        canonical_arrays = {key: archive[key] for key in archive.files}
    if set(canonical_arrays) != set(original_arrays):
        raise AssertionError('An NPZ array key changed')
    with zipfile.ZipFile(source_npz, 'r') as original_zip, zipfile.ZipFile(target_npz, 'r') as canonical_zip:
        if canonical_zip.testzip() is not None:
            raise AssertionError('Canonical NPZ archive is corrupt')
        for key, original in original_arrays.items():
            if key == 'report':
                continue
            canonical = canonical_arrays[key]
            if (canonical.dtype != original.dtype or canonical.shape != original.shape or
                    canonical.tobytes(order='A') != original.tobytes(order='A') or
                    canonical_zip.read(key+'.npy') != original_zip.read(key+'.npy')):
                raise AssertionError(f'Non-report array bytes changed: {key}')
            array_checks.append({'key': key, 'dtype': str(original.dtype), 'shape': list(original.shape),
                'array_payload_sha256': bytes_digest(original.tobytes(order='A')),
                'original_and_canonical_npy_member_sha256': bytes_digest(original_zip.read(key+'.npy')),
                'exact_dtype_shape_payload_and_member_bytes': True})
    canonical_loaded = loader.load_human_result(target_npz)
    canonical_schema = loader.validate_human_result(canonical_loaded, expected_frames=COUNTS[episode])
    if canonical_schema != original_schema or canonical_loaded['report'] != updated or read(target_report) != updated:
        raise AssertionError('Canonical loader/schema or embedded/standalone metadata validation failed')
    after_hashes = {str(path): digest(path) for path in sources}
    if old_hashes != after_hashes:
        raise AssertionError('An original fit/report/observation/source file was modified')
    receipt = {'schema_version': 'human_metadata_correction_receipt.v1', 'recorded_at': now(),
        'track': 1, 'episode': episode, 'passed': True, 'correction_helper_sha256': digest(__file__),
        'originals_sha256_before': old_hashes, 'originals_sha256_after': after_hashes,
        'all_original_bytes_preserved': True, 'canonical_npz': str(target_npz), 'canonical_npz_sha256': digest(target_npz),
        'canonical_report': str(target_report), 'canonical_report_sha256': digest(target_report),
        'changed_npz_array_keys': ['report'], 'unchanged_npz_array_checks': array_checks,
        'changed_report_keys': sorted(key for key in (original_report.keys() | updated.keys()) if original_report.get(key) != updated.get(key)),
        'actual_observation_kind': str(kind.item()), 'actual_observation_report_schema': obs_report['schema_version'],
        'new_method': updated['method'], 'new_observation_semantics': updated['observation_semantics'],
        'executed_source_sha256_preserved': True, 'original_schema': original_schema, 'canonical_schema': canonical_schema,
        'loader': str(loader_path), 'loader_sha256': digest(loader_path), 'classifier': classifier_provenance,
        'neural_or_fit_or_geometry_rerun': False, 'review_evaluation_controller_outputs_replaced': False,
        'quality_or_contact_accepted': False, 'official_scores_computed': False, 'all30_execution_started': False}
    write(target_root / 'correction-receipt.json', receipt)
    print(json.dumps({'episode': episode, 'passed': True, 'canonical_npz': str(target_npz),
                      'canonical_npz_sha256': receipt['canonical_npz_sha256'],
                      'receipt': str(target_root/'correction-receipt.json')}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=int, choices=(16,11), action='append', required=True)
    parser.add_argument('--maximum-wait-seconds', type=int, default=3600)
    args = parser.parse_args()
    if len(args.episode) != len(set(args.episode)) or not 1 <= args.maximum_wait_seconds <= 7200:
        raise ValueError('Use unique authorized clips and a bounded wait of at most two hours')
    if DESTINATION.exists():
        raise ValueError('Preserve the existing correction run; no overwriting or resume without audit')
    DESTINATION.mkdir()
    classifier, provenance = classifier_snapshot()
    loader_path = TASK / 'runtime02/src/v2hoi/human_result.py'
    loader = load_module('metadata_only_frozen_human_result', loader_path)
    write(DESTINATION/'run-started.json', {'recorded_at':now(), 'episodes':args.episode,
        'helper_sha256':digest(__file__), 'classifier':provenance, 'loader_sha256':digest(loader_path),
        'maximum_wait_seconds':args.maximum_wait_seconds, 'all30_execution_authorized':False})
    pending = set(args.episode)
    completed = []
    deadline = time.monotonic() + args.maximum_wait_seconds
    while pending:
        for episode in sorted(pending):
            finished = TASK / f'ep{episode}-fit-egohand02.finished.json'
            if not finished.exists():
                continue
            completed.append(correct_episode(episode,classifier,provenance,loader,loader_path))
            pending.remove(episode)
        progress = {'recorded_at':now(), 'completed_episodes':[x['episode'] for x in completed],
                    'pending_episodes':sorted(pending), 'all30_execution_authorized':False}
        temporary = DESTINATION/'progress.tmp'
        temporary.write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8')
        temporary.replace(DESTINATION/'progress.json')
        if pending:
            if time.monotonic() >= deadline:
                write(DESTINATION/'timeout.json',progress)
                raise TimeoutError('Bounded metadata watcher expired; preserve the completed corrections')
            time.sleep(5)
    write(DESTINATION/'completion.json',{'recorded_at':now(),'status':'completed_metadata_only_copies',
        'episodes':[x['episode'] for x in completed], 'receipts':completed,
        'all_original_bytes_preserved':True,'all30_execution_authorized':False})


if __name__=='__main__':
    main()
