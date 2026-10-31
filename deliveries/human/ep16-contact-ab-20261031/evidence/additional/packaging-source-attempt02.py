"""Package completed EP16 manual-contact A/B evidence; no publishing or promotion."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

TASK = Path(__file__).resolve().parent
RESULTS = Path('J:/video-to-hoi/runs/ep16-contact-ab-20261031').resolve()
FORBIDDEN_EVIDENCE_NAMES = ('rasterizer-preview01', 'hand-skin-display-benchmark01',
                            'render-hand-skin-indices01', 'gpu-health', 'app-inventory',
                            'process-inventory', 'process-tree', 'health-before-deadline')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def checked_binding(row):
    path = Path(row['path']).resolve()
    if not path.is_file() or digest(path) != row['sha256']:
        raise ValueError(f'Bound input bytes differ: {path}')
    if 'bytes' in row and path.stat().st_size != row['bytes']:
        raise ValueError(f'Bound input size differs: {path}')
    return path


def render_identity(receipt_path, video, evaluation_path, manifest_path):
    """Bind the actual renderer invocation and its complete frozen source set."""
    receipt_path = Path(receipt_path).resolve()
    receipt = read(receipt_path)
    if (receipt['input_manifest_sha256'] != digest(manifest_path) or
            receipt['evaluation_sha256'] != digest(evaluation_path) or
            receipt['renderer_sha256'] != video['sources_before_and_after']['renderer']):
        raise ValueError('Renderer invocation binds a different manifest/evaluation/source')
    command = receipt['command']
    if len(command) < 3 or command[1] != '-B' or Path(command[2]).name != 'render_contact_ab.py':
        raise ValueError('Unexpected recorded renderer invocation')
    renderer = Path(command[2]).resolve()
    runtime = renderer.parents[1]
    snapshot_path = runtime / 'snapshot.json'
    if not runtime.is_relative_to(TASK) or digest(snapshot_path) != receipt['render_snapshot_sha256']:
        raise ValueError('Actual renderer snapshot identity differs')
    snapshot = read(snapshot_path)
    for row in snapshot['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.py':
            raise ValueError('Unexpected renderer snapshot source')
        if digest(runtime / relative) != row['sha256']:
            raise ValueError('A frozen renderer dependency changed')
    if (digest(renderer) != receipt['renderer_sha256'] or
            digest(runtime / 'src/v2hoi/contact_review.py') != video['sources_before_and_after']['rasterizer']):
        raise ValueError('Displayed renderer/rasterizer bytes differ')
    return {'receipt': receipt_path, 'renderer': renderer, 'runtime': runtime, 'snapshot': snapshot_path}


def validated_anchor_reference(spec, plan):
    declared = spec.get('contact_anchor_reference')
    if declared != plan.get('contact_anchor_reference'):
        raise ValueError('Evaluation and solve plan name different anchor references')
    b_report = read(spec['b']['report']['path'])
    actual = b_report['contact_provenance'].get('anchor_reference')
    if not declared:
        if actual is not None:
            raise ValueError('B used an undeclared anchor reference')
        return None, None
    native = checked_binding(declared)
    if (not actual or Path(actual['path']).resolve() != native or actual['sha256'] != declared['sha256'] or
            actual['same_optimizer_initialization'] is not True or
            actual['validation']['optimization_initialization_changed'] is not False):
        raise ValueError('Actual B reference provenance or original optimizer initialization differs')
    report = native.parent / 'report.json'
    for row in plan['input_bindings']:
        if Path(row['path']).resolve() == report:
            checked_binding(row)
    # The external sidecar must be the metadata embedded in this exact MP NPZ,
    # not a report or parameter-equivalent NPZ substituted from A.
    import numpy as np
    with np.load(native, allow_pickle=False) as archive:
        embedded = archive['report']
        if embedded.shape != () or json.loads(str(embedded.item())) != read(report):
            raise ValueError('Anchor reference external report differs from its exact embedded metadata')
    return native, report


def numeric_review_paths(paths, final_evaluation):
    paths = paths or [TASK / 'protocol-review01/evaluation01-compact-review.json',
                      TASK / 'protocol-review01/evaluation02-independent-review01.json']
    covered = set()
    for path in paths:
        value = read(path)
        covered.add(checked_binding(value['evaluation']))
        if value.get('quality_accepted', False):
            raise ValueError('Numeric review must not silently accept reconstruction quality')
    if covered != {(RESULTS / 'evaluation01.json').resolve(), Path(final_evaluation).resolve()}:
        raise ValueError('Retain the exact initial and final independent numeric reviews')
    return [Path(path).resolve() for path in paths]


def validate_extra_evidence(path):
    path = Path(path).resolve()
    if not (path.is_relative_to(TASK) or path.is_relative_to(RESULTS)) or path.suffix not in ('.json', '.py', '.txt', '.yml'):
        raise ValueError('Additional evidence must be an explicit small task/result source or receipt, not raw logs/assets')
    if path.stat().st_size > 2 * 1024 * 1024 or any(name in path.name for name in FORBIDDEN_EVIDENCE_NAMES):
        raise ValueError('Oversized/private/unused performance evidence is excluded')
    return path


def number_summary(values):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return {'count': 0, 'min': None, 'median': None, 'max': None}
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle-1]+values[middle]) / 2
    return {'count': len(values), 'min': values[0], 'median': median, 'max': values[-1]}


def compact_evaluation(evaluation, evaluation_path):
    value = copy.deepcopy(evaluation)
    for reference in value['image_reference_motion_screens'].values():
        reference.pop('all_adjacent_frame_rows', None)
    for role in value['native_opening_proxy_curves'].values():
        for kind in role.values():
            for side in kind['sides'].values():
                rows = side.pop('rows')
                side['whole_clip_prediction_proxy_summary'] = {
                    name: number_summary([row[name] for row in rows]) for name in
                    ('palm_width', 'thumb_index_distance', 'thumb_middle_distance',
                     'thumb_index_over_palm', 'thumb_middle_over_palm')}
    value['summary_scope'] = 'Both image references, all boundary/window summaries and all geometry diagnostics retained. Raw per-edge rows and per-frame opening curves remain in the exact full evaluation.'
    value['full_evaluation'] = {'path': str(evaluation_path), 'sha256': digest(evaluation_path),
                              'bytes': Path(evaluation_path).stat().st_size}
    return value


def first_attempt_note(plan, report, outputs):
    anchors = report['contact']['before']['admitted_anchors']
    checks = {}
    with zipfile.ZipFile(outputs['A']) as a, zipfile.ZipFile(outputs['B']) as b:
        for name in ('pose', 'shape', 'scales', 'frame_indices', 'fps'):
            checks[name] = a.read(name + '.npy') == b.read(name + '.npy')
    return {'scope': 'Preserved initial anchor-selection attempt, not the final meaningful-attraction experiment',
            'plan_sha256': digest(TASK / 'ab-plan01.json'),
            'anchor_sides_from_actual_report': sorted({row['side'] for row in anchors}),
            'manual_phase_with_mutable_finger_controls': plan['manual_phase_with_mutable_finger_controls'],
            'exact_numeric_npy_member_equality_A_B': checks,
            'interpretation': 'The initial attraction anchors selected only the unsupported left hand during admitted usable carry. This is a structural limitation of that attraction test. Other tip-penetration terms are not claimed to have zero gradients. Final revision uses an explicitly validated read-only MP anchor reference while preserving original SAM initialization and both variable masks.',
            'quality_accepted': False, 'all30_execution_authorized': False}


def validate_complete(evaluation_path, review, manifest_path, render_receipt):
    evaluation_path, review = Path(evaluation_path).resolve(), Path(review).resolve()
    evaluation, video = read(evaluation_path), read(review / 'summary.json')
    for value in (evaluation, video):
        if value['track'] != 1 or value['episode'] != 16 or value['frames'] != 360:
            raise ValueError('Only the complete authorized EP16 experiment can be packaged')
        if value['quality_accepted'] or value['all30_execution_authorized']:
            raise ValueError('Local evidence packaging must not imply quality acceptance or all30 permission')
    if not evaluation['original_frame_coverage_complete'] or not evaluation['same_original_camera_and_fixed_object_for_both']:
        raise ValueError('A/B geometry/control completeness checks are missing')
    if not all(row['passed'] for row in evaluation['native_decoder'].values()):
        raise ValueError('Native decoder parity failed')
    for role in ('a', 'b'):
        for name in ('fixed_identity_unchanged', 'fixed_skeleton_scales_unchanged', 'root_controls_unchanged',
                     'all_unselected_controls_unchanged', 'finite_parameters'):
            if evaluation['parameter_preservation'][role][name] is not True:
                raise ValueError('Native fixed-parameter preservation failed')
    for key in ('exact_active_hand_masks_identical', 'exact_image_weights_identical', 'heldout_image_weights_zero'):
        if not evaluation['matched_controls'][key]:
            raise ValueError('The controlled pair changed an image/mutable-variable condition')
    if not evaluation['contact_attraction_outside_admitted_phase_zero']:
        raise ValueError('Contact attraction remains active outside the manual grasp phase')
    manifest_path = Path(manifest_path).resolve()
    if digest(manifest_path) != evaluation['input_manifest_sha256']:
        raise ValueError('The exact executed evaluation input manifest is missing or changed')
    spec = read(manifest_path)
    if spec != evaluation['input_bindings']:
        raise ValueError('Evaluation input manifest contents differ')
    plan_path = checked_binding(spec['protocol'])
    plan = read(plan_path)
    if digest(plan_path) != evaluation['protocol_sha256']:
        raise ValueError('Evaluation binds a different solve plan')
    for row in plan['input_bindings'] + plan['fitter_source_bindings']:
        checked_binding(row)
    for role in ('A', 'B'):
        finished = read(TASK / f'{plan_path.stem}-{role}.finished.json')
        if finished['returncode'] != 0 or not finished['frozen_inputs_and_sources_unchanged'] or finished['plan_sha256'] != digest(plan_path):
            raise ValueError('A/B did not finish against the exact immutable plan')
    for name in ('baseline', 'a', 'b', 'historical_no_contact', 'object', 'object_report', 'phases',
                 'protocol', 'criteria', 'legacy_reference', 'fresh_reference', 'manifest', 'source_video'):
        checked_binding(spec[name])
    for role in ('a', 'b'):
        if checked_binding(spec[role]) != Path(plan['outputs'][role.upper()]).resolve() / 'episode_000016_human.npz':
            raise ValueError('Evaluation uses a different A/B native output')
        for name in ('report', 'support', 'weights'):
            checked_binding(spec[role][name])
    validated_anchor_reference(spec, plan)
    if video['evaluation_sha256'] != digest(evaluation_path) or video['source_video_sha256'] != evaluation['source_video_sha256']:
        raise ValueError('Videos bind a different evaluation or source video')
    if video['input_manifest_sha256'] != digest(manifest_path):
        raise ValueError('Video input-manifest identity differs from the actual final manifest')
    required_arrays = {'vertices_m', 'joints_3d', 'faces', 'frame_indices', 'camera_intrinsics'}
    if set(video['native_review_array_sha256']) != {'a', 'b'}:
        raise ValueError('Expected exactly the displayed A/B geometry identities')
    for role, values in video['native_review_array_sha256'].items():
        if not required_arrays <= set(values) or not set(values) <= set(evaluation['native_review_array_sha256'][role]):
            raise ValueError('Displayed native review-array identity scope differs')
        for name, expected in values.items():
            if expected != evaluation['native_review_array_sha256'][role][name]:
                raise ValueError('Displayed native geometry is not the final evaluated geometry')
            if digest(Path(spec[role]['path']).parent / 'review_arrays' / (name + '.npy')) != expected:
                raise ValueError('Displayed native review arrays changed after evaluation/rendering')
    render_identity(render_receipt, video, evaluation_path, manifest_path)
    for attempt in ('01', '02'):
        finished = read(TASK / f'evaluation{attempt}.finished.json')
        snapshot_path = TASK / f'evaluation-runtime{attempt}/snapshot.json'
        if (finished['returncode'] != 0 or finished['input_and_runtime_source_bytes_unchanged'] is not True or
                finished['runtime_snapshot_sha256'] != digest(snapshot_path) or
                finished['output_sha256'] != digest(RESULTS / f'evaluation{attempt}.json')):
            raise ValueError('Evaluation source snapshot or output is not the actually completed invocation')
    expected = {'contact-comparison.mp4': (360, 30),
                'contact-slow-000040-000110.mp4': (71, 10),
                'contact-slow-000230-000300.mp4': (71, 10)}
    if {row['file'] for row in video['videos']} != set(expected):
        raise ValueError('Expected all three complete comparison videos')
    for row in video['videos']:
        if (row['frames_fully_decoded'], row['fps']) != expected[row['file']]:
            raise ValueError('Comparison full-decode frame/fps count differs')
        path = review / row['file']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Video bytes changed since complete decode')
    for row in video['actually_saved_static_samples']:
        if digest(review / row['file']) != row['sha256']:
            raise ValueError('Saved comparison sample changed')
    return evaluation, video, plan


def payload_files(evaluation_path, review, manifest_path, render_receipt, static_reviews, integrity,
                  evidence=(), numeric_reviews=None, context_review=None, context_render_receipt=None):
    evaluation_path, review = Path(evaluation_path).resolve(), Path(review).resolve()
    evaluation, video, plan = validate_complete(evaluation_path, review, manifest_path, render_receipt)
    spec, files = evaluation['input_bindings'], {}

    def add(source, member):
        original = Path(source)
        source = original.resolve()
        member_path = Path(member)
        if original.is_symlink() or not source.is_file() or member_path.is_absolute() or '..' in member_path.parts or member in files:
            raise ValueError(f'Invalid or duplicated review payload: {member}')
        if not (source.is_relative_to(TASK) or source.is_relative_to(RESULTS) or source in permitted_external):
            raise ValueError('Payload is not an explicit source-bound experiment file')
        files[member] = source

    permitted_external = set()
    for name in ('baseline', 'historical_no_contact', 'legacy_reference', 'fresh_reference', 'manifest'):
        path = checked_binding(spec[name])
        permitted_external.add(path)
        if (path.parent / 'report.json').is_file():
            permitted_external.add((path.parent / 'report.json').resolve())
    anchor_native, anchor_report = validated_anchor_reference(spec, plan)
    if anchor_native:
        permitted_external.update((anchor_native, anchor_report))
        add(anchor_native, 'parameters/anchor-selection-reference.npz')
        add(anchor_report, 'checks/reference-native-report.json')
    for row in video['videos']:
        add(review / row['file'], row['file'])
    for row in video['actually_saved_static_samples']:
        add(review / row['file'], row['file'])
    for name in ('index.html', 'summary.json', 'display-roi.npy'):
        add(review / name, name)
    add(evaluation_path, 'checks/full-evaluation.json')
    add(manifest_path, 'checks/' + Path(manifest_path).name)
    add(render_receipt, 'checks/primary-render-invocation.json')
    if context_review:
        if not context_render_receipt:
            raise ValueError('Context review requires its explicit actual renderer receipt')
        _, context_video, _ = validate_complete(evaluation_path, context_review, manifest_path, context_render_receipt)
        for row in context_video['videos'] + context_video['actually_saved_static_samples']:
            add(Path(context_review) / row['file'], 'context/' + row['file'])
        for name in ('index.html', 'summary.json', 'display-roi.npy'):
            add(Path(context_review) / name, 'context/' + name)
        add(context_render_receipt, 'checks/context-render-invocation.json')
    for role in ('baseline', 'a', 'b', 'historical_no_contact'):
        path = checked_binding(spec[role])
        add(path, f'parameters/{role}.npz')
        add(path.parent / 'report.json', f'checks/native-{role}.json')
    for name in ('legacy_reference', 'fresh_reference'):
        path = checked_binding(spec[name])
        add(path, f'references/{name}.npz')
        add(path.parent / 'report.json', f'checks/{name}-observer.json')
    for role in ('a', 'b'):
        root = Path(spec[role]['path']).parent
        names = ['hand_joints_m.npy', 'hand_projected.npy', 'active_hands.npy', 'original_active_hands.npy',
                 'optimization_weights.npy', 'support_transition_envelope.npy']
        if role == 'b':
            names += ['contact_phase_weights.npy', 'contact_anchor_weights.npy', 'contact_anchor_targets_m.npy']
        for name in names:
            add(root / name, f'parameters/{role}/{name}')
    object_root = checked_binding(spec['object']).parent
    for name in ('contact_ready.npz', 'geometry.ply', 'object_candidate.npz', 'feature_evidence.npz',
                 'report.json', 'validation.json', 'contact-export-report.json', 'temporal-summary.json'):
        add(object_root / name, 'object/' + name)
    add(checked_binding(spec['manifest']), 'checks/source-manifest.json')
    plan_path = checked_binding(spec['protocol'])
    add(plan_path, 'checks/' + plan_path.name)
    for name in ('task-start.json', 'manual-contact-phases01.json',
                 'frozen-contact-ab-criteria01.json', 'object-pipeline-completed01.json',
                 'object-contact-eligibility-preflight01.json', 'object-static-review01.json'):
        add(TASK / name, 'checks/' + name)
    for role in ('A', 'B'):
        for suffix in ('started.json', 'child-started.json', 'finished.json'):
            name = f'{plan_path.stem}-{role}.{suffix}'
            add(TASK / name, 'checks/' + name)
    phases = read(checked_binding(spec['phases']))
    review_receipt = Path(phases['source_review_receipt'])
    if digest(review_receipt) != phases['source_review_receipt_sha256']:
        raise ValueError('Manual phase source review changed')
    add(review_receipt, 'checks/manual-phase-source-review.json')
    for row in read(review_receipt)['actual_tool_viewed_images']:
        path = checked_binding(row)
        add(path, 'source-phase-review/' + path.name)
    if not static_reviews:
        raise ValueError('At least one actual post-fit static-review receipt is required')
    for path in static_reviews:
        value = read(path)
        if value.get('episode', 16) != 16 or value.get('quality_accepted', False):
            raise ValueError('Static review has the wrong scope or acceptance claim')
        add(path, 'checks/static-review/' + Path(path).name)
    if integrity:
        add(integrity, 'checks/completed-integrity.json')
    # Preserve the initial pair as separate evidence, without recycling its videos.
    original_plan = read(TASK / 'ab-plan01.json')
    initial_outputs = {role: Path(original_plan['outputs'][role]) / 'episode_000016_human.npz' for role in ('A', 'B')}
    for role, path in initial_outputs.items():
        add(path, f'attempt01/parameters/{role}.npz')
        add(path.parent / 'report.json', f'attempt01/checks/native-{role}.json')
        for name in ('active_hands.npy', 'optimization_weights.npy'):
            add(path.parent / name, f'attempt01/parameters/{role}/{name}')
        if role == 'B':
            for name in ('contact_phase_weights.npy', 'contact_anchor_weights.npy', 'contact_anchor_targets_m.npy'):
                add(path.parent / name, 'attempt01/parameters/B/' + name)
        for suffix in ('started.json', 'child-started.json', 'finished.json'):
            name = f'ab-plan01-{role}.{suffix}'
            if read(TASK / f'ab-plan01-{role}.finished.json')['returncode'] != 0:
                raise ValueError('Initial attempt did not finish; preserve and investigate it separately')
            add(TASK / name, 'attempt01/checks/' + name)
    for name in ('ab-plan01.json', 'evaluation-inputs01.json'):
        add(TASK / name, 'attempt01/checks/' + name)
    initial_evaluation_path = RESULTS / 'evaluation01.json'
    initial_evaluation = read(initial_evaluation_path)
    if initial_evaluation['source_video_sha256'] != evaluation['source_video_sha256'] or initial_evaluation['frames'] != 360:
        raise ValueError('Initial attempt evidence has a different source or coverage')
    for role in ('a', 'b'):
        if checked_binding(initial_evaluation['input_bindings'][role]) != initial_outputs[role.upper()]:
            raise ValueError('Initial evaluation refers to different preserved native outputs')
    for name in ('baseline', 'object', 'phases', 'criteria', 'fresh_reference', 'legacy_reference'):
        if initial_evaluation['input_bindings'][name]['sha256'] != spec[name]['sha256']:
            raise ValueError('Object, phases or other fixed A/B inputs changed between revisions')
    add(initial_evaluation_path, 'attempt01/checks/full-evaluation.json')
    add(TASK / 'hand-skin-selection-note01.json', 'checks/hand-skin-selection-note01.json')
    for path in numeric_review_paths(numeric_reviews, evaluation_path):
        add(path, 'checks/independent-review/' + path.name)
    for path in evidence:
        path = validate_extra_evidence(path)
        if path not in files.values():
            add(path, 'checks/additional/' + path.name)
    note = first_attempt_note(original_plan, read(initial_outputs['B'].parent / 'report.json'), initial_outputs)
    generated = {'attempt01/checks/compact-evaluation.json': json.dumps(compact_evaluation(initial_evaluation, initial_evaluation_path), indent=2, allow_nan=False).encode() + b'\n',
                 'attempt01/checks/structural-attraction-note.json': json.dumps(note, indent=2, allow_nan=False).encode() + b'\n'}
    return files, generated, evaluation, video, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--render-receipt', type=Path, required=True)
    parser.add_argument('--context-review', type=Path)
    parser.add_argument('--context-render-receipt', type=Path)
    parser.add_argument('--static-review', type=Path, action='append', required=True)
    parser.add_argument('--integrity', type=Path)
    parser.add_argument('--evidence', type=Path, action='append', default=[])
    parser.add_argument('--numeric-review', type=Path, action='append')
    parser.add_argument('--output-dir', type=Path, default=TASK / 'local-delivery02')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--preflight-report', type=Path, default=TASK / 'package-preflight01.json')
    args = parser.parse_args()
    files, generated, evaluation, video, plan = payload_files(args.evaluation, args.review, args.manifest,
        args.render_receipt, args.static_review, args.integrity, args.evidence, args.numeric_review,
        args.context_review, args.context_render_receipt)
    destination = args.output_dir.resolve()
    if destination.exists():
        raise ValueError('Preserve previous package attempts; output directory must not exist')
    inventory = [{'member': name, 'bytes': path.stat().st_size, 'sha256': digest(path)} for name, path in sorted(files.items())]
    inventory += [{'member': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} for name, raw in sorted(generated.items())]
    preflight = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'status': 'all_payloads_verified_no_archive_written',
                 'files': len(inventory), 'payload_bytes': sum(row['bytes'] for row in inventory),
                 'evaluation_sha256': digest(args.evaluation), 'video_summary_sha256': digest(args.review / 'summary.json'),
                 'context_video_summary_sha256': digest(args.context_review / 'summary.json') if args.context_review else None,
                 'render_receipt_sha256': digest(args.render_receipt),
                 'context_render_receipt_sha256': digest(args.context_render_receipt) if args.context_review else None,
                 'packager_source_sha256': digest(__file__), 'all30_execution_authorized': False}
    if args.preflight_only:
        write_new(args.preflight_report, preflight)
        print(json.dumps(preflight), flush=True)
        return
    destination.mkdir()
    archive = destination / 'ep16-contact-ab-20261031-review02.zip'
    notes = ('EP16: same-camera manual-contact A/B review\n\n'
             'Extract the entire archive and open index.html. No model server is needed to play the three videos.\n'
             'Columns: original RGB / A same fit without contact / B same fit with manually admitted contact.\n'
             'The full clip retains all360 original frames at30fps; critical windows replay every included frame at10fps.\n'
             'The primary display is a fixed source crop for readable hands/object. If context/ is present, its index.html retains the separately bound global view of the same exact evaluated geometry.\n'
             'All methods and frames use one common display ROI. Blue is the native human; orange is the uncertain reconstructed object.\n'
             'Shared mesh depth testing does not establish real table/background occlusion, physical contact or metric scale accuracy.\n'
             'Both fits have identical observation-supported mutable controls and image weights; manual contact does not activate hidden-hand controls.\n'
             'Revision02 uses a pinned read-only MP-native anchor reference for B only; both optimizer initializations and priors remain original SAM. The initial structurally limited anchor-selection pair is separately preserved under attempt01 and is not silently promoted.\n'
             'The object, camera, scale, manual phases, decoder and fit bounds are fixed for both arms.\n'
             'Image references and manual phases are development evidence, not true hidden motion or official scores.\n'
             'Native parameters require the separately permitted exact MHR decoder for3D reconstruction. Huge body vertices/model weights/source video are excluded.\n'
             'No quality/contact/scale acceptance, learned contact training, EP11 run, all30 run, result promotion or public upload is implied.\n'
             'inventory.json lists exact payload identities, excluding itself and this README.\n')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for name, source in sorted(files.items()):
            out.write(source, name)
        for name, raw in sorted(generated.items()):
            out.writestr(name, raw)
        out.writestr('inventory.json', json.dumps(inventory, indent=2) + '\n')
        out.writestr('README.txt', notes)
    with zipfile.ZipFile(archive) as check:
        if check.testzip() is not None or len(check.namelist()) != len(set(check.namelist())) or len(check.namelist()) != len(inventory) + 2:
            raise ValueError('Review archive CRC/member check failed')
        for row in inventory:
            with check.open(row['member']) as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                    raise ValueError('Archived payload bytes differ')
    receipt = {**preflight, 'status': 'completed_verified_local_archive', 'track': 1, 'episode': 16, 'frames': 360,
               'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive),
               'all_zip_members_rehashed': True, 'primary_videos_fully_decoded': len(video['videos']),
               'context_videos_fully_decoded': len(read(args.context_review / 'summary.json')['videos']) if args.context_review else 0,
               'videos_fully_decoded': len(video['videos']) + (len(read(args.context_review / 'summary.json')['videos']) if args.context_review else 0),
               'quality_accepted': False, 'published': False, 'contact_model_training_started': False,
               'episode11_execution_started': False, 'automatic_promotion': False}
    write_new(destination / 'packaging-receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
