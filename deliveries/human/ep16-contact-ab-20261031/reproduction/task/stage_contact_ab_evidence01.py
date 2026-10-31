"""Copy an explicit small EP16 A/B evidence/source set into human; never Git-stage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from package_contact_ab01 import (
    read, digest, write_new, validate_complete, compact_evaluation, first_attempt_note,
    render_identity, validated_anchor_reference, numeric_review_paths, validate_extra_evidence,
)

TASK = Path(__file__).resolve().parent
REPO = Path('C:/Users/arthasxu/.codex/worktrees/human-all30/video-to-hoi').resolve()
RESULTS = Path('J:/video-to-hoi/runs/ep16-contact-ab-20261031').resolve()
DEST = REPO / 'deliveries/human/ep16-contact-ab-20261031'
MAX_FILE_BYTES = 2 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--render-receipt', type=Path, required=True)
    parser.add_argument('--context-review', type=Path)
    parser.add_argument('--context-render-receipt', type=Path)
    parser.add_argument('--static-review', type=Path, action='append', required=True)
    parser.add_argument('--packaging-receipt', type=Path, default=TASK / 'local-delivery02/packaging-receipt.json')
    parser.add_argument('--integrity', type=Path)
    parser.add_argument('--evaluation-source', type=Path, default=TASK / 'evaluation-runtime02/tools/evaluate_contact_ab.py')
    parser.add_argument('--render-source', type=Path)
    parser.add_argument('--rasterizer-source', type=Path)
    parser.add_argument('--helper-root', type=Path, default=TASK / 'evaluation-runtime02')
    parser.add_argument('--test-source', type=Path, action='append', default=[])
    parser.add_argument('--evidence', type=Path, action='append', default=[])
    parser.add_argument('--numeric-review', type=Path, action='append')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--preflight-report', type=Path, default=TASK / 'stage-preflight01.json')
    args = parser.parse_args()
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip() != 'human':
        raise ValueError('The destination checkout must remain on human')
    status_hash = digest(REPO / 'docs/status.html')
    if status_hash != read(TASK / 'task-start.json')['preexisting_unrelated_status_sha256']:
        raise ValueError('Preserve the unrelated status-page edit')
    evaluation_path, review = args.evaluation.resolve(), args.review.resolve()
    evaluation, video, plan = validate_complete(evaluation_path, review, args.manifest, args.render_receipt)
    primary_render = render_identity(args.render_receipt, video, evaluation_path, args.manifest)
    context_render, context_video = None, None
    if args.context_review:
        if not args.context_render_receipt:
            raise ValueError('Context display requires its explicit actual renderer receipt')
        _, context_video, _ = validate_complete(evaluation_path, args.context_review, args.manifest, args.context_render_receipt)
        context_render = render_identity(args.context_render_receipt, context_video, evaluation_path, args.manifest)
    packaging = read(args.packaging_receipt)
    if (packaging['status'] != 'completed_verified_local_archive' or packaging['episode'] != 16 or
            packaging['frames'] != 360 or packaging['published'] or not packaging['all_zip_members_rehashed'] or
            packaging['packager_source_sha256'] != digest(TASK / 'package_contact_ab01.py') or
            packaging['evaluation_sha256'] != digest(evaluation_path) or
            packaging['video_summary_sha256'] != digest(review / 'summary.json') or
            packaging['context_video_summary_sha256'] != (digest(args.context_review / 'summary.json') if args.context_review else None) or
            packaging['render_receipt_sha256'] != digest(args.render_receipt) or
            packaging['context_render_receipt_sha256'] != (digest(args.context_render_receipt) if args.context_review else None) or
            digest(packaging['path']) != packaging['sha256']):
        raise ValueError('Final local review package does not bind these exact completed outputs')
    planned = {}
    _, anchor_report = validated_anchor_reference(evaluation['input_bindings'], plan)

    def add(source, relative, category, expected=None):
        original = Path(source)
        source, relative = original.resolve(), Path(relative)
        if not (source.is_relative_to(TASK) or source.is_relative_to(RESULTS) or source.is_relative_to(REPO) or source == anchor_report):
            raise ValueError('Evidence must be an explicitly named task/result/worktree source')
        if original.is_symlink() or not source.is_file() or source.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f'Missing/oversized evidence source: {source}')
        if source.suffix.lower() not in ('.json', '.py', '.txt', '.log', '.yml'):
            raise ValueError('Only small source/text evidence belongs in this Git delivery')
        if any(name in source.name for name in ('rasterizer-preview01', 'hand-skin-display-benchmark01', 'render-hand-skin-indices01')):
            raise ValueError('Unused performance probes are excluded from the actual final delivery')
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe evidence destination')
        actual = digest(source)
        if expected and actual != expected:
            raise ValueError(f'Frozen source differs: {source}')
        key = relative.as_posix()
        if key in planned:
            if planned[key]['sha256'] != actual:
                raise ValueError('Conflicting evidence destination')
            return
        target = DEST / relative
        if target.exists():
            raise ValueError(f'Preserve existing delivery attempts; output exists: {target}')
        planned[key] = {'source': source, 'target': target, 'sha256': actual,
                        'bytes': source.stat().st_size, 'category': category}

    runtimes = [(TASK / name, count) for name, count in
                (('runtime01', 6), ('runtime02', 6), ('evaluation-runtime01', 6), ('evaluation-runtime02', 6))]
    render_roots = {primary_render['runtime']}
    if context_render:
        render_roots.add(context_render['runtime'])
    runtimes += [(path, 5) for path in sorted(render_roots)]
    for runtime, expected_count in runtimes:
        snapshot = read(runtime / 'snapshot.json')
        if len(snapshot['files']) != expected_count:
            raise ValueError('Unexpected executed source snapshot scope')
        for row in snapshot['files']:
            if Path(row['path']).is_absolute() or '..' in Path(row['path']).parts or Path(row['path']).suffix != '.py':
                raise ValueError('Unexpected source snapshot member')
            add(runtime / row['path'], Path('reproduction/task') / runtime.name / row['path'], 'exact_attempt_specific_executed_source', row['sha256'])
        add(runtime / 'snapshot.json', Path('reproduction/task') / runtime.name / 'snapshot.json', 'attempt_source_snapshot')
    object_pin = read(TASK / 'object-runtime-source-pin01.json')
    for name, expected in object_pin['files'].items():
        if Path(name).name != name or Path(name).suffix not in ('.py', '.json'):
            raise ValueError('Unexpected object source manifest member')
        add(TASK / 'object-runtime01' / name, Path('reproduction/task/object-runtime01') / name, 'exact_object_preparation_source', expected)
    add(TASK / 'run_object_pipeline01.py', 'reproduction/task/run_object_pipeline01.py', 'object_runner', object_pin['runner_sha256'])
    add(TASK / 'recover_object_export01.py', 'reproduction/task/recover_object_export01.py', 'preserved_environment_recovery_recipe')
    add(args.evaluation_source, 'reproduction/evaluation/tools/evaluate_contact_ab.py', 'executed_evaluator', evaluation['evaluation_source_sha256'])
    add(args.render_source or primary_render['renderer'], 'reproduction/primary-render/tools/render_contact_ab.py', 'executed_renderer', video['sources_before_and_after']['renderer'])
    add(args.rasterizer_source or primary_render['runtime'] / 'src/v2hoi/contact_review.py', 'reproduction/primary-render/src/v2hoi/contact_review.py', 'executed_rasterizer', video['sources_before_and_after']['rasterizer'])
    for name, expected in evaluation['helper_source_sha256'].items():
        add(args.helper_root / 'tools' / (name + '.py'), Path('reproduction/evaluation/tools') / (name + '.py'), 'executed_evaluation_helper', expected)
    test_sources = args.test_source or [REPO / 'tests' / name for name in
                                       ('test_contact_ab.py', 'test_contact_anchor_reference.py', 'test_contact_ab_render.py', 'test_contact_ab_evaluation.py')]
    workflow_sources = None
    for name in ('workflow-python02.json', 'workflow-node02.json', 'workflow-registry02.json', 'workflow-catalog02.json'):
        receipt = read(TASK / name)
        if receipt['returncode'] != 0 or receipt['source_sha256_before'] != receipt['source_sha256_after']:
            raise ValueError('A required workflow02 check failed or its sources changed')
        if workflow_sources is None:
            workflow_sources = receipt['source_sha256_after']
        elif workflow_sources != receipt['source_sha256_after']:
            raise ValueError('Workflow02 checks verified different source versions')
        add(TASK / name, Path('evidence/checks') / name, 'successful_explicit_workflow02_check')
    for path in test_sources:
        key = 'tests/' + path.name
        if key not in workflow_sources:
            raise ValueError('Copied test source was not bound by the successful workflow02 receipt')
        add(path, Path('reproduction/tests') / path.name, 'workflow02_verified_test_source', workflow_sources[key])
    add(REPO / '.github/workflows/tests.yml', 'reproduction/verification/tests.yml', 'workflow02_verified_explicit_test_list', workflow_sources['.github/workflows/tests.yml'])
    for name in ('package_contact_ab01.py', 'stage_contact_ab_evidence01.py'):
        add(TASK / name, Path('reproduction/task') / name, 'delivery_recipe')
    task_evidence = ('task-start.json', 'prior-human-input-bindings.json', 'manual-contact-phases01.json',
                     'frozen-contact-ab-criteria01.json', 'ab-plan01.json', 'object-runtime-source-pin01.json',
                     'object-pipeline-completed01.json', 'object-static-review01.json',
                     'object-contact-eligibility-preflight01.json', 'evaluation-recipe01.json',
                     'evaluation-inputs01.json', 'contact-wrapper-focused01.json',
                     'anchor-reference-focused02.json', 'hand-skin-selection-note01.json')
    for name in task_evidence:
        add(TASK / name, Path('reproduction/task') / name, 'frozen_condition_or_provenance')
    add(args.manifest, Path('reproduction/task') / args.manifest.name, 'exact_final_evaluation_manifest')
    final_plan_path = Path(evaluation['input_bindings']['protocol']['path'])
    add(final_plan_path, Path('reproduction/task') / final_plan_path.name, 'exact_final_solve_plan')
    phases = read(TASK / 'manual-contact-phases01.json')
    add(phases['source_review_receipt'], 'evidence/manual-phase-source-review.json', 'actual_source_RGB_review', phases['source_review_receipt_sha256'])
    for plan_stem in ('ab-plan01', final_plan_path.stem):
        for role in ('A', 'B'):
            for suffix in ('started.json', 'child-started.json', 'finished.json'):
                name = f'{plan_stem}-{role}.{suffix}'
                add(TASK / name, Path('reproduction/task') / name, 'attempt_specific_actual_fit_invocation')
    for stage in ('initial', 'temporal', 'validation', 'contact-export', 'contact-export-recovery', 'samples-recovery'):
        for suffix in ('started.json', 'finished.json'):
            name = f'object-{stage}01.{suffix}'
            add(TASK / name, Path('reproduction/task') / name, 'original_and_recovered_object_attempt')
    object_root = Path(evaluation['input_bindings']['object']['path']).parent
    for name in ('report.json', 'validation.json', 'contact-export-report.json', 'temporal-summary.json'):
        add(object_root / name, 'evidence/object-' + name, 'current_camera_object_diagnostic')
    for role in ('a', 'b'):
        add(evaluation['input_bindings'][role]['report']['path'], 'evidence/native-' + role + '-report.json', 'native_fit_report')
    initial_plan = read(TASK / 'ab-plan01.json')
    initial_outputs = {role: Path(initial_plan['outputs'][role]) / 'episode_000016_human.npz' for role in ('A', 'B')}
    for role, path in initial_outputs.items():
        add(path.parent / 'report.json', f'evidence/attempt01/native-{role}-report.json', 'preserved_initial_native_fit_report')
    for path in numeric_review_paths(args.numeric_review, evaluation_path):
        add(path, 'evidence/independent-review/' + path.name, 'independent_attempt_specific_negative_or_positive_checks')
    add(review / 'summary.json', 'evidence/video-integrity.json', 'full_video_decode_receipt')
    add(args.render_receipt, 'evidence/primary-render-invocation.json', 'exact_actual_primary_renderer_invocation')
    if context_video:
        add(args.context_review / 'summary.json', 'evidence/context-video-integrity.json', 'same_geometry_global_context_video_receipt')
        add(args.context_render_receipt, 'evidence/context-render-invocation.json', 'exact_actual_context_renderer_invocation')
    if anchor_report:
        add(anchor_report, 'evidence/anchor-selection-reference-native-report.json', 'exact_MP_anchor_reference_external_metadata')
    for path in args.static_review:
        add(path, 'evidence/static-review/' + path.name, 'actual_static_QA')
    for path in args.evidence:
        path = validate_extra_evidence(path)
        if not any(row['source'] == path for row in planned.values()):
            add(path, 'evidence/additional/' + path.name, 'explicit_additional_check_or_invocation')
    if args.integrity:
        add(args.integrity, 'completed-integrity.json', 'combined_integrity')
    add(args.packaging_receipt, 'local-review-package.json', 'exact_local_archive_identity')
    initial_evaluation_path = RESULTS / 'evaluation01.json'
    generated = {'summary.json': json.dumps(compact_evaluation(evaluation, evaluation_path), indent=2, allow_nan=False).encode() + b'\n',
                 'attempt01-summary.json': json.dumps(compact_evaluation(read(initial_evaluation_path), initial_evaluation_path), indent=2, allow_nan=False).encode() + b'\n',
                 'attempt01-structural-attraction-note.json': json.dumps(first_attempt_note(initial_plan, read(initial_outputs['B'].parent / 'report.json'), initial_outputs), indent=2, allow_nan=False).encode() + b'\n',
                 'reproduction/README.txt': (
                     'EP16 current-camera manual-contact A/B reproduction\n\n'
                     'These are exact executed source snapshots and small receipts, not a portable installer.\n'
                     'Original absolute J:/C:/ paths must be deliberately relocated with new receipts.\n'
                     'Both six-file fit snapshots runtime01/02 are retained. First and final evaluation sources are separate. Primary close-up and optional global-context renderers use all native faces and separate five-file snapshots; only the viewport differs, not evaluated geometry.\n'
                     'The original object export failure and the environment-only recovery are both retained.\n'
                     'Both A/B use the same original native SAM initialization, fresh MP observations, camera, fixed object geometry/relative scale,600iterations and mutable-variable masks.\n'
                     'Only B adds the existing weak manual-phase contact terms. Long unsupported left-hand controls stay at their baseline; contact is not hidden-hand recovery.\n'
                     'Final revision02 explicitly uses a pinned read-only MP-native anchor-selection reference. It changes no optimization initialization, priors, masks, camera, scale, phase labels or fit bounds. Initial attempt01 attraction selected unsupported left-hand anchors and is separately retained as a structural limitation, not promoted.\n'
                     'The local ZIP contains all three videos, native parameters, object geometry/poses, references and full evaluation; Git keeps compact summaries.\n'
                     'Separately permitted native MHR weights, Python environments and original Track1 EP16 video/masks are required to rerun; they are excluded here.\n'
                     'Physical scale/contact truth, official scores, result promotion, learned contact training, EP11 and all30 execution are not established or authorized by these artifacts.\n'
                 ).encode()}
    for relative, raw in generated.items():
        if len(raw) > MAX_FILE_BYTES or (DEST / relative).exists() or relative in planned:
            raise ValueError('Generated summary is oversized or would overwrite an existing attempt')
    if (DEST / 'evidence-inventory.json').exists() or sum(row['bytes'] for row in planned.values()) + sum(map(len, generated.values())) > 16 * 1024 * 1024:
        raise ValueError('Existing inventory or small-evidence budget exceeded')
    preflight = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'status': 'passed_all_inputs_no_repository_copy',
                 'source_sha256': digest(__file__), 'packager_sha256': digest(TASK / 'package_contact_ab01.py'),
                 'source_files': len(planned), 'generated_files': len(generated),
                 'bytes': sum(row['bytes'] for row in planned.values()) + sum(map(len, generated.values())),
                 'status_page_sha256': status_hash, 'all_planned_targets_absent': True,
                 'original_failed_export_returncode': read(TASK / 'object-contact-export01.finished.json')['returncode'],
                 'recovery_returncode': read(TASK / 'object-contact-export-recovery01.finished.json')['returncode']}
    if args.preflight_only:
        write_new(args.preflight_report, preflight)
        print(json.dumps(preflight), flush=True)
        return
    rows = []
    for relative, row in sorted(planned.items()):
        target = row['target']
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(row['source'].read_bytes())
        if digest(target) != row['sha256'] or digest(row['source']) != row['sha256']:
            raise ValueError('Source or copied bytes changed during staging')
        rows.append({'path': relative, 'bytes': row['bytes'], 'sha256': row['sha256'], 'category': row['category']})
    for relative, raw in generated.items():
        path = DEST / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        rows.append({'path': relative, 'bytes': len(raw), 'sha256': digest(path), 'category': 'generated_compact_evidence_or_reproduction_notes'})
    if digest(REPO / 'docs/status.html') != status_hash:
        raise ValueError('Unrelated status page changed during evidence staging')
    inventory = {**preflight, 'status': 'copied_small_evidence_only', 'track': 1, 'authorized_episodes': [16],
                 'files': rows, 'file_count': len(rows), 'scope': 'These exact copied/generated files only; authored delivery README, module index and this inventory are excluded.',
                 'quality_accepted': False, 'all30_execution_authorized': False,
                 'git_stage_commit_push_performed': False}
    write_new(DEST / 'evidence-inventory.json', inventory)
    print(json.dumps({'inventory': str(DEST / 'evidence-inventory.json'), 'files': len(rows),
                      'sha256': digest(DEST / 'evidence-inventory.json')}), flush=True)


if __name__ == '__main__':
    main()
