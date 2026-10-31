"""Stage an explicit public-safe refined-pilot source/evidence allowlist into human.

Prepared task-local recipe: do not run before final summary, integrity, packaging
and actual static review receipts exist. No Git stage/commit/push is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

TASK = Path(__file__).resolve().parent
REPO = Path('C:/Users/arthasxu/.codex/worktrees/human-all30/video-to-hoi').resolve()
RESULTS = Path('J:/video-to-hoi/runs/egohand-refined-pilots-20261031').resolve()
DEST = (REPO / 'deliveries/human/egohand-refined-pilots-20261031').resolve()
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
PRIVATE_NAMES = ('health-before-deadline', 'process-tree', 'gpu-context', 'gpu_context', 'app-inventory', 'process-inventory')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def task_json(path):
    path = Path(path).resolve()
    if not path.is_relative_to(TASK) or path.suffix != '.json':
        raise ValueError('Final evidence must be an explicit task-local JSON file')
    return path


def require_success(path):
    value = read(path)
    if value.get('returncode', value.get('exit_code')) != 0:
        raise ValueError(f'Unsuccessful stage/check: {path}')
    return value


def verify_metadata_bridge():
    path = TASK / 'metadata-errata-manifest02.json'
    manifest = read(path)
    if manifest['track'] != 1 or {x['episode'] for x in manifest['episodes']} != {16, 11}:
        raise ValueError('Wrong canonical correction scope')
    roles = []
    for item in manifest['episodes']:
        ep = item['episode']
        receipt_path = Path(item['correction_receipt'])
        if digest(receipt_path) != item['correction_receipt_sha256']:
            raise ValueError('Correction receipt bytes changed')
        receipt = read(receipt_path)
        if not receipt['passed'] or not receipt['all_original_bytes_preserved'] or not receipt['executed_source_sha256_preserved']:
            raise ValueError('Metadata correction did not preserve the executed result')
        for source, expected in receipt['originals_sha256_before'].items():
            if digest(source) != expected or receipt['originals_sha256_after'][source] != expected:
                raise ValueError('An as-executed original changed after correction')
        for kind in ('npz', 'report'):
            if digest(item[f'canonical_{kind}']) != item[f'canonical_{kind}_sha256'] or item[f'canonical_{kind}_sha256'] != receipt[f'canonical_{kind}_sha256']:
                raise ValueError('Canonical correction payload changed')
        original_report = read(item['original_report'])
        canonical_report = read(item['canonical_report'])
        executed_refiner = TASK / 'runtime02/tools/refine_human_hands.py'
        executed_hash = digest(executed_refiner)
        if original_report['source_sha256']['tools/refine_human_hands.py'] != executed_hash or canonical_report['source_sha256'] != original_report['source_sha256']:
            raise ValueError('Historical executed source provenance was rewritten')
        if canonical_report['method'] != 'source_bound_model_target_native_mhr_sequence_fit' or 'independent_observation_provenance' in canonical_report:
            raise ValueError('Canonical report still mislabels generated targets')
        classifier = receipt['classifier']
        if digest(classifier['snapshot']) != classifier['snapshot_sha256']:
            raise ValueError('Metadata-only classifier reference changed')
        with zipfile.ZipFile(item['original_npz']) as old, zipfile.ZipFile(item['canonical_npz']) as new:
            if old.namelist() != new.namelist():
                raise ValueError('Canonical NPZ member keys differ')
            for name in old.namelist():
                if name != 'report.npy' and old.read(name) != new.read(name):
                    raise ValueError('A non-report NPY member changed')
        roles.append({'episode': ep, 'executed_fitting_classifier_sha256': executed_hash,
                      'metadata_only_classifier_reference_sha256': classifier['snapshot_sha256'],
                      'canonical_executed_source_hashes_preserved': True,
                      'nonreport_npy_members_identical': True})
    return manifest, roles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=TASK/'completed-summary02.json')
    parser.add_argument('--integrity', type=Path, required=True)
    parser.add_argument('--packaging-receipt', type=Path, default=TASK/'local-delivery02/packaging-receipt.json')
    parser.add_argument('--preflight-only', action='store_true', help='Read/verify all sources and targets; copy nothing into the repository')
    parser.add_argument('--preflight-report', type=Path, default=TASK/'stage-refined-preflight02.json')
    args = parser.parse_args()
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip() != 'human':
        raise ValueError('The authorized destination checkout must remain on human')
    if not DEST.is_relative_to(REPO/'deliveries/human'):
        raise ValueError('Delivery destination escaped the authorized module folder')
    status_path = REPO/'docs/status.html'
    status_hash = digest(status_path)
    if status_hash != read(TASK/'task-start.json')['preexisting_unrelated_status_sha256']:
        raise ValueError('Preserve the unrelated status-page edit; inspect any subsequent change')
    inventory_path = DEST/'evidence-inventory.json'
    if inventory_path.exists():
        raise ValueError('A completed staging inventory already exists; no overwrite/rerun')
    summary_path, integrity_path, packaging_path = map(task_json, (args.summary,args.integrity,args.packaging_receipt))
    require_success(TASK/'postprocessing-finished02.json')
    summary, integrity, packaging = read(summary_path), read(integrity_path), read(packaging_path)
    if summary['track'] != 1 or {x['episode'] for x in summary['episodes']} != {16,11}:
        raise ValueError('Final summary must contain both authorized clips')
    if set(integrity['authorized_episodes']) != {16,11} or integrity['exact_prior_raw_cache_views'] != 2160 or integrity['new_neural_hand_views'] != 314:
        raise ValueError('Final integrity coverage/cache counts differ')
    bridge = integrity['executed_refiner_metadata_erratum']['canonical_correction_manifest']
    if bridge is None:
        raise ValueError('Final integrity must include the canonical metadata bridge')
    errata, classifier_roles = verify_metadata_bridge()
    if packaging['packager_source_sha256'] != digest(TASK/'package_local_reviews02.py') or packaging['metadata_errata_manifest_sha256'] != digest(TASK/'metadata-errata-manifest02.json') or not packaging['canonical_numeric_npy_members_reverified']:
        raise ValueError('Packaging does not bind the executed recipe and validated canonical correction')
    if {item['episode'] for item in packaging['outputs']} != {16,11} or packaging['all30_execution_authorized'] is not False:
        raise ValueError('Packaging scope differs')
    for item in packaging['outputs']:
        if item['published'] or not item['all_zip_members_rehashed'] or digest(item['path']) != item['sha256']:
            raise ValueError('Local package bytes/provenance differ')
    planned = {}

    def add(source, relative, category, expected=None):
        original = Path(source)
        if original.is_symlink():
            raise ValueError('No symlink payloads')
        source = original.resolve()
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe staged path')
        if not (source.is_relative_to(TASK) or source.is_relative_to(RESULTS)):
            raise ValueError('Only explicit task/authorized-result evidence may be copied')
        if any(value in source.name.lower() for value in PRIVATE_NAMES):
            raise ValueError('Private GPU/application/process inventory is excluded')
        if not source.is_file() or source.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f'Missing or oversized evidence: {source}')
        if source.suffix.lower() in {'.pt','.ckpt','.pkl','.mp4','.jpg','.jpeg','.png','.zip','.dll','.pyd','.exe'}:
            raise ValueError('Large assets/media/binaries are delivered separately')
        if source.suffix.lower() in {'.npy','.npz'} and not any(part in {'crops','crops02'} for part in relative.parts):
            raise ValueError('Only small declared crop-condition arrays belong in source evidence')
        actual = digest(source)
        if expected is not None and actual != expected:
            raise ValueError(f'Frozen evidence bytes differ: {source}')
        target = (DEST/relative).resolve()
        if not target.is_relative_to(DEST):
            raise ValueError('Staged target escaped the delivery folder')
        key = relative.as_posix()
        if key in planned:
            if planned[key]['sha256'] != actual or planned[key]['source'] != source:
                raise ValueError('Conflicting duplicate staged path')
            return
        if target.exists():
            raise ValueError(f'Planned output already exists; no overwrites: {target}')
        planned[key] = {'source':source,'target':target,'sha256':actual,'bytes':source.stat().st_size,'category':category}

    # Keep original task-relative layout rather than moving recipes away from their inputs/runtime.
    plan_path = TASK/'runtime-reproduction-bundle-plan02.json'
    plan = read(plan_path)
    for item in plan['files']:
        source = Path(item['absolute_path']).resolve()
        if not source.is_relative_to(TASK):raise ValueError('Runtime allowlist escaped the task')
        add(source,Path('reproduction/task')/item['relative_path'],'inference_recipe_or_bound_evidence',item['sha256'])
    add(plan_path,'reproduction/task/runtime-reproduction-bundle-plan02.json','runtime_allowlist')
    cfg = read(TASK/'postprocessing-config02.json')
    runtime = (TASK/'runtime02').resolve();snapshot_path=runtime/'snapshot.json';snapshot=read(snapshot_path)
    if digest(snapshot_path)!=cfg['runtime_snapshot']['sha256'] or len(snapshot['files'])!=77:
        raise ValueError('Executed frozen runtime snapshot differs')
    for item in snapshot['files']:
        relative=Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix!='.py':raise ValueError('Unexpected runtime source member')
        add(runtime/relative,Path('reproduction/task/runtime02')/relative,'exact_executed_runtime_source',item['sha256'])
    add(snapshot_path,'reproduction/task/runtime02/snapshot.json','executed_runtime_manifest')
    # The actual MP observer and initial crop preparation used separate frozen roots.
    runtime02_hashes={item['path']:item['sha256'] for item in snapshot['files']}
    source_set_roles=[]
    mp_root=TASK/'mp-runtime01';mp_snapshot=read(mp_root/'snapshot.json')
    if len(mp_snapshot['files'])!=3:raise ValueError('Actual fresh-MP snapshot must retain its three source files')
    mp_roles=[]
    for item in mp_snapshot['files']:
        rel=Path(item['path'])
        if rel.is_absolute() or '..' in rel.parts or rel.suffix!='.py':raise ValueError('Unexpected MP source member')
        add(mp_root/rel,Path('reproduction/task/mp-runtime01')/rel,'exact_executed_fresh_MP_source',item['sha256'])
        mp_roles.append({'path':item['path'],'actual_source_sha256':item['sha256'],
                         'same_path_in_runtime02':item['path'] in runtime02_hashes,
                         'byte_equivalent_to_runtime02':runtime02_hashes.get(item['path'])==item['sha256']})
    add(mp_root/'snapshot.json','reproduction/task/mp-runtime01/snapshot.json','actual_fresh_MP_source_manifest')
    source_set_roles.append({'source_root':'mp-runtime01','actual_stage':'fresh MediaPipe observer02',
                             'retained_complete_snapshot':True,'files':mp_roles})
    # Bind initial crop production to the exact per-output source copies, not a different captured wrapper.
    source_note=read(TASK/'crop-preparation-source-note02.json')
    prep_roles=[]
    for ep in (16,11):
        crop_root=TASK/'crops'/f'episode_{ep:06d}'
        declared=read(crop_root/'crop-policy-receipt.json')['source_sha256']
        files=[]
        for name,expected in declared.items():
            source=crop_root/'source_snapshot'/Path(name).name
            add(source,Path('reproduction/task/crops')/f'episode_{ep:06d}'/'source_snapshot'/Path(name).name,
                'actual_initial_crop_producer_source',expected)
            files.append({'recorded_path':name,'actual_producer_sha256':expected,
                          'byte_equivalent_to_runtime02':runtime02_hashes.get(name)==expected})
        if not any(str(value).replace('\\','/').endswith('/mp-runtime01/tools/human_hand/observations.py')
                   for value in read(TASK/f'ep{ep}-fresh-mp02.json')['command']):
            raise ValueError('Fresh observer command no longer identifies its actual frozen MP root')
        prep_roles.append({'episode':ep,'source_root':f'crops/episode_{ep:06d}/source_snapshot',
                           'actual_stage':'initial crop producer as bound by output source receipts','files':files})
    add(TASK/'runtime01/snapshot.json','reproduction/task/runtime01/snapshot.json','captured_historical_manifest_not_actual_prep_wrapper_bytes')
    add(TASK/'crop-preparation-source-note02.json','reproduction/task/crop-preparation-source-note02.json','producer_vs_captured_hashing_wrapper_note')
    source_set_roles.append({'actual_stage':'initial tight-crop preparation01','source_sets':prep_roles,
                             'captured_wrapper_differs_only_in_digest_helper':True,
                             'no_original_receipt_or_snapshot_rewritten':True})
    recipes=['run_job.py','run_postprocessing_job.py','run_refined_postprocessing02.py',
             'render_refined_egohand_comparison.py','verify_refined_completed_evidence02.py',
             'summarize_completed02.py','package_local_reviews02.py','metadata_corrections02.py',
             'stage_refined_delivery_evidence02.py','run_workflow_checks03.py','run_focused_torch_checks02.py']
    for name in recipes:add(TASK/name,Path('reproduction/task')/name,'task_recipe')
    add(TASK/'reproduction-README02.txt','reproduction/README.txt','reproduction_limits')
    task_evidence=['task-start.json','protocol01.json','inputs-manifest01.json','protocol02.json','inputs-manifest02.json',
        'postprocessing-config02.json','postprocessing-preflight02.json','postprocessing-started02.json','postprocessing-finished02.json',
        'routing-decision01.json','adapter-design-and-depth-diagnosis01.json','crop-input-audit01.json','crop-input-audit02.json',
        'mp-router02-observation-audit.json','postprocessing-trial01-archive.json','postprocessing-stopped-for-routing02.json',
        'controller02-stop-selftest01.json','controller02-stop-selftest02.json','metadata-errata-manifest02.json',
        'metadata-corrections02-job01.json','metadata-corrections02-job01.started.json','metadata-corrections02-job01.finished.json']
    for name in task_evidence:add(TASK/name,Path('reproduction/task')/name,'task_evidence')
    for name in ['workflow-python-tests03.json','workflow-node03.json','workflow-registry03.json','workflow-catalog03.json','focused-torch-tests02.json']:
        value=require_success(TASK/name)
        if value.get('source_sha256_before')!=value.get('source_sha256_after'):raise ValueError('Source changed during a required check')
        add(TASK/name,Path('evidence/checks')/name,'explicit_workflow_or_focused_check')
    for name in ['run-started.json','completion.json','classifier-source-reference.py']:
        add(TASK/'metadata-corrections02'/name,Path('reproduction/task/metadata-corrections02')/name,'metadata_only_classifier_evidence')
    metric_coverage=[]
    for ep,count in [(16,360),(11,877)]:
        root=RESULTS/f'episode_{ep:06d}'
        complete=read(root/'postprocessing-completion02.json')
        if complete['frames']!=count or complete['quality_accepted'] or complete['all30_execution_authorized']:
            raise ValueError('Completed stage scope/acceptance metadata differs')
        row=next(item for item in summary['episodes'] if item['episode']==ep)
        if not {'legacy_mediapipe','fresh_mediapipe'}<=set(row['image_reference_motion_screens']):
            raise ValueError('Do not omit legacy-reference failures behind fresh masks')
        compact_path=TASK/f'ep{ep}-compact-evaluation02.json'
        if compact_path.exists() and not {'legacy_mediapipe','fresh_mediapipe'}<=set(read(compact_path)['references']):
            raise ValueError('A supplied compact table must retain both fresh and legacy references')
        metric_coverage.append({'episode':ep,'full_legacy_and_fresh_reference_tables_copied_without_filtering':True,
                               'mandatory_compact_source':str(summary_path),
                               'additional_per_episode_compact_available':compact_path.exists(),
                               'pooled_D2_support':'20 non-wrist hand landmarks per hand; prediction-only diagnostic'})
        for parent in ['crops','crops02']:
            for name in ['crop-policy-receipt.json','crop_boxes.npy','crop_anchors.npz']:
                add(TASK/parent/f'episode_{ep:06d}'/name,Path('reproduction/task')/parent/f'episode_{ep:06d}'/name,'trial_or_final_crop_provenance')
        for stage in ['fresh-mp02','fit-mp02','adapt-raw02','adapt-smoothed02','fit-egohand02','evaluation02','render02','inference-attempt02']:
            require_success(TASK/f'ep{ep}-{stage}.finished.json')
            for suffix in ['.json','.started.json','.finished.json']:
                name=f'ep{ep}-{stage}'+suffix;add(TASK/name,Path('reproduction/task')/name,'actual_stage_command_and_receipt')
        for name in [f'ep{ep}-final02-independent-numeric-review01.json',f'ep{ep}-final02-independent-static-review01.json',
                     f'ep{ep}-prediction-verification02.json',f'ep{ep}-retarget-verification02.json']:
            add(TASK/name,Path('evidence')/name,'numeric_static_or_integrity_QA')
        for name in [f'ep{ep}-compact-evaluation02.json',f'ep{ep}-completed-integrity02.json',f'ep{ep}-hand-update-and-self-motion02.json']:
            if (TASK/name).exists():add(TASK/name,Path('evidence')/name,'additional_compact_numeric_or_integrity_QA')
        root_review=TASK/f'root-static-review-ep{ep}-02.json'
        if root_review.exists():add(root_review,Path('evidence')/root_review.name,'actual_root_static_image_review')
        add(root/'postprocessing-completion02.json',f'evidence/ep{ep}-postprocessing-completion02.json','completed_stage_provenance')
        add(root/'review02/summary.json',f'evidence/ep{ep}-video-integrity02.json','video_file_and_source_bindings')
        add(root/'candidate-egohand02/report.json',f'evidence/ep{ep}-as-executed-generated-native-report.json','historical_metadata_preserved')
        for name in ['report.json','correction-receipt.json']:
            add(TASK/'metadata-corrections02'/f'episode_{ep:06d}'/name,Path('reproduction/task/metadata-corrections02')/f'episode_{ep:06d}'/name,'canonical_metadata_sidecar')
    add(summary_path,'summary.json','complete_fresh_and_legacy_compact_metrics')
    add(integrity_path,'completed-integrity.json','combined_final_integrity_and_errata_bridge')
    add(packaging_path,'local-review-packages.json','verified_local_delivery_archives')
    if sum(item['bytes'] for item in planned.values())>MAX_TOTAL_BYTES:raise ValueError('Small-evidence byte budget exceeded')
    if digest(status_path)!=status_hash:raise AssertionError('Unrelated status page changed during preflight')
    preflight={'recorded_at':datetime.now(timezone.utc).isoformat(),'status':'passed_all_inputs_no_repository_copy',
        'source_recipe_sha256':digest(__file__),'destination':str(DEST),'file_count':len(planned),
        'total_bytes':sum(item['bytes'] for item in planned.values()),'maximum_file_bytes':max(item['bytes'] for item in planned.values()),
        'summary_sha256':digest(summary_path),'integrity_sha256':digest(integrity_path),'packaging_receipt_sha256':digest(packaging_path),
        'all_planned_targets_absent':True,'status_page_sha256':status_hash,
        'executed_vs_metadata_classifier_roles':classifier_roles,'additional_actual_executed_source_sets':source_set_roles,
        'metric_reference_coverage':metric_coverage,'repository_files_copied':0,'git_stage_commit_push_performed':False,
        'files':[{'path':relative,'source':str(item['source']),'bytes':item['bytes'],'sha256':item['sha256'],'category':item['category']}
                 for relative,item in sorted(planned.items())]}
    if args.preflight_only:
        report=task_json(args.preflight_report)
        write(report,preflight)
        print(json.dumps({'status':preflight['status'],'preflight_report':str(report),'preflight_sha256':digest(report),
                          'files':preflight['file_count'],'bytes':preflight['total_bytes'],'maximum_file_bytes':preflight['maximum_file_bytes'],
                          'repository_files_copied':0}),flush=True)
        return
    # All required inputs, hashes and target absence are validated before the first copy.
    copied=[]
    for relative,item in sorted(planned.items()):
        target=item['target'];target.parent.mkdir(parents=True,exist_ok=True)
        with item['source'].open('rb') as source,target.open('xb') as output:shutil.copyfileobj(source,output)
        if digest(target)!=item['sha256'] or digest(item['source'])!=item['sha256']:raise AssertionError('Source or copied bytes changed during staging')
        copied.append({'path':relative,'bytes':item['bytes'],'sha256':item['sha256'],'category':item['category']})
    if digest(status_path)!=status_hash:raise AssertionError('Unrelated status page changed during staging')
    inventory={'recorded_at':datetime.now(timezone.utc).isoformat(),'track':1,'authorized_episodes':[16,11],
        'files':copied,'file_count':len(copied),'total_bytes':sum(x['bytes'] for x in copied),
        'scope':'Exact copied small source/evidence bytes. Authored report/module documents, this inventory, weights, numeric parameter/media payloads and private GPU/app inventory are excluded.',
        'executed_vs_metadata_classifier_roles':classifier_roles,'additional_actual_executed_source_sets':source_set_roles,'metric_reference_coverage':metric_coverage,
        'status_page_sha256_before_and_after':status_hash,'all30_execution_authorized':False,
        'quality_or_contact_accepted':False,'official_scores_computed':False,'git_stage_commit_push_performed':False}
    write(inventory_path,inventory)
    print(json.dumps({'inventory':str(inventory_path),'inventory_sha256':digest(inventory_path),'files':len(copied),'bytes':inventory['total_bytes'],'status_page_unchanged':True}),flush=True)


if __name__=='__main__':main()
