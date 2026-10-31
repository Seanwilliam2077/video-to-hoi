"""Copy an exact allowlist of completed pilot recipes and evidence into human."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

TASK = Path(__file__).resolve().parent
REPO = Path('C:/Users/arthasxu/.codex/worktrees/human-all30/video-to-hoi').resolve()
DEST = (REPO / 'deliveries/human/egohand-pilots-20261031').resolve()
RESULTS = Path('J:/video-to-hoi/runs/egohand-pilots-20261031').resolve()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip() == 'human'
    assert DEST.is_relative_to(REPO / 'deliveries/human')
    status_path = REPO / 'docs/status.html'
    status_hash = digest(status_path)
    assert status_hash == read(TASK / 'task-start.json')['preexisting_unrelated_status_sha256']
    for ep in (16, 11):
        for variant in ('raw', 'smoothed'):
            assert read(TASK / f'ep{ep}-fit-{variant}01.finished.json')['returncode'] == 0
        assert read(RESULTS / f'episode_{ep:06d}/review-raw01/summary.json')['frames'] == (360 if ep == 16 else 877)
    copied = []

    def copy(source, relative):
        source = Path(source).resolve()
        target = (DEST / relative).resolve()
        assert source.is_file() and target.is_relative_to(DEST) and not source.is_symlink()
        source_hash = digest(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert digest(target) == source_hash, 'Preserve conflicting evidence'
        else:
            shutil.copyfile(source, target)
        assert digest(target) == source_hash
        relative_name = target.relative_to(DEST).as_posix()
        previous = next((item for item in copied if item['path'] == relative_name), None)
        if previous is not None:
            assert previous['sha256'] == source_hash
            return
        copied.append({'path': relative_name,
                       'bytes': target.stat().st_size, 'sha256': source_hash})

    plan = read(TASK / 'reproduction-bundle-plan.json')
    for row in plan['files']:
        source = Path(row['absolute_path']).resolve()
        assert source.is_relative_to(TASK) and digest(source) == row['sha256']
        relative = Path(row['relative_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        copy(source, Path('reproduction') / relative)
    scripts = ['run_job.py', 'prepare_after_inference.py', 'run_postprocessing.py',
               'render_egohand_comparison.py', 'grasp_motion_diagnostics.py',
               'summarize_completed_pilots.py', 'package_local_reviews.py',
               'stage_delivery_evidence.py']
    for name in scripts:
        copy(TASK / name, Path('reproduction') / name)
    copy(TASK / 'weights-receipt.json', 'reproduction/weights-receipt.json')
    snapshot = read(TASK / 'runtime01/snapshot.json')
    for row in snapshot['files']:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        source = TASK / 'runtime01' / relative
        assert digest(source) == row['sha256']
        copy(source, Path('reproduction/runtime01') / relative)
    copy(TASK / 'runtime01/snapshot.json', 'reproduction/runtime01/snapshot.json')
    evidence = ['comparison-conditions-and-preservation01.json',
                'skinning-backend-parity.json', 'egohand-domain-smoothing-audit01.json',
                'ep16-motion-statistic-reconciliation01.json',
                'ep16-manual-visual-review01.json', 'ep16-independent-static-review01.json',
                'ep11-manual-visual-review01.json', 'ep11-opening-motion-numeric-review01.json',
                'ep11-independent-static-review01.json', 'completed-integrity-final01.json',
                'input-preprocessing-audit-final01.json',
                'workflow-python-tests01.json', 'workflow-node01.json',
                'workflow-registry01.json', 'workflow-catalog01.json',
                'focused-torch-tests02.json', 'source-checkout-attempt01-disposition.json']
    for name in evidence:
        copy(TASK / name, Path('evidence') / name)
    copy(TASK / 'runtime01/snapshot.json', 'evidence/native-runtime-snapshot.json')
    copy(TASK / 'completed-summary01.json', 'summary.json')
    copy(TASK / 'local-delivery01/packaging-receipt.json', 'local-review-packages.json')
    for ep in (16, 11):
        for name in ('review-windows.json',):
            copy(TASK / f'ep{ep}-{name}', f'evidence/ep{ep}-{name}')
        copy(RESULTS / f'episode_{ep:06d}/review-raw01/summary.json', f'evidence/ep{ep}-video-integrity.json')
        copy(TASK / f'ep{ep}-render-raw01.finished.json', f'evidence/ep{ep}-render-finished.json')
        for variant in ('raw', 'smoothed'):
            for stage in ('fit', 'evaluation'):
                name = f'ep{ep}-{stage}-{variant}01.finished.json'
                copy(TASK / name, Path('evidence') / name)
    inventory_path = DEST / 'evidence-inventory.json'
    inventory = {'recorded_at': datetime.now(timezone.utc).isoformat(),
                 'all30_execution_authorized': False,
                 'scope': 'Exact copied source/evidence bytes; inventory excludes itself and authored Markdown.',
                 'files': copied}
    with inventory_path.open('x', encoding='utf-8') as stream:
        json.dump(inventory, stream, indent=2)
        stream.write('\n')
    assert digest(status_path) == status_hash
    print(json.dumps({'files': len(copied), 'inventory': str(inventory_path), 'status_page_unchanged': True}))


if __name__ == '__main__':
    main()
