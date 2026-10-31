"""Run only the checked-in explicit workflow checks and preserve source bindings."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parent
REPO=Path('C:/Users/arthasxu/.codex/worktrees/human-all30/video-to-hoi')
PY=Path('J:/video-to-hoi/.venv/Scripts/python.exe')
FILES=['src/v2hoi/hand_crop_inputs.py', 'src/v2hoi/hand_crop_router.py', 'src/v2hoi/refined_egohand.py', 'tools/human_hand/observations.py', 'tools/prepare_refined_egohand_crops.py', 'tools/route_refined_hand_crops.py', 'tools/adapt_refined_egohand.py', 'tools/evaluate_refined_egohand_pilots.py', 'tools/refine_human_hands.py', 'tests/test_hand_target_provenance.py', 'tests/test_hand_crop_inputs.py', 'tests/test_hand_crop_router.py', 'tests/test_refined_egohand.py', 'tests/test_refined_egohand_evaluation.py', '.github/workflows/tests.yml']


def hashes():
    return {name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in FILES}


def run(name,command):
    log=ROOT/f'{name}.log'
    receipt={'command':list(map(str,command)),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'source_sha256_before':hashes(),'PYTHONPATH':str(REPO/'src')}
    start=time.monotonic()
    with log.open('xb') as stream:
        proc=subprocess.run(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'PYTHONPATH':str(REPO/'src'),'PYTHONDONTWRITEBYTECODE':'1'},
            creationflags=subprocess.CREATE_NO_WINDOW)
    receipt.update(returncode=proc.returncode,wall_seconds=time.monotonic()-start,source_sha256_after=hashes())
    assert receipt['source_sha256_before']==receipt['source_sha256_after'],'Sources changed during verification'
    with (ROOT/f'{name}.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps({'name':name,'returncode':proc.returncode,'seconds':receipt['wall_seconds']}),flush=True)
    if proc.returncode: raise SystemExit(proc.returncode)


if __name__=='__main__':
    workflow=(REPO/'.github/workflows/tests.yml').read_text()
    tests=re.findall(r'^\s+(tests/test_\w+\.py)\s*$',workflow,re.MULTILINE)
    assert len(tests)>35 and len(tests)==len(set(tests))
    run('workflow-python-tests03',[str(PY),'-B','-m','pytest','-q',*tests])
    run('workflow-registry03',[str(PY),'-B','tools/progress_board.py','validate'])
    run('workflow-catalog03',[str(PY),'-B','tools/build_delivery_catalog.py','--check'])
    run('workflow-node03',['node','--test','tests/test_progress_board.cjs','tests/test_delivery_view.cjs','tests/test_progress_app.cjs'])
