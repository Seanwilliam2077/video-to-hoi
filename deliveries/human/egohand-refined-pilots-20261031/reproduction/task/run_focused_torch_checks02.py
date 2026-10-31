"""Reuse existing Torch and development pytest read-only; no environment changes."""
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import datetime,hashlib,json,re,sys,time
ROOT=Path(__file__).resolve().parent
REPO=Path('C:/Users/arthasxu/.codex/worktrees/human-all30/video-to-hoi')
sys.path.insert(0,str(REPO/'src'))
sys.path.append('J:/video-to-hoi/.venv/Lib/site-packages')
import torch,pytest
torch.set_num_threads(1)
tests=['tests/test_egohand_adapter.py', 'tests/test_egohand_evaluation.py', 'tests/test_hand_crop_inputs.py', 'tests/test_hand_crop_router.py', 'tests/test_refined_egohand.py', 'tests/test_refined_egohand_evaluation.py', 'tests/test_hand_refinement.py', 'tests/test_human_hand_pilots.py', 'tests/test_human_result.py']
files=['tests/test_egohand_adapter.py', 'tests/test_egohand_evaluation.py', 'tests/test_hand_crop_inputs.py', 'tests/test_hand_crop_router.py', 'tests/test_refined_egohand.py', 'tests/test_refined_egohand_evaluation.py', 'tests/test_hand_refinement.py', 'tests/test_human_hand_pilots.py', 'tests/test_human_result.py', 'src/v2hoi/egohand_adapter.py', 'src/v2hoi/hand_crop_inputs.py', 'src/v2hoi/hand_crop_router.py', 'src/v2hoi/refined_egohand.py', 'src/v2hoi/hand_refinement.py', 'tools/adapt_refined_egohand.py', 'tools/evaluate_refined_egohand_pilots.py']
hashes=lambda:{name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in files}
before=hashes();begin=time.monotonic()
log=ROOT/'focused-torch-tests02.log'
with log.open('x',encoding='utf-8') as f,redirect_stdout(f),redirect_stderr(f):
    code=pytest.main(['-q',*[str(REPO/name) for name in tests]])
after=hashes();assert before==after
content=log.read_text();passed=re.search(r'(\d+) passed',content)
receipt={'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':int(code),
         'wall_seconds':time.monotonic()-begin,'tests':tests,'source_sha256_before':before,'source_sha256_after':after,
         'passed':int(passed.group(1)) if passed else None,'torch':torch.__version__,'pytest':pytest.__version__,
         'torch_module':torch.__file__,'pytest_module':pytest.__file__,'environment_packages_modified':False,
         'independent_synthetic_fixtures_only':True}
with (ROOT/'focused-torch-tests02.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps({'exit_code':int(code),'passed':receipt['passed'],'seconds':receipt['wall_seconds']}))
raise SystemExit(code)
