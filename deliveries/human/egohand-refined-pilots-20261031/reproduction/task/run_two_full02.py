"""Run only the two frozen CPU/cache02 inference jobs, preserving every receipt."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
T=Path(__file__).resolve().parent
expected={'run_refined_egohand02.py':'91729499f3454489e9611e4a51498e8c4886d1421acf6df2f367ac53533c4ec4','raw_cache02.py':'e415747f01bfc190ae811bc0452ecc0b1fff1996099a998ced04b7282e5e67ef'}
for name,sha in expected.items():
 if hashlib.sha256((T/name).read_bytes()).hexdigest()!=sha:raise ValueError('Frozen inference source changed')
records=[]
for ep in [16,11]:
 job=T/f'ep{ep}-inference-attempt02.json'
 code=subprocess.run([sys.executable,'-B',str(T/'run_runtime_job.py'),str(job)],cwd=T).returncode
 receipt=json.loads(job.with_suffix('.finished.json').read_text())
 records.append(receipt)
 if code or receipt['returncode']:
  print(json.dumps({'episode':ep,'failed':True,'receipt':str(job.with_suffix('.finished.json'))}),flush=True);sys.exit(1)
with (T/'two-pilot-inference-completion02.json').open('x',encoding='utf-8') as f:json.dump({'status':'completed','recorded_at':datetime.now(timezone.utc).isoformat(),'episodes':[16,11],'records':records,'all30_execution_started':False},f,indent=2);f.write('\n')
