"""Run one source-bound refined-pilot command with durable process receipts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
p=Path(sys.argv[1]).resolve();job=json.loads(p.read_text(encoding='utf-8'))
if job['episode'] not in (11,16) or job['all30_execution_authorized'] is not False:raise ValueError('Wrong pilot scope')
for key in ('started','finished'):
 if p.with_suffix(f'.{key}.json').exists():raise ValueError('Preserve existing attempts and receipts')
for item in job['source_bindings']:
 if sha(item['path'])!=item['sha256']:raise ValueError(f"Frozen source/input changed: {item['path']}")
receipt={**job,'started_at':now(),'launcher_pid':os.getpid()};begin=time.perf_counter()
with Path(job['log']).open('xb') as log:
 process=subprocess.Popen(job['command'],cwd=p.parent,env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
 receipt['pid']=process.pid;p.with_suffix('.started.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'started':True,'pid':process.pid,'log':job['log']}),flush=True)
 code=process.wait()
receipt.update(finished_at=now(),returncode=code,wall_seconds=time.perf_counter()-begin)
p.with_suffix('.finished.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'finished':True,'returncode':code,'wall_seconds':receipt['wall_seconds']}),flush=True)
sys.exit(code)
