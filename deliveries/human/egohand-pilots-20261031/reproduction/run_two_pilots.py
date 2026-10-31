"""Execute only the two authorized controlled EgoHand pilots sequentially."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time,sys
ROOT=Path(__file__).resolve().parent
PY=ROOT/'venv/Scripts/python.exe'
runner=ROOT/'run_controlled_egohand.py'
bench=json.loads((ROOT/'cpu-chunk-benchmark-receipt.json').read_text(encoding='utf-8'))
config=bench['selected'];source=runner.read_bytes();source_sha=hashlib.sha256(source).hexdigest()
snapshot=ROOT/'inference-runtime01';snapshot.mkdir(exist_ok=False)
(snapshot/'run_controlled_egohand.py').write_bytes(source)
env={'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1'}
records=[]
for episode in (16,11):
 assert runner.read_bytes()==source,'Runtime source changed after freezing'
 output=Path('J:/video-to-hoi/runs/egohand-pilots-20261031')/f'episode_{episode:06d}'/'egohand-raw-attempt01'
 assert not output.exists(),f'Immutable output already exists: {output}'
 command=[str(PY),str(runner),'--episode',str(episode),'--output',str(output),'--device','cpu','--dtype','float32','--threads',str(config['threads']),'--batch-size',str(config['batch_size']),'--frame-chunk-size','8','--with-smoothing']
 log=ROOT/f'ep{episode:02d}-inference-attempt01.log';start=time.monotonic()
 with log.open('xb') as stream:
  process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,**env},creationflags=subprocess.CREATE_NO_WINDOW)
  receipt={'episode':episode,'source_sha256':source_sha,'source_snapshot':str(snapshot/'run_controlled_egohand.py'),'command':command,'log':str(log),'pid':process.pid,'launcher_pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all30_execution_authorized':False,'configuration_selected_on_runtime_not_quality':True}
  (ROOT/f'ep{episode:02d}-inference-attempt01.started.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({'episode':episode,'pid':process.pid,'status':'started','output':str(output)}),flush=True)
  code=process.wait()
 receipt.update(returncode=code,wall_seconds=time.monotonic()-start,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (ROOT/f'ep{episode:02d}-inference-attempt01.finished.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');records.append(receipt)
 print(json.dumps({'episode':episode,'returncode':code,'wall_seconds':receipt['wall_seconds']}),flush=True)
 if code:raise SystemExit(code)
(ROOT/'two-pilot-inference-completion.json').write_text(json.dumps({'status':'completed','episodes':[16,11],'records':records,'all30_execution_started':False},indent=2),encoding='utf-8')
