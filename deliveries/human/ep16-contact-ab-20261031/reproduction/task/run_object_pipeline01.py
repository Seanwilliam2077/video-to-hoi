"""Bounded local CPU object preparation with immutable stage receipts."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,time,os
TASK=Path(__file__).resolve().parent
RUNTIME=TASK/'object-runtime01'
RESULT=Path('J:/video-to-hoi/runs/ep16-contact-ab-20261031')
PY='J:/video-to-hoi/work/human-runtime/venv/Scripts/python.exe'

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    with path.open('x',encoding='utf-8') as out:json.dump(value,out,indent=2);out.write('\n')
def main():
    sources={p.name:digest(p) for p in RUNTIME.iterdir() if p.is_file()}
    write(TASK/'object-runtime-source-pin01.json',{'recorded_at':now(),'files':sources,'runner_sha256':digest(Path(__file__))})
    stages=[('object-initial01',[PY,'-B',str(RUNTIME/'prepare_object_initial01.py'),'--output',str(RESULT/'object-initial01')]),
        ('object-temporal01',[PY,'-B',str(RUNTIME/'temporal_object_stronger_rotation.py'),'--source',str(RESULT/'object-initial01'),'--output',str(RESULT/'object-final01'),'--nfev','45']),
        ('object-validation01',[PY,'-B',str(RUNTIME/'validate_temporal_candidate.py'),'--candidate',str(RESULT/'object-final01')]),
        ('object-contact-export01',[PY,'-B',str(RUNTIME/'prepare_contact_mesh.py'),'--candidate',str(RESULT/'object-final01')]),
        ('object-samples01',[PY,'-B',str(RUNTIME/'render_object_samples01.py'),'--candidate',str(RESULT/'object-final01')])]
    completed=[]
    for name,command in stages:
        start=time.monotonic();log=TASK/f'{name}.log'
        with log.open('xb') as output:
            proc=subprocess.Popen(command,cwd=RUNTIME,stdout=output,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},creationflags=subprocess.CREATE_NO_WINDOW)
            write(TASK/f'{name}.started.json',{'recorded_at':now(),'pid':proc.pid,'command':command,'cwd':str(RUNTIME),'source_sha256':sources,'log':str(log),'track':1,'episode':16})
            code=proc.wait()
        receipt={'recorded_at':now(),'returncode':code,'wall_seconds':time.monotonic()-start,'log_sha256':digest(log),'sources_unchanged':sources=={p.name:digest(p) for p in RUNTIME.iterdir() if p.is_file()}}
        write(TASK/f'{name}.finished.json',receipt);print(json.dumps({'stage':name,**receipt}),flush=True)
        if code or not receipt['sources_unchanged']:raise SystemExit(code or 2)
        completed.append(name)
    write(TASK/'object-pipeline-completed01.json',{'recorded_at':now(),'completed_stages':completed,'contact_ready':str(RESULT/'object-final01/contact_ready.npz'),'contact_ready_sha256':digest(RESULT/'object-final01/contact_ready.npz'),'quality_accepted':False,'official_score_available':False,'all30_execution_authorized':False})

if __name__=='__main__':main()
