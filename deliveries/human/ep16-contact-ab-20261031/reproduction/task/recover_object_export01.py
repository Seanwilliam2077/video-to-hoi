"""Recover only optional geometry export in an existing verified local environment."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,time,os
T=Path(__file__).resolve().parent;W=T/'object-runtime01'
R=Path('J:/video-to-hoi/runs/ep16-contact-ab-20261031/object-final01')
PY='J:/video-to-hoi/work/human-hand-round2/expert/venv/Scripts/python.exe'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def main():
    pin=json.loads((T/'object-runtime-source-pin01.json').read_text())['files']
    assert pin=={p.name:sha(p) for p in W.iterdir() if p.is_file()}
    assert json.loads((R/'validation.json').read_text())['weak_contact_experiment_guards_passed']
    assert 'ModuleNotFoundError: No module named' in (T/'object-contact-export01.log').read_text()
    assert not (R/'contact_ready.npz').exists()
    protected={str(R/p):sha(R/p) for p in ('object_candidate.npz','report.json','validation.json','feature_evidence.npz')}
    for name,script in [('object-contact-export-recovery01','prepare_contact_mesh.py'),('object-samples-recovery01','render_object_samples01.py')]:
        command=[PY,'-B',str(W/script),'--candidate',str(R)];start=time.monotonic();log=T/f'{name}.log'
        with log.open('xb') as f:
            proc=subprocess.Popen(command,cwd=W,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            write(T/f'{name}.started.json',{'recorded_at':now(),'command':command,'pid':proc.pid,'protected_inputs':protected,'source_pin_sha256':sha(T/'object-runtime-source-pin01.json'),'reason':'Existing human-runtime lacks optional trimesh; expert environment verified to contain it. No inference/pose numerical solve rerun.'})
            code=proc.wait()
        receipt={'recorded_at':now(),'returncode':code,'wall_seconds':time.monotonic()-start,'log_sha256':sha(log),'protected_inputs_unchanged':all(sha(p)==h for p,h in protected.items()),'runtime_sources_unchanged':pin=={p.name:sha(p) for p in W.iterdir() if p.is_file()}}
        write(T/f'{name}.finished.json',receipt);print(json.dumps({'stage':name,**receipt}),flush=True)
        if code or not receipt['protected_inputs_unchanged'] or not receipt['runtime_sources_unchanged']:raise SystemExit(code or 2)
    write(T/'object-pipeline-completed01.json',{'recorded_at':now(),'status':'completed_with_environment_only_export_recovery','original_export_failure_preserved':str(T/'object-contact-export01.finished.json'),'contact_ready':str(R/'contact_ready.npz'),'contact_ready_sha256':sha(R/'contact_ready.npz'),'geometry_ply_sha256':sha(R/'geometry.ply'),'object_report_sha256':sha(R/'report.json'),'validation_sha256':sha(R/'validation.json'),'source_pin_sha256':sha(T/'object-runtime-source-pin01.json'),'sample_review':str(R/'sample-review/summary.json'),'quality_accepted':False,'contact_or_scale_accepted':False,'official_score_available':False,'all30_execution_authorized':False})

if __name__=='__main__':main()
