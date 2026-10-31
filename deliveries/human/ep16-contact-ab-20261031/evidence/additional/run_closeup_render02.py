from pathlib import Path
import datetime,hashlib,json,subprocess,time,os
T=Path(__file__).resolve().parent
R=Path(r"J:/video-to-hoi/runs/ep16-contact-ab-20261031")
PY=r"J:/video-to-hoi/work/human-runtime/venv/Scripts/python.exe"
def read(p):return json.loads(p.read_text())
def digest(p):
 with p.open("rb") as s:return hashlib.file_digest(s,"sha256").hexdigest()
def write(p,v):
 with p.open("x") as s:json.dump(v,s,indent=2);s.write("\n")
started=time.monotonic()
while not (T/"evaluation02.finished.json").exists():
 if time.monotonic()-started>1200:raise TimeoutError("Evaluation02 completion receipt not available")
 time.sleep(3)
assert read(T/"evaluation02.finished.json")["returncode"]==0
assert not (R/"review-closeup02").exists()
snap=T/"closeup-runtime02/snapshot.json";source=read(snap)
for x in source["files"]:assert digest(snap.parent/x["path"])==x["sha256"]
cmd=[PY,"-B",str(T/"closeup-runtime02/tools/render_contact_ab.py"),"--input-manifest",str(T/"evaluation-inputs02.json"),"--evaluation",str(R/"evaluation02.json"),"--output",str(R/"review-closeup02"),"--workers","4","--size","512"]
record={"recorded_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"wrapper_pid":os.getpid(),"command":cmd,"cwd":str(T/"closeup-runtime02"),"input_manifest_sha256":digest(T/"evaluation-inputs02.json"),"evaluation_sha256":digest(R/"evaluation02.json"),"render_snapshot_sha256":digest(snap),"renderer_sha256":digest(T/"closeup-runtime02/tools/render_contact_ab.py")}
write(T/"closeup-render02.started.json",record)
tick=time.monotonic()
with (T/"closeup-render02.log").open("xb") as stream:
 p=subprocess.Popen(cmd,cwd=record["cwd"],stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
 write(T/"closeup-render02.child-started.json",{**record,"child_pid":p.pid})
 rc=p.wait()
for x in source["files"]:assert digest(snap.parent/x["path"])==x["sha256"]
write(T/"closeup-render02.finished.json",{**record,"child_pid":p.pid,"returncode":rc,"wall_seconds":time.monotonic()-tick,"frozen_sources_unchanged":True})
print(json.dumps({"render_returncode":rc,"output":str(R/"review-closeup02"),"seconds":time.monotonic()-tick}),flush=True)
raise SystemExit(rc)
