from pathlib import Path
import datetime,hashlib,json,os,re,subprocess,time
T=Path(__file__).resolve().parent
R=Path(r"C:/Users/arthasxu/.codex/worktrees/human-all30/video-to-hoi")
PY=Path(r"J:/video-to-hoi/.venv/Scripts/python.exe")
FILES=[".github/workflows/tests.yml","src/v2hoi/contact_review.py","tools/refine_contact_ab.py","tools/evaluate_contact_ab.py","tools/render_contact_ab.py","tests/test_contact_ab.py","tests/test_contact_ab_evaluation.py","tests/test_contact_ab_render.py","tests/test_contact_anchor_reference.py","src/v2hoi/hand_contact.py","src/v2hoi/hand_refinement.py","tools/refine_human_hands.py"]
def hashes():return {n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in FILES}
def run(name,args):
 before=hashes();started=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.monotonic()
 with (T/(name+".log")).open("xb") as out:
  result=subprocess.run(args,cwd=R,stdout=out,stderr=subprocess.STDOUT,env={**os.environ,"PYTHONPATH":str(R/"src"),"PYTHONDONTWRITEBYTECODE":"1"},creationflags=subprocess.CREATE_NO_WINDOW)
 after=hashes();assert before==after
 record={"started_at":started,"command":args,"returncode":result.returncode,"seconds":time.monotonic()-tick,"source_sha256_before":before,"source_sha256_after":after}
 with (T/(name+".json")).open("x") as stream:json.dump(record,stream,indent=2);stream.write("\n")
 print(json.dumps({"check":name,"returncode":result.returncode,"seconds":record["seconds"]}),flush=True)
 if result.returncode:raise SystemExit(result.returncode)
tests=re.findall(r"^\s+(tests/test_\w+\.py)\s*$",(R/".github/workflows/tests.yml").read_text(),re.MULTILINE)
assert len(tests)>40 and len(tests)==len(set(tests))
run("workflow-python02",[str(PY),"-B","-m","pytest","-q",*tests])
run("workflow-registry02",[str(PY),"-B","tools/progress_board.py","validate"])
run("workflow-catalog02",[str(PY),"-B","tools/build_delivery_catalog.py","--check"])
run("workflow-node02",["node","--test","tests/test_progress_board.cjs","tests/test_delivery_view.cjs","tests/test_progress_app.cjs"])
