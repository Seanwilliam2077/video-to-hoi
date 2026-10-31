from pathlib import Path
import hashlib,json,datetime,time
T=Path(__file__).resolve().parent
R=Path(r"J:/video-to-hoi/runs/ep16-contact-ab-20261031")
def digest(p):
 with p.open("rb") as s:return hashlib.file_digest(s,"sha256").hexdigest()
def read(p):return json.loads(p.read_text())
tick=time.monotonic()
while not (T/"closeup-render02.finished.json").exists():
 if time.monotonic()-tick>1200:raise TimeoutError("Closeup completion missing")
 time.sleep(3)
assert read(T/"closeup-render02.finished.json")["returncode"]==0
assert read(T/"render02.finished.json")["returncode"]==0
D=T/"original-review-indexes02";D.mkdir()
note='<p><strong>Interpretation:</strong> Manual attraction OFF can mean either an excluded manual phase or an unreliable object-pose frame. It does not label physical release. Only129 of176 carry frames pass the object gate. Left carry finger controls remain frozen in both methods.</p><p>The paired experiment did not pass all frozen motion/skin checks. No contact-model training, EP11 extension or thirty-video run follows from this result.</p>'
rows=[]
for folder in ["review02","review-closeup02"]:
 root=R/folder;summary=read(root/"summary.json");p=root/"index.html";raw=p.read_bytes();old=digest(p)
 with (D/(folder+".html")).open("xb") as f:f.write(raw)
 text=raw.decode();marker='<p><a href="summary.json">';assert text.count(marker)==1
 with p.open("w",encoding="utf-8",newline="") as f:f.write(text.replace(marker,note+marker))
 for v in summary["videos"]:assert digest(root/v["file"])==v["sha256"]
 rows.append({"directory":str(root),"original_index_sha256":old,"current_index_sha256":digest(p),"original_backup":str(D/(folder+".html")),"all_video_bytes_unchanged":True})
result={"recorded_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source_script_sha256":digest(Path(__file__)),"operation":"Add English interpretation paragraphs to the two display indexes only; preserve original indexes and every video/geometry/evaluation byte.","outputs":rows}
with (T/"review-index-clarification02.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
print(json.dumps({"indexes_clarified":2,"all_videos_unchanged":True}),flush=True)
