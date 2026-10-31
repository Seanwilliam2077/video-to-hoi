"""Run one immutable, logged local pilot command; never enqueue the all-30 batch."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

path = Path(sys.argv[1]).resolve()
job = json.loads(path.read_text(encoding="utf-8"))
if job.get("episode") not in (11, 16):
    raise ValueError("This review round is authorized only for episodes 11 and 16")
if any(value in ("all", "--episodes") for value in job["command"]):
    raise ValueError("Batch execution requires a later explicit user approval")
for item in job.get("input_bindings", []):
    source = Path(item["path"])
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != item["sha256"]:
        raise ValueError(f"Input identity changed: {source}")
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
begin = time.monotonic()
log = Path(job["log"])
with log.open("xb") as stream:
    process = subprocess.Popen(job["command"], cwd=job["cwd"], stdout=stream,
        stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1"}, creationflags=subprocess.CREATE_NO_WINDOW)
    receipt = {**job, "pid": process.pid, "launcher_pid": os.getpid(), "started_at": started}
    path.with_suffix(".started.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"pid": process.pid, "log": str(log)}), flush=True)
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        code = process.wait()
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
receipt.update(returncode=code, wall_seconds=time.monotonic() - begin,
    finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
path.with_suffix(".finished.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps({"returncode": code, "wall_seconds": receipt["wall_seconds"]}), flush=True)
sys.exit(code)
