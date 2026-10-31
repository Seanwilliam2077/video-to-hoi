"""Run one immutable two-pilot stage with exclusive durable receipts."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


path = Path(sys.argv[1]).resolve()
job = json.loads(path.read_text(encoding="utf-8"))
if job.get("episode") not in (11, 16) or job.get("all30_execution_authorized") is not False:
    raise ValueError("Only the two explicitly authorized pilots are permitted")
if any(value in ("all", "--episodes") for value in job["command"]):
    raise ValueError("No all-30 or multi-episode command is permitted")
if path.with_suffix(".started.json").exists() or path.with_suffix(".finished.json").exists():
    raise ValueError("Preserve an already-started stage; inspect PID ownership and use an explicit recovery attempt")
for binding in job["input_bindings"]:
    with Path(binding["path"]).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != binding["sha256"]:
        raise ValueError(f"Stage input identity changed: {binding['path']}")
started, begin = dt.datetime.now(dt.timezone.utc).isoformat(), time.monotonic()
with Path(job["log"]).open("xb") as log:
    process = subprocess.Popen(job["command"], cwd=job["cwd"], stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    receipt = {**job, "pid": process.pid, "launcher_pid": os.getpid(), "started_at": started}
    write(path.with_suffix(".started.json"), receipt)
    print(json.dumps({"pid": process.pid, "job": str(path), "log": job["log"]}), flush=True)
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        code = process.wait()
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
receipt.update(returncode=code, wall_seconds=time.monotonic() - begin,
               finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
write(path.with_suffix(".finished.json"), receipt)
print(json.dumps({"returncode": code, "job": str(path), "wall_seconds": receipt["wall_seconds"]}), flush=True)
sys.exit(code)
