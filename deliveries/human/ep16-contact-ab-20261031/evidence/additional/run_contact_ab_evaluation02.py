"""Run one frozen independent evaluation with durable ownership/source receipts."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import subprocess
import time

TASK = Path(__file__).resolve().parent
JOB = TASK / "prepared-evaluation-job03.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    job = read(JOB)
    root = Path(job["tool_root"])
    snap = root / "snapshot.json"
    snapshot = read(snap)
    def verify():
        assert digest(snap) == job["runtime_snapshot"]["sha256"]
        for row in snapshot["files"]:
            path = root / row["path"]
            assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
        for name in ("source", "input_manifest"):
            assert digest(job[name]["path"]) == job[name]["sha256"]
        assert digest(JOB) == job_hash
    job_hash = digest(JOB)
    verify()
    output = Path(job["output"])
    started, child_started, finished, log = [TASK / ("evaluation02" + suffix) for suffix in (".started.json", ".child-started.json", ".finished.json", ".log")]
    assert not any(p.exists() for p in (output, started, child_started, finished, log))
    record = {"recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "track": 1, "episode": 16, "job": str(JOB), "job_sha256": job_hash,
        "command": job["command"], "cwd": str(root), "wrapper_pid": os.getpid(),
        "source_sha256": job["source"]["sha256"],
        "input_manifest_sha256": job["input_manifest"]["sha256"],
        "wrapper_source_sha256": digest(Path(__file__)),
        "runtime_snapshot_sha256": job["runtime_snapshot"]["sha256"],
        "all30_execution_authorized": False, "quality_accepted": False}
    write(started, record)
    tick = time.monotonic()
    with log.open("xb") as stream:
        child = subprocess.Popen(job["command"], cwd=root, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        write(child_started, {**record, "child_pid": child.pid,
                            "child_started_at": dt.datetime.now(dt.timezone.utc).isoformat()})
        rc = child.wait()
    verify()
    record.update(returncode=rc, child_pid=child.pid,
        finished_at=dt.datetime.now(dt.timezone.utc).isoformat(), wall_seconds=time.monotonic() - tick,
        input_and_runtime_source_bytes_unchanged=True)
    if rc == 0:
        result = read(output)
        assert result["frames"] == 360 and result["input_manifest_sha256"] == record["input_manifest_sha256"]
        assert all(result["native_decoder"][role]["passed"] for role in ("baseline", "a", "b"))
        record.update(output=str(output), output_sha256=digest(output), status="completed_independent_evaluation")
    else:
        record["status"] = "evaluation_failed_preserved_attempt"
    write(finished, record)
    print(json.dumps(record), flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
