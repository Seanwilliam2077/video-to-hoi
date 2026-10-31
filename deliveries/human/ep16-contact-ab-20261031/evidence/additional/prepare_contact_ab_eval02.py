"""Bind completed revision02 pairs with an explicit anchor-only native prior."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import time

import numpy as np

TASK = Path(__file__).resolve().parent
EVAL_SHA = "94a60410753c5fd41bbb44e97c26bc2938b3fa67f6e3c3c1022c68f7dd2a92fb"
FIRST_INPUT_SHA = "6e8e1c8a09cb357e1b224add02eca143edfdb2668de5dfd05f136fdf8a636fc4"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def check(row):
    path = Path(row["path"])
    assert path.is_file() and digest(path) == row["sha256"], path
    if "bytes" in row:
        assert path.stat().st_size == row["bytes"]


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--evaluation-runtime", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    source = args.evaluation_runtime / "tools/evaluate_contact_ab.py"
    assert source.is_file() and digest(source) == EVAL_SHA
    snapshot = read(args.evaluation_runtime / "snapshot.json")
    for row in snapshot["files"]:
        check({**row, "path": str(args.evaluation_runtime / row["path"])})
    assert digest(args.plan) == args.plan_sha256
    plan = read(args.plan)
    assert plan["track"] == 1 and plan["episode"] == 16 and plan["frames"] == 360
    assert plan["all30_execution_authorized"] is False
    old_path = TASK / "evaluation-inputs01.json"
    assert digest(old_path) == FIRST_INPUT_SHA
    old = read(old_path)
    anchor = plan.get("contact_anchor_reference")
    assert anchor and anchor["sha256"] == old["historical_no_contact"]["sha256"]
    check(anchor)
    destination, job_path = TASK / "evaluation-inputs02.json", TASK / "prepared-evaluation-job03.json"
    assert not destination.exists() and not job_path.exists()
    receipts = {arm: args.plan.with_name(args.plan.stem + f"-{arm}.finished.json") for arm in ("A", "B")}
    tick = time.monotonic()
    while not all(path.exists() for path in receipts.values()):
        if time.monotonic() - tick >= args.wait_seconds:
            raise ValueError("Revision02 native pairs are incomplete; no manifest written")
        time.sleep(2)
    for arm, path in receipts.items():
        value = read(path)
        assert value["returncode"] == 0 and value["status"] == "fit_completed"
        assert value["plan_sha256"] == args.plan_sha256 and value["command"] == plan["commands"][arm]
        assert value["frozen_inputs_and_sources_unchanged"]
    for row in plan["input_bindings"] + plan["fitter_source_bindings"]:
        check(row)
    spec = dict(old)
    spec.update(schema_version="ep16_contact_ab_evaluation_inputs.v2",
                recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                protocol=binding(args.plan), contact_anchor_reference=anchor,
                completed_native_fit_receipts={arm: binding(path) for arm, path in receipts.items()},
                prior_controlled_attempt={"input_manifest": binding(old_path),
                    "evaluation": binding(Path("J:/video-to-hoi/runs/ep16-contact-ab-20261031/evaluation01.json")),
                    "retained_reason": "Actual first-run anchors lay only on immutable left fingers; revised anchor-selection prior is explicit and pre-fit frozen."})
    for arm, role in (("A", "a"), ("B", "b")):
        root = Path(plan["outputs"][arm])
        entry = binding(root / "episode_000016_human.npz")
        entry.update(report=binding(root / "report.json"), support=binding(root / "active_hands.npy"),
                     weights=binding(root / "optimization_weights.npy"))
        report = read(root / "report.json")
        assert report["config"] == plan["frozen_config"]
        assert report["baseline_sha256"] == old["baseline"]["sha256"]
        assert report["observations_sha256"] == old["fresh_reference"]["sha256"]
        assert report["source_video_sha256"] == plan["source_video_sha256"]
        assert report["heldout_frame_indices"] == list(range(0, 360, 5))
        if role == "a":
            assert report["contact_provenance"] is None
        else:
            actual = report["contact_provenance"]["anchor_reference"]
            assert actual["sha256"] == anchor["sha256"] and actual["same_optimizer_initialization"] is True
        spec[role] = entry
    for name in ("support", "weights"):
        assert np.array_equal(np.load(spec["a"][name]["path"], allow_pickle=False),
                              np.load(spec["b"][name]["path"], allow_pickle=False))
    for key in ("baseline", "historical_no_contact", "legacy_reference", "fresh_reference", "object", "object_report", "phases", "criteria"):
        check(spec[key])
    assert digest(args.plan) == args.plan_sha256 and digest(source) == EVAL_SHA
    write(destination, spec)
    output = "J:/video-to-hoi/runs/ep16-contact-ab-20261031/evaluation02.json"
    job = {"recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "prepared_not_executed", "track": 1, "episode": 16,
        "input_manifest": binding(destination), "source": binding(source),
        "runtime_snapshot": binding(args.evaluation_runtime / "snapshot.json"),
        "tool_root": str(args.evaluation_runtime), "output": output,
        "command": ["J:/video-to-hoi/work/human-runtime/venv/Scripts/python.exe", "-B", str(source),
            "--input-manifest", str(destination), "--output", output,
            "--tool-root", str(args.evaluation_runtime), "--threads", "1", "--model-batch", "8"],
        "all30_execution_authorized": False, "quality_accepted": False}
    write(job_path, job)
    print(json.dumps({"manifest": str(destination), "manifest_sha256": digest(destination),
                      "job": str(job_path), "job_sha256": digest(job_path),
                      "model_execution_performed": False}), flush=True)


if __name__ == "__main__":
    main()
