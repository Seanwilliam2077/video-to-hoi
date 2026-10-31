"""Bind completed native A/B artifacts; prepare evaluation without running it."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import time

import numpy as np

TASK = Path(__file__).resolve().parent
PLAN_SHA = "82f808435a10cb60f082a562a782e72172d673105e8422e85b81414483aebdfb"
PREVIOUS = Path("J:/video-to-hoi/work/egohand-refined-pilots-20261031/ep16-evaluation-inputs02.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def check(row):
    path = Path(row["path"])
    assert path.is_file() and digest(path) == row["sha256"], path
    if "bytes" in row:
        assert path.stat().st_size == row["bytes"], path
    return path


def write_new(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    started = time.monotonic()
    destinations = [TASK / "evaluation-inputs01.json", TASK / "prepared-evaluation-job01.json"]
    assert not any(p.exists() for p in destinations), "Preserve existing manifests"
    plan_path = TASK / "ab-plan01.json"
    assert digest(plan_path) == PLAN_SHA
    plan = read(plan_path)
    assert plan["track"] == 1 and plan["episode"] == 16 and plan["frames"] == 360
    assert plan["all30_execution_authorized"] is False
    receipts = {arm: TASK / f"ab-plan01-{arm}.finished.json" for arm in ("A", "B")}
    while not all(p.exists() for p in receipts.values()):
        if time.monotonic() - started >= args.wait_seconds:
            raise ValueError("Both native fits have not completed; no evaluation manifest written")
        time.sleep(2)
    for arm, path in receipts.items():
        value = read(path)
        assert value["returncode"] == 0 and value["status"] == "fit_completed"
        assert value["plan_sha256"] == PLAN_SHA and value["command"] == plan["commands"][arm]
        assert value["frozen_inputs_and_sources_unchanged"]
    for row in plan["input_bindings"] + plan["fitter_source_bindings"]:
        check(row)
    previous = read(PREVIOUS)
    old = next(x for x in previous["episodes"] if x["episode"] == 16)
    assert old["video_sha256"] == plan["source_video_sha256"]
    def option(command, name):
        return Path(command[command.index(name) + 1])
    values = plan["commands"]["A"]
    manifest_path = option(values, "--manifest")
    manifest = read(manifest_path)
    assert manifest["video_sha256"] == plan["source_video_sha256"]
    video = Path(manifest["video"])
    assert "track_1" in video.parts
    baseline = check(old["baseline"])
    historic = next(x for x in old["variants"] if x["role"] == "fresh_mediapipe")
    check(historic)
    for name in ("legacy_mediapipe", "fresh_mediapipe"):
        check(old["evidence"][name])
        check(old["evidence"][name]["report"])
    check(old["original_bindings"]["old_active_hands"])
    native = {}
    for arm, role in (("A", "a"), ("B", "b")):
        root = Path(plan["outputs"][arm])
        path = root / "episode_000016_human.npz"
        entry = binding(path)
        entry.update(report=binding(root / "report.json"), support=binding(root / "active_hands.npy"),
                     weights=binding(root / "optimization_weights.npy"))
        report = read(root / "report.json")
        assert report["config"] == plan["frozen_config"]
        assert report["source_video_sha256"] == plan["source_video_sha256"]
        assert report["baseline_sha256"] == old["baseline"]["sha256"]
        assert report["observations_sha256"] == old["evidence"]["fresh_mediapipe"]["sha256"]
        assert report["heldout_frame_indices"] == list(range(0, 360, 5))
        native[role] = entry
        for name in ("vertices_m", "joints_3d", "faces", "frame_indices", "camera_intrinsics", "fps"):
            assert (root / "review_arrays" / (name + ".npy")).is_file()
        assert np.array_equal(np.load(root / "review_arrays/frame_indices.npy", allow_pickle=False), np.arange(360))
        assert np.array_equal(np.load(root / "review_arrays/camera_intrinsics.npy", allow_pickle=False),
                              np.asarray(manifest["camera_intrinsics"], np.float32))
    assert np.array_equal(np.load(native["a"]["support"]["path"], allow_pickle=False),
                          np.load(native["b"]["support"]["path"], allow_pickle=False))
    assert np.array_equal(np.load(native["a"]["weights"]["path"], allow_pickle=False),
                          np.load(native["b"]["weights"]["path"], allow_pickle=False))
    object_path = Path(plan["prepared_object"])
    phases_path = Path(plan["manual_phases"])
    criteria_path = TASK / "frozen-contact-ab-criteria01.json"
    spec = {"schema_version": "ep16_contact_ab_evaluation_inputs.v1",
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "track": 1, "episode": 16, "frames": 360, "authorized_episodes": [16],
        "all30_execution_authorized": False, "source_video": binding(video),
        "manifest": binding(manifest_path), "baseline": binding(baseline), **native,
        "historical_no_contact": binding(historic["path"]),
        "legacy_reference": binding(old["evidence"]["legacy_mediapipe"]["path"]),
        "fresh_reference": binding(old["evidence"]["fresh_mediapipe"]["path"]),
        "original_active_hands": binding(old["original_bindings"]["old_active_hands"]["path"]),
        "object": binding(object_path), "object_report": binding(object_path.parent / "report.json"),
        "phases": binding(phases_path), "criteria": binding(criteria_path),
        "fit_configuration_reference": binding(Path(historic["path"]).parent / "report.json"),
        "protocol": binding(plan_path),
        "completed_native_fit_receipts": {arm: binding(path) for arm, path in receipts.items()},
        "previous_evaluation_input_manifest": binding(PREVIOUS),
        "manual_phase_with_mutable_finger_controls": plan["manual_phase_with_mutable_finger_controls"],
        "manual_phase_without_mutable_finger_controls": plan["manual_phase_without_mutable_finger_controls"],
        "left_hand_limitation": "All129 usable carry frames have frozen left finger controls. This experiment cannot demonstrate recovered left finger articulation; any allowed arm motion is a separate block.",
        "quality_accepted": False, "official_score_available": False,
        "contact_model_training_started": False, "episode11_execution_started": False}
    assert digest(plan_path) == PLAN_SHA
    write_new(destinations[0], spec)
    tool_root = Path(plan["source_root"])
    source = tool_root / "tools/evaluate_contact_ab.py"
    assert source.is_file()
    job = {"recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "prepared_not_executed",
        "track": 1, "episode": 16, "input_manifest": binding(destinations[0]),
        "source": binding(source), "tool_root": str(tool_root),
        "command": ["J:/video-to-hoi/work/human-runtime/venv/Scripts/python.exe", "-B", str(source),
            "--input-manifest", str(destinations[0]), "--output", "J:/video-to-hoi/runs/ep16-contact-ab-20261031/evaluation01.json",
            "--tool-root", str(tool_root), "--threads", "1", "--model-batch", "8"],
        "cwd": str(tool_root), "all30_execution_authorized": False,
        "quality_accepted": False, "official_score_available": False,
        "actual_evaluation_execution_performed": False}
    write_new(destinations[1], job)
    print(json.dumps({"input_manifest": str(destinations[0]), "sha256": digest(destinations[0]),
                      "job": str(destinations[1]), "job_sha256": digest(destinations[1]),
                      "actual_evaluation_execution_performed": False}), flush=True)


if __name__ == "__main__":
    main()
