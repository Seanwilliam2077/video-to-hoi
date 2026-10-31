"""Continue only the two authorized EgoHand pilots after complete inference.

This is a task-local long-running job, not a scheduler or an all-30 launcher.
Every stage preserves its original command, log, inputs, PID and return receipt.
"""
from pathlib import Path
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

TASK = Path(__file__).resolve().parent
RESULTS = Path("J:/video-to-hoi/runs/egohand-pilots-20261031")
PYTHON = Path("J:/video-to-hoi/work/human-runtime/venv/Scripts/python.exe")
RUNTIME = TASK / "runtime01"
STATE = TASK / "postprocessing-progress01.json"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def state(value):
    value = {"recorded_at": now(), "launcher_pid": os.getpid(), "all30_execution_authorized": False,
             "automatic_promotion": False, **value}
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(STATE)


def run_job(job_path):
    finished = job_path.with_suffix(".finished.json")
    if finished.exists():
        receipt = read(finished)
        if receipt.get("returncode") != 0:
            raise RuntimeError(f"Preserve failed attempt and inspect {finished}")
        return receipt
    if job_path.with_suffix(".started.json").exists():
        raise RuntimeError(f"Stage already started without finish; inspect PID ownership: {job_path}")
    result = subprocess.run([str(PYTHON), "-B", str(TASK / "run_job.py"), str(job_path)], cwd=TASK)
    if result.returncode or not finished.exists():
        raise RuntimeError(f"Stage did not complete successfully: {job_path}")
    receipt = read(finished)
    if receipt["returncode"] != 0:
        raise RuntimeError(f"Stage failed: {job_path}")
    return receipt


def make_job(path, episode, command, bindings):
    if path.exists():
        raise ValueError(f"Preserve existing job definition: {path}")
    write(path, {"episode": episode, "command": [str(v) for v in command], "cwd": str(RUNTIME),
                 "log": str(path.with_suffix(".log")),
                 "input_bindings": [{"path": str(p), "sha256": digest(p)} for p in bindings],
                 "all30_execution_authorized": False, "automatic_promotion": False})


def episode(episode):
    root = RESULTS / f"episode_{episode:06d}"
    prediction = root / "egohand-raw-attempt01"
    receipt_path = prediction / "prediction-receipt.json"
    state({"episode": episode, "status": "waiting_for_complete_original_frame_inference", "receipt": str(receipt_path)})
    while not receipt_path.exists():
        failed = TASK / f"ep{episode}-inference-attempt01.finished.json"
        if failed.exists() and read(failed).get("returncode") not in (None, 0):
            raise RuntimeError(f"Inference failed; preserve {failed}")
        time.sleep(5)
    state({"episode": episode, "status": "verifying_complete_inference"})
    verification = TASK / f"ep{episode}-prediction-verification01.json"
    if verification.exists():
        raise ValueError("Prediction verification already exists; inspect prior postprocessing rather than overwrite")
    subprocess.run([str(PYTHON), "-B", str(TASK / "prepare_after_inference.py"),
                    "--episode", str(episode), "--prediction-root", str(prediction)], check=True, cwd=TASK)
    state({"episode": episode, "status": "adapting_raw_and_smoothed_generated_targets"})
    for variant in ("raw", "smoothed"):
        run_job(TASK / f"ep{episode}-adapt-{variant}01.json")
        fit_path = TASK / f"ep{episode}-fit-{variant}01.json"
        fit = read(fit_path)
        if fit.get("status") != "planned_not_started" or fit_path.with_suffix(".started.json").exists():
            raise ValueError("Fit plan has already changed or started")
        planned = fit_path.with_suffix(".planned.json")
        with planned.open("xb") as stream:
            stream.write(fit_path.read_bytes())
        observations = root / f"observations-{variant}01"
        fit.pop("status")
        fit["input_bindings"] += [{"path": str(p), "sha256": digest(p)} for p in [
            observations / "observations.npz", observations / "report.json", receipt_path,
            verification, RUNTIME / "tools/refine_human_hands.py", RUNTIME / "src/v2hoi/hand_refinement.py"]]
        fit["plan_completed_at"] = now()
        fit["generated_proposal_variant"] = variant
        fit_path.write_text(json.dumps(fit, indent=2, allow_nan=False), encoding="utf-8")
    state({"episode": episode, "status": "fitting_two_native_MHR_variants_CPU1_each"})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_job, TASK / f"ep{episode}-fit-{variant}01.json") for variant in ("raw", "smoothed")]
        for future in futures:
            future.result()
    jobs = []
    for variant in ("raw", "smoothed"):
        candidate = root / f"candidate-{variant}01"
        observations = root / f"observations-{variant}01/observations.npz"
        path = TASK / f"ep{episode}-evaluation-{variant}01.json"
        command = [PYTHON, "-B", RUNTIME / "tools/evaluate_egohand_pilots.py",
            "--episode", str(episode), "--candidate", candidate, "--egohand-observations", observations,
            "--output", root / f"evaluation-{variant}01.json", "--input-manifest", TASK / "inputs/manifest.json",
            "--review-windows", TASK / f"ep{episode}-review-windows.json", "--tool-root", RUNTIME,
            "--device", "cpu", "--threads", "1", "--model-batch", "8"]
        bindings = [candidate / "report.json", candidate / f"episode_{episode:06d}_human.npz", observations,
                    TASK / "inputs/manifest.json", RUNTIME / "tools/evaluate_egohand_pilots.py",
                    RUNTIME / "tools/evaluate_human_hand_pilots.py"]
        make_job(path, episode, command, bindings)
        jobs.append(path)
    render = TASK / f"ep{episode}-render-raw01.json"
    command = [PYTHON, "-B", TASK / "render_egohand_comparison.py", "--episode", str(episode),
        "--candidate", root / "candidate-raw01", "--egohand-observations", root / "observations-raw01/observations.npz",
        "--smoothed-egohand-observations", root / "observations-smoothed01/observations.npz",
        "--output", root / "review-raw01", "--input-manifest", TASK / "inputs/manifest.json", "--tool-root", RUNTIME]
    make_job(render, episode, command, [TASK / "render_egohand_comparison.py", TASK / "inputs/manifest.json",
        root / "candidate-raw01/report.json", root / "observations-raw01/observations.npz",
        root / "observations-smoothed01/observations.npz", RUNTIME / "tools/compare_human_hands.py", RUNTIME / "tools/review_human.py"])
    jobs.append(render)
    state({"episode": episode, "status": "fresh_native_evaluations_and_synchronized_video_rendering", "jobs": [str(x) for x in jobs]})
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_job, p) for p in jobs]
        for future in futures:
            future.result()
    summary = {"episode": episode, "completed_at": now(), "status": "complete_numeric_and_video_evidence_awaiting_visual_review",
        "raw_evaluation": str(root / "evaluation-raw01.json"), "smoothed_evaluation": str(root / "evaluation-smoothed01.json"),
        "video_review": str(root / "review-raw01/index.html"), "video_qa": str(root / "review-raw01/summary.json"),
        "quality_accepted": False, "official_score_available": False, "automatic_promotion": False,
        "all30_execution_authorized": False}
    write(root / "postprocessing-completion01.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=(11, 16), action="append", required=True)
    args = parser.parse_args()
    if len(set(args.episode)) != len(args.episode):
        parser.error("Do not repeat an episode")
    started = {"started_at": now(), "pid": os.getpid(), "command": sys.argv,
               "episodes": args.episode, "helper_sha256": digest(__file__),
               "all30_execution_authorized": False, "automatic_promotion": False}
    write(TASK / "postprocessing-started01.json", started)
    try:
        completed = [episode(n) for n in args.episode]
        state({"status": "completed_numeric_and_video_evidence_awaiting_visual_review", "episodes": completed})
        write(TASK / "postprocessing-finished01.json", {**started, "finished_at": now(), "returncode": 0, "episodes": completed})
    except Exception as error:
        state({"status": "failed_preserve_attempt", "error_type": type(error).__name__, "error": str(error)})
        write(TASK / "postprocessing-finished01.json", {**started, "finished_at": now(), "returncode": 1,
              "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
