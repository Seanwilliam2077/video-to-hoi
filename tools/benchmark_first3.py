"""Prepare Track 1 episodes 0/1/2 and plan or execute explicit benchmark recipes.

Preparation, planning and ranking never run models. Execution is a separate
command for the future Linux GPU host and does not install any environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checked_child(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError("input path leaves its Track 1/package root")
    return result


def verify_videos(package):
    manifest = read(package / "benchmarks/first3/episodes.json")
    if manifest.get("dataset") != "track1" or [e["id"] for e in manifest["episodes"]] != [0, 1, 2]:
        raise ValueError("benchmark must contain exactly Track 1 episodes 0, 1, 2")
    for episode in manifest["episodes"]:
        path = checked_child(package, episode["video"])
        if path.stat().st_size != episode["bytes"] or digest(path) != episode["video_sha256"]:
            raise ValueError(f"video verification failed: {episode['id']}")
    return manifest["episodes"]


def prepare(dataset, output):
    repo = Path(__file__).resolve().parents[1]
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if "track_2" in {p.lower() for p in dataset.parts}:
        raise ValueError("Track 2 is prohibited")
    info = read(dataset / "meta/info.json")
    if info.get("robot_type") != "video_only_object_tracking":
        raise ValueError("not the Track 1 video-only dataset")
    manifest = read(repo / "benchmarks/first3/episodes.json")
    sources = []
    for episode in manifest["episodes"]:
        source = checked_child(dataset, episode["source_path"])
        if source.stat().st_size != episode["bytes"] or digest(source) != episode["video_sha256"]:
            raise ValueError(f"source video differs from the frozen input: {episode['id']}")
        sources.append((source, episode["video"]))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(repo / "benchmarks/first3", output / "benchmarks/first3")
    (output / "tools").mkdir()
    for name in ("benchmark_first3.py", "benchmark_moge_video.py", "rank_first3.py"):
        shutil.copy2(repo / "tools" / name, output / "tools" / name)
    shutil.copy2(repo / "benchmarks/first3/README.md", output / "README.md")
    for source, relative in sources:
        target = checked_child(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (output / "metadata").mkdir()
    shutil.copy2(dataset / "meta/info.json", output / "metadata/info.json")
    for name in ("episodes.jsonl", "episodes_metadata.jsonl"):
        rows = [json.loads(line) for line in (dataset / "meta" / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        selected = [r for r in rows if r["episode_index"] in (0, 1, 2)]
        if sorted(r["episode_index"] for r in selected) != [0, 1, 2]:
            raise ValueError("missing or duplicate first-three metadata")
        (output / "metadata" / name).write_text("".join(json.dumps(r) + "\n" for r in selected), encoding="utf-8")
    write(output / "results.json", {"schema_version": 1, "runs": []})
    verify_videos(output)
    subprocess.run([sys.executable, str(output / "tools/rank_first3.py"),
                    "--protocol", str(output / "benchmarks/first3/protocol.json"),
                    "--results", str(output / "results.json"),
                    "--output-json", str(output / "ranking.json"),
                    "--output-md", str(output / "ranking.md")], check=True, capture_output=True)
    files = [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size,
              "sha256": digest(p)} for p in sorted(output.rglob("*")) if p.is_file()]
    write(output / "package-manifest.json", {"schema_version": 1, "dataset": "track1", "episodes": [0, 1, 2],
          "models_executed": False, "files": files})
    print(json.dumps({"package": str(output), "files": len(files), "status": "prepared_not_run"}))


def recipes(package):
    result = []
    for path in sorted((package / "benchmarks/first3").glob("*_recipes.json")):
        result.extend(read(path)["recipes"])
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("duplicate recipe IDs")
    return {r["id"]: r for r in result}


def make_plan(package, runtime, candidate, output):
    episodes = verify_videos(package)
    recipe = recipes(package)[candidate]
    variables = {name: runtime[name] for name in ("python", "assets", "toolkit", "shared")}
    variables.update(runtime.get("paths", {}))
    variables["python"] = runtime.get("python_by_candidate", {}).get(candidate, variables["python"])
    variables["package"] = str(package)
    jobs = []
    for episode in episodes:
        v = {**variables, "episode": episode["id"], "episode6": f"{episode['id']:06d}",
             "frames": episode["frames"], "video": str(checked_child(package, episode["video"])),
             "output": str(output / candidate / f"episode_{episode['id']:06d}")}
        missing = []
        requirements = []
        for item in recipe.get("required_paths", []):
            path = Path(item["path"].format(**v))
            ok = path.is_dir() if item["kind"] == "directory" else path.is_file()
            requirements.append({"path": str(path), "kind": item["kind"], "exists": ok})
            if not ok:
                missing.append(str(path))
        steps = [{"name": s["name"], "argv": [str(a).format(**v) for a in s["argv"]],
                  "cwd": s["cwd"].format(**v)} for s in recipe["steps"]]
        for step in steps:
            if not Path(step["cwd"]).is_dir():
                missing.append(step["cwd"])
        jobs.append({"episode": episode["id"], "frames": episode["frames"],
                     "video_sha256": episode["video_sha256"], "output": v["output"],
                     "requirements": requirements, "missing": sorted(set(missing)), "steps": steps,
                     "expected_outputs": [str(Path(v["output"]) / s.format(**v))
                                          for s in recipe.get("expected_outputs", [])]})
    return {"candidate": candidate, "recipe": recipe, "jobs": jobs,
            "environment": {k: runtime.get(k) for k in ("machine_id", "environment_id")},
            "status": "missing_inputs" if any(j["missing"] for j in jobs) else "paths_present_runtime_unverified"}


def execute(plan, timeout):
    if platform.system() != "Linux":
        raise ValueError("run this recipe on the selected Linux GPU server; local preparation remains available")
    if any(j["missing"] for j in plan["jobs"]):
        raise ValueError("preflight found missing inputs; inspect the plan")
    if not all(plan["environment"].values()):
        raise ValueError("record machine_id and environment_id before a timed run")
    if any(Path(j["output"]).exists() for j in plan["jobs"]):
        raise ValueError("output already exists; use a new immutable run directory")
    for job in plan["jobs"]:
        output = Path(job["output"])
        output.mkdir(parents=True)
        # An output existence check is never a score or full-frame validation.
        report = {"candidate": plan["candidate"], "episode": job["episode"], "frames_expected": job["frames"],
                  "video_sha256": job["video_sha256"], "environment": plan["environment"],
                  "status": "running", "steps": [], "quality_evaluated": False}
        write(output / "execution.json", report)
        start = time.perf_counter()
        try:
            for i, step in enumerate(job["steps"]):
                env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
                with (output / f"step-{i}.log").open("wb") as stream:
                    result = subprocess.run(step["argv"], cwd=step["cwd"], env=env,
                                            stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
                report["steps"].append({**step, "returncode": result.returncode})
                if result.returncode:
                    raise RuntimeError(f"{step['name']} exited {result.returncode}")
            import glob
            absent = [p for p in job["expected_outputs"] if not glob.glob(p)]
            if absent:
                raise RuntimeError("expected artifacts not found: " + ", ".join(absent))
            report["status"] = "execution_completed_quality_pending"
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            report["status"] = "failed"
            report["error"] = str(error)
        finally:
            report["wall_seconds"] = time.perf_counter() - start
            write(output / "execution.json", report)
        print(json.dumps({"candidate": plan["candidate"], "episode": job["episode"], "status": report["status"]}))
        if report["status"] == "failed":
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset-root", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    for name in ("plan", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--package", type=Path, required=True)
        sub.add_argument("--runtime", type=Path, required=True)
        sub.add_argument("--candidate", required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name == "run":
            sub.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.dataset_root, args.output)
        return 0
    plan = make_plan(args.package.resolve(), read(args.runtime), args.candidate, args.output.resolve())
    if args.command == "plan":
        print(json.dumps(plan, indent=2))
        return 0
    return execute(plan, args.timeout)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as error:
        print(f"benchmark: {error}", file=sys.stderr)
        raise SystemExit(2)
