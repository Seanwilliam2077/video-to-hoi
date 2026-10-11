"""After the existing native job exits, verify, officially convert, pack and render."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from v2hoi.native import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--native-pid", type=int, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        process = psutil.Process(a.native_pid)
        command = process.cmdline()
        if "v2hoi.native" not in command or "--output" not in command or Path(command[command.index("--output")+1]).resolve() != a.run.resolve():
            raise ValueError("PID does not belong to this native run")
        print(f"Waiting for native PID {a.native_pid} before starting another GPU stage", flush=True)
        while process.is_running():
            try:
                process.wait(timeout=5)
                break
            except psutil.TimeoutExpired:
                pass
    except psutil.NoSuchProcess:
        pass
    if not json.loads((a.run / "native-status.json").read_text())["complete"]:
        raise RuntimeError("Native pipeline failed; export was not started")
    audit_path = a.run / "silhouette-compatibility-resume.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        memory_path = a.run / "gpu-memory.json"
        memory = json.loads(memory_path.read_text())
        memory["samples"] = audit["prior_successful_gpu_samples"] + memory["samples"]
        memory["peak_observed_mib"] = max(x["used_mib"] for x in memory["samples"])
        memory["scope"] = "Whole-device 2-second samples: successful stages 01-06 plus resumed 07-08; prior failed attempt retained separately in the compatibility audit"
        memory_path.write_text(json.dumps(memory, indent=2))
    output = a.run / "export"
    logs = root / "artifacts/baseline-downloads"
    def execute(name, command):
        print(f"START {name}", flush=True)
        with (logs / f"corrected-{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, *command], cwd=root, check=True, stdout=log, stderr=subprocess.STDOUT)
        print(f"PASS {name}", flush=True)
    execute("native-audit", ["tools/windows/check_native_run.py", "--run", str(a.run), "--preparation", str(a.preparation)])
    execute("official-converter", ["-m", "v2hoi.cuda_budget", "--fraction", ".75", "--report", str(a.run / "export-memory.json"), "--",
        "-m", "v2hoi.native_export", "--run", str(a.run), "--assets", "weights/cari4d/sam3d_body", "--output", str(output), "--model-batch", "64"])
    execute("official-format", ["tools/windows/validate_official_export.py", "--episodes", str(output), "--episode", "16"])
    execute("packed-render", ["-m", "v2hoi.cuda_budget", "--fraction", ".75", "--report", str(a.run / "export-render-memory.json"), "--",
        "tools/windows/render_export.py", "--run", str(a.run), "--export", str(output), "--mesh", str(output / "episode_000016_object_budgeted.glb")])
    execute("runtime", ["tools/windows/check_runtime.py"])
    execute("native-kernels", ["tools/windows/check_native_kernels.py"])
    record = {"date": "2026-10-11", "branch": "windows-cari4d", "episode": 16, "provenance": "track1",
        "scope": "Corrected official model/generation/conversion flow on one Windows video; owner-permitted Kaolin penetration exception. No Linux-output equivalence or official-score claim.",
        "run": a.run.as_posix(), "preparation_sha256": sha256(a.preparation),
        "native": json.loads((a.run / "verification.json").read_text()),
        "generation": json.loads((a.preparation.parent / "object-generation/generation.json").read_text()),
        "generation_memory": json.loads((a.preparation.parent / "object-generation/memory-offload.json").read_text()),
        "human_export": json.loads((output / "export-report.json").read_text()),
        "official_format": json.loads((output / "official-format-report.json").read_text()),
        "reloaded_render": json.loads((output / "render-report.json").read_text()),
        "silhouette_batch_check": json.loads((root / "artifacts/windows-migration/silhouette-batching.json").read_text()),
        "historical_baseline_label_retracted": True, "full_inference_surface_preserved": True,
        "official_submission_mesh_budget_applied": True,
        "environment": ".venv", "environment_packages": 180,
        "permitted_exception": "Open3D/PyTorch3D penetration fallback; Kaolin equivalence unproven",
        "online_claude_pages_updated": False,
        "cleanup_pending": "Automatic approval rejected deletion of the aborted deployment/windows/.venv; main environment is .venv"}
    (root / "deployment/windows/corrected-episode16-result.json").write_text(json.dumps(record, indent=2))
    print("Corrected run, full official conversion, format readback and budgeted-mesh render verified", flush=True)


if __name__ == "__main__":
    main()
