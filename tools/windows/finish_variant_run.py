"""Verify and export the calibrated, colored, reduced-mesh Track 1 variant."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import psutil
from v2hoi.native import native_environment, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--native-pid", type=int)
    p.add_argument("--tune-model-batch", action="store_true")
    p.add_argument("--result",type=Path)
    p.add_argument("--source", type=Path, default=Path("third_party/CARI4D_MHR_Win"))
    p.add_argument("--export-submission", action="store_true",
                   help="Only use after the owner explicitly confirms submission export")
    a = p.parse_args()
    if not a.export_submission:
        p.error("Submission export requires explicit owner confirmation and --export-submission; previews do not require conversion")
    if a.native_pid is not None:
        try:
            process = psutil.Process(a.native_pid)
            command = process.cmdline()
            if "v2hoi.native" not in command or Path(command[command.index("--output")+1]).resolve() != a.run.resolve():
                raise ValueError("PID is not this variant run")
            print("Waiting for this variant's native stages",flush=True)
            while process.is_running():
                try:
                    process.wait(timeout=5)
                    break
                except psutil.TimeoutExpired:
                    pass
        except psutil.NoSuchProcess:
            pass
    if not json.loads((a.run/"native-status.json").read_text())["complete"]:
        raise ValueError("Native variant failed; no export started")
    logs = a.run / "verification-logs"
    logs.mkdir(parents=True, exist_ok=True)
    output = a.run/"export"
    identity = json.loads((a.run/"native-inputs.json").read_text())
    episode = identity["episode"]
    command = identity["command"]
    weights = Path(command[command.index("--weights_path") + 1]).resolve()
    environment = native_environment(a.source.resolve(), weights)
    def execute(name, args):
        print("START",name,flush=True)
        with (logs/f"variant-{name}.log").open("w",encoding="utf-8") as stream:
            subprocess.run([sys.executable,*args],check=True,stdout=stream,stderr=subprocess.STDOUT,env=environment)
        print("PASS",name,flush=True)
    execute("native-audit",["tools/windows/check_native_run.py","--run",str(a.run),"--preparation",str(a.preparation)])
    execute("stage-audit",["tools/windows/audit_stage_commands.py","--run",str(a.run),"--output",str(a.run/"stage-audit.json")])
    model_batch = 64
    if a.tune_model_batch:
        execute("model-batches",["-m","v2hoi.cuda_budget","--fraction",".75","--report",str(a.run/"model-batches-memory.json"),"--",
            "tools/windows/benchmark_mhr_batches.py","--parameters","runs/windows-episode16/faithful-v2/export/episode_000016.npz",
            "--model","weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt","--output",str(a.run/"model-batches.json")])
        model_batch = json.loads((a.run/"model-batches.json").read_text())["selected_batch"]
    execute("official-converter",["-m","v2hoi.cuda_budget","--fraction",".75","--report",str(a.run/"export-memory.json"),"--",
        "-m","v2hoi.native_export","--run",str(a.run),"--assets",str(weights/"sam3d_body"),"--output",str(output),"--model-batch",str(model_batch)])
    execute("official-format",["tools/windows/validate_official_export.py","--episodes",str(output),"--episode",str(episode)])
    execute("packed-render",["-m","v2hoi.cuda_budget","--fraction",".75","--report",str(a.run/"export-render-memory.json"),"--",
        "tools/windows/render_export.py","--run",str(a.run),"--export",str(output),"--mesh",str(output/f"episode_{episode:06d}_object_budgeted.glb")])
    execute("motion-audit",["tools/windows/audit_object_motion.py","--run",str(a.run),"--output",str(a.run/"motion-audit.json")])
    prepared = json.loads(a.preparation.read_text())
    calibration_path = (a.preparation.parent/prepared["mesh_calibration"]["receipt"]["path"]).resolve()
    report = {"date":"2026-10-11","provenance":"track1","episode":episode,
        "scope":"One calibrated colored 4096-face inference variant; original NVIDIA optional rotation optimization enabled. Not the provided-mesh release test or an official accuracy score.",
        "preparation_sha256":sha256(a.preparation),"identity":identity,
        "calibration":json.loads(calibration_path.read_text()),
        "native":json.loads((a.run/"verification.json").read_text()),
        "human_export":json.loads((output/"export-report.json").read_text()),
        "format":json.loads((output/"official-format-report.json").read_text()),
        "motion":json.loads((a.run/"motion-audit.json").read_text()),
        "geometry_preserved_inference_to_export":True,"video_review_complete":False,
        "permitted_exception":"Kaolin penetration fallback equivalence is unproven",
        "online_status_synced":False}
    if prepared.get("mesh_scale_search"):
        search_path = a.preparation.parent/prepared["mesh_scale_search"]["receipt"]["path"]
        report["legacy_scale_search"] = json.loads(search_path.read_text())
        report["scope"] = "Legacy scale-search preprocessing comparison on the control's exact colored 4096-face surface; all original MHR stages and control settings retained. No official accuracy score."
    destination = a.result or Path(f"deployment/windows/{a.run.name}-episode{episode:02d}-result.json")
    destination.write_text(json.dumps(report,indent=2))
    print("Variant processing and export verified; video quality still requires review",flush=True)


if __name__ == "__main__":
    main()
