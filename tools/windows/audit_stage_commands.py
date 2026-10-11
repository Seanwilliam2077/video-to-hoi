"""Compare every completed stage invocation with the pinned NVIDIA launcher.

Runs the original launcher with stage execution and report writing intercepted.
It does not run models, change outputs, or download weights.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types
from v2hoi.native import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    identity = json.loads((a.run/"native-inputs.json").read_text())
    if identity["provenance"] != "track1":
        raise ValueError("Only permitted Track 1 native runs may be audited")
    source_root = Path("third_party/CARI4D_MHR_Win").resolve()
    leaf = "reconstruction/modules/v2d_cari4d/lib/run_inference.py"
    source = subprocess.check_output(["git", "-C", str(source_root), "show", f"193382a184308f6800331f3db829f76dc4294cfe:{leaf}"])
    module = types.ModuleType("pinned_launcher_audit")
    module.__file__ = str(source_root/leaf)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    module.SAM3D_SOURCE_ROOT = source_root/"reconstruction/modules/v2d_sam3d_body/lib"
    planned = []
    recorded_args = identity["command"][3:].copy()
    learning_rate = None
    if "--postopt_lr" in recorded_args:
        index = recorded_args.index("--postopt_lr")
        learning_rate = recorded_args[index+1]
        del recorded_args[index:index+2]
    def intercept(name, command, inputs, outputs, *args):
        if name == "07_refinement" and learning_rate is not None:
            locations = [command.index(flag) for flag in ("--symmetric-object","--optimize-object-rotation") if flag in command]
            index = min(locations) if locations else len(command)
            command[index:index] = ["--lr",learning_rate]
        planned.append((name,command))
        return {"name":name,"audited_without_execution":True}
    module._stage_run = intercept
    module._atomic_json = lambda *args: None
    args = vars(module._parser().parse_args(recorded_args))
    args["download_models"] = not args.pop("skip_weight_download")
    if args["download_models"]:
        raise ValueError("Audit must be offline")
    module.run_inference(**args)
    root = a.run/"native"/f"episode_{identity['episode']:06d}"/".stages"
    checks = []
    for name,expected in planned:
        artifact = root/f"{name}.json"
        actual = json.loads(artifact.read_text())["signature"]["command"]
        if actual[1:3] == ["-m", "v2hoi.cuda_budget"]:
            actual = [actual[0],*actual[actual.index("--")+1:]]
        # Python shim/base path can differ; its stage script and every argument must match.
        if actual[1:] != expected[1:]:
            raise ValueError(f"Stage command differs from the pinned launcher: {name}\n{actual}\n{expected}")
        checks.append({"stage":name,"arguments_match":True,"marker_sha256":sha256(artifact),"command":actual})
    if len(checks) != 8:
        raise ValueError("Released launcher stage coverage is incomplete")
    report = {"provenance":"track1","episode":identity["episode"],"stage_commands":"PASS",
        "scope":"All eight original stage calls with the recorded resource options; excludes object input calibration and output quality acceptance",
        "calls":checks,"upstream_revision":"193382a184308f6800331f3db829f76dc4294cfe"}
    report["explicit_original_postoptimizer_lr"] = learning_rate
    a.output.write_text(json.dumps(report,indent=2))
    print("All eight actual stage invocations match the original launcher with recorded resource options")


if __name__ == "__main__":
    main()
