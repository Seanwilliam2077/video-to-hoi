"""Search scale and compare unchanged MHR inference; no submission export."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-run", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wait-pid", type=int)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=a.resume)
    prepared = a.output/"preparation"
    run = a.output/"run"
    def execute(name, arguments):
        print("START", name, flush=True)
        with (a.output/f"{name}{'-resume' if a.resume else ''}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable,*arguments], check=True, stdout=log, stderr=subprocess.STDOUT)
        print("PASS", name, flush=True)
    search = ["tools/windows/legacy_scale_search.py", "--preparation",str(a.preparation),"--control-run",str(a.control_run),"--output",str(prepared)]
    if a.wait_pid:
        search += ["--wait-pid", str(a.wait_pid)]
    if a.resume:
        search += ["--resume"]
    execute("scale-search", search)
    identity = json.loads((a.control_run/"native-inputs.json").read_text())
    control = identity["command"]
    def setting(flag, default):
        return control[control.index(flag)+1] if flag in control else default
    # Identical recorded control settings, changing only the prepared mesh scale.
    arguments = ["-m","v2hoi.native","--dataset-root","third_party/CARI4D_MHR_Win/data/v2d-challenge/track_1",
                 "--episode",str(identity["episode"]),"--preparation",str(prepared/"preparation.json"),
                 "--allow-budgeted-mesh","--allow-legacy-scale-search","--output",str(run),
                 "--weights",setting("--weights_path","weights/cari4d")]
    for official, adapter, default in (
        ("--postopt_lr","--postopt-learning-rate","0.001"),
        ("--postopt_num_steps","--postopt-steps","300"),
        ("--postopt_batch_size","--postopt-batch-size","0"),
        ("--moge_batch_size","--moge-batch-size","4"),
        ("--sam3d_chunk_size","--sam3d-chunk-size","4"),
        ("--sam3d_refit_batch_size","--sam3d-refit-batch-size","64"),
        ("--depth_alignment_render_batch_size","--depth-render-batch-size","4"),
        ("--coconet_render_batch_size","--coconet-render-batch-size","4"),
        ("--render_batch_size","--render-batch-size","4")):
        arguments += [adapter, setting(official,default)]
    if "--optimize_object_rotation" in control:
        arguments += ["--optimize-object-rotation"]
    for name,value in (("render",256),("refine",512),("depth",32)):
        arguments += [f"--foundationpose-{name}-batch-size",str(value)]
    execute("native", arguments)
    execute("native-audit",["tools/windows/check_native_run.py","--run",str(run),"--preparation",str(prepared/"preparation.json")])
    execute("stage-audit",["tools/windows/audit_stage_commands.py","--run",str(run),"--output",str(run/"stage-audit.json")])
    for label, target in (("control", a.control_run), ("legacy", run)):
        execute(f"motion-{label}",["tools/windows/audit_object_motion.py","--run",str(target),"--output",str(a.output/f"motion-{label}.json"),"--without-export"])
    report = {"control":str(a.control_run),"variant":str(run),"submission_export":"Not requested; only export after explicit submission confirmation",
              "legacy_scale_search":json.loads((prepared/"legacy-scale-search.json").read_text()),
              "motion_control":json.loads((a.output/"motion-control.json").read_text()),
              "motion_legacy":json.loads((a.output/"motion-legacy.json").read_text()),
              "scope":"Same 4096-face surface and MHR settings, replacing input scale search only; prediction diagnostics are not official scores",
              "video_review_complete":False}
    variant_identity = json.loads((run/"native-inputs.json").read_text())
    def model_options(command):
        values = command[3:].copy()
        for flag in ("--video_path","--mask_h5_path","--object_mesh_path","--output_dir"):
            i = values.index(flag)
            del values[i:i+2]
        return values
    same_options = model_options(control) == model_options(variant_identity["command"])
    same_sources = identity["source_sha256"] == variant_identity["source_sha256"]
    if not same_options or not same_sources or identity["video_sha256"] != variant_identity["video_sha256"]:
        raise ValueError("Comparison changed official sources, model settings or video")
    report["controlled_comparison"] = {"same_model_options":same_options,"same_official_sources":same_sources,
        "same_source_video":True,"same_masks":identity["preparation"]["masks"]["sha256"] == variant_identity["preparation"]["masks"]["sha256"],
        "only_preprocessing_change":"One uniform scale chosen by legacy search on the control's exact 4096-face mesh"}
    (a.output/"result.json").write_text(json.dumps(report,indent=2))
    print("Scale-search inference comparison complete; no submission export", flush=True)


if __name__ == "__main__":
    main()
