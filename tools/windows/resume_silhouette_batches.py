"""Audit a storage-only silhouette change before reusing stages 01-06.

Reject every other source change. The original entire module AST must match
after removing the new wrapper and restoring the original method name.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
from v2hoi.native import sha256


MODULE = Path("reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--previous-module", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path("third_party/CARI4D_MHR_Win"))
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    record = json.loads((a.run / "native-inputs.json").read_text())
    status = json.loads((a.run / "native-status.json").read_text())
    if status["complete"] or not record.get("official_preparation") or record["episode"] != 16:
        raise ValueError("Only the failed corrected preparation can resume here")
    if record["entry_sha256"] != sha256(root / "src/v2hoi/native.py"):
        raise ValueError("Native entry changed")
    previous = ast.parse(a.previous_module.read_text())
    current_path = a.source / MODULE
    current = ast.parse(current_path.read_text())
    cls = next(n for n in current.body if isinstance(n, ast.ClassDef) and n.name == "MHRParityPostOptimizer")
    wrapper = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_silhouette_loss")
    cls.body.remove(wrapper)
    next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_silhouette_loss_unchunked").name = "_silhouette_loss"
    if ast.dump(previous) != ast.dump(current):
        raise ValueError("Postopt changes exceed the independent rendering wrapper")
    old_hash, new_hash = hashlib.sha256(), hashlib.sha256()
    for path in sorted((a.source / "reconstruction/modules").rglob("*.py")):
        if ".venv" in path.parts:
            continue
        name = path.relative_to(a.source)
        for digest in (old_hash, new_hash):
            digest.update(name.as_posix().encode())
        new_hash.update(path.read_bytes())
        old_hash.update(a.previous_module.read_bytes() if name == MODULE else path.read_bytes())
    if old_hash.hexdigest() != record["source_sha256"]:
        raise ValueError("Another upstream source changed; cannot reuse prior stages")
    proof = json.loads((root / "artifacts/windows-migration/silhouette-batching.json").read_text())
    if (proof["result"] != "PASS" or proof["module_sha256"] != sha256(current_path)
            or proof["checker_sha256"] != sha256(root / "tools/windows/check_silhouette_batches.py")):
        raise ValueError("Current source lacks a matching loss/gradient check")
    for name in ("masks", "mesh"):
        preparation = record["preparation"][name]
        base = root / "runs/windows-episode16/faithful-preparation"
        if sha256(base / preparation["path"]) != preparation["sha256"]:
            raise ValueError("Immutable preparation changed")
    stage_root = a.run / "native/episode_000016/.stages"
    outputs = []
    for stage in ("01_depth", "02_export", "03_human_initialization", "04_depth_alignment", "05_object_pose", "06_coconet"):
        marker = json.loads((stage_root / f"{stage}.json").read_text())
        for item in marker["outputs"]:
            path = Path(item["path"])
            if path.stat().st_size != item["size"] or path.stat().st_mtime_ns != item["mtime_ns"]:
                raise ValueError(f"Prior stage output changed: {path}")
            outputs.append({"path": str(path), "sha256": sha256(path)})
    audit_path = a.run / "silhouette-compatibility-resume.json"
    if audit_path.exists():
        raise FileExistsError("Preserve the existing compatibility audit")
    audit = {"previous_identity": record, "new_source_sha256": new_hash.hexdigest(),
        "previous_module_sha256": sha256(a.previous_module), "current_module_sha256": sha256(current_path),
        "loss_gradient_check": proof, "reused_outputs": outputs,
        "scope": "Only independent silhouette raster calls checkpointed in batches 16; all frames, 300 steps, optimizer and temporal objective unchanged"}
    old_memory = json.loads((a.run / "gpu-memory.json").read_text())
    cutoff = (stage_root / "06_coconet.json").stat().st_mtime
    audit["prior_successful_gpu_samples"] = [x for x in old_memory["samples"] if x["time"] <= cutoff]
    audit["prior_all_gpu_memory"] = old_memory
    shutil.copy2(stage_root / "07_refinement.memory.json", a.run / "refinement-failed-memory.json")
    audit_path.write_text(json.dumps(audit, indent=2))
    record["source_sha256"] = new_hash.hexdigest()
    (a.run / "native-inputs.json").write_text(json.dumps(record, indent=2))
    print("Verified exact prior source and original module AST; stages 01-06 are reusable")


if __name__ == "__main__":
    main()
