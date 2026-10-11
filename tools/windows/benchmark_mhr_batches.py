"""Measure original float64 MHR model-call batching on saved Track 1 parameters."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
from v2hoi.native import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameters",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    with np.load(a.parameters,allow_pickle=False) as saved:
        pose = torch.as_tensor(saved["pose"],device="cuda",dtype=torch.float64)
        shared = torch.as_tensor(np.r_[saved["scales"],saved["shape"]][None],device="cuda",dtype=torch.float64)
    source = Path("third_party/official-track1-kit/tools/track1/mesh_to_mhr_params.py")
    spec = importlib.util.spec_from_file_location("mhr_batch_benchmark",source)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    mhr = original.MHR(str(a.model),"cuda",64,"float64")
    results = []
    reference = None
    for batch in (64,128,256):
        mhr.chunk = batch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for repetition in range(4):
            torch.cuda.synchronize()
            started = time.perf_counter()
            vertices = mhr.verts(pose,shared)
            torch.cuda.synchronize()
            duration = time.perf_counter()-started
            if repetition:
                elapsed.append(duration)
        candidate = vertices.cpu().numpy()
        if reference is None:
            reference = candidate
        error = float(np.abs(reference-candidate).max())
        if error > 1e-6:
            raise ValueError("MHR model batching changed decoded vertices beyond 1e-6 mm")
        results.append({"batch":batch,"median_seconds":float(np.median(elapsed)),
            "frames_per_second":len(pose)/float(np.median(elapsed)),"max_vertex_difference_mm":error,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()})
        del vertices
        print(json.dumps(results[-1]),flush=True)
    # Reserve headroom for the converter's float64 solver, target and Jacobian arrays.
    suitable = [r for r in results if r["peak_allocated_bytes"] < 8*2**30]
    selected = min(suitable,key=lambda r:r["median_seconds"])["batch"]
    report = {"scope":"Original model forward only; complete converter time and memory verified separately",
        "parameters_sha256":sha256(a.parameters),"model_sha256":sha256(a.model),
        "official_code_sha256":sha256(source),"results":results,"selected_batch":selected}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
