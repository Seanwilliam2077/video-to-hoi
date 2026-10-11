"""Run one Python stage with an explicit PyTorch CUDA allocator budget."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fraction", type=float, default=.75)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not 0 < a.fraction <= 1 or not command:
        p.error("Require 0 < fraction <= 1 and a Python script or -m module after --")
    import torch
    active = torch.cuda.is_available()
    if active:
        torch.cuda.set_per_process_memory_fraction(a.fraction)
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    completed = False
    try:
        if command[0] == "-m":
            sys.argv = command[1:]
            runpy.run_module(command[1], run_name="__main__", alter_sys=True)
        else:
            sys.argv = command
            sys.path.insert(0, str(Path(command[0]).resolve().parent))
            runpy.run_path(command[0], run_name="__main__")
        completed = True
    except SystemExit as exc:
        completed = exc.code in (None, 0)
        raise
    finally:
        report = {"completed": completed, "command": command, "seconds": time.monotonic() - started,
                  "torch_allocator_fraction": a.fraction, "cuda_available": active,
                  "scope": "Per-stage PyTorch allocator only; driver, desktop and other CUDA libraries are outside this limit"}
        if active:
            report.update(device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2))
        print("V2HOI_CUDA_BUDGET " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
