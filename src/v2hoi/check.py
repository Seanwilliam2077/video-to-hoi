"""Recheck module deliveries locally, without executing a model or granting quality acceptance.

    python -m v2hoi.check --run runs/human-v1 --stage human --episodes 16
    python -m v2hoi.check --run runs/human-v1 --before-stage objects --episodes 16
    python -m v2hoi.check --run runs/demo --status --episodes 16

The command prints JSON and exits 0 only when all requested gates pass. An
unrun downstream stage is expected to fail --status; use --before-stage to
check prerequisites. Native stage names are unsupported and fail closed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v2hoi.clips import load_clips
from v2hoi.contracts import Run
from v2hoi.gates import DEPENDENCIES, DeliveryGate, LIMITATIONS, PROFILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--stage", choices=DEPENDENCIES)
    selection.add_argument("--before-stage", choices=DEPENDENCIES)
    selection.add_argument("--status", action="store_true")
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    result = {"profile": PROFILE, "passed": False, "stages": [], "limitations": LIMITATIONS}
    try:
        run = Run.open(args.run)
        catalog = load_clips(run.dataset, root=args.dataset_root)
        clips = load_clips(run.dataset, args.episodes, args.dataset_root) if args.episodes else catalog
        gate = DeliveryGate(run, clips, catalog=catalog)
        stages = list(DEPENDENCIES) if args.status else [args.stage or args.before_stage]
        result["stages"] = [gate.check(stage, before=bool(args.before_stage)) for stage in stages]
        result["passed"] = all(report["passed"] for report in result["stages"])
    except Exception as exc:
        result["error"] = str(exc)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
