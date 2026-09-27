"""Run the pipeline, or some of its stages, on Track 1 inputs only.

    # file-format smoke test with the currently registered fake backends
    python -m v2hoi.run --run-id demo --dataset track1 --episodes 0 6

    # iterate on one stage against a fixed snapshot of the others
    python -m v2hoi.run --run-id motion-try1 --dataset track1 --episodes 0 6 \
        --upstream runs/demo --stages motion refine export

Stages read what they need from this run and fall back to --upstream, so each
person can work on their stage without waiting for the others. These fake
backends do not reconstruct the video. Track 1 has no public ground truth, so
this runner has no ground-truth scoring mode. A run that re-runs human or motion
must re-run refine too; export checks the upstream-run case.
Re-running stages in an existing run overwrites their outputs and keeps the
rest. See docs/contracts.md and docs/workflow.md.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from v2hoi.clips import DATASETS, load_clips
from v2hoi.contracts import Run
from v2hoi.stages import DEFAULT_BACKENDS, STAGES, make

RUNS_DIR = Path("runs")


def parse_backends(specs: list[str]) -> dict[str, str]:
    """["motion=foundationpose", ...] → the default backends with those replaced."""
    chosen = dict(DEFAULT_BACKENDS)
    for spec in specs:
        stage, sep, name = spec.partition("=")
        if not sep or stage not in STAGES or not name:
            raise ValueError(f"--backend expects <stage>=<name> with a stage in {STAGES}, got {spec!r}")
        if name in {"tier2", "reference"}:
            raise ValueError(f"{stage}={name} reads Track 2 assets and is disabled")
        chosen[stage] = name
    return chosen


def run_pipeline(
    root: Path,
    dataset: str = "track1",
    episodes: list[int] | None = None,
    stages: list[str] | None = None,
    backends: dict[str, str] | None = None,
    upstream: Path | None = None,
    dataset_root: Path | None = None,
    log=print,
) -> Run:
    if dataset != "track1":
        raise ValueError("only Track 1 pipeline runs are permitted")
    stages = list(stages or STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stages {unknown}; expected some of {STAGES}")
    backends = {**DEFAULT_BACKENDS, **(backends or {})}
    clips = load_clips(dataset, episodes, dataset_root)
    run = Run.start(
        root, dataset, Run.open(upstream) if upstream else None,
        episodes=[c.episode for c in clips], stages=stages, backends={s: backends[s] for s in stages},
    )
    for stage in STAGES:
        if stage not in stages:
            continue
        start = time.time()
        make(stage, backends[stage]).run(run, clips)
        log(f"{stage:<8} {backends[stage]:<12} {time.time() - start:6.1f}s")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="the run lives in <runs-dir>/<run-id>")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="track1")
    parser.add_argument("--episodes", type=int, nargs="+", help="default: every episode of the dataset")
    parser.add_argument("--stages", nargs="+", choices=STAGES, help="default: all, in pipeline order")
    parser.add_argument("--backend", action="extend", nargs="+", default=[], metavar="STAGE=NAME",
                        help="select a registered Track 1 backend, e.g. --backend motion=mytracker")
    parser.add_argument("--upstream", type=Path, help="run to read missing artifacts from")
    parser.add_argument("--score", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.score:
        parser.error("--score is disabled: Track 1 has no public ground truth")
    run = run_pipeline(
        args.runs_dir / args.run_id, args.dataset, args.episodes, args.stages,
        parse_backends(args.backend), args.upstream,
    )
    print(f"run: {run.root}")


if __name__ == "__main__":
    main()
