"""Run the pipeline, or some of its stages, on Track 1 or on the Tier 1 dev set.

    # the whole pipeline with fake backends on two dev episodes, then score it
    python -m v2hoi.run --run-id demo --dataset tier1 --episodes 7 9 --score

    # iterate on one stage against a fixed snapshot of the others
    python -m v2hoi.run --run-id fp-try1 --dataset tier1 --episodes 7 9 \
        --upstream runs/demo --stages motion export --backend motion=<name> --score

Stages read what they need from this run and fall back to --upstream, so each
person can work on their stage without waiting for the others. Re-running
stages in an existing run overwrites their outputs and keeps the rest.
See docs/contracts.md and docs/workflow.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from v2hoi.clips import DATASETS, load_clips
from v2hoi.contracts import Run
from v2hoi import score as S
from v2hoi.stages import DEFAULT_BACKENDS, STAGES, make

RUNS_DIR = Path("runs")


def parse_backends(specs: list[str]) -> dict[str, str]:
    """["motion=foundationpose", ...] → the default backends with those replaced."""
    chosen = dict(DEFAULT_BACKENDS)
    for spec in specs:
        stage, sep, name = spec.partition("=")
        if not sep or stage not in STAGES or not name:
            raise ValueError(f"--backend expects <stage>=<name> with a stage in {STAGES}, got {spec!r}")
        chosen[stage] = name
    return chosen


def run_pipeline(
    root: Path,
    dataset: str,
    episodes: list[int] | None = None,
    stages: list[str] | None = None,
    backends: dict[str, str] | None = None,
    upstream: Path | None = None,
    dataset_root: Path | None = None,
    log=print,
) -> Run:
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
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="tier1")
    parser.add_argument("--episodes", type=int, nargs="+", help="default: every episode of the dataset")
    parser.add_argument("--stages", nargs="+", choices=STAGES, help="default: all, in pipeline order")
    parser.add_argument("--backend", action="append", default=[], metavar="STAGE=NAME",
                        help="replace a stage's backend; repeatable")
    parser.add_argument("--upstream", type=Path, help="run to read missing artifacts from")
    parser.add_argument("--score", action="store_true", help="score the export against Tier 1 (tier1 only)")
    parser.add_argument("--stride", type=int, default=1, help="frame stride for scoring")
    parser.add_argument("--align", choices=S.ALIGNMENTS, default="first",
                        help="alignment for scoring; first-object is for tracking work with --backend objects=reference")
    args = parser.parse_args()

    if args.score and args.dataset != "tier1":
        parser.error("--score needs --dataset tier1; Track 1 has no public ground truth")
    run = run_pipeline(
        args.runs_dir / args.run_id, args.dataset, args.episodes, args.stages,
        parse_backends(args.backend), args.upstream,
    )
    print(f"run: {run.root}")

    if args.score:
        episodes = [c.episode for c in load_clips("tier1", args.episodes)]
        cfg = S.Config(align=args.align, stride=args.stride)
        report = S.score(DATASETS["tier1"], run.root / "export", episodes, cfg)
        (run.root / "score.json").write_text(json.dumps(S._jsonable(report), indent=1), encoding="utf-8")
        (run.root / "summary.json").write_text(json.dumps(S._jsonable(S.summary(report)), indent=1), encoding="utf-8")
        print(S.format_table(report))
        print(f"\n{run.root / 'score.json'}")


if __name__ == "__main__":
    main()
