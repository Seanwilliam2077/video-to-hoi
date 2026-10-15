"""Reproduce independent-folder handoffs and a blocked stale delivery on CPU.

Only hand-authored synthetic metadata and fake outputs are used. No challenge
video, model, network, native binary or reference data is loaded. Five folders
on one host demonstrate contracts; this is not a two-host model-run claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from v2hoi.clips import load_clips
from v2hoi.contracts import Motion, Run
from v2hoi.gates import DEPENDENCIES, DeliveryGate, GateError, run_lock
from v2hoi.run import run_pipeline


def demo(output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = output / "independent-synthetic-metadata"
    (metadata / "meta").mkdir(parents=True)
    info = {
        "robot_type": "video_only_object_tracking", "fps": 30, "chunks_size": 1000,
        "video_path": "unused/episode_{episode_index:06d}.mp4",
        "features": {"observation.images.exo_camera": {"shape": [24, 32, 3]}},
        "fixture_origin": "independently hand-authored; no challenge pixels or parameters",
    }
    (metadata / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (metadata / "meta/episodes_metadata.jsonl").write_text(json.dumps({
        "episode_index": 0, "object": "synthetic_cube", "object_prompt": "independent cube",
        "camera": "fixture", "video_key": "observation.images.exo_camera",
    }) + "\n", encoding="utf-8")
    (metadata / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 6}) + "\n", encoding="utf-8")
    jobs = [
        ("m1-inputs", ["inputs"]), ("m2-human", ["human"]),
        ("m3-object", ["objects", "motion"]), ("m4-physics", ["refine"]),
        ("m1-export", ["export"]),
    ]
    backends = {stage: "fake" for stage in DEPENDENCIES}
    backends["export"] = "tier1"
    upstream, runs = None, {}
    for label, stages in jobs:
        run = run_pipeline(output / label, episodes=[0], stages=stages, backends=backends,
                           upstream=upstream, dataset_root=metadata, log=lambda *_: None)
        runs[label], upstream = run, run.root
    clips = load_clips("track1", root=metadata)
    final = DeliveryGate(runs["m1-export"], clips)
    before = {stage: final.check(stage) for stage in DEPENDENCIES}
    if not all(check["passed"] for check in before.values()):
        raise AssertionError("The valid independent-folder handoff did not pass")

    # A valid-shaped but altered old tracking result must invalidate descendants.
    producer = runs["m3-object"]
    changed = producer.load(Motion, episode=0)
    changed.T_cam_obj[:, 0, 3] += 0.1
    producer.save(changed, episode=0)
    after = final.check("export")
    if after["passed"]:
        raise AssertionError("Changed upstream bytes were not detected")

    class MustNotRun:
        called = False

        def run(self, run, clips):
            self.called = True
            raise AssertionError("A blocked consumer was started")

    sentinel = MustNotRun()
    blocked = Run.start(output / "blocked-consumer", "track1", runs["m1-export"])
    failure = None
    try:
        with run_lock(blocked):
            DeliveryGate(blocked, clips).execute("export", "sentinel", sentinel)
    except GateError as exc:
        failure = str(exc)
    if failure is None or sentinel.called:
        raise AssertionError("The consumer was not stopped at preflight")
    summary = {
        "scope": "Independent synthetic internal v1 fixture; one host, five module folders",
        "valid_stage_checks": {stage: check["passed"] for stage, check in before.items()},
        "tampered_upstream_blocks_export": not after["passed"],
        "blocked_consumer_started": sentinel.called,
        "failure": after["checks"],
        "native_model_execution": False, "quality_acceptance": False,
        "output_root": str(output),
    }
    (output / "demo-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("work") / "delivery-demo" / str(uuid4()))
    args = parser.parse_args()
    print(json.dumps(demo(args.output), indent=2))


if __name__ == "__main__":
    main()
