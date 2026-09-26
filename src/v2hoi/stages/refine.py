"""Refine (module 4, temporal & physics): the human and the object trajectory
together, after tracking and before export.

This stage owns the physical axis (ACC-H, ACC-O, PEN) and must not cost
Chamfer. Planned real backends: per-joint smoothing with the hands on their
own (design 3.4); static segments locked; low-confidence frames re-filled
by interpolation; CARI4D's joint refinement (CoCoNet, then contact-guided
optimization with ground and table constraints).

Develop on Tier 2, whose noise the organizer sampled from Track 1's error
distributions, and score against Tier 1:

    python -m v2hoi.run --run-id t2 --dataset tier1 --episodes 7 9 10 19 21 \
        --backend human=tier2 motion=tier2 objects=reference --score
"""
from __future__ import annotations

from v2hoi.clips import Clip
from v2hoi.contracts import Human, Motion, RefinedHuman, RefinedMotion, Run


class PassThrough:
    """Changes nothing: the refined trajectories are the inputs."""

    def run(self, run: Run, clips: list[Clip]) -> None:
        for clip in clips:
            run.save(RefinedHuman.like(run.load(Human, episode=clip.episode)), episode=clip.episode)
            run.save(RefinedMotion.like(run.load(Motion, episode=clip.episode)), episode=clip.episode)


BACKENDS = {"fake": PassThrough}
