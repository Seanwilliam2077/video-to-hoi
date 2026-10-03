"""Refine (module 4, temporal & physics): the human and the object trajectory
together, after tracking and before export.

Only pass-through is implemented. Planned repairs preserve observed motion
timing, amplitude and contact while filling occlusions with explicit
uncertainty. Official acceleration is reference-relative; minimizing
self-acceleration alone can hurt it. See docs/implementation-plan.md.

Use Track 1 evidence and independently authored synthetic fixtures only.
Track 2 assets and derivatives are prohibited for every development purpose.
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
