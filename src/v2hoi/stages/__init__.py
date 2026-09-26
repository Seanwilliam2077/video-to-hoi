"""Pipeline stages, in order. Each stage module owns one part of the pipeline:

    inputs   platform  camera intrinsics, human and object masks
    human    human     MHR and SOMA-X parameters per clip
    objects  objects   one metric mesh per object
    motion   motion    object pose on every frame, contact refinement
    export   platform  the Tier 1 layout that v2hoi.score reads

A backend is a class with ``run(run: Run, clips: list[Clip]) -> None`` that
reads upstream artifacts from ``run`` and saves its own (v2hoi.contracts).
Each module lists its backends in BACKENDS. Import heavy dependencies inside
a backend, not at module level, so the fake pipeline and CI stay light.
"""
from __future__ import annotations

import importlib

STAGES = ("inputs", "human", "objects", "motion", "export")
DEFAULT_BACKENDS = {"inputs": "fake", "human": "fake", "objects": "fake", "motion": "fake", "export": "tier1"}


def backends(stage: str) -> dict[str, type]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    return importlib.import_module(f"v2hoi.stages.{stage}").BACKENDS


def make(stage: str, name: str):
    available = backends(stage)
    if name not in available:
        raise ValueError(f"stage {stage} has no backend {name!r}; available: {sorted(available)}")
    return available[name]()
