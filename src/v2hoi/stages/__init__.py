"""Pipeline stages, in order, and the module that owns each (docs/workflow.md):

    inputs   1 platform & perception   camera intrinsics, masks, depth
    human    2 human                   MHR and SOMA-X parameters, the metric depth scale
    objects  3 object                  one metric mesh per object
    motion   3 object                  object pose on every frame
    refine   4 temporal & physics      smoothing, static segments, occlusions, contact
    export   1 platform & perception   the internal parquet + MHR schema

A backend is a class with ``run(run: Run, clips: list[Clip]) -> None`` that
reads upstream artifacts from ``run`` and saves its own (v2hoi.contracts).
Each module lists its backends in BACKENDS. Import heavy dependencies inside
a backend, not at module level, so the fake pipeline and CI stay light.
"""
from __future__ import annotations

import importlib

from v2hoi.contracts import ContractError

STAGES = ("inputs", "human", "objects", "motion", "refine", "export")
DEFAULT_BACKENDS = {
    "inputs": "fake", "human": "fake", "objects": "fake", "motion": "fake", "refine": "fake", "export": "tier1",
}


def backends(stage: str) -> dict[str, type]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    return importlib.import_module(f"v2hoi.stages.{stage}").BACKENDS


def make(stage: str, name: str):
    if name in {"tier2", "reference"}:
        raise ContractError(f"{stage}={name} reads Track 2 assets and is disabled")
    available = backends(stage)
    if name not in available:
        raise ValueError(f"stage {stage} has no backend {name!r}; available: {sorted(available)}")
    return available[name]()


def dev_only(clips, what: str) -> None:
    """Legacy Track 2 readers are disabled, including direct class calls."""
    raise ContractError(f"{what} read Track 2 assets and are disabled")
