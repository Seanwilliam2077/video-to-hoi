# Developer guide

The README covers the project, its pages and the downloads. This guide covers everything a developer needs next: the data policy, the pipeline and modules, setup, running, validation and tests.

## Data policy

**Use Track 1 only.** Track 2 data, including Tier 1 and Tier 2, must not be used for this project's reconstruction, development, parameter tuning, validation, scorer self-checks, baseline normalization, or submissions. This includes videos, meshes, human/object trajectories, labels, and camera, scale, or tuning parameters derived from them.

Develop against Track 1 videos and artifacts reconstructed from those videos. Use independently generated synthetic geometry and trajectories for unit tests; do not derive fixtures or noise distributions from Track 2. Older team pages or code paths that suggest Track 2 use are superseded by this policy. See [workflow.md](workflow.md).

## Pipeline and modules

From one video taken by a single static RGB camera, and a text description of the object, the planned pipeline recovers the person (body and hands, as MHR parameters), the object's mesh, and its pose in every frame, in a common metric frame. The current official evaluator uses reference-dependent CD-H/CD-O and ACC-H/ACC-O; ACC-H covers 22 body joints, and PEN measures predicted hand penetration into the submitted object rather than a reference difference. See [evaluation.md](evaluation.md) for the pinned evidence and remaining limits.

The human supplies the current scale prior that the object module uses (`DepthScale`). Mesh generation and tracking feed each other inside one module. The proposed refine stage makes bounded, evidence-supported human/object repairs; decreasing self-acceleration alone is not an official ACC improvement. Modules develop against immutable Track 1 snapshots, with independent synthetic fixtures for interface tests. Cross-episode geometry sharing is an engineering prior and remains disabled pending organizer clarification of allowable evidence pooling.

Four people build the pipeline in parallel, one module and one branch each. Every stage has a fake backend, so `main` runs end to end from day one and each module replaces its fakes with real models.

- [workflow.md](workflow.md): why the modules are cut this way, how each one develops independently, and the merge gate.
- [contracts.md](contracts.md): the files the stages exchange.
- [design.md](design.md): the architecture, its evidence, and what is known about the official submission.
- [evaluation.md](evaluation.md): current official metric semantics, independent diagnostic evidence and promotion criteria.
- [track1-compliance.md](track1-compliance.md): the six final requirements versus current implementation, evidence gaps and ordered acceptance work.
- [implementation-plan.md](implementation-plan.md): G0 official/native contracts, G1 real native baselines, G2 camera/scale evidence, G3 bounded repairs and G4 full-set export rehearsal.
- [platform.md](platform.md): remote machine sizing, the candidate servers, and baseline acceptance requirements. Remote deployment and the two real baselines are still pending.

## Current state

- **Planning update (2026-10-03):** the official evaluation kit is available; its repository adapter and official export are still pending. The revised proposal is documentation, not a runtime implementation or measured improvement. A local CPU test environment was prepared; no reconstruction model installation, server deployment, model run or submission was performed for this update.
- **Final owner clarification (2026-10-03, this conversation):** one common first-reference-frame Sim(3), native MHR, world-posed object Chamfer, reference-relative ACC, all objects/all frames including occlusions, and Track 1-only geometry/cameras are acceptance targets. See [workflow.md](workflow.md#final-owner-requirements-2026-10-03). The public kit's first-scored-frame selection must be reconciled with the owner's first-reference-frame wording; neither current fake outputs nor the legacy scorer establish compliance.
- **Validation:** Track 1 has no public reconstruction ground truth for reference-dependent CD/ACC. Local review uses independent image observations, motion/event fidelity, completeness, contact and overlays. Self-smoothness remains a diagnostic only. Official PEN also uses reference-derived alignment scale; an unaligned local penetration calculation is not the final official score. Automated evidence extraction and promotion gates remain to be implemented.
- **Legacy scorer:** `v2hoi.score` contains metric primitives, but its default reference and internal score depend on Track 2. Its existing command-line workflow is outside the approved project workflow.
- **Pipeline:** `python -m v2hoi.run` runs all six stages with fake backends through export. No reconstruction models are wired in yet.
- **Server handoff:** [deployment/README.md](../deployment/README.md) covers the source ZIP, the pinned model and source inventory, dependency locks, and verification. It works without SSH. [PREPARATION_STATUS.md](../deployment/PREPARATION_STATUS.md) records what has actually been downloaded and verified.
- **First-three-video comparison kit:** release `first3-benchmark-20260927` (69 MB, three complete videos, no weights). [benchmarks/first3/README.md](../benchmarks/first3/README.md) covers Track 1 episodes 0, 1, 2, the execution recipes and the ranking checks. No models were run and no scores are reported.

## Setup

Requires Python 3.10+ and a CUDA build of PyTorch already installed. The venv below reuses the system torch instead of installing a CPU build:

```bash
uv venv .venv --python 3.10 --system-site-packages
uv pip install --python .venv/Scripts/python.exe numpy scipy pandas pyarrow trimesh huggingface_hub pytest warp-lang rtree cholespy usd-core
uv pip install --python .venv/Scripts/python.exe --no-deps "py-soma-x==0.2.1"
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
```

On first use, `py-soma-x` downloads the SOMA-X assets from Hugging Face, pinned to the revision the official toolkit uses.

## Data

The README's download command writes the Track 1 subtree to `data/v2d/track_1/`. The current `v2hoi.download --videos` implementation is also restricted to Track 1 and accepts `--revision` to pin the dataset snapshot; this change is awaiting remote runtime verification. Neither command removes data already on disk. Existing Track 2 files and derived artifacts must stay out of project runs and baselines.

[track1-videos/index.html](track1-videos/index.html) previews all 30 Track 1 videos, filterable by object and camera. GitHub shows its source, so open it from a clone. `python tools/track1_gallery/build.py` rebuilds it from the downloaded videos.

## Running the pipeline

```bash
.venv/Scripts/python.exe -m v2hoi.run --run-id track1-smoke --dataset track1 --episodes 16 12
```

Runs every stage on Track 1 episodes 16 and 12 into `runs/track1-smoke/` and exports the result in the internal schema. The current fake outputs test the interfaces, not reconstruction quality. `--stages` runs only some stages, `--upstream runs/<run>` reads outputs reconstructed from Track 1 in a fixed earlier run, and `--backend <stage>=<name>` picks an implemented backend.

The runner now defaults to and accepts only `track1`; keep `--dataset track1` explicit in shared commands. It rejects `--score`, `human=tier2`, `motion=tier2`, and `objects=reference`. These code changes await remote runtime verification. The default exporter is still named `tier1`; this is a serializer name only and does not read Track 2 data.

## Validation and official scoring

The internal export contains `meta/info.json`, `meta/episodes_metadata.jsonl`, `data/chunk-000/episode_XXXXXX.parquet`, `mesh/<object>/<object>.glb`, and the corresponding MHR files. This storage schema does not authorize using another dataset's assets.

Start G1 with immutable native baselines for episodes 16 and 12, then review `dev-mini` (`0 6 9 16 24`) by object and interaction type, and all 30 episodes at G4. Preserve `first3` as a limited hoop integration fixture. Freeze observation/annotation and event-review policies before comparisons; distinguish fitted evidence from independent or held-out observations. Check all-frame coverage, first-scored-frame human geometry, reprojection, scale consistency, occlusion reappearance, contact gaps/penetration, motion amplitude and timing. Report per-episode and worst-case regressions, not only means. Model-estimated depth and masks are not ground truth.

Every claimed improvement must belong to one exact exported candidate reloaded for measurement and rendering. Reject copied-first-frame motion even if its self-acceleration is zero, and reject hand/object separation used merely to reduce penetration. Add models only to address a recorded failure that the stable native baseline cannot resolve. Proposed v2 native MHR, observations, typed symmetry and dependency-hash contracts are documented separately from implemented v1 in [contracts.md](contracts.md).

The old Track 2 self-check commands and Tier 2-normalized score are retired. `v2hoi.score --strict` cannot certify Track 1 provenance, current metric semantics or official format. Implement the now-available official kit's native MHR/object adapter, pin its source/configuration and complete export/reload/render checks before submission. Do not convert proxies into a scalar official-score estimate. [evaluation.md](evaluation.md) is the current evaluation reference; [implementation-plan.md](implementation-plan.md) distinguishes planned gates from existing behavior.

## Tests

Use the explicit allowlist in `.github/workflows/tests.yml` as the source of truth. Its independent synthetic tests cover download/bundle tools and fake-stage wiring:

```bash
python -m pytest -q tests/test_bundle.py tests/test_contracts.py tests/test_dataset.py tests/test_fetch_assets.py tests/test_download_foundationpose.py tests/test_model_release_prepare.py tests/test_model_release_restore.py tests/test_handoff_restore.py tests/test_benchmark_first3.py tests/test_rank_first3.py tests/test_geometry.py tests/test_metrics.py tests/test_run.py
```

These files create their own synthetic fixtures. `test_run.py` uses generated metadata and never invokes the legacy scorer. Do not run a bare `pytest`: `test_score.py` still contains Tier 2-derived benchmark assertions and can read Track 2 files when present, so CI excludes it. Passing these tests verifies the covered contracts and wiring, not real model quality, complete data provenance, or the official submission format.

## Team pages

All team pages live on claude.ai and share one navigation bar (team hub, status, reference, then the four modules). Their sources and update rules:

- Team hub: [hub.html](hub.html), published with `pipeline.svg` as a supporting file.
- Status page (Chinese, for leadership): [status.html](status.html); read [status.md](status.md) before editing it.
- Track 1 reference and the four module pages: maintained on claude.ai. Read the live page, edit, and republish to the same URL; their tasks, updates and resources are stored in each page's database.

The page URLs and the navigation rule are in [AGENTS.md](../AGENTS.md).
