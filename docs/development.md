# Developer guide

The README covers the project, its pages and the downloads. This guide covers data policy, modules, setup, running, validation and tests. [track1-requirements.md](track1-requirements.md) is the single active specification for the owner's latest six Track 1 requirements.

## Data policy

**Use Track 1 only.** Track 2 data, including Tier 1 and Tier 2, must not be used for this project's reconstruction, development, parameter tuning, validation, scorer self-checks, baseline normalization, or submissions. This includes videos, meshes, human/object trajectories, labels, and camera, scale, or tuning parameters derived from them.

Develop against Track 1 videos and artifacts reconstructed from those videos. Use independently generated synthetic geometry and trajectories for unit tests; do not derive fixtures or noise distributions from Track 2. Older team pages or code paths that suggest Track 2 use are superseded by this policy. See [workflow.md](workflow.md).

## Pipeline and modules

From one video taken by a single static RGB camera, and a text description of the object, the planned pipeline recovers the person (body and hands, as MHR parameters), the object's mesh, and its pose in every frame, in a common metric frame. The current official evaluator uses reference-dependent CD-H/CD-O and ACC-H/ACC-O; ACC-H covers 22 body joints, and PEN measures predicted hand penetration into the submitted object rather than a reference difference. See [evaluation.md](evaluation.md) for the pinned evidence and remaining limits.

In the planned real pipeline, the human supplies a scale prior for the object module through `DepthScale`; the current fake value is not an estimate. Mesh generation and tracking feed each other inside one module. The proposed refine stage makes bounded, evidence-supported human/object repairs; decreasing self-acceleration alone is not an official ACC improvement. Modules develop against immutable Track 1 snapshots, with independent synthetic fixtures for interface tests. Cross-episode geometry sharing is an engineering prior and remains disabled pending organizer clarification of allowable evidence pooling.

Members work across all four modules with floating roles. Module labels define input/output boundaries; concurrent tasks use separate branches and result directories. Every stage has a fake backend, so `main` runs end to end from day one and each module replaces its fakes with real models.

- [workflow.md](workflow.md): why the modules are cut this way, how each one develops independently, and the merge gate.
- [Module submissions](../deliveries/README.md): the four delivery folders, shared record template, code PRs, check/review evidence and large-output locations. Post progress in the corresponding live module page as well; Git commits do not update those pages.
- [platform-plan.md](platform-plan.md): Module 1 scope, existing progress, proposed implementation rounds and team discussion checklist.
- [delivery-gates.md](delivery-gates.md): executable internal v1 delivery checks, stop/continue behavior, receipts and recheck CLI.
- [team-handoffs.md](team-handoffs.md) and [module capability map](../configs/team-modules.json): independent starting work, interface boundaries and consumer review.
- [distributed-execution.md](distributed-execution.md) and [design-only profile](distributed-execution.json): model-stage responsibilities across independent machines, immutable result packages and remaining transport/native-adapter work.
- [contracts.md](contracts.md): the files the stages exchange.
- [contracts-v2.md](contracts-v2.md): proposed evidence, immutable scene and bounded solve-plan contracts; v1 remains implemented.
- [v2-foundation.md](v2-foundation.md): opt-in executable structural profile, immutable store, typed records, report trust and CPU fixtures.
- [component-audit.md](component-audit.md): source-pinned component decisions, exposure ledger, initial review policy and runtime restoration gaps.
- [design.md](design.md): the architecture, its evidence, and what is known about the official submission.
- [video-coverage.md](video-coverage.md) and [JSON inventory](video-coverage.json): all 30 metadata records, candidate diagnostics, explicit unknowns and exposure-aware validation roles.
- [track1-requirements.md](track1-requirements.md): R1–R6, the current acceptance requirements; conflicting historical rules are superseded.
- [evaluation.md](evaluation.md): pinned external metric mechanics, native-MHR reference diagnostics, independent evidence and limitations.
- [track1-compliance.md](track1-compliance.md): the six final requirements versus current implementation, evidence gaps and ordered acceptance work.
- [implementation-plan.md](implementation-plan.md): P0–P5 work packages and dependencies mapped to the G0–G4 evidence gates.
- [platform.md](platform.md): remote machine sizing, the candidate servers, and baseline acceptance requirements. Remote deployment and the two real baselines are still pending.

## Current state

- **Human local batch execution (2026-10-21):** the received input archive, complete file inventory, all-30 handoff gate and exact pinned native readers passed for 30 videos / 16,563 frames. The new adapter consumes each original mask and estimated camera without fixed-box/FOV fallback. A three-frame GPU smoke and native fixed-identity fit passed; the complete EP16/EP11 pilot is running from an immutable source snapshot. The resumable SAM 3D Body + MediaPipe runner preserves baseline/candidate comparisons and independent hand evidence. See [human-batch.md](human-batch.md). This is execution in progress, not an all-30 result, quality acceptance or an official score.
- **Human batch/dashboard audit (2026-10-20):** upstream `main` was synchronized to `9e671e5` while preserving the local episode-16 Human prototype and its tests. The all-30 Body input package covers 16,563 original frames, but the current receiving checkout still needs the archive and receiver checks. The native Windows estimator supports external masks and camera intrinsics; the local runner still uses a fixed box and assumed field of view, so a new verified input adapter and resumable per-episode orchestration are required. The verified local hand experiment used MediaPipe; HandFlow code, checkpoint and licensed MANO assets are not present in the audited project environments. The public board can register output links and provenance, but it is not an inference launcher. Its current Human declaration requires `human` and `depth_scale` artifacts, and full-stage upstream acceptance remains separate from standalone Body input readiness. The v1 Human schema is not the native 136/68/45 output contract. See [the audit receipt](evidence/human-batch-readiness-20261020.json) for the evidence, remaining adapters and bounded CPU estimate. No new inference, Human acceptance, upload or progress record is claimed by this audit.
- **Delivery gates (2026-10-15):** all six internal v1 stages now run behind automatic preflight/output checks, fresh attempts, copied input snapshots, exact predecessor receipts and fail-closed continuation. A read-only check CLI and independent-folder CPU demonstration are available. This is executable internal/synthetic readiness, not native adapter, quality or official export acceptance. Separate module start/review responsibilities are documented.
- **Distributed design (2026-10-13):** separate model-stage workers replace the shared-server assumption. Native 01-04, 05 and 06-07 retain exact dependencies and form one logical run, followed by atomic scene import and GPU-aware export. Git coordinates small manifests; large artifacts use verified packages. Stage wrappers, transport, relocation and a two-host full-episode rehearsal are still pending; no runtime code or model execution changed.
- **Foundation implementation (2026-10-13):** the P0 source/exposure audit and initial structural/review policy are recorded. `v2hoi.v2` implements strict artifact IO, immutable record publication, atomic scene selection, evidence/scene/plan structural checks and synthetic native-array/rigid-geometry checks. This is partial P1 infrastructure; `v2hoi.run`, v1 artifacts and fake reconstruction backends remain unchanged. A separate local CPU test environment was prepared; no model environment was restored or executed.
- **Design update (2026-10-13):** video evidence and metric-mesh preparation feed one atomic native CARI4D task. Its coupled outputs become an immutable `SceneState`; diagnostic `SolvePlan` repairs produce new candidates, independently reviewed before promotion and export/read-back. The v2 specification, 30-video metadata coverage and implementation backlog are documented; no runtime contract version, model execution or acceptance gate changed in this update.
- **Branch evidence:** `main` at `0e86889696edc4d81a3c287cb93d76baece4a1d5` retains fake reconstruction stages. `origin/windows-cari4d` at `fb0bac9` preserves native experiments and repair/export work. Their provisional scale, orientation and contact findings require component review and complete acceptance; neither a blanket branch restore nor an accepted real baseline is established.
- **Implementation update (2026-10-03):** the requirements and reference-based native-MHR diagnostic are updated. The official packer adapter, real reconstruction backends and measured reconstruction improvement remain pending. No reconstruction model installation, server deployment, real model run or submission is established by this update.
- **Requirements:** follow [R1–R6](track1-requirements.md). The first reference frame has an explicit original-video ID. A later scored subset retains that frame's human-derived alignment; external array indexing cannot change this rule.
- **Validation:** Track 1 has no public reconstruction ground truth for reference-dependent CD/ACC. Local review uses independent image observations, motion/event fidelity, completeness, contact and overlays. Self-smoothness remains a diagnostic only. Official PEN also uses reference-derived alignment scale; an unaligned local penetration calculation is not the final official score. Automated evidence extraction and promotion gates remain to be implemented.
- **Reference diagnostic:** `v2hoi.score` uses explicitly supplied native-MHR predictions, permitted references, canonical object meshes and an authorized decoder. It has no Track 2 reference default or normalization and does not produce an official score or composite ranking. See [evaluation.md](evaluation.md) for its input and equivalence limits.
- **Pipeline:** `python -m v2hoi.run` runs all six stages with fake backends through export. No reconstruction models are wired in yet.
- **Local Human prototype:** [human-local.md](human-local.md) documents the separate native Windows SAM 3D Body environment, verified model/input acquisition, completed 360-frame CPU body/hand inference, shared-identity native MHR export, and complete raw/fitted reviews. Finite values, exact frame IDs, shared identity/scales, camera consistency and browser checks passed. Inference through the final review took 2 h 03 min 48 s on CPU. Its four-worker fitting command, source hashes and parity evidence are in the walkthrough. This Human-only prototype is not registered as a backend of the six-stage runner. Historical external packer observations are recorded in [assessment-20260930.md](assessment-20260930.md); [the current requirements](track1-requirements.md) govern acceptance.
- **Human research:** [human-strategy-20261006.md](human-strategy-20261006.md) records primary-source research available by 2026-10-06, native MHR integration choices, evidence and contact/temporal refinement hypotheses. These are proposed comparisons and development experiments, not measured improvements in official challenge scores.
- **Hand refinement and official control:** [human-hand-experiment-20261015.md](human-hand-experiment-20261015.md) records independent detections, native finger/wrist fitting, failed ablations, our RGB-derived object/masks and the complete eight-stage local CARI4D control. Fresh official execution took 45.20 minutes with all 360 frames and a disclosed full-mesh collision variant. Three-stage and hand-closeup videos are included. Within the original fixed-identity branch, v3 improves visible right-hand image agreement; every accepted left detection is after release, so left grasp remains unobserved. Official CoCoNet recovers lost object tracking, while its default final output worsens right-hand image alignment and preserves local finger coefficients. Our contact candidate was rejected. No reconstruction acceptance or official score is claimed. Original wrapper failure and separately verified repair/cache replay are documented.
- **Server handoff:** [deployment/README.md](../deployment/README.md) covers the source ZIP, the pinned model and source inventory, dependency locks, and verification. It works without SSH. [PREPARATION_STATUS.md](../deployment/PREPARATION_STATUS.md) records what has actually been downloaded and verified.
- **First-three-video comparison kit:** release `first3-benchmark-20260927` (69 MB, three complete videos, no weights). [benchmarks/first3/README.md](../benchmarks/first3/README.md) covers Track 1 episodes 0, 1, 2, the execution recipes and the ranking checks. No models were run and no scores are reported.

## Setup

For the standalone real Human workflow on this workstation, follow
[human-local.md](human-local.md). Its isolated Python 3.11 environment lives under
`work/human-runtime/venv`; the older general setup below is not its installer.

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

[Video preparation through the Body handoff](body-input-preparation.md) gives the pinned all-30 download, executable CPU byte/decode/timeline gate, and the remaining human-observation, camera and receiver checks. Install the optional `video` extra for decoding. Raw-data readiness does not accept the camera/masks/depth stage or make a Body runtime ready.

## Running the pipeline

```bash
.venv/Scripts/python.exe -m v2hoi.run --run-id track1-smoke --dataset track1 --episodes 16 12
```

Runs every stage on Track 1 episodes 16 and 12 into `runs/track1-smoke/` and exports the result in the internal schema. The current fake outputs test the interfaces, not reconstruction quality. `--stages` runs only some stages, `--upstream runs/<run>` reads outputs reconstructed from Track 1 in a fixed earlier run, and `--backend <stage>=<name>` picks an implemented backend.

The runner now defaults to and accepts only `track1`; keep `--dataset track1` explicit in shared commands. It rejects `--score`, `human=tier2`, `motion=tier2`, and `objects=reference`. These code changes await remote runtime verification. The default exporter is still named `tier1`; this is a serializer name only and does not read Track 2 data.

## Validation and official scoring

The internal export contains `meta/info.json`, `meta/episodes_metadata.jsonl`, `data/chunk-000/episode_XXXXXX.parquet`, `mesh/<object>/<object>.glb`, and the corresponding MHR files. This storage schema does not authorize using another dataset's assets.

Start G1 with immutable native baselines for episodes 16 and 12, then review `dev-mini` (`0 6 9 16 24`) by object and interaction type, and all 30 episodes at G4. Preserve `first3` as a limited hoop integration fixture. Freeze observation/annotation and event-review policies before comparisons; distinguish fitted evidence from independent or held-out observations. Check all-frame coverage, first-reference-frame mapping and human geometry, reprojection, scale consistency, occlusion reappearance, contact gaps/penetration, motion amplitude and timing. Report per-episode and worst-case regressions, not only means. Model-estimated depth and masks are not ground truth.

Every claimed improvement must belong to one exact exported candidate reloaded for measurement and rendering. Reject copied-first-frame motion even if its self-acceleration is zero, and reject hand/object separation used merely to reduce penetration. Add models only to address a recorded failure that the stable native baseline cannot resolve. Proposed v2 native MHR, observations, typed symmetry and dependency-hash contracts are documented separately from implemented v1 in [contracts.md](contracts.md).

Historical Track 2 self-check commands, reference defaults and normalized totals are removed and superseded. The replacement diagnostic requires explicit permitted references and reports its limits; it cannot certify provenance truth or official format. Implement the official kit's native MHR/object adapter, pin its source/configuration and complete export/reload/render checks before submission. Do not convert proxies into a scalar official-score estimate. [evaluation.md](evaluation.md) records evaluation mechanics; [implementation-plan.md](implementation-plan.md) distinguishes planned gates from existing behavior.

## Tests

Use the explicit allowlist in `.github/workflows/tests.yml` as the source of truth. Its independent synthetic tests cover tools, contracts, fake-stage wiring and the replacement reference-based diagnostic:

```bash
python -m pytest -q tests/test_bundle.py tests/test_contracts.py tests/test_dataset.py tests/test_fetch_assets.py tests/test_download_foundationpose.py tests/test_model_release_prepare.py tests/test_model_release_restore.py tests/test_handoff_restore.py tests/test_benchmark_first3.py tests/test_rank_first3.py tests/test_geometry.py tests/test_metrics.py tests/test_track1_score.py tests/test_run.py tests/test_human_result.py tests/test_hand_refinement.py tests/test_hand_contact.py tests/test_review_human.py tests/test_human_runtime_assets.py tests/test_sam_body_inputs.py tests/test_human_batch.py tests/test_deliverables.py tests/test_delivery_gates.py tests/test_v2_artifacts.py tests/test_v2_geometry.py tests/test_v2_contracts.py tests/test_progress_board.py tests/test_prepare_track1_data.py tests/test_prepare_body_masks.py tests/test_prepare_body_camera.py tests/test_check_body_handoff.py tests/test_package_body_handoff.py tests/test_delivery_catalog.py
```

These files create their own independent synthetic fixtures. The former `test_score.py` and its prohibited benchmark assumptions have been replaced by `test_track1_score.py`; do not recover or reuse its historical data or values. Continue to use the explicit CI allowlist rather than bare `pytest`. Passing these tests verifies covered mathematics, contracts and wiring, not real authorized MHR model execution, reconstruction quality, complete data provenance or the official submission format.

## Team pages

All team pages live on claude.ai and share one navigation bar (team hub, status, reference, then the four modules). Their sources and update rules:

- Team hub: [hub.html](hub.html), published with `pipeline.svg` as a supporting file.
- Status page (Chinese, for leadership): [status.html](status.html); read [status.md](status.md) before editing it.
- Track 1 reference and the four module pages: maintained on claude.ai. Read the live page, edit, and republish to the same URL; their tasks, updates and resources are stored in each page's database.

The page URLs and the navigation rule are in [AGENTS.md](../AGENTS.md).
