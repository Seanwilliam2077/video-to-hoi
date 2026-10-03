# Team workflow

Four people build six stages in parallel, one module each. `main` currently runs end to end with fake backends, which test the wiring but do not reconstruct a scene or produce a validated challenge submission. The stages exchange files specified in [contracts.md](contracts.md). A real backend replaces a fake stage and is checked on Track 1 inputs against a fixed Track 1 upstream run.

Planning update, 2026-10-03: [design.md](design.md) defines the bounded refinement proposal, [evaluation.md](evaluation.md) records the current official evaluator semantics, and [implementation-plan.md](implementation-plan.md) defines gates G0–G4. These are implementation requirements, not completed backends or measured quality improvements. Retain the six stages and four owners; an internal coordinator may repeat affected stages only after the proposed dependency and acceptance gates exist.

## Final owner requirements (2026-10-03)

Source: the project owner's final clarification in this conversation on 2026-10-03. These are project acceptance requirements; this provenance is not a public organizer URL or a claim that the current pipeline satisfies them.

1. Fit one Sim(3) using only human geometry at the first reference frame and apply the same transform to human and object throughout the sequence. No later refitting or separate object alignment is acceptable.
2. Deliver native MHR with a validated official converter and decoder contract.
3. Evaluate object Chamfer on the posed object in the common world frame; canonical-shape or independently ICP-aligned comparisons are diagnostics only.
4. Use reference-relative acceleration error, not prediction-only smoothness, for ACC.
5. Cover every required object and every source frame, including occluded frames. A passing subset is not a complete submission.
6. Use Track 1 only, including all camera and geometry inputs. Do not import Track 2 calibration or iron, bowl, table or other meshes, including renamed, rescaled or cached derivatives.

The inspected public kit selects the first **scored** frame for its alignment; the owner's requirement says the first **reference** frame. Preserve both source statements and resolve their index mapping before G0/export acceptance. Do not silently substitute frame zero, choose a convenient visible frame or report this ambiguity as solved. The legacy scorer and fake pipeline are not implementations of these final requirements.

## Track 1-only data rule

All project development, hyperparameter selection, validation, scorer self-checks, benchmark references, and submissions use only the provided Track 1 videos and metadata, plus synthetic fixtures created independently of Track 2. Do not use Track 2 Tier 1 or Tier 2 data or assets, including meshes, human or object trajectories, camera estimates, noise statistics, or benchmarks derived from them. Do not transfer a model choice, scale, threshold, smoothing weight, or other parameter selected using Track 2 into this project.

The runner accepts only `--dataset track1` and rejects `--score`, `human=tier2`, `motion=tier2`, and `objects=reference`. The legacy backend implementations are also disabled at their entry points. The repository still contains a legacy ground-truth scorer and `test_score.py` with Tier 2-derived benchmark values; those are outside this workflow and excluded from CI. Do not use their data paths or benchmark values as acceptance evidence. The default `export=tier1` backend is only an internal serializer and does not read Track 2 data; it can export a Track 1 run. Complete source-provenance checks remain a follow-up implementation task.

Track 1 has no public reconstruction ground truth. The official evaluation kit is now available, but this repository has not implemented or validated its submission adapter. CD-H/CD-O and reference-relative ACC require the organizer's reference; the current ACC-H uses 22 body joints, not all finger joints. PEN measures predicted hand penetration into the submitted object and is not a difference from reference penetration, but the final value uses reference-derived alignment scale. An unaligned local penetration diagnostic is therefore not final official PEN. Use Track 1 reprojection, depth consistency, coverage, event timing, predicted contact checks and visual inspection as development evidence. Self-acceleration measures smoothness only; lowering it can make reference-relative ACC worse. Do not combine proxies into a purported official total. Scorer mathematics may be self-tested with independent analytic examples. No aggregate score or acceptance threshold may be normalized to Tier 2.

## Modules

The boundaries follow three dependencies:

1. **The scale anchor.** The human supplies the current metric scale prior for depth, which is used by the object module. The human module delivers a `DepthScale`; estimated human scale is not ground truth. Proposed refinement must track its uncertainty and avoid freely changing focal length, human scale and object depth together.
2. **Mesh and tracking.** Candidate mesh selection needs tracking, and tracking needs a mesh, so one module owns both.
3. **Contact refinement changes the human and object together.** It is a separate stage after both.

[pipeline.svg](pipeline.svg) draws the six stages and four modules. The metric names below identify challenge outcomes, not measurements already produced by the repository. The current official semantics and availability of local evidence are specified in [evaluation.md](evaluation.md).

| Module | Owner | Stages | Challenge outcome | Independent starting input | Main risk |
|---|---|---|---|---|---|
| 1 Platform & perception | Sean | inputs, export | all, as integration gatekeeper | Track 1 videos and metadata; synthetic contract fixtures | on the critical path for the first two weeks |
| 2 Human | TBD | human | CD-H, ACC-H; hand geometry for PEN | Track 1 videos, then a fixed Track 1 inputs snapshot | human vertices on the first scored frame determine the alignment |
| 3 Object | TBD | objects, motion | CD-O | Track 1 frames, masks, and scaled depth; a fixed Track 1 upstream run | the largest workload: the hula hoop, white furniture, and foot-pushed objects |
| 4 Temporal & physics | TBD | refine | ACC-H, ACC-O, PEN, subject to image fidelity | independent synthetic trajectories, then a fixed Track 1 human and motion snapshot | over-smoothing can reduce self-acceleration while worsening reference-relative ACC |

What each module delivers:

- **1 Platform & perception:** Track 1 masks and depth, verified camera hypotheses, a native CARI4D baseline, the official export adapter, evidence reports, CI and run provenance. Camera metadata supplies a grouping hypothesis, not proof of identical intrinsics. Deployment and submissions remain separate execution work; this planning update authorizes neither.
- **2 Human:** SAM 3D Body → MHR; one identity per clip; MHR → SOMA-X as an internal representation if needed; the `DepthScale` that makes the human the metric anchor. Review the first frame, reprojection, identity consistency, and depth alignment on Track 1. No Track 2 human labels are used.
- **3 Object:** Track 1-derived meshes, scale hypotheses, structured symmetry and a torus model for the hula hoop; registration and tracking with explicit visibility and ambiguity. Shared geometry across an object's clips is an engineering prior, not a confirmed submission requirement. Keep cross-episode reconstruction disabled pending organizer clarification and record each contributing source. Deliver every frame, distinguishing evidence-backed poses from inferred occlusion filling.
- **4 Temporal & physics:** bounded repairs of demonstrated failures, with body motion and hand contact evaluated separately. Preserve motion onset, amplitude and event timing; do not optimize self-acceleration toward zero. Use typed contact edges for simultaneous support, grasp and sliding interactions, uncertainty-aware constraints, and symmetry-aware rotations. Native MHR is the planned authority and SOMA is derived. Global root/orientation changes require direct image evidence and a bounded update; adding a learned refiner requires a demonstrated benefit over the fixed baseline.

Module 3 starts with meshes while module 1 builds the Track 1 baseline. Select the smallest working native model path before adding candidates. Separate incompatible environments, but do not make installing every candidate a prerequisite for a real baseline. Record versions and runtime identity when execution is separately undertaken.

Each module has a shared page for tasks, progress, run results, and reading: [Platform](https://claude.ai/artifact/A9hrz9B8qed9WWqkFWpAHg), [Human](https://claude.ai/artifact/3jiAK6fqoNwqGtLwDKf3pk), [Object](https://claude.ai/artifact/35K4apKHN75wiUuB8qxmq8), and [Physics](https://claude.ai/artifact/DUFpvGZmA2DyuU68afM7DV). Record each run's ID, Track 1 source episodes, upstream ID, code revision, model versions, visual evidence, and limitations. The [Track 1 reference](https://claude.ai/artifact/YXL887jxTTp7aQByCQvC6b) page holds challenge rules and organizer answers; keep those answers dated and linked.

## Development sets and evidence

| Set | Episodes | Use |
|---|---|---|
| `first3` | Track 1: 0, 1, 2 | existing hula-hoop integration fixture; not evidence of general HOI quality |
| `dev-mini` | Track 1: 0 hula hoop, 6 iron, 9 white desk, 16 foam block, 24 sitting on a stool | every stage PR: thin object, hand-held object, furniture, simple object, and body contact |
| `dev-full` | Track 1: all 30 | weekly integration and coverage review |
| `synthetic` | Analytic cameras, shapes, poses, and trajectories created without Track 2 | contract, geometry, scorer-math, occlusion, and contact self-checks |

Use the same immutable input snapshot and baseline for before/after comparisons. Freeze independently reviewed observation frames and event intervals before comparing candidates; estimated masks/depth used by reconstruction are not independent ground truth. Report per-episode and per-object results, visible/occluded intervals, contact transitions and worst regressions, alongside macro summaries. Preserve held-out review observations and record their role. Do not count adjacent frames as independent samples or infer generalization from the three hoop clips. A proxy can disagree with the hidden official metric; the PR must say what its evidence supports.

Reject frozen-motion controls that reduce self-acceleration but fail observed motion, and hand/object separation that reduces penetration but breaks contact. Compare all metrics and overlays from one reloaded exported candidate; never assemble a claimed improvement from different versions of the human, mesh and trajectory.

## Branches and pull requests

- `main` stays green and runnable. Everything reaches it through a pull request.
- Each module works on its own branch: `platform` (module 1), `human` (2), `object` (3), and `physics` (4). Merge `main` into your branch at least once a week, and open a pull request when the change is reviewable.
- **Contracts first.** A change to a field's meaning or shape goes in a small PR of its own, bumps `CONTRACT_VERSION`, and is reviewed by the owners of the stages that read it. Implementation PRs follow.
- Module 1 owns the stage contracts, validation tools, and CI. Version any evaluator change so reports from different versions are not treated as directly comparable.
- The repository is in English: code, docs, commit messages, and PR descriptions.

## Merge review

The Track 1-only quantitative evaluator, automated baseline comparison, and data-provenance gate are **not implemented yet**. The following is the review procedure, not a claim that CI currently enforces it.

1. Run the explicit synthetic test list in `.github/workflows/tests.yml`. Do not run bare `pytest`: the excluded `test_score.py` still contains legacy Tier 2-derived assertions. Fake-stage runs through export test wiring only and never invoke the legacy scorer. CI does not yet establish provenance or validate real model outputs.
2. Run changed real stages on `dev-mini` with `--dataset track1` and a fixed Track 1 `--upstream runs/baseline-vN`. Rerun dependent stages after any changed input, scale, mesh, or pose. Do not pass `--score`: the existing scorer requires non-Track 1 ground truth.
3. Attach a before/after table for each episode and links to overlays and native-output evidence. Include run/baseline IDs, code/model/runtime versions, input and output hashes, observation versions, failure intervals and resource use. Explain regressions and uncertainty; do not label proxies as official CD, ACC or PEN. The proposed hash-based dependency graph is not yet implemented, so explicitly audit each dependency before promotion.
4. Review official-format compatibility, source provenance, complete per-frame outputs, first-scored-frame quality, contact and motion fidelity, and the module's target evidence. Record accept/reject/unresolved and any trade-off; retain a recoverable baseline. A documentation design score is not an empirical quality score. Until the evaluator and gate are implemented, there is no automatic numerical pass threshold.

A stage can be developed against a fixed Track 1 upstream run using `--upstream`. For example, a human backend can run with `--dataset track1 --episodes 0 6 9 16 24 --upstream runs/baseline-vN --stages human refine export --backend human=<name>`. This is a run command, not a score command. Use only Track 1-derived upstream artifacts; a fake upstream verifies plumbing but cannot establish reconstruction quality.

## Progress reporting after each task

The project owner requests an update to the [project progress page](https://claude.ai/artifact/U2vgULC1h3DurtZwyQRHS1) after completed project work. Keep its versioned source, [status.html](status.html), current as part of task completion. This Chinese page is intended for leadership and product updates; preserve its existing layout and explain progress in plain language.

Record the update date, completed deliverables, what was actually verified, outstanding blockers, and the next required action. Link to the supporting PR or run artifacts. Distinguish downloaded files, installed environments, synthetic tests, real baselines, and official submissions. Publish the updated artifact through an available authorized editing session. If direct publication is unavailable, deliver the updated source and explicitly state that the public Claude URL has not been republished; a Git commit alone does not update that URL.

## Weekly integration

- **Tuesday:** merge reviewed PRs, run `dev-full` on Track 1 with the newest real backends, inspect coverage and visual evidence, and pin the result as the next baseline (`runs/baseline-vN`, never overwritten). Everyone moves their `--upstream` to it.
- **Kaggle:** only module 1 submits after the now-available official evaluator's adapter and export rehearsal pass, with provenance and continuity reviewed. The legacy `v2hoi.score --strict` is not a Track 1 submission gate. Recheck current competition dates, team and submission rules against the official sources linked in [evaluation.md](evaluation.md) before submission; historical schedules are not execution authorization.

## Runs and storage

`runs/` is not committed. Code travels through git; artifacts live on the rented GPU machine or another team-controlled store and are referred to by immutable run ID. Name runs `<module>-<topic>-<n>`, for example `human-sam3d-2`; baselines are `baseline-vN`. Store the Track 1 episode list, input provenance, model/weight versions, dependency environment, upstream ID, and review evidence with each run. Do not reuse an output after its input or parameters change.

## Open items

| Item | Module |
|---|---|
| Remove remaining legacy scorer/test code; extend the runner's Track 1 restrictions to end-to-end provenance checks | 1 |
| Track 1-only reprojection, coverage, self-smoothness, and visualization reports with baseline comparison | 1, with each module defining its evidence |
| Automatic Track 1 source-provenance checks for inputs, upstream runs, and submissions | 1 |
| Real backends for every stage, starting with the CARI4D baseline on Track 1 | everyone; baseline by 1 |
| Keep module 4's human refinement consistent with submitted MHR | 4, reviewed by 2 |
| Per-module CODEOWNERS and required reviews on `main` once teammates are collaborators | 1 |
| Official-kit-compatible native MHR export adapter and reload/render rehearsal (G0/G4) | 1, reviewed by 2 and 3 |
| Proposed v2 observations, symmetry, provenance and dependency-hash contracts; keep v1 explicitly documented | 1, reviewed by all consumers |
| Immutable native baselines (G1), evidence-driven geometry checks (G2), bounded failure repairs (G3) | everyone |
| Clarify whether cross-episode reconstruction/sharing is allowed before enabling it | 1 and 3 |
