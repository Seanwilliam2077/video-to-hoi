# Team workflow

Four people build six stages in parallel, one module each. `main` currently runs end to end with fake backends, which test the wiring but do not reconstruct a scene or produce a validated challenge submission. The stages exchange files specified in [contracts.md](contracts.md). A real backend replaces a fake stage and is checked on Track 1 inputs against a fixed Track 1 upstream run.

## Track 1-only data rule

All project development, hyperparameter selection, validation, scorer self-checks, benchmark references, and submissions use only the provided Track 1 videos and metadata, plus synthetic fixtures created independently of Track 2. Do not use Track 2 Tier 1 or Tier 2 data or assets, including meshes, human or object trajectories, camera estimates, noise statistics, or benchmarks derived from them. Do not transfer a model choice, scale, threshold, smoothing weight, or other parameter selected using Track 2 into this project.

The runner accepts only `--dataset track1` and rejects `--score`, `human=tier2`, `motion=tier2`, and `objects=reference`. The legacy backend implementations are also disabled at their entry points. The repository still contains a legacy ground-truth scorer and `test_score.py` with Tier 2-derived benchmark values; those are outside this workflow and excluded from CI. Do not use their data paths or benchmark values as acceptance evidence. The default `export=tier1` backend is only an internal serializer and does not read Track 2 data; it can export a Track 1 run. Complete source-provenance checks remain a follow-up implementation task.

Track 1 has no public reconstruction ground truth. We cannot calculate official CD-H, CD-O, or ground-truth-based PEN locally or claim a local result is an official leaderboard score. Use Track 1 reprojection and depth consistency, mask overlap, frame coverage, trajectory self-smoothness, predicted-geometry contact checks, and visual inspection as development evidence. These are proxies, not the official metrics. Scorer mathematics may be self-tested with analytic synthetic examples that have no Track 2 source. No aggregate score or acceptance threshold may be normalized to Tier 2.

## Modules

The boundaries follow three dependencies:

1. **The scale anchor.** The human sets the metric scale of the depth, and scaled depth sets the scale of the object meshes. The human module therefore delivers a `DepthScale`.
2. **Mesh and tracking.** Candidate mesh selection needs tracking, and tracking needs a mesh, so one module owns both.
3. **Contact refinement changes the human and object together.** It is a separate stage after both.

[pipeline.svg](pipeline.svg) draws the six stages and four modules. The metric names below identify the challenge outcomes each module aims to improve; without public Track 1 ground truth, they are not locally measurable as official scores.

| Module | Owner | Stages | Challenge outcome | Independent starting input | Main risk |
|---|---|---|---|---|---|
| 1 Platform & perception | Sean | inputs, export | all, as integration gatekeeper | Track 1 videos and metadata; synthetic contract fixtures | on the critical path for the first two weeks |
| 2 Human | TBD | human | CD-H | Track 1 videos, then a fixed Track 1 inputs snapshot | a bad first frame affects the whole aligned clip |
| 3 Object | TBD | objects, motion | CD-O | Track 1 frames, masks, and scaled depth; a fixed Track 1 upstream run | the largest workload: the hula hoop, white furniture, and foot-pushed objects |
| 4 Temporal & physics | TBD | refine | ACC-H, ACC-O, PEN | independent synthetic trajectories, then a fixed Track 1 human and motion snapshot | over-smoothing can improve acceleration while damaging pose and contact |

What each module delivers:

- **1 Platform & perception:** the rented GPU machine; Track 1 masks (GroundingDINO → SAM2), intrinsics merged per physical camera, and depth (MoGe2); a CARI4D baseline using Track 1 only; export, Track 1-only validation tools, CI, run provenance, and Kaggle submissions.
- **2 Human:** SAM 3D Body → MHR; one identity per clip; MHR → SOMA-X as an internal representation if needed; the `DepthScale` that makes the human the metric anchor. Review the first frame, reprojection, identity consistency, and depth alignment on Track 1. No Track 2 human labels are used.
- **3 Object:** meshes for the 10 Track 1 objects, generated and selected from Track 1 frames; one metric scale per object from human-scaled Track 1 depth; symmetry handling and a torus model for the hula hoop; FoundationPose registration and tracking. Deliver a pose and confidence on every frame, including occlusions. Inspect Track 1 silhouette/depth reprojection, temporal coverage, and overlays rather than reference meshes.
- **4 Temporal & physics:** smoothing with fingers handled separately, static segments, low-confidence frame filling, and human–object contact refinement. Start with independently generated synthetic motion and contact cases; then compare Track 1 outputs against the same frozen upstream run. Report self-acceleration, pose stability, contact plausibility, and visual trade-offs. Keep the MHR trajectory usable for submission.

Module 3 is the heaviest. It starts with meshes while module 1 builds the Track 1 baseline; module 1 can later take a hard object case. Each module installs its own models on the GPU machine, with versions recorded in its run report.

Each module has a shared page for tasks, progress, run results, and reading: [Platform](https://claude.ai/artifact/A9hrz9B8qed9WWqkFWpAHg), [Human](https://claude.ai/artifact/3jiAK6fqoNwqGtLwDKf3pk), [Object](https://claude.ai/artifact/35K4apKHN75wiUuB8qxmq8), and [Physics](https://claude.ai/artifact/DUFpvGZmA2DyuU68afM7DV). Record each run's ID, Track 1 source episodes, upstream ID, code revision, model versions, visual evidence, and limitations. The [Track 1 reference](https://claude.ai/artifact/YXL887jxTTp7aQByCQvC6b) page holds challenge rules and organizer answers; keep those answers dated and linked.

## Development sets and evidence

| Set | Episodes | Use |
|---|---|---|
| `dev-mini` | Track 1: 0 hula hoop, 6 iron, 9 white desk, 16 foam block, 24 sitting on a stool | every stage PR: thin object, hand-held object, furniture, simple object, and body contact |
| `dev-full` | Track 1: all 30 | weekly integration and coverage review |
| `synthetic` | Analytic cameras, shapes, poses, and trajectories created without Track 2 | contract, geometry, scorer-math, occlusion, and contact self-checks |

Use the same input snapshot and baseline for before/after comparisons. Inspect per-episode mask or silhouette reprojection, depth residuals where available, frame and occlusion coverage, acceleration of the predicted trajectories, contact plausibility, and rendered overlays. State which inputs are model predictions rather than labels. A proxy can disagree with the hidden official metric; the PR must say what its evidence supports.

## Branches and pull requests

- `main` stays green and runnable. Everything reaches it through a pull request.
- Each module works on its own branch: `platform` (module 1), `human` (2), `object` (3), and `physics` (4). Merge `main` into your branch at least once a week, and open a pull request when the change is reviewable.
- **Contracts first.** A change to a field's meaning or shape goes in a small PR of its own, bumps `CONTRACT_VERSION`, and is reviewed by the owners of the stages that read it. Implementation PRs follow.
- Module 1 owns the stage contracts, validation tools, and CI. Version any evaluator change so reports from different versions are not treated as directly comparable.
- The repository is in English: code, docs, commit messages, and PR descriptions.

## Merge review

The Track 1-only quantitative evaluator, automated baseline comparison, and data-provenance gate are **not implemented yet**. The following is the review procedure, not a claim that CI currently enforces it.

1. Run the explicit synthetic test list in the README, which is also the CI allowlist. Do not run bare `pytest`: the excluded `test_score.py` still contains legacy Tier 2-derived assertions. Fake-stage runs through export test wiring only and never invoke the legacy scorer. CI does not yet establish provenance or validate real model outputs.
2. Run changed real stages on `dev-mini` with `--dataset track1` and a fixed Track 1 `--upstream runs/baseline-vN`. Rerun dependent stages after any changed input, scale, mesh, or pose. Do not pass `--score`: the existing scorer requires non-Track 1 ground truth.
3. Attach a before/after table for each episode and a link to overlays or short inspection clips. Include run ID, baseline ID, code and model versions, episode list, the exact Track 1 inputs, and the chosen proxies. Explain regressions and uncertainty; do not label proxies CD-H, CD-O, or an official PEN score.
4. Review contract validity, complete per-frame outputs, first-frame quality, physical plausibility, and the module's target evidence. A reviewer records the decision and any trade-off in the PR. Until the evaluator and gate are implemented, there is no automatic numerical pass threshold.

A stage can be developed against a fixed Track 1 upstream run using `--upstream`. For example, a human backend can run with `--dataset track1 --episodes 0 6 9 16 24 --upstream runs/baseline-vN --stages human refine export --backend human=<name>`. This is a run command, not a score command. Use only Track 1-derived upstream artifacts; a fake upstream verifies plumbing but cannot establish reconstruction quality.

## Progress reporting after each task

The project owner requests an update to the [project progress page](https://claude.ai/artifact/U2vgULC1h3DurtZwyQRHS1) after completed project work. Keep its versioned source, [status.html](status.html), current as part of task completion. This Chinese page is intended for leadership and product updates; preserve its existing layout and explain progress in plain language.

Record the update date, completed deliverables, what was actually verified, outstanding blockers, and the next required action. Link to the supporting PR or run artifacts. Distinguish downloaded files, installed environments, synthetic tests, real baselines, and official submissions. Publish the updated artifact through an available authorized editing session. If direct publication is unavailable, deliver the updated source and explicitly state that the public Claude URL has not been republished; a Git commit alone does not update that URL.

## Weekly integration

- **Tuesday:** merge reviewed PRs, run `dev-full` on Track 1 with the newest real backends, inspect coverage and visual evidence, and pin the result as the next baseline (`runs/baseline-vN`, never overwritten). Everyone moves their `--upstream` to it.
- **Kaggle:** only module 1 submits, using Track 1-derived reconstruction artifacts after the official export format is available and provenance and continuity have been reviewed. The legacy `v2hoi.score --strict` depends on Track 2 ground truth and is not a Track 1 submission gate. Up to 5 submissions a week, unlimited in the last 3 days before the 2026-11-04 freeze. All four people must be in one Kaggle team, with the same members across all five Track 1 competitions.

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
| Official submission export backend once `eval_reconstruction.py` is published | 1 |
| Resume keyed on input hashes (design section 3) | 1 |
