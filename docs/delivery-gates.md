# Automatic module delivery gates

Updated 2026-10-15. The six-stage internal v1 runner now checks every module's delivery automatically and stops before a dependent stage when required outputs or predecessor records fail. The independent [check CLI](../src/v2hoi/check.py) revalidates received local files without executing a model. Passing this gate means **internal format and dependency readiness**, not reconstruction quality, native MHR acceptance or official submission compliance.

## Execution sequence

1. Acquire the local run lock before changing run history. Resolve the selected backend and registered stage contract.
2. Mark a new uniquely identified attempt as running. Recheck predecessor receipts, current file hashes, schemas and selected clip/video identities. A failed local attempt masks older upstream success.
3. Copy the declared prerequisite files into a fresh input snapshot and verify the copied bytes. Run the backend in a fresh output directory. An old file or upstream fallback cannot fulfill this attempt's output obligation.
4. Reload every required output and run its structural checks. Compare input snapshots and current predecessors again. Missing, malformed, changed or unsupported deliveries fail closed.
5. Publish only verified files, preserving attempt evidence. Export publishes a fresh complete tree and retains the previous tree, excluding stale leftover episodes. A failed publication never produces a passing pointer.
6. Write an immutable report with checks, input/output hashes, clip identities, validator identity, backend and predecessor receipt references. Only a passing current pointer permits the next internal stage.

The controller owns this sequence. A backend returning zero or writing its own `passed` field does not skip verification. Interrupted attempts remain running and block continuation until deliberately rerun. The local lock is not a distributed lock; investigate an interrupted process before manually clearing a leftover lock.

## Implemented stage requirements

| Stage | Prerequisites | Required outputs and actual checks |
| --- | --- | --- |
| `inputs` | Selected clip metadata; source video hash when the file exists | Camera, masks and depth; correct image size, finite positive focal lengths, packed-mask shape, positive depth stride, nonnegative finite full-frame depth |
| `human` | `inputs` | Internal v1 human arrays and depth-scale record; every required field, complete frame shapes, finite arrays and positive finite scale |
| `objects` | `inputs`, `human` | Object record and mesh; exact object name, requested episode membership, positive scale, readable nonempty finite geometry, valid triangles and metre sanity bounds |
| `motion` | `inputs`, `human`, `objects` | Every frame's object transform and confidence; finite arrays, proper rotations, homogeneous last row and confidence in [0,1] |
| `refine` | `inputs`, `human`, `objects`, `motion` | Both refined human and object outputs; complete frame shapes and structural checks |
| `export` | All five predecessors | Internal metadata, parquet, decomposed MHR arrays and meshes; exact requested episode/file set, full ordered frames, timestamps, row totals, finite arrays and equality with the refined/source data |

The dependency policy is intentionally conservative. These are declared internal responsibilities, not a traced read set of a real model. Empty synthetic masks and stationary synthetic poses can pass; meaningful visibility, contact and reconstruction still require separate evidence. Missing source videos are allowed only because this profile supports the existing metadata-only fake smoke path; their absence is recorded as `video: null`. This profile must never be used as a real-input acceptance gate.

## Running and checking

Run from the repository root in the development environment. These commands select the currently registered **fake/internal** backends; they do not run native reconstruction models.

```bash
python -m v2hoi.run --run-id e16-inputs-a --episodes 16 --stages inputs --dataset-root data/v2d/track_1
python -m v2hoi.run --run-id e16-human-a --episodes 16 --stages human --upstream runs/e16-inputs-a --dataset-root data/v2d/track_1
python -m v2hoi.check --run runs/e16-human-a --stage human --episodes 16 --dataset-root data/v2d/track_1
python -m v2hoi.check --run runs/e16-human-a --before-stage objects --episodes 16 --dataset-root data/v2d/track_1
python -m v2hoi.check --run runs/e16-human-a --status --episodes 16 --dataset-root data/v2d/track_1
```

`run` applies preflight and post-run checks automatically. `check` prints JSON and exits **0** only when all requested checks pass, **1** for missing/failed/stale delivery, and **2** for invalid CLI arguments. `--stage` checks that producer's completed delivery; `--before-stage` checks a consumer's prerequisites; `--status` checks all six stages, so unrun downstream stages make its overall result false. A standalone check does not launch the next stage or publish a new passing producer receipt.

The default episode selection is all episodes in the local metadata. Use `--episodes` for an isolated task. The metadata catalog must also contain other episodes referenced by a selected batch receipt. Unknown/native stage names have no registered validator and are rejected. There is no skip-gate option in the runner.

## Reports, stale results and independent experiments

```text
runs/<run-id>/.gates/
  attempts/<attempt-id>/input-snapshot/  fixed prerequisite copies
  attempts/<attempt-id>/outputs/         independently checked producer output
  receipts/<attempt-id>.json             immutable passing or failed report
  current/<stage>/<episode>.json         current running/passed/failed pointer
  previous/<attempt-id>/export/          retained prior export tree when replaced
```

A report binds the complete episode batch, metadata/video identities, validator source/profile, backend, exact predecessor receipts and input/output bytes. Validator source line endings are normalized for Windows/Linux checkouts. The actual data hashes remain exact bytes. Reports always leave `quality_accepted` and `official_export_valid` false and do not change any v2 scene selection.

Re-running a producer creates a new attempt, even when output bytes happen to match. Descendant receipts bind the original attempt and become stale. Recheck reports do not silently bless old files with new lineage. Legacy runs without delivery receipts need to be reproduced through the gated runner; a post-hoc existence check cannot establish what their backend consumed.

**Prefer one episode and one experiment per run directory for team work.** A multi-episode invocation is one atomic delivery batch. Replacing only part of it invalidates the original batch's receipt; rerun the batch or use isolated episode runs. This conservative behavior avoids mixing attempts. Keep accepted baselines in separate fixed run directories and create new `--run-id` values for alternatives.

The current v1 object layout still groups fake assets by object name; it is not the target episode-local native asset design. These delivery checks do not resolve that scientific migration. Absolute/relative upstream paths in `run.json` also remain local configuration: native package transport and path relocation are separate pending work, not completed merely by copying a report to another machine.

Receivers reopen real files and recursively verify their available upstream chain. An edited or stale producer flag is insufficient. Reports are local consistency evidence, not cryptographic attestations against a malicious producer. Run local semantic checks and the required source/quality review after future transport import. Do not synchronize local locks, current pointers or mutable cache directories through Git.

## Native integration remains a separate gate

The [distributed native profile](distributed-execution.json) stays `design_only`. Native 01-04, 05 and 06-07 need pinned typed adapters for their H5/PKL/PTH schemas, actual input closure, frame/mesh/camera binding and relocation. A nonempty binary file does not pass by default. After native normalization, reuse the v2 `EpisodeEvidence`/`SceneState`/`SolvePlan` checks and separate execution, quality and export reports. See [team-handoffs.md](team-handoffs.md) for module ownership and outstanding boundary checks.

The gate tests use only independent synthetic fixtures. Full-suite results are recorded with the platform delivery; they do not establish a real multi-host model run.

For a standalone CPU demonstration without videos, models or downloads, run `python tools/demo_delivery_gates.py`. It creates fresh independent module folders, passes all six internal deliveries, then changes an upstream trajectory and confirms that the blocked consumer backend never starts. An optional `--output PATH` must name a directory that does not yet exist.
