# Proposed v2 contracts: evidence, scene state and bounded repair

**Status: PROPOSED, 2026-10-13. Specification revision `2.0-draft.1`.**
This document specifies the full target migration. An opt-in, partial executable profile is now documented in
[v2-foundation.md](v2-foundation.md); it implements structural checks and immutable storage, not the native pipeline.
Implemented `CONTRACT_VERSION` and every [v1 contract](contracts.md) remain unchanged.
MUST, MUST NOT and SHOULD describe requirements for that future implementation.
[Track 1 requirements R1–R6](track1-requirements.md) remain authoritative.
Review changes to this proposal with Platform, Human, Object and Physics before implementation.

## 1. Scope and authority

- `EpisodeEvidence` indexes video observations and their provenance; it is not reconstructed truth.
- `SceneState` binds one complete scene candidate and its validation results.
- `SolvePlan` authorizes one bounded computation from exact inputs to a new candidate.
- Native MHR is the human authority; derived meshes/SOMA views are diagnostics with derivation links.
- A coupled backend may produce human and motion together; stage ownership does not require separate inference.
- Track 2 data and derivatives MUST NOT be downloaded, opened or used at any step, including validation.
- Unknown scene-asset provenance MUST be resolved before payload loading. A free-text label is insufficient.
- Independent synthetic fixtures are permitted for interface/mathematical checks and MUST NOT enter real outputs.
- Cross-episode asset/camera estimation defaults to disabled. Shared object labels do not grant permission.
- Reconstruction anchor selection is independent of evaluation alignment. R1 uses the explicit first
  reference-frame original ID and a single human-derived Sim(3), only with a permitted reference.
  Neither this schema nor an exported scene supplies hidden reference data or an official local score.

## 2. Common envelope, references and identities

Every record MUST contain `record_type`, `spec_revision`, `id`, `created_at`, `provenance` and `payload`.
`record_type` is one of the three names above. `id` is a new UUID v4; `created_at` is UTC RFC 3339.
IDs identify records, not file contents. Reusing an ID for different bytes is forbidden.
`spec_revision` is exactly `2.0-draft.1` for examples conforming to this draft; it is not a v1 version bump.
Unknown required semantics, unsupported revisions, duplicate JSON keys and non-finite numbers MUST fail.
Numbers MUST preserve integer frame IDs exactly; absent, unknown and empty MUST remain distinguishable.
Units, array shapes, dtypes, coordinate names and enum meanings MUST be explicit in typed payloads.

A file reference `ArtifactRef` MUST contain `path`, `sha256` and `media_type`.
`sha256` is 64 lowercase hexadecimal characters over the exact stored bytes, including JSON references.
JSON used to compute a semantic/configuration hash MUST use RFC 8785 canonicalization and UTF-8.
Array archives MUST contain no pickle/object arrays or executable payloads; loading MUST disable pickle.
Reference validation MUST precede payload parsing and MUST verify actual bytes, not size/mtime alone.

`path` is relative to one declared artifact-store root, uses `/`, and names a regular file.
Absolute paths, drive/UNC paths, empty components, `.`/`..`, URI schemes and alternate data streams are forbidden.
Resolve links/reparse points and enforce final path containment before opening; reject escaping paths.
Import upstream artifacts into this store or explicitly register a contained immutable store root;
do not interpret a reference relative to the current working directory or silently search upstream runs.
Referenced files MUST remain immutable; a changed file under the same path is a hash mismatch.
Write to temporary contained paths, verify, then publish by atomic rename; retain previous accepted records.

`provenance` MUST identify `source_kind` (`track1` or `independent_synthetic`), source video/metadata refs,
parent artifact refs, generating operation, code revision, configuration hash and any manual edits.
Model/runtime/decoder dependencies MUST have separate pinned identities and hashes; they are not source videos.
Scene assets/estimates MUST trace to permitted inputs. Record policy review evidence before their use.
Record producer invocation, random seed, environment and material nondeterminism; execution reproducibility
may use declared numerical tolerances rather than requiring identical GPU bytes.

## 3. EpisodeEvidence

The payload MUST contain `episode_id`, `video`, `timeline`, `entities`, `observations`, `events`,
`partitions` and `evidence_policy_ref`. Large arrays MAY be referenced/chunked; the index stays immutable.

### Timeline and association

- `video` pins original encoded bytes and decode identity; `timeline` pins the decoded frame map.
- Original frame IDs are zero-based decoded-display indices `0..T-1` of the untrimmed input video.
  Record original PTS with rational time base, presentation order, dimensions and decoded frame count.
- Every observation identifies its original frame IDs; crop-local row numbers are never original IDs.
- Retain exact row-to-source mappings for resampling, trimming or inference strides. No silent FPS conversion.
- A real complete scene MUST cover every decoded input frame, including invisible/out-of-view intervals.
- `entities` assigns episode-local stable IDs and role (`target_human`, `target_object`, `context`).
  Record the metadata/manual/model evidence for each identity and ambiguous association candidates.
- Identity changes and background-person swaps MUST be explicit failures or unresolved hypotheses.

### Observations, predictions and visibility

Each observation MUST record `observation_id`, `entity_id`, `frame_ids`, `kind`, `origin`,
`data_ref`, `producer`, `coordinate_frame`, `visibility`, `validity_ref` and `derivation_refs`.
`kind` is `pixels`, `mask`, `keypoints`, `depth`, `flow`, `feature_matches`, `landmarks` or `metadata`.
`origin` is `raw`, `model_prediction` or `human_annotation`; masks/depth inferred by a model are predictions.
`review_status` (`unreviewed`, `reviewed`, `corrected`) is separate from origin; review does not make truth.
Manual corrections MUST create new observations and retain the prior record and correction rationale.

`visibility` is `visible`, `partially_occluded`, `fully_occluded`, `out_of_view` or `unknown`,
with an evidence reference and per-pixel/per-component detail where available.
Unknown or occluded pixels MUST NOT be encoded as certain background/free space.
Depth records specify native units, metric/relative status, validity, resampling and any scale derivation.
Visible and hypothesized amodal masks MUST have distinct semantics and cannot be substituted silently.

Image observations MUST record source dimensions, processed dimensions, pixel-centre convention and
`T_processed_from_source` (3-by-3 homogeneous transform) for crop/resize/pad, plus valid image regions.
Record lens distortion/undistortion separately; nonlinear warps require pinned forward/inverse maps.
Validate landmark round trips and intrinsic transformations; inverse crop mapping must preserve holes/thin edges.
Flow and matches MUST name both source frames and both image transforms.

Estimator score, fit residual and posterior uncertainty MUST be separate fields with method/units.
No scalar or posterior score is a calibrated probability without explicit calibration evidence.
Observability SHOULD distinguish translation, rotation and contact components, including competing hypotheses.
Optimizers MUST NOT freely decrease confidence/weights to conceal residuals; weight rules are pinned inputs.

### Events and partitions

`events` contains candidate intervals for grasp, slide, support, foot push, release, rolling and visibility
transitions, with evidence refs and alternatives. Simultaneous relations are permitted; events are predictions.
Intervals use inclusive original `start_frame_id` and `end_frame_id`, never local array offsets.

`partitions` MUST pin explicit observation IDs/frame IDs, selection method, selection seed/configuration,
selection time and roles: `fit`, `review`, `diagnostic` or `unused`.
Fit/review sets MUST be disjoint by observation lineage, not only by record ID; duplicates/corrections inherit lineage.
Common video/model identity alone is not duplicate-observation lineage; shared pixels/derived measurements are.
Record temporal exclusion buffers and shared producer, video and scene dependencies as independence limitations.
If a whole-video model consumed review pixels, those pixels are not held out from that model; report this
explicitly. They MAY be held out from a subsequent local repair, whose exclusion boundary must be declared.
An observation used to select candidates/weights becomes development evidence for future comparisons.
Within-video repair review is distinct from object-group generalization holdout; neither implies 3D ground truth.
Anchor selection MUST record its observable criterion and MUST NOT use review residuals to choose fit frames.

Illustrative observation fragment (all hashes below are placeholders, not executable fixture identities):

```json
{
  "observation_id": "mask-000016-120", "entity_id": "object-0",
  "frame_ids": [120], "kind": "mask", "origin": "model_prediction",
  "data_ref": {"path": "evidence/mask-120.npz", "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "media_type": "application/x-npz"},
  "producer": {"model_id": "segmentation-pinned", "config_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
  "coordinate_frame": "processed_pixels", "visibility": "partially_occluded",
  "validity_ref": {"path": "evidence/valid-120.npz", "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", "media_type": "application/x-npz"},
  "derivation_refs": [], "review_status": "unreviewed"
}
```

This fragment omits the enclosing video provenance and coordinate-transform record, which are still required.

## 4. SceneState

The payload MUST contain `scene`, `scene_hash`, `status` and `validation_reports`.
`scene_hash` is SHA-256 of canonical JSON `{spec_revision, provenance, scene}`.
Reports bind this hash and their validator/configuration hashes, avoiding a circular report/manifest hash.
Status changes produce a new immutable SceneState envelope with the same scene hash when geometry/inputs agree.
For status-only revisions, the scene-generation `provenance` MUST remain byte-for-byte unchanged.
Such revisions MUST add `payload.review_lineage` with the previous envelope ref, review operation identity
and report refs; this field, envelope ID/time, status and reports are outside the semantic scene hash.
Do not replace the generating operation or scene parents with the status-review operation or prior envelope.
Changing any scene dependency changes that hash and makes previous validation inapplicable.

`scene` MUST contain episode/entity manifests, timeline ref, evidence ref, coordinate conventions, camera,
native humans, object assets/trajectories, support/contact, hypothesis lineage and execution provenance.
Preparation estimates and actual coupled-backend estimates MUST have distinct refs and roles.
Import a coupled backend's camera/human/depth/object outputs atomically; do not claim external estimates
controlled that run without recorded consumption. Material bootstrap/internal scale disagreement is unresolved.

Atomic import concerns one complete logical reconstruction, not one physical machine. The proposed
[distributed execution profile](distributed-execution.md) links native stage results by exact predecessor
refs, a logical run ID and per-stage source/model/configuration/environment identities. Validate the complete
dependency graph before publishing a full scene. Its proposed `StageResult` transport envelope is not an
implemented v2 record type, and the current foundation does not validate multi-host stage sequencing or
prove actual input consumption. Importing bytes must not grant report trust or promote scene status.

### Camera and native human

- Scene geometry uses metres in a named OpenCV camera frame (x right, y down, z forward).
  Any alternative backend/world convention requires a pinned, verified conversion into that scene frame.
- Camera records contain image size, intrinsics, distortion model, extrinsics when relevant, source evidence
  and crop mapping. Fixed camera is an explicit tested assumption; shared camera names do not verify intrinsics.
- Each human pins native `pose[T,136]`, static `scales[68]`, static `shape[45]`, original frame map,
  exact MHR rig/model assets, decoder code and native-to-scene conversion identities.
- Native 204 parameters are pose136 plus scales68; shape45 is separate. Native parameter units are decoder-defined,
  not uniformly metres/radians. No v1 concatenation, padding, renaming or metre-root copying is a conversion.
- Decoded geometry MUST be finite and in the declared scene frame. Preserve original native artifacts and
  measured conversion error; geometric fit from an off-manifold mesh is not automatically an exact conversion.
- Derived human geometry MUST cite the native content and decoder hashes; no independently optimized SOMA truth.

### Objects and trajectories

For every required object, pin mesh bytes, canonical origin/axes, topology, collision proxy and generation lineage.
`source_episodes` and `sharing_policy_ref` MUST record permitted contributors; default contribution is episode-local.
Typed symmetry is `none`, `finite` (explicit proper transforms) or `continuous` (axis/origin); record evidence
and distinguish geometric symmetry from texture/asymmetric features. A representative orientation is not observation.

`scale_mode` is `baked_metric` or `explicit_uniform`; trajectory geometry is always `P[t] = s * R[t] * V + t[t]`.
For `baked_metric`, stored vertices are metres and `s = 1`; provenance scale MUST NOT be applied again.
For `explicit_uniform`, vertices use declared canonical units and positive static `s` converts them to metres.
Translation is metres in both modes. Per-frame object scale is forbidden in this initial rigid-object contract.
Record anisotropic mesh preparation as a new mesh and canonical conversion, never as an undeclared pose scale.
Pose files MUST bind mesh hash, scale convention and canonical transform hash; rotations MUST be proper SO(3).
Canonical-origin changes require transforming mesh and pose together and invalidate trajectory-based checks,
even when world geometry is preserved. Origins MUST NOT be adjusted to improve an acceleration metric.

Trajectories contain all original frame IDs, finite translations/rotations, hypothesis IDs, evidence status
(`observed`, `inferred`, `unresolved`) and component uncertainty. Invisible frames still require explicit estimates.
Unresolved finite completion is permitted for execution validation but does not establish quality acceptance.
Geometry compilation/export MUST check thin parts, holes, contact surfaces and the actual compiled mesh/pose pair.

### Support, contact and operation evidence

Support records pin scene planes/patches and uncertainty. Contact edges name both entities and body/mesh patches,
inclusive frame intervals, evidence, state (`candidate`, `active`, `inactive`, `unresolved`) and constraint type.
Types are `grasp`, `sliding`, `body_support`, `foot_push`, `ground_support`, `rolling` and `no_contact`.
Record normal/tangential freedoms, softness and weight policy; sliding MUST NOT inherit a no-slip grasp rule.
Multiple edges may coexist; low speed alone cannot impose a hard static lock. Contact cannot override image evidence.
Each repair result MUST record input SceneState/EpisodeEvidence/SolvePlan refs, changed variable blocks,
actual budget, backend consumed-input manifest, outputs, residuals, unresolved intervals and parent scene hash.
Preparation/initialization instead pins its invocation and prepared-input manifest; it is not a repair operation.

### Status and transitions

`status` contains independent booleans `execution_valid`, `quality_accepted`, `export_valid`, plus `selection`
(`candidate`, `accepted`, `rejected`, `unresolved`). False means not established; reports distinguish fail/not-run.

| Transition / check | Required evidence |
|---|---|
| Candidate to `execution_valid` | Real permitted inputs, successful invocation, complete finite coverage, coherent units/identities, native decode and pinned dependencies |
| `execution_valid` to `quality_accepted` | Frozen review protocol, independent-within-declared-scope evidence, full-clip regressions and signed/recorded decision |
| `execution_valid` to `export_valid` | Exact scene export, pinned packer validation, reload/decode/render checks, complete expected manifest |
| Candidate to rejected/unresolved | Explicit failed/insufficient evidence; retain parent/rollback refs when a scene predecessor exists, otherwise the initial prepared-input/invocation manifest |
| Final candidate promotion | All three validity booleans true; complete challenge manifest; atomic promotion record |

Quality and export checks MAY run in either order; neither substitutes for the other. Execution-only baselines
remain valuable frozen baselines. They MUST NOT be described as accepted reconstructions or final submissions.
Each true flag requires a passing report for the same scene hash and current policy/configuration.
`selection: accepted` requires execution and quality acceptance; final export promotion additionally requires export validity.
Rejected/unresolved trials remain inspectable. A new attempt uses a new ID and never rewrites their status.

Illustrative status fragment after a real run with unresolved quality:

```json
{
  "status": {"execution_valid": true, "quality_accepted": false, "export_valid": false, "selection": "unresolved"},
  "validation_reports": [{"path": "reports/execution.json", "sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "media_type": "application/json"}]
}
```

## 5. SolvePlan

The payload MUST contain `input_scene_ref`, `evidence_ref`, `operation`, `entity_ids`, `intervals`,
`mutable_variables`, `fixed_variables`, `bounds`, `fit_partition_ref`, `review_partition_ref`, `budgets`,
`acceptance_config_ref`, `acceptance_config_hash`, `rollback_scene_ref`, `failure_action` and `unresolved_action`.
Refs are ArtifactRefs; the acceptance hash additionally pins canonical configuration JSON. No default may enlarge scope.
Variable blocks MUST be disjoint and cover all solver-accessible state; undeclared variables are fixed.
Native human variable slices require a decoder-pinned control map; do not guess native parameter indices.

| Operation enum | Mutable scope | Default fixed scope |
|---|---|---|
| `orientation_branch` | Object rotation candidates and explicitly declared coupled translation | Camera, mesh/scale, human |
| `contact_transition` | Contact states and declared local object/human root/body/hand blocks | Camera, mesh/scale, human shape/scales, unrelated intervals |
| `occlusion_recovery` | Object trajectory hypotheses in the declared interval and boundary overlap | Global geometry, camera, trusted anchors, human |

The initial repair registry MUST reject other operations; proposed enum support is not implementation.
Human edits require a separate evidence-supported `contact_transition` plan with explicit mutable blocks;
they MUST NOT be added to an `occlusion_recovery` plan through an authorization flag.
Each plan is for one operation and requires an execution-valid parent and an explicit rollback ref.
Initialization and geometry/camera/scale/representation changes are separate preparation revisions, outside
this repair enum. They invalidate consumers and trigger reinitialization; no hidden edits inside a pose repair.
Operator code/config identities MUST be pinned. A failed initial run records no valid baseline.

Budgets MUST bound candidate count, iterations, wall time and resource use; specify units and enforce limits.
Bounds MUST constrain changes from the parent, and repairs MUST declare boundary continuity checks.
The acceptance configuration MUST have a hash and contain metric definitions, sampling, tolerances,
allowed regressions, reviewed-decision criteria, review scope and required reports, frozen before fitting.
Uncalibrated diagnostics MUST NOT be presented as universal numeric quality thresholds or official scores.
The solver cannot change acceptance thresholds, partitions, weight policy or its own candidate budget.
Validation MUST rerun on the full clip and exact output scene, not merely the optimized interval.
Review-driven repeated tuning invalidates the claimed independence and must be recorded as development use.

On timeout, failed checks or insufficient evidence: preserve the parent, retain candidate/logs, record
`rejected` or `unresolved`, and do not auto-promote. An unresolved interval keeps explicit estimates/uncertainty;
it does not authorize dropping frames or silently asserting that a held pose was observed.
Rollback preserves execution validity only. Export MUST NOT implicitly fall back to an unaccepted baseline;
the exact selected parent must independently satisfy quality and export gates before final promotion.

Illustrative bounded operation fragment (its refs and full acceptance configuration are required separately):

```json
{
  "operation": "orientation_branch", "entity_ids": ["object-0"],
  "intervals": [{"start_frame_id": 120, "end_frame_id": 180}],
  "mutable_variables": ["object-0.rotation[120:180]"],
  "fixed_variables": ["camera", "human", "object-0.mesh", "object-0.scale", "object-0.translation", "object-0.rotation[outside:120:180]", "support_contact"],
  "bounds": {"max_rotation_change_rad": 3.141593},
  "budgets": {"max_candidates": 2, "max_iterations": 100, "max_wall_seconds": 300, "max_gpu_memory_bytes": 17179869184},
  "failure_action": "retain_parent", "unresolved_action": "retain_parent_and_record_uncertainty"
}
```

Bracketed ranges here use inclusive original IDs as declared above, not Python slicing semantics.

## 6. Dependency invalidation and required validators

Reuse MUST compare consumed content hashes plus producer code/model/config/decoder/policy identities.
File existence, stage completion, run names and nearest-upstream distance are insufficient.
Same-run overwrites MUST be detected before reuse; direct byte mutations are rejected, not adopted silently.
Import/export MUST use a single exact SceneState; assembling latest human/mesh/motion independently is forbidden.

| Changed dependency | Invalidated descendants |
|---|---|
| Video/decode/frame map or target association | Affected observations, preparation, scenes, plans, checks and exports |
| Camera/crop/coordinate adapter | Consuming projections, depth/scale, geometry/poses, contacts and checks |
| Observation or partition/weight/acceptance policy | Consuming plans/candidates, ranking and quality reports |
| Mesh/origin/scale | Bound trajectories, contact/collision, rendering, compiled geometry and export checks |
| Native human/rig/decoder | Derived geometry, contact, consuming object repairs, quality and export checks |
| Pose/contact/support | Consuming repairs, full-clip checks and exports |
| Packer or compiler | Compiled payload, reload/geometry checks and export validity |

Invalidation follows recorded consumers transitively; a changed unused artifact alone need not invalidate a scene.
Explicit permitted sharing expands invalidation to all consuming episodes. No automatic cross-episode pooling.
Rollback selects a still-valid parent by exact ref; stale ancestors must be regenerated before reuse.

Required validator groups: (1) envelope/path/hash/source policy; (2) decode timeline/entity/coverage;
(3) observation transforms/visibility/partition lineage; (4) native rig/units/geometry conversion;
(5) mesh/pose/scale/canonical frame; (6) contact/operation bounds; (7) dependency freshness;
(8) frozen quality decision/full-clip regression; (9) export/reload/compiled geometry and expected manifest.
Each report records validator revision, input hashes, checks, measured values, tolerances, failures and limitations.
Contract fixtures, real execution, independent quality review and export correctness are separate evidence types.
Migration acceptance requires negative cases for stale same-run files, wrong meshes/rigs, double scale,
cropped frame IDs, leaked review lineage, missing occluded frames and unknown provenance, plus real native
Track 1 round-trip evidence. The [foundation profile](v2-foundation.md) implements a subset with independent
synthetic tests; native decoding, operational source provenance, quality, official export and real migration
acceptance remain open. Its structural pass must not be reported as completion of this target specification.
