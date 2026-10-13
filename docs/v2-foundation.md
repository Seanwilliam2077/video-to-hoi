# Executable v2 foundation

Updated 2026-10-13. `v2hoi.v2` implements an **opt-in structural foundation** with `spec_revision: "2.0-draft.1"` and `implementation_profile: "p1-foundation.1"`. It is a concrete profile of [contracts-v2.md](contracts-v2.md), not a declaration that the complete proposed contract or migration gate is finished. [R1-R6](track1-requirements.md) remain authoritative.

The existing `v2hoi.contracts.CONTRACT_VERSION == 1`, fake stage registry and `v2hoi.run` execution path remain unchanged. Importing the v2 package does not register a native backend, run a model, decode MHR, invoke the official packer or accept a reconstruction. No v1 array concatenation or automatic v1-to-v2 migration is supplied.

## API and completion boundary

| Module | Implemented foundation | Still required for real acceptance |
|---|---|---|
| [`artifacts.py`](../src/v2hoi/v2/artifacts.py) | Strict JSON, canonical content hashes, verified file refs, immutable content/UUID publication and atomic compare-and-swap selection | Producer integration, recovery orchestration and a dependency scheduler; authenticity of the claimed evidence |
| [`contracts.py`](../src/v2hoi/v2/contracts.py) | `EpisodeEvidence`, `SceneState`, `SolvePlan`; explicit source-domain context, reference verification, structural and declared binding checks, report trust policy | Actual source video decoding, observation generation/review, native decoder/converter, meaningful quality/export validators and solver execution |
| [`geometry.py`](../src/v2hoi/v2/geometry.py) | Numeric array/coverage checks, proper rigid rotations, exactly-once object scale, world vertices, canonical rebase and affine pixel/intrinsic transforms | Native parameter semantics, model/rig compatibility, mesh topology/appearance, real camera calibration and reconstruction quality |

This is partial P1.1-P1.3 infrastructure. The [P0 audit](component-audit.md) leaves external/manual exposure history and runtime restoration unresolved. [validation-policy.json](validation-policy.json) freezes the initial structural/review protocol, while numerical quality thresholds and runtime-specific conversion tolerances remain unfrozen. No P2 baseline or G0-G4 acceptance gate is closed by these interfaces or synthetic tests.

## Store, references and canonical JSON

Construct `ArtifactStore(root)` on a local filesystem with atomic hard-link/rename and advisory-lock support. Network filesystems with weaker semantics are unsupported. Each `ArtifactRef` has **exactly** three fields:

```json
{
  "path": "objects/ab/abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789.json",
  "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  "media_type": "application/json"
}
```

These bytes/hashes are illustrative, not a usable fixture. `path` is a store-relative POSIX path; absolute paths, drives, backslashes, `..`, empty components, Windows device names, alternate streams and unsafe trailing characters are rejected. Internal symbolic links/reparse points are rejected. `sha256` is 64 lowercase hexadecimal characters. A bare MIME type is required; JSON readers accept `application/json` or a `+json` suffix. NPZ readers require `application/x-npz` and disable pickle loading.

`put_bytes(data, media_type, suffix)` stores a content-addressed immutable blob; `put_json(mapping)` first canonicalizes JSON. `read_bytes(ref)` recomputes the byte hash, and `read_json(ref)` also performs strict JSON parsing. Existing content with different bytes cannot be silently overwritten. These protections detect changed bytes; they do not authenticate a producer's statement about their origin.

`canonical_json_bytes(value)` uses pinned `rfc8785==0.1.4` for RFC 8785 JSON Canonicalization Scheme serialization. The dependency emits minimally encoded UTF-8 bytes and does not coerce non-string object keys. The repository additionally rejects duplicate keys on parsing, non-finite numbers, integers outside `[-(2**53-1), 2**53-1]`, lone surrogates and unsupported Python values. `content_hash(value)` is SHA-256 over the canonical bytes, not over pretty-printed JSON. See the maintainer's [rfc8785 package documentation](https://pypi.org/project/rfc8785/).

```python
from pathlib import Path
from v2hoi.v2.artifacts import ArtifactStore, content_hash

store = ArtifactStore(Path("work") / "v2-artifacts")
policy_ref = store.put_json({"policy_id": "example", "frozen": False})
assert store.read_json(policy_ref)["frozen"] is False
policy_hash = content_hash(store.read_json(policy_ref))
```

This example stores a policy blob only. It does not create scene validity or a usable SolvePlan acceptance configuration.

`publish_record(record, validate)` binds a canonical UUID v4 to one immutable validated value. A same-ID retry with identical bytes is idempotent; changed content requires a new ID. The callback must not mutate its input. `lookup_record(id)` returns the registered ref. `compare_and_swap_selection(name, ref, expected, validate)` updates a named pointer only when its old ref equals `expected`; `None` means that no selection may exist. The selected record must already be published. Low-level callers own the semantic callback; normal scene consumers should use `ValidationContext.select_scene`.

## Envelope and source-domain context

Every concrete record contains:

| Field | Concrete representation |
|---|---|
| `record_type` | `EpisodeEvidence`, `SceneState` or `SolvePlan` |
| `spec_revision`, `implementation_profile` | Exact strings above |
| `id` | Canonical lowercase UUID v4 |
| `created_at` | UTC RFC3339 string ending in `Z` |
| `provenance` | Source, producer, policy-review and dependency record below |
| `payload` | Type-specific object |

`provenance` requires `source_kind`, nonempty `source_refs`, `parent_refs`, `operation`, `code_revision`, `configuration_ref`, `configuration_hash`, `manual_edits`, `policy_review_ref`, `producer` and nonempty `dependencies`. `producer` contains `invocation`, nonnegative integer `seed`, `environment` and `nondeterminism`. `operation: "fake"` is rejected. `configuration_ref` identifies the actual configuration JSON; its canonical content hash must equal the lowercase SHA-256 `configuration_hash`. Parent refs must identify v2 records; their dependencies are recursively validated and cycles are rejected.

The referenced source-policy review contains `decision`, `source_kind`, `source_hashes`, `reviewer` and `limitations`. Its decision must be `approved`, its domain must match the context, and its source hashes must cover the exact declared `source_refs`. This verifies the declaration's bindings, **not its truth**: an approved string or matching hash cannot prove that data has permitted provenance. The calling application must review the actual acquisition/derivation chain and must not create an approval merely to satisfy the parser.

`ValidationContext(store, source_kind="track1", trusted_reports=())` keeps real Track 1 and `independent_synthetic` domains separate. Unknown/other domains are rejected. Independent synthetic fixtures may exercise identical interfaces in their own context; they cannot enter the Track 1 context or certify real native decoding, source provenance, quality or submission compliance. Declared artifact references are byte-verified transitively, including references inside supplemental JSON documents. Referenced v2 records undergo their own source-policy and semantic validation; JSON dependency cycles are rejected. Supplemental metadata is permitted, but verified bytes do not mean a specialized semantic validator exists for every supplemental field.

## EpisodeEvidence payload

The required fields are `episode_id` (integer 0-29), `video` (reviewed source ArtifactRef), `timeline` (JSON ArtifactRef), `entities`, `observations`, `events`, `partitions` (JSON ArtifactRef) and `evidence_policy_ref`.

The timeline JSON contains `frame_count`, `frame_ids`, `pts`, `time_base`, `width`, `height` and `decode_ref`. IDs must be the complete sequence `0..T-1`; original presentation timestamps are increasing integers with positive rational `time_base: [numerator, denominator]`. Sizes and count are positive integers. These checks validate a declared timeline; an actual decoder still has to establish its correspondence to the original bytes.

Entities contain `id`, `role` and `evidence_ref`. Exactly one `target_human` and at least one `target_object` are required; `context` entities may also exist. IDs are episode-local and unique. A source-video actor association still requires independent review.

Each observation requires:

- `observation_id`, `entity_id`, sorted unique original `frame_ids` and `kind` (`pixels`, `mask`, `keypoints`, `depth`, `flow`, `feature_matches`, `landmarks`, `metadata`).
- `origin` (`raw`, `model_prediction`, `human_annotation`), `data_ref`, `producer`, `coordinate_frame` (`source_pixels` or `processed_pixels`), `validity_ref` and `visibility` (`visible`, `partially_occluded`, `fully_occluded`, `out_of_view`, `unknown`).
- `derivation_refs` for supporting artifacts, `derived_from` for parent observation IDs, and nonempty `lineage_roots`. Parent-observation cycles are rejected.
- `image_transform` containing `source_size`, `processed_size`, `pixel_centres` and `T_processed_from_source`. Sizes use `[width, height]`; this profile checks an invertible affine 3-by-3 transform and matching source dimensions. Nonlinear warp support and actual landmark round-trip review remain later work.

The partition JSON contains `fit`, `review`, `diagnostic` and `unused` lists of observation IDs, plus `scope` (`local_repair` or `baseline_review`), `selection_config_ref`, `selected_at`, `temporal_exclusion_frames`, `limitations` and Boolean `whole_video_model_consumed_review`. Every observation belongs to exactly one role. Fit/review separation checks declared lineage roots, observation/derivation byte hashes, parent observations and frame separation including the declared temporal buffer. It does not discover undeclared shared pixels, unseen manual use or correlated estimator errors. Whole-video consumption requires a stated limitation and never creates an unseen-data claim.

Events contain `id`, `type`, `entity_ids`, `interval`, `state` and `evidence_refs`. Event `type` is `grasp`, `slide`, `support`, `foot_push`, `release`, `rolling` or `visibility_transition`. Intervals use inclusive original `start_frame_id` and `end_frame_id`. Event state is `candidate` or `unresolved`; text-derived event proposals do not become contact truth. Multiple relations may coexist.

## SceneState payload

The payload contains `scene`, `scene_hash`, `status` and `validation_reports`. Compute its semantic identity with `scene_hash(record)`, which hashes canonical `{spec_revision, provenance, scene}`. The envelope ID/time, status and report list are outside this hash. Changing generating provenance or scene dependencies changes the semantic identity.

`scene` requires `episode_id`, `evidence_ref`, `timeline_ref`, `coordinate_frame: "opencv_camera_m"`, `camera_ref`, `humans`, `objects`, `support_contact_ref`, `hypothesis_lineage_ref`, `execution_manifest_ref`, `preparation_refs` and `dependencies`. Evidence episode and timeline bindings must agree, and every target entity appears exactly once. Preparation estimates remain separate from actual native outputs.

| Referenced or embedded record | Concrete required contents |
|---|---|
| Camera JSON | `width`, `height`, `K`, `distortion`, `evidence_ref`; source dimensions, finite 3-by-3 intrinsic matrix, positive focal lengths |
| Human descriptor | `entity_id`, `native_ref`, `rig_ref`, `decoder_ref`, `conversion_ref`, `native_units: "decoder_defined"` |
| Human NPZ | `pose[T,136]`, `shape[45]`, `scales[68]`, integer `frame_ids[T]`; finite arrays and exact full-frame coverage |
| Conversion binding JSON | `rig_sha256` and `decoder_sha256` equal the pinned refs; this is a binding check, not execution of that decoder |
| Object descriptor | `entity_id`, `mesh_ref`, `trajectory_ref`, `canonical_transform_ref`, `scale_mode`, `scale`, `source_episodes`, `symmetry`, `pose_binding`, `evidence_status`, `uncertainty_ref` |
| Mesh NPZ | `vertices[V,3]`, nonempty integer triangle `faces[F,3]` with in-range indices; this CPU profile uses NPZ, not direct GLB parsing |
| Trajectory NPZ | `rotations[T,3,3]`, `translations[T,3]`, integer `frame_ids[T]`; complete finite rigid trajectory |
| Exact `pose_binding` | `mesh_sha256`, `canonical_sha256`, `scale_mode`, `scale`, matching the descriptor |
| Canonical transform JSON | `T_new_from_source`, a finite proper rigid 4-by-4 homogeneous transform; scale, shear and reflection are rejected |
| Object source/ambiguity | `source_episodes == [episode_id]`; `symmetry.type` is `none`, `finite` or `continuous`; per-frame `evidence_status` is `observed`, `inferred` or `unresolved` |
| Execution manifest JSON | `job_id`, `backend`, `source_kind`, `timeline_ref`, `inputs`, `object_inputs`, `outputs`, `state`; coupled camera/human/trajectory/contact hashes belong to outputs and metric meshes to inputs; the exact scene `evidence_ref` must appear in inputs |

`execution.object_inputs` must equal the ordered list `[{"entity_id": obj["entity_id"], **obj["pose_binding"]} for obj in scene["objects"]]`. This binds the invoked object's mesh, canonical frame and scale together, beyond checking that its mesh bytes appeared somewhere in the invocation.

For finite symmetry, provide a nonempty `rotations` array of proper 3-by-3 rotations and `evidence_ref`. Continuous symmetry requires a finite unit `axis`, finite `origin` (both three-component vectors) and `evidence_ref`. The validator checks the declared representation and numerical properties; verifying that the real object has that symmetry remains an evidence-review task.

`scale_mode` is `baked_metric` (scale exactly one) or `explicit_uniform` (one positive static scale). Rotations must be proper SO(3), checked numerically at absolute tolerance `1e-6` with zero relative tolerance. These math checks do not calibrate physical object size or validate native parameter units. Per-frame scale and cropped/missing original frame maps are rejected.

The support/contact JSON has `supports` and `edges`. A support records `id`, `geometry_ref`, `uncertainty_ref`, `evidence_refs`. An edge records `id`, two `entity_ids`, `type`, `state`, inclusive `interval`, `evidence_refs`, two `patches`, `normal_freedom`, `tangential_freedom` and `weight_policy_ref`. Types are `grasp`, `sliding`, `body_support`, `foot_push`, `ground_support`, `rolling`, `no_contact`; states are `candidate`, `active`, `inactive`, `unresolved`. Freedoms are `free`, `soft` or `constrained`; a sliding edge cannot constrain tangent motion as no-slip contact. This checks representation consistency, not actual physical contact.

## Reports and promotion

`status` has separate Boolean `execution_valid`, `quality_accepted`, `export_valid` and a `selection` of `candidate`, `accepted`, `rejected` or `unresolved`. Quality/export require execution validity. Accepted selection requires both execution and quality; final-export selection additionally requires export validity.

A report JSON contains `kind` (`execution`, `quality`, `export`), exact `scene_hash`, Boolean `passed`, `validator_ref`, `configuration_ref`, nonempty `checks` and `limitations`. Duplicate report kinds and stale scene hashes are rejected. In a passing report, every check must be an object with `passed: true`; failed or unrun checks cannot be hidden behind the top-level flag. Each true status flag needs a passing report and an **exact caller-trusted triple**:

```python
(report_kind, validator_sha256, configuration_sha256)
```

The context does not execute or authenticate that validator and cannot judge whether its check list is truthful. Trust is an external reviewed caller policy, not a flag obtained by adding a passing JSON report. The default empty trust set allows candidate inspection but cannot establish true validity flags. Execution validity also requires the bound invocation state to be `complete`.

An initial scene starts with all three validity flags false. Any true flag requires `payload.review_lineage` pointing to an immutable prior SceneState. For each status-only revision, retain the same generating `provenance` and scene hash, assign a new UUID, and provide `previous_ref`, nonempty `operation` and `report_refs` exactly equal to the current `validation_reports` list. Recursive review validation must terminate at the initial all-false scene; a later revision may refer to a previously reviewed scene. A status review must not replace the generating operation. Rejected and unresolved records remain immutable.

```python
from v2hoi.v2.contracts import SceneState, ValidationContext

# scene_document and reviewed_report_identities come from the caller's importer
# and independently reviewed validator policy; neither is synthesized here.
context = ValidationContext(store, source_kind="track1",
                            trusted_reports=reviewed_report_identities)
scene = SceneState.from_dict(scene_document, context)
scene_ref = scene.publish(context)

# This succeeds only for an accepted scene with all three trusted report kinds.
prior_ref = store.get_selection("submission-candidate")
context.select_scene("submission-candidate", scene_ref,
                     expected=prior_ref, require_export=True)
```

The general record API is `from_dict(value, context)`, `load(ref, context)`, `to_dict()` and `publish(context)`. `to_dict()` returns a copy. A concurrent selection change causes compare-and-swap failure; retry requires deliberate fresh review. Promotion does not submit anything externally.

## SolvePlan payload

The concrete required fields are `input_scene_ref`, `evidence_ref`, `operation`, `entity_ids`, `intervals`, `mutable_variables`, `fixed_variables`, `bounds`, `fit_partition_ref`, `review_partition_ref`, `budgets`, `acceptance_config_ref`, `acceptance_config_hash`, `rollback_scene_ref`, `failure_action`, `unresolved_action`, `operator_ref`.

The parent must be execution-valid in the same context. `rollback_scene_ref` must equal that exact parent ref. Both partition refs point to the **same** evidence partition manifest, which must declare `scope: "local_repair"` and nonempty fit/review sets. The allowed operation names are `orientation_branch`, `contact_transition`, `occlusion_recovery`; this registry does not execute them.

Mutable variables are objects, not illustrative strings from earlier design fragments:

```json
{
  "operation": "orientation_branch",
  "entity_ids": ["object-0"],
  "intervals": [{"start_frame_id": 120, "end_frame_id": 122}],
  "mutable_variables": [
    {"entity_id": "object-0", "block": "rotation", "frame_ids": [120, 121, 122]}
  ],
  "fixed_variables": ["camera", "mesh", "scale", "human_shape", "human_scales", "human", "all_undeclared"],
  "bounds": {"max_rotation_change_rad": 0.5},
  "budgets": {"max_candidates": 2, "max_iterations": 20, "max_wall_seconds": 60, "max_gpu_memory_bytes": 1073741824},
  "failure_action": "retain_parent",
  "unresolved_action": "retain_parent_and_record_uncertainty"
}
```

This is a fragment, not a complete executable plan. Refs, envelope and acceptance configuration are still required. IDs must be sorted/unique, within the source timeline and declared inclusive intervals. Duplicate mutable entity/block/frame assignments and fixed/mutable overlap are rejected.

All plans fix `camera`, `mesh`, `scale`, `human_shape`, `human_scales` and `all_undeclared`. Optional `human` fixes all human blocks; exact `<entity_id>.<block>` entries fix individual blocks. Object blocks are `rotation`/`translation`; `contact_transition` additionally permits object `contact` and human `root`/`body`/`hand`/`contact`. Human non-contact edits require `control_map_ref`. Its JSON contains `decoder_sha256`, `rig_sha256`, `block`, nonempty sorted unique `indices` in `0..135`, nonempty `units` and `review_ref`; decoder, rig and block must match the scene's human and requested block. These bindings cannot establish that the chosen native indices actually control the intended anatomy; that requires the pinned decoder review. Orientation plans require a rotation block. Global geometry, camera or scale changes require a new preparation revision.

Bounds must include the corresponding nonnegative finite value: `max_rotation_change_rad`, `max_translation_change_m`, `max_native_control_change` or `max_contact_changes`. All four budget fields above are required positive exact integers. Runtime enforcement and measured deltas remain the future solver's responsibility.

The acceptance JSON requires `frozen`, `metrics`, `tolerances`, `allowed_regressions`, `review_criteria`, `full_clip`, `boundary_checks`, `required_reports`, `limitations`; `frozen`, `full_clip` and `boundary_checks` must be true. Its canonical hash must equal `acceptance_config_hash`. This verifies a declared policy and does not establish that its metrics/tolerances are valid. The initial repository policy intentionally leaves numerical quality acceptance unresolved, so it is not itself a ready repair acceptance configuration.

## Geometry helpers and verification

`validate_native_human` checks shape, finite values and complete source frame IDs only. It cannot prove a compatible rig, meaningful pose values, unit conversion or native decoding. `validate_object_trajectory` checks the finite rigid-object arrays, static scale and complete frame map. `world_object_vertices` computes `P[t] = s R[t] V + t[t]`; it cannot inspect artifact lineage.

`rebase_object_canonical` accepts only a proper rigid canonical transform, transforms mesh and poses together, and returns a `RebasedObject` marked `requires_revalidation=True`. It preserves world geometry numerically; callers must produce new bindings and invalidate contact/trajectory/export checks. It must not be used to manipulate an acceleration metric. `validate_image_transform`, `transform_image_points` and `transform_intrinsics` support explicit affine pixel maps with `K_processed = T_processed_from_source @ K_source`; they do not infer crops, undistort lenses or estimate calibration.

Run the complete explicit CPU test allowlist from [`.github/workflows/tests.yml`](../.github/workflows/tests.yml), not a bare `pytest`. Its current command, from the repository root in the development environment, is:

```text
python -m pytest -q tests/test_bundle.py tests/test_contracts.py tests/test_dataset.py tests/test_fetch_assets.py tests/test_download_foundationpose.py tests/test_model_release_prepare.py tests/test_model_release_restore.py tests/test_handoff_restore.py tests/test_benchmark_first3.py tests/test_rank_first3.py tests/test_geometry.py tests/test_metrics.py tests/test_track1_score.py tests/test_run.py tests/test_v2_artifacts.py tests/test_v2_geometry.py tests/test_v2_contracts.py
```

The independent synthetic fixtures in [`test_v2_artifacts.py`](../tests/test_v2_artifacts.py), [`test_v2_geometry.py`](../tests/test_v2_geometry.py) and [`test_v2_contracts.py`](../tests/test_v2_contracts.py) are the executable examples of this profile. They test failure boundaries without challenge inputs or model inference. Storage integrity, strict declarations and synthetic geometry identities are useful foundations; real video evidence, authentic native outputs, independently reviewed reconstruction quality and an exact export/read-back remain required.
