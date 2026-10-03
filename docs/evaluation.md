# Track 1 evaluation mechanics and evidence

The single active requirements are [R1–R6](track1-requirements.md), supplied by the owner on **2026-10-03**. This document records external implementation evidence and local diagnostic limits; it cannot override those requirements. The official submission kit was available on **2026-10-02** and its allowlisted code was reverified on **2026-10-03**. No reconstruction result or official score is reported here.

Use only permitted Track 1 inputs and independently created synthetic fixtures. No Track 2 content, scores, statistics, cached outputs or derived parameters may be used to implement, tune, validate or normalize this work. The former scorer's defaults and normalization are historical and superseded; they must not be restored in the native-MHR reference diagnostic.

## Owner-confirmed requirements, October 3

Follow [the current requirements](track1-requirements.md). Earlier notes are superseded wherever they conflict. Requirement authority comes from the owner's latest instruction, not from choosing between historical documents.

**Frame mapping is an implementation responsibility.** R1 fixes alignment to the first reference frame's explicit original-video ID. A selected scoring window may begin later and must retain that alignment. The inspected external code fits the first entry in its scored arrays; an adapter must verify what original reference frame that entry represents and preserve R1. This observation is not a competing rule and does not permit replacing the reference first frame with raw frame zero or a convenient visible frame.

## Sources and verification boundary

- [Official challenge page](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/): submission entry point and public context.
- [Official FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf), Track 1 section on page 2: MHR, first-frame human alignment, reference-relative acceleration, continuous object trajectories, and the Track 1 data boundary. Read on October 2; the October 3 browser fetch failed, so no later FAQ revision is asserted.
- [Official submission kit](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip): the member paths and line numbers below refer to the exact versions recorded in [the evidence manifest](evidence/official-track1-20261002.json).
- [Official Track 1 dataset README](https://huggingface.co/datasets/nvidia/video_to_data_challenge/blob/main/track_1/README.md): 30 videos, 10 objects with three episodes each, and no public reconstruction ground truth.

The archive was inspected with HTTP Range requests for its directory and an explicit code/documentation allowlist. No entire-archive download, sample payload, model execution, or held-out reference read was needed. Each inspected member has a SHA-256 of its exact uncompressed bytes. The manifest deliberately leaves the whole-ZIP hash null. An ETag, modification date, or source filename is not a cryptographic identity.

The public `metric_code/README.md:3-6` says the scorers omit host reference values and the MHR model asset. The generated Track 1 scorers contain an empty body-asset slot. Their mathematical definitions are inspectable; complete local host equivalence and official Track 1 scores are not established. The sample's row set, host role-index arrays, and model version are separate dependencies, not values to infer from an old local schema.

## Current implementation and planned work

| Component | Current repository behavior | Required next implementation |
| --- | --- | --- |
| Pipeline | Fake reconstruction stages and internal export exercise wiring | One real Track 1 baseline, then incremental adapters |
| Human contract | Parallel SOMA-X and wrapper `mhr_*` arrays; shape/finite checks | Native MHR parameters as the authoritative state; explicit tested conversions |
| Export | `stages/export.py` writes the internal layout and separate MHR arrays | Adapter to the pinned official packer, with round-trip geometry checks |
| Metrics | Generic primitives plus a native-MHR reference diagnostic with explicitly supplied decoder/reference | Validate real authorized MHR decoding and pinned official sampling/serialization; do not claim official equivalence |
| Comparison | `rank_first3.py` checks submitted records for three hoop episodes | Independently produced observations, machine-generated measurements, and broader grouped evaluation |
| Provenance | Run paths and invocation history; partial stale-output detection | Immutable artifacts and hashes covering every dependency |

These are implementation requirements. Publishing this document does not mean those capabilities are complete. Passing the existing synthetic CI allowlist does not certify reconstruction quality or an official submission.

## Local native-MHR reference diagnostic

`v2hoi.score` is a replacement diagnostic interface, not the official evaluator or a submission adapter. Its old reference defaults, alternative alignment modes and normalized total are removed. It requires explicitly supplied permitted prediction/reference manifests, native parameters, canonical meshes, body-role indices and a local authorized decoder. Nothing in the command downloads or supplies a hidden reference or body model.

```bash
python -m v2hoi.score --pred prediction.json --reference reference.json --decoder your_package.mhr:decode --roles roles.json --out report.json
```

`your_package.mhr:decode` is a placeholder for a locally implemented, verified decoder. It accepts `v2hoi.score.NativeMHR` and returns `DecodedHuman(vertices, joints)` in metres in one consistent scene frame with corresponding topology. The caller must preserve the rig's native parameter meanings and document its asset/version. An arbitrary synthetic decoder is useful only for independently constructed synthetic tests.

Each manifest has `schema_version: 2`, `provenance: {"kind": "track1" | "independent_synthetic", "evidence": "..."}`, `required_objects`, `expected_frames`, and `episodes`. `expected_frames` maps each episode ID to integer `start` and `count` in **original-video frame coordinates**; `count` must be at least three. `episodes` has exactly those episode IDs. Each entry declares `object_id`, `artifact` (native NPZ path), `mesh` (canonical triangle-mesh path), optional `sample_count` (default 2048) and optional explicit provenance. Paths are resolved relative to their manifest. Prediction and reference must have the same declared timeline and object coverage. The declared timeline must itself be verified against the source video before any complete-coverage claim.

The diagnostic CLI accepts self-contained **GLB, PLY and STL** meshes. It reads mesh bytes in memory with no external resolver and rejects external buffer/texture declarations and sidecar formats. This is deliberately narrower than the official packer's format support listed below. JSON rejects duplicate keys and nonfinite constants; NPZ uses `allow_pickle=False`. Explicit fake/smoke declarations cannot be labelled Track 1, and a report cannot mix Track 1 and independent synthetic provenance. The output path cannot overwrite an input manifest, role file, mesh or native artifact.

| Native diagnostic NPZ field | Required content |
| --- | --- |
| `pose`, `scales`, `shape` | `[T,136]`, `[68]`, `[45]`; native values, not decomposed v1 wrapper fields |
| `frame_indices` | All consecutive original frame IDs declared by the manifest, in order |
| `object_poses` | `[T,4,4]` finite proper rigid transforms in the common scene frame |
| `object_scale` | Positive scalar applied exactly once to canonical mesh vertices |
| `object_visible` | Optional `[T]` boolean metadata; invisibility never removes a frame |

`roles.json` declares `alignment_vertices`, `surface_vertices`, `body_joints22` (exactly 22 explicit joint indices), `hand_vertices`, and a nonempty `source_id`. No host role indices are guessed or supplied by default. The internal `Reconstruction` API can accept explicitly described canonical diagnostic point samples; the CLI requires a mesh and derives its samples internally.

The CLI samples triangle surfaces with a deterministic episode-derived seed, records the mesh hash, seed and count, and constructs object geometry as `s*R*V+t`. It fits one human-only Sim(3) at the first reference ID and applies it to the complete human/object scene. The programmatic `score_episode` API can select later `scored_frame_indices` without changing that initial alignment. The CLI evaluates every declared frame. ACC uses original consecutive triplets, reference-relative body-joint/object-translation second differences and cm/frame² units. Results aggregate by equal episode means, with no composite total.

PEN uses the predicted canonical mesh, triangle distances and absolute winding-number sign, then the object and shared alignment scales; exterior hand points contribute zero to the mean. Zero-area padding faces are excluded. Open or self-intersecting meshes retain winding-number ambiguity, and this CPU calculation is not certified against the host GPU evaluator. The programmatic API reports PEN unavailable when no signed-distance function is supplied; a dataset report does not compute a partial PEN mean when any episode is unavailable.

Provenance declarations and prohibited-path checks run before payload loading and decoder import, but **declarations are not authenticated provenance**. The report explicitly sets `official_equivalence: false`. It does not verify hidden host roles, body assets, official sampling/mesh compilation or an actual authorized MHR model; synthetic tests do not fill those gaps. Public Track 1 reconstruction ground truth remains unavailable, so this interface does not create local official scores. Use independently constructed synthetic references for mathematical tests and independent video evidence for real Track 1 development.

## Official packer inputs and final payload

The kit's `tools/pack_reconstruction.py:16-27,125-161` consumes one `episode_XXXXXX.npz` and one `episode_XXXXXX_object` mesh file. Array index `t` is the original video frame, not the position in a compacted scored-frame array.

| NPZ field | Shape | Meaning |
| --- | --- | --- |
| `pose` | `[T,136]` | Native MHR model parameters `0:136`, including root translation, root rotation, and articulated pose |
| `scales` | `[68]` | Native MHR model parameters `136:204`, constant across the episode; not one scalar metric scale |
| `shape` | `[45]` | Constant identity coefficients |
| `object_rotation` | `[T,3,3]` | Proper rotation from object coordinates to the scene |
| `object_translation` | `[T,3]` | Object-coordinate origin in the metric scene |
| `object_scale` | scalar | Positive multiplier applied to the object mesh |

The mesh suffix can be `.glb`, `.obj`, `.ply`, `.stl`, or `.off`. The packer derives required episode/frame rows and mesh budgets from the official sample; do not hard-code them from a historical example. It produces one Parquet with `row_id`, `x`, `y`, `z`, and `code_commit_url`, used for all five Track 1 metric competitions. The commit must identify the generating code with a full GitHub commit URL, identical throughout the payload. Masks and depth are internal evidence, not fields of this payload. See `tools/pack_reconstruction.py:92-103,110-122` and `tracks/track_1/README.md:3-23`.

Parameter arrays are packed three numbers per row. The unused tail must be zero; mesh face rows contain integer indices represented as numeric coordinates. Rotations are validated and projected to SO(3). This transport representation is distinct from the episode NPZ and from the repository's current internal exporter. The inspected packer does not add a general numeric quantization step. See `v2dlb/mhr_submission.py:128-152`.

### Native MHR units and conversion

The existing `mhr_global_rot`, `mhr_body_pose`, `mhr_hand_pose`, `mhr_scale`, and metre-valued `mhr_transl` fields do **not** establish a mapping to the official native 204-parameter vector. Do not concatenate them, truncate them, or copy metre translation into the native root parameters.

The official converter distinguishes native pose parameters from the geometry they produce. Its `tools/track1/mesh_to_mhr_params.py:227` identifies one native root-translation parameter unit as 100 mm; `:294-352` fits MHR-topology vertices and returns pose, scales, shape, and fit diagnostics. Its expected input is `[T,18439,3]` vertices in metres. This is an optimization-based conversion, not proof of an exact mapping for arbitrary off-manifold meshes.

The scorer's native forward path applies the model's parameter transform before skeletal transforms and skinning. The resulting native vertices/joints are divided by 100 and multiplied by `diag(1,-1,-1)` to reach the metric scene (`metric_code/track_1/ACC-H.py:516-546`). Scale parameters also pass through the native model; they are not independently interpretable as object scale factors.

Prefer preserving the actual native model parameters from a compatible backend. Otherwise use a verified model/converter and measure the geometry after conversion and after serialization. Check coordinates, hands, constant identity, invalid frames, and per-frame vertex error. The converter's fit claims are not measurements of this project's outputs.

## Alignment, frames, and reductions

**Project behavior follows R1:** fit one Sim(3) from human geometry at the reference's explicit first original frame, then apply it to both human and object for all frames. The inspected external implementation fits corresponding role-selected human vertices at the first scored-array entry (`metric_code/track_1/ACC-H.py:397,503-504,634-655,679-686`). Preserve the original-frame mapping when adapting that code; selecting later diagnostic rows must not refit alignment. Host alignment-role indices belong to its body asset and are not exposed by the public code alone.

`v2dlb/mhr_submission.py:34-37` describes scored frames within the hand-object contact span, separated into contiguous stretches. The official sample determines the actual row set. This audit has not parsed that sample, so exact scored first frames, episode membership, and budgets remain unverified. Preserve complete video trajectories, including occlusion, as the FAQ requires; inspect both whole-video behavior and the eventual sample-defined windows.

Acceleration is evaluated only within consecutive-frame stretches, with no difference across a gap. The implementation rejects stretches shorter than three frames. Within an episode it averages valid differences; across episodes it averages episode metrics equally. A long episode therefore does not receive extra weight at the final reduction. See `metric_code/track_1/ACC-H.py:672-697,989-1010`.

## Metric definitions

All five metrics are lower-is-better. Do not infer a combined leaderboard point formula from that fact or manufacture a local total.

### Chamfer distance

CD is the **sum** of the two directed mean nearest-neighbor Euclidean distances, converted from metres to centimetres. It is neither squared distance nor half that sum (`metric_code/track_1/CD-H.py:246-258`).

CD-H uses the body's selected surface-role vertices. CD-O uses area-weighted samples from the submitted, budgeted object mesh, with the sample count set by the body role and a seed derived from episode identity. The points are posed with the submitted object transform and then the common human alignment. Do not substitute all human vertices, a different sampler, or an independently aligned object and call the result official. See `metric_code/track_1/ACC-H.py:441-453,652-658`.

### Acceleration error

For trajectory `x`, let `D2(x)[t] = x[t+1] - 2*x[t] + x[t-1]`. ACC measures the mean norm of `D2(prediction) - D2(reference)` after the common alignment. ACC-H uses 22 selected body joints; ACC-O uses **object translation**, not its centroid or angular acceleration. The units are **cm/frame^2**, with no FPS-squared multiplier. See `metric_code/track_1/ACC-O.py:261-280` and `metric_code/track_1/ACC-H.py:655,683-695`.

Prediction-only acceleration is a useful local diagnostic but cannot estimate reference-relative ACC. A frozen trajectory can have zero self-acceleration while missing genuine motion. Report `self_acceleration` separately; when reporting m/s^2, explicitly apply the sampling interval and retain a separate frame-based value. Fingers need dedicated diagnostics even though this official ACC-H definition selects body joints.

### Hand-object penetration

PEN transforms the predicted hand points into the submitted object's mesh frame, computes their nonnegative interior depths, and converts those depths using both object scale and human alignment scale. A point outside contributes **zero**, and the denominator includes **all selected hand points**, not only penetrating points. The final value averages over frames and then episodes. The reference affects this definition only through the alignment scale; the metric does not subtract reference penetration. See `v2dlb/mhr_submission.py:450-466` and `v2dlb/mhr_metrics.py:320-337`.

For hand count `N`, frame count `F`, local depth `d[t,i] >= 0`, object scale `s`, and alignment scale `a`, the episode formula is `100*a*s*sum(d)/(F*N)`. All-exterior hands yield zero, but that alone does not prove contact or accurate reconstruction. A hand-object separation failure must still be caught by image fit, contact evidence, and CD.

The inside test uses absolute winding number greater than 0.5 and distance to triangle surfaces; it excludes zero-area padding faces (`v2dlb/mhr_submission.py:401-430`). This is different from the repository's surface-sample normal vote and maximum-depth primitive. Whole-body, foot, seat, and support penetration remain useful separate diagnostics, not this official PEN.

## Mesh scale, origin, and export effects

The public pose convention is `scene_point = object_scale * R * mesh_point + translation`. `v2dlb/mesh_budget.py:27-39` loads, welds, simplifies to the sample's budget, then pads; it does not normalize the object origin or scale. If vertices are already metric, an adapter may use scale 1. It must not apply the stored mesh provenance scale a second time. Validate the actual compiled mesh, including thin structures, holes, orientation, and contact surfaces; checking only the original high-resolution mesh is insufficient.

ACC-O's use of translation makes the canonical origin material. Replacing vertices by `V-c` while replacing translation by `t+s*R*c` preserves world geometry but can change translation acceleration when rotation varies. No canonical-origin requirement was found in the inspected sources. Freeze and record the object's origin and every canonical transform; do not optimize an origin to improve a score. Request clarification before asserting official origin invariance.

The payload carries a mesh and scale per episode. It does not enforce one shared mesh across an object's clips. A shared-object reconstruction may be an algorithmic choice, but cross-clip fitting permission is not established by that schema. Keep a per-clip mode available until the applicable rule is confirmed.

## Superseded history and current limits

| Issue | Interpretation for this project |
| --- | --- |
| Old repository notes say the official script is unavailable | Superseded: the kit exists; the repository adapter is still missing |
| Old local Chamfer averages the two directions | Preserve its identity as a local primitive; do not compare its number directly to official CD |
| Old notes call self-acceleration the official smoothness term | Superseded by reference-relative second differences |
| General descriptions compare penetration with the reference | The inspected PEN code uses own-mesh hand depth and reference alignment scale only |
| `mhr_submission.py:25-28` suggests an identical reference copy gives all-zero scores | That is not universal for PEN: a reference pose with nonzero self-penetration remains positive under the inspected formula; this is a mathematical observation, not a data experiment |
| Code-only scorers are public | Host reference/body assets and role arrays are omitted; exact local official scores remain unavailable |
| Official packer returns a file | This establishes packaging, not image fidelity, data provenance, valid licenses, or geometric accuracy |

R1–R6 are settled project requirements. Version external adapters and record technical mismatches without reinstating historical rules as active alternatives. Never use an unavailable reference, guessed host role list or historical Track 2 result to close an implementation gap.

## Planned validation and candidate decisions

Use these checks in order. Synthetic diagnostic coverage is engineering evidence; full pipeline, native-export and official-equivalence gates remain open:

1. **Contract and serialization:** analytic synthetic camera/mesh/trajectory cases, native-MHR FK conversion checks with authorized model assets, original frame indices, constant identity, proper rotations, finite values, all-frame coverage, and compile/read-back equivalence. Freeze the official sample and packer identity before using them.
2. **Mathematics:** independent synthetic examples for sum-versus-mean Chamfer, rigid/global scale transformations, reference-first alignment followed by later scoring windows, second differences with gaps, rotating objects with fixed canonical origins, and penetration inside/outside a known mesh. Test padding and export simplification. Use the replacement diagnostic with explicit permitted references; do not recreate historical Track 2 defaults, statistics or normalization.
3. **Independent Track 1 evidence:** freeze video hashes, annotation frames, visibility policy, prompts, observed landmarks/masks, evaluators, and review windows before comparison. Model predictions are observations, not labels. An optimizer and its evaluator must not share the same predictions as an unquestioned truth source.
4. **Controls:** a copied-first-frame trajectory must fail motion fidelity; lagged motion must fail timing; detached hands must fail contact evidence; missing difficult frames must fail coverage. Smoothness or penetration improvements cannot waive these controls.
5. **Broader review:** keep all three clips of an object in the same development/holdout group. Do not fit shared meshes or camera priors using a held-out group. After all clips have informed choices, call the set a regression set, not unseen validation. Report object-level and event-level results, tails, failures, manual intervention, and runtime. Three hoop clips support only a hoop-subset claim.
6. **Selection and provenance:** generate metrics from artifacts, verify referenced evidence exists, and preserve an immutable dependency hash graph. Compare fidelity, motion/contact plausibility, failure rate, and cost as separate axes. Set tolerances before viewing candidate results; use evidence from multiple episodes rather than an invented official composite score.

`rank_first3.py` currently verifies metadata and reviewer declarations, not whether the evidence proves a claim. A lower proxy number is therefore a hypothesis to examine, not proof of a leaderboard improvement. Official scores, when received from the organizer, must be recorded separately with their submission receipt and exact generating commit.
