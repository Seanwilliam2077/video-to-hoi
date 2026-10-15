# Implementation plan: Track 1 baseline and bounded repair

Updated 2026-10-13. This delivery backlog implements the video-driven architecture in [design.md](design.md), the proposed [v2 contracts](contracts-v2.md), and the [video coverage and validation protocol](video-coverage.md). Initial P0 audit and partial P1 infrastructure are implemented below; real reconstruction-quality and submission gates remain open.

## Foundation delivery status

**2026-10-15 internal delivery increment:** the six v1 stages now have fresh-attempt execution, typed output
checks, fixed prerequisite copies, immutable reports, dependency invalidation and a read-only recheck CLI.
The CPU demonstration crosses independent local module folders and blocks a consumer after upstream tampering.
See [delivery-gates.md](delivery-gates.md) and [team-handoffs.md](team-handoffs.md). Native stages still need
typed adapters and transport; the P2 real-model and G0-G4 acceptance gates remain open.

| Work package | Recorded progress | Remaining acceptance work |
|---|---|---|
| P0.1 | [Component audit](component-audit.md): exact branch/upstream/patch identities and individual reuse decisions; provisional reproduction stack nominated | Restore and verify an actual runtime on the selected host |
| P0.2 | [Exposure ledger](exposure-ledger.json): all 30 episodes, 10 groups, 13 source-pinned historical records; no clean holdout assigned | External/manual history and actual video observation audit |
| P0.3 | [Initial validation policy](validation-policy.json): structural/review protocol, grouping, intervention log and failure taxonomy | Empirical quality thresholds and native conversion tolerances after baseline characterization |
| P1.1 | [Executable foundation](v2-foundation.md): source-domain separation, three record validators, structural/frame/asset/report bindings | Actual provenance verification, native decoder, complete target schema and reviewed real evidence |
| P1.2 | Content hashes, immutable UUID records, verified transitive declared dependencies and atomic compare-and-swap scene selection | Runtime producer integration, dependency scheduling/recomputation and crash recovery orchestration |
| P1.3 | Independent synthetic native-shape, frame, SO(3), scale, canonical-frame and pixel-transform checks | Real rig/decoder semantics and native-to-imported-to-exported geometry equivalence |

This is foundation delivery, not a completed P1 migration. The next integration work starts from the audited invocation/import components and requires a pinned restored runtime plus authoritative native artifacts; it must not promote synthetic fixtures or set validity flags without separately trusted reports. All G0-G4 gates below remain open.

[evaluation.md](evaluation.md) records external-kit identities, metric/format mechanics and diagnostic limitations. [contracts.md](contracts.md) describes implemented v1 artifacts; the v2 proposal does not change their meanings or register a working backend. [workflow.md](workflow.md) owns branch/review and reporting procedures. Only Track 1 inputs and independently created synthetic fixtures are permitted; Track 2 and all its derivatives remain excluded from every gate.

The single current authority is [track1-requirements.md](track1-requirements.md), R1–R6, effective 2026-10-03. G0-G4 must implement those requirements. The first reference frame has an explicit original-video ID; a later scored subset must retain its human-derived alignment. External-kit row selection is an adapter detail and cannot supersede the required frame or justify dropping input frames.

## Starting point and delivery boundary

### Independent-machine execution decision (2026-10-13)

Team members have different server access. A shared host is no longer a planning prerequisite: follow
[distributed-execution.md](distributed-execution.md) and its design-only JSON profile. Preserve one logical
native run and its scientific configuration while assigning native 01-04, 05 and 06-07 to capable workers.
M1 owns coordination, full-scene import and export validation; GPU conversion/rendering may execute on M2/M4.

Add three bounded tasks to P1/P2: (1) pack/verify/import with immutable manifests and synthetic failure checks;
(2) per-stage commands, relative-path materialization and verified predecessor consumption; (3) one full
episode across at least two real hosts, followed by native-to-exported geometry comparison and the second
engineering episode. Current local validators and historical Release tools do not implement these tasks.
Keep preparation and the research prototype running in parallel. If a boundary fails, temporarily co-locate
adjacent stages on an available worker; do not change the baseline algorithm to make transport easier.
These adapter tasks do not close G0/G1 by themselves or change the existing milestone dates.

The reviewed `main` at `0e86889` retains fake reconstruction stages, pass-through refinement, and the internal v1 exporter after the Windows development changes were reverted. `origin/windows-cari4d` contains candidate adapters, native conversion/export, runtime patches, previews and repair experiments. Those components and recorded runs are audit inputs, not an accepted current baseline. Historical preparation receipts establish the identity of downloaded files; they do not establish a runnable environment on the selected machine.

Before reusing any branch component, record its exact commit/path, upstream and model dependencies, intended behavior, local modifications, input provenance, coordinate/scale assumptions, frame handling, run evidence, and portability limits. Classify it as reusable after verification, requiring revision, or deferred; give it a task record and a focused acceptance check. Review components individually rather than treating the branch as a single approved implementation. In particular, distinguish a portability patch from a changed inference or optimization method.

Runtime version selection remains open. The historical external-code pin in `evaluation.md`, deployment manifests, and the branch's actually executed revisions must be reconciled before P2. Record the selected source commits and patches, model/rig identities, OS, Python, CUDA/PyTorch and native dependencies, plus the effective runtime configuration. A documentation source pin or a mutable container tag alone cannot close that task. Any different runtime needs a compatibility decision against R1–R6 and explicit conversion checks.

The first engineering delivery ends at P2: episodes 12 and 16 pass native conversion, full-frame structural checks, atomic scene import, and export/reload comparisons. P3–P4 establish whether evidence-driven decisions improve reconstruction. P5 establishes completeness and reproducibility of the submission candidate. A successful export or a visually plausible preview does not establish 3D accuracy.

## Milestones and gate mapping

P0–P5 specify execution order and work packages. G0–G4 below remain the authoritative evidence gates; milestone labels neither replace nor automatically close them. Gate review records must link the exact artifacts and unresolved findings. No gate is marked passed by this plan.

| Milestone | Scope | Gate evidence produced | Completion boundary |
|---|---|---|---|
| P0 | Inventory, Track 1 coverage, prior exposure history, validation protocol | G0 scope/provenance decisions; future G2/G4 review protocol | Audited implementation inventory and documented development/validation roles |
| P1 | Reviewed v2 contracts, compatibility, dependency invalidation, native mapping | G0 representation and structural checks | Executable contracts and independent fixtures; real conversion remains pending P2 |
| P2 | One atomic native baseline on episodes 12 and 16; early native export/reload | Remaining G0 real-conversion evidence and G1 | Complete, reproducible real baseline with known weaknesses recorded |
| P3 | Video evidence, diagnostics, multi-interaction `SolvePlan` dry runs | G2 | Reviewed diagnostic decisions under a frozen observation protocol |
| P4 | Bounded orientation, contact-switch and occlusion repairs | G3 | Each enabled repair has acceptance evidence and rollback; rejected repairs remain disabled |
| P5 | Frozen grouped validation, all 30 episodes, export/restoration rehearsal | G4 | One coherent full-set submission candidate and reproducibility record |

### Ordered backlog

Capability areas: **M1** Platform, **M2** Human, **M3** Object, **M4** Temporal/Physics. Member roles float across tasks; no permanent assignee is required. `A → B` means B cannot be accepted before A's recorded result; independent preparation may proceed earlier. The four modules retain their input/output boundaries while a single native task produces coupled human/object state. The tables name work and review scopes; execution and review participants are recorded per task.

| ID | Depends on | Capability; review scopes | Concrete artifact and acceptance |
|---|---|---|---|
| P0.1 | None | M1; M2/M3/M4 | Main/branch component inventory and runtime decision register. Every reuse candidate has a source identity, provenance assessment, unresolved dependencies and verification task; experiments are not labelled accepted baselines. |
| P0.2 | None | M1; M3/M4 | All-30 coverage matrix plus exposure ledger: geometry, view, interaction, occlusion, reviewed frames and prior tuning/selection history. Unreviewed video properties and unknown history remain explicit. |
| P0.3 | P0.1, P0.2 | M1/M4; M2/M3 | Frozen initial fitting/review policy, grouping, intervention policy, per-episode failure taxonomy, metric names and decision record template. Same-object clips are grouped; unseen status is not inferred from lack of current logs. |
| P1.1 | P0.1, P0.3 | M1; M2/M3/M4 | Versioned `EpisodeEvidence`, `SceneState`, `SolvePlan` and artifact-manifest contracts from `contracts-v2.md`, with an explicit v1 compatibility boundary. Wrong frame/camera/mesh bindings, fake artifacts and unknown provenance fail clearly. |
| P1.2 | P1.1 | M1; M2/M3/M4 | Immutable scene snapshots, content dependencies, atomic promotion and stale-input checks. A consumer loads one declared scene; it cannot assemble unrelated latest human, mesh and motion files. |
| P1.3 | P0.1, P1.1 | M2/M3; M1/M4 | Native decoder/converter and mesh/pose-frame specification with independent fixtures and declared numerical tolerances. Known transformations decode as intended; old v1 arrays cannot masquerade as native MHR. |
| P2.1 | P0.1, P1.1, P1.3 | M1/M2/M3; M4 | Pinned runnable adapter plus metric-mesh preparation on episodes 12 and 16. Bootstrap estimates and the native pipeline's actual consumed/recomputed camera, depth and human state have separate lineage and a reviewed consistency check. |
| P2.2 | P1.2, P2.1 | M1; M2/M3/M4 | One logical native run per episode, executed in verified stage blocks across available machines or one host; imported as a complete `SceneState` with native parameters, aligned mesh/poses, source frame map, exact consumed predecessors, logs and per-stage environment. Missing required artifacts prevent promotion; prior valid snapshots remain available. |
| P2.3 | P2.2, P1.3 | M1/M2/M3; M4 | Native conversion, official-format-compatible packing, reload and geometry comparison for both full clips, plus source overlays and failure inventory. All source frames/objects survive; native-to-imported-to-reloaded comparisons isolate adapter error. |
| P3.1 | P0.2, P0.3, P1.1 | M1; M2/M3/M4 | `EpisodeEvidence` generation: stable target IDs, visible observations, explicit unavailable states, anchor selection, event candidates and lineage. Thin structures, image transforms and uncertainty survive; a rendered prediction is not marked independent evidence. |
| P3.2 | P2.3, P3.1 | M1/M4; M2/M3 | Regenerable diagnostic reports over multiple interactions, separating structural failure, image disagreement and ambiguity. Every report identifies fitting versus reserved observations and correlated model dependencies. |
| P3.3 | P3.2 | M4; M1/M2/M3 | Reviewed `SolvePlan` dry runs for orientation, contact-switch and occlusion cases. Each names its evidence, fixed/free variables, hypotheses, budget, full-clip checks and rollback. No repair executes merely because a plan was generated. |
| P4.1 | P3.3 | M3/M4; M1/M2 | Orientation candidate selection using fixed camera, mesh and scale. Known symmetry, appearance cues, visible anchors and interval boundaries are checked; a smooth wrong orientation is rejected. |
| P4.2 | P3.3 | M4/M2/M3; M1 | Contact-state transitions for grasp/release, slide/push and support, with bounded state-owner-approved changes. Contact is rechecked after geometry/motion changes; low penetration alone cannot promote a candidate. |
| P4.3 | P3.3 | M4/M3; M1/M2 | Bidirectional occlusion recovery with a fixed hypothesis budget. Reappearance and window boundaries constrain candidates; complete coverage preserves explicit uncertainty and does not assert unique hidden motion. |
| P4.4 | Applicable P4.1–P4.3 | M1/M4; affected M2/M3 owners | Independent accept/reject report and atomic promotion/rollback per trial. Declared failure improves on reserved evidence, other checks respect predeclared limits, and full-clip structure remains valid. If none pass, retain the valid baseline. |
| P5.1 | P3.3; P4.4 for any enabled repair | M1; M2/M3/M4 | Freeze all effective rules, models, budgets, observation corrections and acceptance criteria. Execute grouped validation only after exposure-history review; record results used for selection as development evidence thereafter. |
| P5.2 | P5.1 | M1; M2/M3/M4 | Full 30-episode run, accepted scene IDs, per-object/interaction and worst-case reports, failures, interventions and compute use. Required frames/objects are complete; no unrecorded episode-specific policy choice. |
| P5.3 | P5.2, P2.3 | M1; M2/M3/M4 | One full-set export manifest, exact scene-bound packing/reload/render checks and clean restoration rehearsal. Unresolved submission-validity issues block G4; local export remains distinct from external submission. |

P1 contract work and P3 evidence-producer prototypes may overlap, but P3 diagnostic acceptance depends on a real P2 baseline. P1.3 native mapping and P1.2 scheduling can proceed in parallel after the shared contract is reviewed. M2/M3 prepare scale/native geometry while M1 resolves runtime/import and M4 defines event review and negative controls. P4.1–P4.3 can run independently against the same immutable baseline; composing their changes requires a fresh P4.4 full-clip review. Do not allow concurrent repairs to overwrite a shared scene.

### Acceptance layers and stop rules

| Layer | Required evidence | What it does not certify |
|---|---|---|
| Engineering MVP, P2 / G0–G1 | Permitted inputs, native semantics, coherent coordinates/scale, all-frame state, exact asset binding, reproducible invocation, export/reload equivalence | Correct unknown 3D geometry or generalization |
| Reconstruction quality, P3–P4 / G2–G3 | Independent/reserved image evidence, event and contact review, perturbation/negative controls, no unacceptable full-clip regressions | Hidden Track 1 official scores or unbiased generalization after repeated tuning |
| Submission candidate, P5 / G4 | Full expected set, frozen rules and artifacts, operational format requirements, restoration and read-back evidence | Submission receipt, leaderboard success or broader unseen-data performance |

Structural failures stop promotion immediately: missing source frames, incompatible canonical frames, unknown provenance, stale descendants, fake contributions, invalid native conversion, or mixed scene versions. Scale/bootstrap disagreement returns to preparation; it must not be hidden by an all-variable optimizer. Geometric failures reopen the relevant state owner rather than being silently repaired in the exporter.

Quality thresholds are chosen and documented after baseline characterization and before candidate ranking. Use reviewed qualitative decisions where evidence cannot justify a numerical threshold. Exhausted budget, ambiguous observations or no reserved-evidence improvement stop that repair and retain the prior accepted snapshot. A complete valid baseline remains an admissible fallback when all optional repairs fail; an invalid baseline never becomes a fallback by renaming its status. Record failed attempts as well as selected candidates.

The entire 30-video collection has already informed architecture discussion, and branch experiments may have additional exposure. Audit that history before assigning holdout roles. If no untouched object group remains, report grouped retrospective or prospective frozen-policy validation with its limits; do not call it an unbiased holdout. A validation group becomes regression/development data once it influences a policy change. Full-set coverage and input perturbations are useful stress tests, but cannot independently prove unseen-object generalization.

## Gate overview and review scopes

Gates close on recorded evidence, not on a date, downloaded checkpoint count, or a design-review score. A later gate cannot make an incomplete earlier gate disappear.

| Gate | Deliverable | Primary capability | Required review scopes | Current state |
|---|---|---|---|---|
| G0 | Evaluation/native-conversion/provenance contract | Module 1 Platform | Module 2 Human; Module 3 Object; Module 4 Physics for metric tests | Open: diagnostic/source review exists; runtime, real decoder/export and end-to-end provenance acceptance remain unresolved |
| G1 | Immutable real baseline on episodes 16 and 12 | Module 1 Platform | Modules 2, 3 and 4 | Open: branch run records require audit and reproduction through the accepted contracts |
| G2 | Reliable observations, camera/scale/asset lineage | Module 1 Platform + Module 3 Object | Module 2 Human; Module 4 Physics | Open: proposed design; diagnostic decisions require multi-interaction review |
| G3 | Bounded repairs with evidence and rollback | Module 4 Physics + relevant state owner | Module 1 integration; Modules 2/3 as affected | Open: branch experiments are candidates for audit, not accepted policies |
| G4 | Complete 30-episode, reproducible submission candidate | Module 1 Platform | All module owners | Open: G0-G2 and any enabled G3 policy need evidence; G3 may remain disabled |

Any available member with the required capability may execute or review work in M1-M4. Record the actual participants per result; permanent assignment is not a gate. Artifact contracts and required review evidence continue to apply when participants change. G4 may ship the accepted G1/G2 baseline if no G3 repair survives review; research novelty is not a completeness requirement.

## G0: establish the executable truth

### Required work

1. Implement the current R1–R6 specification and record reviewed external source identities. Resolve the runtime pin and branch-component audit in P0.1 separately from the historical evaluation-code pin; document any compatibility patches and changes to numerical behavior. Make the reference's first original-video frame ID explicit and verify it independently of scored-row selection. Historical September assumptions are superseded, not alternative configuration choices. Do not download an entire upstream data archive to obtain code or run examples with unknown provenance.
2. Specify and review a native-MHR migration. The proposed authoritative record is the official converter's native 204 parameters plus shape45 and decoder identity. Current decomposed v1 fields have no automatic mapping. Preserve v1 compatibility explicitly or supply a versioned converter; do not change field meanings in place.
3. Implement a source-pinned native decoder/export adapter. Record frame IDs, native-unit conversions, image/world conventions and rig assets. Validate the mapping to the official schema through decoded geometry, not array dimensions alone.
4. Bind object poses to mesh content, canonical origin/axes and exactly-once scale application. Verify the native CARI4D aligned-mesh transform and the exported mesh/pose pair together.
5. Implement Track 1 provenance checks for original videos, frame-derived observations, mesh candidates, manual corrections, model inputs, upstream runs and exports. Reject unknown sources and fake artifacts from a real/submission run.
6. Replace metadata-only resume assumptions with content-hash dependencies. A changed mask, camera, mesh, scale, model, converter or parameter invalidates downstream artifacts. Preserve the previous valid run.
7. Validate the reference-based native-MHR diagnostic with independent synthetic cases and an explicitly injected authorized decoder. Require complete episode/object/frame manifests and explicit permitted reference provenance. Do not restore historical Track 2 defaults, statistics, normalization or alignment modes. Use the explicit test list in `.github/workflows/tests.yml`; never run bare `pytest` in this repository.

### Acceptance evidence

- Known-answer fixtures cover native neutral/non-neutral poses, translation, global rotation, anatomical scale, hands, image transforms and camera/world conversion.
- Both conversion paths decode corresponding geometry within predeclared numerical tolerances; no direct concatenation of the old 133/108/28 arrays is used.
- Object tests include baked versus explicit scale, a non-origin-centred mesh, rotation plus translation, and the mesh/pose pair after canonical-frame conversion. Check compile/read-back behavior after official mesh welding, simplification and padding, particularly holes and contact surfaces. Geometry preservation alone does not certify an origin-based trajectory metric.
- Independently generated trajectories establish the new evaluator implications: unchanged accelerating predictions have zero reference-relative acceleration error; a moving sequence frozen in place need not. The exact official definitions remain in evaluation.md.
- First-reference-frame alignment uses its explicit original-video ID even when the scored subset begins later. A synthetic common-transform case applies one human-derived Sim(3) to human and object; a later-drift case must remain erroneous under that fixed alignment rather than being repaired by whole-clip, per-frame or independent object fits.
- Stale-input, wrong-rig, wrong-mesh, missing-frame, fake-output and unknown-provenance cases fail clearly.
- A reviewer can trace every fixture and asset to Track 1 or independent synthetic construction.

Reject G0 closure if conversion is justified only by shape checks, old normalization values are reused, source selection is ambiguous, or a local proxy is labelled an official reference-based score. A working fake export is not a native-conversion acceptance test.

## G1: produce one real baseline before comparing models

### Fixed scope

Start with Track 1 episode 16 (foam block) and episode 12 (pan), both explicitly treated as development cases. Run the toolkit MHR CARI4D runtime selected in P0.1/P2.1, without unrecorded changes to its learned components or optimization behavior. Reusing an earlier branch result requires the same provenance, conversion, full-coverage and reproducibility checks as a new result. Use episode-local meshes until cross-episode estimation is explicitly permitted. Keep a single baseline segmentation route and a small recorded mesh-candidate set.

### Required work and capability areas

| Owner | Work |
|---|---|
| Platform | Verify video decoding/frame map; prepare masks; invoke pinned native pipeline; import its coupled human/object results atomically |
| Human | Verify decoded MHR, identity stability, root/frame conversion and the required first reference frame |
| Object | Reconstruct and establish metric mesh scale; inspect topology, canonical frame and pose compatibility |
| Physics | Inspect motion, contact and failures without modifying the baseline; establish the initial diagnostic report |

Every stage used for acceptance must consume real Track 1-derived inputs. The native pipeline estimates its own camera/depth/human initialization; record those outputs as the baseline truth about what ran. Do not claim that a separate project camera file controlled the native run.

Resolve the metric-mesh bootstrap explicitly. Before the full CARI4D call, a preparation adapter obtains real depth and standalone human estimates with documented intrinsics, aligns visible human depth, and fits the candidate object's scale on visible evidence. Preserve that pass and compare it with the native pipeline's subsequently recomputed human/depth estimates. A material disagreement reopens scale preparation; it is not resolved by assuming the monolithic call consumed external initialization. No placeholder mesh or fake human may contribute to an accepted baseline.

### Deliverables and acceptance

- Immutable baseline ID with video/model/code/config identities, invocation, full logs, stage outputs, frame mapping, resource record and mesh provenance.
- Native human parameters and object poses covering every decoded source frame, including occlusions, with no fake contribution or silent trimming. Compare decoded video counts and original frame IDs with metadata and output coverage; a finite array of the metadata length alone is insufficient. Review discontinuities at occlusion entry/exit as well as missing values.
- Baseline export accepted by the G0 converter checks and the permitted official-format validation path. Do this at P2, before repair implementation: reload and compare human geometry, posed object geometry, canonical frame, scale application and original source-frame identities for the exact exported scene. Format acceptance and 3D-quality judgments remain separate.
- Source-image overlays and a three-stage comparison where available; clear marking of occlusion, low-confidence intervals, the explicit initial-frame mapping and known failures.
- A reproducible command/configuration on an identified execution environment. Bitwise equality across hardware is not required; declared comparison tolerances and provenance are.
- A review covering Platform, Human and Object concerns, with the actual reviewers recorded per result; the task's Physics work records the failure inventory.

Reject if output existence substitutes for coverage, supplied reference geometry is used, a generated mesh is treated as metric without a scale check, or the original mesh is paired with poses for a recentered mesh. This gate can close with visible reconstruction weaknesses if they are documented; it cannot close with missing or invalid artifacts.

**Comparison at this gate:** native output versus imported/exported output, using identical reconstruction. It must isolate adapter correctness. Do not start a broad SAM3D/GVHMR/WHAM or temporal-model tournament.

## G2: make observations and asset consistency inspectable

### Work packages

| ID | Owner | Change | Acceptance / rejection | Comparison |
|---|---|---|---|---|
| G2.1 | Platform | Observation records: visible masks, invalid/occluded/out-of-frame states, image transforms, depth validity, correction history | Thin contours/holes and crop mapping survive; reject unknown visibility encoded as certain empty space | Original versus corrected observations on fixed held-out frames |
| G2.2 | Platform + Human | Camera/depth/initialization override adapter, only if replacing native inputs is justified | Adapter reproduces the unchanged baseline path before overrides; source identity proves override was consumed | Native inputs versus one controlled override, fixed models and mesh |
| G2.3 | Human + Object | Fixed-camera, stable-human scale initialization and bounded object-scale fitting | Relative hand/object depth is consistent on visible evidence; report scale sensitivity; reject unrestricted per-frame scale | Single-frame estimate versus robust multi-frame estimate |
| G2.4 | Object | Candidate geometry, canonical origin/scale and typed symmetry | Correct asset binding, inspectable topology and visibility-aware selection; reject PEN-only mesh selection | Same observations/budget, candidate changes only |
| G2.5 | Platform | Automatic diagnostic extraction and per-episode reports | Report can be regenerated from frozen artifacts and exposes failures; reject hand-entered unverified values as measured evidence | Recompute stored baseline records independently |

Preserve the native baseline. Add explicit camera/depth/initialization hooks rather than editing hidden intermediate files and assuming the run used them. Do not introduce an all-variable solver.

### Review protocol

The historical `dev-mini = [0, 6, 9, 16, 24]` is a development/regression set, not an untouched holdout. Use the reviewed coverage matrix and exposure ledger in [video-coverage.md](video-coverage.md) to select additional clips covering sitting/carrying, foot push, contact switching and occlusion; do not infer verified frame-level events solely from metadata labels. P3 acceptance must span multiple interaction types rather than only episodes 12 and 16. Freeze fitting frames, diagnostic frames, synthetic cases, candidate budget, display conventions and review criteria before comparing candidates. A review must include all episodes in its declared set, including unsuccessful ones. Group all clips of an object together for broader development/validation comparisons; same-object leave-one-clip-out checks are within-development stress tests. A group that has influenced model or policy selection is subsequently a regression group. Record architecture-level exposure and uncertain history rather than presenting the all-30 set as an unbiased holdout.

Required evidence includes visible-contour agreement, independent image landmarks or reviewed keyframes, reliable-depth residuals, forward/backward discrepancies, asset scale/origin checks, contact plausibility and motion continuity. Record which observations are predictions and which were manually checked. Neither constitutes hidden reference geometry.

A reserved observation is excluded from the candidate-fitting and selection steps named in the experiment protocol; it does not imply that its source image was unseen by a full-video baseline or pretrained model. Record that distinction and shared upstream dependencies. Reusing review outcomes to choose a new repair transfers those outcomes into development evidence; it does not create repeated independent confirmation.

Choose degradation limits and numerical tolerances after baseline characterization and before ranking. No universal automatic quality threshold is currently implemented. If evidence does not support a numerical threshold, use an explicit reviewed decision rather than inventing one.

Cross-episode camera or asset sharing remains disabled. Enabling it requires documented permission, confirmed object/camera identities, a separate experiment flag and a record of which clips influenced each output. Three asynchronous clips never become synchronized multiview measurements by sharing an object label.

## G3: repair a diagnosed failure with bounded freedom

A repair proposal is a versioned `SolvePlan`: it states the failed interval, independent supporting observations, permitted parameter block, fixed variables, trust-region/displacement limits, hypothesis budget, fitting/review split, expected checks and rollback artifact. In P3 it is reviewed without executing a mutation; in P4 a registered repair operator may execute it against a specified immutable `SceneState`. Check the full clip after repairing a window. The validator decides promotion under the frozen rules; the repair operator's own fitting loss is insufficient.

### Order and admissible changes

| Order | Owner | Initial operation | Keep fixed initially | Essential rejection case |
|---|---|---|---|---|
| Precondition | Object + Platform | Route mesh/pose-frame, scale or camera inconsistency back to G0–G2 preparation; invalidate descendants | All unrelated representations | Improved appearance obtained by inconsistent units, mesh origin or hidden camera changes |
| P4.1 | Object + Physics | Orientation branches and associated bounded object pose fitting from clear anchors | Mesh, scale, camera, human | Wrong orientation becomes smooth; silhouette improves by moving behind the person |
| P4.2 | Physics + Human + Object | Contact-state switching and bounded local human/object changes supported by independent image anchors | Shape, anatomical scale, camera, object asset and unrelated pose blocks | Contact prior pulls a visible body part away from image evidence; PEN falls because the hand stops touching |
| P4.3 | Physics + Object | Bidirectional occlusion hypotheses and symmetry-aware selection | Mesh, scale, camera, identity and declared candidate budget | Long occlusion is presented as uniquely observed; boundary jumps or lost real acceleration |

Within contact-switch repair, release native human root/body or hand blocks only after a fixed-human candidate exposes the corresponding failure and independent observations can constrain the release. Record this as a separate intervention and ablation. Broad human refinement, camera refitting or new geometry generation are not additional automatic first-release repair operators.

Do not optimize every camera, scale, shape and pose parameter together. The native postopt's fixed parameters and hand-contact gate are baseline behavior; changing them is a separate intervention requiring its own ablation.

Treat grip, supported sliding, foot push, body support, release and no contact separately. Use soft constraints when the state or support plane is uncertain. A low estimated speed is insufficient evidence for a hard static lock. Pure smoothing toward zero acceleration is a negative control, not the official optimization objective.

### Repair acceptance

A candidate passes provenance/native-format/coverage checks; improves its declared failure on reserved observations or permitted official results; respects the predeclared degradation limits elsewhere; preserves first-reference-frame fidelity and window boundaries; and retains a reproducible rollback. Review per-episode and worst-case outcomes. An aggregate loss reduction cannot override a demonstrated geometric failure.

A conditional repair policy must use information available from the input/prediction. Do not cherry-pick variants per episode using hidden test answers or unrecorded manual score-driven decisions. Official results used for selection must be logged as such.

### Research ablations

These are hypotheses to test after a safe baseline, not claims of novel or superior methods.

| Hypothesis | Exact progression | Required failure/negative controls | Evidence needed |
|---|---|---|---|
| Observability-gated shared assets, only after permission | Single-frame asset -> multi-frame candidate/scale selection -> shared optimization -> shared optimization with gated variable release | Initial scale/focal perturbations; unconstrained all-variable fitting; leave-one-clip-out asset estimation | Stability and reserved-view consistency improve without sacrificing official results where available; fitting-only gain is insufficient |
| Event-conditioned symmetry-aware occlusion repair | Native -> bidirectional tracking -> matched-budget hypotheses -> symmetry distance -> event-conditioned branch selection | Occlusion duration, real spin, abrupt acceleration, release/regrasp, wrong static/contact hypothesis | Fewer flips and faster reappearance recovery, retained geometric and reference-relative motion fidelity |
| Evidence-constrained hand contact | Frozen hands -> PEN-only free hands -> contact distance -> persistent contact region -> observation-gated parameter release | Hands moved away, incorrect contact activation, hidden hand, cavity/thin-object collision failure | Lower penetration with preserved independently reviewed contact and image fidelity, same mesh and sampling |

Factor graphs, contact losses, shared shape and multiple hypotheses are established ideas. Claim a method contribution only when the proposed additional mechanism survives those ablations. Use matched input/compute budgets and independent synthetic trajectories with known answers; do not train and test on the same synthetic generator settings without disclosing it.

## G4: complete, freeze and rehearse

### Required work

1. Freeze the policy and validation grouping in P5.1 before inspecting new validation results, using the exposure audit from P0.2. Produce complete outputs for all 30 Track 1 episodes. Accepted baseline output is the fallback wherever a repair is disabled or rejected. A failed or structurally invalid baseline is a blocking issue, not an acceptable fallback.
2. Freeze the final mesh, its origin/scale, rig, converter, evaluator source identity, models and configuration. Recompute all changed descendants from content-verified inputs.
3. Validate native MHR/mesh/pose/frame contracts and provenance for every episode. Reproduce permitted prediction-geometry calculations through the pinned path and declare any assumed alignment. Official PEN also needs reference-derived alignment scale; do not label an unaligned local calculation a final official result.
4. Produce a per-episode report, source overlays, known-failure inventory, baseline/candidate decision record and the final export manifest. Include per-object/per-interaction and visible/occluded outcomes, worst cases, failure rates, manual interventions and resource use. State which conclusions are retrospective and which use a frozen policy; all-30 completion does not create unseen-data evidence.
5. Rehearse restoration and the permitted validation/export commands from the frozen package in a clean, identified environment. Exercise incomplete-file and stale-input failures.
6. Verify the current official operational requirements separately: kit authority/version, canonical-origin convention, permitted cross-episode use, competition format and submission limits. Do not infer them from an old planning calendar.
7. Prepare a concrete reviewable submission candidate. The designated submission owner handles any external upload under the team's authorization procedure; a local export is not a submission.

### Acceptance / rejection

Close G4 only when every required episode passes structural/provenance checks, reviewers have seen failures as well as averages, the converter reproduces the accepted geometry, unresolved issues affecting submission validity are resolved, and the package can be restored and checked. Publishable novelty and a particular leaderboard score are not prerequisites.

Reject a candidate if it silently falls back to fake data, omits occluded frames, mixes incompatible mesh/pose versions, changes canonical origin across variants, relies on Track 2, or claims complete official scoring from proxies. An unresolved optional research experiment stays disabled rather than delaying a valid baseline package.

## Evidence required for every experiment

The future run manifest/report should record:

- Run, baseline, parent-artifact and content identities; permitted source episodes and corrections.
- Code/model/rig/converter/evaluator identities and exact effective configuration.
- Native frame mapping, image transforms, mesh canonical origin and scale convention.
- Fitting versus review observations, their producers, reliability and common dependencies.
- Changed variable blocks, fixed variables, interval boundaries, limits and hypothesis budget.
- Per-episode diagnostics, failure categories, overlays, resource use and comparison results.
- Reviewer decision: accept, reject or unresolved; explicit trade-offs and rollback ID.

This manifest extension and automatic report extraction are planned work. Existing `run.json`, recipes, glob checks and scalar confidence fields do not yet provide this evidence.

## Questions that control scope

| Question | Owner | Effect until resolved |
|---|---|---|
| Which pinned external-kit adapter reproduces R1–R6 and records the first reference frame's original-video ID? | Platform | Requirements remain fixed; validate row/index mapping and compatibility without changing the alignment frame |
| Which runtime commit, patches, model/rig identities and native environment are selected after comparing historical manifests with branch execution records? | Platform with Human/Object | P2 runtime acceptance remains open; historical source receipts and review pins are not enough |
| Which branch components meet provenance, coordinate/frame semantics, full-coverage and portability checks? | Relevant module owner; Platform integration | Reuse only audited components; branch-wide availability does not establish acceptance |
| Which object groups have already influenced architecture, tuning or candidate selection? | Platform + Physics, all owners contribute history | Treat unknown exposure explicitly; do not claim an untouched holdout without evidence |
| Exact native mapping from the toolkit output to the official 204/45 record, including units and rig assumptions | Human + Platform | G0/G1 export acceptance remains open |
| Official object canonical origin and pose-scale conventions | Object + Platform | Freeze origin; no recentering as a tuning variable; consult evaluation.md |
| Is cross-episode shared geometry, scale, camera or identity estimation allowed? | Platform | Feature disabled; baseline remains episode-local |
| How are metrics/episodes combined and which operational submission limits currently apply? | Platform | Report separate metrics; no invented overall score or expired calendar |
| Which camera settings/image transforms are actually shared across recordings? | Platform | No hard cross-clip parameter tying |
| Which observation reliability/degradation limits are defensible on the accepted baseline? | All owners | Review explicitly; no fabricated automatic promotion threshold |
| Does a proposed new model address a measured bottleneck after G1/G2? | Relevant state owner | Keep the single baseline model path |

## Execution priority

The critical path is native conversion and verified evaluation adapters -> real complete baseline -> trustworthy observations and asset lineage -> targeted repairs -> full-set freeze. A reference-based diagnostic does not close native export or official-equivalence gaps. Source/runtime packaging may support that work but is not a substitute for it. Stop adding models when the next unresolved problem is an adapter, a coordinate convention, a bad mask, or a missing acceptance report.
