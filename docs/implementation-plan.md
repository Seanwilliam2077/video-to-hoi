# Implementation plan: Track 1 baseline and bounded repair

Updated 2026-10-03. This backlog implements [design.md](design.md). It does not report completed model runs, measured gains, a working native exporter, or an approved submission.

[evaluation.md](evaluation.md) owns official-kit identities, exact metric/format semantics and their unresolved conflicts. [contracts.md](contracts.md) distinguishes implemented v1 artifacts from the proposed next contract. [workflow.md](workflow.md) owns branch/review and reporting procedures. Only Track 1 inputs and independently created synthetic fixtures are permitted; Track 2 and all its derivatives remain excluded from every gate.

The owner's final Track 1 requirements supplied in chat on 2026-10-03, recorded in [design.md](design.md#latest-project-authority-final-requirements-supplied-on-2026-10-03), are the latest project authority. G0-G4 must preserve one human-derived first-frame alignment applied to the complete human/object sequence, native MHR, posed-world object geometry, reference-relative acceleration, all-frame object coverage and Track 1-only estimation. The kit's "first scored frame" wording remains an explicit mapping question when it differs from "first frame"; do not resolve that ambiguity by dropping input frames.

## Gate overview and ownership

Gates close on recorded evidence, not on a date, downloaded checkpoint count, or a design-review score. A later gate cannot make an incomplete earlier gate disappear.

| Gate | Deliverable | Primary owner | Required reviewers | Current state |
|---|---|---|---|---|
| G0 | Evaluation/native-conversion/provenance contract | Module 1 Platform, Sean | Module 2 Human; Module 3 Object; Module 4 Physics for metric tests | Source review documented; executable gates pending |
| G1 | Immutable real baseline on episodes 16 and 12 | Module 1 Platform | Modules 2, 3 and 4 | Pending real inputs, adapters and inference |
| G2 | Reliable observations, camera/scale/asset lineage | Module 1 Platform + Module 3 Object | Module 2 Human; Module 4 Physics | Planned |
| G3 | Bounded repairs with evidence and rollback | Module 4 Physics + relevant state owner | Module 1 integration; Modules 2/3 as affected | Planned |
| G4 | Complete 30-episode, reproducible submission candidate | Module 1 Platform | All module owners | Pending G0-G3 evidence; G3 may remain disabled |

Module 2, 3 and 4 individual assignees remain TBD. A module is accountable for its artifacts even when another module runs a coupled upstream method. G4 may ship the accepted G1/G2 baseline if no G3 repair survives review; research novelty is not a completeness requirement.

## G0: establish the executable truth

### Required work

1. Preserve the owner's final 2026-10-03 requirements, reviewed official source identities, and differences from the September email and legacy scorer. Map the kit's selected first frame to the clarification without assuming it always means source frame zero. Do not download an entire upstream data archive to obtain code or run examples with unknown provenance.
2. Specify and review a native-MHR migration. The proposed authoritative record is the official converter's native 204 parameters plus shape45 and decoder identity. Current decomposed v1 fields have no automatic mapping. Preserve v1 compatibility explicitly or supply a versioned converter; do not change field meanings in place.
3. Implement a source-pinned native decoder/export adapter. Record frame IDs, native-unit conversions, image/world conventions and rig assets. Validate the mapping to the official schema through decoded geometry, not array dimensions alone.
4. Bind object poses to mesh content, canonical origin/axes and exactly-once scale application. Verify the native CARI4D aligned-mesh transform and the exported mesh/pose pair together.
5. Implement Track 1 provenance checks for original videos, frame-derived observations, mesh candidates, manual corrections, model inputs, upstream runs and exports. Reject unknown sources and fake artifacts from a real/submission run.
6. Replace metadata-only resume assumptions with content-hash dependencies. A changed mask, camera, mesh, scale, model, converter or parameter invalidates downstream artifacts. Preserve the previous valid run.
7. Isolate permitted synthetic numerical checks from prohibited legacy scorer paths. Use the explicit test list in `.github/workflows/tests.yml`; never run bare `pytest` in this repository.

### Acceptance evidence

- Known-answer fixtures cover native neutral/non-neutral poses, translation, global rotation, anatomical scale, hands, image transforms and camera/world conversion.
- Both conversion paths decode corresponding geometry within predeclared numerical tolerances; no direct concatenation of the old 133/108/28 arrays is used.
- Object tests include baked versus explicit scale, a non-origin-centred mesh, rotation plus translation, and the mesh/pose pair after canonical-frame conversion. Check compile/read-back behavior after official mesh welding, simplification and padding, particularly holes and contact surfaces. Geometry preservation alone does not certify an origin-based trajectory metric.
- Independently generated trajectories establish the new evaluator implications: unchanged accelerating predictions have zero reference-relative acceleration error; a moving sequence frozen in place need not. The exact official definitions remain in evaluation.md.
- Initial scored-frame selection and its mapping to the clarified first-frame rule are tested without altering source frame numbering. A synthetic common-transform case applies one human-derived Sim(3) to human and object; a later-drift case must remain erroneous under that fixed alignment rather than being repaired by per-frame or independent object fits.
- Stale-input, wrong-rig, wrong-mesh, missing-frame, fake-output and unknown-provenance cases fail clearly.
- A reviewer can trace every fixture and asset to Track 1 or independent synthetic construction.

Reject G0 closure if conversion is justified only by shape checks, old normalization values are reused, source selection is ambiguous, or a local proxy is labelled an official reference-based score. A working fake export is not a native-conversion acceptance test.

## G1: produce one real baseline before comparing models

### Fixed scope

Start with Track 1 episode 16 (foam block) and episode 12 (pan). Run the pinned toolkit MHR CARI4D without changing its learned components or optimization behavior. Use episode-local meshes until cross-episode estimation is explicitly permitted. Keep a single baseline segmentation route and a small recorded mesh-candidate set.

### Required work and owners

| Owner | Work |
|---|---|
| Platform | Verify video decoding/frame map; prepare masks; invoke pinned native pipeline; import its coupled human/object results atomically |
| Human | Verify decoded MHR, identity stability, root/frame conversion and the initial scored region |
| Object | Reconstruct and establish metric mesh scale; inspect topology, canonical frame and pose compatibility |
| Physics | Inspect motion, contact and failures without modifying the baseline; establish the initial diagnostic report |

Every stage used for acceptance must consume real Track 1-derived inputs. The native pipeline estimates its own camera/depth/human initialization; record those outputs as the baseline truth about what ran. Do not claim that a separate project camera file controlled the native run.

Resolve the metric-mesh bootstrap explicitly. Before the full CARI4D call, a preparation adapter obtains real depth and standalone human estimates with documented intrinsics, aligns visible human depth, and fits the candidate object's scale on visible evidence. Preserve that pass and compare it with the native pipeline's subsequently recomputed human/depth estimates. A material disagreement reopens scale preparation; it is not resolved by assuming the monolithic call consumed external initialization. No placeholder mesh or fake human may contribute to an accepted baseline.

### Deliverables and acceptance

- Immutable baseline ID with video/model/code/config identities, invocation, full logs, stage outputs, frame mapping, resource record and mesh provenance.
- Native human parameters and object poses covering every decoded source frame, including occlusions, with no fake contribution or silent trimming. Compare decoded video counts and original frame IDs with metadata and output coverage; a finite array of the metadata length alone is insufficient. Review discontinuities at occlusion entry/exit as well as missing values.
- Baseline export accepted by the G0 converter checks and the permitted official-format validation path.
- Source-image overlays and a three-stage comparison where available; clear marking of occlusion, low-confidence intervals, first scored region and known failures.
- A reproducible command/configuration on an identified execution environment. Bitwise equality across hardware is not required; declared comparison tolerances and provenance are.
- A review signed by Platform, Human and Object owners; Physics records the failure inventory.

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

Expand to `dev-mini = [0, 6, 9, 16, 24]`. Freeze fitting frames, diagnostic frames, synthetic cases, candidate budget, display conventions and review criteria before comparing candidates. A review must include all episodes in its declared set, including unsuccessful ones. Group all clips of an object together for broader development/holdout comparisons; same-object leave-one-clip-out checks are within-development stress tests. A group that has influenced model selection is subsequently a regression group.

Required evidence includes visible-contour agreement, independent image landmarks or reviewed keyframes, reliable-depth residuals, forward/backward discrepancies, asset scale/origin checks, contact plausibility and motion continuity. Record which observations are predictions and which were manually checked. Neither constitutes hidden reference geometry.

Choose degradation limits and numerical tolerances after baseline characterization and before ranking. No universal automatic quality threshold is currently implemented. If evidence does not support a numerical threshold, use an explicit reviewed decision rather than inventing one.

Cross-episode camera or asset sharing remains disabled. Enabling it requires documented permission, confirmed object/camera identities, a separate experiment flag and a record of which clips influenced each output. Three asynchronous clips never become synchronized multiview measurements by sharing an object label.

## G3: repair a diagnosed failure with bounded freedom

A repair proposal states the failed interval, independent supporting observations, permitted parameter block, fixed variables, trust-region/displacement limits, hypothesis budget, fitting/review split and rollback artifact. Check the full clip after repairing a window.

### Order and admissible changes

| Order | Owner | Allowed first experiment | Keep fixed initially | Essential rejection case |
|---|---|---|---|---|
| 1 | Object + Platform | Mesh/pose-frame, scale or camera consistency correction | All unrelated representations | Improved appearance obtained by inconsistent units, mesh origin or hidden camera changes |
| 2 | Object + Physics | Object translation/rotation repair from clear anchors and visible features | Mesh, scale, camera, human | Wrong orientation becomes smooth; silhouette improves by moving behind the person |
| 3 | Human + Physics | Human root/body repair using independent image anchors | Shape, anatomical scale, camera, object asset | Pose prior or contact alone pulls the person away from image evidence |
| 4 | Human + Physics | Selectively release native hand controls | Mesh and unrelated body blocks | PEN falls because the hand stops touching the object |
| 5 | Physics + Object | Event states, bidirectional hypotheses and symmetry-aware selection | Candidate budget and representation contracts | Long occlusion is presented as uniquely observed; boundary jumps or lost real acceleration |

Do not optimize every camera, scale, shape and pose parameter together. The native postopt's fixed parameters and hand-contact gate are baseline behavior; changing them is a separate intervention requiring its own ablation.

Treat grip, supported sliding, foot push, body support, release and no contact separately. Use soft constraints when the state or support plane is uncertain. A low estimated speed is insufficient evidence for a hard static lock. Pure smoothing toward zero acceleration is a negative control, not the official optimization objective.

### Repair acceptance

A candidate passes provenance/native-format/coverage checks; improves its declared failure on reserved observations or permitted official results; respects the predeclared degradation limits elsewhere; preserves the initial scored region and window boundaries; and retains a reproducible rollback. Review per-episode and worst-case outcomes. An aggregate loss reduction cannot override a demonstrated geometric failure.

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

1. Produce complete outputs for all 30 Track 1 episodes. Accepted baseline output is the fallback wherever a repair is disabled or rejected.
2. Freeze the final mesh, its origin/scale, rig, converter, evaluator source identity, models and configuration. Recompute all changed descendants from content-verified inputs.
3. Validate native MHR/mesh/pose/frame contracts and provenance for every episode. Reproduce permitted prediction-geometry calculations through the pinned path and declare any assumed alignment. Official PEN also needs reference-derived alignment scale; do not label an unaligned local calculation a final official result.
4. Produce a per-episode report, source overlays, known-failure inventory, baseline/candidate decision record and the final export manifest.
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
| Which official kit version and first-frame selection implement the owner's final 2026-10-03 requirements? | Platform | Follow the final requirements; reproduce pinned implementation behavior, preserve all original frames, and keep first-frame versus first-scored-frame mapping explicit |
| Exact native mapping from the toolkit output to the official 204/45 record, including units and rig assumptions | Human + Platform | G0/G1 export acceptance remains open |
| Official object canonical origin and pose-scale conventions | Object + Platform | Freeze origin; no recentering as a tuning variable; consult evaluation.md |
| Is cross-episode shared geometry, scale, camera or identity estimation allowed? | Platform | Feature disabled; baseline remains episode-local |
| How are metrics/episodes combined and which operational submission limits currently apply? | Platform | Report separate metrics; no invented overall score or expired calendar |
| Which camera settings/image transforms are actually shared across recordings? | Platform | No hard cross-clip parameter tying |
| Which observation reliability/degradation limits are defensible on the accepted baseline? | All owners | Review explicitly; no fabricated automatic promotion threshold |
| Does a proposed new model address a measured bottleneck after G1/G2? | Relevant state owner | Keep the single baseline model path |

## Execution priority

The critical path is native conversion and evaluator reconciliation -> real complete baseline -> trustworthy observations and asset lineage -> targeted repairs -> full-set freeze. Source/runtime packaging may support that work but is not a substitute for it. Stop adding models when the next unresolved problem is an adapter, a coordinate convention, a bad mask, or a missing acceptance report.
