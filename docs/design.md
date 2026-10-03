# Track 1 reconstruction design

Updated 2026-10-03. This is the implementation direction following the repository and official-kit review. It is a plan, not a claim that the real reconstruction pipeline has been implemented or validated.

Build one reproducible native-MHR CARI4D baseline, preserve it, and add bounded repairs only where independent image evidence supports them. More models, a smaller optimization loss, and a smoother animation are not evidence of better reconstruction.

Follow [track1-requirements.md](track1-requirements.md), the single authoritative statement of the owner's latest six requirements. Read [evaluation.md](evaluation.md) for source-pinned external mechanics and diagnostic limitations, [implementation-plan.md](implementation-plan.md) for the G0-G4 delivery gates, and [contracts.md](contracts.md) for implemented v1 files and proposed future contracts.

### Latest project authority: final requirements supplied on 2026-10-03

The requirements are R1–R6 in [the current specification](track1-requirements.md). Older interpretations are superseded. In particular, alignment uses the first reference frame's explicit original-video ID even when a diagnostic scoring subset starts later. External array indexing is an adapter concern; it does not create an alternative alignment rule.

## 1. Scope, evidence, and current status

The task is to reconstruct human motion with hands, object geometry, and object motion from the 30 Track 1 monocular videos and their metadata. The released videos are static-camera sequences, but every adapter must preserve their frame indices, dimensions, image transforms, and timestamps. A camera name is not sufficient evidence of identical intrinsics or extrinsics across videos.

**Track 1 only.** Never download, load, or use Track 2 content or any derivative, including Tier 1/Tier 2 meshes, trajectories, labels, camera parameters, statistics, cached outputs, scores, or tuned parameters. There is no validation, debugging, normalization, or demonstration exception. Independently created synthetic fixtures may test methods and interfaces; they are not submission assets. Exclude assets with unknown provenance. Reading approved evaluation source does not authorize fetching its example data or reference bundles.

| Area | Evidence in the current repository | Still required |
|---|---|---|
| Orchestration | Six stages, upstream runs, v1 files, fake-stage smoke tests | Content-addressed dependencies and real artifact provenance |
| Perception and reconstruction | Fake inputs/human/object/motion; prohibited legacy backends are blocked | Registered, verified real adapters |
| Refinement | Pass-through backend | Real, observation-constrained repairs |
| Export | Internal metadata/parquet/mesh plus decomposed MHR arrays | Native MHR conversion and official export validation |
| Candidate comparison | Prepared recipes and record/ranking validation | Verified extraction of real measurements and comparisons |
| Runtime assets | Source/model manifests and acquisition records | Successful inference and a reproducible accepted baseline |

The reconstruction findings originated from the audit of `6a53e84` and still describe the fake-stage pipeline. The replacement reference-based diagnostic scorer is a separate implementation from reconstruction/export; its current scope is recorded in evaluation.md. Downloaded weights and passing synthetic tests do not establish reconstruction quality. No numerical quality improvement is asserted by this design.

### Evaluation consequences

The current requirements replace the September design's prediction-only acceleration assumption. The inspected external kit uses body-only human acceleration; indiscriminate smoothing and a finger-acceleration optimization campaign are not justified by that metric. Native MHR parameterization must be converted explicitly rather than inferred from matching array sizes. See evaluation.md for formulas, units, explicit frame mapping, source identities and adapter limits.

Track 1 does not expose the reference needed for local official Chamfer and reference-relative acceleration scores. Prediction-geometry penetration can be diagnosed locally, but official PEN also uses reference-derived alignment scale; an unaligned calculation is not the final official score. Keep official metrics, prediction-geometry calculations, image-consistency diagnostics, and synthetic experiments separately named.

## 2. Baseline and upstream boundaries

Use the MHR implementation in NVIDIA's video_to_data toolkit at `33129dd0f2d2dcfd1164d43fd076542660756ed2`. Keep its code, model identities, preprocessing, and outputs intact for the initial baseline. The standalone NVlabs/CARI4D repository and paper are related methods, not interchangeable runtime specifications.

The pinned toolkit accepts RGB, human/object masks, and an already metric object mesh. SAM 3D Objects is external mesh preparation. Its transform/intrinsics sidecars are not automatically consumed; the supplied geometry retains its scale. The native launcher constructs its own MoGe2 depth/intrinsics, human initialization, depth alignment, object tracking, CoCoNet output, and post-optimization. It has no public override for the project's planned shared camera/depth/human artifacts. The internal stages must be adapted explicitly before claiming that those artifacts govern inference.

The default final optimizer is limited: it updates object translation and MHR body rotation controls; object rotation is opt-in. Human root, hand, shape, and scale parameters remain fixed. Its symmetry flag changes temporal treatment; it does not implement a general symmetry group. Its contact eligibility is gated at initialization. Floor/table constraints and independent human image reprojection are additions to develop, not existing baseline capabilities.

Sources: [pinned toolkit README](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_cari4d/README.md), [native launcher](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_cari4d/lib/run_inference.py), [native optimizer](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py).

Do not transplant the standalone implementation's SMPL-era parameterization, mesh scaling procedure, optimizer settings, or paper metrics into this baseline. The [standalone custom-video instructions](https://github.com/NVlabs/CARI4D/blob/main/docs/custom_video.md) are useful research context; their limitations motivate tests, not unmeasured claims about the toolkit fork.

## 3. Architecture: observations, baseline, optional repair

The existing six-stage interface remains useful for ownership and storage. Do not force native CARI4D into six supposedly independent estimators: it already couples human and object inference. A baseline adapter should import the coupled outputs atomically and expose stage identities without rerunning hidden estimators.

```text
Track 1 video + metadata
  -> observations: masks, image transforms, camera/depth proposals
  -> object preparation: candidate geometry + explicit metric-scale estimate
  -> pinned native MHR CARI4D
  -> immutable baseline: decoded geometry, native parameters, object poses
  -> diagnostics and flagged intervals
  -> optional bounded repair, with the baseline retained
  -> verified native export -> official-kit-compatible artifact
```

An observation records what was measured or predicted from an image; an estimate records the scene explanation being optimized. Rendering an estimate back into a pseudo-observation does not create independent evidence. Record common upstream models: human initialization, depth, and contact predictions can have correlated errors.

The proposed future artifacts include:

- Native MHR trajectory and rig identity; derived human geometry and optional SOMA-X views.
- Object mesh identity, canonical origin/axes, baked scale and explicit pose-scale convention.
- Camera and image-transform lineage, including crop/resize/padding and distortion assumptions.
- Visible masks, invalid/occluded/out-of-frame regions, depth validity and observation provenance.
- Pose hypotheses, visibility state, diagnostic uncertainty, contact events and repair decisions.
- Content hashes for every dependency and an immutable run manifest.

These are contract proposals, not additional fields already available in `CONTRACT_VERSION = 1`. Introduce a reviewed migration before a consumer depends on them.

## 4. Representation and coordinates

### One authoritative human trajectory

Propose `NativeMHR204` as the authoritative future representation: the official converter's 204 native parameters, comprising a 136-parameter pose trajectory and 68 static scale parameters, plus 45 static shape coefficients and the exact rig/decoder identity. The current v1 decomposed arrays are not this representation. Do not concatenate, pad, truncate, or rename them to manufacture compatibility.

Native parameter units follow the pinned decoder; they are not uniformly metres or radians. In particular, native translation slots must not be populated directly from metre-valued camera translation without the documented conversion. Validate the mapping by decoding known poses, translations, rotations, scale changes, and a real Track 1 initialization through both paths.

Keep identity and anatomical scale stable within a clip through constrained fitting, not just overwriting each frame with a median while leaving inconsistent translations. Cross-clip identity sharing requires both supporting evidence and permission under the data rules.

MHR is the only editable human state. Derive SOMA-X or other diagnostic geometry from that state; do not smooth a second representation independently. Alternative SMPL-family models are candidate proposals only after a documented conversion demonstrates that global motion, scale, and hands survive.

### Camera world and object asset identity

Use a fixed OpenCV camera frame per static-camera clip for internal geometry. Human and object outputs must share that frame for the whole sequence. A scene plane can be represented in camera coordinates; gravity reasoning does not require changing the submission world. Do not invent shared world extrinsics across unrelated videos.

The official object transform, its scale handling, and the unresolved canonical-origin question are specified in evaluation.md. Bake a scale into geometry or apply it through the declared pose contract once, never twice. Native CARI4D can recenter/reorient an input mesh: pair its output poses with the corresponding aligned mesh, or perform a verified inverse conversion.

Bind every pose artifact to the exact mesh hash and canonical frame. A rigid recanonicalization `v_new = C v_old` requires `T_new = T_old C^-1` to preserve world geometry, but that does not necessarily preserve an origin-based trajectory metric. Freeze the origin across comparisons; do not recenter to improve a score. A scale change requires refitting and invalidating dependent results, not hiding scale in a purported SE(3) rotation.

Record the first reference frame's original-video ID explicitly. A later scoring window keeps that fixed human-derived alignment; the first row of a selected array is not an implicit replacement. Inspect initial-frame sensitivity without choosing frames based on unavailable reference results. Export the common scene coordinates; reference alignment belongs to evaluation, not to an exporter that has no reference. No independent object alignment or later per-frame alignment may conceal reconstruction drift.

## 5. Observation quality and scale

### Camera and depth

Start with the native baseline camera. Check background consistency, image dimensions, crop transforms, and static-camera assumptions before replacing it. Shared intrinsics are an optional prior after camera identity and image formation are verified; shared extrinsics must not be inferred from recording dates.

Metric human and depth networks provide scale priors, not physical measurements. Monocular projection is unchanged under a common spatial scale, so a static camera and learned depth do not make absolute scale uniquely observable. Evaluation alignment can remove some common ambiguity; it cannot repair relative human-object scale errors, shape errors, or temporal drift.

Initialize with one human identity/scale estimate and fixed camera, robustly align reliable visible human depth, then estimate object scale from visible object evidence. Inspect stability across well-observed frames and perturbations of the initialization. Do not independently anchor the object to raw depth and the human to a differently scaled depth stream.

This is an explicit mesh-preparation pass before the full native pipeline: obtain real depth and a real standalone human initialization under a documented camera convention, align them, and scale the candidate mesh. Record these preparation estimates separately from the human/depth estimates that the native pipeline recomputes. If they materially disagree, revisit scale before accepting the run. Do not assume that an unscaled mesh can become metric merely by passing it to CARI4D; the preparation adapter is part of G1 work.

During repair, release one coupled variable block at a time. Do not jointly free focal length, human size, object size, and depth bias without sufficient independent constraints. Per-frame scale freedom is especially dangerous. Report sensitivity and competing solutions; do not label an uncalibrated residual score a probability of correctness.

### Masks, depth boundaries, and independent checks

Use one baseline segmentation path, with reviewed prompts/corrections where permitted. Store both the original prediction and any edits. Distinguish absent object evidence from a reliable empty region. Retain holes and fine contours of rings, handles, and thin furniture.

Compare observed visible regions to jointly rendered visible human/object surfaces, accounting for occlusion and the image boundary. A fully projected object silhouette compared to a partially visible mask can reward an incorrectly shrunken mesh. Predicted depth near thin silhouettes, reflections, occlusion boundaries, or invalid pixels must not be treated as precise geometry.

Keep high-resolution image evidence for thin objects and hands. A stride-only depth schema cannot describe every resizing convention; explicit sample/image transforms are part of the future contract. Use held-out visible frames, independently reviewed image landmarks, and forward/backward discrepancies as cross-checks. Agreement between models with the same upstream estimate is weaker evidence than an independent observation.

## 6. Object geometry and difficult cases

Default to episode-local reconstruction until the organizer confirms whether cross-episode joint estimation is permitted. Reusing or jointly fitting one object's three clips is a disabled-by-default experiment, not an official mesh-sharing requirement. Even if permitted, asynchronous interaction clips are not synchronized multiview observations.

Begin with SAM 3D Objects candidates from a small, recorded set of clear, diverse frames. Add another generator only after a specific geometry failure remains. Rank candidates on visibility-aware held-out image evidence, scale stability, and topology inspection. A realistic texture or low penetration score is not sufficient. Never change mesh sampling, holes, scale, or origin merely to make a metric easier.

| Case | Initial modeling choice | Main failure to test |
|---|---|---|
| Hula hoop | Analytic ring candidate alongside generated geometry; fit visible contours and thickness | Lost hole, perspective/normal ambiguity, fast rotation, body passing through |
| Bowl/cylinder-like geometry | Explicit candidate symmetry after checking geometry and observed appearance | Arbitrary yaw jumps or incorrect hard symmetry |
| Pan, iron, brush and block | Preserve handles and other orientation cues | Occlusion mistaken for missing geometry; wrong grip-side pose |
| White desk/cart | Reliable contours, corners and support evidence | Weak texture, foot interaction, off-screen geometry |
| Stool | Complete geometry consistent with visible intervals | Long body occlusion and unsupported assumptions about seat contact |

For bowl cavities, open meshes and thin surfaces, verify the collision representation separately from a convenience convex proxy. Geometry that fills an actual opening can invent penetration. The exact submission mesh remains the authority for official-format calculations.

The official packer can weld, simplify and pad a mesh to its budget. Inspect and render the compiled mesh after read-back, including thin structures, holes and contact surfaces. A high-resolution candidate that is correct before packing can still be an invalid geometric choice after packing.

If cross-episode fitting is approved, optimize shared shape/scale with clip-specific motion and camera assumptions. Test leave-one-clip-out asset estimation: fit geometry from two clips, then fit only motion on the third. This tests transfer of image consistency, not hidden 3D accuracy or generalization to unseen objects.

## 7. Bounded, evidence-gated repair

The complete model can be described as a sum of visible-image, visible-depth, temporal-correspondence, contact, collision, motion and prior terms. This is a specification, not a requirement to implement one large unconstrained optimizer.

Use robust residuals with explicit units and observation reliability. Keep original observations separate from learned priors. Specify the allowed parameter block, interval, displacement bounds, hypothesis budget and rollback rule before each experiment. A lower objective does not justify promotion if independent observations deteriorate.

The initial repair order is:

1. Correct camera/scale or mesh/pose-frame inconsistencies before refining motion.
2. Repair flagged object pose intervals with fixed geometry and camera, using visible contours/features and clear anchor frames.
3. Release human root or body controls only with independent image anchors and a bounded change from the baseline.
4. Release hand parameters only when the visible hand evidence or stable interaction state can constrain them.
5. Recheck the full sequence, including boundaries of repaired windows, and export from the authoritative native MHR state.

Use reference-relative acceleration implications from evaluation.md. Prediction self-acceleration is a diagnostic, not the target official error. Do not turn rapid real motion into a smooth, wrong trajectory. A static segment must be established from image/support evidence and contact state, not merely low estimated speed; ambiguous intervals retain a soft prior.

### Symmetry and occlusion

Use symmetry-aware rotation distances over the verified equivalence group, rather than averaging distinct poses. Geometric symmetry and appearance symmetry can differ. Approximate symmetry is a soft constraint. Selecting a continuous representative does not make an unobserved spin physically observable.

Track from clear anchors forward and backward. A small fixed set of competing hypotheses may bridge an occlusion; record their differences and use reappearance evidence to choose. During a reliable grip, object motion relative to the hand is more informative than holding the last camera-space pose. Supported sliding, release and regrasp require different states.

Long complete occlusion can remain non-identifiable. Preserve that diagnostic uncertainty even though export requires one continuous trajectory. Filling every frame establishes coverage, not correctness. Do not introduce a learned motion model until these simpler repairs expose a specific remaining failure.

### Contact and collision

Separate hand grasp, foot push, body support, object support and no-contact states. Minimum hand-surface distance alone does not establish a realistic grasp. Where visible evidence supports it, maintain a contact region in object coordinates, allow explicit release/slip, and check tangential motion as well as penetration.

Reducing penetration by moving the hand away can destroy the interaction. Keep the mesh fixed for contact ablations and pair penetration with image fidelity and independently reviewed contact persistence. Uncertain floor/table planes are soft geometric evidence; they are not grounds to invent forces, friction or exact dynamics.

## 8. Validation and promotion

Begin with episodes 16 (foam block) and 12 (pan). Then use `dev-mini = [0, 6, 9, 16, 24]` for varied failures and `dev-full = all 30` for integration. The first-three-video kit covers only hula hoops and cannot select a universal model or establish whole-project readiness.

Every experiment preserves the baseline and records source/model/config hashes, frame mapping, upstream identities, mesh origin/scale, changed variable blocks, runtime, observations used for fitting, observations reserved for review, and before/after overlays. Metadata-only cache signatures are not content hashes; project dependency checks must verify content explicitly. For broader model-selection claims, keep all clips of an object in the same development or holdout group. Same-object leave-one-clip-out checks remain within-development consistency tests, not unseen-object validation. Once a group has influenced decisions, call it a regression group.

Use independent synthetic sequences for known-answer geometry, native conversion, alignment, occlusion, symmetry, collision topology, and acceleration-error tests. Include genuinely accelerating motion: an unchanged correct prediction should have zero reference-relative acceleration error even when its own acceleration is nonzero. Do not estimate test distributions or tolerances from Track 2.

Freeze a comparison protocol after baseline observation and before candidate ranking. Publish per-episode outcomes and the worst failures, not only averages. Frame-adjacent samples are correlated; do not present them as independent replication. No universal numerical pass threshold or automatic promotion is implemented today.

Promote only when:

- Provenance, native-format validity, frame coverage and asset identity remain valid.
- The targeted failure improves on held-out observation evidence or permitted official results.
- Required image/contact checks do not exceed predeclared degradation limits.
- Repaired-window boundaries and the initial scored region remain coherent.
- A reviewer records trade-offs and an immutable rollback baseline.

A Pareto comparison across accuracy, motion fidelity and contact is preferable to an invented scalar leaderboard formula. Full official scores require the approved reference/evaluation process. Keep publication, deployment and official submission as separate actions from local validation.

## 9. Research hypotheses, not established contributions

Potential research directions are observability-gated shared-asset optimization, event-conditioned symmetry-aware occlusion recovery, and evidence-constrained hand contact repair. Factor graphs, multiple hypotheses, shared shape, and contact losses are established ideas; using them together is not automatically a new method.

Each hypothesis requires an incremental ablation, matched input and compute budgets, natural failure cases, independent synthetic known-answer tests, and comparison against the preserved native baseline. Cross-episode experiments additionally require rule confirmation. Detailed experiment and rejection plans are in implementation-plan.md. If evidence demonstrates integration reliability rather than algorithmic novelty, report that result honestly.

## 10. Historical assumptions and open authority questions

The 2026-09-26 organizer email, preserved in [the earlier design revision](https://github.com/Seanwilliam2077/video-to-hoi/blob/6a53e84/docs/design.md#64-organizers-answers), said acceleration measured predicted trajectories alone. This is retained as dated history, not the active implementation specification. The owner's final requirements supplied on 2026-10-03 explicitly require reference-relative second differences, consistent with the inspected kit. That design choice is resolved; evaluation.md retains the source history and the implementation version to reproduce. Do not silently combine the email, the legacy scorer, the paper, and the current kit.

The old 77-joint/finger acceleration assumption, body-joint alignment approximation, reference-relative penetration approximation, and claim that the official adapter is simply unpublished are superseded by the current source review. No part of that update relaxes the Track 1-only policy.

Before submission, verify the external kit adapter against the current requirements, including explicit original-frame mapping, native conversion, canonical object origin and operational compatibility. Resolve permitted cross-episode estimation separately. These implementation questions do not reopen R1–R6. Use G0-G4 gates rather than the expired date-based September schedule.
