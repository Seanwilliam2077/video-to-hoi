# Track 1 video coverage for pipeline design

Updated 2026-10-13. This is a metadata inventory and design proposal. It records no new video inspection, model run, decoded frame verification, accepted reconstruction, or official score. [R1-R6](track1-requirements.md) remain authoritative and the [G0-G4 delivery gates](implementation-plan.md) remain in force.

## Source and evidence boundaries

The inventory copies all 30 `videos` entries from `const DATA` in [the local gallery](track1-videos/index.html), source repository revision `0e86889696edc4d81a3c287cb93d76baece4a1d5`. [video-coverage.json](video-coverage.json) preserves every source entry, including exact task, prompt, camera label, frame count, sequence and source links. The entries were compared with the pinned Git revision. No source video or preview was decoded, and no data was downloaded for this inventory.

- Full gallery Git-blob SHA-256: `bafb940da99620ca4c88534f62ef6c181af1f667fca49ad9c47e98fe9d154d56`.
- Full local gallery file SHA-256: `806ad92292d0545bca0fe68d7c3f6342e50cc903f44644337883a2217fde3ee9`.
- Hashes cover complete HTML bytes, including styling/scripts. Git and checkout bytes can differ due to line endings; the extracted `DATA` objects must match exactly.

The gallery metadata contains 30 episodes, ten objects with three episodes each, and 16,563 frames. View labels are back (5), front (9), left (11), and right (5). These are metadata counts; complete source decoding and original-frame/timestamp mapping remain required before an R5 completeness claim. The camera labels ending in `stereo_camera_left` do not provide a second input view or measured intrinsics.

The inventory separates three evidence classes:

| Class | Meaning | Permitted use |
|---|---|---|
| `source_metadata` | Exact gallery statements, including intended task instructions | Identify an episode and propose what to inspect |
| `inferred_design_hypotheses` | Plausible geometry, events, support relations and diagnostics inferred from those statements | Plan observations and candidate solver constraints |
| `unobserved` | No observation or measurement in this inventory; represented as `null` or explicit status | Keep unknown until video-derived evidence exists |

Task descriptions are intended actions, not observed event truth. Their order does not establish event times, continuous contact, no-slip grasp, object rigidity, or successful task completion. Geometry symmetry and weak-texture labels are hypotheses, not certified properties. All episodes have visual review status `not_reviewed_in_this_inventory`; occlusion, out-of-view and motion-blur intervals, anchors, contacts, source hashes, calibration and local support planes are unknown. A null interval list does not mean the episode has no occlusion.

## Coverage and gaps that shape the architecture

| Geometry / object group | Metadata episodes and views | Candidate pipeline need | Remaining coverage gap |
|---|---|---|---|
| Hula hoop | 00 back, 01 back, 02 back | retain ring hole and thin boundary; separate body silhouette from ring | All three have the back view; geometry and view effects cannot be separated with this group alone. |
| Big red bowl | 03 left, 04 front, 05 front | resolve opening versus underside; distinguish sliding from rigid grasp | Three task variants but only front/left labels; axial rotation observability must be inspected. |
| Iron | 06 left, 07 left, 08 right | resolve horizontal versus upright support; allow sliding during ironing | Left/right only; specular behavior and hand visibility remain unmeasured. |
| White desk | 09 front, 10 left, 11 left | track edges and corners; allow foot-driven motion | Front/left only; tipping, foot pushing and lowering occur in different clips. |
| Black pan | 12 front, 13 left, 14 front | resolve handle and bottom orientation; allow simultaneous grasp and sliding wipe contact | All three specify wiping; repeated action across views does not demonstrate action transfer. |
| Foam grass block | 15 right, 16 front, 17 left | use face appearance to disambiguate rotations; check opposite-corner contacts | Two rotation tasks and one carry/place task; texture uniqueness is unverified. |
| Paint roller | 18 back, 19 left, 20 left | retain thin structure; distinguish whole-object pose from roller rotation | All three specify table rolling, with back/left views; relative roller rotation is unresolved. |
| Pink foam roll | 21 left, 22 front, 23 front | represent uncertain axial rotation; distinguish rolling from sliding using available evidence | Tap/handover and table rolling use different supports; friction/no-slip is unobserved. |
| Short wood stool | 24 front, 25 right, 26 right | check pelvis or thigh support; do not force stool to follow seated body rotation | Sitting front/right and carrying right; object identity alone must not choose a contact model. |
| White laptop cart | 27 right, 28 back, 29 left | preserve sparse silhouette; change support face during lay-down | Lay-down right/back and push/pull left; action and view covary. |

Support and contact coverage extends beyond hand-held objects: floor, table tops, a chair seat and the stool seat are candidate local supports. A single global floor plane cannot represent their heights and local contact surfaces. Contact must be a time-varying graph: a pan can have a grasping hand and a wiping hand simultaneously; a rolling object can touch both hands and a table; a seated human and the stool can participate in body-seat and stool-floor relations concurrently. A mutually exclusive per-frame label such as `grasp OR support` would erase this evidence.

The metadata does not establish how much of each video is occluded, how fast the image motion is, whether the target leaves the image, whether background people appear, or which surface is visible. Those are first-pass observation tasks. Candidate high-motion tasks justify inspecting blur, not labelling a clip blurry. Similar names, dates, tasks or views do not establish independent sampling, and the challenge selection's bias-reduction intent is not demonstrated by these counts.

## Thirty-episode matrix

Camera and frame counts below are source metadata. Task summaries, geometry needs and diagnostic priorities are inferred hypotheses. Exact unabridged task strings and every unknown observation field are in the JSON. `E` = engineering baseline candidate; `X` = proposed expanded coverage case; `D` = established dev-mini; `F` = existing first3 integration fixture. A dash has no selection meaning. Every row is visually unreviewed in this inventory and has an unassigned holdout status.

| Ep | Object | Camera | Frames | Intended task, summarized | Candidate diagnostic priority | Scope |
|---:|---|---|---:|---|---|---|
| 00 | Hula hoop | back | 790 | step inside; lift over body; replace on floor | ring boundary; body occlusion; reappearance | D, F |
| 01 | Hula hoop | back | 668 | grasp right; pass body through; handover left; place down | handover timing; correct limb association; ring hole | F |
| 02 | Hula hoop | back | 866 | both-hand pickup; jump through three times | fast orientation; thin-boundary sampling; jump timing | X, F |
| 03 | Big red bowl | left | 592 | flip bowl onto round table | opening orientation; contact face change | - |
| 04 | Big red bowl | front | 747 | both-hand push while walking around round table | sliding versus grasp; local table plane; changing visibility | - |
| 05 | Big red bowl | front | 668 | floor pickup right; handover left; place on chair | handover; height change; distinct support surfaces | - |
| 06 | Iron | left | 816 | right-hand pickup to upright; left-hand pickup to horizontal | upright/horizontal modes; support switches | D |
| 07 | Iron | left | 777 | both-hand pickup; rotate while walking; place on floor | rotation branches; moving person; ground recontact | - |
| 08 | Iron | right | 634 | kneel; right-hand pickup; iron ground; place down | surface sliding; kneeling body; no forced no-slip grasp | - |
| 09 | White desk | front | 415 | both-hand flip desk from one side to other | pivot versus translation; changing support face | D |
| 10 | White desk | left | 577 | push desk with left foot | foot contact; release timing; sliding freedom | X |
| 11 | White desk | left | 877 | both-hand grasp near edge; lower desk to ground | support transition; controlled tipping; hand-edge contact | - |
| 12 | Black pan | front | 405 | one-hand pickup; other-hand circular bottom wipe; replace | grasp plus wiping; bottom/handle orientation; scale | E |
| 13 | Black pan | left | 425 | one-hand pickup; other-hand circular bottom wipe; replace | grasp plus wiping; bottom/handle orientation; scale | - |
| 14 | Black pan | front | 442 | one-hand pickup; other-hand circular bottom wipe; replace | grasp plus wiping; bottom/handle orientation; scale | - |
| 15 | Foam grass block | right | 501 | both-hand corner pickup; repeated rotations; replace | face correspondence; rotation amplitude; corner contacts | - |
| 16 | Foam grass block | front | 360 | both-hand corner pickup; repeated rotations; replace | face correspondence; rotation amplitude; corner contacts | E, D |
| 17 | Foam grass block | left | 419 | both hands to one-hand underside support; turn; floor placement | contact switch; support versus grasp; table-to-floor scale consistency | - |
| 18 | Paint roller | back | 535 | pickup; roll on table while moving around; lay down | roller articulation ambiguity; surface constraint; thin handle | - |
| 19 | Paint roller | left | 443 | pickup; roll on table while moving around; lay down | roller articulation ambiguity; surface constraint; thin handle | - |
| 20 | Paint roller | left | 549 | pickup; roll on table while moving around; lay down | roller articulation ambiguity; surface constraint; thin handle | - |
| 21 | Pink foam roll | left | 563 | one-hand hold end; tap ground; switch hand; repeat | end identity; intermittent impact timing; axial ambiguity | - |
| 22 | Pink foam roll | front | 533 | pickup; hand-driven rolling back and forth on table | rolling/sliding distinction; axial ambiguity; local table plane | X |
| 23 | Pink foam roll | front | 552 | one-hand hold end; tap ground; switch hand; repeat | end identity; intermittent impact timing; axial ambiguity | - |
| 24 | Short wood stool | front | 420 | sit; change body facing while seated; stand | seated support; occlusion; body/object motion separation | X, D |
| 25 | Short wood stool | right | 365 | both-hand pickup; one-hand carry; both-hand replacement | carry versus support; contact release; thin legs | X |
| 26 | Short wood stool | right | 399 | sit; change body facing while seated; stand | seated support; occlusion; body/object motion separation | - |
| 27 | White laptop cart | right | 440 | both-hand pickup; lay sideways; release | support face changes; sparse geometry; release timing | - |
| 28 | White laptop cart | back | 366 | both-hand pickup; lay sideways; release | support face changes; sparse geometry; release timing | - |
| 29 | White laptop cart | left | 419 | one-hand forward push; two-hand sideways pull; release | push/pull transition; rolling/sliding ambiguity; no rigid hand lock | - |

## Development exposure and evaluation assignment

The established regression scope in [development.md](development.md) is `dev-mini = [0, 6, 9, 16, 24]`; `first3 = [0, 1, 2]` remains a limited hoop integration fixture. The engineering baseline candidates are episode 12 (pan) and 16 (foam block), consistent with G1. Proposed second-round coverage is episode 2 (ring/jumping), 10 (foot push), 22 (table rolling), 24 (seated support), and 25 (carrying the same object category). These cases supplement existing regression scope and do not replace acceptance rules or prove reconstruction quality.

This expansion still leaves important modes for later review: bowl handover to a chair, iron surface sliding, roller articulation, desk/cart tipping and cart push/pull. All 30 episodes remain the final coverage target. IDs are coverage/reporting selectors only; solver routing must use observable properties and inferred contact states rather than per-episode or per-object patches.

All holdout assignments are **unassigned pending historical-use audit**. Absence from the proposed case set does not make an episode unseen: first3 experiments, prior previews, manual asset work, threshold selection, branch experiments and human corrections can all create exposure. Audit complete object groups, record what was viewed/fitted/tuned, and then freeze the evaluation policy before claiming transfer. Already exposed groups may serve regression evaluation, with that limitation stated. If all groups were previously used, report retrospective coverage and robustness instead of inventing a clean holdout.

Separate object-group transfer, interaction-group transfer and input-perturbation robustness. They answer different questions. Frame-level holdouts from the same sequence remain correlated; shared model errors and fitting dependencies must be recorded. Repeatedly tuning to review observations consumes their independence. Metadata exposure tags alone do not guarantee independent generalization, and local image checks cannot establish hidden-reference 3D accuracy or official CD/ACC.

## Requirements for the four modules

| Module | Evidence or state to add | Coverage motivation |
|---|---|---|
| Platform / inputs | Verified source frame/timestamp map, target identity, visibility and uncertainty, fit/review separation, direction-diverse anchors | All episodes; thin structures and moving bodies need source-resolution observations, with no frame deletion during occlusion |
| Human | Verified native MHR; separate body, hand, foot and pelvis/thigh observations; stable shape; uncertainty by time and anatomical region | Foot push, kneeling, seated rotation and handover cannot use hand visibility as a proxy for whole-body reliability |
| Object | Episode-local metric geometry, frozen origin/axes, exactly-once scale, candidate geometry/appearance symmetries, full-frame pose hypotheses | Rings/leg gaps need preserved holes; vessels need opening direction; cylinders need axial ambiguity; rollers may violate a single rigid-body appearance model |
| Physics / refine | Simultaneous contact graph, local support surfaces, contact-release events, sliding/rolling alternatives, bidirectional occlusion recovery and bounded SolvePlans | Grasp plus wipe, hand plus table, seat plus floor, tipping, foot push and intermittent end taps |
| Platform / validation and export | Immutable complete SceneState references, dependency invalidation, per-episode/worst-case diagnostics and round-trip checks | A good subset or a smooth frozen trajectory cannot satisfy whole-video coverage and motion fidelity |

Local support surfaces are nuisance geometry inferred from the current permitted video, with their own frame convention, visible extent and confidence. They are not preloaded round-table/chair/floor calibrations. A support label is only a candidate: determine whether the surface is stationary, estimate its geometry, and verify the applicable interval before imposing it. Separate whole-object SE(3) from evidence of a roller or wheel rotating locally; any richer internal state must still map to the required export without inventing unsupported degrees of freedom.

## Next evidence pass and boundaries

1. Decode each authorized source video, hash its bytes, verify metadata frame counts and preserve original IDs and timestamps. Register preview/resizing transforms separately; preview counts are not source counts.
2. Review target identity, observable geometry, visibility, image quality and potential anchors. Attach frame-indexed evidence to every claim and retain unknowns where evidence is absent.
3. Confirm or reject candidate events, simultaneous contacts, local support surfaces and symmetry hypotheses. Preserve alternative explanations and confidence instead of treating text instructions as labels.
4. Audit historical use and freeze fitting/review policies, solver rules and budgets. Report disagreements and worst cases without converting proxy losses into official scores.
5. Feed observations to EpisodeEvidence, accepted state to SceneState and bounded repair decisions to SolvePlan. This inventory is a coverage plan, not a substitute for those executable contracts.

Track 2 content and derivatives remain excluded from every step. Cross-episode pooling of geometry, scale, camera or human identity remains disabled pending applicable permission; even the three clips of an object are reconstructed independently. Reconstruction anchors may be selected by video evidence, while R1 alignment remains at the reference trajectory's explicit first original-video frame when a permitted reference exists. Metadata frame counts alone cannot close R5, and a complete pose estimate must still be exported for every source frame including hidden intervals.
