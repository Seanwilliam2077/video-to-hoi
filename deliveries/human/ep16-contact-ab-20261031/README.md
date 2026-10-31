# EP16 shared-camera manual-contact A/B

Status on 2026-10-31: object preparation, both controlled pairs and independent
native evaluation are complete. The revised contact pair does not pass the
frozen motion and usable-object skin guards. All six comparison videos and the
verified local review archive are complete. This is a local
Track 1 EP16 experiment, not an accepted reconstruction or official submission.
Contact-network training, EP11 execution and an all-30 rerun have not started.

## Purpose and fixed controls

The owner requested one current-camera human/object preparation followed by a
no-contact versus manually phased contact experiment. The executor is Codex on
the owner's Windows workstation, with independent object, solver and evaluation
contributors recorded in the task receipts. Work is on `human`, starting from
`262b7b1d3e83787820cca0026af23995d7b08fe2`.

Both arms use the same original native SAM initialization and final routed
MediaPipe observations from the [refined hand pilots](../egohand-refined-pilots-20261031/README.md).
The current MediaPipe reconstruction is a historical parity check and, in the
final pair, a read-only contact-anchor selector. It does not replace the common
optimizer initialization, supply hidden-contact labels or add image targets.

The numerical solver is the existing native-MHR sequence fitter: CPU1,
600 iterations, learning rate0.001, model batch8, original stride5 image
holdout, eight-frame support collar, correction-budget weight1 and safety
fraction0.2. Shape[45], skeleton scales[68], root and all nonselected controls
remain fixed. Finger changes are bounded by0.65rad, selected arm controls by
0.1rad and wrist displacement by0.04 native estimated metres. Object geometry,
poses, camera and scale are fixed during both human solves.

The intervention is:

- **A:** the unchanged image/pose/temporal objectives, without contact losses.
- **B:** the same objectives, support masks, variables and budget, plus the
  existing soft contact-anchor weight0.01 and fingertip-interior proxy weight0.005.

The optimizer's hand-support policy is unchanged in both arms. Manual phases
do not activate unsupported hidden-hand controls. In the129 object-usable
carry frames, left finger controls have **zero mutable frames**, while right
finger controls are available throughout. This limits what this experiment can
establish about the hidden left hand.

## One camera and coordinate preparation

The old contact object used an assumed focal length near1330px and a different
human depth gauge. Reusing its old 3D points/poses would not match the current
human initialization. This experiment reconstructs object geometry/poses under
the current estimated camera:

```text
K = [[818.7458496, 0, 767.5],
     [0, 818.7458496, 575.5],
     [0, 0, 1]]
OpenCV camera: x right, y down, z forward; estimated metres.
Original video: 360 consecutive frames, 1536 x 1152, 30 fps.
```

Only EP16 RGB, its verified original-frame masks and its own human estimates
are used. The rounded-cube shape family was previously inferred from this
video's silhouettes. Frame0 shape/pose fitting, SIFT-to-surface ray intersections
and the complete temporal object trajectory are recomputed under the new K;
old 3D reference points are not reused.

A single depth gauge is frozen before the A/B solves:2.879471779m, the pooled
original-SAM wrist depth over frames90–230 excluding stride5 IDs. The source
review shows a conservative airborne-carry interval. Object centre depth near
wrist depth is an explicit initialization assumption, not a measurement of true
object depth. The wrist distribution is not a calibrated uncertainty interval.

The final object is one static baked-metric mesh with0.20902645m extent,
1,474 vertices and2,944 faces. It is closed, convex, consistently wound and has
positive volume. Object scale is applied once; the contact solver's scale is1.
World/camera geometry is `P[t] = V @ R[t].T + translation[t]`, with proper SO(3)
rotations and all original IDs0–359 retained.

Conditional object checks are:

| Check | Result |
| --- | ---: |
| Mean visible-mask IoU | 0.951746 |
| Spatially distinct heldout image observations | 2,773 |
| Mean heldout reprojection | 0.763px |
| Maximum translation step | 0.01848m |
| Maximum rotation step | 6.822 degrees |
| Object-pose reliability gate | 313 / 360 frames |
| Usable frames within the manual carry interval | 129 / 176 frames |

These checks support input consistency and conditional 2D tracking; they do
not establish hidden geometry, physical size or true contact. All360 poses are
retained, including unreliable/inferred frames. Both filtered and unfiltered
geometry diagnostics are reported. No reference alignment or official score is
computed.

The first contact export failed because the human environment lacked `trimesh`.
It was recovered using the existing expert environment. The failed attempt,
successful recovery and exact unchanged fitted geometry/source identities are
preserved separately.

## Manual phases and anchor provenance

Manual review inspected26 predefined source frames before fitting. Root also
inspected the pickup/release sheets and prepared-object overlays. The frozen
schedule is intentionally conservative:

| Original frames | State | Attraction |
| --- | --- | ---: |
| 0–74 | Approach/pickup; exact first touch unresolved | 0 |
| 75–250 | Coarse bimanual airborne carry | 1, with the existing boundary ramp |
| 251–289 | Placement/release transition unresolved | 0 |
| 290–359 | Clearly released in inspected RGB | 0 |

Weights also respect object-pose reliability. A positive coarse grasp interval
does not verify particular fingertip contacts or no-slip. The existing contact
implementation selects at most two already-near-surface tips per hand; stable
local anchors are a weak hypothesis. No attraction is applied in uncertain
pickup/release intervals. Within admitted carry, fixed 3D anchors assume stable
local contact and can suppress unobserved rolling/sliding; tangential freedom
is not modelled. No attraction persists outside the admitted interval, although the unchanged whole-clip optimizer can still
couple nearby frames through its other objectives and best-iterate selection.

The first preserved pair selected anchors from the original SAM initialization.
It admitted only two left-hand anchors; left controls were frozen throughout
usable carry. The attraction distance therefore remained exactly unchanged.
This is a structural intervention failure, not evidence that contact modelling
cannot help. A matches the historical current MediaPipe parameters bitwise.

The second pair adds an explicit, validated `--contact-anchor-reference` input
for **selection only**, using the fixed current image-only MediaPipe native MHR
result. This corrects the mismatch between the preliminary eligibility check
and actual anchor-selection input. Source video, original IDs, decoder, camera,
identity, scales and root must match exactly. Both optimizers still start from
the same original SAM parameters. Object, manual phases, loss weights, bounds,
support policy and iteration budget do not change. Both attempts and their
separate frozen source snapshots remain inspectable.

## Independent checks and current limitations

The evaluation criteria were frozen before the first human solve and retained
for the second pair. They include complete fresh native decoding, exact
parameter preservation, identical A/B image weights/support masks, both original
and final MediaPipe references, heldout image agreement, all adjacent original
frame motion, original support boundaries, manual phase windows, opening changes
and actual retained native updates.

Full native hand-skin depth against the same convex object is measured for all
360 frames, including release. The verified skin selection uses blend-weight
threshold0.5 and2,321 vertices per hand. Conditional penetration is stratified
by phase, both with and without the pose-usable gate. Tip-to-surface distance,
anchor residuals and object-local drift are separate hypothesis diagnostics,
not independent contact truth or the official penetration score.

Inherited B-versus-A engineering nonregression tolerances are5%+0.25px for image
agreement and0.5mm mean/2mm worst for hand-skin depth. Original motion guards are
also retained. These tolerances are local development choices, not universal
accuracy thresholds. Improvement in a loss's own anchors alone is insufficient
to establish useful contact reconstruction or authorize contact-model training.

The first pair's right mean/worst depth rises slightly relative to matched A;
neither its unchanged anchor distance nor its additional motion failures
supports a successful contact intervention. Existing left-hand penetration is
already severe in the original hidden-hand prediction under this geometry and
is not created by B. Numerical consistency of camera/coordinates does not
resolve this human/object hypothesis conflict.

## Final paired numerical result

The revised A remains bitwise equal to the preserved current MediaPipe native
parameters. All360 freshly decoded baseline/A/B frames pass mesh/joint parity.
Identity, scales, root, unselected parameters and original IDs are preserved;
the A/B active-hand masks and image weights are identical. These are execution
and representation checks, not reconstruction accuracy.

B now admits left thumb/middle and right thumb/middle anchors from the explicit
read-only current-MediaPipe selector. Right attraction therefore has mutable
variables; left carry fingers still do not. Compare A and B against the same
frozen B anchor locations:

| Hypothesis/self-consistency diagnostic | A | B |
| --- | ---: | ---: |
| Left weighted frozen-anchor distance | 65.569mm | 65.569mm |
| Right weighted frozen-anchor distance | 73.032mm | 64.532mm |
| Right skin worst depth, all176 carry frames | 24.125mm | 23.253mm |
| Right skin worst depth, fixed129 usable carry frames | 14.027mm | 16.274mm |
| Right skin mean depth, fixed129 usable carry frames | 0.01677mm | 0.02674mm |
| Left skin mean depth, all176 carry frames | 41.270mm | 41.270mm |
| Left skin worst depth, all176 carry frames | 104.112mm | 104.112mm |

The right usable-frame worst depth increases2.246mm, exceeding the frozen2mm
nonregression tolerance. The all-frame worst depth improves, so reporting only
that aggregate would hide a regression in the predeclared reliable subset.
These depths refer to the same uncertain reconstructed cube and estimated units;
they are not physical truth or the official penetration score. Clearly released
frames290–359 have zero sampled skin penetration in both arms under this object
hypothesis; that alone does not establish correct finger articulation or release
timing.

The same final MediaPipe image reference gives the following stride5 diagnostic:

| Hand | Supported heldout frames | A reprojection | B reprojection |
| --- | ---: | ---: | ---: |
| Left | 5 | 5.408px | 5.225px |
| Right | 63 | 4.375px | 4.159px |

Both image guards pass, but these are method-dependent detector distances, not
independent hand/contact truth. The left-hand sample is particularly sparse.
The paired motion check finds11 failing supported finger edges and14 failing
supported wrist edges against A under each detector reference. Finger support
denominators differ:312 original-reference and314 final-reference edges. No
unknown-motion correction-budget failures are added. A's paired zero failures
are by construction and do not establish that A is accurate.

Independent world-space convex-hull checks, local-coordinate checks and sampled
triangle distances agree on the left-hand conflict. Fresh native decoding,
side mapping, metres/centimetres, rigid inverse transforms and normals pass.
The hidden left pose and assumed object shape/depth remain physically ambiguous;
the conflict cannot be attributed to a transpose/units error or to B creating it.

**Decision:** useful manual-contact reconstruction has not been established.
Keep both trials as experimental evidence. Do not train the contact predictor,
run EP11, expand to30 videos or replace selected dashboard results from this
experiment. The next investigation should address the occluded-left pose and
relative-depth ambiguity, use actual hand-surface contact rather than joint-tip
attraction alone, and separately test bounded hidden-hand variables with the
same variable policy in both arms. A learned contact switch cannot repair
frozen variables or an inconsistent hand/object initialization by itself.

## Software and review evidence

Implementation is on `human` in
[`ff0abfd`](https://github.com/Seanwilliam2077/video-to-hoi/commit/ff0abfd).
The final explicit workflow list passed **1071 tests, 25 skipped**. Node passed
**44 tests**; registry/catalog checks passed without result promotion. Five
independent synthetic rendering tests include exact serial/process-pool output
agreement and cross-entity depth occlusion. No fitting/contact mathematics was
changed: the solver interface adds a source-bound anchor selector and strict
controlled-pair execution.

All-triangle human/object rendering uses a perspective-correct shared Z-buffer.
Context and closeup sources retain their own exact identities. The closeup is a
display-only variant using fixed original-pixel ROI[520,500,850,830] for every
frame, source RGB and both methods. It changes one viewport assignment, not
input crops, camera, object, optimization or evaluation. The source difference
is preserved explicitly. No table/background depth is supplied, and mesh-to-mesh
occlusion must not be interpreted as true real-scene visibility.

## Videos, checks and sharing

All six MP4 files were fully decoded: each view contains360 original frames
at30fps, plus frames40–110 and230–300 replayed at10fps without interpolation.
The independent reviewer inspected20 predefined closeup and10 context JPEGs;
root reviewed an additional six images with overlap. Static samples show no
clear overall B grasp-fidelity advantage. At sampled release frames, both poses
remain imperfect relative to RGB. Full file decoding and static sampling do not
claim continuous human playback, precise true release timing or hidden-hand
correctness.

Local entry points are:

```text
J:/video-to-hoi/runs/ep16-contact-ab-20261031/review-closeup02/index.html
J:/video-to-hoi/runs/ep16-contact-ab-20261031/review02/index.html
```

Columns are source RGB / A without contact / B with manual-phase contact.
Blue denotes the estimated native human; orange denotes the uncertain object.
The `Manual attraction OFF` footer includes both excluded manual intervals and
object-pose uncertainty. For example, RGB110 is carrying but its object gate
abstains. OFF must not be read as a measured release label. The two indexes have
an explicit interpretation note; original generated indexes and a separate
HTML-only clarification receipt are preserved. Every video and geometry byte
remains unchanged.

The self-contained shareable archive is:

```text
J:/video-to-hoi/work/ep16-contact-ab-20261031/local-delivery03/ep16-contact-ab-20261031-review02.zip
Bytes: 17,279,521
SHA256: d69108fea22f435a044f8e30e34e64cc826d9cef44a682418372fd39c1f92fd3
```

A reviewer extracts the entire ZIP and opens `index.html`; `context/index.html`
provides the separately bound full view. No model server is needed to watch.
The archive contains six videos, samples, exact baseline/A/B and anchor-selector
parameters, object geometry/poses, image references, phases, both full numerical
evaluations and actual execution/review records. Full per-frame human vertices,
source video and model weights are excluded. Recreating3D requires the exact
separately permitted MHR decoder and source inputs.

The first archive attempt was correctly rejected by full payload verification:
Windows backslashes in context-image inventory names differed from normalized
ZIP members. It and its original packager source are preserved. Recovery
normalizes member names before inventory construction and writes a new archive
directory. CRC and every final member SHA were checked; no model, geometry,
video or evaluation was rerun or altered for this packaging repair.

Git includes the [final compact summary](summary.json),
[first-pair summary](attempt01-summary.json),
[first-pair structural note](attempt01-structural-attraction-note.json),
[combined integrity record](completed-integrity.json),
[exact archive identity](local-review-package.json),
[source/evidence inventory](evidence-inventory.json) and
[reproduction notes](reproduction/README.txt).
The full numerical reports and large media are in the ZIP, not Git.
Executed source snapshots distinguish both fitting/evaluation attempts and both
display variants. Recipes retain local absolute paths and require deliberate
relocation plus independently permitted assets; this is not a portable installer.

Live Claude module/status publication is unavailable without the required
Artifact tool. The preexisting unrelated `docs/status.html` edit is preserved
and excluded from commits. No selected dashboard result is promoted and no
main deployment is performed.
