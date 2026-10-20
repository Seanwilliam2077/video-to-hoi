# Episode 16: independent hand observations and native MHR sequence refinement

Experiment date: 2026-10-15. Scope: the permitted Track 1 episode 16 only,
360 original frames at 30 fps. The hand and contact experiments are complete;
the official control's actual execution status is recorded below.

## Current conclusion

Independent image observations and a finger-enabled native MHR sequence fit
improve the visible right hand substantially over the existing baseline. Within
the original fixed-identity hand experiment, the preferred Human-only candidate
is `coupled-wrist-v3-spatial`. It reduces right-hand image discrepancy on withheld
frames from 7.781 to 4.010 original-image pixels. A visible palm flip near frame
110 is removed in the inspected time window. These measurements support better
image agreement and motion continuity, not verified 3D accuracy or competition
scores. The occluded left grasp has no accepted independent observations and
is still unresolved.

The principal failure was broader than finger articulation: noisy upper-arm and
forearm twist rotated the whole hand. Finger-angle smoothing alone could not
repair this inherited palm rotation. Bounded native arm/wrist coupling and a
phase-conditioned wrist-relative spatial motion term address it without
replacing the hand mesh or changing the person's identity and skeleton scale.

The full local official control also completed, with mixed quality. CoCoNet
recovers an object track lost during carrying; post-optimization makes its
translation more continuous. Right-hand image agreement worsens through those
stages, and local finger parameters stay frozen. Our weak-contact experiment
also fails the native hand-skin penetration check. No current candidate is
accepted as a complete, physically correct grasp reconstruction or a proven
competition-score improvement.

## Requested experiments and evidence

| Requested step | Actual work | Status |
|---|---|---|
| Independent finger observations | Stable crops, independent MediaPipe image detections, side association, handedness and mirrored-crop agreement checks | Right-hand evidence obtained with a disclosed alternative; left grasp unobserved; HaMeR/HandFlow not run |
| Whole-sequence finger/wrist optimization | Five full-360-frame variants, true native finger controls, bounded arm chain, image and motion losses | Completed; v3 preferred for further review |
| Full official CARI4D control | Pinned source, verified generic models, our own masks/textured object, measured representation adapter | All eight stages executed over 360 frames; all three videos decoded; disclosed full-mesh configuration variant |
| Stable object-local grasp contact | Closed RGB-derived object hypothesis, manual conservative phase, weak anchors, release, matched control and scale tests | Completed negative experiment; nominal rejected and both scale tests abstained |

The baseline run is preserved at `runs/human-ep16-cpu-full/`. New variants are
under `runs/human-ep16-hand-round2/`; observations, official preparation and
object experiments are under `work/human-hand-round2/`. No other-track data,
reference geometry, reference poses, scores or cached benchmark outputs were
used. Generic pretrained model assets and source code have explicit provenance.

For the earlier literature review and proposed comparisons, see
[Human reconstruction strategy as of 6 October](human-strategy-20261006.md).
That research cutoff is retained; the measurements in this report are from the
separate 15 October execution.

## 1. Independent observations: what is supported by the video

The licensed `MANO_RIGHT.pkl` asset required by the considered HaMeR/HandFlow
paths was absent from the verified local inventory. Those models were **not
run**. The actual observation experiment uses MediaPipe HandLandmarker 0.10.21
on CPU, with its public generic hand model. It supplies 2D image targets, not
a standalone verified 3D hand reconstruction.

SAM hand locations seed the crop only. Crop centers use a five-frame median;
crop size uses a fifteen-frame median and a bounded 112–208-pixel extent,
then resizes to 384 pixels for detection. IMAGE mode computes a fresh detection
in every frame. There is no interpolated target, copied tracking prediction,
or fitting target taken from SAM's own projected hand.

Anatomical side is associated using the nearest original SAM wrist. Gates reject
duplicate assignment, ambiguous/wrong wrist association, handedness disagreement,
and disagreement after horizontal flip and inverse mapping. Broad local skin-color
support is a heuristic; the brown object can resemble skin. MediaPipe supplies
no calibrated per-joint visibility probabilities. Consequently, stored confidence
values are binary fitting reliability weights, not measured visibility or truth.

| Observation result | Left | Right |
|---|---:|---:|
| Accepted hand frames out of 360 | 9 | 307 |
| Positive landmark weights | 183 | 5,511 |
| Accepted hand frames in every-fifth-frame validation subset | 2 | 61 |

The full gated observation pass took **68.047 seconds**. Missing/ambiguous points
receive zero weight. **All nine accepted left-hand frames are after release
(281–359). There are zero accepted left-hand observations during approach,
pickup, stable carry or placement/release.** The two left validation frames
therefore validate post-release image agreement only, not grasp articulation.
The left hand is mostly behind the object during carrying; regularizing that
missing interval does not constitute new visual evidence.

Observation NPZ SHA-256:
`4841efedc111b40e79e1d8c8a7b7fa6617a02b1f6e7bf34bb067f01e020122dd`.
The detailed report and rejection counts are in
`work/human-hand-round2/expert/gated360/report.json`.

## 2. Fingers participate in the full native optimization

The optimizer runs all 360 frames together through the official native MHR
kinematic path. The 42 hand landmarks were verified against the rig's joint
names and against full native model forward output. Right finger controls are
`68:95`, left finger controls `95:122`. They are optimized before full native
mesh export; separate hand meshes are not pasted onto fixed wrists.

The objective combines robust absolute and wrist-relative 2D evidence, observed
relative velocity, motion-gated angular acceleration, bounded pose updates,
and native arm/wrist coupling. Coupled variants allow fourteen verified
arm/wrist controls to change by at most 0.10 radians, with wrist displacement
limited to 0.04 metres from the initializer. Finger updates are bounded to
0.65 radians per control. Shared identity `[45]`, skeletal scales `[68]`, root,
camera, and all other controls stay fixed.

These delta trust regions limit change from the initializer; they are not an
explicit projection into anatomical absolute joint limits. Anatomical validity
still needs inspection and stronger constraints.

Every fifth original frame is excluded from all image position, velocity and
motion-gate targets. Temporal terms can still update these frames. The withheld
observations were used to compare variants, so this is a **validation set used
for selection**, not an untouched final benchmark. The detector and SAM crop
seeds can have correlated errors. The left subset has only two detected frames.

| Variant | CPU fit plus native export (s) | Right validation image discrepancy (px) | Left discrepancy (px; only 2 frames) | Right relative second-difference P95 (px/frame²) | Left P95 |
|---|---:|---:|---:|---:|---:|
| Original fixed-identity baseline | — | 7.781 | 5.063 | 7.768 | 3.503 |
| Finger observations, preserve unsupported gaps | 50.797 | 7.141 | 4.752 | 6.288 | 3.920 |
| Finger observations with bounded temporal prior | 50.922 | 7.128 | 5.143 | 5.963 | 3.486 |
| Coupled arm/wrist v1 | 67.141 | 7.083 | 5.524 | 5.554 | 2.831 |
| Coupled arm/wrist v2 | 64.781 | 4.586 | 3.864 | 5.735 | 3.261 |
| Coupled v3 with stable-carry spatial prior | **62.938** | **4.010** | **3.865** | **2.449** | **1.855** |

All values use the original 1536×1152 pixels. Image discrepancy is a weighted
mean distance to independent detections. Motion P95 is over all per-landmark
wrist-relative predicted second differences, not the organizer's acceleration
error against reference motion. Neither number is an official score.

The general every-frame viewer aggregates its cards over the projectable native
whole-body joints, rather than the 20 wrist-relative finger landmarks per hand
used here. Its second-difference card at frame N uses N-2,N-1,N, centered on N-1;
the diagnostic plots explicitly label the center frame. Do not compare these
different aggregations as if they were the same hand-quality measurement.

The first observation-only candidate worsens the left-hand motion P95 because
sparse supported intervals introduce boundaries. Coupled v1 also introduces
periodic corrections around withheld frames. V2 smooths the **arm correction**
to remove that periodic artifact but leaves a genuine palm flip at frame 110.
These variants are retained as failed or incomplete ablations.

### Why frame 110 needed a different constraint

Raw upper-arm twist at frames 109/110/111 is approximately
`0.290 / -0.001 / 0.322` radians; forearm twist is
`-0.601 / -0.908 / -0.598`. Finger controls in v2 are already smoother, but the
whole palm still inherits this chain rotation. Frame 110 is also a withheld
image frame. Inspecting RGB frames 108–112 shows a stable grip, rather than the
predicted one-frame curl/rotation.

V3 adds a robust second-difference penalty on native wrist-relative 3D hand
landmarks during the **manually admitted stable-carry interval 75–250 only**.
Independent observed articulation reduces its strength. No object surface or
object contact target is used in this experiment. The existing angle and wrist
bounds suffice; they were not enlarged to repair this frame.

The right-hand frame-mean second difference centered at 110 changes from
13.511 in the baseline, to 19.458 in v2, to 2.125 px/frame² in v3. The inspected
108–112 window no longer has the visible palm flip. Pickup, placement/release,
and released-hand motion remain: mean right relative step in v2→v3 is
0.612→0.611, 0.385→0.380, and 0.887→0.888 pixels respectively. This is evidence
of selective change, not proof that every true finger motion is preserved.

The baseline root-depth trajectory also has a 0.06959-metre second difference
centered on frame 110. This Human hand experiment keeps that root fixed. Body
and global motion therefore require a separate validated optimization.

### Structural and visual verification

All 360 native output frames are present and finite. An independent reopen and
full native forward on frames 0,40,110,120,240,285,359 reproduces saved vertices
and joints within `4.77e-7` metres. Identity/scales and nonselected controls are
bitwise unchanged. Camera and every native face are preserved. Withheld image
weights are zero. See the candidate's `independent-acceptance.json`.

Videos use the same original frame, timestamp, camera and crop across panels.
Mesh-only hand panels hide large joint dots that otherwise obscure the fingers.
The renderer draws human surfaces even behind the object: an overlaid hidden
hand is **not** evidence of penetration or true visible contact. Numeric contact
diagnostics, where available, use actual 3D geometry separately.

Useful review windows include 38–42 (hidden left hand), 108–112 (palm flip),
and 283–287 (release). The original baseline and all failed ablations remain
available for comparison.

To repeat the preferred fit from the existing verified baseline, use a fresh
output directory:

```powershell
& work/human-runtime/venv/Scripts/python.exe tools/refine_human_hands.py `
  --baseline runs/human-ep16-cpu-full/fixed-fit/episode_000016_human.npz `
  --observations work/human-hand-round2/expert/gated360/observations.npz `
  --manifest runs/human-ep16-cpu-full/manifest.json `
  --review-arrays runs/human-ep16-cpu-full/fixed-fit/review_arrays `
  --output runs/human-ep16-hand-round2/v3-repeat `
  --device cpu --threads 1 --iterations 180 --holdout-stride 5 `
  --unobserved-policy bounded_temporal_prior --relative-weight 0.5 `
  --optimize-wrist-chain --arm-update-temporal-weight 0.08 `
  --spatial-acceleration-weight 0.2 `
  --motion-phases work/human-hand-round2/contact/phases.json
```

This reuses independent observations and pretrained models. It is test-time
sequence fitting, not training a new reconstruction network. It does not rerun
the original approximately 96-minute CPU body/hand inference.

## 3. Official CARI4D: inputs, adapter and current execution status

The control is pinned to official v0.3.0 source
`595ad68d897b986cd154d4575a022411af096eb3`. It uses our own full-resolution
person/object masks and textured object reconstructed from this same video.
No example mesh, reference trajectories, opaque collision proxy or unknown
benchmark asset is used.

The actual SAM2.1 tiny GPU mask pass covered all 360 frames in 344.97 seconds,
with 0.709 GiB peak allocated and 1.064 GiB reserved CUDA memory. Tiny is an
explicit local input-generation choice, not a claim of parity with a larger
official segmenter. All 720 PNGs match the packed HDF5 bytes, frame IDs are exact,
and no mask is empty. Sampled overlays stay on the person and object. These are
predicted masks, not reference annotations. Packed mask SHA-256:
`38d5fd6eb836c725e95b5c568e29315a8acfe5bc12a1988d0a5c08ab8969fb33`.

### Representation conversion was measured, not assumed

Native submission pose/scales/shape `[136]/[68]/[45]` differ from the official
network's internal body, continuous-hand and scale-PCA blocks. An adapter audit
ran both raw and shared-identity outputs through the exact official head and
native geometry for **all 360 frames**:

| Conversion audit | Worst frame-mean vertex difference | Worst vertex difference |
|---|---:|---:|
| Raw native round trip | 0.000707 mm | 0.002007 mm |
| Shared fitted scales projected into official PCA layout | 0.056241 mm | 0.560209 mm |

The maximum anatomical-joint discrepancy is 0.562054 mm and projected vertex
discrepancy 0.158185 px. The scale map has rank 25 with three null modes;
minimum-norm PCA projection is needed. Shared fitted 68-dimensional scales do
not lie exactly in the official subspace, so this is sub-millimetre agreement,
not a lossless conversion. Root and body/hand six-dimensional rotations use
different column layouts; the adapter uses upstream helpers rather than
truncation. Dummy joint 0 is excluded from anatomical-joint comparison.

Audit-only NPZs are not drop-in official initializers: preprocessing metadata,
coordinate conventions and stage contracts still matter. The full control
reinitializes from RGB/masks; it does not silently inject the existing NPZ.
See `work/human-hand-round2/official/adapter-audit/conversion-report.json`.

A separate audit of the newly executed official initialization checked the
reverse internal-to-native direction on frames 0,120,240,359. Maximum vertex
discrepancy is `9.82e-7` metres and anatomical-joint discrepancy `5.07e-7` metres.
This direction decodes the official scale subspace exactly; it does not have the
forward projection limitation above. It is a four-frame conversion check, not a
new fitted candidate. Official initialization retains per-frame shape and scale
coefficients, whereas our controlled hand experiment uses one fixed identity
and skeleton. Exporting one shared identity requires a genuine geometry refit,
not field renaming or a claim of unchanged geometry after averaging.

### Full control execution

WSL Docker and the NVIDIA container runtime were installed locally. The exact
PyTorch 2.5.1 / CUDA 12.4 base image passed GPU arithmetic on the existing RTX
3080. The Windows driver and other applications were left unchanged. Source-only
dependency caches avoid upstream example/data downloads. CUDA compilation is
limited to SM8.6 and four jobs. Generic hand-surface data was regenerated from
the verified neutral MHR model. The supported full-human-mesh collision mode
replaces an excluded collision proxy of unverified provenance; this is a
disclosed configuration variant.

The built environment passed the full CUDA contract in 9.90 seconds, including
PyTorch3D, nvdiffrast, Kaolin, FoundationPose native modules and an actual Warp
kernel. SAM and MoGe imports also pass. Building the runtime took 2,974.56 seconds
(49.6 minutes), including diagnosis and retries; this excludes earlier model
acquisition and runtime installation. An optional real-model depth-refit probe
selected batch 256, with 5.62 GB peak reserved CUDA memory and a separate
desktop-memory headroom guard. Batch 360 was not admitted by that guard. The
actual wild-video initialization preserves native SAM parameters and does not
execute this optional 500-step refit, because `--no-align-to-gt-depth` is enabled.
The probe is a resource diagnostic, not a fitting stage used in this result.

All eight actual reconstruction stages completed in **2,712.123 seconds
(45.20 minutes)**, including stage process startup and storage. This excludes
the first-time environment build above, earlier installation/model acquisition
and the later independent audits. All 360 frames and the 96-frame network window
were retained. Configured post-optimization steps are 300; pinned upstream uses
an inclusive loop over indices 0–300, so the saved Adam counters confirm **301
updates**, with no early stop. All recorded full-clip batches contain 360 frames
and all recorded losses are finite. Loss reduction is an optimizer diagnostic,
not an accuracy measure.

| Actual stage | Wall seconds | Sampled GPU-wide peak (MiB) |
|---|---:|---:|
| Depth estimation | 101.972 | 6,749 |
| Export | 37.879 | 3,118 |
| SAM 3D Body initialization | 160.498 | 7,230 |
| Depth alignment and storage | 1,081.693 | 4,338 |
| FoundationPose object initialization/tracking | 121.028 | 9,980 |
| CoCoNet | 205.668 | 5,599 |
| Full-clip refinement | 805.502 | 9,566 |
| Three-stage rendering | 172.892 | 2,752 |

GPU-wide memory is sampled about once per second and includes desktop
applications, driver/context memory and possible sampling omissions. These are
not model allocator peaks or minimum-memory recommendations. The slow depth
storage stage used two PNG encoders at compression level 9 with optimization;
its wall time is not all GPU fitting. The full-human-mesh collision term starts
after step 180 and accounts for the slower later refinement updates.

The original outer Bash wrapper returned 127 **after** all eight subprocesses
and videos completed. Editing that running script inserted 54 bytes and shifted
Bash's later read offset into a trailing argument. This was an avoidable launcher
mistake, not a model-stage failure. Original logs, reports, markers and GPU samples
are preserved with unchanged hashes. A repaired immutable-snapshot launcher
passed a **separate 25.084-second cached replay**, exit 0, reusing all eight stages.
That replay is not relabelled as fresh inference. Details are in
`work/human-hand-round2/official/wrapper-repair-report.json`.

All stage outputs passed exact-frame, finite-array and object-transform audits.
The two original official videos and source-inclusive hand closeup each decode
360 frames at 30 fps, 12 seconds, H.264. They also play and seek offline in Edge,
including the 4,608-pixel-wide three-stage video. Time seeking at an exact frame
boundary can show the preceding frame; the saved-image Human viewer selects
exact frame IDs, and frame-center seeks were independently checked.

MoGe exports a fixed camera from its first-frame estimate: focal length
826.675 pixels and horizontal field of view 85.786 degrees. Our existing Human
comparison uses focal length 1,330.215 pixels, derived from an assumed 60-degree
field of view. MoGe's individual estimates range from 799.624 to 829.543 pixels;
the official export policy uses the first estimate as the fixed camera. Neither
camera hypothesis is measured calibration. This difference changes the
depth/scale interpretation, so raw metric positions or hand-object distances
cannot be compared directly across the two methods. Their projected overlays
must use their respective camera contracts. The record is
`work/human-hand-round2/official/camera-comparison.json`.

The released configuration has training supervision weight `w_mhr_hand: 0`
and contract `body12_freeze_hand_face_all_losses`. The inference evaluator applies
that contract and zeros the frozen delta outputs. Separately, default
`smplh_parity` post-optimization freezes hand, identity, scale, root and human
translation; it optimizes selected body rotations and object translation, with
optional object rotation. Changing only the training hand-loss weight would
neither supply trained finger predictions nor change the post-fit parameter
groups. The full pipeline is a valuable whole-HOI control, but its presence alone
does not establish an independently observed finger-articulation correction.

### Measured stage quality, separate from execution success

The following rows use the actual committed stage outputs and the
same fixed official camera. They are input-consistency diagnostics, not official
scores. The object masks were supplied to the pipeline and are not held-out
annotations. The right-hand subset is the same 61 detected every-fifth frames
used in the earlier experiment; the official stages use none of these MediaPipe
targets during inference.

| Official stage | Right image discrepancy, 61-frame subset (px) | Right image discrepancy, carry (px; 131 detected frames) | Object visible-region mask IoU, all frames | Object visible-region IoU, carry 75–250 |
|---|---:|---:|---:|---:|
| SAM + FoundationPose initialization | 3.987 | 4.154 | 0.501464 | 0.157447 |
| CoCoNet | 5.977 | 6.343 | 0.838676 | 0.794254 |
| Post-optimization | 8.021 | 8.292 | 0.841400 | 0.797481 |

FoundationPose loses the object during carrying and stays near the former table
position. Its internal 360/360 reliability flags and zero fallback count do not
detect that tracking failure. CoCoNet follows the carried object substantially
better, while right-hand image agreement worsens. Its local hand coefficients,
scale coefficients and face coefficients are exactly unchanged from initialization;
body/global/shape changes can still move the hand in the image. The predicted
right-hand relative motion P95 falls from 2.060 to 1.922 px/frame² despite the
worse image fit. This is a direct example of why smoother output alone is
insufficient evidence of better reconstruction. The committed post-optimization
output continues this tradeoff: right-hand subset discrepancy rises to 8.021 px,
while predicted relative motion P95 falls further to 1.769 px/frame². Wrist-only
image discrepancy over all detected right-hand frames changes from 4.032 to
8.026 to 9.287 px across initialization, CoCoNet and final output. The final
local hand parameters remain inherited; smoother wrist-relative motion does
not show that the grasp is more accurate. Moreover, the overall motion reduction
does not hold for stable carry: right relative motion P95 is 1.576, 1.718 and
1.725 px/frame² there. Carry image discrepancy rises from 4.154 to 6.343 to
8.292 px over 131 detected carry frames. No left carry observation exists.

The fresh official initialization's right-hand subset discrepancy, 3.987 px, is
similar to v3's 4.010 px. It also lacks the old branch's frame-110 palm reversal:
its centered right-hand relative second difference is 0.410 px/frame². Camera,
mask prompt, initialization path and identity policy differ, so these results
do not isolate a single cause or establish v3 as an overall winner. A controlled
shared-identity refit of this new initialization is a useful next comparison.

CoCoNet also has a maximum symmetry-unreduced object rotation step of 134.69
degrees, versus 26.61 for FoundationPose. Cube-like geometry can make different
orientations look similar; these raw steps are not reference angular errors.
They still warn against assuming a stable material-local grasp anchor without
resolving the pose branch. The maximum conditional translation step changes
from 12.10 to 5.42 cm; post-optimization reduces it to 2.98 cm. The final object
rotations are bitwise unchanged from CoCoNet, so the large symmetry-unreduced
rotation steps persist. The final mask-IoU gain is small, while translation is
more continuous by this input diagnostic. Those values use the same estimated object geometry and
official camera; no cross-camera metric ranking is inferred.

Independent CPU reports and original-frame contact sheets are under
`runs/human-ep16-hand-round2/official-hand-diagnostics/` and
`runs/human-ep16-hand-round2/official-object-diagnostics/`.

## 4. Weak stable-contact experiment: explicit geometry uncertainty

The object is an RGB/mask-derived rounded-cube hypothesis, not a reference mesh.
Rounded-cube powers 4/8/16 and a cylinder were compared against the actual frame-0
mask; the cylinder fits worse. Shape exponent, hidden surfaces, camera and metric
size remain ambiguous. The final sharper-cube hypothesis has 0.21681-metre extent,
anchored to the prior Human depth and assumed 60-degree horizontal field of view.
Another plausible initial shape has 0.24876-metre extent; this difference itself
warns against treating estimated metric scale as measured truth.

The final trajectory is a full-sequence image/mask optimization with training
SIFT/PnP initialization and quadratic temporal terms. It is not a filtered copy
of the original pose curve. Rejected silhouette-only and earlier global fits
are retained. The final maximum translation step is 1.798 cm and rotation step
6.93 degrees. Mean visible-mask IoU is 0.951831. A spatially deduplicated validation
subset of 2,773 image matches has 0.761-pixel mean discrepancy. These diagnostics
are conditional on the hypothesized mesh and were used for selection.

Analytically zero superellipsoid axes needed correction before topology welding:
fractional powers amplified tiny trigonometric residuals. The explicit maximum
geometry change is 1.381 mm and projected change 0.3824 pixels over all frames.
Mean mask IoU changes from 0.951978 to 0.951831; held-out feature projection does
not change. The final closed convex collision mesh has 1,474 vertices and 2,944
faces. The OBJ retains UV seams for rendering. The final texture is rebaked
using the actual triangle-to-UV mapping and only five visibly unoccluded source
frames (0,15,300,330,359). Observed UV texel coverage is **23.13%**, with unknown
texels gray; this is not surface-area coverage. The earlier 45.23%-coverage bake
has UV stretching and some skin contamination because predicted person masks
miss fingers. It is preserved as a failed asset, not used for official input.
The corrected object is in `candidate-v7-texture-corrected/`; its geometry and
contact-ready NPZ are byte-identical to the validated final geometry.

The stable-carry interval is conservatively annotated as frames 75–250 inclusive.
Approach 0–59, pickup 60–74, placement/release 251–280 and released frames 281–359
have zero contact attraction. Only 126/176 carry frames pass the object's
per-frame image-quality gate. Object-local anchors may attract at most two
already-near-surface fingertips per hand; no requirement forces all fingers
onto the guessed shape. The contact experiment starts from v3, with smaller
incremental angle/wrist trust regions and weak contact weights.

The first nominal contact run completed in **80.250 seconds** and was rejected
by the actual native hand-skin penetration guard. Only left thumb/index anchors
were admitted. Anchor discrepancy falls from 70.65 to 44.53 mm, and right-hand
validation image discrepancy falls from 4.010 to 3.617 px. However, maximum right
hand-skin penetration grows from 0.483 to 9.348 mm. Left-hand penetration remains
severe: mean depth over selected hand vertices changes from 20.77 to 17.86 mm,
with maximum depth around 105 mm. All depths are against this uncertain object
hypothesis, not official PEN. A better fingertip proxy or image residual would
have accepted a worse surface interaction; the candidate is therefore **not
promoted**. Preferred Human output remains v3 without object contact.

The **matched no-contact control** completed in 57.719 seconds. Its right-hand
maximum skin depth is 9.503 mm, compared with 9.348 mm for nominal contact and
0.483 mm for v3. The regression also occurs without contact; nominal maximum
depth 9.348 mm versus control 9.503 mm gives no evidence that contact is its main
cause. Contact gives a limited local benefit relative to that matched control:
left mean skin depth falls from 20.811 to
17.862 mm. Maximum left penetration remains around 105 mm and the penetrating
vertex fraction increases. Neither additional fit is promoted over v3.

Contact incremental bounds are 0.20 radians for fingers, 0.03 radians for the
arm chain and 10 mm wrist displacement relative to v3. Cumulative maxima relative
to the original baseline are 0.85 radians, 0.13 radians and 49.94 mm respectively.
They must not be confused with v3's original 0.65/0.10-radian/40-mm bounds.
Direct contact loss/gradients are zero outside admitted reliable carry frames;
temporal optimization can still propagate changes to other frames.

Both **0.8 and 1.2 scale** image-preserving gauge tests abstain because no
fingertips remain consistently near the hypothesized object; no new Human NPZ
is written for either test. All projected object vertices change by less than
0.000012 pixels, yet the minimum per-tip median surface distance becomes roughly
0.65–0.70 metres at 0.8 and 0.90–0.96 metres at 1.2, versus 0.020 metres for the
left hand and 0.071 metres for the right at nominal scale. These
0.8/1.0/1.2 scale tests scale object geometry and camera translation together,
preserving its projection while changing depth relative to the fixed Human.
This range is a chosen sensitivity test, not a calibrated confidence interval.
No improvement is claimed until image agreement, detachment and actual native
hand-skin penetration checks pass. A failed admission gate produces an explicit
abstention record, not a fabricated reconstructed result.

## Viewing and sharing

The main full-length close-up comparison is
`runs/human-ep16-hand-round2/comparison-hands-v3/hands-comparison.mp4`.
The frame-exact preferred Human review is
`runs/human-ep16-hand-round2/coupled-wrist-v3-spatial/review/index.html`.
Figures are under `figures-v3/`; observation and object videos have their own
provenance reports. The portable comparison/report package is
`runs/human-ep16-hand-round2/report/`; the share archive is
`runs/human-ep16-hand-round2/share/human-ep16-hand-round2-review-20261015.zip`.
It includes the complete video comparisons, exact-frame Human viewer, safe
parameter NPZs, source snapshots, geometry figures and diagnostic records.
Official internal NPZs retain per-frame identity/scales and are explicitly
different from the preferred native shared-identity NPZ.

The packaged saved-image viewer was independently checked directly from
`file://` with network access disabled, at desktop and phone widths. Coverage,
frame 110 to 111, playback and the last frame 359 all work without Python, model
weights or a local server. The supplied MP4s also support offline playback.

The loopback preview is local to this computer. To serve this experiment:

```powershell
& .venv/Scripts/python.exe tools/serve_human_review.py --port 8767 --directory runs/human-ep16-hand-round2
```

Open `http://127.0.0.1:8767/` and select the comparison or review. To share,
send the completed portable package or MP4 files; sending a `127.0.0.1` URL does
not give another computer access. Native NPZs contain parameters and require
the matching licensed/native model to regenerate geometry. Weights are excluded
from view-only packages. No external upload or competition submission occurred.

## What this means for the project

Keep native MHR, shared identity/scales, independent hand image evidence and
bounded phase-aware full-sequence fitting as the working Human path. Keep the
original baseline and rejected variants. Improve the occluded left hand and
global body/root motion before claiming overall motion acceptance. Activate
object contact only when the reconstructed relative depth and geometry support
it and sensitivity checks do not invalidate the conclusion.

Reuse the official implementation as a measured HOI prior rather than accepting
every stage unconditionally. Add projected-mask/feature checks to detect object
tracking loss; a visibility-gated re-registration experiment is warranted.
Retain image evidence while refining body/wrist/object motion, and handle
cube-like pose symmetries before locking material-local anchors. The existing
`fit_shared_identity` implementation accepts per-frame native shape/scales and
real target vertices, so exact official-to-native conversion can feed a genuine
shared-identity refit before the finger fitter. This combined route has not yet
been run; its geometry/image changes must be measured rather than assumed.

The useful hypothesis for a competition improvement is evidence-gated, native
finger/wrist/object refinement with release-aware persistent contact, rather
than uniform smoothing. It is an engineering hypothesis tested on one clip,
not a demonstrated novel state of the art or a promise of better leaderboard
rank. Generalization still needs the agreed multi-video Track 1 subset and
eventual organizer evaluation.

## Primary sources and reproducibility

- [Official challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).
- [Pinned official CARI4D source](https://github.com/nvidia-isaac/video_to_data/tree/595ad68d897b986cd154d4575a022411af096eb3/reconstruction/modules/v2d_cari4d).
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body),
  [native MHR](https://github.com/facebookresearch/MHR), and
  [HaMeR](https://github.com/geopavlakos/hamer).
- [MediaPipe hand-landmark documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

Source video SHA-256:
`a37a978ef0f3d0eeaad74f2ed75fb32502d8a1a047587291c14433d052f9eb74`.
Each fit stores source snapshots, input/model hashes, configuration, objective
trace, exact frame IDs, validation policy and timing in `report.json`.

The explicit 18-file test list in `.github/workflows/tests.yml` passed:
**245 passed, 10 optional Torch tests skipped**, 7.81 seconds in the lightweight
project environment. The actual Torch runtime passed **all 41 focused optimization
and contact tests**, with no skips, in 25.68 seconds. Syntax and whitespace checks
passed. Never run bare pytest. Local technical records and
the Chinese status page are maintained; publishing to the existing Claude team
pages remains unavailable because no Claude Artifact tool is exposed.
