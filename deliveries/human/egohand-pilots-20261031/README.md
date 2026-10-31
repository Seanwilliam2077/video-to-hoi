# Controlled EgoHand versus MediaPipe hand pilots

Status on 2026-10-31: complete EP16/EP11 neural inference, four native MHR fits,
fresh native evaluations, 16 fully decoded comparison videos, sampled static
review and two rehashed offline packages have finished. Only these two Track 1
videos are authorized.
No new all-30 run or automatic result promotion is authorized.

**This integration has not demonstrated an improvement over the current bounded
MediaPipe pilot.** Filtering reduces aggregate EgoHand motion spikes, but both
variants retain image-placement problems and conservative transition-screen
failures. They remain research candidates, without replacing selected dashboard
results or establishing an accepted Human/official competition submission.

## Scope and comparison

The comparison preserves the exact 360/877 original source frames, 30 fps,
estimated camera, source-pixel hand boxes, fixed full-body native MHR identity
and skeleton scales. The reference is the bounded MediaPipe candidate completed
on 2026-10-23, not the earlier rejected correction. Original outputs remain
read-only and are bound by hashes.

Both methods use their native image preprocessing. EgoHand uses its trained
256-pixel bilinear affine sampler; existing MediaPipe used a 384-pixel cubic
resize of an integer region. The source bounds match, but the image tensors
are not byte-identical. The official right-hand camera/mirroring convention
is retained. Perspective cropping and pixel-aspect rectification are disabled;
no new detector, box expansion or camera fitting is introduced.

EgoHand receives all original hand boxes, including frames without accepted
MediaPipe landmarks. Its generated geometry is projected through the same K and
semantically mapped from SOMA-25 to 21 fitting targets, excluding extra
metacarpals 5, 10, 15 and 20. This is anatomical correspondence, not equivalence
of standalone hand and full-body decoders. The wrist origins need not coincide.

Targets are admitted only when finite, in front of the camera, inside the source
image and inside the frozen crop. These binary weights indicate geometric
admission, **not visibility, calibrated confidence or ground truth**. The model's
optional crop-presence and uncertainty values are retained as raw diagnostics.
Missing predictions receive zero target weights, without invented observations.

All candidates are fitted into the same original native MHR. Identity, skeleton
scales, root and unselected controls stay fixed; the original bounded arm/wrist
controls maintain wrist coupling. The controlled solve uses 600 iterations,
learning rate 0.001, holdout stride 5, transition collar 8, correction budget
weight 1 and internal safety fraction 0.2. The optimizer mathematics are
unchanged from the MediaPipe pilot. Target values and positive support can differ.

Raw and official adaptively smoothed EgoHand outputs are kept separate. Smoothing
rebuilds geometry from filtered parameters; it is not new image evidence or a
learned long-occlusion infiller. Both target sets use the same downstream solve.

The filter uses the whole clip, including future and stride-5 frames. The native
solve excludes those targets, but this remains within-video development
validation, not unseen-frame accuracy. See the [official filter description](https://nvidia.github.io/IsaacCapture/main/ego_post_processing/egohand.html#adaptive-hand-smoothing)
and [pinned domain/filter audit](evidence/egohand-domain-smoothing-audit01.json).
The public model disclosure does not identify Track 2 content; undisclosed or
inherited pretraining was not completely audited.

## Deployment and source binding

Official source: [NVIDIA Isaac Capture EgoHand](https://nvidia.github.io/IsaacCapture/main/ego_post_processing/egohand.html).
The source is pinned to IsaacCapture
`9dfc9f25fe0b8c69a2f4f0bc01eb669097f7d1b9`.

The isolated local environment reuses existing Torch 2.5.1/cu124 read-only and
adds its dependencies only to the task-local environment. Native Windows is a
local compatibility path; the official documentation targets Ubuntu. The upstream
model constructor's hardcoded CUDA hand wrapper is replaced process-locally by
the same wrapper using CPU/torch geometry. Learned weights and prediction
equations are unchanged. No MANO model is required by this path.

Thirteen selected public checkpoint, generic rig, neutral-geometry and license
files were downloaded and verified. The checkpoint is 10,657,169,753 bytes,
SHA-256 `f940ecb6e3b73d5bd9e562cff232550e1c072f6c063eb4f705951122b15d8a75`.
All 817 saved model-state keys load with no missing learned or unexpected keys.
The checkpoint's camera mode is `sam3db_perspective` and hand identity model is
`mhr`; this standalone hand output does not replace the full native MHR export.

The first source checkout accidentally materialized a small set of bundled
upstream example files. They were isolated unread, the attempt was recorded,
and subsequent source access used an exact code/configuration allowlist. None
of those examples or calibrations was used as an experiment input. No Track 2
data, derivatives, scores or statistics were used.

At launch, actual free CUDA memory was approximately 345 MiB, so the experiment
uses CPU without terminating other applications. A fixed-input throughput probe
selected 12 Torch threads, batch size 8 and eight source frames per decode chunk.
Its maximum joint difference from the initial batch-one smoke was below
0.000001 m. This was a throughput choice, not selection of a favourable pose.

## Verification and interpretation

The explicit workflow Python test list passed **948 tests, 25 skipped**. The
Node workflow passed **44 tests**. Read-only reuse of the existing Torch runtime
and existing development pytest passed **128 focused tests**. An initial focused
invocation lacked pytest; its failed log is preserved, and the successful check
reused the existing package without modifying the inference environment.

The original registry and catalog checks passed for 30 episodes / 16,563 frames,
without replacing their selected Human results. Software tests establish
interfaces and mathematical behavior, not real-video quality.

The real 11-frame smoke completed on both hands with raw and smoothed geometry.
Independent source-camera projection agreed with inversion of upstream saved
crop projections within 0.0000472 pixels. Wrist-local joint zero was exactly
zero; no double translation or frame-mapping defect was found. Static review
still found visible placement discrepancies. These must not be hidden by
successful import, finite arrays or small projection round-trip error.

The saved `pred_keypoints_2d` is a projection of generated 3D geometry; it is not
the auxiliary direct 2D head. The current export cannot distinguish an auxiliary
2D-head failure from a 3D/camera-head failure. Fully hidden finger correctness
cannot be judged from that consistency check.

The full comparison will independently decode every saved MHR frame, retain the
existing adjacent-edge/wrist thresholds, check original and new support
boundaries, and report raw/smoothed specialist proposals separately. MediaPipe
agreement is method-dependent and can favour its own fitted candidate; EgoHand
target agreement is self-consistency. Neither is ground truth. The original
holdout has already influenced development and remains development validation.

Visible source review examines grasp closure, release, wrist attachment and
motion timing as well as jitter. Opening-distance proxies are descriptive, not
true contact or calibrated joint-angle labels. No object-depth compositing,
physical grasp acceptance or official challenge score is implied by an overlay.

## Execution evidence and final results

The executor is Codex on the owner's Windows workstation, with separate runtime,
adapter and independent evaluation contributors recorded in evidence. All 1,237
original frames have both raw and officially filtered left/right predictions.
All four native MHR candidates passed all-frame decoder parity and preserve
shape[45], scales[68], root and unselected controls. Arm/wrist limits are reached
in both clips; EP11 raw finger correction also reaches its 0.65-radian cap.
These results describe bounded retargeting, not unconstrained specialist quality.

Prediction motion below is the P95 of pooled 20-finger wrist-relative second
difference norms in source pixels/frame squared at 30 fps. It includes genuine
motion and palm rotation; it is not official ACC-H or reference-relative error.
Freezing the hand could lower this quantity without improving reconstruction.

| Video / hand | Current MP | EgoHand raw | EgoHand filtered |
| --- | ---: | ---: | ---: |
| EP16 left | 3.787 | 7.357 | 5.864 |
| EP16 right | 6.033 | 11.558 | 9.017 |
| EP11 left | 5.483 | 9.630 | 7.129 |
| EP11 right | 5.678 | 9.998 | 6.692 |

The following MP-referenced added-correction checks retain the original
tolerances: observed velocity discrepancy at most baseline x 1.05 + 0.25 pixels;
unknown-motion correction steps at most 1 pixel and 0.005 native estimated
metres. They are conservative engineering diagnostics. "Unknown" means
insufficient accepted MP evidence, not an occlusion annotation. The unchanged
baseline trivially adds no correction. An edge identifies one anatomical hand
between consecutive original frames.

| MP-referenced finger screen | Current MP failures | EgoHand raw | EgoHand filtered |
| --- | ---: | ---: | ---: |
| EP16, all 718 adjacent hand-edges | 0 | 262 | 180 |
| EP16, 11 original detector-support boundaries | 0 | 5 | 5 |
| EP11, all 1,752 adjacent hand-edges | 0 | 615 | 461 |
| EP11, 35 original detector-support boundaries | 0 | 21 | 14 |

EP16 has two sufficiently MP-observed original boundary edges and nine unknown
ones; EP11 has two and 33. Absolute wrist screens retain failures too. Generated
EgoHand-target agreement is recorded separately as self-consistency: EP16 can
pass every such edge while failing the MP-referenced screen. EP11 filtered-target
wrist checks also have four failures. Neither screen establishes hidden truth.

Weighted source-pixel agreement on the original stride-5 development frames:

| Video / hand | MP-supported frames | Current MP | EgoHand raw | EgoHand filtered |
| --- | ---: | ---: | ---: | ---: |
| EP16 left | 5 | 5.256 | 5.501 | 6.168 |
| EP16 right | 64 | 4.302 | 9.298 | 9.177 |
| EP11 left | 71 | 6.221 | 8.083 | 7.946 |
| EP11 right | 78 | 5.724 | 8.307 | 8.157 |

The fallible MP reference can favour its own fitted candidate; these are not
ground-truth accuracy estimates. Five EP16 left-hand frames do not establish
hidden grasp accuracy. Opening curves retain nonzero variation, which alone
does not establish correct grasp/release. Nearly collapsed EP11 MP palm
projections make some 2D opening ratios unstable; they are not true angle/contact
errors. Improvement is not uniform across windows: the filtered EP11 late
680-850 window has slightly lower native 3D second differences than MP, while
its left-hand 511-512 edge is worse. Aggregate results must not imply every
interval regresses.

| CPU stage | EP16 | EP11 |
| --- | ---: | ---: |
| Predictor.run wall, including crop/output work | 802.311 s | 1,793.776 s |
| Official filter and geometry rebuild | 26.375 s | 47.735 s |
| Inference runner total, including load/export | 859.625 s | 1,877.656 s |
| Native raw fit and geometry export | 249.829 s | 342.234 s |
| Native filtered fit and geometry export | 255.359 s | 344.063 s |

Variants were fitted in parallel. These are stage durations, not an isolated
kernel benchmark or total deployment elapsed time. Peak inference process
working set is about 8.67 GiB. CPU torch versus Warp hand geometry agrees
exactly at checked joints and within 0.000000202 m at vertices. Full inference
projection round trips are below 0.000103 pixels. These establish implementation
consistency, not real geometry accuracy.

## Outputs, reproduction and next action

Task receipts and results are stored under:

`J:/video-to-hoi/work/egohand-pilots-20261031/`

`J:/video-to-hoi/runs/egohand-pilots-20261031/`

Each episode opens at `review-raw01/index.html`. The top row shows the anatomical
left hand; the bottom row shows the right. Original frame IDs are printed.

- `raw-smoothed-mhr-comparison.mp4`: RGB / current bounded MP / raw EgoHand MHR /
  officially filtered EgoHand MHR.
- `hands-comparison.mp4`: RGB / original SAM / current bounded MP / raw EgoHand MHR.
- `proposal-comparison.mp4`: RGB / MP image detections / raw generated geometry /
  filtered generated geometry. Admission counts do not establish visibility.
- Critical clips include every source frame at 10 fps, one-third speed. EP16
  windows are 40-110 and 240-308; EP11 windows are 280-340, 480-620 and 680-850.
  These are review windows, not labelled true phases.

The [compact results](summary.json), [package identities](local-review-packages.json)
and [evidence inventory](evidence-inventory.json) bind the output bytes.
Large videos, specialist geometry and native parameter arrays stay outside Git.
Self-contained local ZIPs open at `index.html` after extraction; no model server
is required to watch. Native parameter decoding still requires the exact allowed
MHR model. No new external video publication or catalog promotion is performed.

The [archived execution recipes](reproduction/README.md) preserve actual source,
commands, dependency/asset/license identities and native snapshot bytes. They
are not an installed production backend or verified portable native importer.
Large weights/rigs/models and original videos remain separate dependencies;
machine-local paths require hash-preserving relocation and fresh attempts.

All 16 comparison videos passed full decoding. Every member of both offline
archives was rehashed: EP16 is 72,855,848 bytes; EP11 is 204,690,579 bytes.
The [combined integrity receipt](evidence/completed-integrity-final01.json)
reverifies all 14 postprocessing stages and unchanged original inputs/results.
The [preprocessing audit](evidence/input-preprocessing-audit-final01.json)
checks BGR-to-RGB conversion, pixel-unit crop transforms and normalization
against the pinned official inference implementation; no simple integration
error was found by these checks.

Sampled static image review is recorded separately for [EP16](evidence/ep16-manual-visual-review01.json)
and [EP11](evidence/ep11-manual-visual-review01.json), with independent reviews
beside them. EP11 frames 511/512 show visibly displaced/enlarged generated
proposals; frame 803 shows a visible blurred left hand despite missing admitted
MP targets. No consistent sampled visual advantage was established. Exact true
grasp/release timing, fully hidden finger correctness, physical contact, metric
scale and official export acceptance remain unestablished. No official
challenge scores or continuous human playback claims are made.

Next, test visibility-aware fusion of image evidence and specialist proposals,
verifying wrist alignment before fitting finger articulation. Check reappearance
frames and opening/closure timing explicitly. Object-relative contact requires
separately verified Track 1 object geometry, motion and phase evidence.
Third-person small/blurred hands and estimated camera parameters are plausible
input/domain confounds, not proven sole causes. Stronger smoothing alone does
not establish a correct hidden grasp.

The adapter and evaluator implementation is committed on `human` as
[`76745e4`](https://github.com/Seanwilliam2077/video-to-hoi/commit/76745e4).
The preexisting unrelated `docs/status.html` edit remains preserved. Live Claude
module/status publishing has not occurred; the required Artifact tool is not
available in this session.
