# Local SAM 3D Body and MediaPipe batch

Status on 2026-10-21: all 30 videos and all 16,563 original frames have completed
Human reconstruction, preview generation and native checks. All 30 portable
packages are published and registered as `submitted`, with exact-record Human
previews in the delivery board. This completes this Human-only review delivery;
reconstruction quality, scene scale, physical contact and official export remain
unaccepted.

The selected result is the fixed-body baseline in **all 30 videos**. Every
MediaPipe hand candidate failed its separate support-boundary motion screen,
despite better held-out image agreement in many clips. Both versions remain
available for comparison. This run has not solved hand twitching.

The executor is Codex on the owner's Windows workstation. Code is isolated on
the `human` branch; execution uses an immutable source snapshot. The existing
single-episode experiment remains unchanged. A reviewer has not accepted these
new results.

## Results and how to review

- [Published packages, videos and evidence](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/sam-mediapipe-20261021).
- [Exact 30-entry Human preview catalog](../../../tracking/human-catalog.json)
  and [submitted delivery records](../../../tracking/records).
- [Completion receipt](all30-completion.json),
  [pilot publication](pilot-publication.json) and
  [remaining-28 publication](remaining-publication.json).
- [Full numerical analysis](review/all30-analysis-summary.json),
  [sampled visual-review index](review/all30-visual-review-index.json),
  [browser review](review/final-review-summary.json), and
  [mask/camera implementation audit](review/mask-camera-semantics-audit-20261021.json).
- [Independent final delivery audit](review/final-delivery-audit.json) and
  [byte-preserved review inventory](review/inventory.json).

Each archive opens at `index.html` after extraction and includes complete body
and hand videos, frame viewers, checks, observations and native parameters.
`parameters/selectedhuman.npz` is a byte-identical copy of the selected baseline;
`parameters/refined.npz` is the rejected hand candidate. Native arrays are
`pose[frames,136]`, one `scales[68]` vector and one `shape[45]` vector per clip.
Identity/scales are fixed within a clip; no shared identity across all videos
was fitted. Offline viewing needs no model server. Regenerating 3D geometry
requires the exact permitted MHR model identified in the archive.

The publisher verified 720 public files: 360 result assets and their 360 checksum
sidecars, including 30 archives and 90 comparison videos. Already published EP11
and EP16 bytes were preserved. The main branch board awaits normal review/merge
and deployment; the local preview at `http://127.0.0.1:8769/` has all 30 results.

## Findings and limits

| Check or observation | Result | What it establishes |
|---|---|---|
| Original-frame coverage | 30/30 clips, 16,563/16,563 frames | Complete Human output mapping |
| Native representation | Both variants finite, fixed per-clip identity/scales | Parameter and saved-geometry contract |
| Recorded decoder consistency | Maximum difference 0.000001431 m | Numerical consistency, not true 3D accuracy |
| Selected candidate | 30 baseline, 0 refined | Conservative delivery choice |
| Hand transition screens | 544/587 failed; all 30 candidates rejected | Added motion at support boundaries remains a problem |
| Accepted independent hand observations | Left 50.18%; right 44.12% of frames | Detector support after reliability gates, not measured visibility |
| Visual inspection | 291 unique sampled source frames across all 30 clips | No gross person/camera/axis/render failure found in these samples |

The rejected hand candidate improved held-out image error strictly in 26 left-hand
and 25 right-hand clips. Across the 29 clips with nonzero held-out observations
per side, the median of per-clip mean errors changed from 7.10 to 5.90 pixels
on the left and 6.79 to 5.63 pixels on the right. These are **rejected-candidate**
image diagnostics, not measurements of the selected result, hidden-reference
errors, pooled landmark errors or competition scores. Empty-observation zero
placeholders were excluded. Image agreement alone missed transition artifacts.

The visual index includes 270 baseline body sample pairs, two historical pilot
extras and 44 unique hand sample pairs, with overlap across modalities. The 291
unique sampled source frames do not mean every frame was watched continuously.
The review found unresolved finger/grip mismatches when hands were occluded or
independent detections were missing. Mesh rendering through a table without
object-depth composition does not establish physical penetration or correct depth.

The [implementation audit](review/mask-camera-semantics-audit-20261021.json)
confirms received masks select/crop the body and enter learned body-mask
conditioning. Separate hand reruns use RGB crops and camera parameters without
external mask prompts. Received estimated intrinsics override temporary defaults.
An upstream final SAM70 2D-projection overwrite introduces about +0.5 pixels on
both axes in the audited pilots; native 3D geometry and downstream full-camera
review are unaffected. The constructor fallback messages do not mean the supplied
body masks or camera were ignored.

Camera parameters are Track 1-derived estimates, not measured calibration.
Monocular human depth/size, shared human-object scale, required reference-frame
alignment, contact, penetration, object motion and official export remain
unverified or outside this Human-only delivery. HaMeR, HandFlow and full official
CARI4D were not run in this batch.

## Measured runtime and validation

The source-bound all-30 job ran from 2026-10-21 03:14:54 to 22:40:56 Shanghai time:
69,960.672 seconds, or 19 hours 26 minutes. It reused the two completed pilots;
their current-stage receipts sum to another 1.569 hours and are not added to the
main-job wall time. Historical experiments and rejected attempts are also
excluded. Main-job time includes observed GPU resource contention. Packaging,
publication and the Human branch push completed at 23:46 Shanghai time.

Execution used the owner's Windows workstation and RTX 3080: GPU SAM/native body
fitting and CPU MediaPipe/finger refinement. Source is fixed at
`21982effde7da5ba429aa618283ffb9f9bcc0277`, with immutable runtime02/05/06 stage
snapshots and exact model/environment receipts. Later commits register evidence.

Validation: the explicit workflow Python list passed 898 tests with 17 skips;
34 actual Torch hand tests passed; 44 board Node checks passed. Final registry,
both catalogs, exact archive/check/native bindings and publication checks passed.
Actual Edge loaded, sought and played all three public review videos for EP00,
EP24 and EP29 without JavaScript/media errors, in addition to the two pilots.
These checks establish execution and delivery, not reconstruction acceptance.

## Next optimization work

1. Improve side assignment, stable hand crops and occlusion-aware observations;
   evaluate a permitted specialist hand estimator once its dependencies are available.
2. Replace abrupt observation-support transitions with confidence-aware sequence
   fitting and explicit uncertainty. Preserve observed grasp changes; do not
   flatten motion merely to lower prediction-only acceleration.
3. Evaluate wrist/finger correspondence, identity across clips where justified,
   and human/object camera and scale consistency before physical refinement.
4. Add object-aware grasp/release contact, penetration and floating checks, then
   compare official CARI4D initialization, CoCoNet and post-optimization stages
   using only our own permitted Track 1 assets and verified parameter conversion.

The repository technical record is current. Live Claude Human module/status
publication remains pending because its required Artifact editing/publishing
capability is unavailable here. The existing unrelated local status-page edit
was preserved. No main merge, deployment or result-quality acceptance was performed.

## Verified input receipt

- The received archive is 627,464,556 bytes, SHA256
  `c5d09ae93d5608ec0d24beebc1f7dabec7fcc60eb5cb1d24b42a3fd743d4f5aa`.
- [Receiver summary](receiver/receiver-summary-20261021.json): 30 original videos,
  16,563 original frames, complete file inventory, no missing or extra files.
- [Handoff gate](receiver/receiver-checks-20261021-attempt01.json): passed all 30.
- [Pinned native readers](receiver/receiver-native-reader-checks-20261021-attempt01.json):
  passed every RGB/mask pair and camera matrix at upstream revision
  `33129dd0f2d2dcfd1164d43fd076542660756ed2`.
- [Receiver runtime](receiver/receiver-runtime-record.json): execution environment
  and measured gate times. Input checks do not certify model inference or accuracy.

## Pilot and execution contract

EP16 has 360 frames; EP11 has 877, the longest episode, with walking, turning and
occlusion. Each frame consumes the delivered human mask and estimated camera.
EP16's new focal length is approximately 818.746 pixels, so the earlier assumed
camera result is not rebound to this input package.

The route is real SAM full-body-and-hands inference, shared native MHR identity
and skeleton fitting, independent MediaPipe observations, bounded finger/wrist
refinement, complete body and hand comparison videos, and native decoding checks.
Missing independent observations remain uncertain. Both baseline and hand
candidate are retained; candidate selection is a diagnostic and does not grant
quality acceptance. See [the execution guide](../../../docs/human-batch.md) and
[portable packaging guide](../../../docs/human-delivery-packaging.md).

A real three-frame GPU smoke passed. Warm raw inference was approximately
1.4 seconds per frame. A GPU native-fit chunk comparison took 18.328 seconds
with batch 8 and 77.000 seconds with batch 1; maximum geometry difference was
0.04323 mm. These are short smoke measurements, not all-episode throughput or
accuracy against hidden references. The executed native model uses float32
with a float64 solver. No model-math precision change is inferred from chunking.

## Hand transition review

[Independent EP16 review](pilot-review/ep16-initial-review.json) found extra
finger motion when independent observations resumed or disappeared. At original
frames 285 to 286, mean wrist-relative image motion was 1.30 pixels in the
baseline and 8.67 pixels in the initial hand candidate. The 17 common independent
finger observations moved about 1.51 pixels. These are image diagnostics, not
hidden-reference errors.

[Two correction-continuity probes](pilot-review/ep16-transition-probes.json)
reduced that onset jump but still failed 9 of 11 boundary screens. Held-out image
agreement alone passed, so it cannot authorize the candidate. The final runner
checks every fitted support boundary, keeps both candidates and falls back to
the native baseline when needed. The occluded left grasp still lacks independent
observations; hand twitching and physical contact have not been solved.

## Published EP16 result

The [EP16 portable review and native parameters](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/sam-mediapipe-20261021/sam-mediapipe-20261021-ep16.zip)
contain every original frame, both candidates and their checks. The archive is
150,520,145 bytes, SHA256
`be81e717f6fd80b4d9b2fbf79ff854e2bb14c68430e6108fb1de4dae7f45df75`.
Source code is pinned to
[`21982effde7da5ba429aa618283ffb9f9bcc0277`](https://github.com/Seanwilliam2077/video-to-hoi/tree/21982effde7da5ba429aa618283ffb9f9bcc0277).

- [Selected baseline: full-body comparison](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/sam-mediapipe-20261021/sam-mediapipe-20261021-ep16-baseline-comparison.mp4).
- [Unselected hand candidate: full-body comparison](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/sam-mediapipe-20261021/sam-mediapipe-20261021-ep16-refined-comparison.mp4).
- [Synchronized hand close-ups](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/sam-mediapipe-20261021/sam-mediapipe-20261021-ep16-hands-comparison.mp4).
- [Native and hand checks](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/sam-mediapipe-20261021/sam-mediapipe-20261021-ep16-checks.json),
  [scale limitations](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/sam-mediapipe-20261021/sam-mediapipe-20261021-ep16-depth-scale.json),
  and [publication/result summary](pilot-review/ep16-published-result.json).

The publisher anonymously downloaded and verified all 12 result assets and
their 12 checksum sidecars. The immutable publication attempt receipt has
SHA256 `e8529ae6f6f11e6f12be1b4506504228c1321f791e52bdf9679979aec6f86adc`.
The exact submitted registry record and Human preview entry are linked by the
same record ID, `sam-mediapipe-20261021-ep16`; no old camera result is reused.
The registry and catalog update is on the `human` branch. The deployed main
board changes only after a reviewed merge and Pages publication.

Native decoding checks passed for both candidates. Held-out image agreement
improved from 5.76 to 5.53 pixels over five left-hand frames and 6.79 to 4.95
pixels over 64 right-hand frames, but nine support boundaries still violated
the separate motion screen. Therefore the hand candidate is retained for
inspection and is not the selected native trajectory. Passing a native check
or an image screen is not reconstruction-quality or true-contact acceptance.

## Published EP11 result

The [EP11 portable review and native parameters](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/sam-mediapipe-20261021/sam-mediapipe-20261021-ep11.zip)
contain all 877 original frames and both candidates. The archive is 327,819,561
bytes, SHA256
`929c357bb697d6770a8ccbca965c740bbeeb0a4cc3f4d03eb360ea60c09a04a8`.
It uses the same pinned source revision as EP16.

- [Selected baseline: full-body comparison](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/sam-mediapipe-20261021/sam-mediapipe-20261021-ep11-baseline-comparison.mp4).
- [Unselected hand candidate: full-body comparison](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/sam-mediapipe-20261021/sam-mediapipe-20261021-ep11-refined-comparison.mp4).
- [Synchronized hand close-ups](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/sam-mediapipe-20261021/sam-mediapipe-20261021-ep11-hands-comparison.mp4).
- [Native and hand checks](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/sam-mediapipe-20261021/sam-mediapipe-20261021-ep11-checks.json),
  [scale limitations](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/sam-mediapipe-20261021/sam-mediapipe-20261021-ep11-depth-scale.json),
  and [publication/result summary](pilot-review/ep11-published-result.json).

All 12 result assets and their 12 checksum sidecars were independently downloaded
and verified. The immutable publication receipt has SHA256
`024984f594a6c8ffb609192fea5d8b4236c8f0386669f7d38fc8311ee3d75ae2`.
The complete hand movie was decoded, and selected original frames were visually
reviewed. The reviewer did not watch every frame continuously or independently
rerun the native decoder. Local browser checks loaded, sought and played all
three actual Release videos for each pilot without JavaScript errors; their
receipt identities are included in the operational summary below.

The hand candidate improved held-out image agreement from 7.36 to 5.90 pixels
over 71 left-hand frames and 7.48 to 5.18 pixels over 78 right-hand frames.
However, 33 of 35 support boundaries failed the independent motion screen.
The exact native baseline remains selected, and the hand candidate is available
for inspection. Occluded hand depth, contact, penetration and release timing
remain unverified.

## Pilot gate provenance

Both pilots passed the operational coverage, native-decoding and sampled visual
gate before the remaining episodes ran. The historical
[operational start and runtime summary](pilot-review/all30-operational-start.json)
binds that gate, actual batch launch, planned queue and the original 19.59-hour
forecast. The completed run and exact published results are recorded above and
in the final completion/publication receipts. Historical forecasts are not current
delivery status or quality acceptance.
