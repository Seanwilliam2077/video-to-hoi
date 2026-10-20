# Local SAM 3D Body and MediaPipe batch

Status on 2026-10-21: all inputs received and independently checked. EP16's
complete 360-frame Human result is published and registered as `submitted`,
with a rich preview in the Human catalog. Its hand candidate failed 9 of 11
support-boundary motion screens, so the fixed-body baseline remains selected.
EP11 and the remaining episodes are pending final delivery. This is not a
completed all-30 delivery or a quality acceptance.

The executor is Codex on the owner's Windows workstation. Code is isolated on
the `human` branch; execution uses an immutable source snapshot. The existing
single-episode experiment remains unchanged. A reviewer has not accepted these
new results.

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

## Remaining delivery work

Finish and review the other pilot before continuing the remaining episodes.
Preserve the published EP16 package and immutable record; do not regenerate its
bytes or identifier when adding later episodes. Publish the remaining native
parameters, comparison videos and check receipts, then append their Human
records and exact preview entries. No all-30 output package, Human board
acceptance, verified scene scale, physical contact, official export or competition
score is claimed. The camera and learned human size remain monocular estimates.
No object result or hidden reference is supplied.
