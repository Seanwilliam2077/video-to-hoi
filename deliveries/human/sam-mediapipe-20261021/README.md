# Local SAM 3D Body and MediaPipe batch

Status on 2026-10-21: all inputs received and independently checked. EP16's
complete 360-frame result and EP11's complete 877-frame result are published
and registered as `submitted`, with exact-record Human previews. The hand
candidates failed 9 of 11 and 33 of 35 support-boundary motion screens,
respectively, so each fixed-body baseline remains selected. The remaining
28 episodes are queued and the source-bound batch has started. These planned
declarations contain no completed outputs. This is not a completed all-30
delivery or a quality acceptance.

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

## Remaining delivery work

Both pilots passed the operational coverage, native-decoding and sampled
visual review gate. The [operational start and runtime summary](pilot-review/all30-operational-start.json)
binds the final gate, actual batch launch, planned queue, and measured two-pilot
forecast by receipt hashes. The remaining 15,326 frames have a 19.59-hour compute
estimate, with an empirical pilot envelope of 18.74 to 20.62 hours and a planning
allowance of 19 to 26 hours. This estimate includes complete preview generation
and native checks; packaging, upload, manual review and interruptions are excluded.
It is a planning estimate, not a completion promise.

Preserve both published pilot packages and immutable records; do not regenerate
their bytes or identifiers when adding later episodes. Publish the remaining native
parameters, comparison videos and check receipts, then append their Human
records and exact preview entries. No all-30 output package, Human board
acceptance, verified scene scale, physical contact, official export or competition
score is claimed. The camera and learned human size remain monocular estimates.
No object result or hidden reference is supplied.
