# Local SAM 3D Body and MediaPipe batch

Status on 2026-10-21: inputs received and independently checked; complete EP16
and EP11 pilots are running. This is not a completed all-30 reconstruction
delivery or a declaration of reconstruction quality.

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

## Remaining delivery work

Review the complete pilots before continuing the remaining episodes. Publish
exact native parameters, comparison videos and check receipts at receiver-accessible
URLs, then append immutable Human records to the board. Until then no all-30
output package, Human board acceptance, verified scene scale, physical contact,
official export or competition score is claimed. The camera and learned human
size remain monocular estimates. No object result or hidden reference is supplied.
