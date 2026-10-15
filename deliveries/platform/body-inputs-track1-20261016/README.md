# All 30 Track 1 videos ready for standalone Body input

Delivery `body-inputs-track1-20261016`, 2026-10-16. All **30 videos / 16,563 original frames** have actual target-human masks, per-episode estimated camera intrinsics, exact source timelines and recorded sampled visual reviews. The complete input gate passed, the selected upstream Body readers loaded every RGB/mask pair and camera, and the archive passed verification after extraction into a separate directory on the producer workstation.

## Download and receive

- [Complete delivery archive](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/body-inputs-track1-20261016/body-inputs-track1-20261016.zip) — 627,464,556 bytes.
- SHA-256: `c5d09ae93d5608ec0d24beebc1f7dabec7fcc60eb5cb1d24b42a3fd743d4f5aa`.
- [Release and verification receipts](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/body-inputs-track1-20261016).
- [Receiving guide](../../../docs/body-input-handoff.md), including exact commands, directory layout and the standalone weight view.

Extract the archive into a fresh directory and follow its `README.md`. The video paths are under `data/v2d/track_1`; each prepared episode is under `body-inputs/episode_XXXXXX`. Each episode contains all masks, observations, intrinsics, timeline, provenance, review records and checksums. Source video and frame-index bytes remain unchanged.

The receiving check must return `all_30_body_inputs_verified: true` before starting the selected Body consumer. The supplied runtime requirements are for CPU input validation. The inference worker separately supplies the pinned Body GPU environment and authorized model weights. This delivery avoids having every teammate repeat segmentation and camera estimation.

## Evidence and provenance

| Evidence | Result |
| --- | --- |
| [Producer gate](checks.json) | All 30 episodes, every source frame and mask verified |
| [Native reader check](native-reader-checks.json) | All 16,563 RGB/mask pairs plus 30 cameras loaded by the pinned upstream readers |
| [Extracted receiver check](receiver-checks.json) | All 30 passed from a separate extracted directory on the same host |
| [Archive verification](package.json) | Every ZIP member and extracted file hash verified |
| [Per-video inventory](inventory.json) | Source/input/review identities, frame counts and camera estimates |
| [Tool checks](tool-validation.json) | Explicit CI allowlist and dashboard checks; independent synthetic fixtures |
| [Release publication](release.json) | Public asset identities and GitHub SHA-256 digests |
| [Anonymous public download](public-download.json) | All four public assets downloaded without credentials and matched their exact hashes and byte counts |

Dataset: NVIDIA Video to Data Challenge Track 1 at `5f68335f3acc802033d1e80728c1633197521de8`, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). No other challenge track or its derivatives were used. Source code commit: `3238f9c6657605e6fac168d91a1f0b12edc889be`; individual producer script and checkpoint hashes are retained in each episode's provenance.

The mask producer is SAM2.1 Hiera Large, using reviewed episode-local target prompts. Camera estimates use the pinned MoGe2 implementation and 13 endpoint-inclusive frames per episode; the median is computed from every estimate. Inputs retain the original 1536 x 1152 grid and full frame sequence. Confidence is unknown where the model does not provide a meaningful calibrated value. QA includes beginning, exact midpoint, end, uniform samples and all automatically flagged frames, with original/overlay target crops. Reviews are explicitly machine-assisted visual inspections, not a claim of human review or exhaustive contour validation.

Executor: Codex on the project Windows workstation. Review: Codex visual review, with actual sample IDs and findings recorded per episode. Any teammate may receive and continue the task; roles remain flexible.

## Every episode

| Episode | Original frames | Human masks | Camera check | Body input gate | Evidence |
| --- | ---: | ---: | --- | --- | --- |
| 00 | 790 | 790 | Passed | Passed | [Input + review](episodes/episode_000000/input.json) |
| 01 | 668 | 668 | Passed | Passed | [Input + review](episodes/episode_000001/input.json) |
| 02 | 866 | 866 | Passed | Passed | [Input + review](episodes/episode_000002/input.json) |
| 03 | 592 | 592 | Passed | Passed | [Input + review](episodes/episode_000003/input.json) |
| 04 | 747 | 747 | Passed | Passed | [Input + review](episodes/episode_000004/input.json) |
| 05 | 668 | 668 | Passed | Passed | [Input + review](episodes/episode_000005/input.json) |
| 06 | 816 | 816 | Passed | Passed | [Input + review](episodes/episode_000006/input.json) |
| 07 | 777 | 777 | Passed | Passed | [Input + review](episodes/episode_000007/input.json) |
| 08 | 634 | 634 | Passed | Passed | [Input + review](episodes/episode_000008/input.json) |
| 09 | 415 | 415 | Passed | Passed | [Input + review](episodes/episode_000009/input.json) |
| 10 | 577 | 577 | Passed | Passed | [Input + review](episodes/episode_000010/input.json) |
| 11 | 877 | 877 | Passed | Passed | [Input + review](episodes/episode_000011/input.json) |
| 12 | 405 | 405 | Passed | Passed | [Input + review](episodes/episode_000012/input.json) |
| 13 | 425 | 425 | Passed | Passed | [Input + review](episodes/episode_000013/input.json) |
| 14 | 442 | 442 | Passed | Passed | [Input + review](episodes/episode_000014/input.json) |
| 15 | 501 | 501 | Passed | Passed | [Input + review](episodes/episode_000015/input.json) |
| 16 | 360 | 360 | Passed | Passed | [Input + review](episodes/episode_000016/input.json) |
| 17 | 419 | 419 | Passed | Passed | [Input + review](episodes/episode_000017/input.json) |
| 18 | 535 | 535 | Passed | Passed | [Input + review](episodes/episode_000018/input.json) |
| 19 | 443 | 443 | Passed | Passed | [Input + review](episodes/episode_000019/input.json) |
| 20 | 549 | 549 | Passed | Passed | [Input + review](episodes/episode_000020/input.json) |
| 21 | 563 | 563 | Passed | Passed | [Input + review](episodes/episode_000021/input.json) |
| 22 | 533 | 533 | Passed | Passed | [Input + review](episodes/episode_000022/input.json) |
| 23 | 552 | 552 | Passed | Passed | [Input + review](episodes/episode_000023/input.json) |
| 24 | 420 | 420 | Passed | Passed | [Input + review](episodes/episode_000024/input.json) |
| 25 | 365 | 365 | Passed | Passed | [Input + review](episodes/episode_000025/input.json) |
| 26 | 399 | 399 | Passed | Passed | [Input + review](episodes/episode_000026/input.json) |
| 27 | 440 | 440 | Passed | Passed | [Input + review](episodes/episode_000027/input.json) |
| 28 | 366 | 366 | Passed | Passed | [Input + review](episodes/episode_000028/input.json) |
| 29 | 419 | 419 | Passed | Passed | [Input + review](episodes/episode_000029/input.json) |

## Acceptance boundary and next action

`body_inputs_verified` and native reader compatibility passed. **Body inference has not run** in this delivery. The camera values are monocular estimates, not ground-truth calibration or verified metric scale. Sampled review establishes no visible localization blocker in the inspected frames; full reconstruction quality must be assessed on actual Body results.

This is the standalone `sam3d_body_masks_camera.v1` input profile. It does not accept full M1 Inputs, object masks/depth/scale, formal CARI4D, native MHR reconstruction or official export. The six-stage progress registry therefore records these partial Inputs deliveries as `submitted`, with full-stage checks left `not_run`; it does not label them accepted reconstructions. Actual second-machine execution remains to be recorded by the receiver.

Next: run the receiving checks and pinned Body consumer on the selected worker, record input/model identities, then verify complete native outputs and overlays. Keep the prepared input package immutable during iteration; corrections produce a new delivery.
