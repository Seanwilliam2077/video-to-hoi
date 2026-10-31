# Image-supported EgoHand crop and native-depth pilots

Status on 2026-10-31: both complete native fits, numerical evaluations,
synchronized comparison videos and verified local review packages are finished.
The numerical outcome is partial improvement over the prior EgoHand route,
without an overall win over matched MediaPipe. Only Track 1 EP16/EP11 are
authorized. No new all-30 run, result promotion or official scoring is authorized.

## Scope and implementation

The executor is Codex on the owner's Windows workstation, with separate runtime,
adapter and independent evaluation contributors recorded in the evidence.
Implementation is on `human` at
[`28deb40`](https://github.com/Seanwilliam2077/video-to-hoi/commit/28deb40).
Related deliveries are the [first EgoHand pilots](../egohand-pilots-20261031/README.md)
and [bounded MediaPipe pilots](../hand-continuity-pilots-20261023/README.md).
Original SAM, MediaPipe and EgoHand artifacts remain read-only.

All 360/877 original frames, anatomical left/right identities, estimated source
K, fixed per-clip native MHR shape/scales and original inspection windows are
retained. Same-new-crop MediaPipe is rerun as the controlled comparison. The
preserved MediaPipe and old filtered EgoHand results provide historical context.
Neither MediaPipe's detector evidence nor generated EgoHand geometry is ground
truth. MP-dependent crop selection, wrist anchoring and target admission are
shared inputs, not independent validation.

The main changes are:

1. Training-image-supported tight crops with 1.5 padding, 64-pixel floor,
   208-pixel cap, robust crop-state filtering and short crop-only interpolation.
   The tight-only trial lost detections, including a source-visible hand at
   EP11 frame 511. It is preserved as trial01. Final routing keeps the exact
   original wide crop unless a tight crop has reliable wrist/eight-landmark
   support without reduced finger count for three consecutive training slots.
   A new mixed left/right duplicate pair widens both crops. Heldout routing uses
   only adjacent training routes. No missing image target is interpolated.
2. Standalone EgoHand palm breadth is close to native MHR, but its camera-depth
   estimates are much nearer in the audited frames. Retarget wrist-local 3D
   geometry at the original body's wrist depth; use only supported training MP
   wrists for lateral placement. A single raw-reference/native palm scalar is
   fixed per hand and clip. No per-frame 2D shrinking, clamping or copied MP
   finger coordinates is introduced.
3. Require accepted image support for each generated joint, plus wrist and at
   least eight of the twenty non-wrist hand landmarks. Unknown/stride-5 shared anchor/gate weights are zero;
   continuous prior geometry is retained without invented image evidence.
   These gates are heuristics, not calibrated visibility or confidence models.

Crop anchors, routing, scale calibration and lateral wrist/gate construction
exclude stride-5 detector coordinates/weights before any decision. Original
images, body initialization and whole-clip EgoHand filtering remain
transductive; this is within-video development, not unseen-frame generalization.

## Frozen conditions and inputs

The native solve reproduces the complete previous MP configuration: CPU1,
600 iterations, learning rate 0.001, original stride-5 mask, transition collar8,
correction budget1, safety fraction0.2, original 0.1-radian arm and 0.04-native-metre
wrist bounds. Native shape[45], scales[68], root and all unselected controls are
fixed. No contact anchors, object constraints, phase labels or camera refit are
added. Both methods use identical revised source bounds, preserving their
trained samplers (MP384 cubic versus EgoHand256 bilinear).

Only the two original Track 1 videos and their own permitted estimates/metadata
are used. Generic pretrained EgoHand/SOMA-X/MediaPipe assets are reused from the
previous verified deployment; no new challenge dataset or example is downloaded.
IsaacCapture stays pinned to `9dfc9f25fe0b8c69a2f4f0bc01eb669097f7d1b9`, with
the same EgoHand checkpoint SHA
`f940ecb6e3b73d5bd9e562cff232550e1c072f6c063eb4f705951122b15d8a75`.

## Actual observer and cache checks

Supported hand frames below require an admitted wrist and at least eight admitted
non-wrist hand landmarks out of twenty on that hand.
They measure detector support, not true visibility or pose correctness.

| Video / hand | Original wide | Tight-only trial | Final routed |
| --- | ---: | ---: | ---: |
| EP16 left | 27 | 10 | 27 |
| EP16 right | 310 | 205 | 312 |
| EP11 left | 344 | 256 | 346 |
| EP11 right | 384 | 277 | 392 |

Final routing retains all originally supported training hand frames. EP16 right
frame315, a development-heldout frame, loses detector support; no other original
supported hand frame is lost. This exception is retained in the audit. Tight
routes are 0/78 frames per EP16 hand and111/125 per EP11 hand; all other input
boxes remain exactly the original wide boxes.

Exact original-frame RAW EgoHand caches are reused only when source video,
frame, side, camera, pixel crop, checkpoint, implementation, dtype and sampler
match. There are2,160 cached hand views and314 newly inferred views across
1,237 source frames. Old filtered results are never reused: whole-clip official
filtering and geometry rebuilding run again on the complete merged raw clip.
The cache contract restores only omitted source K/side/ID metadata, without
generating geometry or interpolating predictions.

The geometry-only cache proof forbids neural forward calls. Restoring all EP16
RAW results and repeating the official filter reproduces both old filtered
NPZs byte-for-byte. Ten exact-input CPU golden views agree within0.000001m and
0.001px. This proves the cache path's numerical/input consistency, not accuracy.

A task-owned GPU probe did not reach a first forward before the runtime health
limit. It was stopped with PID ownership verified, leaving other applications
untouched; no GPU parity or speed result is claimed. Final inference is CPU
float32/12 threads/batch8, with the existing CPU geometry compatibility path.

The explicit final workflow Python list passed **1015 tests, 25 skipped**; Node
passed **44 tests**. The existing Torch runtime passed **189 focused tests**.
Registry/catalog checks passed without changing selected all-30 results.
These checks establish software interfaces and mathematics, not real-video
quality or competition acceptance.

## Results and review

The final evaluations cover all **360 EP16 / 877 EP11 original frames** and all
five native variants. Fresh decoder parity, fixed identity/scales/root and
unselected-parameter checks pass. These are software and geometry consistency
checks, not reconstruction acceptance. Exact receipts and executed recipes are
kept under `work/egohand-refined-pilots-20261031`; large outputs are under
`runs/egohand-refined-pilots-20261031`. No large models/videos are committed to Git.

The primary video columns are RGB / preserved bounded MP / final same-crop MP /
refined EgoHand MHR. A separate old/new EgoHand ablation and raw/filtered/generated
proposal diagnostics retain uncertainty. All methods share one display region
per frame; input crops are a separate saved artifact. Top/bottom rows are
anatomical left/right. Full clips retain 30fps; original critical windows replay
every included source frame at 10fps, without interpolation. Green proposal
points are admitted generated targets. Orange points are unadmitted generated
estimates, including protocol-heldout and missing/gated support: orange is not
an occlusion label or proof that a point was an explicit fitted prior.

The following engineering checks use the **fresh same-crop MP reference** for
all three compared outputs. Counts are failed adjacent hand-frame edges, pooled
across both hands. Thresholds are unchanged from the earlier evaluator. The
reference is method-dependent and shares the refined route's wrist/support
inputs. A zero count is not evidence of accurate hidden motion.

| Video / output | Supported finger motion failures | Unsupported finger correction-budget failures | Supported wrist failures | Unsupported wrist correction-budget failures |
| --- | ---: | ---: | ---: | ---: |
| EP16 prior EgoHand | 58 / 314 | 121 / 404 | 85 / 321 | 133 / 397 |
| EP16 refined EgoHand | 68 / 314 | 0 / 404 | 1 / 321 | 0 / 397 |
| EP16 matched MP | 0 / 314 | 0 / 404 | 0 / 321 | 0 / 397 |
| EP11 prior EgoHand | 102 / 669 | 353 / 1083 | 137 / 670 | 331 / 1082 |
| EP11 refined EgoHand | 110 / 669 | 0 / 1083 | 1 / 670 | 0 / 1082 |
| EP11 matched MP | 0 / 669 | 0 / 1083 | 0 / 670 | 0 / 1082 |

The preserved original MP reference gives a separate sensitivity check. It must
not be mixed with the fresh-reference denominators above:

| Video / output | Supported finger failures | Unsupported finger failures | Supported wrist failures | Unsupported wrist failures |
| --- | ---: | ---: | ---: | ---: |
| EP16 prior EgoHand | 59 / 312 | 121 / 406 | 87 / 321 | 133 / 397 |
| EP16 refined EgoHand | 69 / 312 | 0 / 406 | 7 / 321 | 0 / 397 |
| EP11 prior EgoHand | 104 / 655 | 357 / 1097 | 137 / 656 | 335 / 1096 |
| EP11 refined EgoHand | 111 / 655 | 9 / 1097 | 26 / 656 | 8 / 1096 |

Thus EP11 still has nine unsupported finger and eight unsupported wrist failures
against the original detector mask. New image support changes classification;
the fresh-reference zero is not an unconditional unknown-motion pass. Fresh MP
also fails some checks against the original mask (EP16:4 supported finger/6 wrist;
EP11:26 supported finger/24 wrist and10 unsupported finger/9 wrist). The full
evaluations retain both references, all original boundaries and each variant's
support boundaries without changing the thresholds.

At the original detector-support boundaries, prior EgoHand's unsupported
finger-budget failures fall from **5/9 to 0/9 (EP16)** and **14/33 to 0/33
(EP11)**. These are loss/reacquisition boundaries, not verified occlusion events.
Visible-supported finger checks worsen slightly in both clips. Consequently,
neither refined EgoHand result passes the complete motion guard set.

Heldout MP-reference image distance below is weighted full-hand reprojection
in original pixels, using exactly the same fresh reference per row. The
stride-5 detector coordinates are withheld from fitting, cropping, gating and
calibration. Source RGB and specialist filtering are still transductive.

| Video / hand | Supported heldout frames | Prior EgoHand | Refined EgoHand | Matched MP |
| --- | ---: | ---: | ---: | ---: |
| EP16 left | 5 | 6.168 px | 4.911 px | 5.408 px |
| EP16 right | 63 | 9.210 px | 6.738 px | 4.375 px |
| EP11 left | 71 | 8.068 px | 7.634 px | 6.000 px |
| EP11 right | 79 | 8.284 px | 7.247 px | 5.608 px |

EP16's five supported left-hand heldout frames are insufficient for a broad
left-hand accuracy claim. Most rows remain farther from the shared image
reference than the matched MP fit. Neither agreement nor disagreement with MP
establishes actual 3D accuracy.

Prediction-only wrist-relative finger second-difference p95, pooled over twenty
non-wrist hand landmarks, is recorded below in **native estimated mm/frame²**. It is not official
reference-relative acceleration error, and native scale is not independently
accepted.

| Video / hand | Original SAM body | Prior EgoHand | Refined EgoHand | Matched MP |
| --- | ---: | ---: | ---: | ---: |
| EP16 left | 15.432 | 24.445 | 15.496 | 16.352 |
| EP16 right | 29.447 | 40.058 | 29.544 | 28.208 |
| EP11 left | 29.564 | 37.053 | 30.712 | 29.675 |
| EP11 right | 27.953 | 30.956 | 26.510 | 27.199 |

The reductions often approach the original body's motion. For example, EP16
left finger and arm controls at frames60/110/308 remain exactly the SAM baseline
because no reliable new image support exists. This is conservative abstention,
not inferred hidden articulation. Other frames retain real nonzero updates:
the refined maximum finger changes are0.344rad (EP16) and0.504rad (EP11), below
the unchanged0.65rad bound. Nonzero updates and opening curves prevent a
zero-update result from being mistaken for a successful optimizer; they do not
prove that true grasp amplitude or timing is preserved.

Static review of the predefined EP16 release sequence shows excessive right-hand
curl at frame287 relative to source RGB and matched MP. The hand continues to
change, but the sampled release shape is questionable; a true timing lag is
not established. EP11 samples511/512 and559/560 show no obvious large native
wrist jump at the displayed resolution, but articulation/placement is imperfect
and no grasp-fidelity win over matched MP is established. EP11 left559/560 is
exactly the body baseline with zero admitted image targets; left803 has zero
image targets and a small neighboring-prior correction. At803 the RGB hand is
visible but blurred: missing detector support is not physical occlusion.
The predefined static sample receipts record exactly which images were viewed;
neither static sampling nor complete file decoding claims continuous human
playback or hidden-hand correctness.

**Decision:** retain this as a controlled experiment. The image-support router
and native-depth wrist placement reduce prior EgoHand's added wrist/unsupported
motion, but visible-supported finger stability and grasp fidelity are unresolved.
Do not replace the selected MediaPipe delivery or expand to30 videos. The next
two-video experiment should investigate finger-shape/retarget mismatch, improve
independent image evidence and use an object-relative contact/release model only
after object geometry, coordinate scale and phase evidence have been verified.
Additional temporal filtering alone would not resolve the sampled grasp error.

## Review artifacts and reproducibility

All **16 MP4s** were fully decoded and their original-frame counts/hashes checked:
seven for EP16 and nine for EP11. The independent static reviewer inspected46
predefined comparison JPEGs; root image-review receipts provide additional
sample checks with overlap. This is static source inspection plus complete
file decoding, without a claim of continuous playback or true hidden pose.

Local reviews are under:

```text
J:/video-to-hoi/runs/egohand-refined-pilots-20261031/episode_000016/review02/index.html
J:/video-to-hoi/runs/egohand-refined-pilots-20261031/episode_000011/review02/index.html
```

Start with `hands-comparison.mp4`, then `old-new-egohand-comparison.mp4`.
Inspect the original windows EP16:40-110/240-308 and
EP11:280-340/480-620/680-850 using the slow clips linked in each index.
`proposal-comparison.mp4` explains target admission separately from final native
motion. Original frame IDs are printed in every comparison frame.

The self-contained local archives are under
`J:/video-to-hoi/work/egohand-refined-pilots-20261031/local-delivery02`:

| Archive | Bytes | SHA256 prefix |
| --- | ---: | --- |
| `egohand-refined-pilots-20261031-ep16-review.zip` | 85,120,152 | `f5636f3d3d7f088b` |
| `egohand-refined-pilots-20261031-ep11-review.zip` | 244,720,252 | `6857b292dd54e307` |

Send an entire archive to a reviewer. They extract it and open `index.html`;
playing the videos needs no model server. Each archive includes all comparison
clips, sampled images, original/full evaluation JSON, all five native parameter
variants, generated proposals, crop metadata, source bindings and metadata
correction receipts. Recreating 3D from parameters requires the exact permitted
native MHR decoder. ZIP CRC and every payload SHA were checked, and the
canonical/as-executed numeric NPY members were independently compared again
before packaging. These archives are local files, not public download URLs.

Git contains the [compact full-reference summary](summary.json),
[combined integrity receipt](completed-integrity.json),
[exact local archive identities](local-review-packages.json),
[small-evidence inventory](evidence-inventory.json) and
[reproduction notes](reproduction/README.txt).
Full evaluations/native numeric results/media stay in the review archives.
The recipes retain executed source snapshots and per-stage commands; moving to
another machine requires explicit path/config relocation and the separately
permitted pretrained assets/environments. They are not a portable one-command
installer. The selected dashboard catalog and earlier published EP11/EP16
deliveries remain unchanged.

Source sets are explicit: fresh MP used `mp-runtime01`, the final native solve
and evaluator used `runtime02`, and initial tight-crop preparation used the
producer snapshots saved beside those crop artifacts. The initial CLI's
`5e1c9d4a...` differs from the later captured `9078744b...` only in equivalent
SHA256 file-reading helpers; its numerical code and imported modules match.
The [source note](reproduction/task/crop-preparation-source-note02.json) retains
the exact distinction. No historical receipt is rewritten to a later tool hash.
An [independent producer audit](evidence/crop-producer-source-review.json)
recomputed both clips' crop boxes, all seven trace arrays and core report fields
exactly in memory. This later audit and its recipe have a separate
[small inventory](additional-source-review-inventory.json), preserving the
original staging inventory unchanged.

## Metadata erratum and preservation

The executed frozen refiner recognized the earlier EgoHand report schema but
missed the new hybrid schema. Its top-level report incorrectly called the
generated targets independent MP observations. The actual NPZ kind, target
coordinates/weights, input hashes and fitted numeric arrays were correct.
Future CLI classification is fixed in implementation commit
[`2bd09dd`](https://github.com/Seanwilliam2077/video-to-hoi/commit/2bd09dd), with six
synthetic regression checks.

The executed source/results remain unchanged. Canonical NPZ/report copies correct
only provenance/method/semantics and add explicit lineage. Every non-report NPY
member (pose, shape, scales, fps and frame indices) is byte-identical to the
as-executed file, verified independently. Evaluation and videos bind the
as-executed numeric arrays; the archives contain both copies and the correction
bridge. Nested generated-target residuals remain self-consistency diagnostics,
regardless of generic legacy field labels; they are not independent accuracy.

Full video decoding, all-frame native parity and parameter/budget checks,
original/fresh support boundaries, absolute wrist motion, opening curves,
admission counts and actual retained updates remain separate checks. A frozen
or unsupported hand can look smoother while remaining wrong. MP-reference
distances are method-dependent, especially when MP also anchors/gates EgoHand.

No hidden pose, true grasp timing, physical contact, metric scale or official
submission acceptance is inferred from coverage, projected meshes or smoothness.
Transparent human overlays do not composite object depth; apparent overlap is
not a penetration measurement. Human review is required before any extension
to the 30-video batch.

The preexisting unrelated `docs/status.html` edit is preserved and excluded from
commits. Live Claude module/status publication is unavailable without the
required Artifact tool; this technical record does not claim live publication.
