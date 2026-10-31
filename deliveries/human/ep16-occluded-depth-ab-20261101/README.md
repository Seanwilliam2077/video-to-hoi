# EP16 occluded-left and shared-depth contact A/B pilots

The hidden left hand now adjusts, but its front/back relationship to the object is **not solved**. Two complete controlled A/B pairs show a modest conditional skin-depth improvement with residual deep penetration and contact-only wrist-motion regressions. The preserved primary02 comparison and negative findings are delivered for review.

This is a local research candidate. Reconstruction, physical depth/contact and
official export remain unaccepted. No contact network was trained, no new EP11
experiment ran, and no all-30 rerun or selected-dashboard promotion occurred.

## Scope and identity

| Field | Actual value |
| --- | --- |
| Delivery ID / date | `ep16-occluded-depth-ab-20261101` / 2026-11-01 |
| Module / capability | M2 Human / bounded occluded-hand native refinement and conditional human-object geometry review |
| Work type | Code, independent synthetic checks and real Track 1 EP16 optimization |
| Executor / contributors | Codex on the owner's Windows workstation; separate solver, object-depth, independent evaluation and report contributors recorded in the archive |
| Code PR / implementation | [PR9](https://github.com/Seanwilliam2077/video-to-hoi/pull/9) / [`b7f40591544e3868db0f16cdc8e4ce38b925c8cb`](https://github.com/Seanwilliam2077/video-to-hoi/commit/b7f40591544e3868db0f16cdc8e4ce38b925c8cb) on `human` |
| Related delivery | [Previous EP16 contact experiment](../ep16-contact-ab-20261031/README.md); preserved independently, never overwritten |
| Consumer / review scope | Human and interaction reviewers; full-video grasp, occlusion transitions, source-image fidelity and conditional skin geometry |

The previous contact experiment could not change the hidden left fingers during
reliable carry. This implementation separates manually admitted pose variables
from image observations, improves the shared depth initializer, and adds actual
native-skin constraints. Both A/B arms receive the same newly available controls;
only B receives the contact/surface/ordering intervention.

## Inputs and reproduction

- **Source:** only EP16 RGB, its own masks/estimates and verified generic native
  decoder assets. Source video SHA256:
  `a37a978ef0f3d0eeaad74f2ed75fb32502d8a1a047587291c14433d052f9eb74`.
  No Track 2 content or derivatives are consumed.
- **Coverage:** all original IDs0–359, 360 consecutive frames, 30fps. No missing
  frames, reindexing or differentiation across omitted-frame gaps.
- **Common initialization:** the previous matched MediaPipe native result,
  SHA256 `4a58a3af46a19032b16b74c8f06f93fc5a3dc097c6d533a361ad0cd971e1f169`.
  Its numeric pose/shape/scales match the preserved old A; both original SAM and
  old A/B remain comparison roles.
- **Image evidence:** unchanged final routed MediaPipe observations,
  SHA256 `941012c727c11b2039abce7309b66e5da4bc27b1e2c8944740fe2f93cf5646f3`.
  Original stride5 image targets remain zero during fitting. Manual control
  eligibility does not create observations, confidence or hidden fingertip labels.
- **Native decoder:** generic MHR model SHA256
  `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc`.
  Pose136, shape45 and scales68 follow the existing verified native contract.
  Identity, skeleton scales, root and all unselected controls remain fixed.
- **Environment:** Windows build26200, Python3.11.16, PyTorch2.5.1+cu124, NumPy1.26.4 and OpenCV4.11.0; actual fitting/evaluation/rendering use CPU.
  Fitting uses CPU8 threads; independent native evaluation uses CPU1 and batch8;
  rendering uses eight task-owned workers. Peak memory was not measured.

Exact executed arguments, source bytes and raw receipts are in the review ZIP.
Public summaries redact executor-local paths; they are explicitly summaries,
not byte-identical raw receipts. Git normalizes source line endings; archived
runtime source bytes retain their original SHA256. The code-commit receipt
verifies equality after CRLF/LF normalization.

Reproduction requires separately permitted source inputs and the exact decoder.
Relocate paths deliberately and create a new set of bound manifests/receipts;
the archive is a review package, not a portable native importer or installer.
The core command pattern is:

```text
python tools/refine_human_hands.py --baseline <shared-native.npz>
  --observations <source-observations.npz> --manifest <EP16-manifest.json>
  --review-arrays <shared-native-review-arrays> --model <licensed-mhr-model.pt>
  --output <fresh-attempt> --device cpu --threads 8 --model-batch 8
  --iterations 600 --learning-rate <0.001-or-0.0005> --holdout-stride 5
  --unobserved-policy preserve --relative-weight 1 --acceleration-weight 0.025
  --pose-prior-weight 0.03 --max-delta-radians 0.65 --optimize-wrist-chain
  --arm-delta-limit-radians 0.1 --wrist-max-displacement-metres 0.04
  --arm-update-temporal-weight 0.08 --finger-update-temporal-weight 0.08
  --support-transition-frames 8 --correction-velocity-budget-weight 1
  --correction-budget-safety-fraction 0.2 --spatial-acceleration-weight 0
  --control-prior-support <control-prior-support02.json>
  --control-prior-arm-delta-limit-radians 0.35
  --control-prior-wrist-max-displacement-metres 0.12
  --control-prior-wrist-ray-weight 0.5
  --control-prior-wrist-max-projection-pixels 3
```

Join the wrapped lines into one argument list. A has no contact argument.
B adds `--surface-contact-config <surface-contact-config01.json>`; that file
binds the frozen object, manual phases, common seed, decoder and weights.
No model weights are optimized.

### One shared, conditional object-depth hypothesis

The camera stays fixed at fx=fy818.7458496px, cx767.5, cy575.5, using OpenCV
x-right/y-down/z-forward estimated metres. One projective factor **0.99625**
multiplies the original canonical mesh and all360 camera translations exactly
once. Rotations and image projections remain unchanged; contact scale is1.
The new geometry SHA256 is
`51576aca6fc1bc00f6ce2ca39f13d8c1796cb185c6c01c8d75a09f0e88002e94`.

Selection uses126 native right-skin support points in34 training frames90–230,
excluding stride5 before selection, with fixed confidence, reprojection,
surface-distance and front-facing criteria. The selected skin/triangle pairs
remain fixed during the scan. The data-only factor is0.99575; prior sigma0.01,
0.02 and0.04 gives0.9975,0.99625 and0.99575. None hits the design bounds0.94–1.06.
All360 object projections agree to2.27e-13px in the mathematical invariance check.

This is a visible-right contact-supported depth/scale initializer, approximately
1cm nearer the camera. It is not physical calibration or R1 reference alignment.
The hidden left hand never selects the gauge. Both new A/B arms use exactly this
object, and every evaluated human is also measured under the preserved old
object so a gauge change cannot silently earn credit for a human improvement.

### Shared controls and contact intervention

Manually reviewed LEFT control eligibility covers40–289 with a40-frame internal
cosine ramp. It allows preparation/recovery around uncertain pickup/release;
it does not assert that all those frames are occluded or in contact. The contact
schedule itself remains75–250, further gated by the unchanged object reliability.
Carry contains176 frames, of which129 pass that gate.

LEFT prior-supported arm controls allow0.35rad and wrist displacement0.12 native
estimated metres. Ordinary/RH bounds remain0.1rad/0.04m; fingers remain0.65rad.
The common3px LEFT wrist trust region refers to the initializer's projection,
not an independently observed wrist. Source inspection cannot locate the hidden
anatomical wrist precisely. The exact native Jacobian/Gauss–Newton projection
removes lateral constraint violations before a bounded binary fallback.

B evaluates128 distributed actual skin vertices per hand with exact native pose
correctives and LBS;16 fixed vertex IDs per hand provide weak surface attraction.
IDs are selected from training-only common-seed clearances, without fixed3D
targets, so tangential movement is possible. The objectives are:

- skin interior clearance with3mm slack: mean plus0.25 worst weighted point,
  normalized1cm, weight0.10;
- selected-patch absolute supporting-plane clearance, pseudo-Huber normalized
  2cm, weight0.02;
- LEFT wrist and mean MCP depth against frozen common-seed far-ray targets with
  3mm margin, normalized2cm, weight0.10.

Supporting-plane clearance is exact inside this convex object, but only a lower
bound on outside Euclidean distance. Fixed skin patches can still bias genuine
rolling/contact-patch changes. The far-order prior is not applied to all fingers.
Current-ray intersection coverage remains independently reported; moving off
the object projection cannot erase the frozen reference-ray penalty.

Early TRAIN-only probes exposed lateral ray escape and radial-clamp stalling.
Their actual outputs and executed/recovered exact sources are preserved. The
final four-pose probe improves sampled skin geometry within the unchanged limits,
but frame181's fixed palm-depth conflict remains unresolved. It is feasibility
evidence, not quality acceptance.

### Paired optimizer stability ablation

The second complete pair was frozen before B01 finished or any new heldout
evaluation was reviewed. A01's training trace showed correction-budget spikes
while other losses decreased. Pair02 halves only the common learning rate;
iterations, weights, observations, controls and thresholds remain unchanged.

| Pair | Common learning rate | Retained A TRAIN total | Retained B TRAIN total | CPU fit wall A / B |
| --- | ---: | ---: | ---: | ---: |
| 01 | 0.001 | 0.11729753 | 4.38510799 | 240.83s / 1025.57s |
| 02 | 0.0005 | 0.10458227 | 4.19623184 | 201.28s / 944.61s |

The primary video is pair**02**, selected solely by the
predeclared minimum retained B TRAIN total under identical B objectives. A
has no contact terms, so A/B objective totals are not comparable quality scores.
Both full numerical outcomes remain available; selection is not acceptance.

## Results and producer checks

| Hand / fixed scope | Old A mean / worst (mm) | New A mean / worst (mm) | New B mean / worst (mm) |
| --- | ---: | ---: | ---: |
| Left / 176 raw | 44.036 / 103.502 | 44.070 / 103.212 | 41.856 / 102.786 |
| Left / 129 usable | 43.469 / 103.502 | 43.510 / 103.212 | 40.577 / 102.786 |
| Right / 176 raw | 0.126 / 27.054 | 0.379 / 32.734 | 0.208 / 28.312 |
| Right / 129 usable | 0.049 / 15.917 | 0.286 / 27.329 | 0.055 / 12.123 |

Every table entry uses the **same NEW object**, not a mixture of old/new gauges.
Mean is over all2321 selected skin vertices, including points outside the
object; worst is the deepest selected point/frame. The old A under its OLD
object has raw left mean41.270mm, so the table does not establish an end-to-end
physical improvement. The two-object crossmatrix is retained in both evaluations.

The hidden left fingers now change throughout all176 carry frames. Wrist
movement from the shared seed averages9.56mm, peaks18.06mm, and remains within
2.999975px of the initializer projection. Frame181 still has a fixed-reference
wrist/palm shortfall of33.137/129.630mm with current rays valid. Control mutability
and valid intersection coverage do not establish correct hidden pose.

| Motion guard / reference | Contact-only new A→new B | Cumulative old A→new B |
| --- | --- | --- |
| Fresh finger / wrist / unknown failures | 2/314 / 21/321 / 0 | 0 / 0 / 0 |
| Legacy finger / wrist / unknown failures | 2/312 / 23/321 / 0 | 2 / 3 / 0 |

These two comparisons answer different questions. A failed relative step guard
is not a visually observed twitch count. Both detector references remain
visible; neither is ground truth. The contact-only wrist regressions include
placement/release and clear-release frames. Image-mean guards pass for both
references, but fresh LEFT heldout support is only5 frames. Fresh LEFT mean
projection error changes5.158→4.735px; RIGHT3.552→3.555px.

Frozen full-skin nonregression guards against matched A pass under both object
hypotheses, but the approximately103mm absolute left conflict remains severe.
Pair01 also remains in the record: raw LEFT44.050→42.405mm, worst103.502→103.271mm;
usable LEFT43.490→40.885mm. Its fresh finger/wrist failures are6/314 and23/321.

Halving the step reduces correction-budget levels/peaks. It does not prove full
convergence: B's standard deviation over the last five20-step log checkpoints
increases about27%. A separate post-hoc four-TRAIN-pose test raises penetration
and far-order weights0.10→0.50 without changing the primary result. Full-skin
changes are small, one worst depth worsens, and frame181's palm shortfall remains
131.49mm. Stronger weights alone do not resolve this setup.

See the explicitly redacted [primary findings](evidence/primary-findings.json),
[cumulative review](evidence/cumulative-review.json),
[fixed/current ray review](evidence/ordering-review.json),
[optimizer trace review](evidence/training-trace.json) and
[post-hoc native audit](evidence/posthoc-native-audit.json). Exact raw sources
are identified by archive member and SHA in each summary.


All displayed and evaluated native trajectories are freshly decoded over360
frames. Independent checks bind exact parameters, image/control weights, manual
eligibility, per-side limits and every native review array. Full2321 skin vertices
per hand are checked;128 optimization samples are not used as a substitute for
the complete diagnostic geometry.

Engineering thresholds remain unchanged: image/observed velocity tolerance
5%+0.25px; unknown correction steps1px/0.005 estimated metres; phase/whole-clip
skin nonregression0.5mm mean and2mm worst depth. Both legacy/fresh detector
references, raw frames and the same fixed reliability subset are reported.
These are internal guards, not official scores or evidence of true hidden motion.

| Check | Actual outcome / evidence | Scope |
| --- | --- | --- |
| Explicit repository workflow03 | 1148 Python tests passed,38 skipped;44 Node tests; registry/catalog passed | Software/contract checks; optional native cases covered separately |
| Existing-Torch focused solver/evaluator checks | 111 solver and35 independent evaluator synthetic checks passed | No challenge data in those fixtures |
| Selected native skin / control gradients | Zero difference at inspected original and modified poses | Exact decoder path; not real geometry accuracy |
| Complete real native evaluation | Both evaluations completed; original IDs0–359 and fresh decoder/array/control/source parity checked. [Primary findings](evidence/primary-findings.json) | Five human variants under both fixed object hypotheses in each paired attempt |
| Comparison media | All six outputs fully decoded, with360/71/71 frames per view; final renderer665.21s controller wall. First snapshot-closure failure is preserved separately and recovered without altering fits/evaluations. [Media summary](evidence/media-summary.json) | Original30fps and every-frame10fps slow replay; no interpolated frames |
| Actual static review | Root inspected12 actual images at6 IDs; independent Codex reviewer inspected26 images at13 IDs. LEFT articulation changes are visible in the transparent view, RGB benefit is subtle, and hidden depth/release correctness remains unverified. [Independent review](evidence/static-review.json). Offline HTML passed224 checks at1280/768/390/320px; all40 images decoded; six videos each started approximately320ms per width, zero HTTP requests. This is startup verification, not full human video review. [Browser QA](evidence/browser-qa.json) | Source/grasp/release/transparent geometry sampling; no claim of continuous human playback |
| Archive verification | All317 members CRC/SHA verified; six videos freshly extracted and fully decoded. Archive SHA `90482f091b43887d6d3c65b35d9d070aae7c20dc243d4ebacc84be18a797cade` | CRC/SHA for every member and full decoding of archived videos |

## Output inventory and location

| Output | Verified archive member / location | Bytes | SHA256 |
| --- | --- | ---: | --- |
| Review ZIP | Owner-local task store / `ep16-occluded-depth-ab-20261101.zip`; not remotely published | 38392601 | `90482f091b43887d6d3c65b35d9d070aae7c20dc243d4ebacc84be18a797cade` |
| Offline Chinese report | `documentation/interactive/ep16-left-hand-ab-20261101.html` | 10058376 | `f3521be23f777b49d9642c7e95f365b32b92f9ac98bcd972c141c4324580c8dc` |
| RGB full comparison | `rgb-comparison.mp4` | 1431905 | `c98d799c906a5c67ef883968d1b03d8766604f0f76e19865b3dc8ee8bdc1545b` |
| Transparent full comparison | `transparent-comparison.mp4` | 510966 | `252258b19bdf958f21376ca4c443ee88f35b9910a24d0604f7597659fe43eb8d` |
| Primary A native | `parameters/a.npz` | 207179 | `90948f1041779d8263b97009f5eeab49be8b99da9e089fbc2da0827a31c6c33c` |
| Primary B native | `parameters/b.npz` | 218373 | `96d04e56534f6f86489ed77ce0e41b0ba8d48b32a1ea6f8dc76d949f4852f96e` |
| Primary full evaluation | `checks/full-evaluation.json` | 65127298 | `7aa0fca41f25cf8a87b5f356c69a3743ba06513b2a3a7b23ba80bbbc70de6fd8` |
| Secondary full evaluation | `attempts/01/full-evaluation.json` | 65150045 | `e869ed445f5e1e718da666166988784bbf84bf116d7fcbf32f9ffc2321afec8c` |

The [publication manifest](publication-manifest.json) lists every archived member, exact bytes and SHA, including four slow videos, forty samples and both paired native runs. Its archive receipt is producer-side local verification, not remote publication or consumer native-import acceptance.

The owner-local runtime archive has not been published to a remote result
release. Large videos, raw receipts, native arrays and geometry remain outside
Git. It contains source-bound native SAM/shared MP/old A/B/new paired A/B,
both object hypotheses, image references, policies, both complete evaluations,
actual runtime sources and execution/review records. Decoder weights, the raw
source video, full per-frame human vertices, raw logs and unknown caches are
excluded. Dependency identities remain explicit; native regeneration requires
the separately permitted inputs and model.

RGB columns are original / preserved old A with old object / new A / new B;
the two new columns share the same object and display crop. Transparent-object
views use a synthetic fixed object-local60degree camera and show anatomical L/R
skin through the object; wrists are clipped by the skin selection. These displays
do not establish real-scene visibility or physical contact. No table/background
depth is supplied.

## Receiver verification and review

Producer and independent Codex reviewers verified the exact local output bytes,
native contracts, all original frames and the recorded static samples. Local
browser/video checks are recorded separately. Transfer to another machine,
consumer dependency closure and validated native package import remain
**pending**. The self-contained Chinese report and extracted ZIP viewer need
no model server for video viewing; native3D reproduction remains separate.

## Separate conclusions and next action

| Conclusion | Actual state |
| --- | --- |
| Execution | Completed: four360-frame fits, two full independent evaluations, six comparison videos and offline report; original failed attempts retained |
| Structural local review package | Producer-side local CRC/SHA and archived video decoding passed; dependencies and exclusions are explicit; remote transfer pending |
| Consumer import / relocation | Pending; no receiver-side native importer acceptance claimed |
| Reconstruction / physical contact quality | Not accepted: severe conditional left penetration, unresolved palm depth and contact-only motion regressions remain |
| Official scene export / reload | Not accepted; this is a human research experiment, not a complete challenge submission |

Before a learned contact model, separate incorrect hidden-arm initialization from an incorrect object depth/shape hypothesis. Review native arm-depth candidates against visible forearm/occlusion evidence, and test wrist-projection trust and arm-control limits separately under the same heldout checks. Do not force every finger behind an uncertain box. Then revisit changing contact patches and release constraints while preserving genuine grip changes. There is no evidence here that simply raising contact weights solves the error.

The explicit workflow evidence is [Python](evidence/workflow-python.json),
[Node](evidence/workflow-node.json), [registry](evidence/workflow-registry.json)
and [catalog](evidence/workflow-catalog.json). The [primary selection](evidence/primary-choice.json)
preserves both declared attempts; [code binding](evidence/code-commit.json)
records the source/runtime newline convention. All public JSON in `evidence/`
is clearly labelled as a path-redacted review summary; raw receipts remain in
the ZIP. The archive's English findings document and self-contained Chinese
viewer explain the same negative/mixed conclusion.

The selected all-30 catalog is unchanged. Neither successful execution nor this
record starts training, EP11 validation or a new batch. Live Claude Human/Status
publication is unavailable without the required Artifact publisher; the
unrelated status-page edit is preserved and no main merge/deployment is made.
