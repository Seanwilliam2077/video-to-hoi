# Two-video hand continuity pilots

Status on 2026-10-23: both EP16 and EP11 have completed full native geometry export
and independent fresh decoding of all three comparison variants on all 360 and
877 original frames respectively. The new candidates pass the unchanged local
motion screens, including every original adjacent hand and wrist edge. This is
numerical validation of the hand-correction experiment, not acceptance of true
3D reconstruction, grasp or contact quality.

All twelve generated videos have been completely decoded, and sampled image review
is complete for both episodes. Both portable review packages and 32 public files
(16 result assets and 16 checksum sidecars) were anonymously downloaded and verified.
Automated browser playback was **not completed**: browser security policy
blocked opening the local `file://` page. Video decoding and static-frame QA are
recorded separately and do not establish browser UI playback.

Only EP16 and EP11 are authorized for this experiment. **A new all-30 run requires
the owner's explicit approval after reviewing these two pilots.** This report,
the evaluator and passing software tests cannot authorize that run. The existing
30-video delivery, its selected baselines, published bytes, catalog and submitted
records are preserved.

## What changed

The previous hand optimizer abruptly removed its correction when admitted hand
observations disappeared. This could introduce a jump even when the initialization
had a continuous motion. These pilots correct the transition policy while reusing
the original Track 1 inputs:

`Track 1 masks and estimated camera -> SAM 3D Body -> fixed-identity native MHR -> independent MediaPipe observations -> bounded hand/arm correction`

SAM 3D Body and the image detector were **not rerun**. The original baseline,
observations, camera, source-video hashes and original-frame IDs are reused.
Each clip retains one fixed body-shape vector and one fixed skeleton-scale vector.
Root and all unselected controls remain fixed; the existing bounded arm/wrist
controls remain available to maintain wrist coupling.

The new options are opt-in; previous default behavior is unchanged:

| Option | Pilot value | Purpose |
|---|---:|---|
| Support transition frames | 8 | Carry the added correction into a bounded cosine taper outside original support, rather than switching it immediately to zero |
| Correction velocity budget weight | 1 | Penalize excessive added finger and wrist correction steps, including the worst edge so a long clip cannot dilute it |
| Internal budget safety fraction | 0.2 | Reserve margin inside the independent engineering checks; this tightens training, not the acceptance thresholds |
| Learning rate | 0.001 | Avoid the native-geometry overshoot observed at the former 0.025 step size |
| Iterations | 600 | Optimize the complete hand sequence jointly |
| Device / Torch threads | CPU / 1 | Use the existing local workstation runtime |
| Original holdout stride | 5 | Keep every fifth original frame excluded from image fitting |

The taper regularizes **added corrections**, rather than flattening the original
hand trajectory. Frames far outside the bounded collar remain exactly at the
baseline. A collar frame receives no invented hand position, depth or contact
target. Original positive image-fitting masks are frozen: partial observations
that were insufficient for the old support policy are not newly admitted.

Training-observed fast articulation still relaxes the correction-continuity
prior. Heldout image coordinates remain excluded from image position, velocity
and motion-gate fitting. Manual source-review intervals select material to inspect;
they are not used as phase constraints or synthetic grasp targets.

HandFlow was not used. The local environment audit did not identify a ready,
usable HandFlow code/checkpoint/MANO setup. The actual independent observations
in these results are MediaPipe landmarks, not HandFlow or HaMeR outputs. This
experiment does not establish which specialist model would perform best.

## Results to compare

Every review must distinguish three versions:

1. **Baseline:** the fixed-body result selected in the existing all-30 delivery.
2. **Old candidate:** the previous MediaPipe correction that was rejected by the
   support-boundary stability screen.
3. **New candidate:** this bounded-transition correction, saved separately and
   never automatically promoted into the all-30 catalog.

The following numbers are weighted **absolute 2D reprojection errors in pixels**
on the original heldout detector observations. They measure image agreement,
not true 3D hand accuracy. Lower is better for this particular diagnostic.

| Episode / hand | Baseline | Old rejected candidate | New candidate | Validation state |
|---|---:|---:|---:|---|
| EP16 left | 5.76 | 5.53 | 5.26 | Full candidate, fresh native decode complete |
| EP16 right | 6.79 | 4.95 | 4.30 | Full candidate, fresh native decode complete |
| EP11 left | 7.36 | 5.90 | 6.22 | Full candidate, fresh native decode complete |
| EP11 right | 7.48 | 5.18 | 5.72 | Full candidate, fresh native decode complete |

Both new candidates improve on the baseline's heldout image agreement. The new
EP11 candidate has a higher pixel error than the old rejected candidate. This is
a measured tradeoff: reducing problematic transition corrections does not improve
every aggregate image metric. A lower average pixel error in the old candidate
did not prevent its large boundary jumps.

The EP16 heldout means use only five left-hand frames and 64 right-hand frames;
EP11 uses 71 and 78 respectively. The sparse EP16 left-hand evidence
cannot establish the correctness of its long hidden interaction.

These heldout checks were consulted repeatedly while developing the pilot
configuration. Although their image targets are excluded from optimization,
they are development validation, not an untouched final test. Generalization
beyond these two clips remains unverified.

| Local motion screen | EP16 old candidate | EP16 new candidate | EP11 old candidate | EP11 new candidate |
|---|---:|---:|---:|---:|
| Failed original support boundaries | 9 / 11 | 0 / 11 | 33 / 35 | 0 / 35 |
| Failed edges across all original adjacent hand frames | 41 / 718 | 0 / 718 | 191 / 1,752 | 0 / 1,752 |

Counts refer to side-specific consecutive-frame edges, not bad frames or official
competition errors. EP16 contains 360 frames, hence 359 adjacent pairs per hand
and 718 hand edges. EP11 contains 877 frames and 1,752 hand edges. Both full
candidates reproduce their corresponding probe03 numerical results; their pose,
shape, scales and original-frame arrays are byte-identical to those probes.

EP16's full candidate also passes its three additional collar boundaries, all
718 separate wrist-edge checks, 127 observed edges touching the original holdout,
and 17 independently observed fast-motion edges under the unchanged local
nonregression policy. Its identity, skeleton scales, root and unselected native
controls are unchanged. These are engineering checks, not accepted grasp quality.

EP11's full candidate also passes its 29 additional collar boundaries, all 1,752
separate wrist-edge checks, 269 observed edges touching the original holdout, and
71 independently observed fast-motion edges. Its identity, scales, root and all
unselected controls are unchanged. The old EP11 candidate had 253 wrist-edge
failures under this separate wrist diagnostic; the new one has none.

## How continuity is checked

The evaluator checks the original old-candidate support boundaries explicitly.
It also checks any additional boundaries introduced by the collar and **every
original adjacent frame pair**, including interior observed motion, heldout-touching
edges, fast observed motion and separate absolute wrist motion. Moving a boundary
cannot remove the original edge from evaluation.

The independent finger policy remains unchanged:

- With at least eight accepted common finger landmarks and both accepted wrists,
  the new velocity discrepancy must be at most `baseline * 1.05 + 0.25 pixels`.
- Otherwise, the mean wrist-relative step of the **added correction** must be at
  most `1 pixel` and `0.005 native estimated metres`.

Separate wrist diagnostics use the same numerical tolerances on the wrist's
absolute correction and observed velocity. These are local engineering choices,
not official wrist scoring rules.

The internal safety fraction of 0.2 uses tighter training margins: `0.2 pixels`,
`0.001 native estimated metres` and an observed absolute allowance of `0.05 pixels`.
The independent thresholds above were not weakened. Loss normalizers stay fixed
to avoid making the optimizer abruptly much stiffer when reducing the margin.
The complete heldout targets are used only for post-fit evaluation.

Fresh native validation decodes each full baseline, old candidate and new candidate
with the identified MHR model, then checks saved geometry against that decode.
Small decoder differences establish representation consistency, not physical
accuracy or a shared human/object metric scale.

## Measured local runtime

| Measured component, seconds | EP16 | EP11 |
|---|---:|---:|
| Refiner report: fitting plus complete geometry export | 158.203 | 405.688 |
| Independent fresh three-variant native evaluation | 69.969 | 138.890 |
| Hand/body videos, slow intervals and critical screenshots | 301.516 | 727.329 |

The refiner's reported interval excludes launcher/input-hash overhead. Failed
attempts, code development, software tests and publication are not included in
these component times. Jobs for the two episodes overlapped, so these values are
not a total elapsed project time. Existing body inference and image observations
were reused; this is sequence fitting, not training a new neural model.

## Failed attempts and software verification

EP16 probe01 used the new budget with the old learning rate of 0.025. The native
objective overshot; the best retained result was exactly the unchanged initializer.
It is recorded as an optimization stall, not a successful stable reconstruction.

Probe02 lowered the learning rate to 0.001 for 600 iterations. Its image agreement
improved, but two of the 11 original EP16 boundary checks still failed. One was
slightly over the unknown-motion budget; the other touched a heldout frame and
could not be repaired by inserting that frame's hidden fitting target.

Probe03 reserved an internal safety margin with fraction 0.2. The same configuration
was then used for both episodes. Full `candidate01` runs repeat that configuration
with complete geometry export; they do not introduce another learned model or
new observations. The optimizer retains its best total objective, which can occur
before the final iteration. Objective improvement alone is not a quality pass.

Verification completed so far:

- The explicit Python workflow test list: **917 passed, 25 skipped**. Source and
  test hashes stayed unchanged throughout that final verification.
- Existing Torch runtime focused hand/contact/native-result tests: **97 passed**.
  These include heldout isolation, unchanged image-positive support, bounded
  collars, retained fast observed articulation, retained fast initializer motion
  during absent evidence, and explicit zero-update stall reporting.
- Node workflow tests: **44 passed**.
- Existing delivery registry and catalog remain valid: **90 records, 30 videos,
  16,563 original frames**, without promoting these new pilots.
- Default opt-out fitting was checked against the immutable previous runtime on
  an independent synthetic case: identical pose bytes and identical objective.

The focused test receipt records exact commands, existing Python/Torch/pytest
locations, source hashes, and the frozen-runtime/worktree hash match. No package
installation or environment replacement was needed. Synthetic fixture success
does not establish success on the two real videos.

## Remaining problems and review instructions

The experiment targets **additional jumps introduced by hand refinement**. It does
not solve every twitch already present in the baseline. For example, the EP11
right-hand transition at original frames 559 to 560 still has a roughly 8.09-pixel
velocity-vector discrepancy against independent wrist-relative image observations.
Passing nonregression relative to a poor baseline does not establish recovery of
that motion. EP11 frame 803 also retains a visible left-hand pose mismatch with
no accepted independent left-hand observation.

EP16 has only 27 frames with accepted left-hand detector evidence. Its left hand
has no accepted observations through the long central interaction interval.
The bounded collar cannot recover a verified hidden grasp across that interval.
A detector dropout is also not proof of visual occlusion: crop, side assignment,
confidence filtering and ambiguous fingers can all remove support.

For each episode, review the synchronized source, baseline, old candidate and new
candidate in the hand video. First inspect the old problematic boundaries, then
the approach/grasp/release intervals at normal speed and slow speed. Check that
wrists stay attached, fingers do not snap at detector return, and action timing
still matches the source. A contact sheet alone cannot establish temporal quality.

EP16 review windows cover approach/lift/tilt around frames 40-110 and lower/release
around frames 240-308. EP11 windows cover table interaction around 280-340,
dropouts around 480-620, and grasp/table tipping around 680-850. These broad manual
windows are inspection aids, not ground-truth action boundaries.

No object-contact constraint, physical penetration guarantee, calibrated world
scale, hidden reference trajectory or official competition score is provided.
Improved image agreement and nonregressing observed motion cannot establish true
occluded 3D finger motion. Full CARI4D/CoCoNet comparison is outside this pilot.

The next decision is a human review of both completed comparisons and their
independent checks. The owner may then approve or reject a new all-30 run. This
pilot does not change optimizer defaults or enable automatic batch promotion.

## Files, reproduction and publication state

Local source snapshots and job receipts:

`J:/video-to-hoi/work/human-hand-pilots-20261023/`

Local outputs:

`J:/video-to-hoi/runs/human-hand-pilots-20261023/episode_000016/`

`J:/video-to-hoi/runs/human-hand-pilots-20261023/episode_000011/`

Each episode's completed local layout is:

| Path | Contents / state |
|---|---|
| `candidate01/` | New native parameters, report and complete geometry; both episodes complete |
| `evaluation01.json` | Fresh three-variant native and motion validation; both episodes complete |
| `review-attempt01/index.html` | Synchronized comparison report; both episodes generated; automated local-file browser opening blocked |
| `review-attempt01/hands/hands-comparison.mp4` | Complete baseline / old candidate / new candidate hand comparison |
| `review-attempt01/body/index.html` | New-candidate full-body overlay and frame review |
| `review-attempt01/critical/` | Critical-frame snapshots, contact sheet and slow-motion windows |

The exact argument arrays and immutable input bindings are in `ep16-candidate01.json`
and `ep11-candidate01.json` under the task directory. For example, the EP16 run is:

```powershell
J:/video-to-hoi/work/human-runtime/venv/Scripts/python.exe `
  J:/video-to-hoi/work/human-hand-pilots-20261023/runtime03/tools/refine_human_hands.py `
  --baseline J:/video-to-hoi/runs/human-all30-20261021/episode_000016/baseline/attempt-0001/episode_000016_human.npz `
  --observations J:/video-to-hoi/runs/human-all30-20261021/episode_000016/observations/attempt-0001/observations.npz `
  --manifest J:/video-to-hoi/runs/human-all30-20261021/episode_000016/inference/attempt-0001/manifest.json `
  --review-arrays J:/video-to-hoi/runs/human-all30-20261021/episode_000016/baseline/attempt-0001/review_arrays `
  --model J:/video-to-hoi/weights/human/mhr_model.pt --device cpu --threads 1 --model-batch 8 `
  --iterations 600 --learning-rate 0.001 --holdout-stride 5 --optimize-wrist-chain `
  --support-transition-frames 8 --correction-velocity-budget-weight 1 `
  --correction-budget-safety-fraction 0.2 `
  --output J:/video-to-hoi/runs/human-hand-pilots-20261023/episode_000016/candidate-reproduction-fresh
```

Use a fresh output directory to preserve attempts. EP11 uses the corresponding
`episode_000011` paths with the identical options. These commands refine only
the two authorized videos; they do not launch the all-30 batch.

Frozen fitting source SHA-256:

- `src/v2hoi/hand_refinement.py`: `eb0c2560f8ebbbbc6d2065f1d482b0c14c2b730d61bf97bb754d1ff326baa56f`
- `tools/refine_human_hands.py`: `ed5daabe7b3448a5405590865f64ccfe9aee2410bc420b4063fc4f16dc9a8ff7`
- Runtime03 snapshot manifest: `0a7468f9e84833d7deee7db4d8bf8c4fd7dcd82da770e314c658ba1f77591ce7`

The implementation was committed and pushed to `human` as
[`b8768fea0731c84a7dc21c8576481ba2899d49bf`](https://github.com/Seanwilliam2077/video-to-hoi/commit/b8768fea0731c84a7dc21c8576481ba2899d49bf).
Git blob bytes for the refiner, CLI and independent evaluator match runtime03.
The [source-binding receipt](review/implementation-source-binding.json) records
those exact hashes.

Published review links:

- [All pilot assets and portable packages](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/human-hand-pilots-20261023).
- EP16: [three-variant hands](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/human-hand-pilots-20261023/human-hand-pilots-20261023-ep16-hands-comparison.mp4), [release-transition slow view](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/human-hand-pilots-20261023/human-hand-pilots-20261023-ep16-transition-slow.mp4), [portable package](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/human-hand-pilots-20261023/human-hand-pilots-20261023-ep16-review.zip).
- EP11: [three-variant hands](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/human-hand-pilots-20261023/human-hand-pilots-20261023-ep11-hands-comparison.mp4), [table-tipping slow view](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/human-hand-pilots-20261023/human-hand-pilots-20261023-ep11-transition-slow.mp4), [portable package](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/human-hand-pilots-20261023/human-hand-pilots-20261023-ep11-review.zip).

Each portable package opens at `index.html` after extraction. It includes the
three-variant hand video, body video, all four slow intervals, sampled images,
the three native parameter variants and numerical checks; model weights and
original source videos are excluded. `parameters/candidate.npz` remains an
experimental candidate, not a replacement for the all-30 selected result.

Evidence copies: [EP16 checks](review/ep16-summary.json),
[EP11 checks](review/ep11-summary.json), [publication receipt](review/publication.json),
[completion and approval boundary](review/pilot-completion.json),
[static visual review and playback limitation](review/static-visual-review.json),
[EP16 full video decoding](review/ep16-video-qa.json),
[EP11 full video decoding](review/ep11-video-qa.json), and
[original inputs preserved](review/previous-inputs-preserved.json).
The larger complete all-edge evaluations are available as Release JSON files,
with their exact hashes bound by the committed summaries.

Source and documentation are on the `human` branch. No main merge, deployment,
all-30 rerun or automatic result promotion was performed.

The preexisting unrelated `docs/status.html` change remains preserved. Live Claude
Human/status publication is pending because the required live Artifact publishing
tool is unavailable; repository documentation is not a claim that those live
pages were updated.
