# Human reconstruction strategy as of 6 October 2026

**Keep native MHR as the project representation. Improve its sequence fit using
independent image evidence, motion observations, and human/object contact.**
SAM 3D Body remains the practical initializer already working on this machine.
FlowHMR, HTD-Refine, and HandFlow are useful new comparisons or specialist
observations; none has demonstrated a higher Track 1 score in this project.

Research cutoff: **2026-10-06 inclusive**. Prepared on 2026-10-09. Publication
dates and pre-cutoff code/checkpoint availability are distinguished below.
Only primary papers, author repositories, and official challenge documentation
support factual claims. Recommendations and expected metric effects are project
judgments and hypotheses. No new research model was installed or benchmarked.
Only the 30 Track 1 videos and verified metadata may be used for development,
tuning, validation, and submission. No Track 2 content or derivatives were used.

## The current baseline

Execution follow-up, 15 October: see the separate
[hand and official-control experiment](human-hand-experiment-20261015.md)
for actual independent observations, finger-enabled sequence fits, contact
failures and the locally executed official control. The literature cutoff above
is unchanged. The baseline limitations described below refer to the earlier
CPU reconstruction, not to every later experiment.

The completed local experiment is the full foam-block video, Track 1
episode 16: 360 original frames at 30 fps, 1536 by 1152 pixels. It uses full
SAM 3D Body body/hand inference on CPU, then fits one identity and one skeletal
scale vector with the official native MHR model. Raw predictions and the fitted
result have separate frame-exact reviews, each with 360/360 coverage and no
missing frames. Inference through the final fitted review took **2 h 03 min
48 s on CPU**; raw preview generation overlapped fitting. Reopened native
parameters, forward geometry, camera consistency and browser checks passed.
See [human-local.md](human-local.md) for execution commands and measured results.

The earlier three-frame run was a deliberately partial smoke test at frames
0, 120, and 240. It did not establish continuous motion. A viewer position of
`1 / 360` was the current frame number; the `3 / 360 reconstructed` badge was
the actual coverage. For the full run, those three cached predictions were
reused only after source/model/camera settings and file hashes were verified.
The remaining 357 frames were inferred individually; no interpolation counts
as a reconstructed frame.

The current fitter uses SAM's own predicted vertices as its targets. A small
change after fixing identity measures fidelity to the initializer, not accuracy
against the real person. It has no independent image-fitting term, temporal
motion objective, object geometry, or contact refinement. The manual person box
and assumed 60-degree horizontal field of view are recorded uncertainties.
Completion proves that the Human workflow runs over a whole video; it does not
prove a complete HOI submission or a competitive reconstruction.

## Recent methods and what they contribute

| Method and public date | Available by the cutoff | Output and project value |
| --- | --- | --- |
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body), checkpoint release 2025-11-19; [paper](https://arxiv.org/abs/2602.15989) 2026-02-17 | Official code and model | Native MHR body and articulated hands from images. Keep as the initializer; it does not provide video continuity by itself. |
| [SAM-Body4D](https://arxiv.org/abs/2512.08406), 2025-12-09 | [Author code](https://github.com/gaomingqi/sam-body4d) | Training-free video masking and occlusion recovery around SAM Body. A compatible MHR route; generated hidden appearance needs confidence gating. |
| [DuoMo](https://arxiv.org/abs/2603.03265), 2026-03-03, CVPR 2026 | [Meta code and checkpoints](https://github.com/facebookresearch/DuoMo), publicly released before October | Two-stage camera/world motion diffusion. It predicts 595 sparse MHR-LOD6 vertices rather than native MHR parameters; detailed articulated finger fidelity is not established. Useful motion evidence after fitting; preserve detailed SAM hands. |
| [Fast SAM 3D Body](https://arxiv.org/abs/2603.15603), 2026-03-16 | [Author implementation](https://github.com/yangtiming/Fast-SAM-3D-Body), including a September 29 decoder-parity fix | Parallel crops and a reduced decoding path target throughput. Its reported GPU speedup is not a CPU measurement or proof of better accuracy. Audit the retained native-MHR/hand path before adoption. |
| [HTD-Refine](https://arxiv.org/abs/2605.26879), 2026-05-26, CVPR 2026 oral | [Code](https://github.com/ant-research/HTD-Refine) released 2026-08-08, PVA checkpoint linked | Estimates position, velocity, and acceleration evidence for sequence refinement. Original optimization uses SMPL/SMPL-X; adapt verified motion targets to MHR. Particularly relevant to preserving movement timing while reducing jitter. |
| [HandFlow](https://arxiv.org/abs/2607.11221), 2026-07-13 | [Code](https://github.com/mxxu00/HandFlow) and checkpoints released 2026-07-13 | Temporal generative MANO hands with confidence-masked observations. Fixed-camera mode is relevant. Right-hand training requires explicit left-hand mirroring/restoration, wrist-chain coupling, and MANO-to-MHR fitting. |
| [Native-MHR CARI4D module](https://github.com/nvidia-isaac/video_to_data/blob/v0.3.0/reconstruction/modules/v2d_cari4d/README.md), toolkit v0.3.0, 2026-09-17 | Official toolkit and [gated MHR CoCoNet model](https://huggingface.co/nvidia/cari4d_commercial) | Joint human/object/contact reconstruction using masks, depth, SAM Body, object tracking, and contact refinement. This is the most relevant HOI baseline once our own scaled textured object mesh and masks exist; it performs object tracking internally. The released native checkpoint and default post-optimizer freeze finger articulation; see the dated source audit below. |
| [Unified MHR upper-body reconstruction and retargeting](https://arxiv.org/abs/2609.37776), 2026-09-29 | Paper; no public code/weights link verified | Combines frame-wise body, video, and hand observations in one differentiable MHR state. Confirms that shared identity plus body/hand temporal fitting alone is already published. Its upper-body results are not challenge scores. |
| [FlowHMR](https://arxiv.org/abs/2610.03691), 2026-10-02 | [Code](https://github.com/flowhmr/flowhmr) 2026-10-04/05; [checkpoint](https://huggingface.co/fafsaf1/flowhmr-0.46B) added 2026-09-30 | The newest verified runnable full-body motion candidate in this review. Video-conditioned flow matching and physics-reward post-training produce SMPL-H body/hands/global motion. Use as a controlled temporal comparison or alternative motion hypothesis; converting to MHR is required. |

FlowHMR's physics reward is applied during training. Its normal inference does
not automatically run a simulator to choose the best reconstruction. The paper's
tracking success on its own benchmark is not a V2D score. Training a new GRPO
system from 30 unlabeled challenge clips is a poor first investment; comparing
the released pretrained model is a more practical experiment.
[Paper](https://arxiv.org/abs/2610.03691),
[official inference documentation](https://github.com/flowhmr/flowhmr).

Other screened work includes [OnlineHMR](https://arxiv.org/abs/2603.17355)
(streaming world-space SMPL without articulated fingers),
[Human3R](https://arxiv.org/abs/2510.06219) (scene/camera/human SMPL-X recovery),
[DanceHMR](https://arxiv.org/abs/2605.18102) (temporal body/hand SMPL-X fusion;
no runnable release verified), and
[Sapiens2](https://github.com/facebookresearch/sapiens2) (pose, segmentation,
normals, and pointmaps as possible image observations). Moving-camera and
multi-person advantages are less valuable for our single-person static-camera
task than accurate hands, object-relative motion, and complete coverage.

The October 5 paper
[Robotizing Human Videos with Physically Consistent Interactions](https://arxiv.org/abs/2610.06137)
also studies stabilized grasp and contact reconstruction, but was not identified
as a runnable native-MHR Human baseline. The October 4
[VICON](https://arxiv.org/abs/2610.05180) result requires a contact glove and
RGB-D; its sensor inputs are unavailable in our monocular challenge setting.

Check licenses and dependency stacks before integrating a candidate. In
particular, FlowHMR restricts code use to educational/research/nonprofit purposes,
HTD-Refine uses AGPL, and DuoMo has research-license restrictions. MANO, SMPL,
HaMeR, and auxiliary checkpoints have their own terms. Keep new environments
isolated from the proven Windows CPU runtime and source-pin authorized assets.
Do not fetch upstream example data of unknown provenance.

## Recommended project pipeline

1. **Initialize every frame with SAM 3D Body in native MHR.** Preserve body and
   hands, exact frame IDs, camera conventions, provenance, and raw predictions.
2. **Fit shared identity, skeletal scale, and a static camera using reliable
   frames.** Add independently observed image keypoints and visibility-aware
   silhouette evidence. Treat predicted depth as uncertain. Do not fit strong
   naked-body contours to loose clothing or assume focal length is known.
3. **Refine the whole motion against observed dynamics.** Preserve starts,
   stops, reversals, and contact changes; regularize unsupported jitter more
   strongly than visible fast motion. HTD/PVA targets and FlowHMR candidates
   are optional observations after representation and unit checks.
4. **Spend hand computation on difficult intervals.** Use crop prompts or a
   temporal hand expert where blur, occlusion, disagreement, or image residuals
   indicate failure. Fit its evidence through the same MHR wrist/arm chain;
   never attach an unrelated hand mesh or copy incompatible parameters.
5. **Optimize Human and Object together when valid object inputs exist.**
   Reproduce the official native-MHR CARI4D route on our Track 1-derived assets
   as a control, then test contact improvements. Export the official native
   Human parameters and full object trajectory with one verified coordinate
   system and frame mapping.

This is a staged engineering recommendation, not an empirically ranked best
model. The current SAM/shared-identity baseline is implemented. Independent
image fitting, dynamic-target refinement, alternative experts, and joint contact
optimization remain proposed work. A new model should replace a component only
after the same Track 1 inputs show a useful improvement at an acceptable cost.

## Three priority experiments

**Independent evidence with selective reconstruction.** Optimize shared MHR
identity/camera and per-frame pose against visible image evidence. Use residuals,
mask/blur visibility, expert disagreement, and forward/backward consistency to
decide which intervals need an extra body/hand hypothesis. Compare always-SAM,
always-expert, and gated-expert policies under the same compute budget. The
potential contribution is the evidence-selection policy, not merely combining
two established models.

**Motion-aware refinement.** Constrain movement using image motion and verified
velocity/acceleration predictions. Weaken regularization at observed fast
transients, strengthen it in poorly observed static intervals, and separate
root/body/hand weights. Compare no temporal loss, constant smoothing, adaptive
smoothing, and motion-target fitting. HTD-Refine establishes the broader idea;
our hypothesis is a confidence/contact-conditioned native-MHR implementation.
[HTD-Refine](https://arxiv.org/abs/2605.26879).

**Contact phases with persistent object-local surface anchors.** Infer approach,
grasp/support, release, and slide phases. During a stable grasp, retain a small
set of palm/finger surface anchors in object coordinates; update uncertain
contact assignments during optimization. Jointly penalize penetration and
detachment while preserving image evidence. Compare against no contact,
nearest-wrist attraction, and the baseline's fixed contact gate. This is the
strongest interaction-specific hypothesis once object geometry and scale are
trustworthy. Contact optimization itself is established by
[CARI4D](https://arxiv.org/abs/2512.11988) and other HOI methods.

An additional experiment is sequence-level selection of a few plausible
depth/handedness/object-symmetry hypotheses using later visible frames and
contact continuity. Give extra review to the first scored-frame neighborhood,
since its human geometry anchors episode alignment. Do not confuse a smoothly
wrong hypothesis with a correct reconstruction.

These experiments can support an engineering contribution if they show repeatable
quality/cost benefits. They are not yet demonstrated score improvements or
claims of research novelty. Generative completion can hallucinate hidden hands;
incorrect object scale can corrupt correct human pose; strong smoothing can
erase a real grasp change. Retain the raw baseline and bounded-update controls.

## What the competition metrics imply

The current official contract is recorded in
[assessment-20260930.md](assessment-20260930.md), based on selectively extracted,
provenance-checked participant text. No reference arrays or scorer outputs were
loaded. The
[organizer FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf)
also distinguishes predicted motion from reference-relative acceleration error.

- **CD-H:** the entire human surface matters. Root/limb/shape failures can
  outweigh detailed finger improvements.
- **ACC-H:** evaluates reference-relative second differences of 22 body joints,
  including wrists and excluding finger joints. Low self-acceleration is not
  automatically good; finger-only smoothing cannot directly improve this metric.
- **PEN:** uses 512 hand points against the submitted object mesh. Separating a
  hand from the object can reduce penetration while making the reconstruction
  worse, so also inspect surface distance, visible grasp, and relative motion.
- **Shared alignment:** the first scored frame, which need not be video frame 0,
  sets one human-derived similarity transform reused for the episode and object.
  Relative scale, articulated anchor errors, and time-varying drift matter.
- **Coverage across actions:** episodes contribute equally. One polished foam
  clip cannot establish quality on all 30 videos and object types.

Freeze episodes **0, 6, 9, 16, 24** as a diverse Track 1 development subset, then
check all 30 for integration. Compare raw SAM, current shared self-fit, independent
image/camera fitting, motion refinement, hand gating, and contact one change at
a time. Report full coverage, independent body/hand reprojection, mask agreement,
observed-motion timing, contact/penetration diagnostics, failure cases, runtime,
memory, and manual interventions. Only organizer evaluation establishes the five
official scores. Local proxies and paper benchmarks cannot promise a high rank.

## Research records and next decision

Detailed release evidence and implementation caveats are saved in
`work/human-research-20261006/sota-evidence.md`; metric/innovation analysis is in
`work/human-research-20261006/innovation-evidence.md`. These working files are
local research records. This page is the durable project recommendation.

The complete CPU baseline now runs, but the owner observed clear finger twitching.
Next invest in independent image/camera fitting and motion-aware native-MHR
refinement that actually optimizes finger controls. Trial FlowHMR and
HandFlow on the same difficult Track 1 windows when suitable server resources
are available. Add joint contact refinement when the Object module supplies
verified geometry and trajectories. A large-memory GPU mainly enables efficient
full-sequence comparisons and joint optimization; it cannot resolve monocular
ambiguity or replace validation.

## Follow-up source audit: NVIDIA and Sharpa, 2026-10-09

This follow-up records a later inspection of the official website and source;
it does not extend the paper/release cutoff above. NVIDIA toolkit tag `v0.3.0`
resolves to `595ad68d897b986cd154d4575a022411af096eb3`; observed main was
`193382a184308f6800331f3db829f76dc4294cfe`. The inspected native optimizer,
supervision contract and production configuration are identical between them.
Only source text and permitted episode 16 predictions were inspected; no
reference data, upstream examples, new weights or model inference were used.

The [official CARI4D MHR module](https://github.com/nvidia-isaac/video_to_data/blob/595ad68d897b986cd154d4575a022411af096eb3/reconstruction/modules/v2d_cari4d/README.md)
requires RGB, human/object masks and a separately prepared scaled textured object
mesh. It orchestrates MoGe 2, SAM 3D Body, depth-scale alignment, FoundationPose,
temporal CoCoNet corrections and full-clip contact refinement. It tracks the
object itself. Its inference command does not generate the input object mesh.
The geometric contact gate is fixed from the initial hand-surface distance;
that gate is not a persistent fingertip grasp constraint.

**The default native pipeline is not a ready-made finger-twitch repair.** Its
[released configuration](https://github.com/nvidia-isaac/video_to_data/blob/595ad68d897b986cd154d4575a022411af096eb3/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/configs/mhr-daniel-commercial-moge2-behave79-val-fp16.yml)
sets `w_mhr_hand: 0.0` and `body12_freeze_hand_face_all_losses`.
The [supervision contract](https://github.com/nvidia-isaac/video_to_data/blob/595ad68d897b986cd154d4575a022411af096eb3/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_supervision.py)
zeroes unsupervised corrections, so local finger articulation inherits the
initializer. This does not mean the model produces no hands: body and wrist
updates can still move hands in space. It means no learned local-finger correction
is enabled by that checkpoint contract.

The [post-optimizer](https://github.com/nvidia-isaac/video_to_data/blob/595ad68d897b986cd154d4575a022411af096eb3/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py)
defaults to `smplh_parity`, optimizing body rotation controls and object translation
while freezing `mhr_hand` and the human root/identity/scale. Temporal joint losses
therefore cannot directly change local finger articulation. Its alternate
`legacy_staged` path enables hand parameters in later stages, with temporal/prior
and contact losses, but provides no independent finger-image fitting objective.
It is an experiment to compare, not established better reconstruction. Changing
a training loss weight in inference does not train a frozen released hand head.

The [official SAM initializer](https://github.com/nvidia-isaac/video_to_data/blob/595ad68d897b986cd154d4575a022411af096eb3/reconstruction/modules/v2d_cari4d/lib/cari4d/prep/run_sam3d_mhr_export.py)
uses body inference without the dedicated hand-crop pass. The current local run
used both body and hands. Our `pose136/scales68/shape45` NPZ also differs from
the official pipeline's intermediate body/hand/scale blocks. Reuse needs a
verified representation/camera/unit adapter and a geometry round trip; do not
truncate or rename arrays. Reproducing the official initialization in its own
environment is a useful control. Its [model card](https://huggingface.co/nvidia/cari4d_commercial)
validates Linux/CUDA on A100, not a minimum-memory guarantee for our machine.

**Sharpa is downstream robot hardware and retargeting in this public V2D stack.**
The [NVIDIA repository](https://github.com/nvidia-isaac/video_to_data) places
Sharpa in robotic grounding after human reconstruction. Sharpa's public
[vision-retargeting SDK](https://github.com/sharpa-robotics/sharpa-vision-retargeting-sdk)
uses MediaPipe webcam landmarks or Meta Quest tracking, hand-size calibration
and dex-retargeting to command Wave robot hands. This is not evidence of superior
human-mesh reconstruction on our occluded third-person clips. NVIDIA's separate
[egocentric hand module](https://github.com/nvidia-isaac/video_to_data/blob/main/reconstruction/modules/v2d_ego_hand_reconstruction/README.md)
uses ViPE/Dyn-HaMR; that camera/domain pipeline must be adapted and tested before
using it on Track 1.

The local code audit found almost unchanged raw-to-fitted hand jump patterns:
step-magnitude correlation 0.9938 and second-difference correlation 0.9933 in
native control units. A one-frame reversal occurs around frames 285-287,
9.50-9.57 seconds. These are diagnostics for locating the defect, not official
errors. Shared identity fixes proportions; it did not validate or stabilize
fingers.

Next compare the same Track 1 inputs across current SAM, official default joint
reconstruction, and a finger-enabled MHR sequence optimizer. The latter needs
independent visible hand keypoints, crop/handedness confidence, coupled wrist and
finger motion, and contact-phase constraints on trustworthy object geometry.
Check image fidelity and movement timing alongside jitter and penetration;
selecting the smallest acceleration alone can erase real grasp changes.
The [current FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf)
confirms reference-relative acceleration and native MHR submission. It now also
describes sum-of-metric-ranks aggregation; the homepage retains equal axis
weighting language. Record that public wording discrepancy rather than inventing
a weighted local leaderboard score.
