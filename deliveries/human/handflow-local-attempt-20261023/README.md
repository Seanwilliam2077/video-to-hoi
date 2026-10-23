# Local HandFlow deployment attempt

Status on 2026-10-23: **partial deployment verified; real-video comparison blocked
by missing owner-authorized MANO models**. The owner confirmed that those models
are not yet available. This is not a completed HandFlow reconstruction, a quality
comparison, or a new all-30 delivery.

The authorized scope remains Track 1 EP16 (360 frames) and EP11 (877 frames),
using the local workstation. No all-30 inference, result promotion, or changes
to existing published MediaPipe results were performed.

## What actually ran

The official target is the July 2026
[HandFlow: Fully Generative 4D Hand Recovery with Flow Matching](https://github.com/mxxu00/HandFlow),
not another project with the same name. Source-only sparse checkouts were pinned:

| Component | Commit |
|---|---|
| HandFlow | `d0f429b6355058fcd4366fbde18ccbfd7c871628` |
| HaMeR, matching HandFlow's submodule | `3a01849f4148352e9260b69bf28b65d1671a4905` |
| manopth | `4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b` |

Bundled videos, datasets and standalone MANO model files were excluded. No Track 2
content or derivatives were obtained or used.

The task-local virtual environment passed all 13 import checks, including the
HandFlow denoiser, the HaMeR class and the MANO calculation library. `pip check`
passed. Added dependencies are installed in that environment; existing Torch
packages are shared through a read-only import path. The existing Human runtime
was not upgraded or replaced.

Five selected public weight/configuration files, totaling 3,357,448,567 bytes,
were downloaded from the official HandFlow model repository and HaMeR author's
linked Hugging Face Space. Local hashes were verified; LFS hashes match upstream,
and the small model configuration matches its upstream Git blob:

- `handflow_denoiser.pt` and `normalization_stats.npz`;
- `hamer.ckpt`, `model_config.yaml`, and `mano_mean_params.npz`.

The HaMeR checkpoint was **downloaded but not deserialized or instantiated**.
Any MANO buffers potentially embedded in that checkpoint were not extracted or
used as a replacement for the separately licensed models. In the original
download receipt, `mano_models_downloaded: false` means that no standalone MANO
`.pkl` file was downloaded; it makes no claim about checkpoint buffer contents.

A CPU smoke loaded the genuine trained HandFlow temporal checkpoint. All 504
non-backbone keys matched, with no unexpected keys. Three independent synthetic
16-frame input cases produced finite forward outputs: all observed, middle
unobserved, and all unobserved.

This smoke supplied synthetic compressed image tokens. The upstream constructor's
unused HaMeR backbone was uninitialized and guarded against accidental forward
execution. The smoke did not extract real image features, integrate a full video
trajectory, decode MANO geometry, fit MHR, or measure hand quality. Its roughly
0.12-second forward calls are **not** end-to-end inference speed estimates.

## Local environment and compatibility limits

Task directory:

`J:/video-to-hoi/work/handflow-pilots-20261023/`

Python:

`J:/video-to-hoi/work/handflow-pilots-20261023/venv/Scripts/python.exe`

This attempt uses Python 3.11, Torch 2.5.1/cu124 and torchvision 0.20.1/cu124.
Upstream's setup instead specifies Python 3.10 and Torch 2.7/cu128. The verified
imports and temporal forward establish limited compatibility, not full-pipeline
equivalence. Small CUDA matrix computation also passed; the temporal smoke used
CPU. Available GPU memory varied during preparation and is not reserved.

`runtime_compat.install()` supplies process-local legacy NumPy/`inspect` aliases
for chumpy and selects Windows OpenGL imports. Upstream tracked source files
remain unchanged. The stock PyTorch3D renderer, ViPE and YOLO demo stack were not
installed. A fixed-camera, staged, headless runner and a controlled crop/MHR
adapter remain to be implemented and validated.

The initial missing `torchdiffeq` import and manopth packaging failure were
preserved in task logs and resolved locally. A GitHub API rate limit during
source discovery was handled using source-only sparse Git checkouts.

## Required input before real reconstruction

The [MANO website](https://mano.is.tue.mpg.de/) requires registration and agreement
to its [model license](https://mano.is.tue.mpg.de/license.html). The owner must
obtain the official models under the applicable terms. The stock HandFlow FK
wrapper constructs both hand models, so retain `MANO_RIGHT.pkl` and
`MANO_LEFT.pkl` from the authorized download. A private local destination is:

`J:/video-to-hoi/work/handflow-pilots-20261023/private-mano/`

These model files must not be committed or included in public result packages.
The task's preflight records the missing input and exits 78; it is a current
blocking-status check, not an implemented end-to-end inference launcher.

Once the owner supplies that authorized directory, the existing instruction
already authorizes continuing the two local pilots. No separate approval to
resume EP16/EP11 is required. A new all-30 run remains outside scope.

## Frozen comparison protocol

The original source-video and input hashes were rechecked. The comparison will
retain original frame IDs, 30 fps, estimated camera, fixed SAM/MHR body shape
and skeleton scales. It will compare the current bounded MediaPipe candidate,
not only the older rejected hand correction. Existing outputs remain immutable.

1. **Controlled image input.** Reuse the same original crop bounds and RGB
   content for MediaPipe and HandFlow's HaMeR frontend, retaining each model's
   required resize/normalization. HandFlow's stock YOLO boxes and extra expansion
   would constitute a separate diagnostic, not the controlled comparison.
2. **Correct anatomical mapping.** Explicitly mirror the left hand into the
   right-trained model, transform the principal point consistently, and reflect
   results back. `--side left` alone does not implement that transformation.
   Verify 21-joint ordering, wrist position, original-pixel projection and
   metre/millimetre conversion before fitting the same native MHR. Do not copy
   MANO pose arrays into MHR controls.
3. **Same downstream solve.** Freeze the current control restrictions, iteration
   count, regularization and numerical checks. Use the same fitting/validation
   original-frame mask for both MHR fits, excluding validation targets from
   image-position, velocity and motion-gate losses. Keep HandFlow's generated
   occlusion proposals separate from independent image observations. Any added
   3D-prior loss needs a separately identified ablation. Predetermine random
   seeds; do not select a favourable seed after watching results.
4. **Independent checks.** Reuse every adjacent-frame and absolute-wrist check,
   original observation boundaries, new dropout/return boundaries and temporal
   window joins. Retain the unchanged local 1-pixel/5-mm unknown-motion budgets
   on native estimated geometry, not calibrated world-scale accuracy, and the
   observed velocity nonregression policy. Existing every-fifth-frame
   MediaPipe validation has influenced development and remains development
   validation, not an untouched final test. The detector's outputs are not
   ground truth and can favour its own fitting method.
5. **Actual grasp review.** Inspect closure amplitude, direction and timing,
   wrist attachment, release/reopening and reappearance, not only low motion
   acceleration. Prioritize EP16 frames 40-110 and 240-308, EP11 frames 280-340,
   480-620 and 680-850, including 559-560 and frame 803. These are review windows,
   not verified event annotations. Hidden finger correctness, contact,
   penetration and world scale remain unverified without additional evidence.
6. **Video delivery.** Render source RGB, baseline, current MediaPipe and
   HandFlow-driven MHR on identical display crops, plus normal-speed and 3x-slow
   transition views with original frame IDs. Show generated missing-observation
   frames with uncertainty labels; upstream's stock overlay hides those frames.
   Provide a raw MANO diagnostic separately and fully decode all exported videos.

This protocol is planned, not executed. Actual intrinsics through HaMeR's
projection code, left-side dropout handling, geometric retargeting and headless
memory staging still need validation after MANO is available.

## Evidence and current conclusion

- [Deployment state](evidence/deployment-state.json)
- [Source commits and materialized-file hashes](evidence/source-snapshot.json)
- [Package versions and import checks](evidence/environment-receipt.json)
- [Executed environment commands](evidence/environment-commands.json)
- [Public weight downloads and hashes](evidence/weights-receipt.json)
- [Trained temporal-branch synthetic smoke](evidence/temporal-smoke-receipt.json)
- [Missing-model preflight](evidence/full-inference-preflight.json)
- [Frozen original inputs and current MediaPipe candidates](evidence/original-inputs.json)

Local import and temporal-checkpoint compatibility have been demonstrated.
**No real HandFlow video, native MHR candidate, comparison video or quality
conclusion is available yet.** The prior MediaPipe comparison videos are not
HandFlow results. Live module/status publishing has not occurred; the required
Artifact tool is unavailable, and the preexisting unrelated status-page edit
remains preserved.
