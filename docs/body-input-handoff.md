# Receiving the 30-video Body input package

Updated 2026-10-16. The delivery target is **all 30 complete Track 1 videos**, with an original-frame human mask for every frame and an episode-specific camera estimate. The consumer profile is `sam3d_body_masks_camera.v1`, using the standalone Body interface at upstream revision `33129dd0f2d2dcfd1164d43fd076542660756ed2`.

Read the actual check reports accompanying a delivery to determine its completion. This guide describes the receiving procedure; its existence is not a passing delivery record. Body model inference is not performed by the input preparation or either input check below.

## What to transfer

Use the complete result archive and its externally supplied SHA256. Normal Git history carries scripts, small receipts and documentation; the archive carries videos and frame masks. Every worker may extract it into a different local directory.

```text
<extracted-delivery>/
  body-inputs/
    episode_000000/ ... episode_000029/
      input.json
      timeline.json
      human_masks/000000.png ... <last-original-frame>.png
      observations.json
      masks-manifest.json
      intrinsics.json
      camera-estimation.json
      review.json
      visual-review-receipt.json
      SHA256SUMS
      ... retained visual review evidence
  data/
    v2d/track_1/                 # unchanged original videos and metadata
    prepared/track1/             # verified raw manifest, checks and timelines
  source/track1-source.json
  runtime/
    check_body_handoff.py
    prepare_track1_data.py
    package_body_handoff.py
    check_body_native_readers.py
  requirements.txt
  requirements-native-readers.txt
  producer-checks.json
  runtime-environment.json
  bundle-manifest.json
  SHA256SUMS
  README.md
```

Keep this relative layout. In particular, the raw preparation manifest resolves its source at `../../v2d/track_1`; moving only its directory breaks that reference. `input.json` identifies the exact original video and all model input artifacts by hash. No producer-specific absolute path is needed.

The archive does not include Body model weights, its full GPU environment, object masks or object meshes. Those are separate dependencies. This package targets standalone human initialization; full native CARI4D additionally requires the object inputs described in [distributed execution](https://github.com/Seanwilliam2077/video-to-hoi/blob/main/docs/distributed-execution.md).

## Check the receiving copy before running Body

Compare the archive hash with the value provided alongside the delivery, then extract into a fresh directory. Create and activate a dedicated CPU validation environment using the tested **Python 3.12** version before installing the pinned requirements. Run the following from that extracted directory, with `python` resolving to that Python 3.12 environment:

```bash
python -m pip install -r requirements.txt
python runtime/check_body_handoff.py \
  --package-root body-inputs \
  --dataset-root data/v2d/track_1 \
  --raw-preparation-root data/prepared/track1 \
  --source-inventory source/track1-source.json \
  --output receiver-checks.json
```

Use a new report filename for every attempt. The validator does not overwrite an earlier result. Omit `--episodes` for acceptance: a subset run is diagnostic and cannot demonstrate that every requested video is ready.

Require exit code 0, `passed: true`, `all_30_body_inputs_verified: true`, and `videos_verified: 30`. Check the reported original-frame total against the frozen raw manifest. A gate failure blocks that handoff until the producer corrects and reissues it; changing an artifact also invalidates its review bindings.

This gate verifies original source identities, every decoded source timestamp, all mask filenames and pixels, observation bounds, camera aggregation, complete inventories and current review records. It does not prove segmentation or reconstruction accuracy. The visual review records identify the actual reviewer, sampled frames, findings and remaining issues.

## Check the actual upstream input readers

The handoff validator is independent of the model package. A second CPU check exercises the exact upstream `FrameSource` and `CameraIntrinsics` implementations. It requires only the permitted common-code subtree at the selected revision; it does not import the Body network or load weights.

Obtain a sparse code checkout, or use an existing verified checkout at that revision:

```bash
git clone --filter=blob:none --no-checkout https://github.com/nvidia-isaac/video_to_data.git upstream-body-reader
git -C upstream-body-reader sparse-checkout init --no-cone
git -C upstream-body-reader sparse-checkout set /reconstruction/modules/v2d_common/
git -C upstream-body-reader checkout --detach 33129dd0f2d2dcfd1164d43fd076542660756ed2
python -m pip install -r requirements-native-readers.txt
python runtime/check_body_native_readers.py \
  --upstream-root upstream-body-reader \
  --package-root body-inputs \
  --dataset-root data/v2d/track_1 \
  --handoff-checks receiver-checks.json \
  --output receiver-native-reader-checks.json
```

The check imports the verified common source directly; it does not install the full upstream project. The companion requirements record the actual tested reader dependencies, which can differ from the historical upstream package recipe. Require exit code 0 and `all_30_native_readers_passed: true`. The check sequentially reads all RGB/mask pairs, confirms exact frame stems, dimensions and counts, and loads each camera into the native float32 matrix. It also rechecks video, camera and mask identities against the previously verified input. Preserve both reports with the receiving worker's runtime record.

`body_model_environment_verified` and `body_execution_completed` remain false in this reader report. A successful reader check does not exercise model initialization, GPU compatibility, memory consumption or prediction quality. The receiver must record those separately when starting its human reconstruction task.

## Model runtime and weight layout

Use a Body runtime built from the selected source revision, including its `v2d.common`, `v2d.mv`, `v2d.sam3d_body.lib`, `sam_3d_body` and required GPU dependencies. The archive's `requirements.txt` belongs to CPU input validation and is not a Body environment specification. The historical upstream Docker recipe also needs compatibility review on the receiver's actual GPU; no universal GPU memory guarantee follows from preprocessing on another machine.

The standalone library resolves weights directly beneath `--weights_dir`:

```text
<standalone-weights>/
  sam-3d-body-dinov3/
    model.ckpt
    model_config.yaml
    assets/mhr_model.pt
    ... other files from the same pinned model bundle
  torch_home/
  hf_home/
```

The repository's CARI4D bundle instead stores the model under `sam3d_body/checkpoints/sam-3d-body-dinov3/`. Prepare a separate standalone view by copying the complete verified model directory and required caches, and recheck copied bytes against the model receipts. Passing the CARI4D `sam3d_body/` directory leaves the standalone model path wrong; passing only `checkpoints/` loses the wrapper's expected cache layout. Acquire gated model files through an authorized model account when they are not already available; the input package does not confer model access.

For direct library execution, set `TORCH_HOME` and `HF_HOME` to that view's cache directories. The upstream Docker wrapper sets these paths internally based on `--weights_dir`.

## Start the selected Body consumer

From the extracted package root, in the prepared Body environment, this is the native command for episode 16:

```bash
python -m v2d.sam3d_body.lib.estimate_mhr_params \
  --rgb_path data/v2d/track_1/videos/chunk-000/observation.images.exo_camera/episode_000016.mp4 \
  --mask_path body-inputs/episode_000016/human_masks \
  --cam_intrinsics_path body-inputs/episode_000016/intrinsics.json \
  --weights_dir /absolute/path/to/standalone-weights \
  --output_params_path body-results/episode_000016/mhr_params.pt \
  --output_mesh_path body-results/episode_000016/mhr_mesh.pt \
  --batch_size 1 \
  --debug 1
```

On a worker with the corresponding upstream container image and wrapper installed, use `python -m v2d.sam3d_body.docker.run_estimate_mhr_params` with the same arguments. The container wrapper mounts the provided inputs and outputs. Validate that image/runtime on the receiver before treating it as ready.

Repeat for every episode index `000000` through `000029`, keeping each complete temporal sequence together and writing results outside the immutable input directories. The first run establishes the receiver's actual environment and resource behavior; it does not replace preparation or verification of the other 29 videos. Do not reuse an existing result solely because its filename is present: bind new results to the input and model hashes.

The selected profile supplies masks and camera intrinsics explicitly. The upstream library otherwise has bbox/full-image and default-FoV fallbacks; those fallbacks are not evidence of a correct delivery. Check each result's native metadata: `prompt_counts.mask` must equal the original frame count and both fallback counts must be zero. Separately check native output coverage, parameter semantics and visual overlays before marking human reconstruction complete.

## Camera meaning and limitations

`intrinsics.json` contains exactly `fx`, `fy`, `cx`, `cy`, `width`, and `height` in the unchanged source image grid. These are **estimated intrinsics**, not ground-truth calibration. MoGe2 predicts normalized focal information on 13 deterministic endpoint-inclusive frames from the same episode. The adapter converts to integer pixel-center coordinates using `K_pixel = [[W,0,-0.5],[0,H,-0.5],[0,0,1]] @ K_normalized` and takes the componentwise median of all samples.

The model assumes a centered principal point and does not estimate distortion. Each report retains all raw estimates, source frame IDs, preprocessing conventions, model identities and spread diagnostics. The gate independently rejects a relative p10-p90 focal spread above 0.25 or maximum deviation from the median above 0.40. Passing those screens indicates temporal consistency, not absolute camera accuracy or verified metric scale. Inconsistent results remain candidates and cannot silently become accepted Body inputs.

## Interface evidence

- [Body argument parsing, mask fallback behavior and weight lookup](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_sam3d_body/lib/estimate_mhr_params.py)
- [FrameSource video and PNG directory readers](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_common/video.py)
- [Native six-field CameraIntrinsics](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_common/datatypes.py)
- [Docker wrapper arguments and cache mounts](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_sam3d_body/docker/run_estimate_mhr_params.py)
- [MoGe2 focal inference](https://github.com/microsoft/MoGe/blob/925b8ed835a7a9cdb7578ba15c658a0afc969030/moge/model/v2.py) and [normalized-to-pixel conversion](https://github.com/EasternJournalist/utils3d/blob/3fab839f0be9931dac7c8488eb0e1600c236e183/utils3d/torch/transforms.py)
