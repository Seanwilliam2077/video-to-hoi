# Local SAM 3D Body walkthrough

For the 2026-10-15 follow-up on the observed hand twitching, see
[the hand experiment report](human-hand-experiment-20261015.md). It includes
independent image detections, actual all-frame native finger/wrist optimization,
motion failures, synchronized close-ups, object contact guards and the separate
official-pipeline attempt. The original CPU baseline below remains immutable;
improved detector agreement is not a measured official reconstruction score.

This workflow uses only Track 1 episode 16, the foam-block clip. It runs real
SAM 3D Body with the body and hand decoders, retains native MHR parameters,
fits one identity and skeletal-scale vector, and generates frame-exact reviews.
It is separate from `python -m v2hoi.run`, whose registered defaults remain fake.

## What is ready on this workstation

- Windows Python 3.11.16 at `work/human-runtime/venv/Scripts/python.exe`.
- Torch 2.5.1+cu124 and torchvision 0.20.1+cu124; CUDA arithmetic passed on the
  RTX 3080. This is not a GPU-model memory acceptance result.
- Pinned source-only SAM Body and DINOv3, checksum-verified weights and licenses.
- Complete episode 16: 360 frames, 30 fps, 1536 x 1152.
- Completed CPU body-and-hands inference for **all 360 frames** in
  `runs/human-ep16-cpu-full/`, with exact source indices `0..359`, finite arrays,
  and complete raw and fitted reviews. Shared-identity fitting and reopened
  native output checks passed; see the measured full CPU section below.
- The complete fitted output has `pose[360,136]`, one `scales[68]`, and one
  `shape[45]`, all finite float32 arrays. Its native forward pass produces
  18,439 vertices and 127 joints per frame, with the original frame IDs retained.
- The historical smoke run inferred frames **0, 120, 240**. Each took about
  18 seconds; peak resident memory was about 7.4 GiB.
- Completed shared-identity fitting and native forward validation on those three
  sampled frames: `pose[3,136]`, one `scales[68]`, and one `shape[45]`, all finite
  float32 arrays. The saved model produces 18,439 vertices and 127 joints per
  frame. Visual review retains source-frame IDs and flags incomplete coverage.
- The setup wrapper was rerun successfully with `-SkipDownloads`, checking all
  48 installed packages, source files, selected assets and the official converter.
  A real checkpoint reload verified 1,127 loaded tensors with zero learned-weight
  mismatches. After the parallel-fit and renderer changes, the explicit
  repository test suite passed **214 tests**.

**The historical smoke run covers 3/360 frames; the full CPU raw run covers
360/360.** Full raw inference has now been measured at 96 minutes 14.770 seconds.
Identity fitting and review have separate timings. GPU model inference and
full-clip GPU fitting remain unmeasured. The model's own output, its projection,
and parameter-fit errors are not hidden-reference competition accuracy.

## Directory layout

```text
work/human-runtime/venv/              isolated Human interpreter
tools/human_runtime/                  pinned setup/source/asset helpers
third_party/sam3d-body-runtime/       source-only inference runtime
third_party/track1-mhr-converter/     official conversion code, no reference data
weights/human/                       verified Body and MHR model files
data/v2d/track_1/                     episode 16 and permitted metadata
runs/human-ep16-cpu-smoke/            historical three-frame CPU evidence
runs/human-ep16-cpu-full/             complete raw/fitted CPU run and reviews
runs/human-ep16-cuda-full/            future GPU experiment, not measured
```

The general `.venv` remains a separate Python 3.12 development/test environment.
Do not mix it with Human dependencies. Model/video/runtime directories are
ignored by Git. Source checkout alone does not contain their binary assets.

## Full CPU episode 16: completed on 2026-10-09

The full run has all 360 real source frames, at 30 fps and 1536 x 1152. It
computed 357 new predictions and reused the exact, checksum-verified predictions
for frames 0, 120 and 240 from the smoke run. The video, runtime source, model,
camera and body-and-hands configuration matched before reuse; this is recorded
in `runs/human-ep16-cpu-full/reused_frames.json`. No missing-frame interpolation
was used. `state.json` reports complete coverage; `validation.json` verifies
exact indices, finite per-frame and raw arrays, and camera intrinsics matching
the manifest. The fitted export uses one shared identity and skeletal-scale
vector for all 360 separately fitted poses.

| Stage | Measured time or status | Evidence in `runs/human-ep16-cpu-full/` |
|---|---|---|
| Full raw inference process: 357 new + 3 reused frames | 5,774.770 s (96 min 14.770 s), exit 0 | `inference-timing.json` |
| Model loading within inference | 16.063 s; included in process time | `state.json` |
| Recorded per-frame inference times, including the 3 historical reused timings | Sum 5,768.496 s; mean 16.024 s/frame | `validation.json`, `raw/inference_seconds.npy` |
| Raw review generation | 880.472 s (14 min 40.472 s), exit 0; 360/360 coverage | `reviewraw-timing.json`, `review-raw/summary.json` |
| Full shared identity, pose refits and native export | 1,555.856 s (25 min 55.856 s) process wall, exit 0; fitter 1,513.922 s; internal fitting plus export 1,528.453 s | `fit-timing.json`, `fixed-fit/report.json` |
| Fitted review generation | 80.044 s (1 min 20.044 s), exit 0; 360/360 coverage | `review-timing.json`, `fixed-fit/review/summary.json` |
| Inference launch through final fitted-review completion | 7,427.932 s (2 h 03 min 47.932 s), including stage handling; asset preparation and final QA excluded | `pipeline-summary.json` |
| Final structural, visual and browser QA | Passed; original indices, finite parameters/geometry, camera consistency, both viewers and status light/dark mobile layout | `validation.json`; `work/human-research-20261006/browser-qa/qa-report.json` |

The recorded per-frame sum includes 53.140 seconds spent on the three frames
during the earlier smoke run; the 357 newly computed frames total 5,715.356
seconds. It is not the wall time of this new process. Raw review generation
overlapped full fitting, so adding all stage durations would overstate elapsed
time. This is inference with pretrained weights and parameter fitting, not
model training.

The raw viewer is
`runs/human-ep16-cpu-full/review-raw/index.html`. The completed fitted viewer is
`runs/human-ep16-cpu-full/fixed-fit/review/index.html`. Both have all 360 JPEGs,
an H.264 comparison video, a contact sheet and continuous-frame diagnostics.
The current full-run server uses port **8766**:
[raw review](http://127.0.0.1:8766/review-raw/) and
[fitted review](http://127.0.0.1:8766/fixed-fit/review/).
Port 8765 serves the historical three-frame experiment. To start the full-run
preview later, use a separate terminal:

```powershell
& work\human-runtime\venv\Scripts\python.exe -m http.server 8766 --bind 127.0.0.1 --directory runs\human-ep16-cpu-full
```

The completed full fitting command is below. A reproduction needs a fresh
output directory; the measured result must remain intact.

```powershell
& work\human-runtime\venv\Scripts\python.exe tools/finalize_human.py --run runs/human-ep16-cpu-full --output runs/human-ep16-cpu-full/fixed-fit --device cpu --method fit --threads 1 --model-batch 8 --keyframes 5 --iterations 2 --pose-workers 4
```

The shared identity solve is sequential; four workers then refit independent
poses against that frozen identity. Each worker uses the original per-frame
initializer, including the accepted fitted keyframe poses, and results retain
source order. Torch intra-op and inter-op thread counts are both one. On the
three verified Track 1 smoke predictions, the canonical one-worker and
four-worker paths produced bit-for-bit identical pose, scales, shape, vertex
errors and reconstructed vertices. The six-job frozen-identity benchmark also
left all 55 named model parameter/buffer hashes unchanged. These checks establish
execution parity on that input, not full-clip reconstruction quality. The full
run passed its separate finite-value, coverage, camera and sampled visual checks.
Parity, memory and timing evidence is in
`work/human-research-20261006/pose_threads_canonical_parity.json`,
`pose_threads_benchmark.json` and `pose_threads_reproduction.md`.

The final saved model differs from SAM's raw mesh by a mean vertex distance of
**0.483 mm**, with the largest frame mean **0.932 mm** at frame 308. The original
native-parameter round trip has mean error 0.000329 mm. These are prediction
fidelity and conversion checks, not error against the actual person. The shared
solve and each pose refit used two iterations; this is a limited-budget fit to
SAM predictions, not a converged image/temporal/contact optimization.

Sampled overlays at the first, middle, last and high-change frames broadly align
the body with the input. The owner subsequently observed clear finger twitching;
the code and trajectory audit confirms the independent fit retains upstream hand
instability. Motion quality has not passed acceptance. Hidden fingers, exact grip,
metric depth and 3D contact remain unverified. The raw viewer uses SAM70 joints;
the fitted viewer uses native
MHR127, so their aggregate temporal cards cannot be compared directly. Joint 0
is at zero depth in this native output and cannot be projected; the other 126
joints are inside the image throughout. Adjacent-frame diagnostics have 359
step samples and 358 second-difference samples. Missing initial differences at
frames 0 and 1 are expected, not missing reconstruction.

The review renderer now batches face normals, shading and pixel coordinates,
retaining every face and the existing depth order, camera, geometry and drawing
rules. Nine real-frame comparisons and independent float32/float64 boundary,
degenerate and clipped geometry cases were exactly pixel-identical; existing
review tests and the full 214-test allowlist passed. The draw-only benchmark
averaged 2.337 to 0.210 seconds per frame (11.1 times faster); final review timing
includes video encoding and I/O. Original raw-review renderer SHA-256:
`9a2278d60583801cc4f04957d24d288d5e3d5f55b94ec1bf9c04bba26dac62f9`;
optimized fitted-review renderer:
`856f5c8231900fba1ba1a15f0b33031c76719ca1ea395e8db04cc58080bf3932`.
Proof and exact diff are under `work/human-research-20261006/` in
`validate_review_vectorization.json`, `validate_review_vectorization_analytic.json`
and `review_vectorization.diff`; actual runtime provenance is saved in the run.

`fit-source-provenance.json` records the actual files read when fitting began,
including uncommitted source. Its SHA-256 values are:

| Fit input | SHA-256 |
|---|---|
| `src/v2hoi/human_result.py` | `d2074d2c139c996bca9abeb4fbb585d63f76a17bfae59ae456968e2ab8a4e5c2` |
| `tools/finalize_human.py` | `8231ad5ccd24abe2464af6199846d3b17acaa719c32c212993d27f84f272eb66` |
| `third_party/track1-mhr-converter/mesh_to_mhr_params.py` | `c799ad612fca19620563fcb93bf61e5a4adad0a04251482358746b5f27f8a52e` |
| `weights/human/mhr_model.pt` | `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc` |

`manifest.json` additionally records the SAM checkpoint, runtime source manifest,
video and camera configuration; the full manifest hash is
`a49945ac66e5e1a1da00111f7aa065abae80d2a1f7868108f04a662396fb79cd`.
For the research cutoff and next reconstruction improvements, see
[Human strategy as of 2026-10-06](human-strategy-20261006.md).

## Share the completed result

The run is stored locally because inference, fitting and preview generation ran
on this workstation. `127.0.0.1:8766` is this computer's loopback preview, not a
remotely accessible publication. Sharing that URL alone will not share the files.

The view-only handoff is
`runs/human-ep16-cpu-full/share/human-ep16-cpu-full-review-20261009.zip`
(91,998,144 bytes, approximately 92 MB). It contains the complete fitted review,
all 360 frame images, comparison video, contact sheet, diagnostics, native NPZ,
fit report and provenance/validation records. All archived files passed CRC and
SHA-256 comparison with their sources. The adjacent `.zip.sha256` file records
the archive checksum. Existing run artifacts were preserved.

Send the ZIP through the team's chosen file channel. The recipient extracts the
whole archive and opens `review/index.html`; no model installation or original
video is needed for this saved overlay viewer. Keep `review/frames` beside the
HTML. Alternatively, send only `fixed-fit/review/comparison.mp4` for easy playback.
Absolute paths shown in Details are provenance strings, not required local paths.

The NPZ is the compact Human parameter result, not a video or standalone mesh.
Researchers need the matching native MHR model and unit/axis conventions to
generate and manipulate its 3D geometry. Model weights are not included in this
view-only package. This package does not contain object reconstruction and is
not a complete challenge submission. No external upload or publication was made.

In the current viewer, the cards are per-frame diagnostics. For example, frame
284 at 9.467 seconds has 2.388 px mean step, 3.424 px/frame squared second
difference and 5.037 m median predicted camera-z depth. The second difference
uses frames 282, 283 and 284, centered on 283. Distances are measured in the
original 1536-by-1152 image before preview resizing. Neither motion diagnostic
is a reprojection error or an official accuracy score. Median depth includes all
finite joint depths and is not a measured distance. In-image coverage likewise
does not establish alignment with the actual person.

## 1. Check the prepared installation

Open PowerShell in `J:\video-to-hoi`:

```powershell
Set-Location J:\video-to-hoi
powershell -ExecutionPolicy Bypass -File tools\run_human_episode16.ps1 -Stage Check
```

This rechecks the selected model and input hashes and imports the runtime.
The execution-policy flag applies to this PowerShell process, not the machine.

The installation is already prepared. To reproduce it on a fresh checkout with
a working bootstrap Python, use:

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_human_windows.ps1
```

This downloads only pinned Body/MHR model assets, explicit source files and the
selected Track 1 video/metadata. It does not install WSL or Docker, switch Git
branches, start inference, download other tracks, or modify system Python.
Allow at least 30 GB of disk headroom for packages, model parts and extraction.
The exact installed environment is in `requirements-windows.lock.txt`; the direct
dependency intent is in `requirements-windows.txt`.

## 2. Inspect the completed partial experiment

The smoke manifest is `runs/human-ep16-cpu-smoke/manifest.json`; `state.json`
must show `completed_frames: 3` and `all_video_frames_complete: false`.
The decoded frame indices in `raw/frame_indices.npy` are exactly `[0,120,240]`.

The fixed result is
`runs/human-ep16-cpu-smoke/fixed-fit/episode_000016_human.npz`.
Its `report.json` records two shared-identity iterations and two pose iterations,
one CPU thread and model batch eight. Fitting took 89.359 seconds; fitting plus
native export took 90.640 seconds (104.860 seconds including process startup).
Mean vertex change from the original prediction was 0.493 mm, versus 1.896 mm
for the median-identity initializer. The per-frame changes were 0.728, 0.366 and
0.386 mm. These compare two predictions and are **not reconstruction accuracy**.

To reproduce this fitting step into a fresh directory after the raw run exists:

```powershell
& work\human-runtime\venv\Scripts\python.exe tools/finalize_human.py --run runs/human-ep16-cpu-smoke --output runs/human-ep16-cpu-smoke/fixed-fit-repeat --device cpu --threads 1 --model-batch 8 --keyframes 3 --iterations 2 --allow-partial
```

The raw review is `runs/human-ep16-cpu-smoke/review-raw/index.html`.
The fixed-identity review is `runs/human-ep16-cpu-smoke/fixed-fit/review/index.html`.
Open either in a normal browser. The HTML, JPEGs and video remain together; do
not copy only the HTML file. Move the slider to frames 0, 120 and 240. All other
frames are explicitly labelled missing, rather than interpolated or presented as
successful inference. The contact sheet prioritizes frames that have predictions.

The fitted viewer reports **126/127** joints inside the image: native joint
index 0 is exactly `[0,0,0]` in these saved model outputs and cannot be projected
at zero depth. All other 126 joints are positive-depth and inside the image on
the three sampled frames. This viewer uses native MHR127 joints, while the raw
viewer uses SAM70; their joint counts and skeleton definitions are different.
The review note records the numeric observation without assigning an unverified
anatomical name to joint 0.

The earlier smoke preview served the run artifacts at
`http://127.0.0.1:8765/fixed-fit/review/`; the raw review is at `/review-raw/`.
If that preview is no longer running, start it in a separate terminal:

```powershell
& work\human-runtime\venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1 --directory runs\human-ep16-cpu-smoke
```

Open `http://127.0.0.1:8765/fixed-fit/review/`. Press Ctrl+C in that terminal to
stop a preview you started. The server exposes this run's artifacts only on
localhost, not the repository or the network.

## 3. Verify GPU model inference after releasing resources

The full CPU run does not establish GPU memory use, throughput or fitting time.
No GPU-model smoke run or full-video GPU benchmark has been measured here.
Save work and close resource-heavy applications yourself before a GPU experiment.
Check usage with `nvidia-smi`. Installed CUDA support does not establish that the
complete model fits alongside other GPU programs. Start with a small test:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_human_episode16.ps1 -Stage Smoke -Device cuda -SmokeFrames 3
```

This creates a separate `runs/human-ep16-cuda-smoke/`. It never replaces the CPU
evidence. Inspect its state and review before starting the full video:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_human_episode16.ps1 -Stage ReviewRaw -Device cuda -Smoke
```

An out-of-memory error is a failed experiment, not an instruction to lower
resolution, drop hands, or silently substitute a smaller model. The state file
records failures. The CPU alternative uses `-Device cpu`; its measured full raw
timing is above.

## 4. Process all 360 frames

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_human_episode16.ps1 -Stage Infer -Device cuda
```

This processes the original video frames without temporal subsampling. Outputs
are written atomically per frame. Rerun the same command to resume; different
source/model/camera settings are rejected in an existing run. A complete run's
state must show 360 frames and `all_video_frames_complete: true`.

The first prototype uses the reviewed fixed box `[440,360,850,1000]` and a shared
**assumed 60-degree horizontal field of view**. It does not estimate camera
calibration, masks or depth. This manual initialization is visible in the manifest.
It is sufficient to exercise real inference, but global depth and geometry need
further validation; an overlay alone cannot prove their accuracy.

## 5. Fit shared identity and export

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_human_episode16.ps1 -Stage Fit -Device cuda
```

SAM is no longer resident during this separate process. The fitter jointly
optimizes one identity and skeletal scale on selected frames, then refits each
pose through the official MHR model. It uses the raw predictions as targets, not
any challenge reference. The default model batch is one to limit memory.

The official finite-difference solver evaluates the articulated MHR model many
times per iteration. Inference timings alone do not predict fitting time.
On this workstation, the 521-by-521 shared-identity solve stalled with eight
CPU threads, while the same single-thread path completed the solve in under
0.016 seconds. The finalizer therefore defaults to **one CPU thread**; CUDA
fitting retains eight host threads. SAM inference still uses eight CPU threads.
This changes the execution configuration, not the fitting algorithm. Do not
override the CPU fitter to eight threads on this machine. Direct library callers
should set `torch.set_num_threads(1)` before a CPU fit too.

`--pose-workers 4 --threads 1` is a separate, optional CPU execution setting
for independent pose refits after the shared solve. It does not give the dense
solver four intra-op threads. The finalizer defaults to one pose worker and
rejects multiple workers for CUDA or median-only mode. Use the direct full CPU
command above for the measured run settings; the general wrapper's defaults are
not that command.

Expected Human output:

```text
runs/human-ep16-cuda-full/fixed-fit/episode_000016_human.npz
    pose          (360,136)
    scales        (68,)
    shape         (45,)
    frame_indices (360,) = 0..359
```

Identity and scales are single vectors by construction. The output is the
**Human portion** of the official schema. It cannot yet be uploaded as a complete
challenge submission: object geometry/poses, other episodes and the official
Parquet packer remain separate work. The internal legacy 133/108/28 fields are
not used as a substitute for native 136/68/45 parameters.

`report.json` distinguishes change from the raw prediction from official
accuracy. Inspect both the mean and worst-frame change. A very large fitting
change deserves visual review even if the arrays pass all structural checks.
`-FitMethod median` is an explicitly labelled initial baseline without pose
refitting, not an equivalent replacement for the default fitter.

## 6. Check the result visually and structurally

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_human_episode16.ps1 -Stage ReviewRaw -Device cuda
powershell -ExecutionPolicy Bypass -File tools\run_human_episode16.ps1 -Stage Review -Device cuda
```

Review directories are immutable: the tool refuses to overwrite existing review
artifacts. Choose a new output directory with `tools/review_human.py` for another
render. Its return code 2 means missing/out-of-range frames or decode coverage
failure; labelled smoke wrappers report expected partial coverage explicitly.

Inspect these items:

1. **Coverage:** 360 decoded and predicted frames, no missing or non-finite arrays.
2. **Body:** head, shoulders, elbows, wrists, knees and ankles stay on the person.
   Tight clothing/skin and loose garments need different silhouette expectations.
3. **Hands:** correct left/right identity, sensible wrist direction, fingers remain
   near the observed grasp; zoom source frames 120 and 240 and adjacent frames.
4. **Occlusion:** hidden limbs do not suddenly flip; the body does not pass through
   the table or acquire implausible limb lengths. These are visual diagnostics.
5. **Motion:** play at source speed and step frame by frame; look for root jumps,
   foot sliding, finger flicker, body-size changes and lost contact.
6. **Before/after:** compare raw and fixed results at identical frame IDs. Shared
   shape/scales should remove size variation without sacrificing observed pose.
7. **Diagnostics:** examine `diagnostics.csv`, `summary.json`, `report.json` and
   worst frames. Reprojection of a model's own 3D/2D outputs checks implementation
   consistency, not independent accuracy. Three sparse frames cannot establish
   temporal quality; no acceleration is computed across gaps.

No ground-truth human accuracy, official ACC/CD/PEN, contact score or complete
object collision check is claimed by this Human-only prototype.

## Tests and team updates

Run the explicit file list in `.github/workflows/tests.yml`; never bare pytest.
It now includes synthetic Human schema, review and restoration guards. Real CPU
smoke evidence is separate from those tests. Technical module-page publication
and republishing the existing online status page still require the Claude
Artifact tool, which is unavailable in this session.
