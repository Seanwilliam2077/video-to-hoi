# Native Windows CARI4D MHR runtime

This profile runs the pinned official CARI4D MHR sources through the team's
`python -m v2hoi.native` entry point. The legacy v1 fake-stage interface remains
available for its existing contract tests; it does not feed this real backend.

The [method fidelity record](algorithm_fidelity.md) distinguishes original
methods, resource changes, the permitted penetration exception and validation limits.

The only active project environment is the real directory `<repository>/.venv`.
Do not put a project environment in a user cache or link it outside the project.
The shared uv package cache and uv-managed base Python interpreters remain shared
tool installations; deleting this project removes its environment, not those tools.

## Layout

- `third_party/CARI4D_MHR_Win`: local copy of the official video_to_data checkout,
  including current Windows changes, downloaded MoGe2, and Track 1 episode 16.
- `third_party/CARI4D_Win`: preserved legacy checkout, Windows changes, weights,
  official demo/training downloads, and previous results. It has no active venv.
  These historical assets are archival and are not inputs to the competition
  reconstruction or evaluation pipeline. Only permitted Track 1 inputs may be
  used for challenge reconstruction.
- `wheelhouse/windows`: preserved locally built wheels, including the
  Windows CPython 3.12 PyTorch3D, nvdiffrast and original SAM2 CUDA binaries.
- `artifacts/windows-migration`: original environment lists, wheel hashes,
  transfer verification, installation logs, and cleanup records.

Large files and local upstream checkouts are ignored by Git. A Git push alone
does not transfer them. `sources.json` and `patches/video_to_data.patch` preserve
the pinned upstream revision and reviewable compatibility changes. The patch's
reverse check passes against the local checkout. Keep the local sources, wheels,
and model directories when transferring this runtime.

## Restore and use

From PowerShell in the repository:

```powershell
./tools/windows/setup.ps1
. ./tools/windows/activate.ps1
. ./tools/windows/native-env.ps1
python tools/windows/check_runtime.py
```

Setup is offline by default. It uses `deployment/windows/pyproject.toml` and its
lock, existing uv caches, preserved wheels, and the copied editable sources.
`-Online` explicitly allows downloads when restoring on another machine.
The local binary wheels target CPython 3.12, Windows x64, and the preserved
PyTorch 2.14.1 CUDA 13.0 stack; other combinations need separate builds.

Use the explicit Windows setup script for this environment. A plain `uv sync`
at the repository root uses the team's separate lightweight host manifest and
can remove backend dependencies. This profile includes the editable team package
and its host dependencies. SAM2 uses its original connected-components CUDA
extension and default hole filling (area 8). Segmentation reads the original MP4
with the NVIDIA entry and Decord; inference frames never use a JPEG cache.

## Real Track 1 entry point

Official preparation requires the complete original-frame SAM2 mask sequence,
MoGe2 depth/intrinsics, official SAM3D Body initialization, human-anchored depth
scale alignment and the NVIDIA SAM 3D Objects generative entry. The native
runner rejects the previous convex/JPEG preparation by default. Explicit
`--allow-experimental-preparation` is available only for historical diagnostics.
Generation must complete and produce its receipt before a new preparation
manifest or a faithful full-run result can be reported.

On this 16 GB workstation, generate with `tools/windows/generate_object.py
--cpu-offload`. This adapter stages strictly loaded, unchanged models between
CPU and CUDA. It inherits the original sampling/decoding methods, preserving
float32 parameter storage, original autocast precision, both 25-step samplers,
guidance settings and all requested output formats. Fully resident loading
exceeded the 75% budget; staged generation passed in 84.7 seconds with an
8.50 GiB allocated peak, producing 176,178 vertices and 352,348 faces.
`tools/windows/assemble_preparation.py` freezes its receipt and segmentation
checksums. The corrected complete run and export checks passed on episode 16;
see [the corrected record](corrected-episode16-result.json).

After preparation, run the coupled native backend:

```powershell
python -m v2hoi.native --dataset-root third_party/CARI4D_MHR_Win/data/v2d-challenge/track_1 --episode 16 --preparation runs/windows-episode16/faithful-preparation/preparation.json --output runs/my-episode16
python -m v2hoi.native_export --run runs/my-episode16 --assets weights/cari4d/sam3d_body --output runs/my-episode16/export
python tools/windows/validate_official_export.py --episodes runs/my-episode16/export --episode 16
```

The entry point validates the video against Track 1 metadata and hashes the
prepared masks/mesh. It refuses to reuse a run after its inputs or source change.
Full logs and original-frame identities are stored with the run. Native human
and object states come from the same coupled upstream bundle.

### Explicit 4096-face inference variant

The owner requested pre-inference mesh reduction after the dense reference run.
Keep the full generated surface and prepare a separate variant:

```powershell
python tools/windows/budget_preparation.py --preparation runs/windows-episode16/faithful-preparation/preparation.json --output runs/windows-episode16/budget4096-preparation
python -m v2hoi.native --dataset-root third_party/CARI4D_MHR_Win/data/v2d-challenge/track_1 --episode 16 --preparation runs/windows-episode16/budget4096-preparation/preparation.json --allow-budgeted-mesh --output runs/windows-episode16/budget4096
```

Run GPU jobs sequentially. This reuses all models, segmentation and the original
generative result; it downloads nothing. The new run still calculates its own
tracking and coupled refinement because their object surface has changed.
The tool uses the reviewed, unmodified organizer `v2dlb.mesh_budget.budget_mesh`,
welding and decimating to both 4096-face and 4096-vertex budgets. It removes only
zero-area padding and unreferenced vertices for the inference GLB. Coordinates
are neither recentered nor rescaled. Preparation and receipts keep hashes and
relative paths to the preserved full mesh, official code and masks.

Episode 16 reduction took 0.67 seconds: 352,348 faces/176,178 vertices became
4,096 active faces/2,052 vertices. A face-count ratio does not establish total
speedup. Pre-inference decimation changes tracking, contact and penetration;
the manifest records this deviation and does not label it the full-surface
official reference. The native entry requires `--allow-budgeted-mesh`; the
default full-mesh behavior remains unchanged. End-to-end variant performance
and reconstruction quality have not yet been validated.

The [variant evidence](mesh-budget-episode16-result.json) includes a benchmark
of the unchanged upstream contact-distance/min/square/backward kernel. With
184,320 seeded synthetic query points shaped as 360 frames of two 256-point
hands, median time was 0.987 seconds for the dense mesh and 0.0117 seconds for
the budget mesh: 84.7x faster **for this kernel only**. The synthetic distances
also changed (0.199 mm mean absolute difference); this is not an equivalence
test, video quality score or measured total pipeline speedup.

```powershell
python -m v2hoi.cuda_budget --fraction .75 --report runs/contact-benchmark-memory.json -- tools/windows/benchmark_mesh_contact.py --preparation runs/windows-episode16/budget4096-preparation/preparation.json --output runs/contact-benchmark.json
```

The default `--cuda-memory-fraction 0.75` limits each stage's PyTorch allocator.
Desktop/driver memory and non-PyTorch CUDA allocations are outside that limit.
`gpu-memory.json` records whole-device samples, and `.stages/*.memory.json`
records exact PyTorch allocation/reservation peaks. FoundationPose uses
render/refiner batches 64/128, the unmodified full scorer forward and a 32-item
independent depth-warp batch. Original 512/1024 batches passed on the historical
tiny convex mesh, but exceeded memory with the new 352,348-face generated mesh;
no surface simplification is used to recover memory in the full-surface reference.
Other pipeline batching defaults are 4, with SAM3D refitting at 64; the corrected
full run passed. Do not run two GPU jobs
together on the 16 GB workstation.

For standalone commands, the same budget wrapper is available:

```powershell
python -m v2hoi.cuda_budget --fraction .75 --report runs/export-memory.json -- -m v2hoi.native_export --run runs/my-episode16 --assets weights/cari4d/sam3d_body --output runs/my-episode16/export
python -m v2hoi.cuda_budget --fraction .75 --report runs/render-memory.json -- tools/windows/render_export.py --run runs/my-episode16 --export runs/my-episode16/export
```

Export first verifies the exact wrapper-to-native MHR mapping. It then passes
the decoded vertices to the full official `mesh_to_mhr_params.convert()` flow,
including its float64 arithmetic, initialization, keyframe selection, iteration
counts, cold/backward refits and shared-identity refinement. Only its supported
model call batch is configured (64). The fit's geometric error is reported
separately from the exact conversion check; no custom native-parameter warm
start, median identity or altered fitting policy is used.

The official packer check reads only this episode's rows from the public Track 1
sample. With no `--commit`, it writes `FORMAT_CHECK_ONLY.parquet` using the
organizer's explicit placeholder URL. This is a local format check and must not
be uploaded. A complete submission needs every required episode and an actual
published commit identifying the generating code. No official score is computed.

The official sample allows 4,096 vertex rows and 4,096 face rows per object.
The unmodified organizer packer welds, decimates and pads the full-resolution
input to those limits. This mandatory packaging operation is separate from
inference geometry: the complete generated surface feeds tracking and contact
optimization in the full-surface reference. The explicit variant above moves
decimation before inference and records that deviation. Validation compares Parquet readback exactly to the official
budget result and separately reports sampled surface change. Reload rendering
can use `--mesh <export>/episode_000016_object_budgeted.glb` to display the
surface actually serialized for the competition, with all original source poses.

## Model acquisition and minimal runtime

`python tools/windows/acquire_weights.py` retrieves the team's approved CARI4D
and SAM3D Body Release archives, verifies their hashes, and restores their
receipts/licenses. MoGe2 and FoundationPose weights are reused from the migrated
downloads. SAM2 uses the source revision in `sources.json` and the checkpoint
SHA-256 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`.

`python tools/windows/acquire_weights.py --asset sam3d-objects --download-workers 4`
restores the team's SAM 3D Objects archive. Completed parts are hash-verified
and reused; partial parts resume with checked byte ranges. `--direct-download`
disables the proxy for that transfer only. Choose by measured connectivity.

Training, TensorRT and unused UI packages are not added for this baseline.
Only required SAM 3D Objects inference imports are installed. Its legacy MoGe
and utils3d sources are scoped to the object-generation process; the coupled
backend retains its own current MoGe2 API in the same project environment.
The required penetration objective uses existing Open3D occupancy
and differentiable PyTorch3D triangle distance when Kaolin is unavailable.
This is an owner-permitted exception whose equivalence to Kaolin has **not**
been established. SAM 3D Objects' separate tensor validation helper preserves
the original Kaolin Python function; it does not replace geometry kernels.
DINOv3 imports only its backbone, and CoCoNet uses its checkpoint's complete
DINOv2 state without redundant standalone weight downloads. FoundationPose
uses the original complete scorer forward with independent render/refine batches.

Unused simplification and texture-baking imports are lazy. Their dependencies
are in `object-postprocess`, excluded from default setup; enable this group only
when requesting those upstream options. Core geometry generation retains all
vertices and faces and requires neither VTK/PyVista nor Gaussian rasterization.

`python tools/windows/fetch_official_code.py` restores only the pinned official
code/docs and public Track 1 sample row schema in `official-kit.json`. It reuses
matching local files. Remote reads require exact byte ranges; the complete kit
and reference assets are never downloaded. Updated sources require explicit
review rather than silently replacing pinned code.

`python tools/windows/check_native_kernels.py` verifies analytic penetration
values/gradients and bit-exact independent depth-warp batching. These checks
do not prove Kaolin equivalence. Run this separately from
the explicit CPU test list in `.github/workflows/tests.yml`.

The legacy checkout's old configuration/log files intentionally retain historical
absolute paths. They are evidence, not the active launch configuration. The new
Windows profile and editable imports must resolve inside this repository.

## Validation limits

Transfer hashes establish preservation. CUDA kernel and gradient checks alone
establish execution of the covered GPU operations. Real inference, exported
geometry, official format readback and submission coverage each require their
own evidence. No synthetic result may be represented as reconstruction. A
single-video result is not a complete competition submission.

## Historical Windows experiment (2026-10-11)

The [machine-readable record](episode16-result.json) pins the actual evidence
and retracts the earlier official-baseline label. This run used a depth convex
hull, JPEG mask input, disabled SAM2 hole filling and a custom identity fit.
It demonstrates Windows execution, **not official baseline fidelity**.
`runs/windows-episode16/full-v5` completed all eight upstream stages for episode
16: 360 original frames at 30 fps, including 300 steps of contact optimization.
The 231 explicitly allowed team tests and independent CUDA gradient/batching
checks pass. Offline environment restoration changes no packages.

- Whole-device sampled peak: 9,495 MiB of 16,303 MiB. Exact maximum PyTorch
  allocated/reserved peaks across stages: 5,646 / 7,262 MiB. Sampling can miss
  short whole-device peaks; allocator statistics exclude other CUDA libraries.
- FoundationPose's full-resolution depth warp formerly batched 128 candidates;
  16-item batches reduced its exact allocated peak from 11.05 GiB at a failed
  75%-budget attempt to 2.77 GiB on the successful run. Independent depth-warp
  and scorer batching checks were bit-identical on their synthetic fixtures.
- Native conversion has a maximum decoded vertex error of 0.00124 mm. Fitting
  constant shape/scales then changes vertices by 0.443 mm mean, 1.471 mm p95,
  and 15.280 mm maximum relative to the backend reconstruction; the worst
  frame's mean is 2.280 mm. These are conversion errors, not accuracy scores.
- The official packer produced 20,231 rows for this episode's 240 scored frames.
  All 360 original poses remain in the source NPZ. Readback preserves native
  parameters exactly and the object surface within zero measured error.
- `export/export-reloaded.mp4` renders all 360 frames from saved native
  parameters through the official MHR decoder and saved object mesh/poses.
- A second FoundationPose pass with exactly the same cached inputs reproduced
  all 360 object poses bit-for-bit. Independently repeating upstream estimation
  changed some inputs and later tracking; whole-pipeline repeatability still
  needs evaluation and must not be inferred from the batching fixtures.

The successful resume reused stages 01-04 after verifying that the sole upstream
source change was FoundationPose's depth-warp batching. `compatibility-resume.json`
preserves the previous inputs and patch. The first uncapped complete run remains
in `full-v4`; failed attempts remain separate evidence. They are not advertised
as successful bounded runs.

Quality remains limited: the object is a convex closure of first-frame depth,
its unseen shape is approximate, and visible hand/object alignment and rotation
drift still need work. This is a runnable single-video baseline, not a finished
reconstruction method, a closed multi-video acceptance gate, or an upload-ready
30-episode submission.

## Corrected complete video (2026-10-11)

[The machine-readable result](corrected-episode16-result.json) records successful
completion of all eight stages for all 360 original frames, complete official
float64 human conversion, organizer format readback and a 360-frame render of
the serialized candidate. The dense 352,348-face surface fed tracking and all
300 requested refinement steps. Refinement took 3,222 seconds, original preview
rendering 1,100 seconds, and complete export 672 seconds. The permitted Kaolin
penetration exception remains unproven equivalent. All 233 explicitly allowed
CPU tests pass; independent CUDA runtime/gradient checks pass.

Whole-device 2-second sampled peak was 12,460 / 16,303 MiB (76.4% including other
processes); exact PyTorch allocated/reserved stage maxima were 8,910 / 9,760 MiB.
The 75% setting limits only the PyTorch allocator, not desktop/driver/other
CUDA libraries. Failed memory attempts remain in the compatibility audit.

Official constant-identity fitting differs from the backend human surface by
0.468 mm mean and 15.358 mm maximum. Packing reduced the object to 4,096 faces
and 2,052 active vertices; 20,000 seeded bidirectional surface samples differ
from the dense inference surface by 0.084 mm mean, 0.594 mm p99 and 1.458 mm
sample maximum. These are representation changes, not reconstruction accuracy.
Native parameters and budgeted mesh rows read back exactly. The reloaded
packed-mesh video is `runs/windows-episode16/faithful-v2/export/export-reloaded-budgeted.mp4`.
Visual inspection still shows an undersized inferred object and uncertain hand
alignment. There is no official score or complete 30-video submission.

### Earlier separate component checks

The official SAM2 entry completed all 360 frames with CUDA hole filling and
original-video decoding. Its first-frame masks differ from the historical
experiment, so historical stages must not be reused under the corrected identity.
The complete official MHR converter also passed on the historical native output:
all 360 frames valid, keyframes 0/120/239/359, mean vertex discrepancy 0.443 mm,
maximum 15.275 mm, about 665 seconds. This establishes converter execution;
it does not make the historical object reconstruction faithful.

Dense geometry also exceeds memory when the original postopt silhouette render
materializes all 360 frames at once. Its independent per-frame raster calls now
use checkpointed batches of 16; the original method body, whole-clip optimization,
300-step schedule and temporal objective are preserved. The source-resume audit
verifies the entire original module AST and every other upstream Python file
before reusing stages 01-06. On independent cubes with occlusion and a partial
last batch, losses match exactly and the largest gradient difference is
8.94e-8. This is a bounded synthetic check, not a whole-pipeline equivalence claim.

The online Claude module/status pages have not been updated: this session has
no Claude Artifact publisher. The technical record is local to this branch.

## Migration and environment history (2026-10-11)

- Preserved 3,786 files/links (43.292 GiB of file contents) from both previous
  directories. Regular files were SHA-256 compared. The MoGe2 snapshot was
  restored as an in-project hardlink to its SHA-256-verified blob.
- Restored 154 runtime packages offline. `uv pip check` passed. The setup script
  was rerun successfully without downloads or package changes.
- CUDA forward/backward checks passed for PyTorch, PyTorch3D KNN and mesh
  rasterization, and nvdiffrast. Editable source locations are inside this repo.
- The historical integrated profile contained 162 packages. The corrected
  minimal profile contains 180; optional texture/simplification packages are
  excluded. `uv pip check` passes. All 231
  tests in the explicit team CI allowlist pass on Windows (233 after the new
  pre-inference budget integrity test).
- Automatic tool approval rejected both a bounded batch cleanup and a single
  old-junction removal, returning only `blocked by policy`. No external
  environment was deleted. The old source directories also remain untouched.

The reviewed cleanup helper preserves Conda, Blender MCP, shared interpreters,
and package caches. It targets only the former CARI4D MHR environment, its old
junction, and seven identified uv cached tool environments plus their indexes:

```powershell
./tools/windows/cleanup-external-environments.ps1           # preview
./tools/windows/cleanup-external-environments.ps1 -Execute  # manual cleanup
```

It refuses unexpected targets, environments with active Python/uv processes, or
an incomplete migration. Run it manually only after reviewing its target list.

One later maintenance command omitted `UV_PROJECT_ENVIRONMENT` and began an
unused environment in `deployment/windows/.venv`. The command was stopped;
all active work uses the root `.venv`. Automatic approval rejected deletion of
that newly created directory with only `blocked by policy`, so it remains
pending manual cleanup. This is the exact in-project target:

```powershell
Remove-Item -LiteralPath 'D:\Projects\video-to-hoi\deployment\windows\.venv' -Recurse -Force
```
