# Native Windows CARI4D MHR runtime

This profile runs the pinned official CARI4D MHR sources through the team's
`python -m v2hoi.native` entry point. The legacy v1 fake-stage interface remains
available for its existing contract tests; it does not feed this real backend.

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
- `wheelhouse/windows`: seven existing locally built wheels, including the
  Windows CPython 3.12 PyTorch3D and nvdiffrast binaries.
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
and its host dependencies. SAM2's optional CUDA extension is disabled; its mask
network runs on CUDA normally. The segmentation helper disables optional hole filling.

## Real Track 1 entry point

The episode 16 preparation consists of a complete 360-frame SAM2 mask sequence,
MoGe2 depth/intrinsics, standalone SAM3D MHR initialization, and depth scale
alignment against that estimated human. The object is a convex reconstruction
of visible first-frame depth, closed across unseen surfaces. Its backside is
inferred and remains a quality limitation. This is an image-derived initial
baseline, not a measured reference mesh or a final mesh reconstruction method.

After preparation, run the coupled native backend:

```powershell
python -m v2hoi.native --dataset-root third_party/CARI4D_MHR_Win/data/v2d-challenge/track_1 --episode 16 --preparation runs/windows-episode16/prepared/preparation.json --output runs/my-episode16
python -m v2hoi.native_export --run runs/my-episode16 --assets weights/cari4d/sam3d_body --output runs/my-episode16/export
python tools/windows/validate_official_export.py --episodes runs/my-episode16/export --episode 16
```

The entry point validates the video against Track 1 metadata and hashes the
prepared masks/mesh. It refuses to reuse a run after its inputs or source change.
Full logs and original-frame identities are stored with the run. Native human
and object states come from the same coupled upstream bundle.

The default `--cuda-memory-fraction 0.75` limits each stage's PyTorch allocator.
Desktop/driver memory and non-PyTorch CUDA allocations are outside that limit.
`gpu-memory.json` records whole-device samples, and `.stages/*.memory.json`
records exact PyTorch allocation/reservation peaks. The FoundationPose render,
refiner, independent scorer features and full-resolution depth conversion use
16-item batches. Cross-candidate attention and all original video frames remain
intact. Do not run two GPU jobs together on the 16 GB workstation.

For standalone commands, the same budget wrapper is available:

```powershell
python -m v2hoi.cuda_budget --fraction .75 --report runs/export-memory.json -- -m v2hoi.native_export --run runs/my-episode16 --assets weights/cari4d/sam3d_body --output runs/my-episode16/export
python -m v2hoi.cuda_budget --fraction .75 --report runs/render-memory.json -- tools/windows/render_export.py --run runs/my-episode16 --export runs/my-episode16/export
```

Export first verifies the exact wrapper-to-native MHR mapping. If identity varies
over time, the official converter's joint/per-frame LM solvers fit constant shape
and skeletal scales, warm-started from the actual native parameters. The fit's
geometric error is reported separately from the exact conversion check.

The official packer check reads only this episode's rows from the public Track 1
sample. With no `--commit`, it writes `FORMAT_CHECK_ONLY.parquet` using the
organizer's explicit placeholder URL. This is a local format check and must not
be uploaded. A complete submission needs every required episode and an actual
published commit identifying the generating code. No official score is computed.

## Model acquisition and minimal runtime

`python tools/windows/acquire_weights.py` retrieves the team's approved CARI4D
and SAM3D Body Release archives, verifies their hashes, and restores their
receipts/licenses. MoGe2 and FoundationPose weights are reused from the migrated
downloads. SAM2 uses the source revision in `sources.json` and the checkpoint
SHA-256 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`.

Training, TensorRT, Kaolin, SAM 3D Objects and UI packages are not added for this
baseline. The required penetration objective uses existing Open3D occupancy
and differentiable PyTorch3D triangle distance when Kaolin is unavailable.
DINOv3 imports only its backbone, and CoCoNet uses its checkpoint's complete
DINOv2 state without redundant standalone weight downloads. FoundationPose
renders/refines small batches and chunks only independent scorer features;
cross-candidate attention still sees every candidate together.

`python tools/windows/fetch_official_code.py` restores only the pinned official
code/docs and public Track 1 sample row schema in `official-kit.json`. It reuses
matching local files. Remote reads require exact byte ranges; the complete kit
and reference assets are never downloaded. Updated sources require explicit
review rather than silently replacing pinned code.

`python tools/windows/check_native_kernels.py` verifies analytic penetration
values/gradients and chunked-versus-original scoring. Run this separately from
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

## Verified native baseline (2026-10-11)

The [machine-readable record](episode16-result.json) pins the actual evidence.
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
- The integrated profile contains 162 packages; `uv pip check` passes. All 231
  tests in the explicit team CI allowlist pass on Windows.
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
