# Track 1 assessment, 2026-09-30

This is a historical assessment. The later
[current Track 1 requirements](track1-requirements.md) govern acceptance;
older external format observations below do not replace R1–R6.

Follow-up: the later [local Human walkthrough](human-local.md) records installation
and real partial CPU inference completed after this initial assessment. The
readiness observations below describe the checkout before that follow-up.

This assessment covers the public competition rules and current submission code,
the `main` checkout at `6a53e84`, and the actual Windows workstation. It does not
report a reconstruction experiment or an official leaderboard score.

## Immediate conclusions

1. The official submission kit is now public. The repository's earlier statement
   that the format and metric implementation are unavailable is out of date.
2. The repository implements an integration scaffold and extensive acquisition,
   packaging, and verification tools. All registered reconstruction defaults still
   use fake or pass-through stages. No real Human backend is integrated.
3. SAM 3D Body with native MHR is the recommended Human mainline. Fixed identity
   and skeletal scales, temporal fitting, hand reconstruction, and consistent
   human-object geometry are the highest-priority extensions. This recommendation
   is based on interface fit and engineering risk, not measured candidate scores.
4. This workstation can support development and potentially small Human inference
   experiments. Its current software and available GPU memory do not establish a
   working full reconstruction runtime.

## Current official Track 1 contract

Sources: [challenge homepage](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/)
and the linked [submission kit](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip).
Only selected Track 1 documentation, metric code, and necessary shared source were
retrieved from the kit. Sample/reference arrays, model assets, and other tracks'
payloads were not retrieved. The inspected code is participant code with reference
and embedded model assets omitted; publishing it does not publish the reference.

The final submission is one Parquet file sent to all five Track 1 Kaggle
competitions. The official packer expects per-episode arrays:

| Field | Shape | Meaning |
| --- | --- | --- |
| `pose` | `[T,136]` | Native MHR model parameters, including root and articulated pose |
| `scales` | `[68]` | One skeletal-scale vector per episode |
| `shape` | `[45]` | One identity vector per episode |
| `object_rotation` | `[T,3,3]` | Object rotation |
| `object_translation` | `[T,3]` | Object translation |
| `object_scale` | scalar | Positive object scale |

An object triangle mesh, exact public sample row/frame identities, and one code
commit URL ending in `/commit/<sha>` are also required. Use the official packer
for mesh budgets and row layout. Native MHR root translation uses model units;
do not copy a metre-valued translation into those parameters without conversion.

The current internal Human contract's `133 body / 108 hand / 28 scale` fields are
not equivalent to this `136 pose / 68 scales / 45 shape` submission schema.
SOMA-X is not a required Track 1 submission intermediate.

### Metric definitions that supersede old assumptions

| Metric | Current participant-code definition |
| --- | --- |
| CD-H | Symmetric human-surface Chamfer distance, in centimetres |
| CD-O | Symmetric posed-object-surface Chamfer distance, in centimetres |
| ACC-H | Error relative to the reference second frame differences of 22 body joints, in centimetres per frame squared |
| ACC-O | Error relative to the reference second frame differences of object translation, in centimetres per frame squared |
| PEN | Mean depth of reconstructed hand points inside the submitted object's own mesh, converted to reference scale |

All five are lower-is-better. Chamfer sums the two directional mean Euclidean
distances; it is neither the squared-distance convention nor half of that sum.
Acceleration is reference-relative, not simply the magnitude of the predicted
trajectory's acceleration. Only contiguous scored frames form differences; the
implementation does not multiply by the video frame rate squared.

The scorer fits one similarity transform from corresponding human vertices in the
first **scored** frame, then applies it across the episode. That frame need not be
video frame zero. The object's initial error remains relevant. The PEN implementation
uses 512 hand points, not a whole-body maximum penetration statistic.

The homepage specifies equal weighting of accuracy and physical plausibility.
The retrieved sources do not establish the final cross-metric normalization or
aggregate ranking formula. Do not assume equal weights for all five metrics or
derive an official score from the repository's retired legacy scorer.
Each metric averages equally across episodes, rather than giving longer clips
more weight. The new reference-relative acceleration implementation contradicts
the September 26 organizer reply recorded in `docs/design.md`; reconfirm this
discrepancy with the organizers, while implementing against the current public code.

The deadline is 2026-11-04 17:00 EST, equivalent to **2026-11-05 06:00 Asia/Shanghai**.
Keep the repository's Track 1-only data policy throughout acquisition, debugging,
validation, and submission preparation.

## Repository findings

The audit inventoried 185 tracked files and inspected the main runtime, approved
tests, documentation, benchmark recipes, and deployment inventories. Prohibited
legacy scorer values and data-dependent legacy tests were excluded from validation.

- `inputs`: fake masks, depth, and camera parameters.
- `human`: fake rest-pose body, zero parameters, and fixed depth scale.
- `objects`: placeholder cube.
- `motion`: fixed object position.
- `refine`: pass-through copies.
- `export`: real serialization, but to the internal format rather than the new
  official submission schema.

Useful implemented work includes stage orchestration, contracts, geometry helpers,
Track 1 downloading, provenance fields, asset receipts, release restore tools,
runtime distribution assembly, and benchmark preparation/record validation.
`tools/benchmark_moge_video.py` is a real standalone per-frame depth adapter, but
its GPU execution is unverified and it is not integrated as an inputs backend.

The initial checkout contained no weights, videos, upstream source trees, runtime
environment, or reconstruction runs. Earlier acquisition receipts and published
releases do not imply those files are present on this workstation. Published Linux
wheelhouses do not install into native Windows. Alternative Human/physics source
archives are not complete model environments and do not contain their checkpoints.

The first-three-video kit is entirely hula-hoop footage and has no measured
candidate ranking. Use the planned diverse Track 1 set `0,6,9,16,24` for Human
development, then all 30 episodes for integration. Episodes 16 and 12 remain useful
initial full-pipeline acceptance targets.

## Workstation and completed local setup

Observed on 2026-09-30:

| Component | Observation |
| --- | --- |
| OS | Windows 11 Pro, build 26200, x64 |
| CPU | AMD Ryzen Threadripper PRO 3945WX, 12 cores / 24 threads |
| RAM | About 128 GiB installed; about 15 GiB free during initial inspection |
| GPU | RTX 3080, 10,240 MiB, compute capability 8.6 |
| GPU use | About 7.2-7.3 GiB occupied during inspection |
| Driver | 610.88; NVIDIA-SMI displays CUDA compatibility 13.3 |
| Storage | About 2,709 GiB free on J: during inspection |
| Linux runtime | WSL not installed; Docker unavailable on PATH |
| Existing system Python | 3.14 |

The CUDA driver API successfully initialized, enumerated the device, and created
a CUDA context. This is a driver check, not a PyTorch or model inference test.
The NVIDIA-SMI CUDA number is not proof of a locally installed CUDA toolkit and
does not dictate the CUDA version of a model environment.

An isolated **Python 3.12.14 CPU development environment** was created at
`J:\video-to-hoi\.venv`, and `.[dev]` was installed. It uses the bundled Python
runtime without changing the system Python or closing user applications.

- All 13 test files in `.github/workflows/tests.yml` ran: **161 passed in 9.34 s**.
- `python -m pip check`: no broken requirements.
- Runner and Track 1 downloader help commands passed.
- No model weights, challenge videos, GPU frameworks, WSL, or Docker were installed.
- No reconstruction inference or Kaggle upload was performed.

Use the environment without activation:

```powershell
& J:\video-to-hoi\.venv\Scripts\python.exe -m v2hoi.run --help
```

The installed dependency snapshot is in
`work/assessment-20260930/windows-dev-freeze.txt`. It describes this lightweight
development environment; it is not a Human inference lock. The developer guide's
copied test command omits `test_handoff_restore.py`; the CI list remains canonical.

## Human implementation recommendation

Start with [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) and retain
its native MHR model parameters. Use one shared camera model and full frame indices.
Estimate one identity and skeletal-scale vector from reliable frames, then refit
per-frame pose/root movement. Review the first scored frame, hands, reprojection,
occlusions, and conversion correctness. Apply temporal and contact refinement while
preserving image evidence; low self-acceleration alone is not the competition goal.

The [MHR conversion tools](https://github.com/facebookresearch/MHR/tree/main/tools/mhr_smpl_conversion)
provide SMPL/SMPL-X to MHR fitting, but conversion is an optimization with error,
not field renaming. GVHMR is a valuable optional body/root-motion expert; its output
does not supply detailed hand articulation. PromptHMR video also has hand placeholders
and restrictive non-commercial research terms. WHAM/TRAM/4D Humans add environment
and conversion work without closing the native detailed-hand requirement. HaMeR
may help difficult hands after body integration; MANO and model licenses still apply.

Use official MHR CARI4D as a later joint human-object candidate when Track 1-derived
masks and a correctly scaled object mesh are available. It is not a directly
comparable RGB-only Human model. Preserve MHR as the authority for submission; if
SOMA-X is used internally, verify conversion error and regenerate it after final
MHR changes. Do not refine only SOMA-X and submit stale MHR parameters.

For inference, use a separate Linux/WSL2 Python 3.11 environment or the pinned
per-module container. The recorded Body container uses Torch 2.5.1 / CUDA 12.4,
PyTorch3D 0.7.9, and py-soma-x 0.2.1; its architecture list includes 8.6. This is
source-declared compatibility, not a verified local build. SAM 3D Body publishes
no minimum VRAM guarantee establishing that 10 GiB is enough. Single-person,
batch-one inference is worth testing after freeing resources and establishing Linux.
For the complete unmodified workflow, SAM 3D Objects specifies Linux and at least
32 GB GPU memory; a 48 GB Linux host remains the practical recommendation.

## Next acceptance gates

1. Align the internal contract/exporter with the newly published native MHR schema.
2. Restore only the necessary approved model assets and create the isolated Human
   Linux runtime; verify CUDA, native imports, and one Track 1 frame.
3. Run a short clip, then complete episodes, measuring actual memory and time.
4. Compare raw Body, fixed shape/scales, temporal fitting, and contact refinement
   on the same Track 1 frames. Review failure cases and fidelity, not just smoothness.
5. Integrate real object results, run all episodes, validate the official Parquet,
   and obtain organizer scores. Keep code revision and run provenance with every run.

No score or victory guarantee is inferred from this plan.

## Team-page publication

This file supplies the technical update for the Platform and Human module pages.
The local status-page source was refreshed with current rule availability and the
limited scope of local setup. The session has no Claude Artifact publishing tool;
the module-page databases and existing online status URL have **not** been updated.
The owner must republish to the existing URLs, preserving their database-backed
content and shared navigation. No new online artifact was created.
