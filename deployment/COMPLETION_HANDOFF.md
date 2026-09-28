# Expanded asset handoff — 2026-09-28

The owner requested acquisition and GitHub publication of the remaining assets.
**Deployment and model execution remain deferred.** Acquisition is not a claim
that every candidate is runnable. Some official model routes require manual
downloads, and some software/model licenses do not permit a public mirror.

## Packages

The new handoff uses the `assets-completion-20260928` Release. The checked
publication record and final manifest are stored under `deployment/releases/`.
During preparation, treat a missing publication record as incomplete publication.

| Archive group | Contents | Scope |
| --- | --- | --- |
| Public baseline | Seven public baseline model entries and twelve filtered source entries | Original model bytes, full notices, per-file receipts; no upstream dataset/demo payloads |
| Host wheelhouse | 28 Linux x86_64 CPython 3.10 bootstrap/host wheels | Separate from model runtime environments |
| MoGe3 | Verified `model.pt` and MoGe source at `74fbce054ebed49800de42d0ad0e83495065719a` | Source also supplies the MoGe2 comparison revision |
| SAM3 | Verified video checkpoint and pinned source | Distinct from SAM3D Body/Objects; no output adapter or runtime validation yet |
| YOLOE | Verified official YOLOE-v8L segmentation checkpoint and text-path source | AGPL terms and corresponding source included; optional vendored SAM2 visual-prompt path omitted |
| MobileCLIP | Verified original Apple MobileCLIP-B(LT) checkpoint | Separate research-only package, full current Apple agreement and required attribution |
| MoGe3 dependency sources | Exact `utils3d-moge`, `pipeline` and `FlexGEMM` revisions | Full MIT terms and build metadata; separate NumPy 2 environment and any CUDA build still required |
| Human/physics sources | GVHMR, WHAM, SmoothNet, HTD-Refine, PhysPT and three fixed submodules | Source only; no model weights; four files with unresolved/restrictive terms excluded |
| Baseline runtime acquisition | Public package distributions and supplemental build sources that can be mirrored | CPython 3.10 and a separate CPython 3.11 companion for Torch 2.5.1/cu124; incomplete offline runtime |
| SAM3 / MoGe3 runtime acquisition | 106 public distributions across two independent CPython 3.12 profiles and a small helper supplement | Torch 2.10/cu128; SAM3 uses NumPy 1.26.4 and MoGe3 uses NumPy 2.5.3; 18 distributions acquired locally are withheld from the public mirror |

The existing [FoundationPose add-on](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/foundationpose-addon-20260927)
and [CARI4D / SAM3D model release](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/hf-models-20260927)
remain required for the baseline. The [first-three-video kit](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/first3-benchmark-20260927)
contains Track 1 inputs and preparation scripts; it contains no model weights or measured rankings.

The baseline profiles do not satisfy every expanded candidate. Separate SAM3
CPython 3.12/cu128 and MoGe3 NumPy 2 distributions have also been acquired. Read the
[exact candidate environment differences](EXPANDED_RUNTIME_GAPS.md), the
[baseline runtime inventory guide](RUNTIME_ACQUISITION.md), and the
[CPython 3.11 assembly guide](RUNTIME_CP311.md), plus the
[SAM3/MoGe3 profile guide](RUNTIME_MODERN.md). Legacy human/physics runtimes
remain incomplete; their old pinned stacks have not been silently replaced by
the baseline environment.

## Download and restore

Use the current project checkout or its source ZIP, including both restore scripts:

```bash
python tools/restore_handoff.py deployment/releases/assets-completion-20260928-manifest.json \
  --parts-dir downloads/completion-parts --output-dir downloads/completion-zips --download
```

This downloads only the manifest's exact GitHub Release attachments and verifies
each part and reconstructed ZIP. It does not extract, install, or load a model.
The previous `restore_model_release.py` command remains for the older HF release;
use `restore_handoff.py` for this new tag. Do not concatenate parts from different
versions or use superseded local archives.

Extract the ZIPs into separate roots:

| Archives | Suggested destination | Important resulting paths |
| --- | --- | --- |
| Public baseline, FoundationPose and older HF model ZIPs | `combined-assets/` | `third_party/`, `weights/`, `.receipts/` |
| MoGe3, SAM3, YOLOE, MobileCLIP | `expanded-platform/` | `models/`, `sources/`, `licenses/`, `receipts/` |
| MoGe3 dependency sources | `moge3-dependencies/` | `sources/utils3d-moge3`, `sources/pipeline-moge3`, `sources/flexgemm-moge3` |
| Host wheelhouse | `host-wheelhouse/` | `wheels/`, `requirements/` |
| Human/physics sources | `expanded-human-physics/` | Source packages, full terms and manual acquisition guide; follow the archive's top directory |
| Runtime distributions | `runtime-downloads/` | Follow its README, manifest and public-file inventory |

The modern runtime archive has a top directory named
`runtime-modern-cp312-cu128/`. Use its profile assembler to prepare SAM3 and
MoGe3 in separate destinations. Do not install every wheel in that archive into
one environment. Allow roughly 110 GB for all baseline/new release parts, ZIPs
and extracted files to coexist; 150 GB free leaves preparation headroom before
datasets, container images, compiler caches or experiment outputs.

Keep the current project checkout separate from the baseline asset root; the old
FoundationPose add-on also contains a historical project snapshot. Do not replace
current project scripts with those older files.

After all baseline layers are present, check their declared files:

```bash
python tools/fetch_assets.py verify --root combined-assets --group all
```

Filtered source receipts enumerate the retained files, not the entire upstream
checkout. This check validates those declared bytes plus the twelve baseline
model entries. It does not certify optional source paths, compilation or inference.
The source inventory is [public-baseline-handoff.json](public-baseline-handoff.json).
The toolkit selection contains `v2d_cari4d`, `v2d_common`, `v2d_sam2`, `v2d_sam3d`,
`v2d_sam3d_body`, `v2d_foundation_pose`, `v2d_moge`, `v2d_docker` and `v2d_mv`.
Optional upstream pipelines, calibration and object-preparation modules outside
this list are not supplied as a full upstream mirror. Their inputs must never
be substituted with Track 2 assets.

## What cannot be called complete

- **Human/physics checkpoints:** the reviewed official Drive/Dropbox routes
  require the owner's approved manual channel. See [exact manual commands and
  destinations](MANUAL_MODEL_DOWNLOADS.md). Do not bulk-download mixed data/model folders.
- **SMPL-family models and restricted code:** personal access is separate from
  permission to redistribute. No public body-model mirror is supplied. Four
  restricted/unclear source files are excluded; the candidate source packages
  are explicitly incomplete for execution. See [human/physics inventory](expanded-human-physics.json).
- **Native runtime:** downloaded wheels or source distributions are not built
  extensions. Base image layers, APT closure, module compatibility and server
  validation remain pending. The cu124 profile does not establish compatibility
  with RTX PRO 5000 / Blackwell. A single shared environment is not supported.
- **Legacy comparison runtimes:** GVHMR, HTD-Refine, WHAM, SmoothNet and PhysPT
  have different upstream Torch/CUDA/Python pins. Their complete historical
  environments are not acquired. The helper supplement does not replace a
  selected container or a tested compatibility port; see the candidate gap report.
- **Publication limits:** proprietary NVIDIA SDK distributions and packages
  with unresolved redistribution conditions are not silently mirrored. The
  runtime record provides exact official download origins and hashes for the
  recipient to obtain under the applicable terms.
- **Experiments:** SAM3/YOLOE and human/physics adapters, masks/prompts, Track 1
  meshes, MHR conversion, quality evaluators and actual runs are separate work.
  There are no new scores or rankings.

The MobileCLIP package is for non-commercial research under the full Apple
agreement, not for commercial product development. All packages retain their
original terms; this aggregation does not relicense them. Model records:
[expanded-platform.json](expanded-platform.json).

## Checks performed

- Every current archive was restored from the exact manifest parts and matched
  to its archive SHA-256 without extracting or loading a model.
- The explicit synthetic CPU test suite passed **161 tests**. The corrected
  standalone import path also passed on GitHub Actions at commit `ee7ff4b`.
- Independently created tiny fixtures checked both runtime assemblers' profile
  separation, repeated use, overlap rejection, corruption handling and missing
  distribution reports. These are file-transfer checks, not runtime tests.
- Public attachment URLs and final GitHub digests are recorded separately after
  publication in `releases/assets-completion-20260928-github.json`.

Only the 30 Track 1 videos and metadata, their reconstructed artifacts and
independently created synthetic fixtures are allowed. No Track 2 data, derived
scores/statistics, ground-truth geometry, trajectories or unknown-provenance
upstream examples may be loaded, tested or included in a submission.
