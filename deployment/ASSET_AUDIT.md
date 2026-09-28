# Asset availability audit — 2026-09-28

> **Historical snapshot.** This audit predates the
> [published completion Release](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/assets-completion-20260928). The current Release has eleven
> archives (fifteen parts, 16,873,320,178 bytes) and five companion files;
> all 20 attachments passed public size, digest and access checks. It adds the
> filtered baseline and host packages, four Module 1 models, partial human/physics
> sources and separate CPython 3.10/3.11/3.12 runtime acquisition packages.
> The original 12 baseline model entries are now available across three Releases.
> This does not complete the GPU runtime: manual or restricted model weights,
> SMPL-family access, withheld distributions, legacy candidate environments,
> base images, APT packages, native builds, adapters and server validation remain.
> See the [current handoff guide](COMPLETION_HANDOFF.md). The observations below
> remain evidence of their original audit date, not a current publication inventory.

**The full dependency/model handoff is not complete.** The original baseline's
12 required model entries are present locally and pass integrity checks. Only
five of those entries are published in this repository's GitHub Releases.
The expanded first-three-video candidates and a complete GPU runtime remain
incomplete. A Git clone or GitHub source ZIP does not contain model binaries.

## Verification scope

The audit inspected `main` at `c50c475`, declared local receipts and the live
GitHub Releases API. See the [machine-readable record](asset-audit-20260928.json).

- All 24 source/model receipts' 7,930 paths exist and have their recorded sizes.
- All **12 required model entries (45 files, 23,731,473,555 bytes)** passed fresh
  offline size/SHA-256 checks with `fetch_assets.py verify --group models`.
  Some upstream sources do not publish independent checksums; those checks
  establish consistency with the recorded download, not independent authenticity.
- The 28 Linux CPython 3.10 host/bootstrap wheels and their ZIP passed fresh
  size/SHA-256 checks against the wheelhouse manifest.
- All **three Releases / 21 attachments** remain uploaded. Their GitHub-reported
  sizes and SHA-256 digests match the tracked release records. Remote model
  binaries were not downloaded again in this audit.
- The audited Git tree contains no model checkpoints, wheel binaries or ZIPs.
- No models were executed, no packages installed and no server deployed.

Source trees were checked using receipt paths and filesystem metadata, not a
fresh hash of bundled examples. They include upstream demo videos and meshes;
their presence is not authorization to use them. Only declared generic model
files were read for model verification. Before a new source handoff, exclude
bundled example data unless its provenance is verified. All Track 2 content
and derivatives remain prohibited. This audit does not certify an old source
archive as an approved dataset.

## Local versus GitHub

| Scope | Local state | Git / GitHub state |
| --- | --- | --- |
| Project code, inventories, restore/check tools | Present | Committed on `main` |
| 12 upstream source entries | 7,885 recorded files present; sizes match | URLs/revisions tracked; downloaded trees are not vendored in Git |
| Seven public baseline model entries | Fresh hash checks passed | Upstream links tracked; binaries not published in this repository's Releases |
| FoundationPose scorer + refiner | Owner downloaded manually; fresh hash checks passed | [Add-on release](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/foundationpose-addon-20260927) |
| CARI4D, SAM3D Body, SAM3D Objects | Fresh hash checks passed | [Model release](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/hf-models-20260927), 11 parts plus four companions |
| 28 host/bootstrap wheels | Fresh hash checks passed | Locks/manifest tracked; binary ZIP remains local |
| GPU runtime and system/native dependencies | Incomplete; no validated runtime | Recipes/inventory only |
| Expanded comparison candidates | Partially prepared | References/recipes tracked, not all weights or adapters |
| First-three-video kit | Prepared | [Published](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/first3-benchmark-20260927); contains Track 1 videos and preparation files, **no model weights or measured rankings** |

Two missing uploads exist in the owner's local `C:/Video TO HOI/dist/`:

| File | Bytes | Contents / limit |
| --- | ---: | --- |
| `video-to-hoi-public-assets-20260927.zip` | 5,898,264,359 | Historical base: 12 source entries and seven public model entries. Not published; exclude unverified bundled example data from a future handoff. |
| `linux-cp310-wheelhouse.zip` | 119,277,331 | 28 host/bootstrap wheels. Not published; excludes GPU/model runtime packages. |

The seven public model entries are SAM2, CARI4D MoGe2, Objects MoGe1,
CARI4D DINOv2 ViT-B/ViT-S and Objects DINOv2 register ViT-L/ViT-B, totaling
5,477,409,574 bytes. Published weights are **Release attachments**, not Git blobs.

## Runtime dependencies still needed

[upstream-containers.lock.json](requirements/upstream-containers.lock.json)
inventories source recipes; it is not a built-image or GPU wheel lock.
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` has no pinned digest. The wheelhouse
does not supply PyTorch/CUDA, model-container packages, APT system packages or
built CUDA extensions.

Examples include Kaolin, gsplat, spconv/cumm, FlashAttention, TensorRT and
compiled FoundationPose/PyTorch3D/nvdiffrast components. Downloaded source does
not establish built binaries or compatibility. Unpinned upstream Git/PyPI
references and the APT snapshot also remain unresolved. The target server
configuration is needed before building and validating the complete runtime.

## Expanded first-three-video comparison

The 12-entry baseline count does not cover every candidate in
[benchmarks/first3](../benchmarks/first3/README.md).

| Candidate | Current gap |
| --- | --- |
| MoGe2 | Weights exist; comparison source revision `74fbce054ebed49800de42d0ad0e83495065719a` is not among the two downloaded baseline MoGe revisions. |
| SAM2, SAM3D Body, CARI4D | Selected weights exist; runtime validation and required Track 1-derived masks/mesh inputs remain pending. |
| MoGe3 | `model.pt` (1,481,333,394 bytes) is declared but not downloaded; comparison source/environment also needed. |
| SAM3 | Separate model access/weights and adapter are unverified/unprepared; SAM3D approval does not establish SAM3 access. |
| YOLOE | No verified receipt or publication for model/text-encoder files; no execution adapter. |
| GVHMR, WHAM | Model chains, SMPL-family assets, environments and MHR conversion remain unprepared. |
| SmoothNet, HTD-Refine, PhysPT | Weights, environments, MHR integration and real upstream trajectories remain unprepared. |

`C:/Video TO HOI/yoloe-11s-seg.pt` (27,803,986 bytes) and
`C:/Video TO HOI/mobileclip_blt.ts` (599,764,649 bytes) are untracked. Only names,
sizes and Git state were inspected. Provenance/integrity are unverified, so
these files do not count as ready or published.

Optional standalone Body MoGe2-B and SOMA-X export entries are absent. They are
not required by the selected initial native-MHR baseline.

## Remaining work

1. Prepare a provenance-reviewed public-source/model handoff and publish it
   together with the verified host wheelhouse, retaining licenses and receipts.
   Exclude unverified upstream demo data.
2. Acquire/verify expanded candidate assets and finish their adapters. Codex
   must not access Google Drive; permitted manual downloads remain the owner's task.
3. When the server and entry point are available, prepare compatible GPU/runtime
   packages and validate the two real baselines there. Deployment remains deferred.

The local status-page source is updated. Claude team pages were not republished:
no Artifact editing tool is available in this session.
