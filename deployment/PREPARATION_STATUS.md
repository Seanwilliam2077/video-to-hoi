# Preparation status — 2026-09-27

This is a download and handoff task. No model was installed or executed on the workstation, and no server deployment or SSH connection was attempted. The two real Track 1 baselines remain pending.

## Access

- The Hugging Face login is valid. Exact checkpoint probes for `nvidia/cari4d_commercial`, `facebook/sam-3d-body-dinov3`, and `facebook/sam-3d-objects` returned HTTP 403. Repository approval and the token's gated-repository read scope need checking. Tokens are never part of Git or a transfer ZIP.
- FoundationPose's two Google Drive folders are blocked by enterprise iOA policy. The tools reject them before any network request. An IT-approved delivery method is required.
- Optional standalone Body MoGe2-B and SOMA-X export assets are not selected for the initial native MHR CARI4D baseline.

## Prepared scope

The inventory contains 12 source entries, 12 required model entries, and 2 optional model entries. The following files were actually downloaded and verified:

| Scope | Complete | Downloaded bytes | Contents |
| --- | ---: | ---: | --- |
| Source trees | 12 / 12 | 888,509,283 | Official toolkit, SAM2, SAM3D Objects, both DINO source caches, two MoGe versions, PyTorch3D, nvdiffrast, pybind11, Utils3D |
| Required model assets | 7 / 12 | 5,477,409,574 | SAM2, MoGe2-L, MoGe1-L, and four DINOv2 checkpoints; includes 80 bytes of pinned HF cache refs |
| Host/bootstrap wheels | 28 / 28 | 119,263,919 | Linux x86_64 CPython 3.10 packages; excludes PyTorch/CUDA and model-container dependencies |

All 19 completed source/model receipts passed per-file size and SHA-256 checks (7,898 files). The full asset verification intentionally reports five missing required assets: the three gated models and two organization-blocked FoundationPose folders. It does not certify a runnable baseline. Optional assets were not downloaded.

The machine-readable record is [preparation-report.json](preparation-report.json). [receipts/](receipts/) contains the public-model download receipts and wheelhouse hashes. Where upstream did not publish a content checksum, these record the actual downloaded bytes rather than claiming an independent upstream checksum match.

## Transfer files

The following artifacts are generated under `dist/` on the preparation machine and are transferred separately from Git:

| ZIP | Contents | Checksum companion |
| --- | --- | --- |
| `video-to-hoi-code-latest-20260927.zip` | Project source, updated progress page, preparation tools, docs, locks, and verification records | Same filename plus `.sha256` |
| `video-to-hoi-public-assets-20260927.zip` | Preparation snapshot `6dc5903` plus all 12 source trees, 7 public models, and their portable `.receipts/` | Same filename plus `.sha256` |
| `linux-cp310-wheelhouse.zip` | 28 locked host/bootstrap wheels and their manifest | `linux-cp310-wheelhouse.sha256` |

Extract the public-assets ZIP into a fresh directory, then extract the latest code ZIP over it to bring the project status page and docs up to date. Preserve the folder structure. The later code ZIP has the same asset lock and does not overwrite `weights/`, `third_party/`, or `.receipts/`. Verify the public scope without network access:

```bash
python tools/fetch_assets.py verify --root . --skip-gated
```

This command still reports the two missing FoundationPose assets. To check just the complete sources with a successful exit, use `--group sources`. To check one public model, use `--ids sam2` (or another ID from the report). A successful partial check must not be presented as complete baseline readiness.

Git contains the preparation tools, version locks, source locations, and this status report. Large model binaries, downloaded third-party trees, wheel files, and ZIPs live outside Git under `artifacts/` and `dist/`. Do not interpret GitHub's Download ZIP as an archive containing model weights.

The host dependency lock targets Linux x86_64 with CPython 3.10. It is separate from the model containers. PyTorch/CUDA wheels, native builds, system packages, and container image digests remain dependent on the future machine configuration.

## Verification and handoff

- The downloader and ZIP packager passed 32 synthetic tests for receipt hashes, incomplete downloads, archive paths, internal link materialization, and blocked providers. They do not execute model code.
- The [initial GitHub CI run](https://github.com/Seanwilliam2077/video-to-hoi/actions/runs/36295376997) passed 74 tests. The follow-up adds three packaging cases; see PR #8 for its current check. CI uses an explicit synthetic allowlist. `test_score.py`, challenge datasets, and legacy Track 2 score assertions are excluded.
- Download receipts contain immutable source revisions and per-file SHA-256/size. The asset ZIP preserves verified receipts; the receiver can run `python tools/fetch_assets.py verify --root . --group sources` after extraction. A full `--group all` check must remain incomplete until the missing weights are supplied through approved channels.
- Upstream Git archive links are materialized as ordinary files for ZIP transfer. Pass `--skip_weight_download` to the upstream CARI4D entry after the authorized weight setup; its original downloader expects Git checkout metadata that a source ZIP does not contain.
- The compact project ZIP excludes the optional Track 1 video gallery and downloaded videos. The full Git repository retains the gallery page. Track 1 inputs must be acquired separately using the restricted dataset downloader.

See [README.md](README.md) for preparation commands and [MODEL_ASSETS.md](MODEL_ASSETS.md) for the exact model/cache layout and remaining compatibility checks.
