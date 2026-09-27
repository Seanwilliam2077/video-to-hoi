# Preparation status — 2026-09-27

This is a download and handoff task. No model was installed or executed on the workstation, and no server deployment or SSH connection was attempted. The two real Track 1 baselines remain pending.

## Access

- The Hugging Face login is valid. Exact checkpoint probes for `nvidia/cari4d_commercial`, `facebook/sam-3d-body-dinov3`, and `facebook/sam-3d-objects` returned HTTP 403. Repository approval and the token's gated-repository read scope need checking. Tokens are never part of Git or a transfer ZIP.
- Enterprise iOA blocked Codex's automated Google Drive access. The user downloaded both FoundationPose folders manually; their four required files and two receipts passed an independent offline size/SHA-256 check. `fetch_assets.py` still does not contact Drive. The scorer folder is 190,230,167 bytes and the refiner folder is 68,220,817 bytes. These are locally observed hashes, not independently published upstream checksums.
- Optional standalone Body MoGe2-B and SOMA-X export assets are not selected for the initial native MHR CARI4D baseline.

## Prepared scope

The inventory contains 12 source entries, 12 required model entries, and 2 optional model entries. The following files were actually downloaded and verified:

| Scope | Complete | Downloaded bytes | Contents |
| --- | ---: | ---: | --- |
| Source trees | 12 / 12 | 888,509,283 | Official toolkit, SAM2, SAM3D Objects, both DINO source caches, two MoGe versions, PyTorch3D, nvdiffrast, pybind11, Utils3D |
| Required model assets | 9 / 12 | 5,735,860,558 | Seven previously prepared public models plus two user-downloaded FoundationPose folders; includes 80 bytes of pinned HF cache refs |
| Host/bootstrap wheels | 28 / 28 | 119,263,919 | Linux x86_64 CPython 3.10 packages; excludes PyTorch/CUDA and model-container dependencies |

All 21 completed source/model receipts passed per-file size and SHA-256 checks (7,902 files). The full asset verification still reports three missing required assets: the gated CARI4D, SAM3D Body, and SAM3D Objects models. Their previous HTTP 403 results have not been reprobed. This does not certify a runnable baseline. Optional assets were not downloaded.

The machine-readable record is [preparation-report.json](preparation-report.json). [receipts/](receipts/) contains the public-model and user-manual FoundationPose receipts plus wheelhouse hashes. Where upstream did not publish a content checksum, these record the actual downloaded bytes rather than claiming an independent upstream checksum match.

## Transfer files

The following artifacts are generated under `dist/` on the preparation machine and are transferred separately from Git:

| ZIP | Contents | Checksum companion |
| --- | --- | --- |
| `video-to-hoi-foundationpose-addon-20260927.zip` | Four FoundationPose files, two portable receipts, current project code, progress page, docs and verification records | Same filename plus `.sha256` |
| `video-to-hoi-public-assets-20260927.zip` | Preparation snapshot `6dc5903` plus all 12 source trees, 7 public models, and their portable `.receipts/` | Same filename plus `.sha256` |
| `linux-cp310-wheelhouse.zip` | 28 locked host/bootstrap wheels and their manifest | `linux-cp310-wheelhouse.sha256` |

The existing public-assets ZIP contains only the original seven model entries. The separate `video-to-hoi-foundationpose-addon-20260927.zip` carries the two newly verified FoundationPose folders, their receipts, and the current code and documentation. Extract the original public-assets ZIP into a fresh directory first, then extract the add-on over it. Preserve the folder structure and verify the completed non-gated scope without network access:

```bash
python tools/fetch_assets.py verify --root . --group sources
python tools/fetch_assets.py verify --root . --ids sam2 cari4d-moge2 objects-moge1 cari4d-dinov2-vitb14 cari4d-dinov2-vits14 objects-dinov2-vitl14-reg4 objects-dinov2-vitb14-reg4 foundationpose-scorer foundationpose-refiner
```

On the existing public-assets ZIP alone, the second command still reports the two missing FoundationPose assets. After the add-on is overlaid, the two commands verify all 12 sources and nine completed required model entries. A full `--group all` check still reports the three missing gated models. A successful partial check must not be presented as complete baseline readiness. The manual FoundationPose command and `--register-only` option are in [README.md](README.md).

Git contains the preparation tools, version locks, source locations, and this status report. Large model binaries, downloaded third-party trees, wheel files, and ZIPs live outside Git under `artifacts/` and `dist/`. Do not interpret GitHub's Download ZIP as an archive containing model weights.

The host dependency lock targets Linux x86_64 with CPython 3.10. It is separate from the model containers. PyTorch/CUDA wheels, native builds, system packages, and container image digests remain dependent on the future machine configuration.

## Verification and handoff

- The downloader and ZIP packager passed 43 synthetic tests for receipt hashes, incomplete downloads, archive paths, internal link materialization, blocked automated providers, and user-manual FoundationPose receipts. They do not execute model code. The full explicit CI allowlist also passed locally: 88 synthetic tests.
- The [initial GitHub CI run](https://github.com/Seanwilliam2077/video-to-hoi/actions/runs/36295376997) passed 74 tests. Subsequent changes add packaging and manual-download cases; the current local allowlist result is recorded above. CI uses an explicit synthetic allowlist. `test_score.py`, challenge datasets, and legacy Track 2 score assertions are excluded.
- Download receipts contain immutable source revisions and per-file SHA-256/size. The existing asset ZIP preserves the first 19 verified receipts; the add-on includes the two FoundationPose receipts. The receiver can run `python tools/fetch_assets.py verify --root . --group sources` after extraction. A full `--group all` check must remain incomplete until the three gated weights are acquired.
- Upstream Git archive links are materialized as ordinary files for ZIP transfer. Pass `--skip_weight_download` to the upstream CARI4D entry after the authorized weight setup; its original downloader expects Git checkout metadata that a source ZIP does not contain.
- The compact project ZIP excludes the optional Track 1 video gallery and downloaded videos. The full Git repository retains the gallery page. Track 1 inputs must be acquired separately using the restricted dataset downloader.

See [README.md](README.md) for preparation commands and [MODEL_ASSETS.md](MODEL_ASSETS.md) for the exact model/cache layout and remaining compatibility checks.
