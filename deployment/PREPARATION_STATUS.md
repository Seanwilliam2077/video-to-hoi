# Preparation status — 2026-09-27

This is a download and GitHub handoff task. The owner requested that deployment remain deferred. No model was installed or executed on the workstation, and no server deployment or SSH connection was attempted. The two real Track 1 baselines remain pending.

## Access and redistribution

- Access to `nvidia/cari4d_commercial`, `facebook/sam-3d-body-dinov3`, and `facebook/sam-3d-objects` is approved. Fresh checks and complete downloads resolved the earlier HTTP 403 failures. The 28 selected files passed size/SHA-256 verification. Credentials are excluded from Git and release archives.
- Enterprise iOA blocked Codex's automated Google Drive access. The user downloaded both FoundationPose folders manually; their four files and two receipts passed an independent offline size/SHA-256 check. `fetch_assets.py` still does not contact Drive. The scorer folder is 190,230,167 bytes and the refiner folder is 68,220,817 bytes. These are locally observed hashes, not independently published upstream checksums.
- The three new model packages may be redistributed under their reviewed NVIDIA/SAM terms. [model-redistribution.json](model-redistribution.json) binds that review to the exact asset specification and [full license materials](licenses/). [MODEL_RELEASES.md](MODEL_RELEASES.md) describes the packaging and receiver procedure. Project code licensing does not replace model licenses.
- `assets.lock.json` remains the historical source/identity snapshot so existing receipt hashes stay valid. Its old access observations are not the current download status.
- Optional standalone Body MoGe2-B and SOMA-X export assets are not selected for the initial native MHR CARI4D baseline.

## Prepared scope

The inventory contains 12 source entries, 12 required model entries, and 2 optional model entries. The required local scope is now complete:

| Scope | Complete | Downloaded bytes | Contents |
| --- | ---: | ---: | --- |
| Source trees | 12 / 12 | 888,509,283 | Official toolkit, SAM2, SAM3D Objects, both DINO source caches, two MoGe versions, PyTorch3D, nvdiffrast, pybind11, Utils3D |
| Required model assets | 12 / 12 | 23,731,473,555 | Seven public entries, two user-downloaded FoundationPose folders, and three newly approved models; includes 80 bytes of pinned HF cache refs |
| Host/bootstrap wheels | 28 / 28 | 119,263,919 | Linux x86_64 CPython 3.10 packages; excludes PyTorch/CUDA and model-container dependencies |

All **24 source/model receipts passed per-file size and SHA-256 checks (7,930 files)**. The three newly completed entries are:

| Model | Verified files | Verified bytes |
| --- | ---: | ---: |
| CARI4D | 3 | 2,084,494,424 |
| SAM3D Body | 5 | 2,805,255,047 |
| SAM3D Objects | 20 | 13,105,863,526 |
| New model subtotal | 28 | 17,995,612,997 |

The Objects pipeline and its configurations were inspected without model execution. All 12 referenced checkpoint/config paths are selected; MoGe1 and the DINOv2 register weights are already covered. No additional model file was identified. This confirms file coverage, not a runnable environment or successful inference. Optional assets were not downloaded.

The machine-readable preparation record is [preparation-report.json](preparation-report.json); the corresponding [receipts/](receipts/) identify the checked files. Where upstream did not publish a content checksum, receipts record the actual downloaded bytes rather than an independent upstream checksum match.

## Transfer files

Downloads are stored outside normal Git history under `artifacts/` and `dist/`. GitHub's repository Download ZIP does not include model binaries.

| Handoff | Contents | Publication state |
| --- | --- | --- |
| `video-to-hoi-public-assets-20260927.zip` | Preparation snapshot `6dc5903`, all 12 source trees, seven public models, and 19 portable receipts | Existing local transfer file; not included in either add-on release |
| `video-to-hoi-foundationpose-addon-20260927.zip` | Four FoundationPose files, two receipts, verified code snapshot `5205666`, documentation and verification records | Published with SHA-256/JSON companions |
| `hf-models-20260927` model release | Three reviewed model archives transported as checksum-verified parts, their receipts, licenses and restore tool | Upload in progress; release remains a draft until all uploads are verified |
| `linux-cp310-wheelhouse.zip` | 28 locked host/bootstrap wheels and their manifest | Existing separate local transfer file |

The [FoundationPose add-on](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/foundationpose-addon-20260927) is already published. Its [exact ZIP](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/foundationpose-addon-20260927/video-to-hoi-foundationpose-addon-20260927.zip) is 239,097,418 bytes. GitHub reported the same SHA-256 as the local ZIP: `722995a8c32f26eaefb544d0407fffb9b1c6108a76f5ae5cfe68861dc160c52b`. The [release record](releases/foundationpose-addon-20260927.json) stores asset IDs, URLs, sizes, and server-side digests.

The new model packages are being uploaded to the [model release](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/hf-models-20260927). That link becomes publicly downloadable after the draft is published. Local model verification is complete; remote publication is a separate unfinished step. Follow [MODEL_RELEASES.md](MODEL_RELEASES.md) to download the manifest/restore tool, check every part, and reconstruct the three model ZIPs after publication.

For a complete handoff, extract the separately supplied public-assets base into a fresh directory, overlay the published FoundationPose add-on, then extract the three reconstructed model ZIPs into the same root. Keep the directory layout and receipts. Verify without network access:

```bash
python tools/fetch_assets.py verify --root . --group all
```

With all three layers present, this covers 12 source entries and 12 required models. The old base/add-on pair alone still lacks the three newly approved models. A successful partial check does not establish a complete handoff; a complete file check does not establish baseline readiness. The manual FoundationPose command and `--register-only` option remain documented in [README.md](README.md).

The host dependency lock targets Linux x86_64 with CPython 3.10 and is separate from the model containers. PyTorch/CUDA wheels, native builds, system packages, container digests and compatibility checks still depend on the future machine configuration. The L20/A10/RTX PRO 5000 comparison remains in [platform.md](../docs/platform.md); this task does not select or deploy a server.

## Verification and remaining work

- The current explicit CI allowlist passed 104 synthetic tests locally, including 59 downloader/bundle tests (16 exercise the reviewed model release tools). The three real ZIPs and all 11 parts also passed offline size/SHA-256 restoration checks. The [initial GitHub CI run](https://github.com/Seanwilliam2077/video-to-hoi/actions/runs/36295376997) passed 74 tests. These historical results concern preparation tools, not GPU inference. CI uses an explicit synthetic allowlist; `test_score.py`, challenge data and legacy Track 2 assertions are excluded.
- The required local source/model file verification is now complete: 24 receipts, 7,930 files. Release publication still requires completion of uploads and remote size/SHA-256 checks. The model-release tools also validate the reviewed license bytes before packaging.
- Upstream Git archive links are materialized as ordinary files. When deployment is later authorized, pass `--skip_weight_download` to the upstream CARI4D entry after the prepared assets are placed correctly; its original downloader expects Git metadata absent from a source ZIP.
- The compact project ZIP excludes the optional Track 1 video gallery and downloaded videos. The full Git repository retains the gallery page. Track 1 inputs must be acquired separately with the restricted dataset downloader.
- No remote environment installation, model inference, episode-16/episode-12 baseline, or official score has been produced. The current instruction is to finish the GitHub handoff and defer deployment.

See [README.md](README.md) for preparation commands, [MODEL_ASSETS.md](MODEL_ASSETS.md) for model/cache coverage, and [MODEL_RELEASES.md](MODEL_RELEASES.md) for the reviewed release and restore process.
