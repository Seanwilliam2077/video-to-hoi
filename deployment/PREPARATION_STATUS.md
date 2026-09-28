# Preparation status — 2026-09-27

**Current handoff update (2026-09-28):** The
[completion Release](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/assets-completion-20260928) is public. It holds eleven archives transported
as fifteen verified parts totaling 16,873,320,178 bytes, plus five companion
files (20 attachments). All GitHub-reported hashes and unauthenticated public
attachment paths passed the [publication checks](releases/assets-completion-20260928-github.json).
The release adds seven original baseline model entries and twelve filtered source
entries, the Linux CPython 3.10 host wheelhouse, four Module 1 model packages,
MoGe3 dependency sources, partial human/physics source packages, and separate
CPython 3.10/3.11/3.12 runtime acquisition packages. The other five selected
baseline model entries remain in the older FoundationPose and CARI4D/SAM3D Releases.
The explicit synthetic CPU suite passed 161 tests, and [GitHub Actions](https://github.com/Seanwilliam2077/video-to-hoi/actions/runs/36391458592)
passed for publication source commit `e41383c`. These checks establish package
and tool integrity, not a runnable GPU environment. Manual or restricted
human/physics weights, SMPL-family access, withheld distributions, older candidate
environments, base images, APT packages, native builds, adapters and server
validation remain outstanding. See [COMPLETION_HANDOFF.md](COMPLETION_HANDOFF.md).
No server deployment or model execution has occurred.

**Availability rechecked 2026-09-28:** see the [asset audit](ASSET_AUDIT.md).
The 12/12 model count below covers the original local baseline selection only.
The original record below is retained as historical preparation evidence; its
release-status statements predate the published completion handoff.

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
| `public-baseline-runtime-20260928.zip` | Seven public baseline model entries, twelve filtered runtime source entries, licenses and portable receipts; supersedes the historical unfiltered `video-to-hoi-public-assets-20260927.zip` | Published in the completion Release; part and archive hashes checked |
| `video-to-hoi-foundationpose-addon-20260927.zip` | Four FoundationPose files, two receipts, verified code snapshot `5205666`, documentation and verification records | Published with SHA-256/JSON companions |
| `hf-models-20260927` model release | Three reviewed model archives transported as checksum-verified parts, their receipts, licenses and restore tool | Published: 11 model parts and four companion files; public URLs verified |
| `linux-cp310-wheelhouse.zip` | 28 locked Linux CPython 3.10 host/bootstrap wheels and their manifest | Published in the completion Release; part and archive hashes checked |
| `assets-completion-20260928` remaining archives | Four Module 1 candidates, MoGe3 dependency sources, human/physics partial sources and CPython 3.10/3.11/3.12 runtime acquisition | Public with exact manifest and receipts; not a complete GPU environment |

The [FoundationPose add-on](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/foundationpose-addon-20260927) is already published. Its [exact ZIP](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/foundationpose-addon-20260927/video-to-hoi-foundationpose-addon-20260927.zip) is 239,097,418 bytes. GitHub reported the same SHA-256 as the local ZIP: `722995a8c32f26eaefb544d0407fffb9b1c6108a76f5ae5cfe68861dc160c52b`. The [release record](releases/foundationpose-addon-20260927.json) stores asset IDs, URLs, sizes, and server-side digests.

The [model release](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/hf-models-20260927) is published: 11 parts totaling 17,995,770,414 bytes, plus four companion files. All 15 GitHub asset digests matched local SHA-256 values. Public download URLs were checked without authentication: HEAD verified model-part sizes, and GET plus SHA-256 verified all four companions. The [release record](releases/hf-models-20260927.json) stores asset IDs, URLs, digests, and tag commit; the [download manifest](releases/hf-models-20260927-manifest.json) records model revisions, archive hashes, and ordered parts. Follow [MODEL_RELEASES.md](MODEL_RELEASES.md) to download the manifest/restore tool, check every part, and reconstruct the three model ZIPs.

For a complete handoff, extract the separately supplied public-assets base into a fresh directory, overlay the published FoundationPose add-on, then extract the three reconstructed model ZIPs into the same root. Keep the directory layout and receipts. Verify without network access:

```bash
python tools/fetch_assets.py verify --root . --group all
```

With all three layers present, this covers 12 source entries and 12 required models. The old base/add-on pair alone still lacks the three newly approved models. A successful partial check does not establish a complete handoff; a complete file check does not establish baseline readiness. The manual FoundationPose command and `--register-only` option remain documented in [README.md](README.md).

The host dependency lock targets Linux x86_64 with CPython 3.10 and is separate from the model containers. PyTorch/CUDA wheels, native builds, system packages, container digests and compatibility checks still depend on the future machine configuration. The L20/A10/RTX PRO 5000 comparison remains in [platform.md](../docs/platform.md); this task does not select or deploy a server.

## Verification and remaining work

- The current explicit CI allowlist passed 104 synthetic tests locally, including 59 downloader/bundle tests (16 exercise the reviewed model release tools). The three real ZIPs and all 11 parts also passed offline size/SHA-256 restoration checks. The [initial GitHub CI run](https://github.com/Seanwilliam2077/video-to-hoi/actions/runs/36295376997) passed 74 tests. The [current GitHub CI run](https://github.com/Seanwilliam2077/video-to-hoi/actions/runs/36298879194) also passed for release-source commit `65c9025`. These results concern preparation tools, not GPU inference. CI uses an explicit synthetic allowlist; `test_score.py`, challenge data and legacy Track 2 assertions are excluded.
- The required local source/model file verification is now complete: 24 receipts, 7,930 files. Release upload, server digest checks, and public download checks are complete. The model-release tools also validate the reviewed license bytes before packaging.
- Upstream Git archive links are materialized as ordinary files. When deployment is later authorized, pass `--skip_weight_download` to the upstream CARI4D entry after the prepared assets are placed correctly; its original downloader expects Git metadata absent from a source ZIP.
- The compact project ZIP excludes the optional Track 1 video gallery and downloaded videos. The full Git repository retains the gallery page. Track 1 inputs must be acquired separately with the restricted dataset downloader.
- No remote environment installation, model inference, episode-16/episode-12 baseline, or official score has been produced. The current instruction is to finish the GitHub handoff and defer deployment.

See [README.md](README.md) for preparation commands, [MODEL_ASSETS.md](MODEL_ASSETS.md) for model/cache coverage, and [MODEL_RELEASES.md](MODEL_RELEASES.md) for the reviewed release and restore process.
