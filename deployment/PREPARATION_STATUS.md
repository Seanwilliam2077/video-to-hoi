# Preparation status — 2026-09-27

This is a download and handoff task. No model was installed or executed on the workstation, and no server deployment or SSH connection was attempted. The two real Track 1 baselines remain pending.

## Access

- The Hugging Face login is valid. Exact checkpoint probes for `nvidia/cari4d_commercial`, `facebook/sam-3d-body-dinov3`, and `facebook/sam-3d-objects` returned HTTP 403. Repository approval and the token's gated-repository read scope need checking. Tokens are never part of Git or a transfer ZIP.
- FoundationPose's two Google Drive folders are blocked by enterprise iOA policy. The tools reject them before any network request. An IT-approved delivery method is required.
- Optional standalone Body MoGe2-B and SOMA-X export assets are not selected for the initial native MHR CARI4D baseline.

## Prepared scope

The inventory contains 12 source entries, 12 required model entries, and 2 optional model entries. Download progress and transfer artifacts are being recorded separately; inventory presence is not proof that the corresponding files exist.

Git contains the preparation tools, version locks, source locations, and this status report. Large model binaries, downloaded third-party trees, wheel files, and ZIPs live outside Git under `artifacts/` and `dist/`. Do not interpret GitHub's Download ZIP as an archive containing model weights.

The host dependency lock targets Linux x86_64 with CPython 3.10. It is separate from the model containers. PyTorch/CUDA wheels, native builds, system packages, and container image digests remain dependent on the future machine configuration.

## Verification and handoff

- The downloader and ZIP packager have synthetic tests for receipt hashes, incomplete downloads, archive paths, internal link materialization, and blocked providers. They do not execute model code.
- CI uses an explicit synthetic test allowlist. `test_score.py`, challenge datasets, and legacy Track 2 score assertions are excluded.
- Download receipts contain immutable source revisions and per-file SHA-256/size. The asset ZIP preserves verified receipts; the receiver can run `python tools/fetch_assets.py verify --root . --group sources` after extraction. A full `--group all` check must remain incomplete until the missing weights are supplied through approved channels.
- Upstream Git archive links are materialized as ordinary files for ZIP transfer. Pass `--skip_weight_download` to the upstream CARI4D entry after the authorized weight setup; its original downloader expects Git checkout metadata that a source ZIP does not contain.
- The compact project ZIP excludes the optional Track 1 video gallery and downloaded videos. The full Git repository retains the gallery page. Track 1 inputs must be acquired separately using the restricted dataset downloader.

See [README.md](README.md) for preparation commands and [MODEL_ASSETS.md](MODEL_ASSETS.md) for the exact model/cache layout and remaining compatibility checks.
