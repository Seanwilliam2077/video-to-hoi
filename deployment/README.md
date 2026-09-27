# Deployment preparation (Linux x86_64)

This directory records the software and asset inputs for a future Track 1 run. It is preparation material, not a deployed or validated GPU environment. The owner currently requests downloads and a GitHub handoff only; server deployment and model execution are deferred. The source ZIP can be uploaded manually; no SSH connection has been made. Do not use Track 2 data, assets, trajectories, or reference meshes.

## What is pinned

| File | Scope |
| --- | --- |
| `requirements/requirements.bootstrap.in` and `.txt` | CPython 3.10, Linux x86_64 download/build tooling. `huggingface_hub==1.32.0` is a direct pin; `.txt` pins its resolved Python dependencies. |
| `requirements/requirements.core.lock` | The 26 host runtime packages resolved from this repository's `pyproject.toml` for CPython 3.10 and manylinux2014 x86_64. It excludes optional SOMA-X, PyTorch, and CUDA packages. |
| `requirements/upstream-containers.lock.json` | The upstream `video_to_data` commit and SHA-256 of its SAM2, SAM3D, SAM3D-Body/MHR, and CARI4D Dockerfiles, plus their Git refs and unresolved dependencies. |
| `assets.lock.json` | Historical model/source identities, revisions, access requirements, and destination paths consumed by `tools/fetch_assets.py`; retained unchanged to preserve receipt identity. |
| `model-redistribution.json` and `licenses/` | Redistribution review for the three approved Hugging Face models, bound to their asset-spec hashes and full license/notice bytes. |

The host lock and model containers are **separate environments**. In particular, the host lock resolves NumPy 2.2.6, while the upstream CARI4D container constrains NumPy to 1.26.3. Do not install the host lock into the CARI4D image.

## Package download only

On a networked **Linux x86_64 machine with CPython 3.10, pip, and a matching glibc 2.28 target**, after unpacking the source ZIP, these commands download the pinned host wheels without installing packages or running the pipeline. Use the explicit cross-platform tags below if the download machine differs from that target:

```bash
cd video-to-hoi
python3.10 -m pip download --no-deps --only-binary=:all: \
  --requirement deployment/requirements/requirements.bootstrap.txt \
  --dest wheelhouse/bootstrap
python3.10 -m pip download --no-deps --only-binary=:all: \
  --requirement deployment/requirements/requirements.core.lock \
  --dest wheelhouse/core
```

The lock was resolved with `uv pip compile` for `x86_64-manylinux2014` and Python 3.10. Its versions are fixed, but the lock itself has no wheel hashes. Verify the target's Python minor version and Linux ABI before reusing these wheels. The source ZIP does not contain wheels. A future offline installation would also need compatible wheels for every model container, system packages, base images, Git checkouts, and model files; these are **not** supplied by these host locks.

To cross-download these already resolved packages on Windows, pass the target tags explicitly and **keep `--no-deps`**. Without it, pip may resolve an extra Windows dependency such as `colorama` that is absent from the Linux lock:

```powershell
python -m pip download --no-deps --only-binary=:all: `
  --implementation cp --python-version 310 `
  --abi cp310 --abi abi3 --abi none `
  --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 `
  --requirement deployment/requirements/requirements.bootstrap.txt `
  --requirement deployment/requirements/requirements.core.lock `
  --dest artifacts/linux-cp310-wheelhouse
```

After the download finishes, the following standard-library tool checks that every locked package/version has exactly one compatible CPython 3.10 Linux x86_64 wheel. It rejects non-wheels, unlocked packages, and incompatible tags. It writes a ZIP, a SHA-256 checksum, and a machine-readable JSON sidecar containing every wheel's hash. It does not install or execute wheels, contact the network, or include model/GPU dependencies.

```bash
python tools/prepare_wheelhouse.py
(cd dist && sha256sum -c linux-cp310-wheelhouse.sha256)
```

The output defaults to `dist/linux-cp310-wheelhouse.zip`, `.sha256`, and `.json`; use `--wheelhouse` and `--output` to change paths. These hashes describe the downloaded bytes and do not authenticate them against an upstream release. If an earlier Windows download left **only** an extra `colorama==0.4.6` wheel, `--allow-colorama` permits that specific file but excludes it from the ZIP. The JSON sidecar records its exclusion.

The corresponding metadata-only regeneration commands are:

```bash
uv pip compile deployment/requirements/requirements.bootstrap.in \
  --python-version 3.10 --python-platform x86_64-manylinux2014 \
  --only-binary :all: --no-header --no-annotate --no-python-downloads \
  --output-file deployment/requirements/requirements.bootstrap.txt
uv pip compile pyproject.toml \
  --constraints deployment/requirements/requirements.bootstrap.in \
  --python-version 3.10 --python-platform x86_64-manylinux2014 \
  --only-binary :all: --no-header --no-annotate --no-python-downloads \
  --output-file deployment/requirements/requirements.core.lock
```

## Source ZIP and asset flow

The portable code ZIP can be made locally before the remote machine is known:

```bash
python tools/prepare_bundle.py --root . --output dist/video-to-hoi-code.zip
```

After unpacking it, `plan` needs only the Python standard library. `check` checks remote availability; `fetch` downloads selected permitted source and model assets and writes per-asset receipts; `verify` checks the downloaded files against those receipts. Fetch only assets authorized in `assets.lock.json`; gated models require their own access approval and credentials supplied outside the ZIP. Enterprise iOA blocked Codex's automated access to the two FoundationPose Google Drive folders, so `fetch_assets.py` still leaves those entries untouched. The user obtained both folders manually, and their four required files passed offline receipt checks. They are not in the existing public-assets ZIP; the separate add-on ZIP carries them with current project code.

```bash
python tools/fetch_assets.py plan --group all
# Example restricted to two pinned GitHub sources; no Drive request:
python tools/fetch_assets.py check --ids video-to-data-source sam3d-objects-source
# Once access, disk capacity, and network location are decided:
python tools/fetch_assets.py fetch --root artifacts --ids video-to-data-source sam3d-objects-source
python tools/fetch_assets.py verify --root artifacts --ids video-to-data-source sam3d-objects-source
```

From the extracted project directory on the user's Windows machine (Python 3.10+), the separate FoundationPose downloader can fetch the two official folders and register their file hashes:

```powershell
py -m pip install gdown==5.2.0
py tools/download_foundationpose.py --root "C:/Video TO HOI/artifacts/preparation"
```

If the user has already placed both folders under the target paths in `assets.lock.json`, run the same script with `--register-only` to validate and register those local files without a Drive request. The script checks both `model_best.pth` and `config.yml`, rejects missing/empty files and HTML error pages, and records local size/SHA-256 in `.receipts/`. Upstream SHA-256 values are unavailable: these receipts check later transfer integrity, not upstream authenticity or model inference. Both FoundationPose entries are registered. The three Hugging Face approvals have also been granted, fresh checks succeeded, and all 28 newly selected files were downloaded and verified. The completed required-model count is now **12 / 12**.

The default bundle contains source and manifests only. Verified permitted assets can be included with `--receipts artifacts/.receipts --asset-root artifacts`. The general bundle tool still excludes gated models. The three reviewed models use `tools/prepare_model_release.py`, which requires matching receipts and the separate redistribution policy. See [MODEL_RELEASES.md](MODEL_RELEASES.md) for packaging and restoration; a permission check alone never substitutes for downloaded-file verification.

The existing `video-to-hoi-public-assets-20260927.zip` contains the original seven model entries. The `video-to-hoi-foundationpose-addon-20260927.zip` carries the two newly verified folders, their receipts, and current project code. Extract the original public-assets ZIP first, then overlay the add-on. Neither existing ZIP contains the three newly approved models. Their separately reviewed [release](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/hf-models-20260927) is currently being uploaded as a draft; after publication, restore and extract the three model ZIPs into the same root, then run `python tools/fetch_assets.py verify --root . --group all`. Local preparation already verifies all 24 source/model receipts and 7,930 files. The model scope is 23,731,473,555 bytes including cache refs. This handoff does not install or run the models.

## GPU and native boundaries

The four official recipes at `video_to_data` commit `33129dd0f2d2dcfd1164d43fd076542660756ed2` start from the **tag** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`; its image digest and inherited Python minor version are not locked. Their native builds specify CUDA compute capabilities `8.0`, `8.6`, `8.9`, and `9.0`. This is a **legacy CUDA 12.4 reference for Ampere, Ada, and Hopper only**, pending actual build and run checks. No Blackwell profile is selected or validated. CARI4D also installs TensorRT packages labeled CUDA 12.6 in its CUDA 12.4 base, so its exact ABI combination needs remote validation.

The recipes install Linux APT libraries, compile PyTorch3D, nvdiffrast, and FoundationPose extensions, and use some unpinned Git and PyPI dependencies. `upstream-containers.lock.json` identifies those gaps and records observed Git commit candidates. Pre-fetching a commit does **not** make an unmodified upstream Dockerfile use it. Dockerfile hashes identify the reviewed recipes but do not turn a mutable image tag, Git branch, or APT index into a reproducible binary. Choose the server GPU, driver, OS image, Python version, and CUDA profile before attempting model installation or inference. No remote execution has been performed.
