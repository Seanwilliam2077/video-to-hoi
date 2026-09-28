# Acquired baseline runtime distributions

This handoff collects unmodified third-party distributions for **Linux x86_64,
CPython 3.10, PyTorch 2.5.1 and CUDA 12.4**, from the baseline source recipes at
`nvidia-isaac/video_to_data@33129dd0f2d2dcfd1164d43fd076542660756ed2`.
No package was installed, built or imported for inference on the preparation
computer. No model execution, server deployment or Track 2 input was used.

## What the files establish

- `runtime-acquisition.json` lists the selected distributions, exact versions,
  original download URLs, byte sizes, SHA-256 hashes and redistribution status.
- `SHA256SUMS` records the selected local files. It includes entries withheld
  from the public archive, so it is an inventory rather than a claim that every
  listed file is included in that archive.
- The public archive contains only items marked `redistribution_verified: true`,
  plus supplemental source and license material identified in its manifest.
- Original wheel and source archives are unmodified. Their embedded LICENSE,
  NOTICE and third-party attribution files are retained.
- `fetch_runtime_distributions.py` retrieves the exact original distributions
  and verifies their hashes; it never installs or runs their code. This lets a
  recipient acquire items that cannot be publicly mirrored here directly from
  the upstream distributor, subject to the upstream terms.

## Separate module environments

The upstream project uses separate containers. Do **not** install this entire
inventory into one environment: it aggregates several recipes and some pins or
source revisions intentionally differ between modules.

| Module | Pinned source recipe | Important explicit constraints |
|---|---|---|
| Video masks | `reconstruction/modules/v2d_sam2/docker/Dockerfile` | SAM2 source commit from `assets.lock.json`; torch 2.5.1, torchvision 0.20.1 |
| Object reconstruction | `reconstruction/modules/v2d_sam3d/docker/Dockerfile` | Kaolin 0.18.0 for torch 2.5.1/cu124, cumm-cu124, spconv-cu124, flash-attn 2.7.4.post1, pyrender 0.1.45, pyglet 2.1.15; MoGe1 has its own source and utils3d revision |
| Human reconstruction | `reconstruction/modules/v2d_sam3d_body/docker/Dockerfile` | py-soma-x 0.2.1, networkx 3.2.1, PyTorch3D v0.7.9 source; native MHR route |
| Joint human/object baseline | `reconstruction/modules/v2d_cari4d/docker/Dockerfile` | numpy 1.26.3, torch 2.5.1, torchvision 0.20.1, transformers 5.3.0, imgaug 0.4.0, TensorRT 10.7.0.post1, MoGe2 and FoundationPose native extensions |

The collection pins previously unpinned downloads as an acquisition snapshot.
These selections are **not a tested environment lock**. OpenCV distributions are
limited to `<4.12` here to retain NumPy 1.26.3 compatibility. Lightning packages
are collected at 2.5.6, rather than selecting releases requiring newer Torch.
The full upstream SAM3D Objects development/training requirements are not used:
its baseline container deliberately installs that source with `--no-deps` and
selects inference dependencies separately.

## Known work required on the server

1. Prepare an explicit Python 3.10 Linux environment. The recorded upstream
   PyTorch image digest does not establish its interpreter version; the official
   PyTorch v2.5.1 Dockerfile defaults to Python 3.11. These cp310 wheels must not be
   installed into an unchanged Python 3.11 image. Image layers and the complete
   APT package closure are not included.
2. Select the module's own recipe and sources; resolve any remaining dependency
   constraints before installing. Source-only packages require a remote build.
3. Compile PyTorch3D, nvdiffrast, SAM2, FoundationPose and applicable attention or
   splatting extensions against the selected Torch/CUDA installation.
4. Confirm the actual NVIDIA driver and GPU architecture. The upstream recipes
   target compute capabilities 8.0, 8.6, 8.9 and 9.0. This cu124 handoff does not
   certify RTX PRO 5000 / Blackwell support.
5. Run import, CUDA and baseline checks on allowed Track 1 input, then record the
   built image digest and actual package lock. No such runtime result exists yet.

## Redistribution conditions

- NVIDIA proprietary CUDA/cuDNN/TensorRT development/runtime wheels are acquired
  locally but not publicly mirrored where standalone redistribution rights have
  not been established. Their official origins and checksums remain in inventory.
- The `smplx` package's embedded license prohibits redistribution without separate
  permission; it is excluded. This also does not grant access to SMPL-family model
  assets, which are a separate permission and data-provenance question.
- Native binaries with unresolved copyleft corresponding-source coverage are
  excluded from this public mirror. Their upstream download records are retained.
- Kaolin retains Apache-2.0 and the NVIDIA Source Code License; NSCL-covered
  portions are for non-commercial research/evaluation use.
- A package's presence never authorizes downloading any bundled challenge data.
  Only Track 1 challenge inputs and independently created synthetic fixtures are
  allowed throughout this project.
