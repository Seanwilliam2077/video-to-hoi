# Model and source inventory

This inventory prepares the official **MHR CARI4D + SAM2 + SAM3D Objects** path for a later Linux deployment. The machine, container build and GPU run are still pending. Model binaries are external assets; Git contains the inventory and preparation tools.

The upstream baseline is [`nvidia-isaac/video_to_data@33129dd`](https://github.com/nvidia-isaac/video_to_data/tree/33129dd0f2d2dcfd1164d43fd076542660756ed2). [`assets.lock.json`](assets.lock.json) records full revisions, exact selected files, byte counts, upstream download evidence, target directories and available SHA-256 hashes. Only pretrained weights, model configuration and source code are listed. Challenge episode inputs are handled separately through the project's **Track 1** downloader.

## Current readiness

Metadata was checked on **2026-09-27**. This is an asset inventory, not a completed model bundle or a successful inference run.

- Hugging Face metadata confirms the seven core/standalone model revisions. The account login is valid, but the three selected gated weight probes—CARI4D, SAM3D Body and SAM3D Objects—returned **HTTP 403**. This may reflect missing repository approval or insufficient repository scope in a fine-grained token; the probe does not distinguish those causes. Their files have not been downloaded for this bundle.
- The two FoundationPose Google Drive sources are **blocked by enterprise iOA policy**. Their URLs are retained solely as provenance. The preparation tool must not check, download, retry, proxy or seek a mirror of these sources in this environment. An approved IT process is needed before those assets can be provided.
- Required model files with known sizes total **23,473,022,491 bytes (23.47 GB / 21.86 GiB)**. This excludes FoundationPose, source archives, Python dependencies, container images, cache duplication and ZIP staging. It is not a complete disk-space estimate.
- No GPU inference or complete offline runtime check has been performed. The inaccessible Objects `pipeline.yaml` still needs inspection after authorized access; the selection conservatively includes all nonempty checkpoint/config variants from that snapshot.

## Required pretrained assets

All targets below are relative to the external asset root. Revisions are abbreviated here; the JSON contains full immutable commits.

| Asset | Version | Selected bytes | Runtime target | Access / license record |
| --- | --- | ---: | --- | --- |
| CARI4D step 200000, Aug 25 run | `nvidia/cari4d_commercial@1f7287a` | 2,084,494,424 | `weights/cari4d/cari4d/2026-08-25-09-35-57/` | Gated; current 403. Model card has no license value; review repository terms. |
| SAM3D Body + MHR model | `facebook/sam-3d-body-dinov3@11aaa34` | 2,805,255,047 | `weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/` | Gated; current 403. SAM License. |
| SAM3D Objects | `facebook/sam-3d-objects@2e73555` | 13,105,863,526 | `weights/sam3d/hf-download/` | Gated; current 403. SAM License. |
| SAM2.1 Hiera Large | `facebook/sam2.1-hiera-large@665f8e2` | 898,107,388 | `weights/sam2/` | Public; Apache-2.0. |
| MoGe2 ViT-L Normal | `Ruicheng/moge-2-vitl-normal@b135031` | 1,323,815,928 | `weights/cari4d/hf_home/hub/` | Public; MIT. |
| MoGe1 ViT-L | `Ruicheng/moge-vitl@ad326bf` | 1,256,823,466 | `weights/sam3d/hf_home/hub/` | Public; MIT. |
| DINOv2 ViT-B/14 | Official checkpoint URL; upstream SHA-256 recorded | 346,378,731 | `weights/cari4d/sam3d_body/torch_home/hub/checkpoints/` | Public; Apache-2.0. |
| DINOv2 ViT-S/14 | Official checkpoint URL; upstream SHA-256 recorded | 88,283,115 | Same CARI4D checkpoint directory | Public; Apache-2.0. |
| DINOv2 ViT-L/14 reg4 | Official checkpoint URL; hash pending download receipt | 1,217,607,321 | `weights/sam3d/torch_home/hub/checkpoints/` | Public; Apache-2.0. |
| DINOv2 ViT-B/14 reg4 | Official checkpoint URL; hash pending download receipt | 346,393,545 | Same Objects checkpoint directory | Public; Apache-2.0. |
| FoundationPose scorer | Official folder `2024-01-11-20-02-45` | Unknown | `weights/cari4d/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/` | Organization blocked; model terms and content hashes not verified. |
| FoundationPose refiner | Official folder `2023-10-28-18-33-37` | Unknown | `weights/cari4d/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/` | Organization blocked; model terms and content hashes not verified. |

Each FoundationPose directory needs **both `model_best.pth` and `config.yml`**. The native MHR pipeline explicitly selects the **NVlabs PyTorch** backend. NVIDIA TensorRT checkpoint packages are a different backend and do not replace these required files. See the pinned [CARI4D downloader](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_cari4d/lib/download_weights.py) and [FoundationPose downloader](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_foundation_pose/lib/download_weights.py).

## Required source caches

The source entries use GitHub archives at full commits. They contain no Git metadata. Installing all Python/native dependencies remains a separate step described by the deployment dependency inventory.

| Source | Commit | Target |
| --- | --- | --- |
| `nvidia-isaac/video_to_data` | `33129dd0f2d2dcfd1164d43fd076542660756ed2` | `third_party/video_to_data/` |
| `facebookresearch/sam-3d-objects` | `f91db411c50efee93d8db7aeb323885650f6f722` | `third_party/sam-3d-objects/` |
| `facebookresearch/dinov3` | `6876159a11b4df116f30f667f8c9888617df0751` | `weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov3_main/` |
| `facebookresearch/dinov2` | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | `weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov2_main/` |
| Same DINOv2 revision | Same commit | `weights/sam3d/torch_home/hub/facebookresearch_dinov2_main/` |
| `facebookresearch/sam2` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | `third_party/sam2/` |
| `facebookresearch/pytorch3d` | `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba` | `third_party/pytorch3d/` |
| `NVlabs/nvdiffrast` | `253ac4fcea7de5f396371124af597e6cc957bfae` | `third_party/nvdiffrast/` |
| `pybind/pybind11` | `aa304c9c7d725ffb9d10af08a3b34cb372307020` | `third_party/pybind11/` |
| `microsoft/MoGe` for Objects | `a8c37341bc0325ca99b9d57981cc3bb2bd3e255b` | `third_party/moge-objects/` |
| `microsoft/MoGe` for CARI4D | `925b8ed835a7a9cdb7578ba15c658a0afc969030` | `third_party/moge-cari4d/` |
| `EasternJournalist/utils3d` | `3fab839f0be9931dac7c8488eb0e1600c236e183` | `third_party/utils3d/` |

The SAM3D Objects source pin is this project's reproducibility choice; upstream's container clones its unpinned default branch. DINOv3 uses its own DINOv3 License; DINOv2 uses Apache-2.0; SAM3D Objects uses the SAM License. Source and model licenses must be considered separately.

The two MoGe versions and Utils3D revision are upstream pins. PyTorch3D resolves the upstream `v0.7.9` tag and pybind11 resolves `v2.10.0`. SAM2 and nvdiffrast use recorded project pins for upstream dependencies whose recipes omit a revision. These source choices have not been built or tested on the future GPU server. The JSON includes each source's license and selection basis.

Pre-downloaded sources do not automatically alter the official Dockerfile's Git/PyPI install commands. The build must explicitly use the prepared directories. GPU wheels, the Eigen/system packages, the base-image digest and any remaining native build inputs belong to the separate [dependency inventory](requirements/upstream-containers.lock.json); this source archive is not yet an offline installation image.

After preparation, invoke CARI4D with **`--skip_weight_download`**. Its original downloader rejects pre-existing source-cache directories without `.git`, so rerunning it on these extracted archives would fail. The inference entry accepts existing weight paths and uses the Torch Hub caches. See [`run_inference.py`](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_cari4d/lib/run_inference.py).

## Runtime cache details

These files close known lazy-download paths; they do not establish that every downstream library can run offline yet.

1. **MoGe2-L and MoGe1 require different checkpoints.** CARI4D checks the pinned MoGe2-L file through the Hugging Face cache. Objects locates MoGe1 inside its own cache and rewrites the depth-model configuration to the local `model.pt`. Keep both target layouts. The standalone Body MoGe2-B model does not substitute for either.
2. **Hugging Face cache refs are included.** A snapshot fetched only by commit may omit `refs/main`. `cache_refs.main` points default lazy lookups at the selected immutable snapshot. The fetcher materializes snapshot symlinks for ZIP transfer and receipts include the refs. Use a fresh, dedicated asset root; Objects takes the first cached MoGe1 snapshot it sees.
3. **DINO source and checkpoint caches are both needed.** The Objects downloader preloads register weights but omits the DINO source checkout; this inventory adds it. Body loads DINOv3 with `pretrained=False`; its learned parameters are already in `model.ckpt`, so a separate DINOv3 pretrained checkpoint is unnecessary for this path. PyTorch Hub may still probe the network when resolving an implicit branch; a later offline smoke run must check this behavior before claiming network independence.
4. **MHR default buffers are embedded.** Native CARI4D can extract hand/scale/rig buffers from the selected Body checkpoint. A separately produced `mhr_buffers.pt` is optional. No external `defaultparams` weight is required by the inspected MHR entry path.
5. **Keep the Objects layout unchanged.** `hf-download/checkpoints/pipeline.yaml` and its neighboring files match the official loader. The empty `ss_encoder.safetensors` placeholder and repository demonstration media are excluded. Both mesh `.ckpt`/`.pt` variants are retained until the gated configuration can be verified.

The evidence is in the pinned [Objects downloader](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_sam3d/lib/download_weights.py), [Objects loader](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_sam3d/lib/image_to_mesh.py), [Body DINOv3 backbone](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/backbones/dinov3.py), and [MHR decoder](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/mhr_layer.py).

## Optional profiles

These entries are marked `required: false` and are excluded from the required byte total.

- **`standalone-body-moge2`**: `Ruicheng/moge-2-vitb-normal@ca5f0e0`, 419,110,246 bytes, MIT. Needed for the independent `v2d_sam3d_body` default FOV path. The native CARI4D path already supplies intrinsics through MoGe2-L.
- **`soma-export`**: pretrained MHR/SOMA rig files from `nvidia/soma-x@466879a`, 249,518,051 bytes. Supports a later MHR-to-SOMA internal export; these are model assets, not challenge episode data. `py-soma-x==0.2.1` uses the lowercase repository ID, so the cache directory must also use `models--nvidia--soma-x`. Set `HF_HOME` to the absolute `weights/soma/hf_home` path before imports, or pass the pinned snapshot as `data_root`. Use `enable_procedural_transforms=False` with this asset revision; the newer procedural-transform rig files are absent. The source package is Apache-2.0, while separate asset/model terms still require review.

The source installation and actual MHR-to-SOMA conversion are separate from downloading these optional files. An archive containing them does not prove that conversion or the final project adapter has succeeded.

## Inspect and transfer

The following command only prints the declared inventory; it downloads nothing:

```powershell
python tools/fetch_assets.py plan --manifest deployment/assets.lock.json
```

Asset fetching, access checks and bundle preparation are separate tool commands. Current model preparation is incomplete because of the gates and the organization block above. Do not mark a ZIP as a complete model bundle solely because source downloads succeed. Use the model-access pages to review/request the applicable access through an approved account: [CARI4D](https://huggingface.co/nvidia/cari4d_commercial), [SAM3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3), [SAM3D Objects](https://huggingface.co/facebook/sam-3d-objects).

The fetcher writes `.receipts/<asset-id>.json` containing hashes of downloaded files. The packager consumes those receipts and rechecks their contents. Hugging Face LFS `sha256` is the content checksum. `git_blob_sha1` identifies a Git blob and must not be treated as a SHA-256 checksum. Files whose source publishes no verified content hash remain explicitly unpinned until a download receipt records their actual bytes. FoundationPose receipt acquisition remains blocked here.

Never place credentials, Hugging Face token caches, challenge ground-truth data, or the model binaries themselves in Git. Store binary assets under the separate asset root for an authorized ZIP transfer. License/access approval does not automatically permit publishing model weights in a public repository.
