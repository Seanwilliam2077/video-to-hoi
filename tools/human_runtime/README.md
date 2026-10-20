# Pinned SAM 3D Body runtime source

`prepare_sources.py` downloads only the explicit Python source and licenses in
`source_manifest.json`. It never clones a repository, downloads an archive,
loads a demonstration asset, or fetches challenge data. Every file is verified
against its recorded SHA-256. The source tree is ignored at
`third_party/sam3d-body-runtime`; original bytes remain in its `.upstream` cache.

The SAM implementation comes from NVIDIA `video_to_data` revision
`33129dd0f2d2dcfd1164d43fd076542660756ed2`, specifically
`reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body`. DINOv3 comes from Meta
revision `6876159a11b4df116f30f667f8c9888617df0751`. The source licenses remain
alongside the installed code. This helper does not grant model access or change
the upstream license terms.

## Compatibility changes

Four installed files receive exact, count-checked text changes. The installed
manifest records both upstream and patched hashes:

- The estimator transfers tensors to its model's selected device and only
  empties the CUDA cache when running on CUDA.
- The model creates ray and wrist-index tensors on its selected device, and
  transfers hand batches there.
- The TorchScript MHR file initially loads onto the CPU. The caller transfers
  it along with the model, avoiding an implicit CUDA allocation for CPU runs.
- The backbone calls the pinned local DINOv3 architecture instead of downloading
  a repository through `torch.hub`. It retains `pretrained=False`; the SAM
  checkpoint supplies the trained backbone weights.

These changes do not approximate transforms or substitute geometry libraries.
Detectron2, PyTorch3D, MoGe, SAM segmentation, and PyMomentum are not needed for
this explicit-bounding-box, explicit-camera inference path. Set
`MOMENTUM_ENABLED=0` to use the supplied TorchScript MHR model.

## Required runtime packages and assets

The exercised import path requires PyTorch, torchvision, NumPy, OpenCV, Pillow,
PyTorch Lightning, OmegaConf, YACS, RoMa, einops, and braceexpand. `timm` is needed if
selecting the alternate ViT backbone. The pinned DINOv3 branch does not use it.

Required model files are `model.ckpt`, its adjacent `model_config.yaml`, and
`assets/mhr_model.pt`. The MHR filename can be passed explicitly to the runner.
The runner hashes all model files and verifies every installed source file.

```powershell
& .venv\Scripts\python.exe tools\human_runtime\prepare_sources.py
& .venv\Scripts\python.exe tools\human_runtime\prepare_sources.py --offline
```

`tools/run_sam3d_body.py` adds this source directory to `sys.path`, selects the
device, and disables half precision in the configuration for CPU runs. It uses
the upstream architecture and checkpoint loader. CUDA uses the checkpoint's
configured backbone precision. `--source-url` must identify the permitted
official Track 1 asset; it is provenance, not proof of camera calibration.

## Native MHR and camera placement

The estimator exposes `mhr_model_params[204]` **with zero root translation**.
Its `scale_params[28]` is a latent scale encoding and must not be submitted as
the required 68 native scales. Use native parameters `[136:204]` instead.

`pred_vertices` and `pred_keypoints_3d` are root-local, in metres, with axes
already converted to OpenCV. `pred_cam_t` supplies the camera-space placement.
Add that offset once to the mesh/keypoints. For the native 204-vector, add
`pred_cam_t * [10,-10,-10]` to its first three entries. MHR translation parameter
units are 0.1 metres, and native Y/Z have the opposite signs. Shape is the
45-vector `shape_params`; face expression is zero in the inference head.

Full inference updates the hand parameters before exposing the final native
vector. Review keypoints follow the exact 70-keypoint definition and 65 edges
in the pinned `metadata/mhr70.py` source. The runner writes placed camera-space
meshes/keypoints, absolute-pixel 2D keypoints, and both root-local and placed
native parameters per frame, then combines them as memory-mapped `.npy` arrays.

Raw inference has per-frame identity and scales. Apply the separate shared
identity fitter before using `pose[T,136]`, `scales[68]`, `shape[45]` as the Human
portion of a submission. A complete challenge submission also requires the
object and official episode/frame indexing.
