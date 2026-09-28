# Expanded candidate runtime scope

Reviewed 2026-09-28 from the pinned source metadata, without installing packages,
importing models, or running inference. The baseline CPython 3.10/3.11,
Torch 2.5.1 and CUDA 12.4 acquisitions do **not** cover every expanded candidate.
Each candidate needs a separate, explicitly selected environment and a validated
adapter. Downloaded distributions are not installed or tested environments.

The acquisition task is adding a separate SAM3 CPython 3.12 / Torch 2.10 / cu128
profile, a separate MoGe3 profile with NumPy 2, and selected missing YOLOE
dependencies. Consult the final runtime manifests and publication records for
what actually completed; this statement does not certify those downloads or
their dependency closure. The legacy human/physics environments below have not
been acquired in full. Their large historical Torch/CUDA stacks will require a
target choice and either isolated containers or an explicit, tested port.

## Source-declared environments

| Candidate and fixed source | Upstream declaration | Difference from the baseline acquisition |
| --- | --- | --- |
| [SAM3 `2345a4a`](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/README.md#installation) | README requires Python >=3.12, Torch >=2.7 and CUDA >=12.6; its example installs Torch 2.10.0 from cu128. `pyproject.toml` requires `timm>=1.0.17`, `numpy>=1.26,<2`, `ftfy==6.1.1`. | CP310/311 Torch 2.5.1/cu124 does not match the documented supported stack. The packaging metadata's looser `requires-python>=3.8` is not evidence that the older stack works. Separate acquisition and GPU validation are needed. |
| [MoGe3 `74fbce0`](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/pyproject.toml) | Python >=3.10; NumPy >=2; Gradio >=6; Torch >=2.4; torchvision >=0.19; three fixed Git dependencies listed below. | Baseline NumPy 1.26.3 and Gradio 5.49.0 do not satisfy these declarations. Its NumPy requirement and SAM3's `<2` requirement cannot share one environment. The default uv index is cu130, which may be deliberately overridden; cu130 is not stated here as an unavoidable model requirement. |
| [YOLOE `40cd606`](https://github.com/THU-MIG/yoloe/blob/40cd606cabdbe2b566d6f14a6b162c89206e9a1b/requirements.txt) | README uses Python 3.10; project metadata permits Python >=3.8, Torch >=1.8 and torchvision >=0.9. The install recipe includes editable LVIS API, MobileCLIP and CLIP subpackages. | Broad Torch bounds do not prove the complete pipeline works. Additional distributions, complete editable-package metadata, the text-encoder export and the video mask adapter need preparation and validation. The optional visual-prompt SAM2 path is outside the filtered text-prompt package. |
| [GVHMR `ee960bb`](https://github.com/zju3dv/GVHMR/blob/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/requirements.txt) | Python 3.10 recipe; Torch 2.3.0+cu121, torchvision 0.18.0+cu121, PyTorch3D 0.7.6 for CP310/cu121/Torch2.3, NumPy 1.23.5, timm 0.9.12, Lightning 2.3.0. | These are different pinned binary/API versions. Baseline Torch 2.5.1, torchvision 0.20.1, NumPy 1.26.3, timm 1.0.30 and Lightning 2.5.6 are not substitutes proven by acquisition. |
| [WHAM `2b54f77`](https://github.com/yohanshin/WHAM/blob/2b54f7797391c94876848b905ed875b154c4a295/docs/INSTALL.md) | Tested recipe: Python 3.9, Torch 1.11.0, torchvision 0.12.0, CUDA toolkit 11.3. Requirements pin NumPy 1.22.3, mmcv 1.3.9, timm 0.4.9 and setuptools 59.5.0. | Its original environment is not CP310/311/cu124. ViTPose and optional DPVO also need their own installation/native build. No compatibility port has been performed. |
| [SmoothNet `c03e93e`](https://github.com/cure-lab/SmoothNet/blob/c03e93e8a14f55b9aa087dced2751a7a5e2d50b0/scripts/install_conda.sh) | Install script chooses Python 3.6, Torch 1.10.1, torchvision 0.11.2 and CUDA toolkit 10.2. README reports testing on Torch 1.10.1 with Python >=3.6. Requirements include SciPy 1.5.4 and Matplotlib 3.3.4. | The old recipe is recorded as evidence, not recommended as a new supported deployment. Its exact old binary set is not in the baseline package; modernization or an isolated compatible container is unresolved. |
| [HTD-Refine `2fcd6dd`](https://github.com/ant-research/HTD-Refine/blob/2fcd6ddef3c4eb75a636062245f80a0136c09b7e/requirements.txt) | Python >=3.10; Torch 2.3.0+cu121, torchvision 0.18.0+cu121, PyTorch3D 0.7.6 for CP310/cu121/Torch2.3, NumPy 1.23.5, timm 1.0.12, video-reader-rs 0.2.1. | Even GVHMR's timm pin is different. `pyproject.toml` has an empty dependency list, so installing only the editable package does not install these requirements. |
| [PhysPT `40d8699`](https://github.com/zhangy76/PhysPT/blob/40d869927e53a3cade8ba657cacdaeb993f085b8/README.md) | README creates Python 3.7; most requirements, including Torch, NumPy and torchvision, have no versions. | An unpinned requirement is not a compatibility result. The CP310/311 profile, Qt visualization, initialization adapter and native dependencies remain unvalidated. |

## Concrete supplemental dependencies

MoGe3's pinned `pyproject.toml` lists these distinct repositories. The older
baseline `utils3d` source is not the `utils3d-moge` package below:

- `EasternJournalist/utils3d-moge` at `62f09d58509485564e24d5d9f6aac9ee9ebc0c37`.
- `EasternJournalist/pipeline` at `1c511390d90226c00c101f34b84df26a0f8789b4`.
- `JeffreyXiang/FlexGEMM` at `b2fadb29d41846c7981ade6801ffc689fae119cf`.

YOLOE's project metadata additionally requires `py-cpuinfo`,
`ultralytics-thop>=2.0.0` and `supervision>=0.25.1`; these were absent from the
initial baseline inventory. Its local MobileCLIP/LVIS/CLIP packages are separate
editable sources, not ordinary wheels supplied by installing the root project.
The initial filtered YOLOE archive lacked MobileCLIP's `README.md` and
`requirements.txt` and LVIS API's `requirements.txt`. The final platform source
manifest must explicitly include those files before its installation metadata
is called complete.

Small human/physics dependency gaps include `progress==1.6` for SmoothNet
(WHAM leaves its version open), `munkres` for WHAM, `hydra-zen` for GVHMR and
`pyqtgraph` for PhysPT. Further absent or differently pinned requirements include
`chumpy`, `mmcv==1.3.9`, `xtcocotools`, `cython_bbox`, `lapx`, `wis3d`, `pycolmap`,
`ultralytics==8.2.42` and `video-reader-rs==0.2.1`. Some are native extensions,
GUI components or optional tooling; classify them against the chosen inference
profile before resolving or building them. Fetching a few CPU packages does not
close the historical GPU environments or resolve incompatible version pins.

## Corrected human/physics source package

`expanded-human-physics-sources-runtime.zip` is a **source and build-metadata
package**, not an installed runtime or a model-weight archive. It supersedes the
earlier source ZIP for this handoff while preserving that earlier ZIP unchanged.

The correction adds the seven pinned ViTPose `requirements/*.txt` files and its
`MANIFEST.in`, plus the retained DPVO vendored pybind11 build templates,
`MANIFEST.in` files and CMake helpers. Each added file is checked against its
pinned Git blob and recorded with SHA-256. No upstream script is executed.
The four source files withheld for restrictive or unclear terms remain excluded;
there are still zero human/physics checkpoint files in this package.

Optional DPVO Pangolin/DBoW2 submodules, external Eigen placement and native
compilation remain separate work. Required body models/helper assets, MHR
conversion, contact interfaces and experiment adapters remain unresolved.
Source metadata repair does not establish a complete build or executable model.

## Publication and data boundaries

Use the final inventories to distinguish files acquired locally, files permitted
in a public mirror, and files actually published. In particular, acquiring a
NVIDIA binary does not establish permission for a standalone public mirror, and
SMPL-family access does not grant public redistribution rights. Retain the
separate model/software licenses and manual acquisition requirements.

No installation, inference, benchmark or server deployment was performed by this
review or source correction. Only Track 1 challenge videos/metadata and their
verified reconstructed artifacts may be used later; Track 2 content, derivatives
and unknown-provenance demo/dataset assets remain prohibited for every purpose.
