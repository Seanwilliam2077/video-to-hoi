# SAM3 and MoGe3 runtime distribution handoff

Acquired on 2026-09-28. These are downloaded distributions with exact origins,
checksums, preserved license files and static dependency metadata checks. Nothing
was installed, compiled, imported for GPU execution, deployed or used for inference.

## Separate profiles

| Profile | Python / PyTorch | Important pins | Acquired distributions |
| --- | --- | --- | --- |
| SAM3 video | CPython 3.12 / 2.10.0+cu128, torchvision 0.25.0+cu128 | NumPy 1.26.4, SciPy 1.17.1, tifffile 2025.3.30, setuptools 80.9.0, OpenCV 4.11.0.86 | 65; 48 publicly mirrored |
| MoGe3 | CPython 3.12 / same PyTorch and torchvision | NumPy 2.5.3, SciPy 1.18.1, Gradio 6.28.0 | 87; 71 publicly mirrored |
| YOLOE / human helper supplement | Universal distributions checked against the existing CP310 inventory | py-cpuinfo 9.0.0, ultralytics-thop 2.2.1, supervision 0.30.5, progress 1.6, munkres 1.1.4, hydra-zen 0.16.0, pyqtgraph 0.14.0; colorama, defusedxml, pydeprecate | 10; all publicly mirrored |

Across these profiles there are **124 distinct distributions / 4,483,433,059
bytes acquired**. The public archive contains **106 distributions /
1,482,363,460 bytes**, plus manifests, notices, corresponding source and helpers.
Both complete modern profile wheel selections have a **glibc 2.28 minimum from
their wheel tags**. This is a binary packaging floor; it does not verify system
libraries, host drivers, GPU architecture, CUDA compilation or server compatibility.
The earlier full CP310/CP311 inventory has a higher glibc 2.35 floor due to Open3D.

SAM3 and MoGe3 must use separate environments. A single installation of all wheels
in the shared archive would mix incompatible NumPy and dependency versions.

## Evidence and compatibility choices

- SAM3 source: `facebookresearch/sam3@2345a4ad109ac29c569da749c91d84f10dc08c40`.
  Its README requires Python 3.12+, PyTorch 2.7+ and CUDA 12.6+, and explicitly
  installs PyTorch 2.10 from the cu128 index. Its package requires NumPy below 2.
  SciPy 1.18 and current tifffile require NumPy 2, so the SAM3 profile pins older
  compatible releases. `model_builder.py` imports `pkg_resources`; setuptools
  80.9.0 retains that module. Required video-path imports such as OpenCV, decord,
  einops, Hydra, psutil, SciPy, scikit-image and pycocotools are included in the inventory. Static traversal found pycocotools through
  `sam3.train.masks_ops` and the COCO JSON loader even though it is absent from
  the core package dependencies.
- MoGe3 source: `microsoft/MoGe@74fbce054ebed49800de42d0ad0e83495065719a`.
  Its package requires NumPy 2+, Gradio 6+, torch 2.4+ and torchvision 0.19+.
  Its configuration explicitly documents switching the PyTorch backend to cu128.
  The three exact Git dependencies are delivered separately in
  `moge3-runtime-sources-20260928.zip`: utils3d_moge at `62f09d58509485564e24d5d9f6aac9ee9ebc0c37`,
  pipeline at `1c511390d90226c00c101f34b84df26a0f8789b4`, and FlexGEMM at
  `b2fadb29d41846c7981ade6801ffc689fae119cf`. Their declared runtime dependencies
  are included here. FlexGEMM's optional CUDA extension remains unbuilt.
- The final selected metadata has no missing or incompatible mandatory declared
  dependency specifications for the two modern package profiles. This check does
  not establish binary ABI, import or inference compatibility.

## Receive and assemble one profile

The archive extracts to `runtime-modern-cp312-cu128/`. Its `PUBLIC_MANIFEST.json`
and `PUBLIC_SHA256SUMS` describe only the files actually in the public archive.
The full acquisition ledger also lists the locally acquired, withheld files.

Using a Python 3.10+ control interpreter, assemble each profile into a separate
empty directory; this only copies files and checks their size and SHA256:

```bash
python runtime-modern-cp312-cu128/assemble_runtime_profile.py \
  --bundle runtime-modern-cp312-cu128 --profile sam3 \
  --output sam3-cp312-distributions --allow-missing

python runtime-modern-cp312-cu128/assemble_runtime_profile.py \
  --bundle runtime-modern-cp312-cu128 --profile moge3 \
  --output moge3-cp312-distributions --allow-missing
```

Each output contains a profile-specific `requirements.lock.txt`, acquisition
manifest and `ASSEMBLY_REPORT.json`. The report lists withheld files still needed.
The helper rejects overlapping directories, foreign wheels, another profile's
output, and changed existing files. It does not install the selected packages.

The recipient can retrieve original upstream distributions after reviewing their
licenses. This command downloads exact files and validates checksums; it does not
install, build or run third-party code:

```bash
python runtime-modern-cp312-cu128/fetch_runtime_distributions.py \
  --manifest runtime-modern-cp312-cu128/profiles/sam3.json \
  --root runtime-modern-cp312-cu128
```

Use `profiles/moge3.json` for the other profile, then rerun the corresponding
assembler without `--allow-missing`. The downloader permits only the recorded
official package hosts, including redirects. It does not access Google Drive or
other netdisk services. Package installation and source builds await the selected
remote machine and its environment plan.

## License and remaining runtime scope

Eighteen distinct distributions are acquired locally but omitted from the public
archive: NVIDIA SDK libraries, Triton with bundled NVIDIA tools, decord, and the
two OpenCV variants. Their exact official URLs, sizes and hashes remain in the
manifests. Native copyleft corresponding-source closure or standalone SDK
redistribution permission has not been established for those binary mirrors.

NCCL is distributed under its embedded BSD grant, with the referenced NVTX notice
and full Apache-2.0-with-LLVM-exception terms supplied. CUDA bindings 12.9.4 is
separately licensed under its original NVIDIA CUDA Python license: unchanged
redistribution is allowed under sections 1(b)/2 with consistent downstream terms,
NVIDIA GPU usage restrictions, preserved notices, and its stated compliance
obligations. It is not relicensed under this project or another included package.
The archive is an aggregation of separately licensed distributions.

The ultralytics-thop AGPL wheel is accompanied by its complete official source
distribution; all seven wheel Python files were byte-compared to that source.
Original embedded LICENSE/NOTICE files remain inside every unmodified distribution.

SAM3 optional FA3, xformers, cc_torch and torchcodec paths are not selected; the
default cv2 loader and fallback paths remain the intended initial configuration.
Training and notebook extras are not acquired. OS/APT packages, container layers,
driver validation, source builds and GPU execution remain outstanding.

The helper supplement does not prepare the historical GVHMR, HTD, WHAM, SmoothNet
or PhysPT environments. It also does not supply a Qt backend for pyqtgraph. The
existing runtime gap report tracks those independent environments and model access
restrictions. All challenge input and reconstructed asset usage remains Track 1
only; no Track 2 content or unknown-provenance challenge assets are included.
