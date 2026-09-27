# Remote platform and baseline deployment

Status: all required model downloads verified; GitHub model release published and verified,
2026-09-27. The owner requested acquisition and GitHub handoff only. Deployment
and model execution remain deferred. The owner
cannot connect to the server through SSH; code and preparation materials will
be transferred as ZIP files and used through the server's available console.
See [deployment/README.md](../deployment/README.md) for the preparation commands.
The owner supplied a screenshot of a
candidate host with eight RTX PRO 5000 72 GB GPUs. Remote access, CPU, host RAM,
storage, and container support are not yet verified. Two more server types,
L20 and A10, are also available; see below. The two real baselines
have not been run. Model inference,
environment installation, and acceptance tests will run on the remote host,
not on the project owner's local computer.

## Candidate host: 8 × RTX PRO 5000 72 GB

The screenshot shows eight GPUs, 73,415 MiB reported memory per GPU, driver
595.58.03, and a CUDA compatibility display of 13.2. All GPUs were idle at the
time of the screenshot. This establishes available device capacity at that
moment, not a completed software compatibility test or a team allocation.

The [RTX PRO 5000 72 GB](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/)
uses Blackwell; NVIDIA lists it at
[compute capability 12.0](https://developer.nvidia.com/cuda/gpus).
The pinned CARI4D image below uses CUDA 12.4 and native extension targets ending
at 9.0. Do not deploy it unchanged and assume it supports this host. A Blackwell
container needs a compatible PyTorch/CUDA build and rebuilt, tested native
dependencies, including the geometry, rasterization, and FoundationPose paths.
PyTorch first documented Blackwell wheels in
[2.7 with CUDA 12.8](https://pytorch.org/blog/pytorch-2-7/); that is evidence of
framework support, not validation of the complete CARI4D dependency stack.

Use one GPU for compatibility checks and a short real sequence first. Once the
complete pipeline succeeds, allocate one GPU per episode or independent job.
Do not split one video across GPUs until temporal state and scale consistency
have an explicit implementation. For four concurrent jobs, provision roughly
32–64 vCPUs, 256 GB host RAM or more, and 2 TB of persistent fast storage as an
initial engineering estimate; measure actual use before increasing concurrency.

## Other available servers: L20 and A10 (2026-09-27)

The owner reported two more server types besides the RTX PRO 5000 host. Their
count, CPU, host RAM, storage, OS, driver version, and access method are not
yet known.

| GPU | Memory | Architecture | Fit for the pinned upstream containers |
|---|---|---|---|
| NVIDIA L20 | 48 GB | Ada Lovelace, compute capability 8.9 | 8.9 is in the official recipes' target list (8.0, 8.6, 8.9, 9.0), and 48 GB matches the toolkit's target. The strongest candidate for the first two baselines, pending a driver and container check on the machine itself. |
| NVIDIA A10 | 24 GB | Ampere, compute capability 8.6 | 8.6 is in the target list, but 24 GB is below the 32 GB that SAM 3D Objects requires, so it cannot run the complete unmodified workflow. Use it for lighter stages such as masks and depth, or for tests. |

Recommendation: run the first two baselines on one L20 GPU. Keep the RTX PRO
5000 host for parallel runs across episodes once a Blackwell build of the
container stack is validated. Use A10 for light stages. An in-house L20 makes
renting the L40S below unnecessary unless the L20 check fails.

## Alternative initial machine

Start with one dedicated NVIDIA L40S with 48 GB VRAM, 16–32 vCPUs, 128 GB RAM,
and a 1 TB persistent SSD volume on an x86-64 Ubuntu 22.04 host. Run episodes
16 and 12 sequentially, and profile peak GPU memory, host memory, disk usage,
and elapsed time before scheduling parallel jobs.

These CPU, RAM, and disk sizes are engineering recommendations, not published
model minimums. Prefer a host that supports GPU Docker containers and provides
console or job-runner access, ZIP upload, persistent storage, and approved outbound access to GitHub, Hugging Face,
PyPI, and the required container registries.

| Use | GPU allocation | Host RAM | Persistent storage |
|---|---|---|---|
| First two baselines, one GPU job at a time | 1 × L40S 48 GB; RTX A6000 48 GB is an alternative | 128 GB | 1 TB |
| More headroom for long-clip refinement | 1 × 80 GB GPU, after validating native extensions and headless rendering on that exact model | 128–256 GB | 1–2 TB |
| Two developers running independent experiments | 2 × 48 GB GPUs, one GPU per job | 256 GB recommended | 2 TB |

Multiple GPU memories do not automatically form one pool. A pair of 24 GB GPUs
is not a substitute for a single 48 GB GPU for an implementation that expects
all tensors on one device. Four team members can share one GPU through a queue;
four simultaneous model jobs require separate memory and scheduling estimates.

## Evidence and compatibility

- [SAM 3D Objects setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
  specifies Linux 64-bit and at least 32 GB of GPU memory. This is why a 24 GB
  GPU is not the default for the complete unmodified workflow.
- The [V2D README](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/README.md)
  reports an egocentric end-to-end example on RTX A6000 48 GB, L40, and L40S.
  That example is not a measured memory guarantee for our Track 1 MHR CARI4D
  episodes. Actual acceptance requires running our selected episodes.
- The [MHR CARI4D container](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_cari4d/docker/Dockerfile)
  uses PyTorch 2.5.1 with CUDA 12.4 and builds native extensions for architectures
  8.0, 8.6, 8.9, and 9.0. Use the pinned per-module containers instead of a
  single shared Python environment with an arbitrary latest CUDA stack.
- Install a compatible NVIDIA host driver, Docker Engine, and NVIDIA Container
  Toolkit. Verify CUDA execution and headless rendering inside the actual
  containers before the first full reconstruction. Do not assume VRAM capacity
  alone establishes compatibility with every native extension.

## Model access and handoff

Access to `facebook/sam-3d-body-dinov3`, `facebook/sam-3d-objects`, and
`nvidia/cari4d_commercial` is approved. Fresh access checks and complete downloads
resolved the earlier HTTP 403 failures. Their 28 selected files total
17,995,612,997 bytes and passed size/SHA-256 verification. The
[official CARI4D instructions](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_cari4d/README.md)
explain the Body approval requirement and the separate Objects checkpoint.
All 12 required model entries are now present locally. Redistribution review
permits these three model packages under their respective NVIDIA/SAM terms;
[MODEL_RELEASES.md](../deployment/MODEL_RELEASES.md) records the license copies,
provenance, and restore procedure. The [model release](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/hf-models-20260927)
is published with 11 model parts and four companion files. GitHub digests and unauthenticated public downloads were verified.
No remote token setup or deployment is part of this task. Keep credentials
out of the repository, run manifests, logs, and shared archives.

The selected provider must allow Docker GPU access. A notebook container that
cannot launch the required per-module containers needs a different deployment
strategy and is not interchangeable with a GPU virtual machine.

## Shared storage layout

The following is a proposed path layout, not an existing server address:

```text
/srv/video-to-hoi/
  repo/                 pinned project checkout
  third_party/          pinned upstream checkouts
  weights/              persistent downloaded model cache
  data/track_1/         Track 1 input data only
  runs/                 per-experiment outputs and manifests
  baselines/            immutable accepted runs
  reports/              inspection pages, overlays, and logs
```

Store accepted baselines and their checksums on persistent shared storage that
survives instance termination. Record the real SSH path or object-storage URL
in the deployment record after provisioning; SSH is not required for the ZIP handoff. Local workspace paths and example
paths above are not coworker-accessible baseline addresses.

## First two real baselines

Use Track 1 episode 16 (foam grass block, 360 frames) and episode 12 (black pan,
405 frames). Inputs are 1536 × 1152 at 30 fps according to the Track 1 metadata;
verify decoded frame counts on the remote host before inference.

Generate human/object masks and reconstruct the object mesh from Track 1
frames. The MHR CARI4D inference command requires a supplied object mesh; it
does not generate that mesh itself. Check and establish the candidate mesh's
scale before supplying it. Do not substitute a Track 2 mesh, trajectory,
calibration, or derived parameter.

Acceptance for each real baseline:

1. All stages consume real Track 1-derived inputs; no fake human, mesh, or
   object-motion backend contributes to the accepted run.
2. Record source-video hashes, selected frame IDs, generated-mesh provenance,
   model revisions, container identities, commands, and stage input/output hashes.
3. Export valid MHR and an object pose for every frame, including occlusions;
   label the current storage format as internal until the official adapter is verified.
4. Preserve full stage logs, elapsed times, peak resources, masks/depth previews,
   and reconstruction overlays, with failure and low-confidence frames visible.
5. A coworker can fetch the same code/configuration and inputs, rerun the stages,
   and inspect the same artifacts. Reproducibility does not imply GPU outputs
   will be bitwise identical across different hardware or library versions.
6. Freeze the accepted run under a shared immutable ID and publish its actual
   team-accessible address. Track 1 has no public reconstruction ground truth;
   diagnostic reports are not official leaderboard scores.

Before a host is available, code and deployment preparation can proceed, but
real reconstruction, remote environment validation, and a shared baseline
address remain incomplete.

## Manual FoundationPose handoff (2026-09-27)

The owner clarified that iOA blocks Codex's Google Drive access, while their
own manual downloads are allowed. They obtained both FoundationPose folders.
The manual tool recorded receipts for the required `model_best.pth` and
`config.yml` files; an independent offline size/SHA-256 check then matched
the local files: scorer, 190,230,167 bytes; refiner, 68,220,817 bytes.
Together they add 258,450,984 bytes. These hashes establish local transfer
integrity; no upstream SHA-256 was available for independent authenticity
comparison. The automated fetch/check tool still does not contact Drive.
See [manual commands](../deployment/README.md) and
[preparation status](../deployment/PREPARATION_STATUS.md).

The completed required-model count is now 12/12 following the approved
Hugging Face downloads. Across 12 source and 12 model receipts, all 7,930 files
passed size/SHA-256 verification. The model scope occupies 23,731,473,555 bytes,
including 80 bytes of pinned cache refs. This confirms the prepared files,
not model execution. No Drive download was executed by Codex, and no model
inference or server deployment has been performed. The FoundationPose
add-on was published on 2026-09-27; the three additional model packages are
published separately with verified public download links. The revised local status page must be republished
to its existing online address after the final release update.

### Published add-on

The [FoundationPose add-on release](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/foundationpose-addon-20260927) provides the 239,097,418-byte ZIP, SHA-256 companion, and metadata. All three uploaded assets matched their local SHA-256 values reported by GitHub. The ZIP keeps its verified code snapshot `5205666`; the [release index](../deployment/releases/foundationpose-addon-20260927.json) records the permanent URLs. Extract the separately supplied public-assets base first, then this add-on. This publication adds a downloadable handoff location, not a deployed environment or real baseline result.


## First-three-video comparison preparation (2026-09-27)

The owner requested comparisons for modules 1, 2 and 4, then clarified that the remote entry point is not ready and only the test handoff should be prepared. [The kit](../benchmarks/first3/README.md) fixes Track 1 episodes 0, 1 and 2, all hula-hoop clips: 790/668/866 frames at 30 fps, original 1536 x 1152 resolution. Full CPU decoding and SHA-256 verification checked inputs only; no model execution occurred.

Five recipes cover MoGe2, MoGe3, prompted SAM2, standalone SAM3D Body and joint CARI4D. SAM3/YOLOE, GVHMR/WHAM and SmoothNet/HTD-Refine/PhysPT are recorded as research candidates with source pins and explicit weight/adapter blockers. The three temporal methods do not yet have MHR adapters. MoGe3 adds a separately pinned checkpoint not included in the prior baseline model inventory. No server environment is installed or certified by this kit.

The ranking tool groups comparable tasks and metrics, requires the same three complete episodes, and rejects missing or mismatched evidence. Its protocol remains unfrozen until runtime, upstream, and independent observations exist. Acceleration ranking also requires reviewed video fidelity for each episode. No official score, actual measurements or ranking is supplied. Automatic quality extraction from native candidate outputs is still to be implemented; the tool validates submitted metric records and does not prove their truth.

The package keeps the original input bytes, recipe/protocol files, an empty result report and per-file checksums. Future execution must run on the selected Linux GPU machine. The previous two-baseline acceptance target remains pending; this additional hula-hoop screening does not replace it.

The [published first-three kit](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/first3-benchmark-20260927) is 68,971,507 bytes, SHA-256 `6b2eb93051415fc0a838622bdf5cdba6d8127bd3dc96c78d0f5f4c59c2b4c59b`, from source commit `d822121`. Archive contents and public downloads were checked; the [release record](../deployment/releases/first3-benchmark-20260927.json) preserves asset URLs and GitHub digests. The explicit local allowlist passed 154 synthetic tests and [GitHub CI](https://github.com/Seanwilliam2077/video-to-hoi/actions/runs/36312720815) passed. These checks establish preparation-tool behavior only. The status-page source was updated; its existing Claude online copy still requires republishing.
