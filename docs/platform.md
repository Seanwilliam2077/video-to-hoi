# Remote platform and baseline deployment

Follow the owner's current [Track 1 requirements R1–R6](track1-requirements.md), effective 2026-10-03. Track 2 content and all derivatives are prohibited, including use in tests, diagnostics, baseline normalization, candidate comparisons and rankings. Use Track 1 inputs and their reconstructed artifacts; synthetic fixtures must be independently created. See `AGENTS.md`.

## Module 1 scope for the team discussion (2026-10-15)

[The platform plan](platform-plan.md) records seven capabilities, task/result records, four integration boundaries and three acceptance layers. The proposed order is one complete real input-to-export/reload path, then an actual two-host handoff, then fixed-baseline experiments and broader coverage. Roles float across tasks; no permanent module owner is needed. The meeting checklist covers the real sample, available execution capability, transfer boundary and completion evidence.

This is a documentation update. Existing internal delivery gates and the earlier 453-test record remain the implemented foundation; no new adapter, model execution, transfer implementation or accepted baseline is claimed. Live team-page publication remains pending.

## Automatic deliveries and independent module work (2026-10-15)

The internal v1 runner now applies [automatic delivery gates](delivery-gates.md) before and after each of
its six stages. Fresh attempts cannot borrow missing outputs from older runs. Typed reload checks cover
files, frames, finite geometry, scales, rigid transforms and internal export read-back. Verified input copies,
output hashes and exact predecessor receipts expose same-run changes and stop stale downstream work.
Failed/running attempts block consumers; reports are immutable and local checks never grant quality approval.

The read-only `python -m v2hoi.check` command checks a completed stage, prerequisites for a next stage,
or all six statuses. `tools/demo_delivery_gates.py` uses independent synthetic metadata to pass through five
separate module folders on one host, then alters tracking: export readiness fails and the consumer backend
is never started. This demonstrates local isolation and rejection, not a real two-host model run.

[team-handoffs.md](team-handoffs.md) and [the module capability map](../configs/team-modules.json) define independent
starting work, exact handoffs and consumer review scopes. Development, execution and review participants are chosen per task; no permanent module assignment is required. Use one
episode/experiment per run for isolation; replacing part of a multi-episode batch invalidates that batch's
receipt. Module branches remain unchanged. The map documents roles, not a scheduler.

Native binary schemas, transport/relocation, authentic source verification, decoder normalization and
real quality/export acceptance remain pending. The native distributed profile is still design-only and
unsupported stage names cannot bypass the internal gate. No model, challenge pixels, deployment or upload
was executed. Verification is recorded in [the delivery receipt](evidence/delivery-gates-validation-20261015.json).
The complete explicit CPU allowlist passed: 453 tests, with no failures, errors or skips on local Windows.
Local team/status source updates are complete; live publication remains pending.

The user's floating-role clarification is recorded in the module profile, handoff guide, workflow and agent instructions. There are no permanent module assignees. Task notes and separate branches/run directories support handover and parallel alternatives; they are not a distributed lock. This documentation correction preserves the native dependency graph and existing delivery gates.

## Independent-machine execution design (2026-10-13)

The owner reports different server access for each teammate. [The distributed execution design](distributed-execution.md)
and [non-executable profile](distributed-execution.json) remove the shared-host prerequisite. Source review identifies
eight existing native subprocesses, grouped conservatively into Human's 01-04, Object's 05, and Physics's 06-07.
Platform owns coherent scene import and export; rendering and the historical MHR conversion path need a compatible
GPU worker. Normalize native human parameters before scene publication; export then serializes that same scene
without another independent fit. Native stage 08 previews original bundles, while repaired scenes require an
explicitly compatible renderer. Standalone scale bootstrap and metric asset preparation remain earlier dependencies.

Git carries code, configuration and small immutable result records; common inputs and new stage outputs travel as
hashed packages. Each worker has independent local storage and an environment identity. The proposed transfer and
native adapters must verify dependency closure, relative-path materialization and actual predecessor consumption;
current v2 validators do not enforce these stage transitions. Original native evidence remains preserved.

This delivery changes documentation only. No deployment, new model run, artifact upload or runtime implementation
was performed. The historical sequence proposed pack/verify/import, native stage adapters, then a full episode on two
actual hosts. The 2026-10-15 [platform plan](platform-plan.md) now prioritizes a real import/export/reload path before
accepting cross-machine execution; capability and access are checked per task without permanent member assignments.
Source/profile/document and rendered-page checks accompanied this design; earlier 379 CPU tests remain
evidence for the unchanged foundation, not a new multi-host test. Live module/status publication remains pending.

## P0 audit and P1 foundation delivery (2026-10-13)

[component-audit.md](component-audit.md) records individual branch reuse decisions, exact upstream/patch identities and a provisional Windows reproduction candidate. [exposure-ledger.json](exposure-ledger.json) records all 30 episodes and ten object groups with historical evidence; external/manual use remains unknown. [validation-policy.json](validation-policy.json) defines structural and review rules while leaving empirical quality thresholds unfrozen.

The opt-in [v2 foundation](v2-foundation.md) now implements immutable content-addressed storage, UUID publication, atomic selection, typed evidence/scene/plan structural validation and numerical native-array/object-frame helpers. Validators check actual dependency bytes, exact video/mesh/origin/scale invocation bindings, fit/review lineage and trusted status-report identities. They cannot authenticate source-policy declarations, run a native decoder or certify reconstruction quality. The v1 runner and fake stages are unchanged.

A separate local CPU test environment was created for the explicit CI allowlist, including the three new independent synthetic suites. No model environment, GPU dependency, weights or challenge inputs were installed or loaded for this delivery. Final test counts and interpreter/package versions are recorded in [the foundation verification receipt](evidence/v2-foundation-validation-20261013.json). Tests establish covered storage/schema/math behavior, not completed native integration or a G0-G4 acceptance gate.

Next: restore the nominated runtime under a complete environment/source/model manifest; implement authoritative native import and conversion against the v2 profile; then obtain complete episodes 12/16 and early export/read-back evidence. Human review history, authentic decoder semantics, dependency scheduling and meaningful quality/export validators remain open. Live module/status publication remains pending; the current capability exposes no Claude Artifact publisher.

## Earlier pipeline design record (2026-10-13)

The repository design now separates module ownership from runtime jobs. Platform owns a dependency coordinator that pins the source timeline, preparation estimates, metric object asset and one native CARI4D invocation per episode. Human and Object own conversion and geometry validation of that same invocation; stage directories are views of its output, not permission to rerun independent human/camera estimates. Physics owns bounded diagnostic plans and repairs; Platform promotes only coherent, validated snapshots.

The proposed `EpisodeEvidence`, `SceneState` and `SolvePlan` specifications are in [contracts-v2.md](contracts-v2.md). The [architecture](design.md), [30-video coverage inventory](video-coverage.md), and [P0–P5 backlog](implementation-plan.md) define owners, dependencies, native round-trip evidence and stop/rollback rules. Execution validity, reconstruction acceptance and export validity remain separate decisions. The current runner and v1 contract are unchanged.

Before runtime work, audit reusable components on `origin/windows-cari4d` (`fb0bac9`) against current main (`0e86889696edc4d81a3c287cb93d76baece4a1d5`). Recorded native experiments exist, but provisional metric scale, unresolved pan orientation and invalidated contact after repairs prevent treating them as accepted baselines. Freeze one reviewed upstream/patch/environment combination; the older documented pin and later experimental pin must not be combined implicitly.

This update runs no model, installs no environment and changes no deployment. Next deliverables are the component/exposure audit, executable v2 contracts, and complete native import/export/read-back on episodes 12 and 16. Local page sources are updated; live module/status publication is pending as recorded in [page-publication.md](page-publication.md).

Design verification: all 30 inventory entries match the pinned gallery metadata (16,563 metadata frames), both recorded gallery hashes match, three illustrative JSON fragments parse, modified Markdown file targets exist, SVG XML parses, and `git diff --check` passes. Local browser checks cover the diagram, mobile status in light/dark themes with a ticking countdown, hub language switching and the portable reading copy. These are documentation checks; no runtime unit suite, source-video decoding, model inference or reconstruction-quality validation was performed for this update.

## Earlier planning update (2026-10-03)

The new plan prioritizes a verified native baseline over expanding the model or server inventory. [implementation-plan.md](implementation-plan.md) defines G0 official/native contract validation, G1 real episodes 16/12, G2 camera/scale evidence, G3 bounded repairs and G4 full-set export rehearsal. [evaluation.md](evaluation.md) records the now-available official kit: reference-relative acceleration, 22 body joints for ACC-H, and predicted hand penetration for PEN. Self-smoothness is only a diagnostic. The official adapter, proposed contract v2 and real baselines remain unimplemented or unverified.

For the first accepted run, preserve the upstream native MHR and object outputs, then verify conversion, frame identity, geometry and a reloaded export before comparing algorithms. Each candidate must have one immutable manifest and coherent human/mesh/motion artifacts. Freeze independent observations and inspect motion events, occlusion recovery and contact separation as well as penetration. `first3` remains a limited hoop integration fixture; grouped `dev-mini` review and all 30 clips are separate gates. Cross-episode evidence pooling is disabled pending organizer clarification, rather than treated as a mandatory shared-mesh rule.

Later runtime work should support only the selected baseline and demonstrated repairs first. Downloaded candidates need not all become installed environments. Record GPU time, peak resources, manual review and repair time per accepted clip so quality gains can be compared with total operating cost. The local reference diagnostic and synthetic tests do not establish a model run, deployed environment or submission.

Baseline subsets are development gates only. Record the first reference frame's explicit original-video ID and retain its human-derived alignment when evaluating later windows. External-kit row mapping is an implementation task; it does not reopen R1 or authorize a different fit.

## September acquisition record

The remaining host, download and publication details preserve the dated 2026-09-27/28 preparation record. They are not fresh compatibility checks or authorization to resume server work; current acceptance priorities are the G0–G4 plan above.

Status rechecked 2026-09-28: the original baseline's 12 required model entries
pass fresh local integrity checks. The [completion handoff](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/assets-completion-20260928) is
public: 11 archives in 15 verified parts (16,873,320,178 bytes), plus five
companion files. All 20 attachment hashes and public download paths passed the
[publication checks](../deployment/releases/assets-completion-20260928-github.json).
The handoff adds seven original baseline model entries and twelve filtered source
entries, the CPython 3.10 host wheelhouse, four Module 1 model packages, and
separate CPython 3.10/3.11/3.12 runtime acquisition packages. MoGe3 dependency
sources and partial human/physics candidate sources are included. Manual or
restricted human/physics weights, SMPL-family permissions, withheld distributions,
legacy candidate environments, base images, APT packages, native builds and
adapters remain outstanding. See the [completion handoff](../deployment/COMPLETION_HANDOFF.md)
for scope and limits; the [availability audit](../deployment/ASSET_AUDIT.md)
records an earlier snapshot.

The owner requested acquisition and GitHub handoff only. Deployment and model
execution remain deferred. The owner cannot connect to the server through SSH;
code and preparation materials will be transferred as ZIP files and used through
the server's available console. See [deployment/README.md](../deployment/README.md)
for preparation commands. The two real baselines have not been run; no actual
candidate quality results or rankings exist. Model inference, environment
installation, and acceptance tests will run on the remote host, not on the
project owner's local computer.
The owner supplied a screenshot of a
candidate host with eight RTX PRO 5000 72 GB GPUs. Remote access, CPU, host RAM,
storage, and container support are not yet verified. Two more server types,
L20 and A10, are also available; see below.

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
3. Preserve native MHR and object outputs for every frame, including occlusions;
   verify the now-available official kit's pose/scale/shape and mesh/transform
   conventions through a tested adapter. Label v1 exports internal until then;
   render and measure the exact reloaded candidate intended for submission.
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

Five recipes cover MoGe2, MoGe3, prompted SAM2, standalone SAM3D Body and joint CARI4D. SAM3 and YOLOE now have pinned checkpoints and filtered source packages in the completion handoff, but still need execution adapters and server validation. GVHMR/WHAM and SmoothNet/HTD-Refine/PhysPT remain research candidates with manual or restricted weight, environment and adapter blockers. The three temporal methods do not yet have MHR adapters. MoGe3 has a separately pinned checkpoint in the completion handoff, outside the prior 12-entry baseline inventory. No server environment is installed or certified by this kit.

The ranking tool groups comparable tasks and metrics, requires the same three complete episodes, and rejects missing or mismatched evidence. Its protocol remains unfrozen until runtime, upstream, and independent observations exist. The draft `self_acceleration` diagnostic requires reviewed video fidelity and is distinct from reference-relative ACC. Runs must declare Track 1 reconstruction provenance and no fake outputs; declarations do not independently prove origin or quality. Ambiguous acceleration labels, official metric claims and composite totals are rejected. No official score, actual measurements or ranking is supplied. Automatic quality extraction from native candidate outputs is still to be implemented.

The package keeps the original input bytes, recipe/protocol files, an empty result report and per-file checksums. Future execution must run on the selected Linux GPU machine. The previous two-baseline acceptance target remains pending; this additional hula-hoop screening does not replace it.

The [published first-three kit](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/first3-benchmark-20260927) is 68,971,507 bytes, SHA-256 `6b2eb93051415fc0a838622bdf5cdba6d8127bd3dc96c78d0f5f4c59c2b4c59b`, from source commit `d822121`. Archive contents and public downloads were checked; the [release record](../deployment/releases/first3-benchmark-20260927.json) preserves asset URLs and GitHub digests. The explicit local allowlist passed 154 synthetic tests and [GitHub CI](https://github.com/Seanwilliam2077/video-to-hoi/actions/runs/36312720815) passed. These checks establish preparation-tool behavior only. The status-page source was updated; its existing Claude online copy still requires republishing.
