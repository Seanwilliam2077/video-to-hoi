# First-three-video benchmark handoff

**Prepared, not run.** The owner deferred execution until a remote entry point is available. This package does not install environments or run models during preparation. It is an execution and reporting kit, not a validated GPU environment or a completed SOTA comparison.

## Frozen input

Track 1 episodes **0, 1, 2**, in numeric order: all are hula-hoop videos from the same static camera. Their full lengths are **790, 668, 866 frames** (2,324 total), 30 fps, 1536 x 1152. All frames were decoded on CPU solely to check the input. Video SHA-256 and byte counts are in `benchmarks/first3/episodes.json`. Files are renamed to `episode_000000.0.color.mp4` etc. without changing bytes, as required by CARI4D. This subset tests a difficult thin object but does not represent all objects or all 30 episodes. No Track 2 file, reference mesh, trajectory, or derived value is included.

## Candidate scope and readiness

| Module / comparison | Candidate | What this package supplies | Still required before measurement |
| --- | --- | --- | --- |
| 1, depth | MoGe2 and MoGe3 | Common full-video adapter; pinned source; both checkpoints downloaded and verified | GPU environments; independent depth evidence for accuracy claims |
| 1, prompted masks | SAM2.1 | Pinned official video wrapper recipe | Identical Track 1 prompts and audited human/object IDs; built image |
| 1, text masks | SAM3 and YOLOE | Pinned filtered source and verified weights; separate research-only MobileCLIP package | Output adapters, target environments, fixed text prompts and evaluation masks |
| 2, independent human | SAM3D Body | Pinned official MHR recipe | Track 1 human masks, common intrinsics, built image |
| 2, joint HOI | CARI4D | Pinned official MHR/HOI recipe | Track 1-derived object mesh and two-class masks; built image |
| 4, temporal refinement | SmoothNet, HTD-Refine, PhysPT | Pinned research candidates and comparison gates | Real frozen upstream trajectories, weights, MHR adapters, independent motion evidence |

The kit has **five executable recipes**, plus research candidates that deliberately have no execution command until their adapters are implemented. The module 4 adapter work is not complete. SmoothNet is a mature baseline; public availability does not establish current SOTA or performance on this subset. GVHMR and WHAM are additional human research candidates in `human_recipes.json`; their SMPL-family outputs and separately licensed assets need preparation and alignment before MHR comparison.

Acquisition was expanded on 2026-09-28. [Module 1 receipts](../../deployment/expanded-platform.json)
cover MoGe3, SAM3, YOLOE and MobileCLIP. [Human/physics inventory](../../deployment/expanded-human-physics.json)
covers eight pinned source versions, but their Drive/Dropbox weights remain manual
and restricted body assets cannot be publicly redistributed. See the
[handoff guide](../../deployment/COMPLETION_HANDOFF.md) for package locations and remaining work.
Downloaded weights and source archives do not establish installed environments or working adapters.

SAM3D Body and CARI4D are separate comparison groups: CARI4D uses extra object evidence and the Body initializer, so this is an ablation or end-to-end comparison, not two independent human models. Hold module 3's Track 1 reconstruction fixed when later comparing module 4. Do not use demonstration meshes from another dataset.

Official sources: [MoGe](https://github.com/microsoft/MoGe), [SAM2 toolkit wrapper](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_sam2/docker/run_video_to_masks.py), [SAM3](https://github.com/facebookresearch/sam3), [YOLOE](https://github.com/THU-MIG/yoloe), [Body wrapper](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_sam3d_body/docker/run_estimate_mhr_params.py), [CARI4D](https://github.com/nvidia-isaac/video_to_data/blob/33129dd0f2d2dcfd1164d43fd076542660756ed2/reconstruction/modules/v2d_cari4d/README.md).

[SmoothNet](https://github.com/cure-lab/SmoothNet) smooths temporal channels and does not model object contact. [HTD-Refine](https://github.com/ant-research/HTD-Refine) uses video motion evidence but currently optimizes SMPL/SMPL-X. [PhysPT](https://github.com/zhangy76/PhysPT) operates on SMPL body/ground physics. None supplies a drop-in MHR human-object refinement adapter. Their code and model licenses differ; see the recipe registries. Codex has not accessed the organization-blocked Drive download routes.

## Prepare on the selected server later

1. Transfer this package and the existing combined asset handoff. The base public-assets archive, FoundationPose add-on, and three [HF model release](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/hf-models-20260927) archives remain separate. No weights or environments are duplicated in this test kit.
2. Use the chosen server's compatible CUDA/PyTorch stack. Keep MoGe and the upstream Docker modules in separate environments. Pin MoGe source to `74fbce054ebed49800de42d0ad0e83495065719a`; pin the toolkit to `33129dd0f2d2dcfd1164d43fd076542660756ed2`. Source ZIPs and mutable container tags do not by themselves prove the running image matches those sources. Record the installed versions and image digests in the run's environment identity.
3. Copy `benchmarks/first3/runtime.example.json` to `runtime.json`, then replace the paths and fill in `machine_id` / `environment_id`. Use a separate Python path per candidate if needed. On the host, follow the pinned toolkit's `reconstruction/README.md`: install the selected module `docker/` orchestration packages and their declared dependencies, then build their images. This kit does not automatically install them.
4. MoGe2 uses the previously prepared local checkpoint. MoGe3 is now downloaded and verified; its extra package contains `models/moge3-vitl/model.pt` and source `sources/moge/` at the comparison's required revision. Set `paths.moge3_checkpoint` to that extracted checkpoint. `extra_assets.json` records its identity. The original 12-model completion claim remains scoped to the baseline inventory.

Optional explicit MoGe3 download on an approved machine:

```bash
hf download Ruicheng/moge-3-vitl model.pt --revision 184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be --local-dir /path/to/moge3
```

Prepare and audit shared inputs under `shared/episode_000000/` etc.: `metadata.json` (Track 1 episode, source video hash, producer revision, input/output hashes), `prompts.json`, `human_masks/`, `intrinsics.json`; CARI4D additionally needs `object_masks/`, the official two-class `masks.h5` schema, and metric `object_mesh.glb`. Prompt IDs are human **0**, hoop **1**. Do not put invented boxes, masks, cameras, or meshes in these files to make preflight pass. Use the pinned upstream datatypes; this package does not convert PNG masks to CARI4D's H5 schema or estimate the object mesh.

## Plan, then run

From the extracted package root, planning only verifies the frozen videos and reports commands/missing paths. It does not run Python environments, Docker, or inference:

```bash
python tools/benchmark_first3.py plan --package . --runtime runtime.json --candidate moge2 --output runs/first3-001
```

Later, on the configured **Linux GPU server**, execute a selected candidate over all three complete videos:

```bash
python tools/benchmark_first3.py run --package . --runtime runtime.json --candidate moge2 --output runs/first3-001
```

Other executable IDs are `moge3`, `sam2_prompted`, `sam3d_body_mhr`, and `cari4d_joint_mhr`. Run serially on the same GPU, use a new run directory for each repeat, and freeze the same prompt/camera/upstream inputs. Each subprocess has a default two-hour timeout; change `--timeout` explicitly if needed. Failures retain logs and `execution.json`; completed outputs are never silently reused. Model-loading time is included in wall time. Report warm-up, peak memory and video-by-video times separately. Environment variables request offline HF access; this does not guarantee upstream containers are fully network-isolated.

The MoGe adapter keeps raw numerical depth, valid-pixel masks and normalized intrinsics per frame. Invalid background depth may be infinite; valid-mask non-finite values are reported separately. Plan substantial output space: full-resolution depth for 2,324 frames alone is about 16.5 GB uncompressed per candidate, before masks and other artifacts. File existence and process exit 0 only mean execution completed; neither establishes quality or full-frame validity for third-party formats.

## Scores and ranking

No official Track 1 total can currently be calculated locally from public reconstruction ground truth. Do not run the legacy `v2hoi.score`. `rank_first3.py` produces **internal, per-metric rankings**, grouped by comparable task and inputs; it never combines depth, human, masks and physics into one number.

- Depth: timing and valid-pixel fraction are engineering diagnostics, not depth accuracy. Accuracy needs independent geometric evidence. No reference depth is manufactured here.
- Segmentation: object mask IoU needs frozen, independently reviewed Track 1 labels. Use identical annotation frames and visibility policy. Prompted and text-only methods are separate groups.
- Human: keypoint reprojection needs a fixed joint mapping and independently reviewed Track 1 2D observations. No hidden 3D ground truth is implied. Detection predictions must be labeled as predicted observations, not labels.
- Module 4: report image fit and acceleration separately. Acceleration uses common Cartesian geometry, units, fps, and frame coverage, not mixed body parameter vectors. Before an acceleration ranking, a reviewer must inspect independent video evidence for every episode against the same raw baseline, including visibly moving intervals. Test a copied-first-frame negative control: a frozen trajectory must fail video/motion agreement. No contact score is supplied without valid frozen object/contact evidence.

The included protocol is intentionally **`frozen: false`**. Freeze evaluator code/config, observation hashes, upstream hashes, hardware, budgets and candidate adaptations before looking at scores, then set it true. Every result must cover all three episodes with matching frame counts and video hashes. Missing/failed/skipped data stay unranked, never zero-filled. Each metric uses an equal-weight mean over the three episodes; exact ties receive the same rank. A lone method is not evidence of superiority.

The package does not yet extract comparable quality metrics automatically from all native outputs. The module owner must implement and validate that evaluator, then populate `results.json` from its actual measurements and evidence. The ranker validates submitted records and comparison identities; it cannot independently prove a submitted scalar or reviewer statement is true. Its full result schema is documented in `tools/rank_first3.py`.

```bash
python tools/rank_first3.py --protocol benchmarks/first3/protocol.json --results results.json --output-json ranking.json --output-md ranking.md
```

The shipped empty results produce **no measured score and no ranking**. When the server is available, first complete one baseline and its overlays, audit format/frame coverage, then add compatible candidate comparisons. Report conclusions as first-three-video evidence only.

## Rebuild the handoff from the project checkout

```bash
python tools/benchmark_first3.py prepare --dataset-root data/v2d/track_1 --output dist/first3-benchmark
```

This copies only the three hash-matched videos and relevant metadata, recipe/protocol files and three tool scripts. It does not fetch models, run inference, or read Track 2 data. `package-manifest.json` records every shipped file. Preparation and synthetic tests are separate from real model validation.
