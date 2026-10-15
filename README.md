# video-to-hoi

Monocular 4D human–object reconstruction for Track 1 of the NVIDIA Video to Data (V2D) Challenge. From one static RGB video and an object description, the planned pipeline recovers the person as MHR parameters, the object's metric mesh, and its 6D pose on every frame. The current submission kit evaluates CD-H/CD-O geometry, ACC-H/ACC-O reference-relative acceleration, and hand–object penetration (PEN). The leaderboard freezes on 2026-11-04 at 17:00 EST.

The plan starts with one real CARI4D baseline, then permits bounded corrections supported by independent video evidence. The owner's latest six requirements are the current specification. A native-MHR reference diagnostic checks reconstruction geometry and motion with explicit permitted inputs; it is not an official evaluator and supplies no hidden reference data. The official export adapter and real reconstruction backends remain to be implemented. Registered pipeline backends are still fake, with no measured reconstruction results.

![The video-to-hoi pipeline: six stages across four modules](docs/pipeline.svg)

**Track 1 only.** All reconstructed assets and estimated parameters, including cameras, must come from Track 1. Track 2 data and derivatives are prohibited for development, tuning, validation, tests, scorer checks and submissions. The current requirements, setup, diagnostics and tests are linked from the [developer guide](docs/development.md).

## Team hub

**[Open the team hub](https://claude.ai/artifact/JAeDzfVxpRezsW73C26M4S)** for the project status, Track 1 reference and four module pages. Post work updates in each module page's **Progress** section; use Tasks for current work and Resources for reading. Project status remains available inside the hub. Roles float: any member may contribute across modules.

## Module submissions

| Module | Code area under `src/v2hoi/` | Delivery records and review |
|---|---|---|
| ① Platform & perception | `stages/inputs.py`, `stages/export.py`, shared contracts/checks and `v2/` | [deliveries/platform/](deliveries/platform/) |
| ② Human | `stages/human.py` and related adapters | [deliveries/human/](deliveries/human/) |
| ③ Object | `stages/objects.py`, `stages/motion.py` | [deliveries/object/](deliveries/object/) |
| ④ Temporal & physics | `stages/refine.py` and related adapters | [deliveries/physics/](deliveries/physics/) |

Submit code and a small delivery record through a task branch and PR to `main`. Each record lives at `deliveries/<module>/<unique-delivery-id>/README.md`; include output locations, exact versions, check evidence and remaining work. Use the [submission guide and template](deliveries/README.md). Large outputs belong in a new result-specific Release or agreed team storage, with hashes and exact download references. A progress post or merged record does not by itself accept a model result or start the next stage.

## Model weights and data

Large files are GitHub Release attachments, not ordinary Git blobs or the repository's source ZIP. The current filtered handoff preserves model files and runtime sources while excluding upstream dataset/demo payloads. Use the [handoff guide](deployment/COMPLETION_HANDOFF.md) for download commands, extraction paths, licenses and the verified publication record.

```bash
# Track 1 videos and metadata -> data/v2d/track_1/
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/video_to_data_challenge', repo_type='dataset', local_dir='data/v2d', allow_patterns=['track_1/**'])"

# After restoring and combining the three baseline releases as described in the guide:
python tools/fetch_assets.py verify --root combined-assets --group all
```

The completion handoff is published: 11 archives in 15 verified parts (16,873,320,178 bytes), plus five companion files. It adds seven original baseline model entries, host wheels, four Module 1 model packages, and separate CPython 3.10/3.11/3.12 runtime acquisition packages. The older FoundationPose and CARI4D/SAM3D releases supply the other five selected baseline model entries. Human/physics weights, SMPL-family access, restricted distributions, legacy candidate environments, container/APT/native builds, adapters and server validation remain outstanding. No models have been run and no new rankings are available.
