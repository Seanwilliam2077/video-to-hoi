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

The [30-video progress board](https://video-to-hoi-progress.almondclock256.chatgpt.site) shows each video's stage deliveries and iteration history. This Site is currently owner-private; teammates do not automatically have access. The [Git progress registry](tracking/README.md) is the shared record: export browser drafts, import and validate them locally, then submit the resulting records and snapshot through Git. Unregistered video progress remains unknown.

## Model weights and data

Large files are GitHub Release attachments, not ordinary Git blobs or the repository's source ZIP. The current filtered handoff preserves model files and runtime sources while excluding upstream dataset/demo payloads. Use the [handoff guide](deployment/COMPLETION_HANDOFF.md) for download commands, extraction paths, licenses and the verified publication record.

```bash
# Track 1 videos and metadata -> data/v2d/track_1/
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/video_to_data_challenge', repo_type='dataset', local_dir='data/v2d', allow_patterns=['track_1/**'])"

# After restoring and combining the three baseline releases as described in the guide:
python tools/fetch_assets.py verify --root combined-assets --group all
```

The [all-30 Body input delivery](deliveries/platform/body-inputs-track1-20261016/README.md) is ready: 30 original videos, 16,563 human masks, per-video estimated cameras, exact timelines and sampled review evidence. [Download the verified archive](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/body-inputs-track1-20261016), then follow the [receiving guide](docs/body-input-handoff.md). The complete input gate, pinned native readers and a freshly extracted copy passed on the producer host. Body inference and execution on a second machine remain to be recorded. The [preparation guide](docs/body-input-preparation.md) explains reproduction and acceptance boundaries.

The completion handoff is published: 11 archives in 15 verified parts (16,873,320,178 bytes), plus five companion files. It adds seven original baseline model entries, host wheels, four Module 1 model packages, and separate CPython 3.10/3.11/3.12 runtime acquisition packages. The older FoundationPose and CARI4D/SAM3D releases supply the other five selected baseline model entries. Human/physics weights, SMPL-family access, restricted distributions, legacy candidate environments, container/APT/native builds, adapters and server validation remain outstanding. Those asset receipts do not establish runtime compatibility. The current input delivery executes SAM2 and MoGe2; human reconstruction, physics inference and new rankings remain outstanding.
