# video-to-hoi

Monocular 4D human–object reconstruction for Track 1 of the NVIDIA Video to Data (V2D) Challenge. From one static RGB video and a text description of the object, the pipeline recovers the person (body and hands, as MHR parameters), the object's metric mesh, and the object's 6D pose on every frame, in one metric world frame. Kaggle scores five metrics: CD-H and CD-O for accuracy, ACC-H and ACC-O for smoothness, and PEN for human–object penetration. The leaderboard freezes on 2026-11-04 at 17:00 EST.

![The video-to-hoi pipeline: six stages owned by four modules](docs/pipeline.svg)

**Track 1 only.** No Track 2 data (Tier 1 or Tier 2) is used for development, tuning, validation, tests or submissions. Setup, running, validation and tests are in the [developer guide](docs/development.md).

## Project status

Progress in plain Chinese, for leadership and product: **[open the status page](https://claude.ai/artifact/U2vgULC1h3DurtZwyQRHS1)**.

## Team pages

**[Open the team hub](https://claude.ai/artifact/JAeDzfVxpRezsW73C26M4S)**. It leads to the Track 1 reference (rules, scoring, the 30 videos, organizer Q&A) and to the four module pages (tasks, progress, reading list). Every page carries the same navigation bar, so you can move between them directly.

| Module | Stages | Metrics | Branch |
|---|---|---|---|
| ① Platform & perception | inputs, export | all five, as gatekeeper | `platform` |
| ② Human | human | CD-H | `human` |
| ③ Object | objects, motion | CD-O | `object` |
| ④ Temporal & physics | refine | ACC-H, ACC-O, PEN | `physics` |

## Model weights and data

Large files are GitHub Release attachments, not ordinary Git blobs or the repository's source ZIP. The current filtered handoff preserves model files and runtime sources while excluding upstream dataset/demo payloads. Use the [handoff guide](deployment/COMPLETION_HANDOFF.md) for download commands, extraction paths, licenses and the verified publication record.

```bash
# Track 1 videos and metadata -> data/v2d/track_1/
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/video_to_data_challenge', repo_type='dataset', local_dir='data/v2d', allow_patterns=['track_1/**'])"

# After restoring and combining the three baseline releases as described in the guide:
python tools/fetch_assets.py verify --root combined-assets --group all
```

The completion handoff is published: 11 archives in 15 verified parts (16,873,320,178 bytes), plus five companion files. It adds seven original baseline model entries, host wheels, four Module 1 model packages, and separate CPython 3.10/3.11/3.12 runtime acquisition packages. The older FoundationPose and CARI4D/SAM3D releases supply the other five selected baseline model entries. Human/physics weights, SMPL-family access, restricted distributions, legacy candidate environments, container/APT/native builds, adapters and server validation remain outstanding. No models have been run and no new rankings are available.
