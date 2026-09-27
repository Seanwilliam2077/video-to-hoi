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

Large files are not in Git. The Track 1 data comes from Hugging Face; the model weights come from this repository's releases and the pinned upstream sources. Run from a clone:

```bash
# Track 1 videos and metadata -> data/v2d/track_1/
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/video_to_data_challenge', repo_type='dataset', local_dir='data/v2d', allow_patterns=['track_1/**'])"

# Source trees and the seven public models -> artifacts/
python tools/fetch_assets.py fetch --root artifacts --skip-gated

# FoundationPose weights (239 MB): extract the ZIP into artifacts/
gh release download foundationpose-addon-20260927 -R Seanwilliam2077/video-to-hoi -D downloads

# CARI4D, SAM 3D Body and SAM 3D Objects (18 GB in 11 parts): rebuild three ZIPs, then extract them into artifacts/
python tools/restore_model_release.py deployment/releases/hf-models-20260927-manifest.json --parts-dir downloads/hf-models --output-dir downloads/archives --download

# Check every model file
python tools/fetch_assets.py verify --root artifacts
```

The fetch step leaves FoundationPose and the three gated models incomplete; the two releases supply them. Restoring the three models needs about 54 GB free while parts, ZIPs and extracted files coexist. Read `MODEL_NOTICE.md` in the release before use. Details and licenses: [deployment/MODEL_RELEASES.md](deployment/MODEL_RELEASES.md).
