# video-to-hoi

Recover 3D human and object motion from a single third-person video, for Track 1 of the NVIDIA Video to Data (V2D) Challenge.

The input is a video from one static RGB camera and a description of the object to track. The output is the human body, the object's shape, and the object's pose in every frame, at metric scale in one world frame.

Architecture, trade-offs, and open questions are in [docs/design.md](docs/design.md).

## Status

- Design: `docs/design.md`
- Local Tier 1 scorer: `python -m v2hoi.score`, working
- Test pipeline: `python -m v2hoi.pipeline --out work/demo` runs the object and video layers end to end with stand-in backends. No reconstruction models are wired in yet.

## Setup

Requires Python 3.10+ and a CUDA build of PyTorch already installed. The venv below reuses the system torch instead of installing a CPU build:

```bash
uv venv .venv --python 3.10 --system-site-packages
uv pip install --python .venv/Scripts/python.exe numpy scipy pandas pyarrow trimesh huggingface_hub pytest warp-lang rtree cholespy usd-core
uv pip install --python .venv/Scripts/python.exe --no-deps "py-soma-x==0.2.1"
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
```

On first use, `py-soma-x` downloads the SOMA-X assets from Hugging Face, pinned to the revision the official toolkit uses.

## Data

```bash
.venv/Scripts/python.exe -m v2hoi.download
```

Downloads into `data/v2d/`: Track 2 Tier 1 ground truth (human, object poses, meshes, ground planes), the Tier 2 noisy labels, and Track 1 metadata, about 400 MB. `--videos` also fetches the Tier 1 and Track 1 videos.

## Scoring

A prediction root has the Tier 1 layout: `meta/info.json`, `meta/episodes_metadata.jsonl`, `data/chunk-000/episode_XXXXXX.parquet` (Tier 1 columns), and `mesh/<object>/<object>.glb`.

```bash
.venv/Scripts/python.exe -m v2hoi.score --pred <prediction root>
```

The first five columns of the table are the Track 1 leaderboard metrics (CD-H, CD-O, ACC-H, ACC-O, PEN) in cm, as on Kaggle; the rest are diagnostics. Below the table, the scorer says whether the run used the official settings and whether the prediction is a valid submission, with the reasons if not. The full report is written to `scores/`.

Options: `--episodes 7 9` scores a subset (by default every episode is required); `--stride 3` evaluates the per-frame mesh metrics on every third frame; `--align first|se3|sim3|none` picks the alignment (default `first`: one Sim(3) on the first frame's body joints, the official rule); `--pred-mesh-dir` looks for prediction meshes elsewhere; `--strict` exits with status 1 unless the settings are official and the submission is valid, so run it before submitting to Kaggle.

What the official submission is, and which rules the scorer enforces: `docs/design.md`, sections 6 and 5.1.

Two self-checks:

```bash
.venv/Scripts/python.exe -m v2hoi.score --pred data/v2d/track_2/tier_1_multiview_caption
```

```bash
.venv/Scripts/python.exe -m v2hoi.score --pred data/v2d/track_2/tier_2_synthetic_noise --pred-mesh-dir data/v2d/track_2/tier_1_multiview_caption/mesh
```

The first scores the ground truth against itself, so every error should be close to 0. The second scores the organizer's Tier 2 labels, noise sampled from Track 1 error distributions, as a reference. Both are reported as invalid submissions because their meshes are the reference meshes. Metric definitions: `docs/design.md`, section 5.

## Tests

```bash
.venv/Scripts/python.exe -m pytest
```
