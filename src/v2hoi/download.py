"""Download the parts of the V2D challenge dataset that this project uses.

Default: Track 2 Tier 1 labels (poses, meshes, ground planes, metadata), the
Tier 2 noisy labels, and Track 1 metadata. Videos are large and optional.

    python -m v2hoi.download                 # labels only (~100 MB)
    python -m v2hoi.download --videos        # also Tier 1 and Track 1 videos
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "nvidia/video_to_data_challenge"
DEFAULT_ROOT = Path("data/v2d")

TIER1 = "track_2/tier_1_multiview_caption"
TIER2 = "track_2/tier_2_synthetic_noise"
TRACK1 = "track_1"


def patterns(videos: bool) -> list[str]:
    pats = [
        f"{TIER1}/data/**",
        f"{TIER1}/meta/**",
        f"{TIER1}/mesh/**",
        f"{TIER1}/ground_plane/**",
        f"{TIER1}/README.md",
        f"{TIER2}/data/**",
        f"{TIER2}/meta/**",
        f"{TIER2}/README.md",
        f"{TRACK1}/meta/**",
        f"{TRACK1}/data/**",
        f"{TRACK1}/README.md",
    ]
    if videos:
        pats += [f"{TIER1}/videos/**", f"{TRACK1}/videos/**"]
    return pats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(args.root),
        allow_patterns=patterns(args.videos),
    )
    print(path)


if __name__ == "__main__":
    main()
