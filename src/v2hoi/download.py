"""Download only the Track 1 inputs used by this project.

    python -m v2hoi.download             # Track 1 metadata and input tables
    python -m v2hoi.download --videos    # also Track 1 videos

Pass --revision with a dataset commit to pin a reproducible snapshot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "nvidia/video_to_data_challenge"
DEFAULT_ROOT = Path("data/v2d")

TRACK1 = "track_1"


def patterns(videos: bool) -> list[str]:
    pats = [
        f"{TRACK1}/meta/**",
        f"{TRACK1}/data/**",
        f"{TRACK1}/README.md",
    ]
    if videos:
        pats.append(f"{TRACK1}/videos/**")
    return pats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--revision", help="Hugging Face dataset commit or tag")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(args.root),
        allow_patterns=patterns(args.videos),
        revision=args.revision,
    )
    print(path)


if __name__ == "__main__":
    main()
