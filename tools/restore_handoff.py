"""Restore verified ZIP parts from the project's explicitly approved handoff releases.

Uses only the Python standard library. Does not extract ZIPs, install dependencies,
load models, or contact upstream model/data hosts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

if __package__:
    from . import restore_model_release as core
else:
    # Standalone Release downloads and file-based test imports have no package.
    spec = importlib.util.spec_from_file_location(
        "restore_model_release", Path(__file__).with_name("restore_model_release.py")
    )
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)


RELEASES = {
    "https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/hf-models-20260927":
        "/Seanwilliam2077/video-to-hoi/releases/download/hf-models-20260927/",
    "https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/assets-completion-20260928":
        "/Seanwilliam2077/video-to-hoi/releases/download/assets-completion-20260928/",
}


def restore(manifest: Path, parts_dir: Path, output_dir: Path, *, download: bool = False) -> list[dict]:
    manifest = Path(manifest)
    if core._has_linked_component(manifest) or not manifest.is_file():
        raise core.RestoreError("manifest is missing or linked")
    try:
        page = json.loads(manifest.read_text(encoding="utf-8")).get("release_url")
    except (ValueError, AttributeError) as error:
        raise core.RestoreError("invalid handoff manifest") from error
    if not isinstance(page, str) or page not in RELEASES:
        raise core.RestoreError("manifest does not name an approved project release")
    old_page, old_path = core.RELEASE_PAGE, core.RELEASE_PATH
    try:
        core.RELEASE_PAGE, core.RELEASE_PATH = page, RELEASES[page]
        return core.restore(manifest, parts_dir, output_dir, download=download)
    finally:
        core.RELEASE_PAGE, core.RELEASE_PATH = old_page, old_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="download only missing approved GitHub parts")
    args = parser.parse_args(argv)
    try:
        results = restore(args.manifest, args.parts_dir, args.output_dir, download=args.download)
    except (OSError, core.RestoreError) as error:
        parser.exit(2, f"restore error: {error}\n")
    print(json.dumps({"archives": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
