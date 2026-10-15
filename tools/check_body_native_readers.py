"""Exercise the pinned Body RGB, mask and camera readers without model inference.

Run the complete handoff gate first. This compatibility check imports only the
upstream common package, decodes every original RGB/mask pair, and loads the
native camera JSON. It does not load Body weights or certify its GPU runtime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import types

PIN = "33129dd0f2d2dcfd1164d43fd076542660756ed2"
COMMON = "reconstruction/modules/v2d_common"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_usable_mask_bbox(mask) -> None:
    """Match pinned Body's exclusive-max bbox and greater-than-one-pixel rule."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    require(len(xs) > 0 and int(xs.max()) + 1 - int(xs.min()) > 1
            and int(ys.max()) + 1 - int(ys.min()) > 1,
            "native Body would reject mask bbox and use full-image fallback")


def load_readers(root: Path):
    root = root.resolve(strict=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    require(git("rev-parse", "HEAD") == PIN, "upstream source is not the selected Body revision")
    require(not git("status", "--porcelain", "--untracked-files=no", "--", COMMON), "upstream common source is modified")
    directory = root / COMMON
    tracked = set(git("ls-files", "--", COMMON).splitlines())
    files = sorted(directory.rglob("*.py"))
    require(bool(files), "upstream common source is unavailable")
    require(all(item.relative_to(root).as_posix() in tracked for item in files), "untracked Python file in upstream common source")
    package_name = "_hoi_pinned_native_common"
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    video = importlib.import_module(package_name + ".video")
    datatypes = importlib.import_module(package_name + ".datatypes")
    identity = {"repository": "nvidia-isaac/video_to_data", "revision": PIN,
                "files": [{"path": item.relative_to(root).as_posix(), "sha256": sha256(item)} for item in files]}
    return video.FrameSource, datatypes.CameraIntrinsics, identity


def check(args) -> dict:
    import numpy as np

    prior = json.loads(args.handoff_checks.read_text(encoding="utf-8"))
    require(prior.get("schema_version") == "body_handoff_checks.v1" and prior.get("all_30_body_inputs_verified") is True,
            "a passing all-30 handoff gate report is required first")
    prior_episodes = {row["episode_index"]: row for row in prior["episodes"]}
    require(set(prior_episodes) == set(range(30)), "handoff gate report does not cover all 30 episodes")
    FrameSource, CameraIntrinsics, source_identity = load_readers(args.upstream_root)
    packages = {}
    for name in ("numpy", "Pillow", "scipy", "av", "imageio", "tqdm", "h5py", "opencv-python", "opencv-python-headless"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    import cv2
    report = {"schema_version": "body_native_reader_checks.v1", "checked_at": datetime.now(timezone.utc).isoformat(),
              "handoff_checks_sha256": sha256(args.handoff_checks), "upstream_common_source": source_identity,
              "environment": {"python": sys.version, "packages": packages, "cv2": cv2.__version__},
              "all_30_native_readers_passed": False, "body_model_environment_verified": False,
              "body_execution_completed": False, "episodes": []}
    for index in range(30):
        directory = args.package_root / f"episode_{index:06d}"
        result = {"episode_index": index, "passed": False, "errors": []}
        try:
            require(sha256(directory / "input.json") == prior_episodes[index]["input_sha256"], "input identity changed after handoff gate")
            require(sha256(directory / "review.json") == prior_episodes[index]["review_sha256"], "review changed after handoff gate")
            identity = json.loads((directory / "input.json").read_text(encoding="utf-8"))
            for name in ("intrinsics.json", "timeline.json", "masks-manifest.json"):
                require(sha256(directory / name) == identity["artifacts"][name], f"{name} changed after handoff gate")
            timeline = json.loads((directory / "timeline.json").read_text(encoding="utf-8"))
            masks_manifest = json.loads((directory / "masks-manifest.json").read_text(encoding="utf-8"))
            count, width, height = (timeline[key] for key in ("frame_count", "width", "height"))
            expected_video_path = f"videos/chunk-000/observation.images.exo_camera/episode_{index:06d}.mp4"
            require(identity["source"]["video_path"] == expected_video_path, "unexpected source video path")
            video_path = args.dataset_root / expected_video_path
            require(sha256(video_path) == identity["source"]["video_sha256"], "video changed after handoff gate")
            intrinsics_path = directory / "intrinsics.json"
            camera = CameraIntrinsics.load(str(intrinsics_path))
            matrix = camera.to_matrix()
            require(matrix.shape == (3, 3) and np.isfinite(matrix).all() and matrix[0, 0] > 0 and matrix[1, 1] > 0,
                    "native CameraIntrinsics produced invalid matrix")
            require((camera.width, camera.height) == (width, height), "native camera dimensions mismatch")
            with FrameSource.from_path(video_path) as rgb, FrameSource.from_path(directory / "human_masks") as masks:
                expected_stems = [f"{frame:06d}" for frame in range(count)]
                require(rgb.n_frames == masks.n_frames == count, "native reader frame counts differ")
                require(rgb.stems == masks.stems == expected_stems, "native reader frame stems differ")
                require(rgb.image_size == masks.image_size == (width, height), "native reader image sizes differ")
                decoded = 0
                for frame_id, (image, mask) in enumerate(zip(rgb.iter_frames(), masks.iter_frames(), strict=True)):
                    require(frame_id < count, "native reader returned excess frames")
                    require(image.shape == (height, width, 3) and image.dtype == np.uint8, "native RGB frame layout differs")
                    require(mask.shape == (height, width), "native mask frame layout differs")
                    require(np.all((mask == 0) | (mask == 255)) or np.all((mask == 0) | (mask == 1)), "native mask is not binary")
                    require(bool(np.any(mask)) and not bool(np.all(mask)), "native mask has no usable localization")
                    require_usable_mask_bbox(mask)
                    expected_name = f"human_masks/{frame_id:06d}.png"
                    entry = masks_manifest["files"][frame_id]
                    require(entry["path"] == expected_name and sha256(directory / expected_name) == entry["sha256"],
                            "mask changed after handoff gate")
                    decoded += 1
                require(decoded == count, "native reader terminated early")
            result.update(passed=True, frames_verified=decoded, width=width, height=height,
                          camera_matrix_dtype=str(matrix.dtype), input_sha256=sha256(directory / "input.json"))
        except (ValueError, OSError, KeyError, TypeError, IndexError, RuntimeError) as error:
            result["errors"].append(str(error))
        report["episodes"].append(result)
        print(f"episode_{index:06d}: {'NATIVE READERS PASS' if result['passed'] else 'BLOCKED'}", flush=True)
    report["all_30_native_readers_passed"] = all(row["passed"] for row in report["episodes"])
    report["frames_verified"] = sum(row.get("frames_verified", 0) for row in report["episodes"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--handoff-checks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        require(not args.output.exists(), "output report already exists")
        report = check(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (ValueError, OSError, KeyError, TypeError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Native reader compatibility check failed: {error}\n")
    return 0 if report["all_30_native_readers_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
