"""Render source-bound, target-cropped original/overlay sheets for Body review.

CPU only. This adds QA evidence without changing masks, provenance, or existing
review sheets. A visual reviewer must inspect the generated evidence separately.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import uuid

from prepare_body_masks import (REPO, cleanup_owned_directory, file_entry, mask_observation,
                                now, parse_episodes, raw_data, verified_video, workspace_path,
                                write_json)
from prepare_track1_data import reject_links, safe_relative, sha256


def crop_bounds(bbox, width, height):
    if bbox is None:
        return (0, 0, width, height)
    left, top, right, bottom = bbox
    px, py = max(24, (right - left) * .3), max(24, (bottom - top) * .15)
    return (max(0, math.floor(left - px)), max(0, math.floor(top - py)),
            min(width, math.ceil(right + px)), min(height, math.ceil(bottom + py)))


def check_files(directory, entries):
    paths = set()
    for entry in entries:
        relative = safe_relative(entry["path"])
        if relative in paths:
            raise ValueError("duplicate manifest paths")
        paths.add(relative)
        path = directory / relative
        reject_links(path)
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"QA input/output hash check failed: {path}")


def actor_panel(original, mask, bbox, frame_id, episode_index):
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    crop = crop_bounds(bbox, original.width, original.height)
    rgb = np.asarray(original).copy()
    rgb[mask] = (rgb[mask].astype(np.float32) * .55 + np.array([0, 230, 110]) * .45).astype(np.uint8)
    overlay = Image.fromarray(rgb)
    panel = Image.new("RGB", (640, 512), "#f6f8fa")
    for col, image in enumerate((original, overlay)):
        fitted = ImageOps.contain(image.crop(crop), (312, 472), Image.Resampling.LANCZOS)
        panel.paste(fitted, (col * 320 + 4 + (312 - fitted.width) // 2, 24 + (472 - fitted.height) // 2))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 5), f"EP{episode_index:02d} frame {frame_id:04d} | Original", fill="black")
    draw.text((328, 5), "Actual mask overlay", fill="black")
    draw.text((8, 496), f"Source crop: {crop}" + (" | EMPTY MASK" if bbox is None else ""), fill="black")
    return panel


def write_page(panels, path):
    from PIL import Image
    page = Image.new("RGB", (1280, 512 * math.ceil(len(panels) / 2)), "white")
    for index, panel in enumerate(panels):
        page.paste(panel, (index % 2 * 640, index // 2 * 512))
    page.save(path, quality=93, subsampling=0)


def render_episode(args, manifest, root, episode):
    import av
    import numpy as np
    from PIL import Image
    index = episode["episode_index"]
    directory = workspace_path(args.inputs / f"episode_{index:06d}")
    video, video_hash = verified_video(manifest, root, episode)
    mask_manifest_path = directory / "masks-manifest.json"
    mask_manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
    expected = [f"human_masks/{i:06d}.png" for i in range(episode["frame_count"])]
    if (mask_manifest.get("schema_version") != "body_masks.v1"
            or mask_manifest.get("episode_index") != index or mask_manifest.get("video_sha256") != video_hash
            or [f["path"] for f in mask_manifest["files"]] != expected):
        raise ValueError("mask manifest does not match source and complete frame coverage")
    check_files(directory, mask_manifest["files"] + mask_manifest["artifacts"])
    observations_path, provenance_path = directory / "observations.json", directory / "masks-provenance.json"
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    frames = provenance["visual_review_frame_ids"]
    if (not frames or frames != sorted(set(frames))
            or any(type(i) is not int or i < 0 or i >= episode["frame_count"] for i in frames)
            or [f["frame_id"] for f in observations["frames"]] != list(range(episode["frame_count"]))):
        raise ValueError("invalid or incomplete source frame indices")
    timeline_path = args.manifest.parent / safe_relative(episode["timeline_path"])
    if sha256(timeline_path) != episode["timeline_sha256"]:
        raise ValueError("source timeline hash changed")
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    qa = workspace_path(directory / "mask-qa")
    qa.mkdir(exist_ok=True)
    output_manifest = qa / "targets-manifest.json"
    identity = {"episode_index": index, "video_sha256": video_hash,
                "masks_manifest_sha256": sha256(mask_manifest_path), "frame_ids": frames,
                "observations_sha256": sha256(observations_path), "provenance_sha256": sha256(provenance_path)}
    if output_manifest.exists():
        existing = json.loads(output_manifest.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("existing target QA was rendered from different inputs")
        check_files(directory, existing["files"])
        print(f"EP{index:02d}: existing target QA hashes verified; skipped", flush=True)
        return
    if list(qa.glob("targets-*.jpg")):
        raise ValueError("partial target QA exists; refusing to overwrite evidence")
    stage = directory / f".target-qa-{uuid.uuid4().hex}"
    stage.mkdir()
    selected, panels, page_frames, outputs, seen = set(frames), [], [], [], []
    count = 0
    with av.open(str(video)) as container:
        for frame_id, frame in enumerate(container.decode(video=0)):
            if (frame_id >= episode["frame_count"] or frame.pts != timeline["pts"][frame_id]
                    or (frame.width, frame.height) != (episode["width"], episode["height"])):
                raise ValueError(f"source decode disagrees with timeline at frame {frame_id}")
            count += 1
            if frame_id not in selected:
                continue
            with Image.open(directory / f"human_masks/{frame_id:06d}.png") as loaded:
                pixels = np.asarray(loaded)
                if loaded.mode not in ("L", "1") or loaded.size != (frame.width, frame.height):
                    raise ValueError("mask format or dimensions changed")
                if not set(np.unique(pixels).tolist()) <= {0, 1, 255}:
                    raise ValueError("mask contains non-binary pixels")
                mask = pixels > 0
            actual = mask_observation(mask)
            recorded = observations["frames"][frame_id]
            if any(actual[key] != recorded[key] for key in ("area_pixels", "bbox_xyxy", "valid")):
                raise ValueError("recorded mask observation disagrees with actual mask")
            panels.append(actor_panel(frame.to_image(), mask, actual["bbox_xyxy"], frame_id, index))
            page_frames.append(frame_id)
            seen.append(frame_id)
            if len(panels) == 6:
                path = stage / f"targets-{len(outputs) + 1:03d}.jpg"
                write_page(panels, path)
                outputs.append((path, page_frames))
                panels, page_frames = [], []
    if count != episode["frame_count"] or seen != frames:
        raise ValueError("source decode did not cover all required QA frames")
    if panels:
        path = stage / f"targets-{len(outputs) + 1:03d}.jpg"
        write_page(panels, path)
        outputs.append((path, page_frames))
    entries = []
    for path, ids in outputs:
        destination = qa / path.name
        path.rename(destination)
        entries.append({**file_entry(destination, directory), "frame_ids": ids})
    write_json(output_manifest, {"schema_version": "body_target_qa.v1", "created_at": now(),
                                **identity, "files": entries, "renderer_sha256": sha256(Path(__file__)),
                                "view": "target crop: original left, actual mask overlay right; no approval implied"})
    cleanup_owned_directory(stage, directory, ".target-qa-")
    print(f"EP{index:02d}: {len(frames)} source-bound QA frames in {len(entries)} target sheets", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "data/prepared/track1-20261015/manifest.json")
    parser.add_argument("--inputs", type=Path, default=REPO / "data/body-inputs/track1-20261015")
    parser.add_argument("--episodes", default="all")
    args = parser.parse_args()
    args.episodes = parse_episodes(args.episodes)
    manifest, root = raw_data(args.manifest)
    for episode in manifest["episodes"]:
        if episode["episode_index"] in args.episodes:
            render_episode(args, manifest, root, episode)


if __name__ == "__main__":
    main()
