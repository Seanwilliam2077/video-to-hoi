"""Generate reviewed target prompts and complete Track 1 SAM2 human masks.

Only preparation outputs are produced. Visual review and Body handoff acceptance
remain separate gates. Heavy dependencies are imported only by execution modes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from functools import partial
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid

from prepare_track1_data import reject_links, safe_relative, sha256


REPO = Path(__file__).resolve().parents[1]
SAM_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM_CHECKPOINT_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DETECTOR_URL = "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth"
# Approximate torso locations reviewed in original frame 0 of every episode.
# They select a detector candidate; they do not substitute for a mask or box.
TARGET_POINTS = [
    (.36, .58), (.56, .55), (.49, .54), (.49, .55), (.59, .54),
    (.37, .54), (.40, .55), (.50, .56), (.45, .56), (.64, .56),
    (.39, .57), (.49, .49), (.50, .58), (.58, .55), (.55, .56),
    (.50, .56), (.35, .55), (.57, .54), (.34, .55), (.36, .55),
    (.38, .56), (.62, .57), (.49, .55), (.52, .55), (.35, .54),
    (.54, .64), (.57, .62), (.33, .59), (.56, .59), (.60, .57),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def workspace_path(path):
    path = Path(path).absolute()
    reject_links(path)
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO.resolve()):
        raise ValueError(f"output/cache must stay inside repository: {path}")
    return resolved


def parse_episodes(value):
    result = list(range(30)) if value == "all" else [int(x) for x in value.split(",")]
    if not result or len(set(result)) != len(result) or any(x not in range(30) for x in result):
        raise ValueError("episodes must be unique Track 1 indices 0..29 or all")
    return sorted(result)


def raw_data(manifest_path):
    reject_links(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "track1_raw_preparation.v1"
            or manifest.get("track") != "track_1" or not manifest.get("raw_data_ready")
            or manifest.get("revision") != "5f68335f3acc802033d1e80728c1633197521de8"
            or sorted(e["episode_index"] for e in manifest["episodes"]) != list(range(30))):
        raise ValueError("expected the verified pinned all-30 Track 1 raw manifest")
    root = (manifest_path.parent / manifest["dataset_root_relative"]).resolve()
    reject_links(root)
    return manifest, root


def verified_video(manifest, root, episode):
    relative = safe_relative(episode["video_path"])
    path = root / relative
    reject_links(path)
    entry = manifest["source_files"][relative]
    if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError(f"source video bytes changed: {relative}")
    return path, entry["sha256"]


def first_frame(path):
    import av
    with av.open(str(path)) as container:
        return next(container.decode(video=0)).to_image()


def select_candidate(candidates, point):
    x, y = point
    matches = [c for c in candidates if c["bbox_xyxy"][0] <= x < c["bbox_xyxy"][2]
               and c["bbox_xyxy"][1] <= y < c["bbox_xyxy"][3]]
    if not matches:
        raise ValueError("no person detection contains reviewed target point; inspect and correct prompt")
    return max(matches, key=lambda c: c["score"])


def environment():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "av", "Pillow", "numpy", "hydra-core"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def save_sheet(tiles, destination):
    from PIL import Image, ImageDraw
    columns = 4
    sheet = Image.new("RGB", (384 * columns, 316 * math.ceil(len(tiles) / columns)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (image, label) in enumerate(tiles):
        x, y = i % columns * 384, i // columns * 316
        sheet.paste(image.resize((384, 288)), (x, y))
        draw.text((x + 5, y + 291), label, fill="black")
    sheet.save(destination, quality=90, subsampling=0)


def prompts(args):
    import torch
    from PIL import ImageDraw
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )
    manifest, root = raw_data(args.manifest)
    destination = workspace_path(args.prompts)
    if destination.exists():
        raise ValueError(f"refusing to overwrite prompt record: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch_home = workspace_path(args.torch_home)
    torch_home.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_home))
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    detector = fasterrcnn_mobilenet_v3_large_fpn(weights=weights).eval().to(args.device)
    checkpoint = torch_home / "checkpoints" / DETECTOR_URL.rsplit("/", 1)[1]
    if not checkpoint.is_file():
        raise ValueError("detector checkpoint path was not recorded")
    detector_identity = {"method": "torchvision.fasterrcnn_mobilenet_v3_large_fpn",
                         "weights": "COCO_V1", "url": DETECTOR_URL,
                         "checkpoint_sha256": sha256(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                         "score_threshold": args.detector_threshold, "environment": environment()}
    entries, tiles = [], []
    for episode in manifest["episodes"]:
        index = episode["episode_index"]
        if index not in args.episodes:
            continue
        video, video_hash = verified_video(manifest, root, episode)
        image = first_frame(video)
        if image.size != (episode["width"], episode["height"]):
            raise ValueError("first-frame dimensions changed")
        tensor = weights.transforms()(image).to(args.device)
        with torch.inference_mode():
            prediction = detector([tensor])[0]
        candidates = [{"bbox_xyxy": [float(v) for v in box], "score": float(score)}
                      for box, label, score in zip(prediction["boxes"].cpu().tolist(),
                                                   prediction["labels"].cpu().tolist(),
                                                   prediction["scores"].cpu().tolist())
                      if label == 1 and score >= args.detector_threshold]
        point = [TARGET_POINTS[index][0] * image.width, TARGET_POINTS[index][1] * image.height]
        try:
            selected = select_candidate(candidates, point)
            error = None
        except ValueError as exc:
            selected, error = None, str(exc)
        entry = {"episode_index": index, "frame_id": 0, "target_id": "human_0",
                 "video_sha256": video_hash, "width": image.width, "height": image.height,
                 "target_point_xy": point, "bbox_xyxy": selected["bbox_xyxy"] if selected else None,
                 "selected_score": selected["score"] if selected else None,
                 "candidates": candidates, "selection_error": error}
        entries.append(entry)
        draw = ImageDraw.Draw(image)
        for candidate in candidates:
            draw.rectangle(candidate["bbox_xyxy"], outline="yellow", width=3)
        if selected:
            draw.rectangle(selected["bbox_xyxy"], outline="lime", width=7)
        x, y = point
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="red")
        tiles.append((image, f"EP{index:02d} frame 0 | {'REVIEW' if selected else 'NO MATCH'}"))
        print(f"Prompt EP{index:02d}: {len(candidates)} people, selected={selected is not None}", flush=True)
    sheets = []
    for start in range(0, len(tiles), 12):
        path = destination.parent / f"prompts-review-{start // 12 + 1:02d}.jpg"
        if path.exists():
            raise ValueError(f"refusing to overwrite prompt sheet: {path}")
        save_sheet(tiles[start:start + 12], path)
        sheets.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(destination, {"schema_version": "body_mask_prompts.v1", "created_at": now(),
                            "raw_manifest_sha256": sha256(args.manifest), "detector": detector_identity,
                            "point_selection": "Agent-reviewed torso locations on original frame 0; confirm green boxes",
                            "visual_review": {"status": "pending", "reviewer": None, "reviewed_at": None,
                                              "reviewed_episode_indices": []},
                            "episodes": entries, "contact_sheets": sheets})
    print(f"Prompts written: {destination}; visual review required before run", flush=True)


def mask_observation(mask, previous=None):
    import numpy as np
    if mask.ndim != 2:
        raise ValueError("expected a two-dimensional mask")
    ys, xs = np.where(mask)
    area = int(len(xs))
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if area else None
    reasons = []
    if area == 0:
        reasons.append("empty_mask")
    elif area == mask.size:
        reasons.append("full_image_mask")
    iou = area_change = None
    if previous is not None:
        previous_area = int(previous.sum())
        union = int(np.logical_or(mask, previous).sum())
        iou = float(np.logical_and(mask, previous).sum() / union) if union else 0.0
        area_change = abs(area - previous_area) / max(previous_area, 1)
        if iou < .3:
            reasons.append("adjacent_mask_iou_below_0.3")
        if area_change > .4:
            reasons.append("adjacent_area_change_above_0.4")
    return {"valid": 0 < area < mask.size, "area_pixels": area, "bbox_xyxy": bbox,
            "confidence": None, "adjacent_mask_iou": iou, "adjacent_area_change_ratio": area_change,
            "review_flags": reasons}


def load_prompts(path, manifest_path, episodes):
    reject_links(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    review = document.get("visual_review", {})
    if (document.get("schema_version") != "body_mask_prompts.v1"
            or document.get("raw_manifest_sha256") != sha256(manifest_path)
            or review.get("status") != "approved" or not review.get("reviewer")
            or not review.get("reviewed_at")
            or not set(episodes) <= set(review.get("reviewed_episode_indices", []))):
        raise ValueError("source-bound prompts require explicit visual review of each selected episode")
    entries = document["episodes"]
    if len({e["episode_index"] for e in entries}) != len(entries):
        raise ValueError("duplicate prompt episodes")
    selected = {e["episode_index"]: e for e in entries}
    for index in episodes:
        entry = selected.get(index, {})
        box, point = entry.get("bbox_xyxy"), entry.get("target_point_xy")
        if (entry.get("frame_id") != 0 or entry.get("target_id") != "human_0"
                or not isinstance(box, list) or len(box) != 4
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                or not 0 <= box[0] < box[2] <= entry.get("width", 0)
                or not 0 <= box[1] < box[3] <= entry.get("height", 0)
                or not isinstance(point, list) or len(point) != 2
                or not box[0] <= point[0] < box[2] or not box[1] <= point[1] < box[3]):
            raise ValueError(f"invalid reviewed prompt for episode {index}")
    return document, selected


def decode_cache(video, episode, timeline, cache):
    import av
    cache.mkdir(parents=True, exist_ok=False)
    count = 0
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if (index >= episode["frame_count"] or frame.pts != timeline["pts"][index]
                    or (frame.width, frame.height) != (episode["width"], episode["height"])):
                raise ValueError(f"source decode disagrees with frozen timeline at frame {index}")
            frame.to_image().save(cache / f"{index:06d}.jpg", quality=95, subsampling=0)
            count += 1
    if count != episode["frame_count"]:
        raise ValueError("source decode has incomplete frame coverage")


def file_entry(path, root):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_existing(directory, episode, video_hash, prompt_hash):
    document = json.loads((directory / "masks-manifest.json").read_text(encoding="utf-8"))
    if (document.get("schema_version") != "body_masks.v1"
            or document.get("episode_index") != episode["episode_index"]
            or document.get("video_sha256") != video_hash):
        raise ValueError("existing mask manifest identity mismatch")
    expected = [f"human_masks/{i:06d}.png" for i in range(episode["frame_count"])]
    if [e.get("path") for e in document.get("files", [])] != expected:
        raise ValueError("existing mask manifest coverage mismatch")
    if sorted(p.name for p in (directory / "human_masks").iterdir()) != [f"{i:06d}.png" for i in range(episode["frame_count"])]:
        raise ValueError("existing mask directory contains missing or unexpected entries")
    artifacts = document.get("artifacts", [])
    if not {"observations.json", "masks-provenance.json"} <= {e.get("path") for e in artifacts}:
        raise ValueError("existing output lacks metadata hash inventory")
    for entry in document["files"] + artifacts:
        relative = safe_relative(entry["path"])
        path = directory / relative
        reject_links(path)
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"existing output byte check failed: {path}")
    provenance = json.loads((directory / "masks-provenance.json").read_text(encoding="utf-8"))
    if (provenance.get("checkpoint_sha256") != SAM_CHECKPOINT_SHA256
            or provenance.get("code_revision") != SAM_REVISION
            or provenance.get("prompt_entry_sha256") != prompt_hash):
        raise ValueError("existing output model or prompt identity differs")


def qa_frame_ids(frame_count, observations):
    samples = {round(i * (frame_count - 1) / 8) for i in range(9)}
    # Banker's rounding of (T-1)/2 can miss the gate's exact T//2 frame.
    samples.update({0, frame_count // 2, frame_count - 1})
    samples.update(o["frame_id"] for o in observations if o["review_flags"])
    return sorted(samples)


def create_qa(video, episode, stage, observations):
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    samples = set(qa_frame_ids(episode["frame_count"], observations))
    directory = stage / "mask-qa"
    directory.mkdir()
    tiles, files, page = [], [], 0
    with av.open(str(video)) as container:
        for frame_id, frame in enumerate(container.decode(video=0)):
            if frame_id not in samples:
                continue
            image = frame.to_image()
            with Image.open(stage / f"human_masks/{frame_id:06d}.png") as loaded:
                mask = np.asarray(loaded) > 0
            rgb = np.asarray(image).copy()
            rgb[mask] = (rgb[mask].astype(np.float32) * .55 + np.array([0, 230, 110]) * .45).astype(np.uint8)
            image = Image.fromarray(rgb)
            item = observations[frame_id]
            if item["bbox_xyxy"]:
                ImageDraw.Draw(image).rectangle(item["bbox_xyxy"], outline="lime", width=4)
            label = f"EP{episode['episode_index']:02d} f{frame_id:04d} area={item['area_pixels']}"
            if item["review_flags"]:
                label += " FLAG"
            tiles.append((image, label))
            if len(tiles) == 12:
                page += 1
                path = directory / f"review-{page:03d}.jpg"
                save_sheet(tiles, path)
                files.append(path)
                tiles = []
    if tiles:
        path = directory / f"review-{page + 1:03d}.jpg"
        save_sheet(tiles, path)
        files.append(path)
    return sorted(samples), files


def cleanup_owned_directory(path, parent, prefix):
    resolved = workspace_path(path)
    allowed_parent = workspace_path(parent)
    if resolved.parent != allowed_parent or not resolved.name.startswith(prefix):
        raise ValueError("refusing unsafe cache cleanup")
    reject_links(resolved)
    for item in resolved.rglob("*"):
        reject_links(item)
    shutil.rmtree(resolved)


def run(args):
    import torch
    import numpy as np
    from PIL import Image
    manifest, root = raw_data(args.manifest)
    prompt_document, prompt_entries = load_prompts(args.prompts, args.manifest, args.episodes)
    output, cache_root = workspace_path(args.output), workspace_path(args.cache_root)
    output.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    source = workspace_path(args.sam_source)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if revision != SAM_REVISION or dirty:
        raise ValueError("SAM2 source must be clean and match the pinned revision")
    checkpoint = workspace_path(args.checkpoint)
    if sha256(checkpoint) != SAM_CHECKPOINT_SHA256:
        raise ValueError("SAM2 checkpoint hash mismatch")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("run mode requires the selected CUDA profile")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("selected CUDA profile requires bfloat16 support")
    from sam2.build_sam import build_sam2_video_predictor
    import sam2
    import sam2.sam2_video_predictor as video_predictor_module
    import sam2.utils.misc as sam_misc
    if not Path(sam2.__file__).resolve().is_relative_to(source):
        raise ValueError("imported SAM2 does not come from the pinned local source")
    # Keep explicit episode/every-100-frame progress without per-frame terminal
    # redraws. This only changes progress display in this producer process.
    video_predictor_module.tqdm = partial(video_predictor_module.tqdm, disable=True)
    sam_misc.tqdm = partial(sam_misc.tqdm, disable=True)
    overrides = ["++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                 "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                 "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                 "++model.binarize_mask_from_pts_for_mem_enc=true", "++model.fill_hole_area=0"]
    predictor = build_sam2_video_predictor(SAM_CONFIG, str(checkpoint), device="cuda",
                                          vos_optimized=False, apply_postprocessing=False,
                                          hydra_overrides_extra=overrides)
    runtime = environment()
    runtime.update({"gpu": torch.cuda.get_device_name(), "cuda_runtime": torch.version.cuda,
                    "autocast_dtype": "bfloat16", "device": "cuda"})
    for episode in manifest["episodes"]:
        index = episode["episode_index"]
        if index not in args.episodes:
            continue
        video, video_hash = verified_video(manifest, root, episode)
        prompt = prompt_entries[index]
        if prompt["video_sha256"] != video_hash or (prompt["width"], prompt["height"]) != (episode["width"], episode["height"]):
            raise ValueError("prompt source identity differs from video")
        prompt_hash = fingerprint(prompt)
        directory = workspace_path(output / f"episode_{index:06d}")
        directory.mkdir(exist_ok=True)
        if (directory / "masks-manifest.json").exists():
            if not args.resume:
                raise ValueError(f"refusing to overwrite completed masks: {directory}; use --resume to verify and skip")
            verify_existing(directory, episode, video_hash, prompt_hash)
            print(f"EP{index:02d}: verified existing masks; skipped", flush=True)
            continue
        if any((directory / name).exists() for name in ("human_masks", "observations.json", "masks-provenance.json", "mask-qa")):
            raise ValueError(f"partial published outputs require investigation: {directory}")
        timeline_path = args.manifest.parent / safe_relative(episode["timeline_path"])
        if sha256(timeline_path) != episode["timeline_sha256"]:
            raise ValueError("timeline hash changed")
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        token = uuid.uuid4().hex
        cache = cache_root / f"ep{index:02d}-{token}"
        stage = directory / f".mask-generation-{token}"
        stage.mkdir()
        (stage / "human_masks").mkdir()
        start = time.monotonic()
        print(f"EP{index:02d}: preparing all {episode['frame_count']} source frames", flush=True)
        decode_cache(video, episode, timeline, cache)
        observations, files, previous = [], [], None
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(str(cache), offload_video_to_cpu=True,
                                         offload_state_to_cpu=True, async_loading_frames=False)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1,
                                             box=np.asarray(prompt["bbox_xyxy"], dtype=np.float32))
            for frame_id, ids, logits in predictor.propagate_in_video(state):
                if frame_id != len(observations) or ids != [1]:
                    raise ValueError("predictor returned unexpected frame or target identity")
                mask = (logits[0, 0] > 0).cpu().numpy()
                if mask.shape != (episode["height"], episode["width"]):
                    raise ValueError("predictor mask dimensions differ from source")
                observation = {"frame_id": int(frame_id), **mask_observation(mask, previous)}
                observations.append(observation)
                path = stage / f"human_masks/{frame_id:06d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(path, compress_level=6)
                files.append(file_entry(path, stage))
                previous = mask
                if (frame_id + 1) % 100 == 0 or frame_id + 1 == episode["frame_count"]:
                    elapsed = time.monotonic() - start
                    print(f"EP{index:02d}: {frame_id + 1}/{episode['frame_count']} masks, {elapsed:.1f}s", flush=True)
            del state, logits
        if len(observations) != episode["frame_count"]:
            raise ValueError("predictor produced incomplete frame coverage")
        peak = torch.cuda.max_memory_allocated()
        gc.collect()
        torch.cuda.empty_cache()
        qa_frames, qa_files = create_qa(video, episode, stage, observations)
        write_json(stage / "observations.json", {"schema_version": "body_observations.v1",
                   "episode_index": index, "target_id": "human_0", "video_sha256": video_hash,
                   "frame_count": episode["frame_count"], "frames": observations})
        provenance = {"schema_version": "body_masks_provenance.v1", "created_at": now(),
                      "episode_index": index, "method": "SAM2.1 Hiera Large video predictor",
                      "code_repository": "https://github.com/facebookresearch/sam2", "code_revision": revision,
                      "checkpoint_repository": "facebook/sam2.1-hiera-large",
                      "checkpoint_revision": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
                      "checkpoint_sha256": SAM_CHECKPOINT_SHA256,
                      "checkpoint_bytes": checkpoint.stat().st_size, "configuration": SAM_CONFIG,
                      "configuration_sha256": sha256(source / "sam2" / SAM_CONFIG),
                      "hydra_overrides": overrides, "environment": runtime,
                      "source_video_sha256": video_hash, "timeline_sha256": episode["timeline_sha256"],
                      "raw_manifest_sha256": sha256(args.manifest), "prompts_sha256": sha256(args.prompts),
                      "prompt_entry_sha256": prompt_hash, "annotation": prompt,
                      "annotation_review": prompt_document["visual_review"],
                      "decoder": "PyAV sequential presentation-order decode, exact PTS checked",
                      "image_transform": {"jpeg_quality": 95, "jpeg_subsampling": 0,
                                          "sam_internal_resize": [1024, 1024],
                                          "mask_output_dimensions": [episode["width"], episode["height"]],
                                          "frame_stride": 1, "mask_logit_threshold": 0},
                      "offload_video_to_cpu": True, "offload_state_to_cpu": True,
                      "async_loading_frames": False, "custom_cuda_postprocessing": False,
                      "tqdm_progress_bars_disabled": True,
                      "torch_compile": False, "peak_cuda_allocated_bytes": peak,
                      "elapsed_seconds": time.monotonic() - start,
                      "quality_flagged_frames": [{"frame_id": o["frame_id"], "flags": o["review_flags"]}
                                                 for o in observations if o["review_flags"]],
                      "visual_review_frame_ids": qa_frames, "visual_review_status": "pending",
                      "invalid_frame_ids": [o["frame_id"] for o in observations if not o["valid"]],
                      "producer_script_sha256": sha256(Path(__file__))}
        write_json(stage / "masks-provenance.json", provenance)
        artifacts = [file_entry(path, stage) for path in
                     [stage / "observations.json", stage / "masks-provenance.json", *qa_files]]
        write_json(stage / "masks-manifest.json", {"schema_version": "body_masks.v1", "episode_index": index,
                   "video_sha256": video_hash, "frame_count": episode["frame_count"], "files": files,
                   "artifacts": artifacts})
        for name in ("human_masks", "observations.json", "masks-provenance.json", "mask-qa", "masks-manifest.json"):
            (stage / name).rename(directory / name)
        verify_existing(directory, episode, video_hash, prompt_hash)
        cleanup_owned_directory(cache, cache_root, f"ep{index:02d}-")
        cleanup_owned_directory(stage, directory, ".mask-generation-")
        print(f"EP{index:02d}: complete mask bytes verified; {len(provenance['quality_flagged_frames'])} flagged frames await review", flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("prompts", "run"))
    result.add_argument("--manifest", type=Path, default=REPO / "data/prepared/track1-20261015/manifest.json")
    result.add_argument("--prompts", type=Path, default=REPO / "data/body-inputs/track1-20261015/prompts.json")
    result.add_argument("--output", type=Path, default=REPO / "data/body-inputs/track1-20261015")
    result.add_argument("--torch-home", type=Path, default=REPO / "weights/body-prep/torch_home")
    result.add_argument("--sam-source", type=Path, default=REPO / "third_party/sam2")
    result.add_argument("--checkpoint", type=Path, default=REPO / "weights/body-prep/sam2/sam2.1_hiera_large.pt")
    result.add_argument("--cache-root", type=Path, default=REPO / "work/body-mask-cache")
    result.add_argument("--episodes", default="all")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--detector-threshold", type=float, default=.7)
    result.add_argument("--resume", action="store_true", help="skip complete outputs only after hashes and identities pass")
    return result


def main():
    args = parser().parse_args()
    args.episodes = parse_episodes(args.episodes)
    try:
        if args.mode == "prompts":
            prompts(args)
        else:
            run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Body masks failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
