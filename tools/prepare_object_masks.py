"""Propagate one reviewed object prompt across every original Track 1 frame.

The prompt file must name a point on the target object. A box is optional.
Run refuses prompts that have not been visually approved. Outputs are object
masks only; they do not replace the published human-mask delivery.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid

from prepare_body_camera import episode_inputs, load_raw_inputs
from prepare_body_masks import (
    SAM_CHECKPOINT_SHA256,
    SAM_CONFIG,
    SAM_REVISION,
    create_qa,
    decode_cache,
    file_entry,
    mask_observation,
    qa_frame_ids,
    save_sheet,
    sha256,
)
from prepare_track1_data import reject_links, safe_relative


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def object_names() -> dict[int, str]:
    board = json.loads((Path(__file__).resolve().parents[1] / "tracking" / "board.json").read_text(encoding="utf-8"))
    return {int(episode["ep"]): episode["object"] for episode in board["episodes"]}


def load_prompts(path: Path, manifest_path: Path, episodes: list[int]) -> tuple[dict, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    review = document.get("visual_review", {})
    if (document.get("schema_version") != "object_mask_prompts.v1"
            or document.get("raw_manifest_sha256") != sha256(manifest_path)
            or review.get("status") != "approved" or not review.get("reviewer") or not review.get("reviewed_at")
            or not set(episodes) <= set(review.get("reviewed_episode_indices", []))):
        raise ValueError("object prompts require an approved visual review for every selected episode")
    names = object_names()
    entries = {}
    for entry in document["episodes"]:
        index = entry["episode_index"]
        point = entry["point_xy"]
        if (index in entries or entry.get("object_id") != names[index]
                or len(point) != 2 or any(not isinstance(value, (int, float)) for value in point)):
            raise ValueError(f"invalid object prompt for episode {index}")
        entries[index] = entry
    if any(index not in entries for index in episodes):
        raise ValueError("selected episode has no object prompt")
    return document, entries


def export_frames(args) -> None:
    import av
    from PIL import Image, ImageDraw

    manifest = load_raw_inputs(args.manifest.resolve(), args.dataset_root.resolve())
    names = object_names()
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    tiles = []
    for row in manifest["episodes"]:
        if row["episode_index"] not in args.episodes:
            continue
        video, _timeline = episode_inputs(manifest, row, args.manifest.resolve(), args.dataset_root.resolve())
        with av.open(str(video)) as container:
            image = next(container.decode(video=0)).to_image()
        path = destination / f"episode_{row['episode_index']:06d}-frame0.jpg"
        image.save(path, quality=90)
        preview = image.resize((384, 288))
        tiles.append((preview, f"EP{row['episode_index']:02d} {names[row['episode_index']]}"))
        print(f"EP{row['episode_index']:02d}: frame 0 saved {image.size}", flush=True)
    save_sheet(tiles[:12], destination / "contact-01.jpg")
    if len(tiles) > 12:
        save_sheet(tiles[12:24], destination / "contact-02.jpg")
    if len(tiles) > 24:
        save_sheet(tiles[24:], destination / "contact-03.jpg")


def verify_source(source: Path, checkpoint: Path) -> str:
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if revision != SAM_REVISION or dirty:
        raise ValueError("SAM2 source must be clean and match the pinned revision")
    if sha256(checkpoint) != SAM_CHECKPOINT_SHA256:
        raise ValueError("SAM2 checkpoint hash mismatch")
    return revision


def run(args) -> None:
    import numpy as np
    import torch
    from PIL import Image

    manifest = load_raw_inputs(args.manifest.resolve(), args.dataset_root.resolve())
    prompt_document, prompts = load_prompts(args.prompts, args.manifest.resolve(), args.episodes)
    revision = verify_source(args.sam_source.resolve(), args.checkpoint.resolve())
    if args.device != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("object masks require CUDA with bfloat16")
    from sam2.build_sam import build_sam2_video_predictor
    import sam2
    import sam2.sam2_video_predictor as video_predictor_module
    import sam2.utils.misc as sam_misc
    if not Path(sam2.__file__).resolve().is_relative_to(args.sam_source.resolve()):
        raise ValueError("imported SAM2 does not come from the pinned local source")
    video_predictor_module.tqdm = partial(video_predictor_module.tqdm, disable=True)
    sam_misc.tqdm = partial(sam_misc.tqdm, disable=True)
    overrides = ["++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                 "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                 "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                 "++model.binarize_mask_from_pts_for_mem_enc=true", "++model.fill_hole_area=0"]
    predictor = build_sam2_video_predictor(SAM_CONFIG, str(args.checkpoint.resolve()), device="cuda",
                                          vos_optimized=False, apply_postprocessing=False,
                                          hydra_overrides_extra=overrides)
    runtime = {"python": platform.python_version(), "platform": platform.platform(),
               "gpu": torch.cuda.get_device_name(), "cuda_runtime": torch.version.cuda,
               "packages": sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())}
    args.output.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    for row in manifest["episodes"]:
        index = row["episode_index"]
        if index not in args.episodes:
            continue
        directory = args.output / f"episode_{index:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "masks-manifest.json").is_file() and args.resume:
            print(f"EP{index:02d}: existing object masks present; skipped", flush=True)
            continue
        if (directory / "masks-manifest.json").is_file():
            raise ValueError(f"refusing to overwrite completed object masks: {directory}")
        video, timeline = episode_inputs(manifest, row, args.manifest.resolve(), args.dataset_root.resolve())
        prompt = prompts[index]
        token = uuid.uuid4().hex
        cache = cache_root / f"ep{index:02d}-{token}"
        stage = directory / f".mask-generation-{token}"
        masks_dir = stage / "object_masks"
        masks_dir.mkdir(parents=True)
        started = time.monotonic()
        print(f"EP{index:02d}: preparing all {row['frame_count']} source frames", flush=True)
        decode_cache(video, row, timeline, cache)
        observations, files, previous = [], [], None
        torch.cuda.reset_peak_memory_stats()
        point = np.asarray([prompt["point_xy"]], dtype=np.float32)
        box = None if not prompt.get("bbox_xyxy") else np.asarray(prompt["bbox_xyxy"], dtype=np.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(str(cache), offload_video_to_cpu=True,
                                         offload_state_to_cpu=True, async_loading_frames=False)
            predictor.add_new_points_or_box(state, frame_idx=int(prompt["frame_id"]), obj_id=1,
                                            points=point, labels=np.asarray([1], dtype=np.int32), box=box)
            for frame_id, ids, logits in predictor.propagate_in_video(state):
                if frame_id != len(observations) or ids != [1]:
                    raise ValueError("predictor returned unexpected frame or target identity")
                mask = (logits[0, 0] > 0).cpu().numpy()
                if mask.shape != (row["height"], row["width"]):
                    raise ValueError("predictor mask dimensions differ from source")
                observation = {"frame_id": int(frame_id), **mask_observation(mask, previous)}
                observations.append(observation)
                path = masks_dir / f"{frame_id:06d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(path, compress_level=6)
                files.append({"path": f"object_masks/{frame_id:06d}.png", "bytes": path.stat().st_size, "sha256": sha256(path)})
                previous = mask
                if (frame_id + 1) % 100 == 0 or frame_id + 1 == row["frame_count"]:
                    print(f"EP{index:02d}: {frame_id + 1}/{row['frame_count']} masks, {time.monotonic() - started:.1f}s", flush=True)
            del state
        if len(observations) != row["frame_count"]:
            raise ValueError("predictor produced incomplete frame coverage")
        # Reuse the human QA sheet layout by exposing the same folder name temporarily.
        alias = stage / "human_masks"
        if not alias.exists():
            masks_dir.rename(alias)
        _samples, _qa_files = create_qa(video, row, stage, observations)
        alias.rename(masks_dir)
        video_hash = timeline["video_sha256"]
        provenance = {
            "schema_version": "object_masks_provenance.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "episode_index": index,
            "object_id": prompt["object_id"],
            "method": "SAM2.1 Hiera Large video predictor, one positive object point",
            "code_revision": revision,
            "checkpoint_sha256": SAM_CHECKPOINT_SHA256,
            "prompt": prompt,
            "prompt_file_sha256": sha256(args.prompts),
            "annotation_review": prompt_document["visual_review"],
            "environment": runtime,
            "source_video_sha256": video_hash,
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "invalid_frame_ids": [item["frame_id"] for item in observations if not item["valid"]],
            "visual_review_frame_ids": _samples,
            "visual_review_status": "pending",
        }
        write_json(stage / "observations.json", {"schema_version": "object_observations.v1", "episode_index": index,
                   "object_id": prompt["object_id"], "frame_count": row["frame_count"], "frames": observations})
        write_json(stage / "masks-provenance.json", provenance)
        write_json(stage / "masks-manifest.json", {"schema_version": "object_masks.v1", "episode_index": index,
                   "object_id": prompt["object_id"], "video_sha256": video_hash,
                   "frame_count": row["frame_count"], "files": files})
        for name in ("object_masks", "observations.json", "masks-provenance.json", "mask-qa", "masks-manifest.json"):
            target = directory / name
            if target.exists():
                raise ValueError(f"refusing to replace {target}")
            (stage / name).rename(target)
        shutil.rmtree(cache)
        shutil.rmtree(stage)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"EP{index:02d}: object masks complete", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("frames", "run"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--sam-source", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path(r"D:\V2H\object-depth-20261022\object-cache"))
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(30)))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        if args.mode == "frames":
            export_frames(args)
        else:
            if not args.prompts or not args.sam_source or not args.checkpoint:
                raise ValueError("run mode requires --prompts, --sam-source and --checkpoint")
            run(args)
    except Exception as error:
        print(f"Object masks failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
