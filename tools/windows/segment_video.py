"""Call NVIDIA's official SAM2 video entry, then losslessly adapt its mask schema."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from v2d.sam2.lib.video_to_masks import video_to_masks
from v2d.common.video import FrameSource
from v2hoi.native import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "masks.h5").exists():
        raise FileExistsError("Use a new output directory to preserve previous masks")
    from sam2 import _C  # Required: never silently skip the default CUDA postprocessing.
    from sam2.utils.misc import fill_holes_in_mask_scores
    probe = torch.ones(1, 1, 32, 32, device="cuda")
    probe[:, :, 8, 8] = -1
    if fill_holes_in_mask_scores(probe, 8)[0, 0, 8, 8] <= 0:
        raise RuntimeError("Official SAM2 small-hole filling is unavailable")
    official_prompts = {"prompts": [{"object_id": row["id"], "frame_index": row["frame"],
        "box": dict(zip(("x0", "y0", "x1", "y1"), row["box"]))} for row in prompts["prompts"]]}
    prompt_path = args.output / "official-prompts.json"
    prompt_path.write_text(json.dumps(official_prompts, indent=2))
    official_output = args.output / "official-masks"
    video_to_masks(str(args.video.resolve()), str(prompt_path.resolve()),
                   str(official_output.resolve()), str(args.checkpoint.resolve().parent), mask_extension=".h5")
    sequence = args.video.name.removesuffix(".0.color.mp4")
    temporary = args.output / "masks.partial.h5"
    areas = []
    sources = {ident: FrameSource.from_path(official_output / f"{ident}.h5") for ident in (1, 2)}
    count = prompts["expected_frames"]
    if any(source.n_frames != count for source in sources.values()):
        raise ValueError("Official SAM2 output is missing original video frames")
    cap = cv2.VideoCapture(str(args.video))
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["source_video_sha256"] = sha256(args.video)
            output.attrs["prompts"] = json.dumps(prompts)
            output.attrs["model"] = "facebook/sam2.1-hiera-large"
            output.attrs["official_entry"] = "v2d.sam2.lib.video_to_masks.video_to_masks"
            output.attrs["fill_hole_area"] = 8
            output.attrs["input"] = "Original MP4 directly decoded by official Decord loader; no JPEG cache"
            for index in range(count):
                frame_areas = {"frame": index}
                masks = {ident: source[index] > 0 for ident, source in sources.items()}
                for ident, mask in masks.items():
                    suffix = {1: "person_mask.png", 2: "obj_rend_mask.png"}[ident]
                    output.create_dataset(f"{sequence}/{index:06d}-k0.{suffix}", data=mask, compression="gzip")
                    frame_areas[str(ident)] = int(mask.sum())
                areas.append(frame_areas)
                ok, preview = cap.read()
                if not ok:
                    raise ValueError(f"Missing preview source frame {index}")
                if index % 90 == 0 or index == count - 1:
                    for ident, mask in masks.items():
                        color = np.array([0, 200, 0] if ident == 1 else [0, 0, 255])
                        preview[mask] = (preview[mask] * 0.55 + color * 0.45).astype(np.uint8)
                    cv2.imwrite(str(args.output / f"overlay-{index:06d}.jpg"), cv2.resize(preview, (768, 576)))
                if index % 30 == 0:
                    print(json.dumps(frame_areas), flush=True)
            output.attrs["complete"] = True
    finally:
        cap.release()
        for source in sources.values():
            source.close()
    temporary.replace(args.output / "masks.h5")
    report = {"frames": count, "areas": areas, "official_entry": "v2d.sam2.lib.video_to_masks.video_to_masks",
              "input": "original MP4 / Decord", "fill_hole_area": 8, "jpeg_frame_cache": False,
              "model_sha256": sha256(args.checkpoint), "mask_sha256": sha256(args.output / "masks.h5")}
    (args.output / "mask-report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
