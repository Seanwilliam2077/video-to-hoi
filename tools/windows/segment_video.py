"""Prompt SAM 2 on a Track 1 video and write the native CARI4D mask schema."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    frames = args.output / "frames"
    frames.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        target = frames / f"{count:06d}.jpg"
        if not target.exists():
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(target)
        count += 1
    cap.release()
    if count != prompts["expected_frames"]:
        raise ValueError(f"Expected {prompts['expected_frames']} frames, decoded {count}")
    sequence = args.video.name.removesuffix(".0.color.mp4")
    temporary = args.output / "masks.partial.h5"
    areas = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml", str(args.checkpoint),
            hydra_overrides_extra=["++model.fill_hole_area=0"],
        )
        state = model.init_state(str(frames), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for prompt in prompts["prompts"]:
            model.add_new_points_or_box(
                state, frame_idx=prompt["frame"], obj_id=prompt["id"],
                box=np.asarray(prompt["box"], dtype=np.float32),
            )
        with h5py.File(temporary, "w") as output:
            output.attrs["source_video_sha256"] = hashlib.sha256(args.video.read_bytes()).hexdigest()
            output.attrs["prompts"] = json.dumps(prompts)
            output.attrs["model"] = "facebook/sam2.1-hiera-large"
            for index, ids, logits in model.propagate_in_video(state):
                masks = (logits[:, 0] > 0).cpu().numpy()
                frame_areas = {"frame": index}
                for ident, mask in zip(ids, masks):
                    suffix = {1: "person_mask.png", 2: "obj_rend_mask.png"}[ident]
                    output.create_dataset(f"{sequence}/{index:06d}-k0.{suffix}", data=mask, compression="gzip")
                    frame_areas[str(ident)] = int(mask.sum())
                areas.append(frame_areas)
                if index % 90 == 0 or index == count - 1:
                    preview = cv2.imread(str(frames / f"{index:06d}.jpg"))
                    for ident, mask in zip(ids, masks):
                        color = np.array([0, 200, 0] if ident == 1 else [0, 0, 255])
                        preview[mask] = (preview[mask] * 0.55 + color * 0.45).astype(np.uint8)
                    cv2.imwrite(str(args.output / f"overlay-{index:06d}.jpg"), cv2.resize(preview, (768, 576)))
                if index % 30 == 0:
                    print(json.dumps(frame_areas), flush=True)
            if len(areas) != count or any(row.get("1", 0) == 0 or row.get("2", 0) == 0 for row in areas):
                raise ValueError("Missing or empty human/object masks; inspect the previews and prompts")
            output.attrs["complete"] = True
    temporary.replace(args.output / "masks.h5")
    (args.output / "mask-report.json").write_text(json.dumps({"frames": count, "areas": areas}, indent=2))


if __name__ == "__main__":
    main()
