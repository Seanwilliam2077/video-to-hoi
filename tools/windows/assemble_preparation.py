"""Freeze verified official segmentation and generated geometry for Track 1."""
import argparse
import json
import os
from pathlib import Path
import h5py
from v2hoi.clips import load_clips
from v2hoi.native import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    clip = load_clips("track1", [a.episode], a.dataset_root)[0]
    directory = a.directory.resolve()
    destination = directory / "preparation.json"
    if destination.exists():
        raise FileExistsError("Preserve existing preparation; use a new directory")
    masks = directory / "segmentation/masks.h5"
    mask_report_path = directory / "segmentation/mask-report.json"
    mask_report = json.loads(mask_report_path.read_text())
    generation_path = directory / "object-generation/generation.json"
    generation = json.loads(generation_path.read_text())
    mesh = directory / "object-generation/object.glb"
    video_hash = sha256(clip.video)
    if (mask_report["frames"] != clip.n_frames or mask_report["fill_hole_area"] != 8
            or mask_report["jpeg_frame_cache"] is not False
            or mask_report["official_entry"] != "v2d.sam2.lib.video_to_masks.video_to_masks"
            or sha256(masks) != mask_report["mask_sha256"]):
        raise ValueError("Incomplete or altered official segmentation")
    with h5py.File(masks) as f:
        if f.attrs["source_video_sha256"] != video_hash:
            raise ValueError("Masks do not belong to this Track 1 video")
    if (generation["provenance"] != "track1"
            or generation["entry"] != "v2d.sam3d.lib.image_to_mesh.image_to_mesh"
            or generation["input_record"]["video_sha256"] != video_hash
            or generation["input_record"]["mask_sha256"] != sha256(masks)
            or generation["surface_replacement"] is not False
            or generation["sampler_step_override"] is not False
            or generation["mesh_sha256"] != sha256(mesh)):
        raise ValueError("Incomplete or altered official object generation")
    def artifact(path):
        return {"path": Path(os.path.relpath(path, directory)).as_posix(), "sha256": sha256(path)}
    record = {"provenance": "track1", "episode": a.episode,
        "video_sha256": video_hash, "original_frames": clip.n_frames,
        "masks": artifact(masks), "mesh": artifact(mesh),
        "segmentation": {"entry": mask_report["official_entry"], "fill_hole_area": 8,
            "jpeg_frame_cache": False, "report": artifact(mask_report_path)},
        "mesh_generation": {"entry": generation["entry"], "receipt": artifact(generation_path),
            "surface_replacement": False, "nvidia_entry_defaults": generation["nvidia_entry_defaults"]},
        "bootstrap_frame_id": generation["input_record"]["bootstrap_frame_id"],
        "note": "Bootstrap frame is an original video index; it is not an assumed evaluation alignment frame",
        "limitations": ["Monocular inferred shape and scale; no reference geometry or official score",
            "Owner permits the penetration fallback without a Kaolin equivalence claim"]}
    destination.write_text(json.dumps(record, indent=2))
    print(destination)


if __name__ == "__main__":
    main()
