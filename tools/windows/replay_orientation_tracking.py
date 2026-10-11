"""Replay two fixed-scale Track 1 object tracks with different initial rotations.

Uses existing FoundationPose weights and original MP4 frames for both tracks.
Human reconstruction is retained as context, not refitted. No submission export.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import trimesh

from v2hoi.clips import load_clips
from v2hoi.native import native_environment, sha256
from v2hoi.pose_review import rigid_pose


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--candidate", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path,
                   default=Path("third_party/CARI4D_MHR_Win/data/v2d-challenge/track_1"))
    p.add_argument("--ffmpeg", type=Path, default=Path("C:/Tools/ffmpeg/bin/ffmpeg.exe"))
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    review = json.loads(a.review.read_text())
    run = Path(review["run"])
    identity = json.loads((run / "native-inputs.json").read_text())
    if review.get("provenance") != "track1" or identity.get("provenance") != "track1":
        raise ValueError("Only verified Track 1 review inputs may be replayed")
    if not a.worker:
        weights = Path(identity["command"][identity["command"].index("--weights_path")+1])
        environment = native_environment(Path("third_party/CARI4D_MHR_Win").resolve(), weights)
        subprocess.run([sys.executable, "-m", "v2hoi.cuda_budget", "--fraction", ".75",
                        "--report", str(a.output.parent / (a.output.name+"-memory.json")), "--",
                        str(Path(__file__)), *sys.argv[1:], "--worker"], env=environment, check=True)
        return
    import av
    import torch
    import nvdiffrast.torch as dr
    import Utils
    from prep.run_foundationpose_mhr_export import _pose_estimator
    from prep.mhr_export_utils import camera_calibration, load_edex, read_mask
    from prep.mhr_depth_h5 import read_metric_depth

    logging.getLogger().setLevel(logging.WARNING)
    clip = load_clips("track1", [identity["episode"]], a.dataset_root)[0]
    if sha256(clip.video) != review["video_sha256"] or identity["original_frame_ids"] != list(range(clip.n_frames)):
        raise ValueError("Replay input differs from the complete Track 1 video")
    for path, expected in review["inputs"].items():
        if sha256(Path(path)) != expected:
            raise ValueError("Reviewed native inputs changed")
    mesh_path = Path(review["canonical_mesh"])
    if sha256(mesh_path) != review["mesh_sha256"]:
        raise ValueError("Reviewed canonical mesh changed")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    frame_zero = next(row for row in review["frames"] if row["original_frame_id"] == 0)
    seed_wrong = rigid_pose(next(row for row in frame_zero["poses"] if row["label"] == "Postopt")["pose_canonical_mesh_to_camera"])
    seed_right = rigid_pose(next(row for row in review["local_refinements"] if row["candidate_id"] == a.candidate and row["frame_id"] == 0)["pose_canonical_mesh_to_camera"])
    center = mesh.bounds.mean(axis=0)
    if not np.allclose(seed_wrong[:3, :3]@center+seed_wrong[:3, 3], seed_right[:3, :3]@center+seed_right[:3, 3], atol=1e-6):
        raise ValueError("Reviewed orientation candidate changed the object center")
    root = run / "native" / f"episode_{clip.episode:06d}"
    export = root / "export" / f"episode_{clip.episode:06d}"
    k, world_to_camera = camera_calibration(load_edex(export), 0)
    if not np.allclose(world_to_camera, np.eye(4), atol=1e-6):
        raise ValueError("Replay expects camera and scene coordinates to coincide")
    a.output.mkdir(parents=True, exist_ok=False)
    Utils.set_seed(0)
    estimator = _pose_estimator(mesh, a.output / "fp-debug")
    centered_transform = estimator.get_tf_to_centered_mesh().cpu().numpy()
    states = [torch.as_tensor(seed @ np.linalg.inv(centered_transform), dtype=torch.float32, device="cuda")
              for seed in (seed_wrong, seed_right)]
    render_context = dr.RasterizeCudaContext()
    mesh_tensors = Utils.make_mesh_tensors(mesh)
    depth_path = root / "depth/moge2/aligned.h5"
    # The released wild first-then-track path uses its configured iteration count.
    fp_metadata_path = root / "foundationpose" / f"episode_{clip.episode:06d}.pkl"
    import pickle
    fp_metadata = pickle.loads(fp_metadata_path.read_bytes())["metadata"]
    if fp_metadata.get("first_usable_frame_gt_rotation_oracle") is not False:
        raise ValueError("Reference-assisted initialization is prohibited")
    stage_path = root / ".stages/05_object_pose.memory.json"
    stage_command = json.loads(stage_path.read_text())["command"]
    iteration = int(stage_command[stage_command.index("--iteration")+1])
    input_video = av.open(str(clip.video))
    stream = input_video.streams.video[0]
    fps = float(stream.average_rate)
    preview = a.output / "orientation-tracking.mp4"
    preview_size = (768, 288)  # Three identically cropped 256 px views.
    encoder_log = (a.output / "preview-encoding.log").open("w", encoding="utf-8")
    encoder = subprocess.Popen([str(a.ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{preview_size[0]}x{preview_size[1]}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18", "-b:v", "0", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(preview)], stdin=subprocess.PIPE, stderr=encoder_log,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    poses, ious, rgb_errors = [], [], []
    sampled_frames = {0, 10, 30, 60, 120, 160, 180, 213, 224, 282, 359}
    try:
        for frame_id, decoded in enumerate(input_video.decode(stream)):
            if frame_id >= clip.n_frames:
                raise ValueError("Video exceeds its declared original-frame manifest")
            rgb = decoded.to_ndarray(format="rgb24")
            height, width = rgb.shape[:2]
            name = f"{frame_id:06d}"
            depth = read_metric_depth(depth_path, "aligned", "front_stereo_camera_left", name)
            if depth.shape != (height, width):
                depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
            depth = depth.astype(np.float32)
            depth[(depth < .001) | (depth > 8.)] = 0
            mask_h = read_mask(export, "human", 0, name).astype(bool)
            mask_o = read_mask(export, "object", 0, name).astype(bool)
            if frame_id == 0:
                current = np.stack([seed_wrong, seed_right])
            else:
                current = []
                with torch.inference_mode():
                    for i in range(2):
                        estimator.pose_last = states[i]
                        pose = estimator.track_one(rgb=rgb, depth=depth, K=k, iteration=iteration, extra={})
                        states[i] = estimator.pose_last.detach().clone()
                        current.append(rigid_pose(pose))
                current = np.stack(current)
            poses.append(current)
            observed = mask_o & ~mask_h
            if observed.any():
                ys, xs = np.where(observed)
                cx, cy = int((xs.min()+xs.max())/2), int((ys.min()+ys.max())/2)
                half = max(80, int(max(xs.max()-xs.min(), ys.max()-ys.min())*.8)+30)
                box = (max(0, cx-half), max(0, cy-half), min(width, cx+half), min(height, cy+half))
            else:
                box = (0, 0, width, height)
            x0, y0, x1, y1 = box
            with torch.inference_mode():
                # Score the full image so drift outside the display crop remains a penalty.
                colors, z, _ = Utils.nvdiffrast_render(K=k, H=height, W=width,
                    ob_in_cams=torch.as_tensor(current, dtype=torch.float32, device="cuda"),
                    glctx=render_context, mesh_tensors=mesh_tensors, use_light=False, extra={})
            colors = (colors.cpu().numpy()*255).round().astype(np.uint8)
            z = z.cpu().numpy()
            source = rgb[y0:y1, x0:x1]
            panels = []
            frame_ious, frame_rgb = [], []
            views = [source]
            for i in range(2):
                visible = (z[i] > .001) & ~mask_h
                overlap = visible & observed
                iou = float(overlap.sum()/max(int((visible | observed).sum()), 1))
                error = float(np.abs(colors[i].astype(float)-rgb)[overlap].mean()/255) if overlap.any() else np.nan
                frame_ious.append(iou)
                frame_rgb.append(error)
                view = rgb.copy()
                view[visible] = colors[i][visible]
                views.append(view[y0:y1, x0:x1])
            ious.append(frame_ious)
            rgb_errors.append(frame_rgb)
            for title, view in zip((f"Original {frame_id}", "Wrong orientation seed", "Reviewed upright seed"), views):
                panel = np.full((288, 256, 3), 245, dtype=np.uint8)
                cv2.putText(panel, title, (5, 23), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 20, 20), 1)
                panel[32:] = cv2.resize(view, (256, 256), interpolation=cv2.INTER_NEAREST)
                panels.append(panel)
            image = np.hstack(panels)
            encoder.stdin.write(image.tobytes())
            if frame_id in sampled_frames:
                cv2.imwrite(str(a.output / f"frame-{frame_id:06d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if frame_id % 30 == 0:
                print(f"TRACK_PROGRESS {frame_id+1}/{clip.n_frames}", flush=True)
    finally:
        input_video.close()
        encoder.stdin.close()
        encoder.wait()
        encoder_log.close()
    if encoder.returncode or len(poses) != clip.n_frames:
        raise ValueError("Incomplete orientation replay or preview encoding")
    capture = cv2.VideoCapture(str(preview))
    decoded_frames = 0
    while capture.read()[0]:
        decoded_frames += 1
    capture.release()
    if decoded_frames != clip.n_frames:
        raise ValueError("Derived preview lost frames")
    poses, ious, rgb_errors = np.asarray(poses), np.asarray(ious), np.asarray(rgb_errors)
    np.savez(a.output / "tracks.npz", poses_canonical_mesh_to_camera=poses,
             original_frame_ids=np.arange(clip.n_frames), visible_iou=ious, overlap_rgb_mae=rgb_errors,
             mesh_sha256=np.array(review["mesh_sha256"]))
    summary = {"schema": "v2hoi.orientation-tracking.v1", "provenance": "track1", "episode": clip.episode,
        "frames": clip.n_frames, "fps": fps, "candidate": a.candidate, "scale_changed": False,
        "review_sha256": sha256(a.review), "tool_sha256": sha256(Path(__file__)),
        "mesh_sha256": review["mesh_sha256"], "video_sha256": review["video_sha256"],
        "tracking_iterations": iteration, "rgb_input": "Original MP4 decoded with PyAV for both tracks",
        "tracking_stage_receipt_sha256": sha256(stage_path),
        "random_seed": 0, "visible_iou_scope": "Full original image, excluding human-mask pixels",
        "initialization_difference": "Object rotation only; transformed center, geometry, scale and other inputs fixed",
        "comparison": "Two independent replays; historical native trajectory is not the replay control",
        "segments": {f"{start}-{min(start+119,clip.n_frames-1)}": {"visible_iou_mean": ious[start:start+120].mean(axis=0).tolist(),
            "overlap_rgb_mae_mean": np.nanmean(rgb_errors[start:start+120], axis=0).tolist()} for start in range(0, clip.n_frames, 120)},
        "preview": {"path": preview.name, "sha256": sha256(preview), "decoded_frames": decoded_frames},
        "limitations": ["Object tracking ablation only; no coupled human/object rerun or submission export",
            "Image observations were used for seed fitting; no ground-truth or official accuracy claim",
            "Motion through occlusion remains an estimate; no claim that every frame is accurate",
            "No WAPR weights or agentic tracking framework were installed"]}
    (a.output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
