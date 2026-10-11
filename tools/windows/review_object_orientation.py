"""Audit Track 1 stage orientations and render fixed-scale rotation candidates.

No model inference, trajectory replacement, scale fitting, or submission export.
The image losses are fitted diagnostics, never calibrated correctness scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys

import cv2
import numpy as np
import trimesh

from v2hoi.clips import load_clips
from v2hoi.native import native_environment, sha256
from v2hoi.pose_review import orientation_hypotheses, rigid_pose, rotate_about_mesh_center


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path,
                   default=Path("third_party/CARI4D_MHR_Win/data/v2d-challenge/track_1"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, nargs="+", default=[0, 10, 30])
    p.add_argument("--refine-candidates", type=int, nargs="*", default=[],
                   help="Visually reviewed quarter-turn IDs to refine on the first listed frame only")
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    identity = json.loads((a.run / "native-inputs.json").read_text())
    preparation = json.loads(a.preparation.read_text())
    if identity.get("provenance") != "track1" or preparation != identity["preparation"]:
        raise ValueError("Require the exact Track 1 preparation of this run")
    if not a.worker:
        weights = Path(identity["command"][identity["command"].index("--weights_path") + 1])
        environment = native_environment(Path("third_party/CARI4D_MHR_Win").resolve(), weights)
        subprocess.run([sys.executable, "-m", "v2hoi.cuda_budget", "--fraction", ".75",
                        "--report", str(a.output.parent / (a.output.name + "-memory.json")), "--",
                        str(Path(__file__)), *sys.argv[1:], "--worker"], env=environment, check=True)
        return
    import torch
    import nvdiffrast.torch as dr
    import Utils
    from lib_mhr.object_pose_frame import resolve_object_pose_frame
    from prep.mhr_depth_h5 import read_metric_depth
    from prep.mhr_export_utils import camera_calibration, load_edex, read_mask

    clip = load_clips("track1", [identity["episode"]], a.dataset_root)[0]
    if (sha256(clip.video) != identity["video_sha256"]
            or identity["original_frame_ids"] != list(range(clip.n_frames))):
        raise ValueError("Run video or original-frame manifest differs from Track 1 metadata")
    if len(set(a.frames)) != len(a.frames) or any(i < 0 or i >= clip.n_frames for i in a.frames):
        raise ValueError("Review frames must be distinct original-video frame IDs")
    if len(set(a.refine_candidates)) != len(a.refine_candidates) or any(i < 0 or i >= 24 for i in a.refine_candidates):
        raise ValueError("Refinement IDs must be distinct quarter-turn candidates 0 through 23")
    mesh_path = (a.preparation.parent / preparation["mesh"]["path"]).resolve()
    masks_path = (a.preparation.parent / preparation["masks"]["path"]).resolve()
    if sha256(mesh_path) != preparation["mesh"]["sha256"] or sha256(masks_path) != preparation["masks"]["sha256"]:
        raise ValueError("Prepared geometry or masks changed")
    root = a.run / "native" / f"episode_{clip.episode:06d}"
    before_path, after_path = root / "inference/coconet.pth", root / "inference/refined.pth"
    before = torch.load(before_path, map_location="cpu", weights_only=False)
    after = torch.load(after_path, map_location="cpu", weights_only=False)
    export = root / "export" / f"episode_{clip.episode:06d}"
    export_identity = json.loads((export / "wild_export.json").read_text())
    if Path(export_identity["source_object_mesh"]["path"]).resolve() != mesh_path:
        raise ValueError("Native preparation used another source mesh")
    canonical_path = Path(export_identity["object_mesh_file"]).resolve()
    source_to_canonical = rigid_pose(export_identity["source_object_mesh_to_aligned_transform"])
    for bundle in (before, after):
        if bundle.get("gt") or bundle["metadata"].get("ground_truth_used") is not False:
            raise ValueError("Reference data is prohibited in this experiment")
        if [int(row["src_frame"]) for row in bundle["frame_meta"]] != identity["original_frame_ids"]:
            raise ValueError("Bundle changed the original timeline")
        if Path(bundle["metadata"]["object_mesh"]).resolve() != canonical_path:
            raise ValueError("Bundle and preparation use different meshes")
    fp_path = root / "foundationpose" / f"episode_{clip.episode:06d}.pkl"
    with fp_path.open("rb") as handle:
        fp = pickle.load(handle)
    if fp["metadata"].get("first_usable_frame_gt_rotation_oracle") is not False:
        raise ValueError("Reference-assisted initialization is prohibited")
    if [int(name) for name in fp["frames"]] != identity["original_frame_ids"]:
        raise ValueError("Tracker timeline differs")
    before_frame = resolve_object_pose_frame(before["metadata"])
    after_frame = resolve_object_pose_frame(after["metadata"])
    stages = {
        "FoundationPose": np.asarray(fp["fp_poses_cam"])[:, 0],
        "Network input": np.asarray(before["in"]["pose_abs"]) @ before_frame.mesh_to_training,
        "CoCoNet": np.asarray(before["pr_initial"]["pose_abs"]) @ before_frame.mesh_to_training,
        "Postopt": np.asarray(after["pr"]["pose_abs"]) @ after_frame.mesh_to_training,
    }
    mesh = trimesh.load(canonical_path, force="mesh", process=False)
    source_mesh = trimesh.load(mesh_path, force="mesh", process=False)
    transformed = source_mesh.copy()
    transformed.apply_transform(source_to_canonical)
    if (not np.array_equal(mesh.faces, transformed.faces)
            or not np.allclose(mesh.vertices, transformed.vertices, atol=1e-6, rtol=0)
            or not np.array_equal(mesh.visual.vertex_colors, transformed.visual.vertex_colors)):
        raise ValueError("Canonical mesh changed geometry, scale, topology or appearance")
    if mesh.visual.kind != "vertex":
        raise ValueError("This fixed-scale experiment expects the existing colored mesh")
    k, world_to_camera = camera_calibration(load_edex(export), 0)
    if not np.allclose(world_to_camera, np.eye(4), atol=1e-6):
        raise ValueError("This native one-camera run must use camera space as scene space")
    a.output.mkdir(parents=True, exist_ok=False)
    context = dr.RasterizeCudaContext()
    tensors = Utils.make_mesh_tensors(mesh)
    center = mesh.bounds.mean(axis=0)
    depth_path = root / "depth/moge2/aligned.h5"
    records = []

    refinements = []

    def tile(rgb, rendered, mask, title, box):
        overlay = rgb.copy()
        overlay[mask] = rendered[mask]
        x0, y0, x1, y1 = box
        view = cv2.resize(overlay[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_NEAREST)
        panel = np.full((294, 256, 3), 245, dtype=np.uint8)
        panel[38:] = view
        cv2.putText(panel, title, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, .43, (20, 20, 20), 1)
        return panel

    for frame_id in a.frames:
        name = f"{frame_id:06d}"
        # Review against the original video, independently of the native RGB cache.
        capture = cv2.VideoCapture(str(clip.video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, bgr = capture.read()
        capture.release()
        if not ok:
            raise ValueError(f"Missing original frame {frame_id}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask_o = read_mask(export, "object", 0, name).astype(bool)
        mask_h = read_mask(export, "human", 0, name).astype(bool)
        if not mask_o.any():
            raise ValueError("Orientation review needs a visible target; choose another review frame")
        height, width = rgb.shape[:2]
        ys, xs = np.where(mask_o)
        box = (max(0, int(xs.min()) - 45), max(0, int(ys.min()) - 45),
               min(width, int(xs.max()) + 46), min(height, int(ys.max()) + 46))
        x0, y0, x1, y1 = box
        roi_k = np.asarray(k, dtype=np.float32).copy()
        roi_k[0, 2] -= x0
        roi_k[1, 2] -= y0
        observation = rgb[y0:y1, x0:x1]
        human = mask_h[y0:y1, x0:x1]
        observed = mask_o[y0:y1, x0:x1] & ~human
        depth = read_metric_depth(depth_path, "aligned", "front_stereo_camera_left", name)
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        depth = depth[y0:y1, x0:x1]
        base = rigid_pose(stages["Postopt"][frame_id])
        candidates = orientation_hypotheses(base, center)
        poses = np.concatenate([np.stack([rigid_pose(values[frame_id]) for values in stages.values()]), candidates])
        def render(pose_values):
            rendered, depths = [], []
            with torch.inference_mode():
                for start in range(0, len(pose_values), 8):
                    color, z, _ = Utils.nvdiffrast_render(K=roi_k, H=y1-y0, W=x1-x0,
                        ob_in_cams=torch.as_tensor(pose_values[start:start+8], dtype=torch.float32, device="cuda"),
                        glctx=context, mesh_tensors=tensors, use_light=False, extra={})
                    rendered.extend((color.cpu().numpy() * 255).round().astype(np.uint8))
                    depths.extend(z.cpu().numpy())
            return np.asarray(rendered), np.asarray(depths)

        def metrics(color, z):
            visible = (z > .001) & ~human
            intersection = visible & observed
            union = visible | observed
            iou = float(intersection.sum() / max(int(union.sum()), 1))
            error = float(np.abs(color.astype(float)-observation)[intersection].mean()/255) if intersection.any() else 1.
            return iou, error

        labels = list(stages) + [f"Candidate {i:02d}" for i in range(24)]
        if frame_id == a.frames[0] and a.refine_candidates:
            from scipy.spatial.transform import Rotation
            for candidate_id in a.refine_candidates:
                seed = candidates[candidate_id]
                angles = np.zeros(3)
                calls = 0
                # Bounded coordinate sweeps: a coarse semantic branch stays fixed.
                for radius in (15., 5., 2.):
                    for axis in range(3):
                        trials = np.repeat(angles[None], 7, axis=0)
                        trials[:, axis] = np.clip(angles[axis]+np.linspace(-radius, radius, 7), -15., 15.)
                        trial_poses = np.stack([rotate_about_mesh_center(seed, center,
                            Rotation.from_euler("xyz", value, degrees=True).as_matrix()) for value in trials])
                        trial_rgb, trial_z = render(trial_poses)
                        losses = [1-iou+error for iou, error in (metrics(c, z) for c, z in zip(trial_rgb, trial_z))]
                        angles = trials[int(np.argmin(losses))]
                        calls += len(trials)
                refined = rotate_about_mesh_center(seed, center, Rotation.from_euler("xyz", angles, degrees=True).as_matrix())
                poses = np.concatenate([poses, refined[None]])
                labels.append(f"Refined {candidate_id:02d}")
                refinements.append({"frame_id": frame_id, "candidate_id": candidate_id,
                    "local_xyz_angles_deg": angles.tolist(), "render_evaluations": calls,
                    "pose_canonical_mesh_to_camera": refined.tolist(),
                    "objective": "1 - visible silhouette IoU + overlapping-pixel RGB MAE / 255"})
        rendered, depths = render(poses)
        frame_records, panels = [], []
        source = tile(observation, observation, np.zeros_like(observed), f"Original frame {frame_id}", (0, 0, x1-x0, y1-y0))
        panels.append(source)
        for i, pose in enumerate(poses):
            label = labels[i]
            visible = (depths[i] > .001) & ~human
            intersection = visible & observed
            union = visible | observed
            iou = float(intersection.sum() / max(int(union.sum()), 1))
            error = float(np.abs(rendered[i].astype(float)-observation)[intersection].mean()/255) if intersection.any() else None
            valid_depth = intersection & np.isfinite(depth) & (depth > .001)
            depth_error = float(np.median(np.abs(depths[i][valid_depth]-depth[valid_depth]))) if valid_depth.any() else None
            record = {"label": label, "pose_canonical_mesh_to_camera": pose.tolist(), "visible_iou": iou,
                      "overlap_rgb_mae": error, "estimated_depth_median_absolute_error_m": depth_error}
            if len(stages) <= i < len(stages)+24:
                record["candidate_id"] = i - len(stages)
            frame_records.append(record)
            panels.append(tile(observation, rendered[i], visible, f"{label} IoU {iou:.3f}", (0, 0, x1-x0, y1-y0)))
        # Every candidate is shown, including poor ones; no numerical auto-promotion.
        padding = np.full_like(panels[0], 245)
        while len(panels) % 5:
            panels.append(padding)
        sheet = np.vstack([np.hstack(panels[i:i+5]) for i in range(0, len(panels), 5)])
        output = a.output / f"frame-{frame_id:06d}-orientations.png"
        cv2.imwrite(str(output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        np.savez(a.output / f"frame-{frame_id:06d}-candidates.npz", poses_canonical_mesh_to_camera=candidates,
                 frame_id=np.array(frame_id), mesh_sha256=np.array(sha256(canonical_path)))
        records.append({"original_frame_id": frame_id, "review_roi_xyxy": list(box), "poses": frame_records,
                        "sheet": {"path": output.name, "sha256": sha256(output)}})
        print(f"Rendered original frame {frame_id}: 4 stages and 24 fixed-scale candidates", flush=True)
    report = {"schema": "v2hoi.orientation-review.v1", "provenance": "track1", "episode": clip.episode,
              "video_sha256": identity["video_sha256"], "source_mesh_sha256": sha256(mesh_path),
              "mesh_sha256": sha256(canonical_path), "canonical_mesh": str(canonical_path),
              "source_to_canonical": source_to_canonical.tolist(),
              "mask_sha256": sha256(masks_path), "run": str(a.run.resolve()),
              "inputs": {str(path): sha256(path) for path in (before_path, after_path, fp_path, depth_path)},
              "tool_sha256": sha256(Path(__file__)), "mesh_center": center.tolist(),
              "mesh_extents_m": mesh.extents.tolist(), "scale_changed": False,
              "translation_policy": "Preserve transformed mesh center; compensate canonical-origin translation",
              "candidate_semantics": "24 quarter-turn hypotheses, not declarations of object symmetry",
              "selection": "No automatic selection; inspect original and textured candidates",
              "visible_iou_scope": "Display ROI, excluding human-mask pixels; not the tracker full-image score",
              "local_refinements": refinements,
              "limitations": ["Fitted image diagnostics, not held-out evidence or official scores",
                              "Estimated depth is not ground truth", "Quarter turns are coarse, not precise pose fits",
                              "No motion, native human parameters or submission files are changed"], "frames": records}
    (a.output / "orientation-review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
