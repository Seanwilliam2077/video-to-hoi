"""Reload the exported native parameters and render every original video frame."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from lib_mhr import MHRLayer
from lib_mhr.human_texture import batched_vertex_normals
from prep.mhr_export_utils import camera_calibration, load_edex
from tools.mhr_mesh_renderer import NvdiffMeshRenderer
from v2hoi.native import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--kit", type=Path, default=Path("third_party/official-track1-kit"))
    a = p.parse_args()
    identity = json.loads((a.run / "native-inputs.json").read_text())
    seq = f"episode_{identity['episode']:06d}"
    params_path = a.export / f"{seq}.npz"
    mesh = a.export / f"{seq}_object.glb"
    with np.load(params_path, allow_pickle=False) as data:
        params = {k: data[k].copy() for k in data.files}
    if params["frame_ids"].tolist() != identity["original_frame_ids"] or float(params["object_scale"]) != 1:
        raise ValueError("Unexpected export timeline or non-unit scale")
    assets = Path(os.environ["MHR_ASSETS_ROOT"])
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=assets, device="cuda")
    faces = layer.mesh_faces(device="cpu").numpy().astype(np.int32)
    del layer
    torch.cuda.empty_cache()
    spec = importlib.util.spec_from_file_location("official_converter", a.kit / "tools/track1/mesh_to_mhr_params.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    mhr = official.MHR(str(assets / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"), "cuda", 1, "float32")
    pose = torch.as_tensor(params["pose"], device="cuda", dtype=torch.float64)
    shared = torch.as_tensor(np.concatenate((params["scales"], params["shape"]))[None], device="cuda", dtype=torch.float64)
    renderer = NvdiffMeshRenderer(device="cuda")
    k, extrinsic = camera_calibration(load_edex(a.run / "native" / seq / "export" / seq), 0)
    if not np.allclose(extrinsic, np.eye(4)):
        raise ValueError("Unexpected non-camera-space export")
    video = cv2.VideoCapture(str(a.run / "inputs" / f"{seq}.0.color.mp4"))
    fps = video.get(cv2.CAP_PROP_FPS)
    output = a.export / "export-reloaded.mp4"
    with imageio.get_writer(str(output), format="FFMPEG", fps=fps, codec="libx264", macro_block_size=None, ffmpeg_params=["-crf", "20"]) as writer:
        for i in range(len(pose)):
            ok, bgr = video.read()
            if not ok:
                raise ValueError(f"Missing source frame {i}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Preview at half resolution; geometry remains in its original metric frame.
            rgb = cv2.resize(rgb, None, fx=.5, fy=.5)
            intrinsics = np.asarray(k, dtype=np.float32).copy()
            intrinsics[:2] *= .5
            verts = (mhr.verts(pose[i:i+1], shared).reshape(1, -1, 3) / 1000).float().cpu().numpy()
            normals = batched_vertex_normals(verts, faces)
            transform = np.eye(4, dtype=np.float32)[None]
            transform[0, :3, :3] = params["object_rotation"][i]
            transform[0, :3, 3] = params["object_translation"][i]
            rendered, mask = renderer.render_front_batch_constant_human_textured_object(
                verts, faces, normals, str(mesh.resolve()), transform, intrinsics[None], rgb.shape[:2], (.78, .82, .85))
            panel = np.full_like(rgb, (32, 36, 40))
            panel[mask[0]] = (rendered[0][mask[0]].clip(0, 1) * 255).round().astype(np.uint8)
            frame = np.concatenate((rgb, panel), axis=1)
            writer.append_data(frame)
            if i in (0, 90, 180, 270, len(pose)-1):
                imageio.imwrite(a.export / f"export-preview-{i:06d}.jpg", frame)
            if i % 60 == 0:
                print(f"Reloaded export render: {i+1}/{len(pose)}", flush=True)
    if video.read()[0]:
        raise ValueError("Extra source frames missing from export")
    video.release()
    check = cv2.VideoCapture(str(output))
    decoded = 0
    while check.read()[0]:
        decoded += 1
    check.release()
    if decoded != len(pose):
        raise ValueError("Rendered export frame count changed")
    report = {"frames_decoded": decoded, "fps": fps, "parameters_sha256": sha256(params_path),
              "mesh_sha256": sha256(mesh), "video_sha256": sha256(output), "decoder": "Official MHR TorchScript wrapper",
              "scope": "Reloaded serialized candidate; no reference data or official score"}
    (a.export / "render-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
