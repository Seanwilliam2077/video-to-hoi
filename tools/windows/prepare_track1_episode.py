"""Prepare any Track 1 episode with the existing official inference entries.

Prompt boxes are explicit input observations, not episode-specific code. The
generated metric size is provisional until independent calibration is accepted.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import cv2
import h5py
from v2hoi.clips import load_clips
from v2hoi.native import native_environment, sha256
from v2hoi.object_refine import seed_frame_index


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--episode",type=int,required=True)
    p.add_argument("--prompts",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--mesh-backend",choices=("sam3d","bpy"),default="sam3d")
    p.add_argument("--asset-spec",type=Path)
    p.add_argument("--blender",type=Path)
    a=p.parse_args()
    if a.mesh_backend=='bpy' and (not a.asset_spec or not a.blender):
        p.error("The bpy backend requires an observational --asset-spec and --blender executable")
    clip=load_clips("track1",[a.episode],a.dataset_root)[0]
    prompts=json.loads(a.prompts.read_text())
    if prompts["expected_frames"]!=clip.n_frames or {row["id"] for row in prompts["prompts"]}!={1,2}:
        raise ValueError("Require source-video person/object observations and complete frame count")
    output=a.output.resolve()
    output.mkdir(parents=True,exist_ok=a.resume)
    source=Path("third_party/CARI4D_MHR_Win").resolve()
    env=native_environment(source,Path("weights/cari4d").resolve())
    sequence=f"episode_{clip.episode:06d}"
    video=output/"inputs"/f"{sequence}.0.color.mp4"
    video.parent.mkdir(exist_ok=True)
    if not video.exists(): shutil.copy2(clip.video,video)
    if sha256(video)!=sha256(clip.video): raise ValueError("Bootstrap video changed")
    def stage(name,command,inputs,outputs):
        receipt=output/f"{name}.json"
        before={str(Path(path).resolve()):sha256(Path(path)) for path in inputs}
        if a.resume and receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved["inputs"]!=before or any(sha256(Path(path))!=digest for path,digest in saved["outputs"].items()):
                raise ValueError(f"Completed preparation inputs changed: {name}")
            return
        print("PREPARE_START",name,flush=True)
        log=(output/f"{name}.log").open("w",encoding="utf-8")
        subprocess.run([sys.executable,"-m","v2hoi.cuda_budget","--fraction",".75","--report",str(output/f"{name}.memory.json"),"--",*command],
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        log.close()
        receipt.write_text(json.dumps({"provenance":"track1","episode":clip.episode,"inputs":before,
            "outputs":{str(Path(path).resolve()):sha256(Path(path)) for path in outputs}},indent=2))
        print("PREPARE_PASS",name,flush=True)
    segment=Path("tools/windows/segment_video.py")
    masks=output/"segmentation/masks.h5"
    stage("segmentation",[str(segment),"--video",str(video),"--prompts",str(a.prompts.resolve()),
        "--checkpoint",str(Path("weights/sam2/sam2.1_hiera_large.pt").resolve()),"--output",str(output/"segmentation")],
        [video,a.prompts,segment],[masks,output/"segmentation/mask-report.json"])
    counts=[]
    with h5py.File(masks) as f:
        for frame_id in range(clip.n_frames):
            human=f[f"{sequence}/{frame_id:06d}-k0.person_mask.png"][()].astype(bool)
            obj=f[f"{sequence}/{frame_id:06d}-k0.obj_rend_mask.png"][()].astype(bool)
            counts.append(int((obj & ~human).sum()) if human.sum()>=100 else 0)
    frame=seed_frame_index(counts)
    selected={"policy":"Largest visible object support with a supported human",
        "original_frame_id":frame,"mask_sha256":sha256(masks),"video_sha256":sha256(video),
        "evaluation_alignment_frame_changed":False}
    selection=output/"generation-frame.json"
    if selection.exists() and json.loads(selection.read_text())!=selected:
        raise ValueError("Generation selection changed; preserve the previous preparation")
    selection.write_text(json.dumps(selected,indent=2),encoding="utf-8")
    first=output/"inputs"/f"rgb-{frame:06d}.png"
    if not first.exists():
        cap=cv2.VideoCapture(str(video))
        for _ in range(frame+1): ok,bgr=cap.read()
        cap.release()
        if not ok: raise ValueError("Missing selected original bootstrap frame")
        cv2.imwrite(str(first),bgr)
    depth_script=source/"reconstruction/modules/v2d_cari4d/lib/cari4d/prep/mhr_wild_depth.py"
    raw=output/"bootstrap-depth/raw.npy"; intrinsics=output/"bootstrap-depth/intrinsics.pkl"
    stage("depth",[str(depth_script),"--video",str(video),"--output-depth",str(raw),"--output-intrinsics",str(intrinsics),
        "--output-report",str(output/"bootstrap-depth/report.json"),"--expected-frames",str(clip.n_frames),
        "--batch-size","4","--device","cuda","--local-files-only"],[video,depth_script],[raw,intrinsics])
    bootstrap=Path("tools/windows/bootstrap_human.py")
    body=output/"bootstrap-human.pth"
    stage("human-bootstrap",[str(bootstrap),"--rgb",str(first),"--masks",str(masks),"--intrinsics",str(intrinsics),
        "--sequence",sequence,"--video-sha256",sha256(video),"--frame",str(frame),"--output",str(body)],
        [first,masks,intrinsics,bootstrap,selection],[body,body.with_suffix(".json")])
    conditioning=Path("tools/windows/prepare_object_inputs.py")
    stage("object-conditioning",[str(conditioning),"--video",str(video),"--masks",str(masks),"--raw-depth",str(raw),
        "--intrinsics",str(intrinsics),"--human-mesh",str(body),"--frame",str(frame),"--output",str(output/"object-inputs")],
        [video,masks,raw,intrinsics,body,conditioning,selection],[output/"object-inputs/inputs.json"])
    if a.mesh_backend=='sam3d':
        generator=Path("tools/windows/generate_object.py")
        stage("object-generation",[str(generator),"--inputs",str(output/"object-inputs"),"--output",str(output/"object-generation"),
            "--cpu-offload"],[output/"object-inputs/inputs.json",generator],[output/"object-generation/object.glb",output/"object-generation/generation.json"])
    else:
        generator=Path("tools/windows/prepare_bpy_asset.py")
        asset_spec=json.loads(a.asset_spec.read_text())
        observation_frames=sorted({frame,*[row['original_frame_id'] for row in asset_spec.get('face_observations',[])]})
        stage("object-generation",[str(generator),"--dataset-root",str(a.dataset_root.resolve()),"--episode",str(clip.episode),
            "--spec",str(a.asset_spec.resolve()),"--frames",*[str(i) for i in observation_frames],"--blender",str(a.blender.resolve()),
            "--output",str(output/"object-generation")],[video,masks,a.asset_spec,generator,generator.with_name('build_bpy_asset.py')],
            [output/"object-generation/object.glb",output/"object-generation/asset.json"])
    assembler=Path("tools/windows/assemble_preparation.py")
    stage("assemble",[str(assembler),"--dataset-root",str(a.dataset_root.resolve()),"--episode",str(clip.episode),
        "--directory",str(output),"--mesh-backend",a.mesh_backend],
        [masks,output/"object-generation"/('generation.json' if a.mesh_backend=='sam3d' else 'asset.json'),assembler],
        [output/"preparation.json"])
    print("Preparation complete; scale remains provisional until independent acceptance",flush=True)


if __name__=="__main__": main()
