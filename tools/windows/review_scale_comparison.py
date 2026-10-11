"""Verify controlled settings and create a derived review video, never inference."""
import argparse
import json
from pathlib import Path
import pickle
import subprocess
import cv2
import numpy as np
from v2hoi.native import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--ffmpeg", type=Path, default=Path("C:/Tools/ffmpeg/bin/ffmpeg.exe"))
    a = p.parse_args()
    report = json.loads((a.comparison/"result.json").read_text())
    control, variant = Path(report["control"]), Path(report["variant"])
    identity = [json.loads((r/"native-inputs.json").read_text()) for r in (control,variant)]
    def model_options(command):
        values = command[3:].copy()
        for flag in ("--video_path","--mask_h5_path","--object_mesh_path","--output_dir"):
            i = values.index(flag)
            del values[i:i+2]
        return values
    controlled = {"same_model_options":model_options(identity[0]["command"]) == model_options(identity[1]["command"]),
        "same_official_sources":identity[0]["source_sha256"] == identity[1]["source_sha256"],
        "same_video":identity[0]["video_sha256"] == identity[1]["video_sha256"],
        "same_masks":identity[0]["preparation"]["masks"]["sha256"] == identity[1]["preparation"]["masks"]["sha256"]}
    if not all(controlled.values()):
        raise ValueError("Comparison changed more than input scale")
    seq = f"episode_{identity[0]['episode']:06d}"
    originals = control/"inputs"/f"{seq}.0.color.mp4"
    overlays = [r/"native"/seq/"inference"/f"{seq}_step200000_before_after.mp4" for r in (control,variant)]
    review = a.comparison/"review"
    review.mkdir(exist_ok=True)
    font = "C\\:/Windows/Fonts/arial.ttf"
    def title(label):
        return f"drawtext=fontfile='{font}':text='{label}':fontsize=24:fontcolor=white:x=12:y=12:box=1:boxcolor=black@0.7"
    filters = f"[0:v]scale=768:576,{title('Original')}[source];"
    for i,label in ((1,"v4 - SRT - 16.4 cm"),(2,"Legacy search - 20.8 cm")):
        filters += f"[{i}:v]crop=1536:1152:3072:84,scale=768:576,{title(label)}[panel{i}];"
    filters += "[source][panel1][panel2]hstack=inputs=3[out]"
    video_path = review/"v4-vs-legacy-scale.mp4"
    if video_path.exists():
        raise FileExistsError("Preserve completed review videos")
    # Both original native videos remain untouched. NVENC compresses only this
    # derived side-by-side review; it never changes input images or model output.
    subprocess.run([str(a.ffmpeg),"-hide_banner","-loglevel","error","-i",str(originals),
        "-i",str(overlays[0]),"-i",str(overlays[1]),"-filter_complex",filters,"-map","[out]",
        "-an","-c:v","h264_nvenc","-preset","p4","-cq","18","-b:v","0","-pix_fmt","yuv420p",
        "-movflags","+faststart",str(video_path)],check=True)
    capture = cv2.VideoCapture(str(video_path))
    frames = 0
    while True:
        ok,frame = capture.read()
        if not ok:
            break
        if frames in (0,60,120,160,180,213,224,282,359):
            cv2.imwrite(str(review/f"frame-{frames:06d}.png"),frame)
        frames += 1
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if frames != identity[0]["expected_frames"] or abs(fps-30)>1e-6:
        raise ValueError("Derived preview changed the timeline")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,1,figsize=(10,6),sharex=True)
    for label,run in (("v4 / 16.4 cm",control),("Legacy search / 20.8 cm",variant)):
        data = pickle.loads((run/"native"/seq/"foundationpose"/f"{seq}.pkl").read_bytes())
        time = np.arange(len(data["frames"]))/30
        axes[0].plot(time,data["fp_full_image_iou"],label=label,lw=1.3)
        axes[1].plot(time,data["fp_poses_cam"][:,0,2,3],label=label,lw=1.3)
    axes[0].set(ylabel="FP visible-mask IoU",ylim=(0,1),title="Input-mask / prediction diagnostics; no ground-truth accuracy score")
    axes[1].set(ylabel="FP camera depth (m)",xlabel="Original video time (seconds)")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(review/"tracking-comparison.png",dpi=150)
    plt.close(fig)
    report["controlled_comparison"] = controlled
    report["derived_preview"] = {"path":str(video_path),"sha256":sha256(video_path),"frames":frames,"fps":fps,
        "codec":"h264_nvenc","scope":"Derived review only; original native rendering/encoding and all inference inputs untouched"}
    report["visual_review_frames"] = [0,60,120,160,180,213,224,282,359]
    (a.comparison/"review-report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({"controlled":controlled,"video":str(video_path),"frames":frames,"fps":fps}),flush=True)


if __name__ == "__main__":
    main()
