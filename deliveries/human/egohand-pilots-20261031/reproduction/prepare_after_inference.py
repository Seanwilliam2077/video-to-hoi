"""Verify one completed full pilot and bind new adapter/fit jobs before launch."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json

import numpy as np

TASK = Path(__file__).resolve().parent
RESULTS = Path("J:/video-to-hoi/runs/egohand-pilots-20261031").resolve()
PYTHON = Path("J:/video-to-hoi/work/human-runtime/venv/Scripts/python.exe")
RUNTIME = TASK / "runtime01"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, choices=(11, 16), required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.prediction_root.resolve()
    episode_root = RESULTS / f"episode_{args.episode:06d}"
    if not root.is_relative_to(episode_root) or root == episode_root:
        raise ValueError("Full prediction must be inside this authorized pilot output")
    if "smoke" in root.name.lower():
        raise ValueError("A smoke is not a full comparison result")
    inputs = read(TASK / "inputs/manifest.json")
    if inputs["all30_execution_authorized"] is not False:
        raise ValueError("Wrong task scope")
    item = next(x for x in inputs["episodes"] if x["episode"] == args.episode)
    n = item["frames"]
    receipt_path = root / "prediction-receipt.json"
    receipt = read(receipt_path)
    if (receipt.get("status") != "completed" or receipt.get("episode") != args.episode or
            receipt.get("source_video_sha256") != item["video_sha256"] or
            receipt.get("source_frames") != n or receipt.get("frames") != n or
            receipt.get("decoded_source_frames") != n or receipt.get("complete_source_coverage") is not True or
            receipt.get("selected_original_frame_ids") != list(range(n))):
        raise ValueError("Prediction receipt is not complete for the exact original video")
    for key, expected in [("same_original_camera", True), ("same_crop_bounds", True),
                          ("perspective_cropping", False), ("pixel_aspect_rectification", False),
                          ("box_expansion", False), ("all30_execution_started", False)]:
        if receipt.get(key) is not expected:
            raise ValueError(f"Controlled inference condition differs: {key}")
    expected_revision = read(TASK / "environment-source-receipt.json")["repos"][0]["revision"]
    if receipt["upstream_commit"] != expected_revision:
        raise ValueError("EgoHand source revision differs from the prepared runtime")
    if receipt["runner_sha256"] != digest(TASK / "run_controlled_egohand.py"):
        raise ValueError("Actual inference runner identity changed")
    checkpoint = next(x for x in read(TASK / "weights-receipt.json")["assets"] if x["file"] == "egohand_s060000.ckpt")
    if receipt["checkpoint_sha256"] != checkpoint["sha256"] or checkpoint.get("verified") is not True:
        raise ValueError("EgoHand checkpoint identity differs from verified public asset")
    for name, key in (("video", "video_sha256"), ("camera", "camera_sha256"),
                      ("original_observations", "original_observations_sha256")):
        if digest(item[name]) != item[key]:
            raise ValueError(f"Frozen task input changed: {name}")
    for binding in item["saved_boxes"]:
        if digest(binding["path"]) != binding["sha256"]:
            raise ValueError("Original source crop bounds changed")
    load = receipt["model_load"]
    if load["missing_learned_keys"] or load["unexpected_keys"]:
        raise ValueError("Learned checkpoint did not match model")
    verification = {"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "episode": args.episode, "source_video_sha256": item["video_sha256"],
                    "frames": n, "prediction_receipt_sha256": digest(receipt_path),
                    "upstream_commit": expected_revision, "outputs": {},
                    "source_camera_crops_and_frame_coverage_verified": True,
                    "generated_predictions_not_ground_truth": True, "all30_execution_authorized": False}
    arrays = {}
    for variant, field, declared_variant in (("raw", "raw_outputs", "raw"),
                                              ("smoothed", "smoothed_outputs", "upstream_adaptive_smoothed")):
        outputs = receipt.get(field, [])
        if len(outputs) != 2 or {x["side"] for x in outputs} != {"left", "right"}:
            raise ValueError("Need both anatomical hands in each declared output variant")
        arrays[variant] = {}
        for output in outputs:
            path = Path(output["path"]).resolve()
            if path != (root / variant / f"{output['side']}_results.npz").resolve():
                raise ValueError("Output path does not match its declared side/variant")
            if (output.get("frames") != n or output.get("variant") != declared_variant or
                    output.get("complete_source_coverage") is not True or
                    output.get("finite_camera_geometry") is not True or digest(path) != output["sha256"]):
                raise ValueError("Output SHA, side, variant or original coverage mismatch")
            with np.load(path, allow_pickle=False) as archive:
                data = {k: archive[k] for k in archive.files}
            names = data["frame_names"]
            if names.shape != (n,) or names.dtype.kind not in "US" or int(data["num_frames"]) != n:
                raise ValueError("Output has wrong original-frame dimensions")
            ids = [int(x) for x in names.astype(str)]
            if ids != list(range(n)):
                raise ValueError("Output changed original frame order or omitted a frame")
            if data["joints"].shape != (n, 25, 3) or data["cam_t"].shape != (n, 3):
                raise ValueError("Unexpected upstream SOMA camera-geometry layout")
            if not np.isfinite(data["joints"]).all() or not np.isfinite(data["cam_t"]).all():
                raise ValueError("Nonfinite full specialist output")
            if float(np.linalg.norm(data["joints"][:, 0], axis=-1).max()) >= 1e-4:
                raise ValueError("Joints are not wrist-local; adding cam_t would be incorrect")
            arrays[variant][output["side"]] = path
        verification["outputs"][variant] = outputs
    verify_path = TASK / f"ep{args.episode}-prediction-verification01.json"
    write(verify_path, verification)
    for variant in ("raw", "smoothed"):
        target = episode_root / f"observations-{variant}01"
        if target.exists():
            raise ValueError("Adapter output exists; preserve prior attempts")
        command = [str(PYTHON), "-B", str(RUNTIME / "tools/adapt_egohand_observations.py"),
            "--left", str(arrays[variant]["left"]), "--right", str(arrays[variant]["right"]),
            "--frozen-observations", item["original_observations"], "--manifest", item["manifest"],
            "--intrinsics", str(Path(item["baseline"]["path"]).parent / "review_arrays/camera_intrinsics.npy"),
            "--prediction-receipt", str(receipt_path), "--upstream-commit", expected_revision,
            "--variant", variant, "--output", str(target)]
        bindings = [{"path": str(p), "sha256": digest(p)} for p in
                    [receipt_path, verify_path, arrays[variant]["left"], arrays[variant]["right"],
                     RUNTIME / "tools/adapt_egohand_observations.py", RUNTIME / "src/v2hoi/egohand_adapter.py"]]
        write(TASK / f"ep{args.episode}-adapt-{variant}01.json", {
            "episode": args.episode, "command": command, "cwd": str(RUNTIME),
            "log": str(TASK / f"ep{args.episode}-adapt-{variant}01.log"), "input_bindings": bindings,
            "automatic_promotion": False, "all30_execution_authorized": False})
    print(json.dumps({"episode": args.episode, "verified_original_frames": n,
                      "adapter_jobs_prepared": ["raw", "smoothed"], "all30_authorized": False}))


if __name__ == "__main__":
    main()
