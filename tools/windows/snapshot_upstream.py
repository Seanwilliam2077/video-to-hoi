"""Save upstream compatibility changes as a reviewable patch in the team repo."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
source = root / "third_party/CARI4D_MHR_Win"
patch = subprocess.check_output(["git", "-C", str(source), "diff", "--binary", "HEAD", "--", "reconstruction/modules"])
untracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "--others", "--exclude-standard", "--", "reconstruction/modules"], text=True).splitlines()
for name in untracked:
    if not name.endswith(".py"):
        raise ValueError(f"Review unexpected untracked upstream asset: {name}")
    content = (source / name).read_text().splitlines(keepends=True)
    diff = ''.join(difflib.unified_diff([], content, fromfile="/dev/null", tofile="b/" + name))
    patch += (f"diff --git a/{name} b/{name}\nnew file mode 100644\n" + diff).encode()
destination = root / "deployment/windows/patches/video_to_data.patch"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_bytes(patch)
manifest = {
    "video_to_data": {"url": "https://github.com/nvidia-isaac/video_to_data.git",
        "revision": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "path": "third_party/CARI4D_MHR_Win", "patch": "patches/video_to_data.patch", "patch_sha256": hashlib.sha256(patch).hexdigest()},
    "sam2": {"url": "https://github.com/facebookresearch/sam2.git", "revision": "2b90b9f5ceec907a1c18123530e92e794ad901a4", "path": "third_party/sam2"},
    "dinov2": {"url": "https://github.com/facebookresearch/dinov2.git", "revision": "7764ea0f912e53c92e82eb78a2a1631e92725fc8", "path": "third_party/dinov2"},
    "dinov3": {"url": "https://github.com/facebookresearch/dinov3.git", "revision": "6876159a11b4df116f30f667f8c9888617df0751", "path": "weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov3_main"},
}
for name, url in (
    ("sam-3d-objects", "https://github.com/facebookresearch/sam-3d-objects.git"),
    ("moge-sam3d", "https://github.com/microsoft/MoGe.git"),
    ("utils3d-sam3d", "https://github.com/EasternJournalist/utils3d.git"),
):
    checkout = root / "third_party" / name
    record = {"url": url, "path": checkout.relative_to(root).as_posix(),
        "revision": subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()}
    changes = subprocess.check_output(["git", "-C", str(checkout), "diff", "--binary", "HEAD"])
    new_files = subprocess.check_output(["git", "-C", str(checkout), "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
    for name_in_checkout in new_files:
        if not name_in_checkout.endswith(".py"):
            raise ValueError(f"Review unexpected upstream asset: {name_in_checkout}")
        content = (checkout / name_in_checkout).read_text().splitlines(keepends=True)
        diff = ''.join(difflib.unified_diff([], content, fromfile="/dev/null", tofile="b/" + name_in_checkout))
        changes += (f"diff --git a/{name_in_checkout} b/{name_in_checkout}\nnew file mode 100644\n" + diff).encode()
    if changes:
        local_patch = root / "deployment/windows/patches" / f"{name}.patch"
        local_patch.write_bytes(changes)
        record.update(patch=f"patches/{name}.patch", patch_sha256=hashlib.sha256(changes).hexdigest())
    manifest[name] = record
sam2_wheel = root / "wheelhouse/windows/sam_2-1.0-cp312-cp312-win_amd64.whl"
manifest["sam2"].update(wheel=sam2_wheel.relative_to(root).as_posix(),
    wheel_sha256=hashlib.sha256(sam2_wheel.read_bytes()).hexdigest(),
    cuda_extension="Original SAM2 connected-components CUDA implementation; Windows compilation only")
sam3d_dino_cache = root / "weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov2_main"
if sam3d_dino_cache.is_dir():
    tree = hashlib.sha256()
    for filename in sorted(sam3d_dino_cache.rglob("*.py")):
        tree.update(filename.relative_to(sam3d_dino_cache).as_posix().encode())
        tree.update(filename.read_bytes())
    manifest["sam3d_dinov2_cache"] = {"url": "https://github.com/facebookresearch/dinov2",
        "path": sam3d_dino_cache.relative_to(root).as_posix(), "python_tree_sha256": tree.hexdigest(),
        "scope": "Actual cached source used by the official SAM3D constructor; archive contains no Git revision",
        "checkpoint": "weights/cari4d/sam3d_body/torch_home/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth",
        "checkpoint_sha256": "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51"}
manifest["moge1_checkpoint"] = {"repo_id": "Ruicheng/moge-vitl", "revision": "ad326bfb61facd6c52b5a825bc1e34d7c97d9672",
    "path": "weights/cari4d/hf_home/hub/models--Ruicheng--moge-vitl/snapshots/ad326bfb61facd6c52b5a825bc1e34d7c97d9672/model.pt",
    "sha256": "da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f"}
(root / "deployment/windows/sources.json").write_text(json.dumps(manifest, indent=2))
print(f"Saved {len(patch)} bytes of compatibility changes")
