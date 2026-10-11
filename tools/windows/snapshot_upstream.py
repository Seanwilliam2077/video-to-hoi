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
(root / "deployment/windows/sources.json").write_text(json.dumps(manifest, indent=2))
print(f"Saved {len(patch)} bytes of compatibility changes")
