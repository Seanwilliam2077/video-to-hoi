"""Score predictions against Track 2 Tier 1 ground truth.

    python -m v2hoi.score --pred <root> [--gt <tier1 root>] [--episodes 7 9] [--stride 2]

Both roots use the Tier 1 layout (see v2hoi.dataset). Metric definitions are
in docs/design.md, section 5. They approximate the official Track 1 metrics,
which are not published yet.

Sanity checks:

    # ground truth against itself: every error ~0
    python -m v2hoi.score --pred data/v2d/track_2/tier_1_multiview_caption
    # Tier 2 keeps the original meshes, so point it at the Tier 1 mesh folder
    python -m v2hoi.score --pred data/v2d/track_2/tier_2_synthetic_noise \
        --pred-mesh-dir data/v2d/track_2/tier_1_multiview_caption/mesh
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from v2hoi import metrics as M
from v2hoi.dataset import TIER1_ROOT, Episode, list_episodes, load_episode
from v2hoi.geometry import Similarity, SurfaceSDF, load_mesh, sample_surface

MM = 1000.0


@dataclass
class Config:
    align: str = "se3"  # se3 | sim3 | none; fitted on body joints over all frames
    stride: int = 1  # frame stride for the per-frame mesh metrics
    object_samples: int = 10_000
    shape_frames: int = 10


@dataclass
class ObjectModel:
    samples: np.ndarray  # (N, 3) surface samples, canonical frame
    centroid: np.ndarray  # (3,)
    sdf: SurfaceSDF


class ObjectModels:
    """Loads each mesh once and keeps its samples and SDF."""

    def __init__(self, count: int):
        self.count = count
        self._cache: dict[str, ObjectModel] = {}

    def __call__(self, path: Path) -> ObjectModel:
        key = str(Path(path).resolve())
        if key not in self._cache:
            if not Path(path).is_file():
                raise FileNotFoundError(f"object mesh not found: {path}")
            mesh = load_mesh(path)
            samples, _ = sample_surface(mesh, self.count, seed=0)
            self._cache[key] = ObjectModel(samples, samples.mean(0), SurfaceSDF(mesh))
        return self._cache[key]


def _posed(model: ObjectModel, T: np.ndarray) -> np.ndarray:
    return model.samples @ T[:3, :3].T + T[:3, 3]


def _mean(values) -> float:
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def _norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=-1)


def _pmap(fn, frames) -> list:
    """Map over frames on threads; fn should use single-threaded KD-tree queries."""
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        return list(pool.map(fn, frames))


def score_episode(gt: Episode, pred: Episode, body, objects: ObjectModels, cfg: Config) -> dict:
    T = len(gt)
    if len(pred) != T:
        raise ValueError(f"episode {gt.index}: prediction has {len(pred)} frames, ground truth has {T}")
    frames = np.arange(0, T, cfg.stride)

    gj, gv = body(gt)
    pj, pv = body(pred)

    # Put the prediction into the ground-truth world using body joints only;
    # the Sim(3) scale is always reported as a diagnostic.
    bj, fj = body.body_ids, body.finger_ids
    src, dst = pj[:, bj].reshape(-1, 3), gj[:, bj].reshape(-1, 3)
    sim3 = Similarity.fit(src, dst, with_scale=True)
    align = {
        "none": Similarity.identity(),
        "se3": Similarity.fit(src, dst, with_scale=False),
        "sim3": sim3,
    }[cfg.align]
    pj_w, pv_w = align.points(pj), align.points(pv)

    human = {
        "chamfer_mm": _mean(_pmap(lambda t: M.chamfer(pv_w[t], gv[t], workers=1), frames)) * MM,
        "mpjpe_mm": float(_norm(pj_w - gj).mean()) * MM,
        "mpjpe_body_mm": float(_norm(pj_w[:, bj] - gj[:, bj]).mean()) * MM,
        "accel_err_mm_f2": M.accel_error(pj_w, gj) * MM,
        "accel_err_body_mm_f2": M.accel_error(pj_w[:, bj], gj[:, bj]) * MM,
        "accel_err_fingers_mm_f2": M.accel_error(pj_w[:, fj], gj[:, fj]) * MM,
    }

    gm, pm = objects(gt.mesh_path), objects(pred.mesh_path)
    gT, pT = gt.obj_T, pred.obj_T
    both = gt.obj_visible & pred.obj_visible
    obj_frames = frames[both[frames]]
    shape_frames = obj_frames[
        np.unique(np.linspace(0, len(obj_frames) - 1, min(cfg.shape_frames, len(obj_frames))).astype(int))
    ] if len(obj_frames) else obj_frames
    g_cen = gT[:, :3, :3] @ gm.centroid + gT[:, :3, 3]
    p_cen = align.points(pT[:, :3, :3] @ pm.centroid + pT[:, :3, 3])
    n_gt_vis, n_gt_hidden = int(gt.obj_visible.sum()), int((~gt.obj_visible).sum())

    obj = {
        "chamfer_mm": _mean(_pmap(
            lambda t: M.chamfer(align.points(_posed(pm, pT[t])), _posed(gm, gT[t]), workers=1), obj_frames
        )) * MM,
        "shape_chamfer_mm": float(np.median(_pmap(
            lambda t: M.icp_residual(align.points(_posed(pm, pT[t])), _posed(gm, gT[t]), workers=1), shape_frames
        ))) * MM if len(shape_frames) else float("nan"),
        "accel_err_mm_f2": M.accel_error(p_cen, g_cen, both) * MM,
        "ang_accel_err_deg_f2": math.degrees(
            M.angular_accel_error(align.R @ pT[:, :3, :3], gT[:, :3, :3], both)
        ),
        "coverage": int(both.sum()) / max(1, n_gt_vis),
        "false_visible": int((pred.obj_visible & ~gt.obj_visible).sum()) / n_gt_hidden if n_gt_hidden else 0.0,
    }

    # Penetration is invariant to the rigid alignment, so each side is measured
    # in its own world, on the same frames.
    def penetration(verts, obj_T, model) -> np.ndarray:
        def depth(t):
            R, tr = obj_T[t, :3, :3], obj_T[t, :3, 3]
            return M.penetration_depth((verts[t] - tr) @ R, model.sdf, workers=1)

        return np.asarray(_pmap(depth, obj_frames))

    pen_pred, pen_gt = penetration(pv, pT, pm), penetration(gv, gT, gm)
    contact = {
        "penetration_err_mm": float(np.abs(pen_pred - pen_gt).mean()) * MM if len(pen_gt) else float("nan"),
        "penetration_pred_mm": float(pen_pred.mean()) * MM if len(pen_pred) else float("nan"),
        "penetration_gt_mm": float(pen_gt.mean()) * MM if len(pen_gt) else float("nan"),
    }
    if gt.ground_plane is not None:
        plane = M.orient_plane(gt.ground_plane, gj[:, bj])
        contact.update({
            "ground_human_pred_mm": _mean(M.plane_penetration(pv_w[t], plane) for t in frames) * MM,
            "ground_human_gt_mm": _mean(M.plane_penetration(gv[t], plane) for t in frames) * MM,
            "ground_object_pred_mm": _mean(
                M.plane_penetration(align.points(_posed(pm, pT[t])), plane) for t in obj_frames
            ) * MM,
            "ground_object_gt_mm": _mean(M.plane_penetration(_posed(gm, gT[t]), plane) for t in obj_frames) * MM,
        })

    return {
        "sequence_id": gt.sequence_id,
        "object": gt.object_name,
        "frames": T,
        "evaluated_frames": int(len(frames)),
        "align": {
            "mode": cfg.align,
            "rot_deg": align.angle_deg,
            "trans_m": float(np.linalg.norm(align.t)),
            "sim3_scale": sim3.s,
        },
        "human": human,
        "object_metrics": obj,
        "contact": contact,
    }


SECTIONS = ("human", "object_metrics", "contact")


def aggregate(per_episode: dict[int, dict]) -> dict:
    out: dict[str, dict] = {}
    for section in SECTIONS:
        keys = {k for res in per_episode.values() for k in res[section]}
        out[section] = {k: _mean(res[section].get(k, float("nan")) for res in per_episode.values()) for k in sorted(keys)}
    out["align"] = {"sim3_scale": _mean(res["align"]["sim3_scale"] for res in per_episode.values())}
    return out


def score(
    gt_root: Path,
    pred_root: Path,
    episodes: list[int] | None = None,
    cfg: Config | None = None,
    pred_mesh_dir: Path | None = None,
    device: str | None = None,
    log=print,
) -> dict:
    from v2hoi.body import SomaBody

    cfg = cfg or Config()
    available = set(list_episodes(pred_root))
    episodes = episodes or [e for e in list_episodes(gt_root) if e in available]
    missing = [e for e in episodes if e not in available]
    if missing:
        raise ValueError(f"prediction root has no episodes {missing}")

    body = SomaBody(device=device)
    objects = ObjectModels(cfg.object_samples)
    per_episode = {}
    for e in episodes:
        start = time.time()
        res = score_episode(load_episode(gt_root, e), load_episode(pred_root, e, pred_mesh_dir), body, objects, cfg)
        per_episode[e] = res
        log(f"episode {e:2d} {res['object']:<20} {time.time() - start:5.0f}s")
    return {
        "gt": str(gt_root),
        "pred": str(pred_root),
        "pred_mesh_dir": str(pred_mesh_dir) if pred_mesh_dir else None,
        "config": asdict(cfg),
        "git": _git_rev(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "mean": aggregate(per_episode),
        "per_episode": per_episode,
    }


COLUMNS = [
    ("human", "chamfer_mm", "h_cham"),
    ("human", "accel_err_body_mm_f2", "h_acc_body"),
    ("human", "accel_err_fingers_mm_f2", "h_acc_fing"),
    ("object_metrics", "chamfer_mm", "o_cham"),
    ("object_metrics", "shape_chamfer_mm", "o_shape"),
    ("object_metrics", "accel_err_mm_f2", "o_acc"),
    ("object_metrics", "ang_accel_err_deg_f2", "o_angacc"),
    ("contact", "penetration_err_mm", "pen_err"),
    ("object_metrics", "coverage", "coverage"),
]


def format_table(report: dict) -> str:
    def fmt(v) -> str:
        return "-" if v is None or not np.isfinite(v) else f"{v:.2f}"

    header = ["ep", "object"] + [c[2] for c in COLUMNS]
    rows = [header]
    for e, res in report["per_episode"].items():
        rows.append([str(e), res["object"]] + [fmt(res[s].get(k)) for s, k, _ in COLUMNS])
    mean = report["mean"]
    rows.append(["mean", ""] + [fmt(mean[s].get(k)) for s, k, _ in COLUMNS])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    units = "units: cham/shape/pen_err mm, acc mm/frame^2, angacc deg/frame^2, coverage fraction"
    return "\n".join(lines + [units])


def _git_rev() -> str | None:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True, help="prediction root (Tier 1 layout)")
    parser.add_argument("--gt", type=Path, default=TIER1_ROOT, help="ground-truth root")
    parser.add_argument("--pred-mesh-dir", type=Path, help="look up prediction meshes here instead of <pred>/mesh")
    parser.add_argument("--episodes", type=int, nargs="*", help="default: every episode in both roots")
    parser.add_argument("--align", choices=["se3", "sim3", "none"], default="se3")
    parser.add_argument("--stride", type=int, default=1, help="frame stride for per-frame mesh metrics")
    parser.add_argument("--device", help="torch device for SOMA-X (default: cuda if available)")
    parser.add_argument("--out", type=Path, help="report JSON (default: scores/<pred>_<time>.json)")
    args = parser.parse_args()

    cfg = Config(align=args.align, stride=args.stride)
    report = score(args.gt, args.pred, args.episodes, cfg, args.pred_mesh_dir, args.device)
    out = args.out or Path("scores") / f"{args.pred.name}_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=1, ensure_ascii=False), encoding="utf-8")
    print(format_table(report))
    print(f"\n{out}")


if __name__ == "__main__":
    main()
