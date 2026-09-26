"""Score predictions against Track 2 Tier 1 ground truth.

    python -m v2hoi.score --pred <root> [--gt <tier1 root>] [--episodes 7 9] [--stride 2] [--strict]

Both roots use the Tier 1 layout (see v2hoi.dataset). The report leads with
the five Track 1 leaderboard numbers in cm (CD-H, CD-O, ACC-H, ACC-O, PEN);
everything else is a diagnostic. It also says whether the run used the
official settings and whether the prediction would be a valid submission.
Rules and metric definitions: docs/design.md, sections 5 and 6. The official
script is not published, so the metric internals are an approximation.

Sanity checks (both are flagged as invalid submissions, because the
prediction meshes are the reference meshes):

    # ground truth against itself: every error ~0
    python -m v2hoi.score --pred data/v2d/track_2/tier_1_multiview_caption
    # Tier 2 keeps the original meshes, so point it at the Tier 1 mesh folder
    python -m v2hoi.score --pred data/v2d/track_2/tier_2_synthetic_noise \
        --pred-mesh-dir data/v2d/track_2/tier_1_multiview_caption/mesh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
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

# Track 1 leaderboard: one Kaggle competition per metric
# (v2d-challenge-track1-cd-h, -cd-o, -acc-h, -acc-o, -pen), lower is better, cm.
# key -> (section, local key in mm). PEN has no live competition yet.
LEADERBOARD = {
    "cd_h_cm": ("human", "chamfer_mm"),
    "cd_o_cm": ("object_metrics", "chamfer_mm"),
    "acc_h_cm": ("human", "accel_mm_f2"),
    "acc_o_cm": ("object_metrics", "accel_mm_f2"),
    "interpenetration_cm": ("contact", "penetration_err_mm"),
}

# Largest side of an object mesh's bounding box. Track 1 objects run from a
# brush to a desk; outside this range the mesh is almost surely in mm or cm.
MESH_EXTENT_M = (0.02, 3.0)


@dataclass
class Config:
    # first: Sim(3) on the first frame's body joints, applied to the whole clip
    # (the official rule). se3 / sim3: fitted on body joints over all frames.
    align: str = "first"
    stride: int = 1  # frame stride for the per-frame mesh metrics
    object_samples: int = 10_000
    shape_frames: int = 10

    def deviations(self) -> list[str]:
        """Settings that make the numbers differ from an official run."""
        out = []
        if self.align != "first":
            out.append(f"align={self.align} (official: first-frame Sim(3))")
        if self.stride != 1:
            out.append(f"stride={self.stride} (official: every frame)")
        return out


@dataclass
class ObjectModel:
    samples: np.ndarray  # (N, 3) surface samples, canonical frame
    centroid: np.ndarray  # (3,)
    sdf: SurfaceSDF

    @property
    def extent(self) -> float:
        return float(np.ptp(self.samples, axis=0).max())


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


def _has_pose(obj_T: np.ndarray) -> np.ndarray:
    return np.isfinite(obj_T).all(axis=(1, 2))


def check_scorable(gt: Episode, pred: Episode) -> None:
    """Breaks that make an episode impossible to score at all."""
    if len(pred) != len(gt):
        raise ValueError(f"episode {gt.index}: prediction has {len(pred)} frames, ground truth has {len(gt)}")
    human = {
        "pose": pred.pose, "translation": pred.transl, "identity": pred.identity,
        "scale": pred.scale, "bone_flex": pred.bone_flex,
    }
    bad = [name for name, x in human.items() if not np.isfinite(x).all()]
    if bad:
        raise ValueError(f"episode {gt.index}: human parameters not finite: {', '.join(bad)}")


def check_prediction(pred: Episode) -> list[str]:
    """Submission rules that still leave the episode scorable."""
    problems = []
    missing = np.flatnonzero(~_has_pose(pred.obj_T))
    if len(missing):
        problems.append(
            f"object pose missing on {len(missing)} of {len(pred)} frames "
            f"(first: {missing[:5].tolist()}); occluded frames need a pose too"
        )
    return problems


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reference_copy(mesh: Path, reference_root: Path) -> Path | None:
    """The reference mesh file that ``mesh`` is a byte copy of, if any.

    Track 1 may not use Track 2 assets. This only catches verbatim copies,
    which is the likely accident (pointing the pipeline at the GT mesh folder).
    """
    size = Path(mesh).stat().st_size
    candidates = [p for p in (Path(reference_root) / "mesh").rglob("*") if p.is_file() and p.stat().st_size == size]
    if not candidates:
        return None
    digest = _sha256(mesh)
    return next((p for p in candidates if _sha256(p) == digest), None)


def check_mesh(name: str, path: Path, model: ObjectModel, reference_root: Path) -> list[str]:
    problems = []
    lo, hi = MESH_EXTENT_M
    if not lo <= model.extent <= hi:
        problems.append(f"mesh {name} is {model.extent:.3g} across; expected metres ({lo}-{hi})")
    copy = reference_copy(path, reference_root)
    if copy is not None:
        problems.append(f"mesh {name} is a copy of reference asset {copy.name}; Track 2 assets are not allowed")
    return problems


def score_episode(gt: Episode, pred: Episode, body, objects: ObjectModels, cfg: Config) -> dict:
    check_scorable(gt, pred)
    T = len(gt)
    frames = np.arange(0, T, cfg.stride)

    gj, gv = body(gt)
    pj, pv = body(pred)

    # Put the prediction into the ground-truth world using body joints only.
    # The organizer did not say which points the first-frame fit uses; body
    # joints are our choice. The whole-clip Sim(3) scale is a diagnostic.
    bj, fj = body.body_ids, body.finger_ids
    src, dst = pj[:, bj].reshape(-1, 3), gj[:, bj].reshape(-1, 3)
    sim3 = Similarity.fit(src, dst, with_scale=True)
    align = {
        "first": lambda: Similarity.fit(pj[0, bj], gj[0, bj], with_scale=True),
        "none": Similarity.identity,
        "se3": lambda: Similarity.fit(src, dst, with_scale=False),
        "sim3": lambda: sim3,
    }[cfg.align]()
    pj_w, pv_w = align.points(pj), align.points(pv)

    human = {
        "chamfer_mm": _mean(_pmap(lambda t: M.chamfer(pv_w[t], gv[t], workers=1), frames)) * MM,
        "mpjpe_mm": float(_norm(pj_w - gj).mean()) * MM,
        "mpjpe_body_mm": float(_norm(pj_w[:, bj] - gj[:, bj]).mean()) * MM,
        # Official smoothness: second difference of the prediction alone.
        # *_ref is the same quantity on the reference, for scale.
        "accel_mm_f2": M.accel_magnitude(pj_w) * MM,
        "accel_body_mm_f2": M.accel_magnitude(pj_w[:, bj]) * MM,
        "accel_fingers_mm_f2": M.accel_magnitude(pj_w[:, fj]) * MM,
        "accel_ref_mm_f2": M.accel_magnitude(gj) * MM,
        # Diagnostic: difference from the reference's acceleration.
        "accel_err_mm_f2": M.accel_error(pj_w, gj) * MM,
        "accel_err_body_mm_f2": M.accel_error(pj_w[:, bj], gj[:, bj]) * MM,
        "accel_err_fingers_mm_f2": M.accel_error(pj_w[:, fj], gj[:, fj]) * MM,
    }

    # The prediction's visibility flag is ignored: a valid submission has a pose
    # on every frame. Frames without one are skipped here and flagged by
    # check_prediction.
    gm, pm = objects(gt.mesh_path), objects(pred.mesh_path)
    gT, pT = gt.obj_T, pred.obj_T
    has_pose = _has_pose(pT)
    both = gt.obj_visible & has_pose
    obj_frames = frames[both[frames]]
    shape_frames = obj_frames[
        np.unique(np.linspace(0, len(obj_frames) - 1, min(cfg.shape_frames, len(obj_frames))).astype(int))
    ] if len(obj_frames) else obj_frames
    g_cen = gT[:, :3, :3] @ gm.centroid + gT[:, :3, 3]
    p_cen = align.points(pT[:, :3, :3] @ pm.centroid + pT[:, :3, 3])

    obj = {
        "chamfer_mm": _mean(_pmap(
            lambda t: M.chamfer(align.points(_posed(pm, pT[t])), _posed(gm, gT[t]), workers=1), obj_frames
        )) * MM,
        "shape_chamfer_mm": float(np.median(_pmap(
            lambda t: M.icp_residual(align.points(_posed(pm, pT[t])), _posed(gm, gT[t]), workers=1), shape_frames
        ))) * MM if len(shape_frames) else float("nan"),
        # Official smoothness of the predicted trajectory alone, over the whole
        # clip including occluded frames, on the mesh centroid so the choice
        # of mesh origin does not matter.
        "accel_mm_f2": M.accel_magnitude(p_cen, has_pose) * MM,
        "ang_accel_deg_f2": math.degrees(M.angular_accel_magnitude(pT[:, :3, :3], has_pose)),
        "accel_ref_mm_f2": M.accel_magnitude(g_cen, gt.obj_visible) * MM,
        "ang_accel_ref_deg_f2": math.degrees(M.angular_accel_magnitude(gT[:, :3, :3], gt.obj_visible)),
        # Diagnostics: difference from the reference.
        "accel_err_mm_f2": M.accel_error(p_cen, g_cen, both) * MM,
        "ang_accel_err_deg_f2": math.degrees(
            M.angular_accel_error(align.R @ pT[:, :3, :3], gT[:, :3, :3], both)
        ),
        # Share of the reference's visible frames where the prediction has a pose.
        "coverage": int(both.sum()) / max(1, int(gt.obj_visible.sum())),
    }

    # Each side is measured in its own world, on the same frames. Rotation and
    # translation do not change depths; the alignment scale does, so apply it.
    def penetration(verts, obj_T, model) -> np.ndarray:
        def depth(t):
            R, tr = obj_T[t, :3, :3], obj_T[t, :3, 3]
            return M.penetration_depth((verts[t] - tr) @ R, model.sdf, workers=1)

        return np.asarray(_pmap(depth, obj_frames))

    pen_pred, pen_gt = penetration(pv, pT, pm) * align.s, penetration(gv, gT, gm)
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

    res = {
        "sequence_id": gt.sequence_id,
        "object": gt.object_name,
        "frames": T,
        "evaluated_frames": int(len(frames)),
        "align": {
            "mode": cfg.align,
            "scale": align.s,
            "rot_deg": align.angle_deg,
            "trans_m": float(np.linalg.norm(align.t)),
            "sim3_scale": sim3.s,
        },
        "human": human,
        "object_metrics": obj,
        "contact": contact,
        "violations": check_prediction(pred),
    }
    res["leaderboard"] = {key: res[s][k] / 10.0 for key, (s, k) in LEADERBOARD.items()}
    return res


SECTIONS = ("leaderboard", "human", "object_metrics", "contact")


def aggregate(per_episode: dict[int, dict]) -> dict:
    out: dict[str, dict] = {}
    for section in SECTIONS:
        keys = {k for res in per_episode.values() for k in res[section]}
        out[section] = {k: _mean(res[section].get(k, float("nan")) for res in per_episode.values()) for k in sorted(keys)}
    out["align"] = {
        key: _mean(res["align"][key] for res in per_episode.values()) for key in ("scale", "sim3_scale")
    }
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
    reference = list_episodes(gt_root)
    available = set(list_episodes(pred_root))
    subset = bool(episodes)
    episodes = episodes or reference
    missing = [e for e in episodes if e not in available]
    if missing:
        hint = "" if subset else "; pass --episodes to score a subset"
        raise ValueError(f"prediction root has no episodes {missing}{hint}")

    body = SomaBody(device=device)
    objects = ObjectModels(cfg.object_samples)
    per_episode, meshes = {}, {}
    for e in episodes:
        start = time.time()
        pred = load_episode(pred_root, e, pred_mesh_dir, mask_hidden=False)
        res = score_episode(load_episode(gt_root, e), pred, body, objects, cfg)
        per_episode[e] = res
        meshes[pred.object_name] = pred.mesh_path
        log(f"episode {e:2d} {res['object']:<20} {time.time() - start:5.0f}s")

    violations = [f"episode {e}: {msg}" for e, res in per_episode.items() for msg in res["violations"]]
    for name, path in meshes.items():
        violations += check_mesh(name, path, objects(path), gt_root)
    deviations = cfg.deviations()
    if set(episodes) != set(reference):
        deviations.append(f"scored {len(episodes)} of {len(reference)} episodes")
    return {
        "gt": str(gt_root),
        "pred": str(pred_root),
        "pred_mesh_dir": str(pred_mesh_dir) if pred_mesh_dir else None,
        "config": asdict(cfg),
        "git": _git_rev(),
        "created": datetime.now().isoformat(timespec="seconds"),
        # official_settings: the run matches the official evaluation settings.
        # valid: the prediction obeys the submission rules. Only when both hold
        # are the leaderboard numbers comparable to Kaggle (up to the metric
        # approximations).
        "submission": {
            "official_settings": not deviations,
            "deviations": deviations,
            "valid": not violations,
            "violations": violations,
        },
        "mean": aggregate(per_episode),
        "per_episode": per_episode,
    }


# (section, key, label, factor to the printed unit). The five leaderboard
# columns come first; the rest are diagnostics, also in cm.
COLUMNS = [
    ("leaderboard", "cd_h_cm", "CD-H", 1.0),
    ("leaderboard", "cd_o_cm", "CD-O", 1.0),
    ("leaderboard", "acc_h_cm", "ACC-H", 1.0),
    ("leaderboard", "acc_o_cm", "ACC-O", 1.0),
    ("leaderboard", "interpenetration_cm", "PEN", 1.0),
    ("human", "accel_ref_mm_f2", "ACC-H_ref", 0.1),
    ("object_metrics", "accel_ref_mm_f2", "ACC-O_ref", 0.1),
    ("object_metrics", "shape_chamfer_mm", "o_shape", 0.1),
    ("object_metrics", "coverage", "coverage", 1.0),
]


def format_table(report: dict) -> str:
    def fmt(v, factor) -> str:
        return "-" if v is None or not np.isfinite(v) else f"{v * factor:.3f}"

    header = ["ep", "object"] + [c[2] for c in COLUMNS]
    rows = [header]
    for e, res in report["per_episode"].items():
        rows.append([str(e), res["object"]] + [fmt(res[s].get(k), f) for s, k, _, f in COLUMNS])
    mean = report["mean"]
    rows.append(["mean", ""] + [fmt(mean[s].get(k), f) for s, k, _, f in COLUMNS])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    lines.append(
        "units: cm as on the leaderboard (ACC per frame^2 at 30 fps, prediction alone; "
        "_ref = same on the reference); PEN is a proxy, no live competition yet; coverage fraction"
    )

    sub = report["submission"]
    lines.append("")
    lines.append("official settings: " + ("yes" if sub["official_settings"] else "no"))
    lines += [f"  - {d}" for d in sub["deviations"]]
    lines.append("valid submission: " + ("yes" if sub["valid"] else "NO"))
    lines += [f"  - {v}" for v in sub["violations"]]
    return "\n".join(lines)


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
    parser.add_argument("--episodes", type=int, nargs="*", help="default: every reference episode, all required")
    parser.add_argument(
        "--align", choices=["first", "se3", "sim3", "none"], default="first",
        help="first: Sim(3) on frame 0 (official rule); se3/sim3: whole clip; none: as submitted",
    )
    parser.add_argument("--stride", type=int, default=1, help="frame stride for per-frame mesh metrics")
    parser.add_argument("--device", help="torch device for SOMA-X (default: cuda if available)")
    parser.add_argument("--out", type=Path, help="report JSON (default: scores/<pred>_<time>.json)")
    parser.add_argument(
        "--strict", action="store_true",
        help="exit 1 unless the settings are official and the prediction is a valid submission",
    )
    args = parser.parse_args()

    cfg = Config(align=args.align, stride=args.stride)
    report = score(args.gt, args.pred, args.episodes, cfg, args.pred_mesh_dir, args.device)
    out = args.out or Path("scores") / f"{args.pred.name}_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=1, ensure_ascii=False), encoding="utf-8")
    print(format_table(report))
    print(f"\n{out}")
    sub = report["submission"]
    if args.strict and not (sub["official_settings"] and sub["valid"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
