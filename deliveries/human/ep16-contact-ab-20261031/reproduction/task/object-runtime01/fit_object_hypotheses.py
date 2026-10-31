"""Fit image-derived parametric object hypotheses; never claim hidden geometry."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

OBJECT = Path(__file__).resolve().parent
CONFIG = json.loads((OBJECT / "object-config01.json").read_text())
VIDEO = Path(CONFIG["video"])
MASKS = Path(CONFIG["masks"])
BASE = Path(CONFIG["human_joints"]).parent
K = np.load(CONFIG["camera_intrinsics"], allow_pickle=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signed_power(value, exponent):
    return np.where(np.abs(value) < 1e-12, 0.0, np.sign(value) * np.abs(value) ** exponent)


def surface(nxz, ny, height_ratio=1., depth_ratio=1., nlat=24, nlon=64):
    lat = np.linspace(-np.pi / 2, np.pi / 2, nlat + 1)
    lon = np.linspace(0, 2 * np.pi, nlon + 1)
    ll, tt = np.meshgrid(lon, lat)
    radius = signed_power(np.cos(tt), 2 / ny)
    xyz = np.stack([
        radius * signed_power(np.cos(ll), 2 / nxz),
        height_ratio * signed_power(np.sin(tt), 2 / ny),
        depth_ratio * radius * signed_power(np.sin(ll), 2 / nxz),
    ], axis=-1).reshape(-1, 3)
    uv = np.stack([ll / (2 * np.pi), (tt + np.pi / 2) / np.pi], axis=-1).reshape(-1, 2)
    faces = []
    for i in range(nlat):
        for j in range(nlon):
            a = i * (nlon + 1) + j
            b, c, d = a + 1, a + nlon + 1, a + nlon + 2
            if i > 0:
                faces.append([a, c, b])
            if i < nlat - 1:
                faces.append([b, c, d])
    return xyz, np.array(faces), uv


def object_rotation(params):
    yaw, tilt, roll = params[3:6]
    return Rotation.from_euler("z", roll).as_matrix() @ Rotation.from_euler("y", yaw).as_matrix() @ Rotation.from_euler("x", np.pi + tilt).as_matrix()


def project(vertices, params, depth):
    rotation = object_rotation(params)
    trans = np.array([(params[0] - K[0, 2]) * depth / K[0, 0], (params[1] - K[1, 2]) * depth / K[1, 1], depth])
    points = np.exp(params[2]) * vertices @ rotation.T + trans
    homogeneous = points @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def sample_polygon(polygon, count):
    start = polygon
    end = np.roll(polygon, -1, axis=0)
    lengths = np.linalg.norm(end - start, axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    positions = np.linspace(0, cumulative[-1], count, endpoint=False)
    segments = np.minimum(np.searchsorted(cumulative, positions, side="right") - 1, len(lengths) - 1)
    fraction = (positions - cumulative[segments]) / np.maximum(lengths[segments], 1e-8)
    return start[segments] + fraction[:, None] * (end - start)[segments]


def point_polygon_distance(points, polygon):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    edges = b - a
    diff = points[:, None, :] - a[None, :, :]
    fraction = np.clip((diff * edges[None]).sum(-1) / np.maximum((edges * edges).sum(-1), 1e-8), 0, 1)
    closest = a[None] + fraction[:, :, None] * edges[None]
    return np.linalg.norm(points[:, None] - closest, axis=-1).min(axis=1)


class FrameEvidence:
    def __init__(self, index):
        self.index = index
        self.mask = cv2.imread(str(MASKS / "object" / f"{index:06d}.png"), 0) > 0
        self.human = cv2.imread(str(MASKS / "human" / f"{index:06d}.png"), 0) > 0
        contours, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contour = max(contours, key=cv2.contourArea)[:, 0].astype(float)
        self.box = cv2.boundingRect(self.contour.astype(np.float32))
        self.center = self.contour.mean(0)
        self.observed = sample_polygon(self.contour, 96)
        self.ignore = cv2.dilate(self.human.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.observed_visible = map_coordinates(self.ignore.astype(float), self.observed[:, ::-1].T, order=0, mode="constant", cval=0) < .5
        outside = cv2.distanceTransform((~self.mask).astype(np.uint8), cv2.DIST_L2, 5)
        inside = cv2.distanceTransform(self.mask.astype(np.uint8), cv2.DIST_L2, 5)
        self.distance = outside - inside

    def residual(self, params, vertices, depth, previous=None):
        xy = project(vertices, params, depth)
        polygon = xy[ConvexHull(xy).vertices]
        boundary = sample_polygon(polygon, 128)
        sampled = map_coordinates(self.distance, boundary[:, ::-1].T, order=1, mode="constant", cval=50)
        occluded = map_coordinates(self.ignore.astype(float), boundary[:, ::-1].T, order=0, mode="constant", cval=0) > .5
        sampled[occluded] *= .1
        reverse = point_polygon_distance(self.observed, polygon)
        reverse[~self.observed_visible] *= .1
        if previous is None:
            priors = np.zeros(6)
        else:
            # A modest continuity prior breaks silhouette symmetries; it is not
            # independent evidence for yaw or hidden surfaces.
            priors = (params - previous) * np.array([.03, .03, 2., .8, .8, .8])
        return np.r_[sampled, reverse, priors]

    def evaluate(self, params, vertices, depth):
        xy = project(vertices, params, depth)
        polygon = xy[ConvexHull(xy).vertices]
        rendered = np.zeros(self.mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(rendered, np.rint(polygon).astype(np.int32), 1)
        comparable = ~self.ignore
        observed = self.mask & comparable
        predicted = (rendered > 0) & comparable
        intersection = (observed & predicted).sum()
        union = (observed | predicted).sum()
        distance = point_polygon_distance(self.observed[self.observed_visible], polygon)
        return {"visible_mask_iou": float(intersection / max(union, 1)), "visible_contour_mean_px": float(np.mean(distance)) if len(distance) else None,
                "silhouette": polygon, "rendered": rendered}


def fit_frame(evidence, vertices, depth, initial, max_nfev=80, previous=None):
    lower = np.array([500, 450, np.log(.03), -1.6, -.5, -.7])
    upper = np.array([900, 850, np.log(.35), 1.6, 1., .7])
    return least_squares(evidence.residual, np.clip(initial, lower + 1e-6, upper - 1e-6),
                         bounds=(lower, upper), args=(vertices, depth, previous),
                         loss="soft_l1", f_scale=2., diff_step=1e-3, max_nfev=max_nfev,
                         xtol=1e-5, ftol=1e-5, gtol=1e-5)


def export_obj(vertices, faces, uv, path):
    lines = ["# Track 1 episode 16 image-fitted parametric hypothesis; hidden geometry is unverified.", "mtllib object.mtl"]
    lines.extend("v " + " ".join(f"{x:.9f}" for x in point) for point in vertices)
    lines.extend("vt " + " ".join(f"{x:.9f}" for x in point) for point in uv)
    lines.append("usemtl track1_observed_texture")
    lines.extend("f " + " ".join(f"{i+1}/{i+1}" for i in face) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path.parent / "object.mtl").write_text("newmtl track1_observed_texture\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd texture.png\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype", action="store_true")
    parser.add_argument("--output", type=Path, default=OBJECT / "candidate-v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        raise FileExistsError("Use a new candidate directory")
    start = time.perf_counter()
    joints = np.load(BASE / "joints_3d.npy", allow_pickle=False)
    depth = float(np.median(joints[60:270, [42, 78], 2]))
    evidence0 = FrameEvidence(0)
    x, y, width, height = evidence0.box
    initial = np.array([x + width/2, y + height/2, np.log(width * depth / K[0,0] / 2.6), .6, .22, 0.])
    hypotheses = [("rounded_cube_n4", 4., 4.), ("rounded_cube_n8", 8., 8.), ("rounded_cube_n16", 16., 16.), ("rounded_cylinder", 2., 16.)]
    results = []
    best = None
    for name, nxz, ny in hypotheses:
        vertices, faces, uv = surface(nxz, ny, nlat=16, nlon=48)
        alternatives = []
        for yaw in [.25, .7, 1.15]:
            guess = initial.copy()
            guess[3] = yaw
            fit = fit_frame(evidence0, vertices, depth, guess, max_nfev=160)
            score = evidence0.evaluate(fit.x, vertices, depth)
            alternatives.append((score["visible_mask_iou"], fit.x, score))
        iou, params, score = max(alternatives, key=lambda item: item[0])
        record = {"name": name, "nxz": nxz, "ny": ny, "frame0_params": params.tolist(), "frame0_iou": iou, "frame0_contour_mean_px": score["visible_contour_mean_px"]}
        results.append(record)
        print(json.dumps(record), flush=True)
        if best is None or iou > best[0]:
            best = (iou, name, nxz, ny, params)
    (args.output / "hypotheses.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    if args.prototype:
        cap = cv2.VideoCapture(str(VIDEO)); _, frame = cap.read(); cap.release()
        panels = []
        for result in results:
            vertices, _, _ = surface(result["nxz"], result["ny"], nlat=16,nlon=48)
            evaluation = evidence0.evaluate(np.array(result["frame0_params"]), vertices, depth)
            image = frame.copy()
            cv2.polylines(image, [np.rint(evaluation["silhouette"]).astype(np.int32)], True, (0, 0, 255),1)
            image = cv2.resize(image[620:750,635:775],(560,520))
            cv2.putText(image,f'{result["name"]}: {result["frame0_iou"]:.3f}',(5,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),1)
            panels.append(image)
        cv2.imwrite(str(args.output / "hypotheses-frame0.jpg"),np.concatenate(panels,axis=1))
        print(f"Prototype completed in {time.perf_counter()-start:.2f} seconds; depth anchor={depth:.3f}m",flush=True)
        return
    _, name, nxz, ny, first = best
    vertices, faces, uv = surface(nxz, ny, nlat=24, nlon=64)
    poses = []
    diagnostics = []
    previous = first.copy()
    for index in range(360):
        evidence = FrameEvidence(index)
        x,y,w,h = evidence.box
        guess = previous.copy()
        if index:
            # Center-of-mask displacement supplies translation initialization,
            # while the solver matches the actual visible outline.
            guess[:2] += evidence.center - previous_evidence.center
        fit = fit_frame(evidence, vertices, depth, guess, max_nfev=60 if index else 120, previous=previous if index else None)
        previous = fit.x
        previous_evidence = evidence
        poses.append(fit.x)
        score = evidence.evaluate(fit.x, vertices, depth)
        diagnostics.append({"frame": index,"visible_mask_iou":score["visible_mask_iou"],"visible_contour_mean_px":score["visible_contour_mean_px"],"nfev":fit.nfev,"solver_success":bool(fit.success)})
        if (index+1)%30==0:
            print(f"Image-derived object pose {index+1}/360",flush=True)
    params = np.array(poses)
    half_extent = float(np.exp(params[0,2]))
    metric_vertices = vertices * half_extent
    transforms = np.repeat(np.eye(4)[None],360,axis=0)
    for i,p in enumerate(params):
        z = depth * half_extent / np.exp(p[2])
        transforms[i,:3,:3] = object_rotation(p)
        transforms[i,:3,3] = [(p[0]-K[0,2])*z/K[0,0],(p[1]-K[1,2])*z/K[1,1],z]
    export_obj(metric_vertices,faces,uv,args.output/"object.obj")
    np.savez_compressed(args.output/"object_candidate.npz",frame_indices=np.arange(360),object_to_camera=transforms,vertices=metric_vertices,faces=faces,uv=uv,camera_intrinsics=K,fit_parameters=params)
    report = {"source_video":str(VIDEO),"source_video_sha256":digest(VIDEO),"source":"Track 1 episode16 RGB and independently generated SAM2 masks only", "selected_hypothesis":name,"hypotheses":results,"depth_anchor_m":depth,"metric_half_extent_m":half_extent,"scale_assumption":"SAM MHR wrist median depth while held, existing assumed 60 degree HFOV. Monocular scale and hidden geometry unverified; use at least +/-20% sensitivity experiments.","mesh_assumption":"Closed superellipsoid is a fitted geometric hypothesis, not a recovered true backside; profile family chosen by frame0 silhouette only.","coordinate_convention":"Object axes x,y,z; object_to_camera is OpenCV right/down/forward, meters.","diagnostics":diagnostics,"wall_seconds":time.perf_counter()-start,"mean_visible_mask_iou":float(np.mean([d['visible_mask_iou'] for d in diagnostics])),"script_sha256":digest(Path(__file__)),"official_baseline_status":"Input candidate only, not a completed or verified CARI4D reproduction"}
    (args.output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ('diagnostics','hypotheses')},indent=2),flush=True)


if __name__ == "__main__":
    main()
