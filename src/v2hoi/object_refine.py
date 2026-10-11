"""Training-free, episode-local observation refinement of native object tracks.

The human, camera, topology and canonical origin remain fixed. One object scale
is shared by every frame. This is an explicit experimental pipeline stage.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from v2hoi.clips import load_clips
from v2hoi.native import native_environment, sha256


@dataclass(frozen=True)
class RefineConfig:
    steps: int = 60
    batch_size: int = 24
    roi_size: int = 192
    scale_min: float = .8
    scale_max: float = 1.2
    rotation_limit_deg: float = 35.
    learning_rate: float = .025
    refine_scale: bool = True
    preview_codec: str = "auto"
    temporal_strength: float = 1.
    tracking_backend: str = 'observations'
    full_hoi_preview: bool = True

    def validate(self):
        if self.steps < 1 or self.batch_size < 1 or self.roi_size < 64:
            raise ValueError("Positive steps/batches and ROI size >= 64 are required")
        values = [self.scale_min, self.scale_max, self.rotation_limit_deg, self.learning_rate,self.temporal_strength]
        if not np.isfinite(values).all() or not 0 < self.scale_min < 1 < self.scale_max:
            raise ValueError("One finite, positive scale interval must bracket 1")
        if not 0 < self.rotation_limit_deg < 90 or self.learning_rate <= 0:
            raise ValueError("Invalid bounded pose-refinement settings")
        if self.temporal_strength<0: raise ValueError("Temporal strength must be nonnegative")
        if self.preview_codec not in {"auto", "h264_nvenc", "libx264"}:
            raise ValueError("Unsupported preview encoder")
        if self.tracking_backend not in {'observations','foundationpose'}:
            raise ValueError('Unsupported motion initializer')
        return self


def require_timeline(frame_ids, expected_frames):
    if expected_frames < 1 or list(frame_ids) != list(range(expected_frames)):
        raise ValueError("Require every consecutive original-video frame exactly once")


def anchor_indices(visible_pixels, count=8, minimum_pixels=32):
    """Even coverage of supported observations, with no episode-specific IDs."""
    values = np.asarray(visible_pixels)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any() or count < 1:
        raise ValueError("Invalid visible support or anchor budget")
    eligible = np.flatnonzero(values >= minimum_pixels)
    if not len(eligible):
        return np.array([], dtype=np.int64)
    offsets = np.linspace(0, len(eligible)-1, min(count, len(eligible))).round().astype(int)
    return eligible[offsets]


def seed_frame_index(visible_pixels):
    """Select the largest supported observation; ties retain the earliest ID."""
    values = np.asarray(visible_pixels)
    anchor_indices(values)
    if not len(values) or values.max() < 32:
        raise ValueError("No supported seed observation")
    return int(values.argmax())


def observation_bbox(mask):
    """Ignore tiny disconnected islands for ROI initialization, not scoring.

    Substantial disconnected parts, including handles separated by occlusion,
    remain supported. The actual fitting and full-image diagnostics retain masks.
    """
    from scipy.ndimage import label
    labels,count=label(np.asarray(mask,dtype=bool),structure=np.ones((3,3)))
    if not count: raise ValueError("Empty observation bbox")
    sizes=np.bincount(labels.ravel());sizes[0]=0
    supported=sizes>=max(8,float(sizes.max())*.01)
    if not supported.any(): supported[sizes.argmax()]=True
    ys,xs=np.where(supported[labels])
    return int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())


def observation_centers(vertices, rotations, center, scale, surface_depth, uv, k):
    """Center projected geometry on a visible bbox at an estimated front depth.

    Geometry centers need not coincide with projected bbox centers (e.g. handles).
    This is an initialization from observations, not a metric reference alignment.
    """
    relative = (np.asarray(vertices)-np.asarray(center))*scale
    oriented = relative[None] @ np.asarray(rotations).transpose(0, 2, 1)
    z = surface_depth-oriented[..., 2].min(axis=1)
    centers = np.stack([(uv[0]-k[0,2])/k[0,0]*z, (uv[1]-k[1,2])/k[1,1]*z, z], axis=1)
    for _ in range(3):
        camera = oriented+centers[:,None]
        projected = camera @ np.asarray(k).T
        pixels = projected[...,:2]/projected[...,2:]
        midpoint = (pixels.min(axis=1)+pixels.max(axis=1))/2
        centers[:,:2] += (np.asarray(uv)-midpoint)/np.array([k[0,0],k[1,1]])*z[:,None]
    return centers


def face_observation_targets(asset,frames):
    """Weak front/back constraints authored from permitted source observations.

    Signs interpolate through unknown transitions; these are neither reference
    rotations nor stationary-motion annotations.
    """
    rows=asset.get('face_observations',[]) if asset else []
    if not rows: return None,None
    axis=np.asarray(asset.get('semantic_normal'),dtype=np.float64)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:
        raise ValueError('Face observations require a finite local semantic normal')
    ids=[row['original_frame_id'] for row in rows]
    sides=[row['visible_side'] for row in rows]
    source_ids={row['original_frame_id'] for row in asset['observations']}
    if (ids!=sorted(set(ids)) or any(i not in source_ids or i<0 or i>=frames for i in ids)
            or any(side not in {'front','back'} for side in sides)):
        raise ValueError('Face hints require unique permitted original observation frames')
    targets=np.interp(np.arange(frames),ids,[1. if side=='front' else -1. for side in sides])
    return axis/np.linalg.norm(axis),targets


def image_rotation_evidence(rgb,masks,occluders):
    """Weak no-rotation evidence from stable visible image patches, not GT poses."""
    rgb=np.asarray(rgb,dtype=float);visible=np.asarray(masks,dtype=bool)&~np.asarray(occluders,dtype=bool)
    values=[]
    for a,b,va,vb in zip(rgb[:-1],rgb[1:],visible[:-1],visible[1:]):
        common=va&vb;union=va|vb
        if common.sum()<128:values.append(0.);continue
        overlap=common.sum()/max(union.sum(),1)
        difference=np.abs(a[common]-b[common]).mean()
        values.append(float(np.clip((overlap-.93)/.06,0,1)*np.clip((.04-difference)/.03,0,1)))
    return np.asarray(values,dtype=np.float32)


def rotation_anchor_pairs(rgb,masks,occluders,window=60):
    """Compare distant textured patches directly; adjacent similarity cannot prove a hold."""
    rgb=np.asarray(rgb);masks=np.asarray(masks);occluders=np.asarray(occluders)
    if window<2:raise ValueError('Rotation evidence window must span multiple frames')
    rows=[]
    for b in range(2,len(rgb)):
        for a in range(max(0,b-window),b-1,10):
            common=np.asarray(masks[a],bool)&np.asarray(masks[b],bool)&~np.asarray(occluders[a],bool)&~np.asarray(occluders[b],bool)
            # Featureless surfaces can retain their patch under real rotation.
            # Do not anchor them from color similarity alone.
            if common.sum()<128 or min(rgb[a][common].std(axis=0).mean(),rgb[b][common].std(axis=0).mean())<.08:continue
            weight=float(image_rotation_evidence(rgb[[a,b]],masks[[a,b]],occluders[[a,b]])[0])
            if weight>.5:rows.append((a,b,weight))
    return np.asarray(rows,dtype=float).reshape(-1,3)


def appearance_rotation_observable(asset):
    """Missing or legacy appearance never establishes a long-range rotation anchor."""
    value=asset.get('appearance_rotation_observable',False) if asset else False
    if not isinstance(value,bool):
        raise ValueError('Appearance rotation observability must be an explicit boolean observation')
    return value


def landmark_observations(asset,frames):
    """Explicit local geometry to original RGB pixel correspondences, never reference poses."""
    result={}
    appearance_rotation_observable(asset)
    source_ids={row['original_frame_id'] for row in asset.get('observations',[])} if asset else set()
    for row in asset.get('landmark_observations',[]) if asset else []:
        frame=row['original_frame_id'];points=np.asarray(row['points_local'],dtype=float)
        pixels=np.asarray(row['pixels'],dtype=float)
        if (frame not in source_ids or not 0<=frame<frames or frame in result
                or points.ndim!=2 or points.shape[1:]!=(3,) or len(points)<4
                or pixels.shape!=(len(points),2) or not np.isfinite(points).all() or not np.isfinite(pixels).all()):
            raise ValueError('Landmarks require finite paired geometry and permitted original RGB observations')
        result[frame]=(points,pixels)
    return result


def select_orientation_path(rotations,costs,frame_ids,angular_step_deg=10.,weight=.02):
    """Global branch selection avoids independent flips of ambiguous silhouettes."""
    rotations=np.asarray(rotations);costs=np.asarray(costs);frame_ids=np.asarray(frame_ids)
    if (rotations.shape!=(*costs.shape,3,3) or costs.ndim!=2 or len(costs)==0
            or frame_ids.shape!=(len(costs),) or (np.diff(frame_ids)<=0).any()
            or not np.isfinite(rotations).all() or not np.isfinite(costs).all()
            or angular_step_deg<=0 or not np.isfinite(weight).all() or (np.asarray(weight)<0).any()
            or np.asarray(weight).ndim>1 or (np.asarray(weight).ndim==1 and len(weight)!=len(costs)-1)):
        raise ValueError('Invalid orientation-path candidates')
    accumulated=costs[0].copy();parents=[]
    for i in range(1,len(costs)):
        traces=np.einsum('aij,bij->ab',rotations[i-1],rotations[i])
        angles=np.arccos(np.clip((traces-1)/2,-1,1))
        strength=weight[i-1] if np.asarray(weight).ndim else weight
        transition=strength*(angles/(np.deg2rad(angular_step_deg)*(frame_ids[i]-frame_ids[i-1])))**2
        alternatives=accumulated[:,None]+transition
        predecessor=alternatives.argmin(axis=0);parents.append(predecessor)
        accumulated=costs[i]+alternatives[predecessor,np.arange(len(predecessor))]
    choices=[int(accumulated.argmin())]
    for parent in reversed(parents):choices.append(int(parent[choices[-1]]))
    return np.asarray(choices[::-1],dtype=np.int64)


def camera_centers_to_poses(rotations, camera_centers, canonical_center, scale=1.):
    """Bake scale into vertices exactly once and preserve the canonical origin."""
    rotations = np.asarray(rotations, dtype=np.float64)
    camera_centers = np.asarray(camera_centers, dtype=np.float64)
    center = np.asarray(canonical_center, dtype=np.float64)
    if (rotations.ndim != 3 or rotations.shape[1:] != (3, 3)
            or camera_centers.shape != (len(rotations), 3) or center.shape != (3,)
            or not np.isfinite(rotations).all() or not np.isfinite(camera_centers).all()
            or not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0):
        raise ValueError("Invalid shared-scale object geometry/pose contract")
    if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-4)):
        raise ValueError("Object rotations must be proper")
    poses = np.repeat(np.eye(4)[None], len(rotations), axis=0)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = camera_centers - (rotations @ (center*scale)[:, None])[..., 0]
    return poses


def scene_identity(human_parameters, intrinsics):
    """Prevent combining per-object results from incompatible human/camera runs."""
    digest=hashlib.sha256()
    for name,value in sorted(human_parameters.items()):
        array=np.asarray(value)
        if not np.isfinite(array).all(): raise ValueError("Nonfinite native human parameters")
        digest.update(name.encode())
        digest.update(str((array.dtype.str,array.shape)).encode())
        digest.update(array.tobytes(order="C"))
    k=np.asarray(intrinsics,dtype=np.float64)
    if k.shape!=(3,3) or not np.isfinite(k).all(): raise ValueError("Invalid scene camera")
    digest.update(k.tobytes())
    return digest.hexdigest()


def candidate_paths(run, candidate):
    """Explicit downstream selection with input and serialized mesh/pose checks."""
    run,candidate=Path(run).resolve(),Path(candidate).resolve()
    identity=json.loads((run/"native-inputs.json").read_text())
    report=json.loads((candidate/"result.json").read_text())
    if (identity.get("provenance")!="track1" or report.get("provenance")!="track1"
            or report.get("episode")!=identity["episode"] or report.get("frames")!=identity["expected_frames"]
            or report.get("video_sha256")!=identity["video_sha256"] or report.get("human_parameters_changed") is not False):
        raise ValueError("Object candidate does not match the original Track 1 native run")
    original=run/"native"/f"episode_{identity['episode']:06d}"/"inference/refined.pth"
    if report["inputs"].get(str(original.resolve()))!=sha256(original):
        raise ValueError("Object candidate source bundle changed")
    paths=[]
    for name in ("refined.pth","object.glb"):
        path=candidate/name
        if report.get("outputs",{}).get(name)!=sha256(path):
            raise ValueError("Serialized object candidate changed")
        paths.append(path)
    return tuple(paths)


def plan_jobs(runs, dataset_root, asset_manifests=()):
    """Validate all jobs before GPU work; missing prerequisites never create fakes."""
    jobs, instances = [], set()
    assets={}
    for value in asset_manifests:
        path=Path(value).resolve();asset=json.loads(path.read_text())
        if (asset.get("schema")!="v2hoi.observed-asset.v1" or asset.get("provenance")!="track1"
                or asset.get('entry')!='blender.bpy.observed-profile'
                or asset.get("reference_geometry_used") is not False):
            raise ValueError("Replacement assets require verified Track 1 observational provenance")
        key=(asset['episode'],asset['object_instance_id'])
        if key in assets: raise ValueError("Duplicate replacement asset")
        mesh=(path.parent/asset['mesh']['path']).resolve()
        if sha256(mesh)!=asset['mesh']['sha256']: raise ValueError("Replacement mesh changed")
        if not asset.get('observations'): raise ValueError("Replacement asset needs source observations")
        for observation in asset['observations']:
            if sha256(path.parent/observation['path'])!=observation['sha256']:
                raise ValueError("Replacement asset observation changed")
        assets[key]=(path,asset)
    for value in runs:
        run = Path(value).resolve()
        identity = json.loads((run / "native-inputs.json").read_text())
        if identity.get("provenance") != "track1":
            raise ValueError("Object refinement accepts Track 1 runs only")
        episode = identity["episode"]
        clip = load_clips("track1", [episode], dataset_root)[0]
        object_id = identity.get("object_instance_id", clip.object_name)
        if not isinstance(object_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", object_id):
            raise ValueError("Object instance ID must be a safe, explicit name")
        instance = (episode, object_id)
        if instance in instances:
            raise ValueError("Duplicate episode/object run; compare alternatives in separate batches")
        require_timeline(identity["original_frame_ids"], clip.n_frames)
        if not clip.video.is_file() or sha256(clip.video) != identity["video_sha256"]:
            raise ValueError("Original Track 1 source video is missing or changed")
        if not json.loads((run / "native-status.json").read_text()).get("complete"):
            raise ValueError("Native source reconstruction is incomplete")
        root = run / "native" / f"episode_{episode:06d}"
        for path in (root / "inference/refined.pth", root / "inference/coconet.pth",
                     root / "depth/moge2/aligned.h5", root / "export" / f"episode_{episode:06d}" / "wild_export.json"):
            if not path.is_file():
                raise FileNotFoundError(f"Missing observation-stage prerequisite: {path}")
        selected={**identity, "object_instance_id":object_id}
        if instance not in assets and identity.get('preparation',{}).get('mesh_generation',{}).get('entry')=='blender.bpy.observed-profile':
            receipt=identity['preparation']['mesh_generation']['receipt']
            preparation_path=Path(identity['preparation_manifest'])
            asset_path=(preparation_path.parent/receipt['path']).resolve()
            if sha256(asset_path)!=receipt['sha256']:raise ValueError('Native bpy preparation receipt changed')
            asset=json.loads(asset_path.read_text())
            asset_mesh=asset_path.parent/asset['mesh']['path']
            if (asset.get('schema')!='v2hoi.observed-asset.v1' or asset.get('entry')!='blender.bpy.observed-profile'
                    or (asset.get('episode'),asset.get('object_instance_id'))!=instance
                    or asset.get('provenance')!='track1' or asset.get('reference_geometry_used') is not False
                    or sha256(asset_mesh)!=asset['mesh']['sha256']):raise ValueError('Native bpy asset provenance changed')
            for row in asset['observations']:
                if sha256(asset_path.parent/row['path'])!=row['sha256']:raise ValueError('Native bpy observation changed')
            assets[instance]=(asset_path,asset)
        if instance in assets:
            path,asset=assets[instance]
            if asset['video_sha256']!=identity['video_sha256'] or any(
                    row['original_frame_id']<0 or row['original_frame_id']>=clip.n_frames for row in asset['observations']):
                raise ValueError("Replacement asset does not belong to this original video")
            face_observation_targets(asset,clip.n_frames)
            landmark_observations(asset,clip.n_frames)
            selected['asset_manifest']=str(path)
        jobs.append((run, clip, selected))
        instances.add(instance)
    if not jobs:
        raise ValueError("No native runs supplied")
    if assets.keys()-instances: raise ValueError("Unused replacement asset; specify its native run")
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--asset-manifests",type=Path,nargs="*",default=[],help="Optional observed replacement assets per episode/object")
    p.add_argument("--dataset-root", type=Path, default=Path("data/v2d/track_1"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--roi-size", type=int, default=192)
    p.add_argument("--scale-min", type=float, default=.8)
    p.add_argument("--scale-max", type=float, default=1.2)
    p.add_argument("--rotation-limit-deg", type=float, default=35.)
    p.add_argument("--fixed-scale", action="store_true", help="Ablation: refine pose with the source mesh scale fixed")
    p.add_argument("--preview-codec", choices=("auto","h264_nvenc","libx264"), default="auto")
    p.add_argument("--temporal-strength",type=float,default=1.,help="Observation-constrained motion continuity; zero for ablation")
    p.add_argument('--tracking-backend',choices=('observations','foundationpose'),default='observations',
        help='Observation mainline makes no FoundationPose calls; foundationpose is an explicit comparison')
    p.add_argument('--object-only-preview',action='store_true',help='Explicit diagnostic ablation; default renders the complete native HOI')
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    config = RefineConfig(a.steps, a.batch_size, a.roi_size, a.scale_min, a.scale_max,
                          a.rotation_limit_deg, refine_scale=not a.fixed_scale, preview_codec=a.preview_codec,
                          temporal_strength=a.temporal_strength,tracking_backend=a.tracking_backend,
                          full_hoi_preview=not a.object_only_preview).validate()
    jobs = plan_jobs(a.runs, a.dataset_root,a.asset_manifests)
    if config.tracking_backend=='observations' and any('asset_manifest' not in identity for _,_,identity in jobs):
        raise ValueError('The bpy observation mainline requires an observed asset for every object; no automatic generated-mesh fallback')
    if not a.worker:
        if a.output.exists():
            raise FileExistsError("Preserve existing candidates; choose a new output")
        identity = jobs[0][2]
        weights = Path(identity["command"][identity["command"].index("--weights_path")+1])
        env = native_environment(Path("third_party/CARI4D_MHR_Win").resolve(), weights)
        subprocess.run([sys.executable, "-m", "v2hoi.cuda_budget", "--fraction", ".75",
                        "--report", str(a.output.parent/(a.output.name+"-memory.json")), "--",
                        "-m", "v2hoi.object_refine", *sys.argv[1:], "--worker"], env=env, check=True)
        return
    from v2hoi.object_refine_runtime import refine_native_run
    a.output.mkdir(parents=True, exist_ok=False)
    reports = []
    scenes = {}
    for run, clip, identity in jobs:
        result=refine_native_run(run, clip, identity,
            a.output/f"episode_{clip.episode:06d}"/identity["object_instance_id"], config)
        if clip.episode in scenes and scenes[clip.episode]!=result["scene_identity"]:
            raise ValueError("Objects in one episode require identical native humans and camera")
        scenes[clip.episode]=result["scene_identity"]
        reports.append(result)
    (a.output/"batch.json").write_text(json.dumps({"schema":"v2hoi.object-refine-batch.v1",
        "provenance":"track1", "config":asdict(config), "episodes":reports,
        "scope":"Episode-local experimental object refinement; native humans retained; no submission conversion"},
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
