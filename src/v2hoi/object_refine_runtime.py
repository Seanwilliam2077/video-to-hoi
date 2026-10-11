"""GPU implementation for the generic observation-refinement pipeline stage."""
from __future__ import annotations

from dataclasses import asdict
import copy
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from v2hoi.native import sha256
from v2hoi.object_refine import (anchor_indices, camera_centers_to_poses, require_timeline,
    scene_identity, seed_frame_index, observation_centers, observation_bbox,face_observation_targets,
    select_orientation_path,image_rotation_evidence,rotation_anchor_pairs,landmark_observations)
from v2hoi.pose_review import orientation_hypotheses, rigid_pose


class ObservationRenderer:
    """Differentiable vertex-color/silhouette rendering in OpenCV coordinates."""

    def __init__(self, mesh, device="cuda"):
        import torch
        import nvdiffrast.torch as dr
        self.torch, self.dr = torch, dr
        self.context = dr.RasterizeCudaContext()
        self.vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        self.faces = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
        self.colors = torch.tensor(np.asarray(mesh.visual.vertex_colors)[:, :3]/255., dtype=torch.float32, device=device)[None]

    def render(self, rotations, centers, scale, canonical_center, intrinsics, size):
        import Utils
        torch, dr = self.torch, self.dr
        relative = (self.vertices-canonical_center)*scale
        if relative.ndim==2: relative=relative[None]
        camera = relative @ rotations.transpose(1, 2) + centers[:, None]
        projections = torch.tensor(np.stack([Utils.projection_matrix_from_intrinsics(k,
            height=size[0], width=size[1], znear=.01, zfar=100.) for k in intrinsics]),
            dtype=torch.float32, device=camera.device)
        flip = torch.tensor(Utils.glcam_in_cvcam, dtype=torch.float32, device=camera.device)
        homogeneous = torch.cat([camera, torch.ones_like(camera[..., :1])], dim=-1)
        clip = homogeneous @ (projections @ flip).transpose(1, 2)
        raster, _ = dr.rasterize(self.context, clip, self.faces, resolution=size)
        color, _ = dr.interpolate(self.colors, raster, self.faces)
        z, _ = dr.interpolate(camera[..., 2:3].contiguous(), raster, self.faces)
        foreground = (raster[..., 3:4] > 0).float()
        color = dr.antialias(color*foreground, raster, clip, self.faces)
        foreground = dr.antialias(foreground, raster, clip, self.faces)
        return color.flip(1), foreground.flip(1), z.flip(1)


def refine_native_run(run, clip, identity, output, config):
    import av
    import cv2
    import torch
    import torch.nn.functional as functional
    import trimesh
    import Utils
    from pytorch3d.transforms import so3_exp_map
    from lib_mhr.object_pose_frame import resolve_object_pose_frame
    from prep.mhr_depth_h5 import read_metric_depth
    from prep.mhr_export_utils import camera_calibration, load_edex, read_mask

    logging.getLogger().setLevel(logging.WARNING)
    runtime_hash = sha256(Path(__file__))
    root = run/"native"/f"episode_{clip.episode:06d}"
    export = root/"export"/f"episode_{clip.episode:06d}"
    bundle_path = root/"inference/refined.pth"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if bundle.get("gt") or bundle["metadata"].get("ground_truth_used") is not False:
        raise ValueError("Reference-bearing native bundles are prohibited")
    require_timeline([int(row["src_frame"]) for row in bundle["frame_meta"]], clip.n_frames)
    export_identity = json.loads((export/"wild_export.json").read_text())
    prepared_mesh = Path(identity["command"][identity["command"].index("--object_mesh_path")+1])
    source_masks = Path(identity["command"][identity["command"].index("--mask_h5_path")+1])
    if sha256(prepared_mesh) != identity["preparation"]["mesh"]["sha256"]:
        raise ValueError("Prepared mesh identity changed")
    if sha256(source_masks) != identity["preparation"]["masks"]["sha256"]:
        raise ValueError("Prepared Track 1 segmentation changed")
    mesh_path = Path(export_identity["object_mesh_file"])
    if mesh_path.resolve() != Path(bundle["metadata"]["object_mesh"]).resolve():
        raise ValueError("Native object mesh and trajectory pairing differs")
    source_to_aligned = rigid_pose(export_identity["source_object_mesh_to_aligned_transform"])
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    source_mesh = trimesh.load(prepared_mesh, force="mesh", process=False)
    source_mesh.apply_transform(source_to_aligned)
    if (not np.array_equal(mesh.faces, source_mesh.faces)
            or not np.allclose(mesh.vertices, source_mesh.vertices, atol=1e-6, rtol=0)
            or not np.array_equal(mesh.visual.vertex_colors, source_mesh.visual.vertex_colors)):
        raise ValueError("Canonical mesh lost its verified geometry, scale or appearance")
    frame = resolve_object_pose_frame(bundle["metadata"])
    # Materialize the exact training-frame geometry paired with pose_abs.
    mesh.apply_transform(frame.mesh_to_training)
    if mesh.visual.kind != "vertex":
        raise ValueError("Observation refinement currently requires native vertex-color geometry")
    base = np.asarray(bundle["pr"]["pose_abs"], dtype=np.float64)
    for pose in base:
        rigid_pose(pose)
    k, world_to_camera = camera_calibration(load_edex(export), 0)
    if not np.allclose(world_to_camera, np.eye(4), atol=1e-6):
        raise ValueError("The native static-camera adapter expects scene space to equal camera space")
    scene_hash=scene_identity({key:np.asarray(value) for key,value in bundle["pr"].items() if key.startswith("mhr_")}, k)
    output.mkdir(parents=True, exist_ok=False)
    snapshots = {str(path.resolve()):sha256(path) for path in (bundle_path, mesh_path, prepared_mesh,
        root/"depth/moge2/aligned.h5", export/"wild_export.json", run/"native-inputs.json", source_masks,
        export/"human_masks/front_stereo_camera_left.h5", export/"object_masks/front_stereo_camera_left.h5")}
    baseline_mesh=mesh.copy()
    asset=None
    asset_observations={}
    if "asset_manifest" in identity:
        from PIL import Image
        asset_path=Path(identity['asset_manifest']);asset=json.loads(asset_path.read_text())
        replacement=asset_path.parent/asset['mesh']['path']
        mesh=trimesh.load(replacement,force="mesh",process=False)
        if mesh.visual.kind!='vertex':
            raise ValueError("Replacement assets require verified explicit vertex colors, not a material fallback")
        if not mesh.is_watertight or len(mesh.faces)>4096 or not np.isfinite(mesh.vertices).all():
            raise ValueError("Replacement asset must be a finite closed low-budget surface")
        snapshots[str(asset_path.resolve())]=sha256(asset_path)
        snapshots[str(replacement.resolve())]=sha256(replacement)
        for row in asset['observations']:
            path=asset_path.parent/row['path'];snapshots[str(path.resolve())]=sha256(path)
            asset_observations[row['original_frame_id']]=np.asarray(Image.open(path).convert('RGB'))
    semantic_axis,face_targets=face_observation_targets(asset,clip.n_frames)
    landmarks=landmark_observations(asset,clip.n_frames)
    def face_cost(rotations,centers,ids):
        if semantic_axis is None: return torch.zeros(len(rotations),device='cuda')
        normal=rotations @ torch.tensor(semantic_axis,dtype=torch.float32,device='cuda')
        toward=-functional.normalize(centers,dim=1)
        sign=torch.tensor(face_targets[ids],dtype=torch.float32,device='cuda')
        facing=(normal*toward).sum(1)
        return .8*sign.abs()*torch.relu(.2-sign*facing).square()
    depth_path = root/"depth/moge2/aligned.h5"
    center_np = mesh.bounds.mean(axis=0)
    canonical_center = torch.tensor(center_np, dtype=torch.float32, device="cuda")
    def landmark_cost(rotations,centers,scale,ids):
        values=[]
        for j,frame_id in enumerate(ids):
            if int(frame_id) not in landmarks:
                values.append(rotations[j].sum()*0);continue
            points,pixels=landmarks[int(frame_id)]
            local=torch.tensor(points-center_np,dtype=torch.float32,device='cuda')
            factor=scale[j] if torch.is_tensor(scale) and scale.ndim else scale
            camera=(local*factor) @ rotations[j].T+centers[j]
            projected=camera @ torch.tensor(k,dtype=torch.float32,device='cuda').T
            uv=projected[:,:2]/projected[:,2:].clamp_min(.001)
            target=torch.tensor(pixels,dtype=torch.float32,device='cuda')
            values.append(.3*functional.smooth_l1_loss((uv-target)/10.,torch.zeros_like(uv),reduction='mean'))
        return torch.stack(values)
    renderer = ObservationRenderer(mesh)
    crops, crop_masks, crop_humans, crop_depths, roi_k, visible_counts, boxes = [], [], [], [], [], [], []
    observed_centers, depth_targets = [], []
    with av.open(str(clip.video)) as container:
        for i, decoded in enumerate(container.decode(container.streams.video[0])):
            rgb = decoded.to_ndarray(format="rgb24")
            if i in asset_observations and not np.array_equal(rgb,asset_observations[i]):
                raise ValueError("Asset observation differs from the exact original-video frame")
            name = f"{i:06d}"
            obj = read_mask(export, "object", 0, name).astype(bool)
            human = read_mask(export, "human", 0, name).astype(bool)
            depth = read_metric_depth(depth_path, "aligned", "front_stereo_camera_left", name)
            if depth.shape != obj.shape:
                depth = cv2.resize(depth, obj.shape[::-1], interpolation=cv2.INTER_NEAREST)
            visible = obj & ~human
            count = int(visible.sum())
            visible_counts.append(count)
            initial_center = base[i, :3, :3]@center_np + base[i, :3, 3]
            if count >= 32:
                xmin,ymin,xmax,ymax = observation_bbox(visible)
                uc, vc = float((xmin+xmax)/2), float((ymin+ymax)/2)
                side = max(128., float(max(xmax-xmin, ymax-ymin)*1.9+40))
                valid = visible & np.isfinite(depth) & (depth > .001)
                depth_targets.append(float(np.median(depth[valid])) if valid.any() else np.nan)
            else:
                uv = k @ initial_center
                uc, vc = float(uv[0]/uv[2]), float(uv[1]/uv[2])
                side = max(128., float(np.max(mesh.extents)*k[0,0]/max(initial_center[2], .1)*2.5))
                depth_targets.append(np.nan)
            observed_centers.append([uc, vc])
            x0, y0 = uc-side/2, vc-side/2
            factor = config.roi_size/side
            affine = np.array([[factor, 0, -x0*factor], [0, factor, -y0*factor]], dtype=np.float32)
            shape = (config.roi_size, config.roi_size)
            crops.append(cv2.warpAffine(rgb, affine, shape, flags=cv2.INTER_LINEAR))
            crop_masks.append(cv2.warpAffine(visible.astype(np.float32), affine, shape, flags=cv2.INTER_NEAREST))
            # Out-of-image pixels also have no observation; do not treat them as empty.
            support = cv2.warpAffine(np.ones_like(obj, dtype=np.uint8), affine, shape, flags=cv2.INTER_NEAREST)
            h = cv2.warpAffine(human.astype(np.uint8), affine, shape, flags=cv2.INTER_NEAREST)
            crop_humans.append((h > 0) | (support == 0))
            crop_depths.append(cv2.warpAffine(depth, affine, shape, flags=cv2.INTER_NEAREST))
            adjusted = np.asarray(k, dtype=np.float32).copy()
            adjusted[:2] *= factor
            adjusted[0,2] -= x0*factor
            adjusted[1,2] -= y0*factor
            roi_k.append(adjusted)
            boxes.append([x0, y0, side])
    require_timeline(range(len(crops)), clip.n_frames)
    anchors = anchor_indices(visible_counts)
    if not len(anchors):
        if config.tracking_backend=='observations':
            raise ValueError('No visible object evidence supports the bpy observation mainline; cannot produce a verified complete HOI candidate')
        report = {"schema":"v2hoi.object-refine.v1", "provenance":"track1", "episode":clip.episode,
            "object_instance_id":identity["object_instance_id"], "frames":clip.n_frames,
            "status":"unsupported; retain complete native estimates", "scale_shared_all_frames":1.,
            "unsupported_frame_ids":list(range(clip.n_frames)), "inputs":snapshots,
            "human_parameters_changed":False, "submission_export":False}
        (output/"result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        # The unchanged native bundle remains authoritative; no frozen replacement.
        return {"episode":clip.episode, "object_instance_id":identity["object_instance_id"],
            "frames":clip.n_frames,"scene_identity":scene_hash,
            "result":str((output/"result.json").resolve()),"status":report["status"]}
    crops, masks, humans, depths = np.asarray(crops)/255., np.asarray(crop_masks), np.asarray(crop_humans), np.asarray(crop_depths)
    roi_k = np.asarray(roi_k)
    active = np.asarray(visible_counts) >= 32
    def align_visible_centers(rotations,centers,scale,frame_id):
        """Match only the projected unoccluded support, not a full-mesh bbox."""
        centers=np.asarray(centers).copy()
        xmin,ymin,xmax,ymax=observation_bbox(masks[frame_id]>.5)
        target_mid=np.array([(xmin+xmax)/2,(ymin+ymax)/2])
        with torch.inference_mode():
            for _ in range(2):
                _,alpha,_=renderer.render(torch.tensor(rotations,dtype=torch.float32,device='cuda'),
                    torch.tensor(centers,dtype=torch.float32,device='cuda'),scale,canonical_center,
                    np.repeat(roi_k[frame_id:frame_id+1],len(rotations),axis=0),(config.roi_size,config.roi_size))
                projected=(alpha[...,0].cpu().numpy()>.5)&~humans[frame_id]
                for j,mask in enumerate(projected):
                    if mask.sum()<8:continue
                    x0,y0,x1,y1=observation_bbox(mask)
                    shift=np.clip(target_mid-[(x0+x1)/2,(y0+y1)/2],-40,40)
                    centers[j,:2]+=shift/np.array([roi_k[frame_id,0,0],roi_k[frame_id,1,1]])*centers[j,2]
        return centers
    first = min((i for i in landmarks if active[i]),default=seed_frame_index(visible_counts))
    # Joint coarse initialization handles provisional generated scales. Subsequent
    # refinement keeps ONE scale for the complete clip, never a per-frame scale.
    seed_reference=base[first] if config.tracking_backend=="foundationpose" else np.eye(4)
    if config.tracking_backend=='observations' and first in landmarks:
        points,pixels=landmarks[first]
        success,rv,tv=cv2.solvePnP(points,pixels,k,None,flags=cv2.SOLVEPNP_SQPNP)
        if success and tv[2]>0:
            seed_reference=rigid_pose(np.block([[cv2.Rodrigues(rv)[0],tv],[np.zeros((1,3)),np.ones((1,1))]]))
    seeds = orientation_hypotheses(seed_reference, center_np)
    seed_rotations = seeds[:,:3,:3]
    coarse_scales = np.geomspace(.25,4.,9) if config.refine_scale else np.array([1.])
    surface_depth = depth_targets[first]
    if not np.isfinite(surface_depth):
        available=depths[first][np.isfinite(depths[first]) & (depths[first]>.001)]
        if not len(available):raise ValueError('No estimated depth supports the observation initializer')
        surface_depth=float(np.median(available))
    seed_scores, seed_centers_all = [], []
    with torch.no_grad():
        target = torch.tensor(masks[first], device="cuda")[None, ..., None]
        care = torch.tensor(~humans[first], device="cuda")[None, ..., None]
        for coarse_scale in coarse_scales:
            seed_centers = observation_centers(mesh.vertices, seed_rotations, center_np,
                coarse_scale, surface_depth, observed_centers[first], k)
            seed_centers=align_visible_centers(seed_rotations,seed_centers,coarse_scale,first)
            rgb_pred, alpha, _ = renderer.render(torch.tensor(seed_rotations, dtype=torch.float32, device="cuda"),
                torch.tensor(seed_centers, dtype=torch.float32, device="cuda"), coarse_scale, canonical_center,
                np.repeat(roi_k[first:first+1], len(seeds), axis=0), (config.roi_size, config.roi_size))
            alpha = alpha*care
            intersection = (alpha*target).sum((1,2,3))
            union = (alpha+target-alpha*target).sum((1,2,3)).clamp_min(1)
            silhouette = 1-intersection/union
            color_loss = ((rgb_pred-torch.tensor(crops[first], dtype=torch.float32, device="cuda")).abs()*target).sum((1,2,3))/(target.sum()*3).clamp_min(1)
            seed_scores.append((silhouette+color_loss+face_cost(
                torch.tensor(seed_rotations,dtype=torch.float32,device='cuda'),
                torch.tensor(seed_centers,dtype=torch.float32,device='cuda'),np.full(len(seeds),first))+landmark_cost(
                torch.tensor(seed_rotations,dtype=torch.float32,device='cuda'),torch.tensor(seed_centers,dtype=torch.float32,device='cuda'),coarse_scale,np.full(len(seeds),first))).cpu().numpy())
            seed_centers_all.append(seed_centers)
    scores = np.asarray(seed_scores)
    # Refine all broad branches before selecting one. A quarter-turn grid alone
    # can prefer an upside-down vessel or a side face because of angular gaps.
    branch_scale_ids=scores.argmin(axis=0)
    initial_branch_scales=torch.tensor(coarse_scales[branch_scale_ids],dtype=torch.float32,device='cuda')
    branch_centers=np.stack([seed_centers_all[sid][j] for j,sid in enumerate(branch_scale_ids)])
    branch_uv=(branch_centers[:,:2]/branch_centers[:,2,None])*np.array([k[0,0],k[1,1]])+np.array([k[0,2],k[1,2]])
    br=torch.tensor(seed_rotations,dtype=torch.float32,device='cuda')
    bc=torch.tensor(branch_centers,dtype=torch.float32,device='cuda')
    buv=torch.tensor(branch_uv,dtype=torch.float32,device='cuda')
    offset_rotation=torch.nn.Parameter(torch.zeros(len(seeds),3,device='cuda'))
    offset_uv=torch.nn.Parameter(torch.zeros(len(seeds),2,device='cuda'))
    offset_z=torch.nn.Parameter(torch.zeros(len(seeds),device='cuda'))
    offset_scale=torch.nn.Parameter(torch.zeros(len(seeds),device='cuda'))
    seed_depth=torch.tensor(np.nan_to_num(depths[first]),dtype=torch.float32,device='cuda')[None,...,None]
    seed_optimizer=torch.optim.Adam([offset_rotation,offset_uv,offset_z]+([offset_scale] if config.refine_scale else []),lr=.04)
    def branch_state():
        rs=so3_exp_map(torch.tanh(offset_rotation)*1.) @ br
        uv=buv+torch.tanh(offset_uv)*40.
        z=bc[:,2]*torch.exp(torch.tanh(offset_z)*.2)
        cs=torch.cat([(uv-torch.tensor([k[0,2],k[1,2]],device='cuda'))/
            torch.tensor([k[0,0],k[1,1]],device='cuda')*z[:,None],z[:,None]],dim=1).float()
        ss=initial_branch_scales*torch.exp(torch.tanh(offset_scale)*.3) if config.refine_scale else initial_branch_scales
        return rs,cs,ss
    for _ in range(40):
        seed_optimizer.zero_grad(set_to_none=True)
        rs,cs,ss=branch_state()
        rgb_pred,alpha,z=renderer.render(rs,cs,ss[:,None,None],canonical_center,
            np.repeat(roi_k[first:first+1],len(seeds),axis=0),(config.roi_size,config.roi_size))
        alpha=alpha*care
        intersection=(alpha*target).sum((1,2,3))
        union=(alpha+target-alpha*target).sum((1,2,3)).clamp_min(1)
        photometric=((rgb_pred-torch.tensor(crops[first],dtype=torch.float32,device='cuda')).abs()*target).sum((1,2,3))/(target.sum()*3).clamp_min(1)
        valid=target*(seed_depth>.001).float()*(z.detach()>.001).float()
        depth_residual=functional.smooth_l1_loss(z,seed_depth.expand_as(z),beta=.08,reduction='none')
        depth_cost=(depth_residual*valid).sum((1,2,3))/valid.sum((1,2,3)).clamp_min(1)
        branch_losses=1-intersection/union+photometric+.6*depth_cost+face_cost(rs,cs,np.full(len(seeds),first))+landmark_cost(rs,cs,ss,np.full(len(seeds),first))
        branch_losses.sum().backward();seed_optimizer.step()
    with torch.inference_mode():
        rs,cs,ss=branch_state()
        # Recompute the selection score at the final state, after the last update.
        rgb_pred,alpha,z=renderer.render(rs,cs,ss[:,None,None],canonical_center,
            np.repeat(roi_k[first:first+1],len(seeds),axis=0),(config.roi_size,config.roi_size))
        alpha=alpha*care;intersection=(alpha*target).sum((1,2,3))
        union=(alpha+target-alpha*target).sum((1,2,3)).clamp_min(1)
        valid=target*(seed_depth>.001).float()*(z>.001).float()
        depth_cost=(functional.smooth_l1_loss(z,seed_depth.expand_as(z),beta=.08,reduction='none')*valid).sum((1,2,3))/valid.sum((1,2,3)).clamp_min(1)
        branch_losses=1-intersection/union+((rgb_pred-torch.tensor(crops[first],dtype=torch.float32,device='cuda')).abs()*target).sum((1,2,3))/(target.sum()*3).clamp_min(1)+.6*depth_cost+face_cost(rs,cs,np.full(len(seeds),first))
        branch_losses+=landmark_cost(rs,cs,ss,np.full(len(seeds),first))
        seed_id=int(branch_losses.argmin());seed_scale=float(ss[seed_id])
        seed=camera_centers_to_poses(rs[seed_id:seed_id+1].cpu().numpy(),
            cs[seed_id:seed_id+1].cpu().numpy(),center_np,seed_scale)[0]
    tracked=np.repeat(seed[None],clip.n_frames,axis=0)
    if config.tracking_backend=="foundationpose":
        from prep.run_foundationpose_mhr_export import _pose_estimator
        tracking_mesh = mesh.copy()
        tracking_mesh.vertices *= seed_scale
        Utils.set_seed(0)
        estimator = _pose_estimator(tracking_mesh, output/"tracking-debug")
        transform = estimator.get_tf_to_centered_mesh().cpu().numpy()
        tracked = base.copy()
        tracked[first] = seed
        decoded_rgb = []
        with av.open(str(clip.video)) as container:
            for i, decoded in enumerate(container.decode(container.streams.video[0])):
                if i < first:
                    decoded_rgb.append(decoded.to_ndarray(format="rgb24"))
        for direction, frame_ids in (("forward",range(first+1,clip.n_frames)),("backward",range(first-1,-1,-1))):
            estimator.pose_last = torch.tensor(seed@np.linalg.inv(transform),dtype=torch.float32,device="cuda")
            # Re-decode forward frames; only the prefix is held for reverse tracking.
            forward_container = av.open(str(clip.video)) if direction=="forward" else None
            forward_frames = iter(forward_container.decode(forward_container.streams.video[0])) if forward_container else None
            if forward_frames:
                for _ in range(first+1): next(forward_frames)
            for i in frame_ids:
                rgb = next(forward_frames).to_ndarray(format="rgb24") if forward_frames else decoded_rgb[i]
                depth = read_metric_depth(depth_path, "aligned", "front_stereo_camera_left", f"{i:06d}")
                if depth.shape != rgb.shape[:2]:
                    depth = cv2.resize(depth, rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST)
                depth = np.nan_to_num(depth.astype(np.float32))
                depth[(depth < .001) | (depth > 8.)] = 0
                with torch.inference_mode():
                    tracked[i] = rigid_pose(estimator.track_one(rgb=rgb, depth=depth, K=k, iteration=5, extra={}))
                if i % 60 == 0:
                    print(f"OBJECT_REFINE_TRACK episode={clip.episode} direction={direction} frame={i+1}/{clip.n_frames}", flush=True)
            if forward_container: forward_container.close()
        del decoded_rgb
        del estimator
        torch.cuda.empty_cache()
    static_rotation_evidence=image_rotation_evidence(crops,masks,humans)
    anchor_pairs=rotation_anchor_pairs(crops,masks,humans) if asset.get('appearance_rotation_observable',False) else np.empty((0,3))
    reinitialized=[]
    if asset is not None or config.tracking_backend=="observations":
        # An independently authored geometry has different axes/appearance from
        # the native asset. Re-estimate every supported view instead of trusting
        # a tracker trained against the discarded shape. Scale stays shared.
        candidate_frames,rotation_states,center_states,cost_states=[],[],[],[]
        with torch.inference_mode():
            for i in np.flatnonzero(active):
                alternatives=orientation_hypotheses(tracked[i],center_np*seed_scale)
                candidate_rotations=alternatives[:,:3,:3]
                surface=depth_targets[i]
                if not np.isfinite(surface): continue
                candidate_centers=observation_centers(mesh.vertices,candidate_rotations,center_np,
                    seed_scale,surface,observed_centers[i],k)
                candidate_centers=align_visible_centers(candidate_rotations,candidate_centers,seed_scale,i)
                c,a,z=renderer.render(torch.tensor(candidate_rotations,dtype=torch.float32,device='cuda'),
                    torch.tensor(candidate_centers,dtype=torch.float32,device='cuda'),seed_scale,canonical_center,
                    np.repeat(roi_k[i:i+1],len(alternatives),axis=0),(config.roi_size,config.roi_size))
                target_i=torch.tensor(masks[i],dtype=torch.float32,device='cuda')[None,...,None]
                care_i=torch.tensor(~humans[i],device='cuda')[None,...,None];a=a*care_i
                intersection=(a*target_i).sum((1,2,3));union=(a+target_i-a*target_i).sum((1,2,3)).clamp_min(1)
                photo=((c-torch.tensor(crops[i],dtype=torch.float32,device='cuda')).abs()*target_i).sum((1,2,3))/(target_i.sum()*3).clamp_min(1)
                # A small angular preference stabilizes equivalent silhouettes;
                # image observations can override it for a real face change.
                angles=np.arccos(np.clip((np.trace(candidate_rotations@tracked[i,:3,:3].T,axis1=1,axis2=2)-1)/2,-1,1))
                costs=1-intersection/union+photo+torch.tensor(angles*.015,device='cuda')+face_cost(
                    torch.tensor(candidate_rotations,dtype=torch.float32,device='cuda'),
                    torch.tensor(candidate_centers,dtype=torch.float32,device='cuda'),np.full(len(alternatives),i))+landmark_cost(
                    torch.tensor(candidate_rotations,dtype=torch.float32,device='cuda'),torch.tensor(candidate_centers,dtype=torch.float32,device='cuda'),seed_scale,np.full(len(alternatives),i))
                candidate_frames.append(int(i));rotation_states.append(candidate_rotations)
                center_states.append(candidate_centers);cost_states.append(costs.cpu().numpy())
                if i%60==0: print(f'OBJECT_REFINE_ASSET_REGISTER episode={clip.episode} frame={i+1}/{clip.n_frames}',flush=True)
        transition_strength=np.array([.02+.2*static_rotation_evidence[b-1] if b==a+1 else .02 for a,b in zip(candidate_frames[:-1],candidate_frames[1:])])
        choices=select_orientation_path(rotation_states,cost_states,candidate_frames,angular_step_deg=300./clip.fps,weight=transition_strength)
        for i,rs,cs,selected in zip(candidate_frames,rotation_states,center_states,choices):
            tracked[i]=camera_centers_to_poses(rs[selected:selected+1],cs[selected:selected+1],center_np,seed_scale)[0]
            if selected: reinitialized.append(int(i))
    initial_rotations = torch.tensor(tracked[:, :3, :3], dtype=torch.float32, device="cuda")
    initial_centers = tracked[:, :3, :3]@(center_np*seed_scale) + tracked[:, :3, 3]
    corrected_centers = initial_centers.copy()
    for i in np.flatnonzero(active):
        if np.isfinite(depth_targets[i]):
            centers=observation_centers(mesh.vertices,tracked[i:i+1,:3,:3],center_np,
                seed_scale,depth_targets[i],observed_centers[i],k)
            corrected_centers[i]=align_visible_centers(tracked[i:i+1,:3,:3],centers,seed_scale,i)[0]
    center_uv=corrected_centers[:,:2]/corrected_centers[:,2,None]*np.array([k[0,0],k[1,1]])+np.array([k[0,2],k[1,2]])
    z0 = torch.tensor(corrected_centers[:,2], dtype=torch.float32, device="cuda")
    uv0 = torch.tensor(center_uv, dtype=torch.float32, device="cuda")
    delta_rotation = torch.nn.Parameter(torch.zeros(clip.n_frames, 3, device="cuda"))
    delta_uv = torch.nn.Parameter(torch.zeros(clip.n_frames, 2, device="cuda"))
    delta_z = torch.nn.Parameter(torch.zeros(clip.n_frames, device="cuda"))
    scale_log = torch.nn.Parameter(torch.zeros((), device="cuda"))
    optimizer = torch.optim.Adam([delta_rotation, delta_uv, delta_z]+([scale_log] if config.refine_scale else []), lr=config.learning_rate)
    data_rgb = torch.tensor(crops, dtype=torch.float32, device="cuda")
    data_mask = torch.tensor(masks, dtype=torch.float32, device="cuda")[...,None]
    data_care = torch.tensor(~humans, dtype=torch.float32, device="cuda")[...,None]
    data_depth = torch.tensor(np.nan_to_num(depths), dtype=torch.float32, device="cuda")[...,None]
    indices = np.flatnonzero(active)
    focal = torch.tensor([k[0,0], k[1,1]], dtype=torch.float32, device="cuda")
    principal = torch.tensor([k[0,2], k[1,2]], dtype=torch.float32, device="cuda")
    history = []

    def state(ids):
        rotation = so3_exp_map(torch.tanh(delta_rotation[ids])*np.deg2rad(config.rotation_limit_deg)) @ initial_rotations[ids]
        uv = uv0[ids]+torch.tanh(delta_uv[ids])*24.
        z = z0[ids]*torch.exp(torch.tanh(delta_z[ids])*.25)
        centers = torch.cat([(uv-principal)/focal*z[:,None], z[:,None]], dim=1)
        scale = seed_scale*(scale_log.exp().clamp(config.scale_min, config.scale_max) if config.refine_scale else scale_log.detach().exp())
        return rotation, centers, scale

    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for start in range(0, len(indices), config.batch_size):
            ids = indices[start:start+config.batch_size]
            rotations, centers, scale = state(ids)
            color, alpha, z = renderer.render(rotations, centers, scale, canonical_center, roi_k[ids],
                                              (config.roi_size, config.roi_size))
            target, care = data_mask[ids], data_care[ids]
            visible = alpha*care
            intersection = (visible*target).sum((1,2,3))
            union = (visible+target-visible*target).sum((1,2,3)).clamp_min(1)
            mask_loss = (1-intersection/union).mean()
            photo = functional.smooth_l1_loss(color, data_rgb[ids], beta=.08, reduction="none")
            color_loss = (photo*target).sum()/(target.sum()*3).clamp_min(1)
            valid_depth = target*(data_depth[ids] > .001).float()*(z.detach() > .001).float()
            residual = functional.smooth_l1_loss(z, data_depth[ids], beta=.08, reduction="none")
            depth_loss = (residual*valid_depth).sum()/valid_depth.sum().clamp_min(1)
            pose_prior = .002*delta_rotation[ids].square().mean()+.001*delta_uv[ids].square().mean()
            loss = mask_loss+.8*color_loss+.6*depth_loss+pose_prior+face_cost(rotations,centers,ids).mean()+landmark_cost(rotations,centers,scale,ids).mean()
            weighted = loss*(len(ids)/len(indices))
            weighted.backward()
            total += float(weighted.detach())
        # Regularize the correction, rather than flattening the observed motion.
        contiguous = active[1:] & active[:-1]
        pair = torch.tensor(contiguous, device="cuda")
        if pair.any():
            correction_loss = .02*(delta_uv[1:]-delta_uv[:-1])[pair].square().mean()
            correction_loss += .01*(delta_rotation[1:]-delta_rotation[:-1])[pair].square().mean()
            correction_loss.backward()
        triplet=torch.tensor(active[2:] & active[1:-1] & active[:-2],device="cuda")
        temporal_value=0.
        if config.temporal_strength and triplet.any():
            all_rotations, all_centers, _ = state(np.arange(clip.n_frames))
            relative=all_rotations[:-1].transpose(1,2) @ all_rotations[1:]
            # Constant angular velocity has zero relative-rotation acceleration.
            angular=(relative[1:]-relative[:-1])[triplet]/.1
            stationary=torch.tensor(static_rotation_evidence,dtype=torch.float32,device='cuda')
            angular_velocity=(relative-torch.eye(3,device='cuda'))/.04
            stable_loss=functional.smooth_l1_loss(angular_velocity,torch.zeros_like(angular_velocity),reduction='none').mean((1,2))
            uv=uv0+torch.tanh(delta_uv)*24.
            image_acc=(uv[2:]-2*uv[1:-1]+uv[:-2])[triplet]/2.
            depth_acc=(all_centers[2:,2]-2*all_centers[1:-1,2]+all_centers[:-2,2])[triplet]/.01
            temporal_loss=config.temporal_strength*(
                .15*functional.smooth_l1_loss(angular,torch.zeros_like(angular),reduction="mean")
                +.1*functional.smooth_l1_loss(image_acc,torch.zeros_like(image_acc),reduction="mean")
                +.1*functional.smooth_l1_loss(depth_acc,torch.zeros_like(depth_acc),reduction="mean")
                +.2*(stable_loss*stationary).mean())
            if len(anchor_pairs):
                pairs=torch.tensor(anchor_pairs[:,:2],dtype=torch.long,device='cuda')
                weights=torch.tensor(anchor_pairs[:,2],dtype=torch.float32,device='cuda')
                difference=(all_rotations[pairs[:,0]]-all_rotations[pairs[:,1]])/.04
                cost=functional.smooth_l1_loss(difference,torch.zeros_like(difference),reduction='none').mean((1,2))
                temporal_loss+=config.temporal_strength*.3*(cost*weights).mean()
            temporal_value=float(temporal_loss.detach())
            temporal_loss.backward()
        parameters=[delta_rotation,delta_uv,delta_z]+([scale_log] if config.refine_scale else [])
        if any(value.grad is None or not torch.isfinite(value.grad).all() for value in parameters):
            raise ValueError("Nonfinite or absent observation-refinement gradient")
        optimizer.step()
        with torch.no_grad():
            scale_log.clamp_(np.log(config.scale_min), np.log(config.scale_max))
        if step % 10 == 0 or step == config.steps-1:
            history.append({"step":step+1, "fitted_loss":total, "temporal_loss":temporal_value,
                "shared_scale":seed_scale*float(scale_log.exp().detach())})
            print("OBJECT_REFINE_OPT "+json.dumps({"episode":clip.episode, **history[-1]}), flush=True)
    with torch.inference_mode():
        rotations, centers, scale = state(np.arange(clip.n_frames))
        scale_value = float(scale)
        poses = camera_centers_to_poses(rotations.cpu().numpy(), centers.cpu().numpy(), center_np, scale_value)
    # Unsupported observations remain estimates, never a claim of frozen truth.
    # Only the explicit legacy comparison retains upstream object states.
    if (~active).any():
        if config.tracking_backend=='foundationpose':
            native_centers = base[~active,:3,:3]@center_np+base[~active,:3,3]
            poses[~active] = camera_centers_to_poses(base[~active, :3,:3], native_centers, center_np, scale_value)
        else:
            from scipy.spatial.transform import Rotation,Slerp
            supported=np.flatnonzero(active);missing=np.flatnonzero(~active)
            rotation_values=rotations.cpu().numpy();center_values=centers.cpu().numpy()
            if len(supported)>1:
                rotation_values[missing]=Slerp(supported,Rotation.from_matrix(rotation_values[supported]))(np.clip(missing,supported[0],supported[-1])).as_matrix()
            for axis in range(3):center_values[missing,axis]=np.interp(missing,supported,center_values[supported,axis])
            poses=camera_centers_to_poses(rotation_values,center_values,center_np,scale_value)
    new_mesh = mesh.copy()
    new_mesh.vertices *= scale_value
    candidate_mesh = output/"object.glb"
    new_mesh.export(candidate_mesh)
    readback = trimesh.load(candidate_mesh, force="mesh", process=False)
    if not np.allclose(readback.vertices, mesh.vertices*scale_value, atol=1e-6) or not np.array_equal(readback.faces, mesh.faces):
        raise ValueError("Candidate geometry changed beyond the single shared scale")
    if not np.array_equal(readback.visual.vertex_colors,mesh.visual.vertex_colors):
        raise ValueError("Candidate serialization changed the frozen appearance")
    candidate = copy.deepcopy(bundle)
    candidate["source_postopt"]=candidate.pop("postopt",{})
    candidate["postopt"]={"mode":"observation_refinement","source_contact_statistics_invalidated":True,
        "scope":"Object-only image fitting; human and camera frozen; contact metrics must be recomputed"}
    for key in list(candidate["pr"]):
        if key.startswith("pose_abs") and key not in {"pose_abs","pose_abs_postopt"}:
            del candidate["pr"][key]
    candidate["pr"]["pose_abs"] = poses.astype(np.float32)
    candidate["pr"]["pose_abs_postopt"] = poses.astype(np.float32)
    candidate["metadata"].update(object_mesh=str(candidate_mesh.resolve()),
        object_mesh_to_training_transform=np.eye(4).tolist(), object_pose_storage_to_training_transform=np.eye(4).tolist(),
        object_mesh_to_pose_transform=np.eye(4).tolist(),
        object_refinement="observation-render-v1", object_scale_baked=scale_value,
        source_states_mesh=str(mesh_path.resolve()),source_states_scope="in/pr_initial/raw are historical source states",
        source_bundle_sha256=sha256(bundle_path))
    torch.save(candidate, output/"refined.pth")
    np.savez(output/"tracks.npz", poses_canonical_mesh_to_camera=poses, original_frame_ids=np.arange(clip.n_frames),
        initial_poses=base, tracked_poses=tracked, tracking_mesh_scale=np.array(seed_scale),
        shared_scale=np.array(scale_value), observation_supported=active)
    preview, metrics = render_preview(clip, export, renderer, base, poses, k, canonical_center, scale_value, output, config,
        before_renderer=ObservationRenderer(baseline_mesh), before_center=baseline_mesh.bounds.mean(0))
    report = {"schema":"v2hoi.object-refine.v1", "provenance":"track1", "episode":clip.episode,
        "object":clip.object_name, "object_instance_id":identity["object_instance_id"],
        "frames":clip.n_frames, "config":asdict(config), "seed_frame":first,
        "seed_candidate":int(seed_id), "seed_scores":scores.tolist(), "seed_scale":seed_scale,
        "refined_seed_scores":branch_losses.cpu().numpy().tolist(),"seed_refinement_steps":40,
        "coarse_scale_candidates":coarse_scales.tolist(), "scale_shared_all_frames":scale_value,
        "scale_policy":"Single episode-local scale baked exactly once into the canonical mesh; origin preserved",
        "anchor_frame_ids":anchors.tolist(), "unsupported_frame_ids":np.flatnonzero(~active).tolist(),
        "inputs":snapshots, "video_sha256":sha256(clip.video), "runtime_sha256":runtime_hash,
        "human_parameters_changed":False, "training":False, "submission_export":False,
        "source_contact_statistics_invalidated":True,"native_submission_format_verified":False,
        "tracking_backend":config.tracking_backend,
        "foundationpose_calls_in_this_stage":0 if config.tracking_backend=='observations' else clip.n_frames-1,
        "source_human_lineage":"Existing native baseline; its earlier upstream stages included FoundationPose",
        "static_rotation_evidence_pair_ids":np.flatnonzero(static_rotation_evidence>.5).tolist(),
        "long_range_rotation_evidence_pairs":anchor_pairs.tolist(),
        "appearance_rotation_observable":asset.get('appearance_rotation_observable',False),
        "landmark_observation_frame_ids":sorted(landmarks),
        "asset_backend":asset['entry'] if asset else "native_generated_mesh",
        "asset_replacement":asset is not None,
        "asset_orientation_reinitialized_frames":reinitialized,
        "semantic_face_observations":asset.get('face_observations',[]) if asset else [],
        "orientation_path_policy":"Global image-cost and geodesic continuity selection for replacement assets",
        "roi_bbox_policy":"Ignore disconnected islands below max(8 pixels, 1% of largest); raw fitting masks retained",
        "scene_identity":scene_hash,
        "outputs":{p.name:sha256(p) for p in (candidate_mesh, output/"refined.pth", output/"tracks.npz", preview)},
        "fitted_history":history, "image_diagnostics":metrics,
        "limitations":["Experimental rigid-object stage; estimated depth and masks are not ground truth",
            "Image fitting is measured on fitting observations; no official score or unseen-video acceptance",
            "Human pose/hand contact and camera are frozen; contact refinement is still pending",
            "Unobserved motion remains an explicit estimate; scale-depth ambiguity remains monocular"]}
    (output/"result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if config.full_hoi_preview:
        from v2hoi.hoi_preview import render_full_hoi
        full_scene=render_full_hoi(run,clip,output/'refined.pth',output,codec=config.preview_codec)
        report['full_hoi_preview']=full_scene
        report['outputs'].update(full_scene['outputs'])
        (output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return {"episode":clip.episode, "object_instance_id":identity["object_instance_id"],
            "result":str((output/"result.json").resolve()), "frames":clip.n_frames,
            "shared_scale":scale_value,"scene_identity":scene_hash,"status":"candidate; pending visual acceptance"}


def render_preview(clip, export, renderer, before, after, k, canonical_center, scale, output, config,
        before_renderer=None,before_center=None,before_scale=1.,before_title="Native input"):
    import av
    import cv2
    import torch
    from prep.mhr_export_utils import read_mask
    width_out, height_out = 768, 288
    preview = output/"comparison.mp4"
    ffmpeg = shutil.which("ffmpeg") or os.environ.get("IMAGEIO_FFMPEG_EXE")
    if not ffmpeg: raise FileNotFoundError("FFmpeg is required for the derived preview")
    codec=config.preview_codec
    if codec=="auto":
        probe=subprocess.run([ffmpeg,"-hide_banner","-loglevel","error","-f","lavfi","-i","color=size=32x32:rate=1",
            "-frames:v","1","-c:v","h264_nvenc","-f","null","-"],capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        codec="h264_nvenc" if probe.returncode==0 else "libx264"
    options=["-preset","p4","-cq","18","-b:v","0"] if codec=="h264_nvenc" else ["-preset","fast","-crf","18"]
    log = (output/"preview.log").open("w")
    encoder = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width_out}x{height_out}", "-r", str(clip.fps), "-i", "pipe:0", "-an", "-c:v", codec,
        *options,"-pix_fmt", "yuv420p", "-movflags", "+faststart", str(preview)],
        stdin=subprocess.PIPE, stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    diagnostics = []
    before_renderer=before_renderer or renderer
    before_center=np.asarray(before_center) if before_center is not None else canonical_center.cpu().numpy()
    try:
        with av.open(str(clip.video)) as container:
            for i, frame in enumerate(container.decode(container.streams.video[0])):
                rgb = frame.to_ndarray(format="rgb24")
                h, w = rgb.shape[:2]
                target = read_mask(export, "object", 0, f"{i:06d}").astype(bool)
                human = read_mask(export, "human", 0, f"{i:06d}").astype(bool)
                centers = np.stack([before[i,:3,:3]@(before_center*before_scale)+before[i,:3,3],
                                    after[i,:3,:3]@(canonical_center.cpu().numpy()*scale)+after[i,:3,3]])
                colors, alphas = [], []
                with torch.inference_mode():
                    for pose, center, s, engine,origin in zip((before[i],after[i]), centers, (before_scale,scale),
                            (before_renderer,renderer),(before_center,canonical_center.cpu().numpy())):
                        c,a,_ = engine.render(torch.tensor(pose[None,:3,:3],dtype=torch.float32,device="cuda"),
                            torch.tensor(center[None],dtype=torch.float32,device="cuda"),s,
                            torch.tensor(origin,dtype=torch.float32,device="cuda"),k[None],(h,w))
                        colors.append((c[0].cpu().numpy()*255).round().clip(0,255).astype(np.uint8))
                        alphas.append(a[0,...,0].cpu().numpy()>.5)
                visible = target & ~human
                if visible.any():
                    xmin,ymin,xmax,ymax=observation_bbox(visible)
                    uc,vc=int((xmin+xmax)/2),int((ymin+ymax)/2)
                    half=max(80,int(max(xmax-xmin,ymax-ymin)*.8+30))
                    box=(max(0,uc-half),max(0,vc-half),min(w,uc+half),min(h,vc+half))
                else:
                    box=(0,0,w,h)
                x0,y0,x1,y1=box
                views=[rgb]
                values=[]
                for color,alpha in zip(colors,alphas):
                    mask=alpha & ~human
                    values.append(float((mask & visible).sum()/max(int((mask | visible).sum()),1)))
                    view=rgb.copy(); view[mask]=color[mask]; views.append(view)
                panels=[]
                for title,view in zip((f"Original {i}",before_title,"Automatic pipeline"),views):
                    panel=np.full((288,256,3),245,dtype=np.uint8)
                    cv2.putText(panel,title,(6,23),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
                    panel[32:]=cv2.resize(view[y0:y1,x0:x1],(256,256),interpolation=cv2.INTER_NEAREST)
                    panels.append(panel)
                image=np.hstack(panels)
                encoder.stdin.write(image.tobytes())
                if i % max(1,clip.n_frames//12)==0 or i==clip.n_frames-1:
                    cv2.imwrite(str(output/f"frame-{i:06d}.png"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                diagnostics.append(values)
    finally:
        encoder.stdin.close(); encoder.wait(); log.close()
    if encoder.returncode or len(diagnostics)!=clip.n_frames:
        raise ValueError("Incomplete pipeline preview")
    capture=cv2.VideoCapture(str(preview)); count=0
    while capture.read()[0]: count+=1
    capture.release()
    if count!=clip.n_frames: raise ValueError("Preview changed frame coverage")
    values=np.asarray(diagnostics)
    np.save(output/"visible-iou.npy",values)
    boundaries=np.linspace(0,clip.n_frames,4).round().astype(int)
    return preview,{"scope":"Full-image visible IoU on fitted masks; not reconstruction accuracy",
        "preview_codec":codec,"segments":[{"start":int(a),"stop":int(b),"mean_native_candidate":values[a:b].mean(axis=0).tolist()}
                    for a,b in zip(boundaries[:-1],boundaries[1:])],"preview_decoded_frames":count}
