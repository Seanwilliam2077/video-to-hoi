"""Re-estimate one EP16 static mesh and all-frame initialization in current K."""
from pathlib import Path
import argparse
import json
import time
import cv2
import numpy as np
from fit_object_hypotheses import (CONFIG, VIDEO, MASKS, K, digest, surface,
    FrameEvidence, fit_frame, export_obj, object_rotation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve all attempts: output must not exist')
    assert CONFIG['track'] == 1 and CONFIG['episode'] == 16
    assert digest(VIDEO) == CONFIG['video_sha256']
    assert digest(CONFIG['camera_intrinsics']) == CONFIG['camera_intrinsics_sha256']
    assert digest(CONFIG['human_joints']) == CONFIG['human_joints_sha256']
    for row in CONFIG['mask_files']:
        assert digest(row['path']) == row['sha256']
    start = time.perf_counter()
    args.output.mkdir(parents=True)
    depth = CONFIG['depth_anchor_m']
    low_vertices, _, _ = surface(16., 16., nlat=16, nlon=48)
    evidence0 = FrameEvidence(0)
    x, y, width, height = evidence0.box
    initial = np.array([x+width/2, y+height/2,
                        np.log(width*depth/K[0,0]/2.6), .6, .22, 0.])
    alternatives = []
    for yaw in (.25, .7, 1.15):
        guess = initial.copy(); guess[3] = yaw
        fit = fit_frame(evidence0, low_vertices, depth, guess, max_nfev=160)
        score = evidence0.evaluate(fit.x, low_vertices, depth)
        alternatives.append((score['visible_mask_iou'], fit.x, score))
    iou, first, score = max(alternatives, key=lambda row:row[0])
    print(json.dumps({'stage':'current_K_frame0','visible_mask_iou':iou,
                      'half_extent_m':float(np.exp(first[2])),
                      'depth_anchor_m':depth,'parameters':first.tolist()}),flush=True)
    vertices, faces, uv = surface(16.,16.,nlat=24,nlon=64)
    poses=[]; diagnostics=[]; previous=first.copy()
    for frame in range(360):
        evidence=FrameEvidence(frame)
        guess=previous.copy()
        if frame:guess[:2]+=evidence.center-previous_evidence.center
        fit=fit_frame(evidence,vertices,depth,guess,max_nfev=60 if frame else 120,
                      previous=previous if frame else None)
        previous=fit.x; previous_evidence=evidence; poses.append(fit.x)
        score=evidence.evaluate(fit.x,vertices,depth)
        diagnostics.append({'frame':frame,'visible_mask_iou':score['visible_mask_iou'],
            'visible_contour_mean_px':score['visible_contour_mean_px'],
            'nfev':fit.nfev,'solver_success':bool(fit.success)})
        if (frame+1)%30==0:print(f'Current-camera silhouette initializer {frame+1}/360',flush=True)
    params=np.asarray(poses);half=float(np.exp(params[0,2]));metric_vertices=vertices*half
    transforms=np.repeat(np.eye(4)[None],360,axis=0)
    for frame,p in enumerate(params):
        z=depth*half/np.exp(p[2])
        transforms[frame,:3,:3]=object_rotation(p)
        transforms[frame,:3,3]=[(p[0]-K[0,2])*z/K[0,0],(p[1]-K[1,2])*z/K[1,1],z]
    export_obj(metric_vertices,faces,uv,args.output/'object.obj')
    np.savez_compressed(args.output/'object_candidate.npz',frame_indices=np.arange(360),
        object_to_camera=transforms,vertices=metric_vertices,faces=faces,uv=uv,
        camera_intrinsics=K,fit_parameters=params)
    report={'track':1,'episode':16,'source_video':str(VIDEO),
        'source_video_sha256':CONFIG['video_sha256'],
        'source':'Verified EP16 RGB, own SAM2 masks and original fixed native-MHR wrist depth gauge',
        'selected_hypothesis':'rounded_cube_n16',
        'hypotheses':[{'name':'rounded_cube_n16','nxz':16.,'ny':16.,
            'frame0_params':params[0].tolist(),'frame0_iou':diagnostics[0]['visible_mask_iou']}],
        'depth_anchor_m':depth,'metric_half_extent_m':half,
        'scale_assumption':CONFIG['depth_gauge_policy'],
        'mesh_assumption':'Own RGB-derived n16 symmetric profile; hidden shape and physical scale unverified. Analytic trig zeros corrected before geometry is generated.',
        'coordinate_convention':'OpenCV right/down/forward metres; fixed current human K; one baked mesh scale, no per-frame mesh scale.',
        'initialization_note':'Per-frame silhouette log-size is converted into rigid depth before export. It is an initializer, not final contact pose.',
        'diagnostics':diagnostics,'wall_seconds':time.perf_counter()-start,
        'mean_visible_mask_iou':float(np.mean([d['visible_mask_iou'] for d in diagnostics])),
        'script_sha256':digest(Path(__file__)),
        'configuration_sha256':digest(Path(__file__).parent/'object-config01.json'),
        'quality_accepted':False,'official_score_available':False,
        'old_camera_or_3D_reference_points_reused':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('diagnostics','hypotheses')}),flush=True)


if __name__=='__main__':main()
