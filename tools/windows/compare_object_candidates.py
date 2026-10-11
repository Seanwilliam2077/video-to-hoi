"""Compare two verified full-timeline native object candidates, without export."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from v2hoi.native import native_environment,sha256
from v2hoi.object_refine import candidate_paths,plan_jobs,RefineConfig


def motion_diagnostics(poses,center):
    rotations=poses[:,:3,:3]
    positions=rotations@center+poses[:,:3,3]
    velocities=Rotation.from_matrix(rotations[:-1].transpose(0,2,1)@rotations[1:]).as_rotvec()
    angular=np.linalg.norm(velocities[1:]-velocities[:-1],axis=1)*180/np.pi
    translation=np.linalg.norm(positions[2:]-2*positions[1:-1]+positions[:-2],axis=1)*1000
    return {'scope':'Self acceleration per frame squared; not reference-relative competition ACC',
        'angular_deg_p50_p95_max':np.quantile(angular,[.5,.95,1]).tolist(),
        'center_mm_p50_p95_max':np.quantile(translation,[.5,.95,1]).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
    _,clip,identity=plan_jobs([a.run],a.dataset_root)[0]
    paths=[candidate_paths(a.run,value) for value in (a.before,a.after)]
    if not a.worker:
        weights=Path(identity['command'][identity['command'].index('--weights_path')+1])
        env=native_environment(Path('third_party/CARI4D_MHR_Win').resolve(),weights)
        subprocess.run([sys.executable,'-m','v2hoi.cuda_budget','--fraction','.75','--report',str(a.output.parent/(a.output.name+'-memory.json')),
            '--',str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],env=env,check=True);return
    import torch,trimesh
    from v2hoi.object_refine_runtime import ObservationRenderer,render_preview
    from prep.mhr_export_utils import camera_calibration,load_edex
    a.output.mkdir(parents=True,exist_ok=False)
    bundles=[torch.load(bundle,map_location='cpu',weights_only=False) for bundle,_ in paths]
    for name,value in bundles[0]['pr'].items():
        if name.startswith('mhr_') and not np.array_equal(np.asarray(value),np.asarray(bundles[1]['pr'][name])):
            raise ValueError('Cannot compare candidates with different frozen humans')
    meshes=[trimesh.load(mesh,force='mesh',process=False) for _,mesh in paths]
    poses=[np.asarray(bundle['pr']['pose_abs']) for bundle in bundles]
    export=a.run/'native'/f'episode_{clip.episode:06d}'/'export'/f'episode_{clip.episode:06d}'
    k,_=camera_calibration(load_edex(export),0)
    preview,image_metrics=render_preview(clip,export,ObservationRenderer(meshes[1]),poses[0],poses[1],k,
        torch.tensor(meshes[1].bounds.mean(0),dtype=torch.float32,device='cuda'),1.,a.output,RefineConfig(),
        before_renderer=ObservationRenderer(meshes[0]),before_center=meshes[0].bounds.mean(0),before_title='Previous candidate')
    report={'provenance':'track1','episode':clip.episode,'frames':clip.n_frames,
        'inputs':{str(value.resolve()):sha256(value) for pair in paths for value in pair},
        'motion_before_after':[motion_diagnostics(p,m.bounds.mean(0)) for p,m in zip(poses,meshes)],
        'image_diagnostics':image_metrics,'preview_sha256':sha256(preview),'submission_export':False}
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
