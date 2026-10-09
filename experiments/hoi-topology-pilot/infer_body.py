"""120-frame native MHR initializer, using only the approved Track 1 clip."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('MOMENTUM_ENABLED', '0')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for p in ['sources/sam-3d-body', 'sources/moge', 'sources/utils3d']:
    sys.path.insert(0, str(ROOT / p))
import cv2
import numpy as np
import torch
import json
import time

def main():
    out = ROOT / 'baseline'; out.mkdir(exist_ok=True)
    frames = ROOT / 'data' / 'frames'; frames.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(ROOT / 'data' / 'episode_000016.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert abs(fps - 30) < .01
    images = []
    for i in range(210):
        ok, image = cap.read()
        assert ok, i
        if i >= 90:
            images.append(image)
            cv2.imwrite(str(frames / f'{i:06d}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 98])
    cap.release()
    from moge.model.v2 import MoGeModel
    moge = MoGeModel.from_pretrained(str(ROOT / 'assets/moge2/model.pt')).cuda().eval()
    candidates = []
    for i in [0, 60, 119]:
        img = cv2.resize(images[i], (768, 576))
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).cuda().float().permute(2, 0, 1) / 255
        with torch.no_grad():
            result = moge.infer(x)
        k = result['intrinsics'].cpu().numpy()
        k[0] *= images[i].shape[1]; k[1] *= images[i].shape[0]
        candidates.append(k)
    camera = np.median(np.stack(candidates), axis=0).astype(np.float32)
    np.save(out / 'intrinsics.npy', camera)
    (out / 'camera.json').write_text(json.dumps({'producer':'MoGe2 pinned official source',
        'sample_original_frame_ids':[90,150,209], 'intrinsics':camera.tolist(),
        'per_sample':np.stack(candidates).tolist(), 'reference_calibration':False},indent=2))
    del moge; torch.cuda.empty_cache()
    hub_load = torch.hub.load
    def local_hub(repo, model, *args, **kwargs):
        if repo == 'facebookresearch/dinov3':
            kwargs['source'] = 'local'
            return hub_load(str(ROOT / 'sources/dinov3'), model, *args, **kwargs)
        return hub_load(repo, model, *args, **kwargs)
    torch.hub.load = local_hub
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    ckpt = ROOT / 'assets/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3'
    model, cfg = load_sam_3d_body(str(ckpt / 'model.ckpt'), device='cuda',
                                 mhr_path=str(ckpt / 'assets/mhr_model.pt'))
    estimator = SAM3DBodyEstimator(model, cfg)
    np.save(out / 'faces.npy', estimator.faces)
    np.save(out / 'keypoint_mapping.npy', model.head_pose.keypoint_mapping.detach().cpu().numpy())
    start = time.monotonic()
    for i, img in enumerate(images):
        frame_id = i + 90
        path = out / f'{frame_id:06d}.npz'
        if path.exists():
            continue
        # One manually inspected actor ROI, not a reference annotation or detector output.
        result = estimator.process_one_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    bboxes=np.array([[510, 390, 820, 1000]], dtype=np.float32),
                    cam_int=torch.from_numpy(camera[None]).cuda(), inference_type='full')[0]
        result = {k:v for k,v in result.items() if v is not None}
        result['original_frame_id'] = np.array(frame_id)
        np.savez_compressed(path, **result)
        print(f'frame {frame_id}: {i+1}/120; elapsed {time.monotonic()-start:.1f}s', flush=True)
    (out / 'identity.json').write_text(json.dumps({'torch':torch.__version__, 'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0), 'frames':list(range(90,210)),
        'human_roi':[510,390,820,1000], 'roi_provenance':'Manual visual inspection of Track 1 frame 150',
        'native_mhr':True, 'human_state':'Official native model parameters retained; no conversion invented'},indent=2))

if __name__ == '__main__':
    main()
