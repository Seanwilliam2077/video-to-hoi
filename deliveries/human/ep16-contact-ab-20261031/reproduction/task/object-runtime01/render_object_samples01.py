"""Static source/object/native-wrist review; no physical-contact assertions."""
from pathlib import Path
import argparse,json,cv2,numpy as np
from fit_object_hypotheses import CONFIG,VIDEO,K,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);args=p.parse_args()
    out=args.candidate/'sample-review';out.mkdir(exist_ok=False)
    data=np.load(args.candidate/'contact_ready.npz',allow_pickle=False)
    human=np.load(CONFIG['human_joints'],allow_pickle=False)
    frames=[0,40,75,90,120,210,230,250,260,280,289,290,359]
    cap=cv2.VideoCapture(str(VIDEO));rows=[]
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,rgb=cap.read();assert ok
        v=data['vertices']@data['object_to_camera'][frame,:3,:3].T+data['object_to_camera'][frame,:3,3]
        q=v@K.T;xy=q[:,:2]/q[:,2:];overlay=rgb.copy()
        hull=cv2.convexHull(xy.astype(np.float32)).reshape(-1,2)
        cv2.polylines(overlay,[np.rint(hull).astype(np.int32)],True,(0,255,255),2,cv2.LINE_AA)
        for face in data['faces'][::12]:cv2.polylines(overlay,[np.rint(xy[face]).astype(np.int32)],True,(0,190,210),1,cv2.LINE_AA)
        wrists=human[frame,[42,78]];h=wrists@K.T;uv=h[:,:2]/h[:,2:]
        for index,point in enumerate(uv):
            cv2.circle(overlay,tuple(np.rint(point).astype(int)),4,(255,60,255),-1,cv2.LINE_AA)
            cv2.putText(overlay,f'Native wrist z={wrists[index,2]:.3f}',tuple(np.rint(point+np.array([5,-5])).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,60,255),1,cv2.LINE_AA)
        # One fixed source-pixel ROI, preserving relative geometry across frames.
        source=cv2.resize(rgb[480:820,560:900],(680,680),interpolation=cv2.INTER_CUBIC)
        model=cv2.resize(overlay[480:820,560:900],(680,680),interpolation=cv2.INTER_CUBIC)
        canvas=np.zeros((730,1360,3),np.uint8);canvas[50:,:680]=source;canvas[50:,680:]=model
        cv2.putText(canvas,f'EP16 original frame{frame:03d}: sourceRGB',(8,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(canvas,'Current-K object / fixed native wrists; scale uncertain',(688,30),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
        path=out/f'object-native-wrists-{frame:06d}.jpg';cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,93]);rows.append({'frame':frame,'path':str(path),'sha256':digest(path),'object_center_depth_m':float(data['object_to_camera'][frame,2,3]),'native_wrist_depths_m':wrists[:,2].tolist()})
    cap.release();(out/'summary.json').write_text(json.dumps({'review_mode':'Static geometry projection and native wrist-depth gauge, not true contact or scale acceptance','images':rows},indent=2)+'\n');print(json.dumps({'output':str(out),'frames':frames}),flush=True)

if __name__=='__main__':main()
