from pathlib import Path
import cv2,torch,numpy as np
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');D=R/'a6000-runs/hand-contact-20261102-v1';b=torch.load(R/'a6000-runs/direction-correction-v1/source-material-candidate-v6/refined.pth',map_location='cpu',weights_only=False);cap=cv2.VideoCapture(str(R/'runs/full-episode16-20261030/inputs/episode_000016.0.color.mp4'));tiles=[]
for f in [100,175,220]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok;H=np.asarray(b['observations']['K_full'][f])@np.linalg.inv(b['observations']['postopt_K_rois'][f]);obj=cv2.warpPerspective(np.float32(b['observations']['postopt_object_mask'][f]),H,(1536,1152))>.5;human=cv2.warpPerspective(np.float32(b['observations']['postopt_human_mask'][f]),H,(1536,1152))>.5;color=np.zeros_like(im);color[obj]=(0,0,255);color[human]=(255,0,0);overlay=im.copy();sel=obj|human;overlay[sel]=.6*overlay[sel]+.4*color[sel];panel=np.concatenate([im[480:740,460:850],overlay[480:740,460:850]],1);cv2.putText(panel,f'Source{f}; object red, human blue',(5,20),0,.5,(255,255,255),1);tiles.append(panel)
cv2.imwrite(str(D/'source-mask-order-review.jpg'),np.concatenate(tiles,0));cap.release()
