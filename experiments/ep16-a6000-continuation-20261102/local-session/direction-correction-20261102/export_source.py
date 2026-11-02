from pathlib import Path
import cv2,numpy as np,torch,json
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');D=R/'a6000-runs/direction-correction-v1';cap=cv2.VideoCapture(str(R/'runs/full-episode16-20261030/inputs/episode_000016.0.color.mp4'))
fs=[0,30,60,90,100,110,120,130,140,150,155,160,165,170,175,180,190,200,210,220,230,240,250,260,270,280,300,330,359]
for page in range(0,len(fs),4):
 tiles=[]
 for f in fs[page:page+4]:
  cap.set(0 if False else cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok;im=cv2.resize(im[490:790,590:890],(600,600));bar=np.zeros((30,600,3),np.uint8);cv2.putText(bar,f'SOURCE {f}; grid coordinates = ROI pixels',(5,22),0,.6,(255,255,255),1)
  for v in range(0,300,20):
   cv2.line(im,(2*v,0),(2*v,599),(80,80,80),1);cv2.line(im,(0,2*v),(599,2*v),(80,80,80),1);cv2.putText(im,str(v),(2*v+2,14),0,.38,(0,0,255),1);cv2.putText(im,str(v),(2,2*v+14),0,.38,(0,0,255),1)
  tiles.append(np.concatenate([bar,im]))
 while len(tiles)<4:tiles.append(np.zeros_like(tiles[0]))
 cv2.imwrite(str(D/f'source-grid-{page//4}.jpg'),np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)]))
cap.release()
