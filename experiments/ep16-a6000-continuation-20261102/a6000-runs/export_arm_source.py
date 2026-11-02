from pathlib import Path
import numpy as np,cv2
R=Path('/data1/defu/zly/nvda/continuation-20261102-v1');D=R/'a6000-runs/hand-contact-20261102-v1';cap=cv2.VideoCapture(str(R/'runs/full-episode16-20261030/inputs/episode_000016.0.color.mp4'));fs=[60,70,80,90,100,110,120,130,140,150,160,175,190,200,210,220,230,240,250,260,270,280]
for n in range(0,len(fs),4):
 tiles=[]
 for f in fs[n:n+4]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok;x=im[480:740,460:850];x=cv2.resize(x,(780,520));bar=np.zeros((28,780,3),np.uint8);cv2.putText(bar,f'Original source frame{f}; global pixel grid',(5,20),0,.55,(255,255,255),1)
  for xx in range(460,851,20):
   q=2*(xx-460);cv2.line(x,(q,0),(q,519),(100,100,100),1);cv2.putText(x,str(xx),(q+1,12),0,.32,(0,0,255),1)
  for yy in range(480,741,20):
   q=2*(yy-480);cv2.line(x,(0,q),(779,q),(100,100,100),1);cv2.putText(x,str(yy),(1,q+12),0,.35,(0,0,255),1)
  tiles.append(np.concatenate([bar,x]))
 while len(tiles)<4:tiles.append(np.zeros_like(tiles[0]))
 cv2.imwrite(str(D/f'arm-source-grid-{n//4}.jpg'),np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)]))
cap.release()
