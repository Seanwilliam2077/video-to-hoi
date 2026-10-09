"""Render four matched dynamic-contact conditions and frozen proxy modes."""
import json
import hashlib
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from real_trial import ROOT, project_np
from synthetic_trial import CORNERS, EDGES

OUT = ROOT/'results/interaction-v4'
names = ['static_no_mesh','static_mesh','dynamic_no_mesh','dynamic_mesh']
labels = ['Static contact / no mesh','Static contact / 3D mesh','Dynamic contact / no mesh','Dynamic contact / 3D mesh']
results = {name:dict(np.load(OUT/f'{name}.npz')) for name in names}
modes = dict(np.load(OUT/'contact-modes.npz'))
report = json.loads((OUT/'metrics.json').read_text())
geometry = {s:dict(np.load(OUT/f'{s}-structure.npz')) for s in ['static','dynamic']}
faces = np.load(ROOT/'baseline/faces.npy')
k = np.load(ROOT/'baseline/intrinsics.npy')
warm = dict(np.load(ROOT/'results/interaction-v3/sliding_without_graph.npz'))
origin = warm['object_translation'][60]
ids = np.flatnonzero(np.linalg.norm(warm['vertices'][60]-origin,axis=-1)<.30)
triangles = faces[np.all(np.isin(faces,ids),axis=1)]
edges = np.unique(np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1),axis=0)[::3]
view = Rotation.from_euler('yx',[35,-10],degrees=True).as_matrix()
all_points = np.concatenate([np.concatenate([r['vertices'][:,ids].reshape(-1,3),(CORNERS*r['object_half_extents']@r['object_rotation'].transpose(0,2,1)+r['object_translation'][:,None]).reshape(-1,3)]) for r in results.values()])
xy = ((all_points-origin)@view.T)[:,:2]
mid = (xy.min(0)+xy.max(0))/2
scale = min(565/np.ptp(xy[:,0]),175/np.ptp(xy[:,1]))


def novel(p):
    return np.rint((((p-origin)@view.T)[:,:2]-mid)*scale+[320,425]).astype(np.int32)


def text(im,s,where,size=.52,color=(35,35,35)):
    cv2.putText(im,s,where,cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)


writers = [cv2.VideoWriter(str(OUT/name),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,1120)) for name,fps in [('comparison-1x.mp4',30),('comparison-half-speed.mp4',15)]]
assert all(w.isOpened() for w in writers)
colors = [(145,145,145),(45,160,235),(60,160,60),(170,80,180)]
for t in range(120):
    source = cv2.imread(str(ROOT/'data/frames'/f'{t+90:06d}.jpg'))
    panels=[]
    for name,title in zip(names,labels):
        r=results[name]
        v,rr,c,half = r['vertices'][t],r['object_rotation'][t],r['object_translation'][t],r['object_half_extents']
        cube=CORNERS*half@rr.T+c
        uv=np.rint(project_np(v,k)).astype(int)
        cuv=np.rint(project_np(cube,k)).astype(int)
        overlay=source.copy()
        for a,b in edges: cv2.line(overlay,tuple(uv[a]),tuple(uv[b]),(230,155,45),1,cv2.LINE_AA)
        for a,b in EDGES: cv2.line(overlay,tuple(cuv[a]),tuple(cuv[b]),(35,220,60),2,cv2.LINE_AA)
        cv2.circle(overlay,tuple(cuv[0]),5,(20,80,255),-1)
        panel=np.full((560,640,3),248,np.uint8)
        text(panel,title,(12,24),.68)
        text(panel,f'frame {t+90} | {(t+90)/30:.2f}s | shared view and scale',(12,45))
        panel[58:306]=cv2.resize(overlay[470:750,560:900],(640,248))
        text(panel,'Estimated 3D geometry and interaction edges',(12,327))
        puv,ncv=novel(v),novel(cube)
        for a,b in edges:
            if np.max(np.abs(np.r_[puv[a],puv[b]]))<3000: cv2.line(panel,tuple(puv[a]),tuple(puv[b]),(230,155,45),1,cv2.LINE_AA)
        for a,b in EDGES: cv2.line(panel,tuple(ncv[a]),tuple(ncv[b]),(35,150,45),2,cv2.LINE_AA)
        cv2.circle(panel,tuple(ncv[0]),4,(20,80,255),-1)
        if not name.endswith('_no_mesh'):
            geom=geometry[name.split('_')[0]]
            nh=len(geom['hand_vertex_ids'])
            adj=geom['laplacian'][t,:nh,nh:]
            pairs=np.argwhere(np.abs(adj)>1e-8)[::4]
            obj_world=geom['object_surface_samples']@rr.T+c
            nodeuv=novel(v[geom['hand_vertex_ids']])
            objuv=novel(obj_world)
            for a,b in pairs: cv2.line(panel,tuple(nodeuv[a]),tuple(objuv[b]),(155,90,175),1,cv2.LINE_AA)
        state=modes['states'][t]
        for p,vertex in enumerate(modes['contact_vertex_ids']): cv2.circle(panel,tuple(puv[vertex]),3,colors[state[p]],-1)
        m=report['methods'][name]
        text(panel,f"Held-out pixels {m['withheld_texture_reprojection_mean_px']:.2f} | sampled max PEN {m['sampled_max_penetration_mm']:.1f} mm",(12,534),.46)
        text(panel,'Contact states are geometry proxies; not contact ground truth',(12,553),.45)
        panels.append(panel)
    frame=np.concatenate([np.concatenate(panels[:2],1),np.concatenate(panels[2:],1)],0)
    for w in writers: w.write(frame)
    if t in [0,60,119]: cv2.imwrite(str(OUT/f'comparison-frame{t+90}.jpg'),frame)
for w in writers: w.release()

fig,ax=plt.subplots(figsize=(12,4.4))
cmap=ListedColormap(['#929292','#eba42d','#3f9f51','#af59ab'])
image=ax.imshow(modes['states'].T,aspect='auto',interpolation='nearest',cmap=cmap,vmin=-.5,vmax=3.5,extent=[90,210,8.5,.5])
ax.set(xlabel='Original video frame',ylabel='Candidate hand material point',title='Frozen estimated contact modes used by dynamic conditions\nPurple: unreliable deeply penetrating geometry, not confirmed release')
ax.set_yticks(range(1,9))
bar=fig.colorbar(image,ax=ax,ticks=range(4),pad=.02)
bar.ax.set_yticklabels(['Separated','Sliding','Sticking','Uncertain'])
fig.tight_layout()
fig.savefig(OUT/'contact-modes.png',dpi=150)
plt.close(fig)

checks={}
for name,expected in [('comparison-1x.mp4',30),('comparison-half-speed.mp4',15)]:
    cap=cv2.VideoCapture(str(OUT/name)); fps=cap.get(cv2.CAP_PROP_FPS); count=0
    while cap.read()[0]: count+=1
    cap.release()
    assert count==120 and abs(fps-expected)<1e-6
    checks[name]=dict(decoded_frames=count,fps=fps,sha256=hashlib.sha256((OUT/name).read_bytes()).hexdigest())
for name,r in results.items():
    for key in ['vertices','object_translation','object_rotation','mhr_model_params']: assert np.isfinite(r[key]).all(),(name,key)
    assert np.array_equal(r['original_frame_ids'],np.arange(90,210))
    assert np.allclose(np.linalg.det(r['object_rotation']),1,atol=1e-5)
    assert np.allclose(r['object_rotation']@r['object_rotation'].transpose(0,2,1),np.eye(3),atol=1e-5)
(OUT/'verification.json').write_text(json.dumps(dict(passed=True,scope='Native output integrity, frame coverage, proper rotation and video decoding; no 3D accuracy proof',videos=checks),indent=2))
print(json.dumps(checks,indent=2))
