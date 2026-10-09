"""Independent known-truth fixture; not challenge measurements."""
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
CORNERS = np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]],float)
EDGES = [(i,j) for i in range(8) for j in range(i+1,8) if np.count_nonzero(CORNERS[i]-CORNERS[j])==1]

def box_sdf(points, half):
    q = np.abs(points) - half
    return np.linalg.norm(np.maximum(q,0),axis=-1) + np.minimum(q.max(axis=-1),0)

def smooth(x):
    return gaussian_filter1d(x,1.5,axis=0,mode='nearest')

def render_synthetic(methods, rotations, half, radius, destination):
    view = Rotation.from_euler('yx',[25,-20],degrees=True).as_matrix()
    def project(p):
        p = p @ view.T
        return np.column_stack((p[:,0]*1100+240,-p[:,1]*1100+250)).astype(np.int32)
    writer = cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),30,(1440,480))
    for t in range(len(rotations)):
        panels=[]
        for label,points in methods.items():
            image=np.full((480,480,3),248,np.uint8)
            obj=points[t,0]
            cube=CORNERS*half @ rotations[t].T + obj
            uv=project(cube)
            for i,j in EDGES:
                cv2.line(image,tuple(uv[i]),tuple(uv[j]),(70,130,30),2,cv2.LINE_AA)
            for h,p in enumerate(points[t,1:]):
                for axis in [0,1,2]:
                    a=np.linspace(0,2*np.pi,40)
                    circle=np.zeros((40,3));circle[:,(axis+1)%3]=radius*np.cos(a);circle[:,(axis+2)%3]=radius*np.sin(a)
                    cv2.polylines(image,[project(circle+p)],True,(180,90,35) if h==0 else (30,120,220),1,cv2.LINE_AA)
            cv2.putText(image,label,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.62,(20,20,20),2)
            cv2.putText(image,'Known-truth synthetic fixture',(15,460),cv2.FONT_HERSHEY_SIMPLEX,.5,(70,70,70),1)
            panels.append(image)
        writer.write(np.concatenate(panels,axis=1))
    writer.release()

def main():
    out=ROOT/'results/synthetic';out.mkdir(parents=True,exist_ok=True)
    n=120; t=np.arange(n)/30; half=.10;radius=.025
    g=np.column_stack((.06*np.sin(t*1.8),.025*np.cos(t*2.1),.025*np.sin(t*1.2)))
    rot=Rotation.from_euler('z',.35*np.sin(t*1.4)).as_matrix()
    offsets=np.array([[0,0,0],[-half-radius,0,0],[half+radius,0,0]])
    relative=np.einsum('tij,hj->thi',rot,offsets)
    truth=g[:,None]+relative
    rng=np.random.default_rng(20261009)
    raw=truth+rng.normal(0,.008,truth.shape)
    raw[:,0,0]+=.012*np.sin(t*2.5)
    raw[:,1,0]+=.008
    independent=smooth(raw)
    # Closed-form quadratic soft-contact solution with identical observation/smoothing input.
    y=independent-relative
    matrix=np.array([[1+8,-4,-4],[-4,1+4,0],[-4,0,1+4]],float)
    soft=np.einsum('hk,tkj->thj',np.linalg.inv(matrix),y)+relative
    shared=smooth((raw-relative).mean(axis=1))[:,None]+relative
    methods={'Independent':independent,'Soft joint':soft,'Shared structure':shared}
    metrics={}
    for name,x in methods.items():
        local=np.einsum('tji,thj->thi',rot,x[:,1:]-x[:,0,None])
        gap=box_sdf(local,half)-radius
        closure=(x-x[:,0,None])-relative
        metrics[name]={'trajectory_rmse_mm':float(np.sqrt(np.mean((x-truth)**2))*1000),
                      'contact_closure_rms_mm':float(np.sqrt(np.mean(closure[:,1:]**2))*1000),
                      'mean_contact_gap_mm':float(np.mean(np.abs(gap))*1000),
                      'max_penetration_mm':float(np.maximum(-gap,0).max()*1000),
                      'known_truth_second_difference_error_mm':float(np.linalg.norm(np.diff(x,n=2,axis=0)-np.diff(truth,n=2,axis=0),axis=-1).mean()*1000)}
    assert np.max(np.abs(box_sdf(offsets[1:],half)-radius))<1e-12
    assert metrics['Shared structure']['max_penetration_mm']<1e-8
    assert metrics['Shared structure']['trajectory_rmse_mm']<metrics['Independent']['trajectory_rmse_mm']
    # A wrong fixed binding must fail a slip control; it cannot be advertised as universally correct.
    slip_truth=truth.copy();slip_truth[:,1,1]+=.04*np.sin(t*1.8)
    forced=smooth((slip_truth-relative).mean(axis=1))[:,None]+relative
    slip_error=float(np.sqrt(np.mean((forced-slip_truth)**2))*1000)
    assert slip_error>5
    report={'scope':'Independent synthetic fixture only, not Track 1 scores', 'seed':20261009,
            'frames':n,'fps':30,'assumption':'Known rotations and fixed no-slip grasp attachments',
            'metrics':metrics,'wrong_fixed_binding_slip_control_rmse_mm':slip_error,
            'tests':['rotated box contact distance','exact shared closure','shared trajectory improvement','wrong-binding slip control']}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'trajectories.npz',truth=truth,raw=raw,independent=independent,soft=soft,shared=shared,rotation=rot)
    render_synthetic(methods,rot,half,radius,out/'comparison.mp4')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
