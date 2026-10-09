"""Bounded native-MHR/cuboid contact pilot; diagnostics, not official challenge scores.

The box is fitted to image correspondences with a cuboid shape prior. Fixed
material attachments are an explicit no-slip hypothesis, not observed 3D truth.
Collision losses/checks sample the surface and do not provide IPC guarantees.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
from pathlib import Path
import json
import time
import cv2
import numpy as np
import torch
import roma
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from synthetic_trial import CORNERS, EDGES, box_sdf

ROOT=Path(__file__).resolve().parent
DEVICE='cuda'

def project_np(x,k):
    return x[...,:2]/x[...,2:3]*np.array([k[0,0],k[1,1]])+k[:2,2]

def project(x,k):
    return x[...,:2]/x[...,2:3].clamp(min=.1)*torch.stack((k[0,0],k[1,1]))+k[:2,2]

def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)

def sdf(x,half):
    q=x.abs()-half
    return torch.linalg.vector_norm(q.clamp(min=0),dim=-1)+q.max(-1).values.clamp(max=0)

def local(points,rotation,center):
    return torch.einsum('tni,tij->tnj',points-center[:,None],rotation)

def decode(model,shape,params,expr,cam):
    result=[]
    with torch.no_grad():
        for a in range(0,len(params),8):
            verts,_=model(tensor(shape[a:a+8]),tensor(params[a:a+8]),tensor(expr[a:a+8]))
            verts=verts.cpu().numpy()/100
            verts[...,1:]*=-1
            result.append(verts+cam[a:a+8,None])
    return np.concatenate(result)

def observations(images,k,hand_depth,out):
    # AI visual annotation of six visible cuboid corners in original Track 1 frame 150.
    # Rounded/partly occluded corners carry uncertainty; no reference geometry used.
    canonical=np.array([[-1,-.85,-1],[1,-.85,-1],[1,-.85,1],[-1,-.85,1],
                        [1,.85,1],[1,.85,-1]],np.float32)
    seed=np.array([[699,585],[756,605],[716,625],[681,595],[710,666],[747,649]],np.float32)
    gray=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in images]
    tracked=np.zeros((len(images),6,2),np.float32)
    valid=np.zeros((len(images),6),bool)
    tracked[60]=seed;valid[60]=True
    for step in [1,-1]:
        previous=60
        for i in range(60+step,len(images) if step==1 else -1,step):
            p0=tracked[previous].reshape(-1,1,2)
            p1,st,e=cv2.calcOpticalFlowPyrLK(gray[previous],gray[i],p0,None,winSize=(21,21),maxLevel=3)
            p2,st2,_=cv2.calcOpticalFlowPyrLK(gray[i],gray[previous],p1,None,winSize=(21,21),maxLevel=3)
            good=(st[:,0]>0)&(st2[:,0]>0)&(np.linalg.norm(p2[:,0]-p0[:,0],axis=1)<2)&(e[:,0]<35)
            tracked[i]=p1[:,0];valid[i]=good&valid[previous]
            previous=i
    ok,r0,t0=cv2.solvePnP(canonical,seed,k,None,flags=cv2.SOLVEPNP_SQPNP)
    ok,r0,t0=cv2.solvePnP(canonical,seed,k,None,rvec=r0,tvec=t0,useExtrinsicGuess=True,flags=cv2.SOLVEPNP_ITERATIVE)
    assert ok and t0[2]>0
    scale=float(hand_depth/t0[2,0])
    corners=canonical*scale
    rotations=np.zeros((len(images),3,3));centers=np.zeros((len(images),3))
    for step in [1,-1]:
        rv=r0.copy();tv=t0.copy()*scale
        for i in range(60,len(images) if step==1 else -1,step):
            good=valid[i]
            if good.sum()>=4:
                ok,rv2,tv2=cv2.solvePnP(corners[good],tracked[i,good],k,None,
                       rvec=rv.copy(),tvec=tv.copy(),useExtrinsicGuess=True,flags=cv2.SOLVEPNP_ITERATIVE)
                if ok and tv2[2]>0:
                    rv,tv=rv2,tv2
            rotations[i]=cv2.Rodrigues(rv)[0];centers[i]=tv[:,0]
    half=np.array([1,.85,1])*scale
    confidence=valid.astype(float)
    report={'annotation_frame_id':150,'image_points':seed.tolist(),'canonical_points':canonical.tolist(),
            'shape_prior':'cuboid with aspect ratio 1:0.85:1; approximate visible foam block',
            'scale_metres_per_unit':scale,'scale_evidence':'median native-human hand depth; not metric ground truth',
            'tracking':'LK forward-backward check, 2px threshold; failed tracks excluded from image loss',
            'valid_corners_per_frame':valid.sum(axis=1).tolist(),'intrinsics_source':'MoGe2 from this Track 1 video',
            'no_artificial_noise':True}
    (out/'object-fit.json').write_text(json.dumps(report,indent=2))
    return corners,tracked,confidence,half,rotations,centers

def surface_projection(p,half):
    q=np.clip(p,-half,half)
    axis=int(np.argmax(np.abs(p)/half))
    sign=1 if p[axis]>=0 else -1
    q[axis]=sign*(half[axis]+.001)
    return q,axis,sign

def make_jacobian(model,shape,params,expr,cam,ids,control_ids,scales,base):
    jac=np.zeros((len(params),len(ids),3,len(control_ids)),np.float32)
    for j,idx in enumerate(control_ids):
        q=params.copy();q[:,idx]+=.001
        v=decode(model,shape,q,expr,cam)
        jac[...,j]=(v[:,ids]-base[:,ids])/.001*scales[j]
        print('Jacobian',j+1,'/',len(control_ids),'control',idx,flush=True)
    return jac

def optimize(label,base,jac,hand_positions,anchor,half,rv0,center0,obs,weights,canon,k,
             model,shape,native,expr,cam,ids,controls,scales):
    base=tensor(base);jac=tensor(jac);half=tensor(half)
    anchor=tensor(anchor);k=tensor(k);obs=tensor(obs);weights=tensor(weights);canon=tensor(canon)
    # One continuous temporal representation for the scene and native pose corrections.
    knot_times=np.linspace(0,len(center0)-1,16)
    spline_np=CubicSpline(knot_times,np.eye(16),axis=0,bc_type='natural')(np.arange(len(center0)))
    spline=tensor(spline_np)
    rv_knots=torch.nn.Parameter(tensor(np.linalg.lstsq(spline_np,rv0,rcond=None)[0]))
    center_knots=torch.nn.Parameter(tensor(np.linalg.lstsq(spline_np,center0,rcond=None)[0]))
    q_knots=torch.nn.Parameter(torch.zeros((16,jac.shape[-1]),device=DEVICE))
    shape=tensor(shape);native=tensor(native);expr=tensor(expr);cam=tensor(cam)
    basis=torch.zeros((len(controls),native.shape[1]),device=DEVICE)
    basis[torch.arange(len(controls),device=DEVICE),torch.as_tensor(controls,device=DEVICE)]=tensor(scales)
    sign=tensor([1,-1,-1])
    ids=torch.as_tensor(ids,device=DEVICE)
    optimizer=torch.optim.Adam([rv_knots,center_knots,q_knots],lr=.006)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=240,eta_min=.00006)
    history=[]
    started=time.monotonic()
    for it in range(240):
        optimizer.zero_grad()
        rv=spline@rv_knots;center=spline@center_knots;q=spline@q_knots
        rot=roma.rotvec_to_rotmat(rv)
        changed=native+q@basis
        decoded,_=model(shape,changed,expr)
        verts=decoded[:,ids]/100*sign+cam[:,None]
        relative=local(verts,rot,center)
        signed=sdf(relative,half)
        hand=relative[:,hand_positions]
        corners=canon[None]@rot.transpose(1,2)+center[:,None]
        reprojection=(project(corners,k)-obs)/3
        image=(reprojection.square().sum(-1)*weights).sum()/weights.sum().clamp(min=1)
        displacement=(verts-base)/.02
        human=displacement.square().mean()
        human_image=((project(verts,k)-project(base,k))/3).square().mean()
        # Signed-distance sampling is a diagnostic approximation, not continuous mesh feasibility.
        negative=torch.relu(.001-signed)/.003
        collision=negative.square().mean()+negative.max(dim=1).values.square().mean()*.1
        if label=='soft_joint':
            contact=(sdf(hand,half)/.005).square().mean()
        else:
            contact=((hand-anchor)/.005).square().mean()
        temporal=((center[2:]-2*center[1:-1]+center[:-2])/.003).square().mean()
        temporal+=((verts[2:,hand_positions]-2*verts[1:-1,hand_positions]+verts[:-2,hand_positions])/.004).square().mean()
        control=q.square().mean()+((q[2:]-2*q[1:-1]+q[:-2])/.2).square().mean()*.02
        angular=((rv[2:]-2*rv[1:-1]+rv[:-2])/.03).square().mean()
        contact_weight=0 if label=='matched_independent' else (.5 if label=='soft_joint' else 2)
        collision_weight=0 if label=='matched_independent' else 8
        loss=image+human*2+human_image*.5+contact*contact_weight+collision*collision_weight+temporal*.08+control*.08+angular*.03
        loss.backward()
        torch.nn.utils.clip_grad_norm_([rv_knots,center_knots,q_knots],100)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():q_knots.clamp_(-.8,.8)
        if it%40==0 or it==239:
            row={'iteration':it,'loss':float(loss.detach()),'image':float(image.detach()),
                 'contact':float(contact.detach()),'collision':float(collision.detach()),
                 'seconds':time.monotonic()-started}
            history.append(row);print(label,row,flush=True)
    return {'rvec':(spline@rv_knots).detach().cpu().numpy(),'center':(spline@center_knots).detach().cpu().numpy(),
            'q':(spline@q_knots).detach().cpu().numpy(),'history':history}

def measure(vertices,rot,center,half,anchors,contact_ids,faces,canon,obs,confidence,k):
    v=vertices
    # Include every vertex, every edge midpoint and every triangle centroid.
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    maxima=[];counts=[];all_gap=[];closure=[];uv_error=[]
    for t in range(len(v)):
        samples=np.concatenate([v[t],(v[t,edges[:,0]]+v[t,edges[:,1]])*.5,v[t,faces].mean(axis=1)])
        relative=(samples-center[t])@rot[t]
        distances=box_sdf(relative,half)
        maxima.append(float(np.maximum(-distances,0).max()))
        counts.append(int((distances<-.0005).sum()))
        hand=(v[t,contact_ids]-center[t])@rot[t]
        all_gap.extend(np.abs(box_sdf(hand,half)))
        closure.extend(np.linalg.norm(hand-anchors,axis=-1))
        uv=project_np(canon@rot[t].T+center[t],k)
        uv_error.extend(np.linalg.norm(uv-obs[t],axis=-1)[confidence[t]>0])
    return {'tracked_corner_reprojection_mean_px':float(np.mean(uv_error)),
            'contact_surface_distance_mean_mm':float(np.mean(all_gap)*1000),
            'fixed_attachment_residual_mean_mm':float(np.mean(closure)*1000),
            'sampled_max_penetration_mm':float(np.max(maxima)*1000),
            'sampled_mean_frame_max_penetration_mm':float(np.mean(maxima)*1000),
            'frames_with_sampled_penetration_gt_0_5mm':int(np.count_nonzero(counts)),
            'sampled_penetrating_points_total':int(np.sum(counts)),
            'object_prediction_only_second_difference_mean_mm':float(np.linalg.norm(np.diff(center,n=2,axis=0),axis=-1).mean()*1000),
            'per_frame_max_penetration_mm':(np.array(maxima)*1000).tolist()}

def videos(results,images,half,canon,k,ids,faces,out):
    writer=cv2.VideoWriter(str(out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(512*len(results),576))
    viewwriter=cv2.VideoWriter(str(out/'interaction_3d.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(480*len(results),480))
    reference_center=np.median(results['independent']['center'],axis=0)
    view=Rotation.from_euler('yx',[35,-10],degrees=True).as_matrix()
    keep_faces=faces[np.any(np.isin(faces,ids),axis=1)]
    edges=np.unique(np.sort(np.concatenate([keep_faces[:,[0,1]],keep_faces[:,[1,2]],keep_faces[:,[2,0]]]),axis=1),axis=0)
    edges=edges[::3]
    def novel_project(p):
        p=(p-reference_center)@view.T
        return np.column_stack((p[:,0]*800+240,p[:,1]*800+260)).astype(np.int32)
    for t in range(len(images)):
        overlays=[];views=[]
        for name,r in results.items():
            image=images[t].copy();rot=r['rotation'][t];center=r['center'][t];v=r['vertices'][t]
            cube=CORNERS*half@rot.T+center
            uv=project_np(cube,k).astype(np.int32)
            for i,j in EDGES:cv2.line(image,tuple(uv[i]),tuple(uv[j]),(40,220,60),2,cv2.LINE_AA)
            h=v[ids];huv=project_np(h,k).astype(np.int32)
            gap=box_sdf((h-center)@rot,half)
            for i,p in enumerate(huv[::8]):
                cv2.circle(image,tuple(p),1,(220,70,220) if gap[i*8]<-.0005 else (220,160,30),-1)
            # Consistent source crop, resized for a visible hand/object diagnostic.
            panel=cv2.resize(image[350:950,450:1000],(512,556))
            cv2.rectangle(panel,(0,0),(512,32),(255,255,255),-1)
            cv2.putText(panel,name,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,20,20),2)
            pad=np.full((576,512,3),255,np.uint8);pad[:556]=panel
            cv2.putText(pad,'Green: box | magenta: inside samples',(6,571),cv2.FONT_HERSHEY_SIMPLEX,.42,(40,40,40),1)
            overlays.append(pad)
            scene=np.full((480,480,3),248,np.uint8);puv=novel_project(v);cuv=novel_project(cube)
            for i,j in edges:
                a,b=puv[i],puv[j]
                if np.max(np.abs(np.r_[a,b]))<3000:cv2.line(scene,tuple(a),tuple(b),(190,145,90),1,cv2.LINE_AA)
            for i,j in EDGES:cv2.line(scene,tuple(cuv[i]),tuple(cuv[j]),(40,140,40),2,cv2.LINE_AA)
            cv2.putText(scene,name,(10,27),cv2.FONT_HERSHEY_SIMPLEX,.58,(20,20,20),2)
            cv2.putText(scene,'Novel view; approximate block geometry',(7,465),cv2.FONT_HERSHEY_SIMPLEX,.43,(50,50,50),1)
            views.append(scene)
        writer.write(np.concatenate(overlays,axis=1));viewwriter.write(np.concatenate(views,axis=1))
        if t==60:cv2.imwrite(str(out/'comparison-frame150.jpg'),np.concatenate(overlays,axis=1))
    writer.release();viewwriter.release()

def main():
    started=time.monotonic();out=ROOT/'results/real';out.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'baseline'/f'{i:06d}.npz' for i in range(90,210)]
    data=[dict(np.load(p)) for p in files]
    shape=np.stack([d['shape_params'] for d in data]);expr=np.stack([d['expr_params'] for d in data])
    native=np.stack([d['mhr_model_params'] for d in data]);cam=np.stack([d['pred_cam_t'] for d in data])
    initial=np.stack([d['pred_vertices']+d['pred_cam_t'] for d in data])
    faces=np.load(ROOT/'baseline/faces.npy');k=np.load(ROOT/'baseline/intrinsics.npy')
    images=[cv2.imread(str(ROOT/'data/frames'/f'{i:06d}.jpg')) for i in range(90,210)]
    depth=np.median(np.stack([d['pred_keypoints_3d'][[41,62],2]+d['pred_cam_t'][2] for d in data]))
    canon,obs,confidence,half,rot0,center0=observations(images,k,depth,out)
    # Smooth rotations in one local tangent chart; preserve camera/identity conventions.
    reference=rot0[60]
    rel=Rotation.from_matrix(np.einsum('ij,tjk->tik',reference.T,rot0)).as_rotvec()
    rot0=np.einsum('ij,tjk->tik',reference,Rotation.from_rotvec(gaussian_filter1d(rel,1.5,axis=0)).as_matrix())
    rv0=Rotation.from_matrix(rot0).as_rotvec();center0=gaussian_filter1d(center0,1.5,axis=0)
    params=native.copy();params[:,:136]=gaussian_filter1d(native[:,:136],1.5,axis=0)
    cam=gaussian_filter1d(cam,1.5,axis=0)
    model=torch.jit.load(str(ROOT/'assets/weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt')).cuda().eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    raw_decoded=decode(model,shape,native,expr,np.stack([d['pred_cam_t'] for d in data]))
    decoder_error=float(np.max(np.abs(raw_decoded-initial)))
    assert decoder_error<2e-5,decoder_error
    base=decode(model,shape,params,expr,cam)
    keypoints=data[60]['pred_keypoints_3d']+data[60]['pred_cam_t']
    hand_ids=[];contacts=[];anchors=[]
    for indices in [[41,28,32,36,40],[62,49,53,57,61]]:
        palm=keypoints[indices].mean(axis=0)
        candidates=np.flatnonzero(np.linalg.norm(base[60]-palm,axis=1)<.105)
        assert len(candidates)>20
        hand_ids.extend(candidates.tolist())
        loc=(base[60,candidates]-center0[60])@rot0[60]
        ranking=np.argsort(np.abs(box_sdf(loc,half)))
        picked=[]
        for j in ranking:
            if all(np.linalg.norm(base[60,candidates[j]]-base[60,candidates[l]])>.012 for l in picked):
                picked.append(j)
            if len(picked)==4:break
        for j in picked:
            contacts.append(int(candidates[j]));anchors.append(surface_projection(loc[j],half)[0])
    contacts=np.array(contacts);anchors=np.array(anchors)
    ids=np.unique(np.r_[hand_ids,np.flatnonzero(np.linalg.norm(base[60]-center0[60],axis=1)<.32),contacts])
    hand_positions=np.searchsorted(ids,contacts)
    # Select actual native arm/hand controls by their measured influence on the contact vertices.
    candidate=np.arange(6,136);score=[]
    for chunk in np.array_split(candidate,9):
        p=np.repeat(params[60:61],len(chunk),axis=0)
        for row,col in enumerate(chunk):p[row,col]+=.001
        v=decode(model,np.repeat(shape[60:61],len(chunk),axis=0),p,
                 np.repeat(expr[60:61],len(chunk),axis=0),np.repeat(cam[60:61],len(chunk),axis=0))
        hand_motion=np.linalg.norm((v[:,contacts]-base[60,contacts][None])/.001,axis=-1).mean(axis=1)
        body_motion=np.linalg.norm((v-base[60][None])/.001,axis=-1).mean(axis=1)
        score.extend(hand_motion/(body_motion+1e-4)*np.sqrt(hand_motion))
    selected=candidate[np.argsort(score)[-12:]]
    controls=np.r_[np.arange(3),selected]
    root_gains=[]
    for idx in range(3):
        p=params[60:61].copy();p[:,idx]+=.001
        v=decode(model,shape[60:61],p,expr[60:61],cam[60:61])
        root_gains.append(float(np.linalg.norm((v[0]-base[60])/.001,axis=-1).mean()))
    scales=np.r_[.03/np.array(root_gains),np.full(12,.12)]
    print('Controls',controls.tolist(),'collision vertices',len(ids),'contact vertices',contacts.tolist(),flush=True)
    jac=make_jacobian(model,shape,params,expr,cam,ids,controls,scales,base)
    results={'independent':{'rotation':rot0,'center':center0,'vertices':base,'native':params}}
    metrics={}
    metrics['independent']=measure(base,rot0,center0,half,anchors,contacts,faces,canon,obs,confidence,k)
    for label in ['matched_independent','soft_joint','attachment_collision']:
        r=optimize(label,base[:,ids],jac,hand_positions,anchors,half,rv0,center0,obs,confidence,canon,k,
                   model,shape,params,expr,cam,ids,controls,scales)
        changed=params.copy();changed[:,controls]+=r['q']*scales
        actual=decode(model,shape,changed,expr,cam)
        # Validate that optimization and final serialization decode identical native states.
        with torch.no_grad():
            checked,_=model(tensor(shape),tensor(changed),tensor(expr))
            checked=checked.cpu().numpy()/100
            checked[...,1:]*=-1
            checked+=cam[:,None]
        linear_error=float(np.linalg.norm(actual-checked,axis=-1).max())
        assert linear_error<2e-5,linear_error
        rotation=Rotation.from_rotvec(r['rvec']).as_matrix()
        results[label]={'rotation':rotation,'center':r['center'],'vertices':actual,'native':changed}
        m=measure(actual,rotation,r['center'],half,anchors,contacts,faces,canon,obs,confidence,k)
        m['max_native_decode_consistency_error_mm']=linear_error*1000
        m['human_vertex_displacement_mean_mm']=float(np.linalg.norm(actual-base,axis=-1).mean()*1000)
        metrics[label]=m
        (out/(label+'-optimization.json')).write_text(json.dumps(r['history'],indent=2))
    report={'scope':'120-frame Track 1 episode 16 diagnostic pilot, not official scores',
            'original_frame_ids':list(range(90,210)),'fps':30,'no_artificial_noise':True,
            'native_decoder_max_error_metres':decoder_error,'half_extents_metres':half.tolist(),
            'fixed_contact_vertex_ids':contacts.tolist(),'fixed_object_attachment_points':anchors.tolist(),
            'optimized_native_control_ids':controls.tolist(),'bounded_native_control_scales':scales.tolist(),
            'collision_sampling':'all native vertices + all unique edge midpoints + all face centroids',
            'continuous_collision_guarantee':False,'reference_relative_ACC_available':False,
            'solver':'Direct differentiable native MHR decode; linear approximation rejected',
            'temporal_parameterization':'16 natural cubic-spline knots over 120 frames; shared by object motion and native pose corrections',
            'native_translation_gains_metres_per_unit':root_gains,
            'contact_hypothesis':'fixed material correspondences inferred at frame 150; not 3D ground truth',
            'human_image_prior':'initializer-preservation prior, not independent reference annotations',
            'matched_temporal_control':'matched_independent uses the same spline, native controls, image/human priors, optimizer and iteration budget, with contact/collision weights zero',
            'metrics':metrics,'seconds':time.monotonic()-started}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    for name,r in results.items():
        np.savez_compressed(out/(name+'.npz'),original_frame_ids=np.arange(90,210),
            mhr_model_params=r['native'],shape_params=shape,expr_params=expr,cam_t=cam,
            object_rotation=r['rotation'],object_translation=r['center'],object_half_extents=half,
            vertices=r['vertices'])
    videos(results,images,half,canon,k,ids,faces,out)
    print(json.dumps({name:{key:val for key,val in m.items() if not isinstance(val,list)} for name,m in metrics.items()},indent=2),flush=True)
    print('Real trial finished',time.monotonic()-started,flush=True)

if __name__=='__main__':main()
