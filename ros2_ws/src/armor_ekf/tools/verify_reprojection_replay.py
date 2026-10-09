"""Validate saved four-plate projections; compare model-only plates with other detections."""
import argparse,csv,json,itertools
from pathlib import Path
import cv2
import numpy as np


def rotation(row,prefix):
    x,y,z,w=[float(row[f'{prefix}_{k}']) for k in ['qx','qy','qz','qw']]
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def main():
    cli=argparse.ArgumentParser(); cli.add_argument('--output-dir',required=True)
    cli.add_argument('--gimbal-pitch-rad',type=float); cli.add_argument('--direct-dir')
    cli.add_argument('--expect-unchanged-ekf',action='store_true')
    args=cli.parse_args()
    out=Path(args.output_dir); ws=Path(__file__).resolve().parents[3]
    yaw=list(csv.DictReader((out/'yaw.csv').open()))
    baseline=list(csv.DictReader((ws/'output/coordinate_transform/world/yaw.csv').open()))
    unchanged_ekf=args.gimbal_pitch_rad is None and yaw==baseline
    if args.expect_unchanged_ekf: assert unchanged_ekf,'Previous world EKF output changed'
    direct_count=0
    if args.direct_dir:
        direct=list(csv.DictReader((Path(args.direct_dir)/'yaw.csv').open()))
        for a,b in zip(direct,yaw):
            for key in ['frame','time','detection_valid','measurement_used','observed_yaw','filtered_yaw','yaw_rate','x','y','z','qx','qy','qz','qw']:
                x,y=float(a[key]),float(b[key]); assert (np.isnan(x) and np.isnan(y)) or abs(x-y)<1e-9
        direct_p=list(csv.DictReader((Path(args.direct_dir)/'reprojection.csv').open()))
        cached_p=list(csv.DictReader((out/'reprojection.csv').open()))
        assert direct_p==cached_p[:len(direct_p)]
        direct_count=len(direct)
    groups={}
    for row in csv.DictReader((out/'reprojection.csv').open()): groups.setdefault(int(row['frame']),[]).append(row)
    detections={}
    for row in csv.DictReader((ws/'output/tracker_cache/detections.csv').open()):
        if int(row['detection_index'])>=0: detections.setdefault(int(row['frame']),[]).append(row)
    K=np.array([[3571.1890927935297,0,683.5999418028833],[0,3569.0234714126823,565.5838917436471],[0,0,1]])
    D=np.array([-.029757126887136703,.371503828585697,.0011898367108863406,-.0025516396408119418,0])
    obj=np.array([[0,-.0675,.0275],[0,-.0675,-.0275],[0,.0675,-.0275],[0,.0675,.0275]])
    transform=np.array(json.loads((ws/'preparation/coordinate_transform/geometry_baseline.json').read_text())['T_odom_camera_optical_frame'])
    if args.gimbal_pitch_rad is not None:
        def ry(a): return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        transform[:3,:3]=ry(args.gimbal_pitch_rad+.091)@np.array([[0,0,1],[-1,0,0],[0,-1,0]])
        transform[:3,3]=ry(args.gimbal_pitch_rad)@np.array([.0461892,0,-.0603874])
    transform_error=0.0
    valid=missing=in_image=front=0; pixel_error=radius_error=0.; anchor_errors=[]; overlaps=[]; distances=[]; plate_pitch=[]
    samples=[]
    for y in yaw:
        frame=int(y['frame']); rows=groups[frame]
        if y['detection_valid']=='0':
            assert len(rows)==1 and rows[0]['valid']=='0' and rows[0]['plate']=='-1'; missing+=1; continue
        assert len(rows)==4 and [r['plate'] for r in rows]==['0','1','2','3']; valid+=1
        center=np.array([float(rows[0]['center_'+k]) for k in 'xyz'])
        for r in rows:
            t=np.array([float(r['world_'+k]) for k in 'xyz']); R=rotation(r,'world')
            ct=np.array([float(r['camera_'+k]) for k in 'xyz']); cr=rotation(r,'camera')
            transform_error=max(transform_error,float(np.max(np.abs(transform[:3,:3]@ct+transform[:3,3]-t))),float(np.max(np.abs(transform[:3,:3]@cr-R))))
            heading=float(y['observed_yaw'])+int(r['plate'])*np.pi/2
            assert np.max(np.abs(t-(center+.25*np.array([np.cos(heading),np.sin(heading),0]))))<1e-10
            radial=t-center; radius_error=max(radius_error,abs(np.linalg.norm(radial)-.25))
            assert abs(radial[2])<1e-12 and np.dot(radial[:2],R[:2,0])>0
            if r['plate']!='0': assert abs(R[2,0]-np.sin(np.pi/12))<1e-12
            in_image+=int(r['in_image']); front+=int(r['front_facing'])
            if r['depth_valid']=='1':
                ct=np.array([float(r['camera_'+k]) for k in 'xyz']); cr=rotation(r,'camera')
                xy=cv2.projectPoints(obj,cv2.Rodrigues(cr)[0],ct,K,D)[0].reshape(4,2)
                stored=np.array([[float(r[f'k{k}x']),float(r[f'k{k}y'])] for k in range(4)])
                pixel_error=max(pixel_error,float(np.max(np.abs(stored-xy))))
        anchor=rows[0]; selected=int(y['selected_detection_index']); ds=detections[frame]; d=ds[selected]
        corners=np.array([[float(d[f'k{k}x']),float(d[f'k{k}y'])] for k in range(4)],np.float32).astype(np.float64)
        actual=np.array([[float(anchor[f'k{k}x']),float(anchor[f'k{k}y'])] for k in range(4)])
        error=float(np.sqrt(np.mean(np.sum((actual-corners)**2,axis=1)))); anchor_errors.append(error)
        assert abs(error-float(y['reprojection_error']))<1e-7
        plate_pitch.append(float(np.degrees(np.arcsin(-rotation(anchor,'world')[2,0]))))
        others=[d for i,d in enumerate(ds) if i!=selected]
        candidates=[r for r in rows[1:] if r['in_image']=='1']
        if others and candidates:
            target=[np.array([[float(d[f'k{k}x']),float(d[f'k{k}y'])] for k in range(4)],np.float32) for d in others]
            predicted=[np.array([[float(r[f'k{k}x']),float(r[f'k{k}y'])] for k in range(4)],np.float32) for r in candidates]
            costs=np.zeros((len(target),len(predicted)))
            for i,a in enumerate(target):
                for j,b in enumerate(predicted):
                    inter=cv2.intersectConvexConvex(cv2.convexHull(a),cv2.convexHull(b))[0]
                    costs[i,j]=inter/(cv2.contourArea(a)+cv2.contourArea(b)-inter) if inter>0 else 0
            n=min(len(target),len(predicted))
            options=[(sum(costs[i,j] for i,j in zip(ix,jx)),-sum(float(np.linalg.norm(target[i].mean(0)-predicted[j].mean(0))) for i,j in zip(ix,jx)),ix,jx) for ix in itertools.combinations(range(len(target)),n) for jx in itertools.permutations(range(len(predicted)),n)]
            _,_,ix,jx=max(options)
            for i,j in zip(ix,jx):
                overlaps.append(float(costs[i,j])); distances.append(float(np.linalg.norm(target[i].mean(0)-predicted[j].mean(0))))
            if len(samples)<4: samples.append(frame)
    if valid == 0:
        report=dict(status='NO_VALID_MODEL',frames=len(yaw),valid_model_frames=0,
                    missing_model_frames=missing,
                    note='Every frame cleared its model; calibration or pose evidence must be checked before spatial validation.')
        (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        return
    assert pixel_error<1e-7 and radius_error<1e-12 and transform_error<1e-12
    if (out/'annotated.mp4').exists():
        capture=cv2.VideoCapture(str(out/'annotated.mp4'))
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==len(yaw)
        for frame in [0]+samples:
            capture.set(cv2.CAP_PROP_POS_FRAMES,frame//2); ok,img=capture.read(); assert ok
            assert cv2.imwrite(str(out/f'frame_{frame:04}.png'),img)
    report=dict(status='PASS',frames=len(yaw),valid_model_frames=valid,missing_model_frames=missing,
        unchanged_world_ekf=unchanged_ekf,direct_inference_agreement_frames=direct_count,max_world_camera_transform_component_error=transform_error,max_radius_error_m=radius_error,max_independent_projection_error_px=pixel_error,
        anchor_rms_error_median_px=float(np.median(anchor_errors)),anchor_rms_error_p95_px=float(np.percentile(anchor_errors,95)),
        total_in_image_plates=in_image,total_front_facing_plates=front,
        measured_anchor_pitch_median_deg=float(np.median(plate_pitch)),model_other_plate_outward_pitch_deg=-15,
        other_detection_model_pairs=len(overlaps),other_plate_iou_median=float(np.median(overlaps)),
        other_plate_center_error_median_px=float(np.median(distances)),
        other_plate_center_error_p90_px=float(np.percentile(distances,90)),
        other_plate_pairs_over_100px=sum(d>100 for d in distances),
        comparison_limit='Distinct maximum-IoU matching to other same-frame YOLO detections; no physical plate IDs, occlusion or ground truth. In-image does not imply physically visible.')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
