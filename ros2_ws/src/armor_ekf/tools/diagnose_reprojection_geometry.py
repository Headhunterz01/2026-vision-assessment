import csv,json,sys
from pathlib import Path
import numpy as np,cv2
from verify_reprojection_replay import rotation
ws=Path(__file__).resolve().parents[3]
out=ws/'output/reprojection'
yaw=list(csv.DictReader((out/'yaw.csv').open())); ds={}
for d in csv.DictReader((ws/'output/tracker_cache/detections.csv').open()):
 if int(d['detection_index'])>=0:ds.setdefault(int(d['frame']),[]).append(d)
def ry(a):return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
def rz(a):return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
opt=np.array([[0,0,1],[-1,0,0],[0,-1,0]])
K=np.array([[3571.1890927935297,0,683.5999418028833],[0,3569.0234714126823,565.5838917436471],[0,0,1]])
D=np.array([-.029757126887136703,.371503828585697,.0011898367108863406,-.0025516396408119418,0])
obj=np.array([[0,-.0675,.0275],[0,-.0675,-.0275],[0,.0675,-.0275],[0,.0675,.0275]])
reports=[]
for pitch in [1.34,0,.2,.5,.8]:
 for center_sign,board_pitch in [(-1,-np.pi/12),(1,-np.pi/12),(-1,np.pi/12)]:
  C=ry(pitch+.091)@opt; scores=[]; errors=[]; angles=[]
  for y in yaw[::10]:
   frame=int(y['frame'])
   if y['detection_valid']!='1' or len(ds[frame])<2:continue
   t=np.array([float(y['camera_'+k]) for k in 'xyz']); R=rotation({('camera_'+k):y['camera_'+k] for k in ['qx','qy','qz','qw']},'camera')
   wr=C@R; a=np.arctan2(wr[1,0],wr[0,0]); angles.append(np.degrees(np.arcsin(-wr[2,0])))
   p=C@t; center=p+center_sign*.25*np.array([np.cos(a),np.sin(a),0]); predicted=[]
   for i in range(1,4):
    b=a+i*np.pi/2; wp=center-center_sign*.25*np.array([np.cos(b),np.sin(b),0]); ar=rz(b)@ry(board_pitch)
    cr=C.T@ar; ct=C.T@wp
    xy=cv2.projectPoints(obj,cv2.Rodrigues(cr)[0],ct,K,D)[0].reshape(4,2).astype(np.float32);predicted.append(xy)
   for j,d in enumerate(ds[frame]):
    if j==int(y['selected_detection_index']):continue
    xy=np.array([[float(d[f'k{k}x']),float(d[f'k{k}y'])] for k in range(4)],np.float32)
    errors.append(min(np.linalg.norm(xy.mean(0)-p.mean(0)) for p in predicted))
    ss=[]
    for p in predicted:
     inter=cv2.intersectConvexConvex(cv2.convexHull(xy),cv2.convexHull(p))[0]
     ss.append(inter/(cv2.contourArea(xy)+cv2.contourArea(p)-inter) if inter>0 else 0)
    scores.append(max(ss))
  reports.append(dict(gimbal_pitch_rad=pitch,center_offset_sign=center_sign,outward_model_pitch_deg=np.degrees(board_pitch),pairs=len(scores),median_nearest_center_px=float(np.median(errors)),median_best_iou=float(np.median(scores)),anchor_pitch_median_deg=float(np.median(angles))))
(out/'geometry_diagnostics.json').write_text(json.dumps(dict(note='Exploratory sweeps on every tenth cached frame, optimistic independent best matches; same-video diagnostics, not calibrated ground truth. Defaults unchanged.',candidates=reports),indent=2))
print(json.dumps(reports,indent=2))
