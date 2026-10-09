"""Check live odom messages, image timestamps, TF chain, markers and inverse PnP."""
import argparse
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray
from armor_interfaces.msg import YawEstimate
from tf2_ros import Buffer, TransformListener


def quat_rotation(q):
    x,y,z,w = q.x,q.y,q.z,q.w
    assert abs(x*x+y*y+z*z+w*w-1) < 1e-6
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def stamp_key(stamp):
    return stamp.sec*1_000_000_000+stamp.nanosec


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--output-dir',required=True)
    cli.add_argument('--count',type=int,default=300)
    args = cli.parse_args()
    folder = Path(args.output_dir); folder.mkdir(parents=True,exist_ok=True)
    baseline = Path(__file__).resolve().parents[3]/'preparation/coordinate_transform/geometry_baseline.json'
    expected = np.array(json.loads(baseline.read_text())['T_odom_camera_optical_frame'])
    K = np.array([[3571.1890927935297,0,683.5999418028833],
                  [0,3569.0234714126823,565.5838917436471],[0,0,1]])
    D = np.array([-0.029757126887136703,0.371503828585697,0.0011898367108863406,-0.0025516396408119418,0.0])
    w,h = 0.135/2,0.055/2
    object_points = np.array([[0,-w,h],[0,-w,-h],[0,w,-h],[0,w,h]])
    rclpy.init(); node = Node('armor_world_verifier')
    buffer = Buffer(); listener = TransformListener(buffer,node)
    bridge = CvBridge()
    received = []; source_stamps = set(); image_stamps = set(); marker_stamps = set(); marker_messages = {}
    states = {}; last_stamp = -1; last_visual = 0
    tf_checks = 0; max_pnp_error = 0.0

    def receive(msg):
        nonlocal last_stamp,last_visual,tf_checks,max_pnp_error
        stamp = stamp_key(msg.header.stamp)
        assert stamp > last_stamp and msg.header.frame_id == 'odom'
        last_stamp = stamp
        if msg.detection_valid:
            assert msg.visual_track_id > 0 and 0.5 <= msg.detection_confidence <= 1
            assert 0 <= msg.class_id <= 12 and 0 <= msg.reprojection_error <= 3
            r = quat_rotation(msg.pose.orientation)
            yaw = math.atan2(r[1,0],r[0,0])
            assert abs(math.atan2(math.sin(yaw-msg.observed_yaw),math.cos(yaw-msg.observed_yaw))) < 1e-6
            assert np.hypot(r[0,0],r[1,0]) >= 0.001
            t = np.array([msg.pose.position.x,msg.pose.position.y,msg.pose.position.z])
            camera_r = expected[:3,:3].T @ r
            camera_t = expected[:3,:3].T @ (t-expected[:3,3])
            assert camera_t[2] > 0 and np.dot(camera_r[:,0],camera_t) < 0
            corners = np.array([[p.x,p.y] for p in msg.corners])
            assert np.all((corners[:,0]>=0)&(corners[:,0]<1440)&(corners[:,1]>=0)&(corners[:,1]<1080))
            _,rvecs,tvecs,_ = cv2.solvePnPGeneric(object_points,corners,K,D,flags=cv2.SOLVEPNP_IPPE)
            errors = [max(float(np.max(np.abs(cv2.Rodrigues(rv)[0]-camera_r))),
                          float(np.max(np.abs(tv.reshape(3)-camera_t)))) for rv,tv in zip(rvecs,tvecs)]
            assert errors and min(errors) < 1e-5, 'Published world pose does not match either source-corner PnP solution'
            max_pnp_error = max(max_pnp_error,min(errors))
        else:
            assert math.isnan(msg.observed_yaw) and not msg.measurement_used
        if msg.filter_initialized:
            assert all(math.isfinite(v) for v in [msg.filtered_yaw,msg.yaw_rate,msg.yaw_variance])
            assert -math.pi <= msg.filtered_yaw <= math.pi and msg.yaw_variance >= 0
        if msg.target_reset:
            assert msg.measurement_used and abs(msg.filtered_yaw-msg.observed_yaw) < 1e-9
        if msg.target_switched:
            assert msg.target_reset and msg.detection_valid and msg.visual_track_id != last_visual
        last_visual = msg.visual_track_id
        if buffer.can_transform('odom','camera_optical_frame',Time.from_msg(msg.header.stamp)):
            tf = buffer.lookup_transform('odom','camera_optical_frame',Time.from_msg(msg.header.stamp)).transform
            assert np.max(np.abs(quat_rotation(tf.rotation)-expected[:3,:3])) < 1e-10
            assert np.max(np.abs(np.array([tf.translation.x,tf.translation.y,tf.translation.z])-expected[:3,3])) < 1e-10
            for parent,child in [('odom','gimbal_link'),('gimbal_link','camera_link'),('camera_link','camera_optical_frame')]:
                assert buffer.can_transform(parent,child,Time.from_msg(msg.header.stamp))
            tf_checks += 1
        received.append((stamp,msg.detection_valid,msg.measurement_used,msg.visual_track_id,msg.target_switched,msg.target_reset))
        states[stamp] = msg

    def image(msg):
        assert msg.header.frame_id == 'camera_optical_frame'
        assert (msg.width,msg.height,msg.encoding) == (1440,1080,'bgr8')
        image_stamps.add(stamp_key(msg.header.stamp))
        if len(image_stamps) == 1:
            assert cv2.imwrite(str(folder/'ros_annotated.png'),bridge.imgmsg_to_cv2(msg,'bgr8'))

    def markers(msg):
        assert len(msg.markers) == 2
        for marker in msg.markers:
            assert marker.header.frame_id == 'odom'
            if marker.action == marker.ADD:
                quat_rotation(marker.pose.orientation)
                assert min(marker.scale.x,marker.scale.y,marker.scale.z) > 0
        marker_stamps.add(stamp_key(msg.markers[0].header.stamp))
        marker_messages[stamp_key(msg.markers[0].header.stamp)] = msg

    subscriptions = [node.create_subscription(YawEstimate,'/armor/world_yaw',receive,10),
                     node.create_subscription(Image,'/armor/world/image_raw',image,1),
                     # This audit must retain inputs rather than intentionally drop them like inference.
                     node.create_subscription(Image,'/armor/image_raw',lambda m:source_stamps.add(stamp_key(m.header.stamp)),30),
                     node.create_subscription(MarkerArray,'/armor/world_markers',markers,10)]
    try:
        deadline = time.monotonic()+50
        while (len(received)<args.count or tf_checks==0 or len(marker_stamps)==0) and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=0.5)
        # Drain messages already queued before comparing source stamps.
        for _ in range(10): rclpy.spin_once(node,timeout_sec=0.02)
        assert len(received)>=args.count and tf_checks>0
        assert any(row[1] for row in received) and image_stamps and marker_stamps
        stamps = set(states)
        catchup_deadline = time.monotonic()+2
        while not stamps <= source_stamps and time.monotonic()<catchup_deadline:
            rclpy.spin_once(node,timeout_sec=0.05)
        assert stamps <= source_stamps, f'Source audit did not capture {len(stamps-source_stamps)} output timestamps'
        assert stamps & image_stamps and stamps & marker_stamps
        marker_pose_checks = 0
        for stamp in stamps & marker_stamps:
            observed = states[stamp]; cube,arrow = marker_messages[stamp].markers
            assert (cube.action == cube.ADD) == observed.detection_valid
            if observed.detection_valid:
                actual = np.array([cube.pose.position.x,cube.pose.position.y,cube.pose.position.z])
                expect = np.array([observed.pose.position.x,observed.pose.position.y,observed.pose.position.z])
                assert np.max(np.abs(actual-expect)) < 1e-12
                assert np.max(np.abs(quat_rotation(cube.pose.orientation)-quat_rotation(observed.pose.orientation))) < 1e-12
                heading = 2*math.atan2(arrow.pose.orientation.z,arrow.pose.orientation.w)
                assert abs(math.atan2(math.sin(heading-observed.filtered_yaw),math.cos(heading-observed.filtered_yaw))) < 1e-10
                marker_pose_checks += 1
        report = dict(status='PASS',messages=len(received),detections=sum(r[1] for r in received),
                      measurements_used=sum(r[2] for r in received),annotated_images=len(image_stamps),
                      visual_tracks=len({r[3] for r in received if r[3]}),visual_switch_events=sum(r[4] for r in received),
                      filter_reset_events=sum(r[5] for r in received),tf_checks=tf_checks,marker_arrays=len(marker_stamps),
                      marker_pose_checks=marker_pose_checks,
                      source_timestamp_matches=len(stamps & source_stamps),max_inverse_transform_pnp_component_error=max_pnp_error,
                      observed_output_hz=(len(received)-1)/((received[-1][0]-received[0][0])/1e9),
                      checks='odom header, original image stamps, live TF2 chain vs baseline, world pose/yaw, inverse transform vs both PnP candidates, covariance, resets, markers, optical-frame images')
        (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
