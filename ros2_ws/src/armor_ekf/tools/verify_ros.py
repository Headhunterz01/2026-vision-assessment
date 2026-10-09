"""Verify the actual ROS message and annotated-image outputs of the EKF pipeline."""
import argparse
import json
import math
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from armor_interfaces.msg import YawEstimate

cli = argparse.ArgumentParser()
cli.add_argument('--output-dir', required=True)
cli.add_argument('--count', type=int, default=100)
args = cli.parse_args()
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
rclpy.init()
node = Node('armor_ekf_verifier')
received = []
image_count = 0
bridge = CvBridge()
last_stamp = -math.inf
last_visual_id = 0


def receive(msg):
    global last_stamp, last_visual_id
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
    assert stamp > last_stamp, 'Output timestamps did not increase'
    assert msg.header.frame_id == 'camera_optical_frame'
    last_stamp = stamp
    if msg.detection_valid:
        assert msg.visual_track_id > 0, 'Motion association did not publish visual identity'
        assert math.isfinite(msg.observed_yaw)
        assert -math.pi <= msg.observed_yaw <= math.pi
        assert 0.5 <= msg.detection_confidence <= 1
        assert 0 <= msg.class_id <= 12
        assert 0 <= msg.reprojection_error <= 3
        assert msg.pose.position.z > 0
        for p in msg.corners:
            assert 0 <= p.x < 1440 and 0 <= p.y < 1080
        q = msg.pose.orientation
        assert abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1) < 1e-6
        nx = 1-2*(q.y*q.y+q.z*q.z)
        nz = 2*(q.x*q.z-q.y*q.w)
        pose_yaw = math.atan2(nx, -nz)
        difference = math.atan2(math.sin(pose_yaw-msg.observed_yaw), math.cos(pose_yaw-msg.observed_yaw))
        assert abs(difference) < 1e-6, 'Pose and yaw disagree'
    else:
        assert math.isnan(msg.observed_yaw) and not msg.measurement_used
    if msg.filter_initialized:
        assert all(math.isfinite(v) for v in [msg.filtered_yaw, msg.yaw_rate, msg.yaw_variance])
        assert msg.yaw_variance >= 0
    if msg.target_reset:
        assert msg.measurement_used and abs(msg.filtered_yaw-msg.observed_yaw) < 1e-9
    if msg.target_switched:
        assert msg.target_reset and msg.detection_valid and msg.visual_track_id > 0
        if last_visual_id > 0:
            assert msg.visual_track_id != last_visual_id, 'Switch flag reused the same visual track'
    last_visual_id = msg.visual_track_id
    received.append((stamp, msg.detection_valid, msg.measurement_used, msg.track_id,
                     msg.visual_track_id, msg.target_switched, msg.target_reset))


def receive_image(msg):
    global image_count
    assert (msg.width, msg.height, msg.encoding) == (1440, 1080, 'bgr8')
    frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
    image_count += 1
    if image_count == 1:
        assert cv2.imwrite(str(output/'ros_annotated.png'), frame)


subscription = node.create_subscription(YawEstimate, '/armor/yaw', receive, 10)
images = node.create_subscription(Image, '/armor/ekf/image_raw', receive_image, 1)
try:
    deadline = time.monotonic()+45
    while (len(received) < args.count or image_count < 1) and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
    assert len(received) >= args.count, f'Only received {len(received)} yaw messages'
    assert image_count >= 1, 'No annotated image received'
    assert any(row[1] for row in received), 'No valid detections in live run'
    report = {
        'messages': len(received), 'detections': sum(row[1] for row in received),
        'measurements_used': sum(row[2] for row in received), 'annotated_images': image_count,
        'tracks': len({row[3] for row in received}),
        'visual_tracks': len({row[4] for row in received if row[4]}),
        'visual_switch_events': sum(row[5] for row in received),
        'filter_reset_events': sum(row[6] for row in received),
        'observed_output_hz': (len(received)-1)/(received[-1][0]-received[0][0]),
        'checks': 'timestamps, frame, corners, pose/quaternion, yaw, covariance, visual identity/switch/reset, annotated image',
    }
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
finally:
    node.destroy_node()
    rclpy.shutdown()
