"""Receive actual ROS messages and check geometry, markers and image stamps."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-clear', action='store_true')
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node('verify_armor_reprojection_redo')
    poses, images, markers = {}, {}, {}
    def key(msg):
        return msg.header.stamp.sec, msg.header.stamp.nanosec
    def pose(msg):
        assert msg.header.frame_id == 'odom' and len(msg.poses) in [0, 4]
        if msg.poses:
            points = np.array([[p.position.x, p.position.y, p.position.z] for p in msg.poses])
            center = points.mean(0)
            np.testing.assert_allclose(np.linalg.norm(points - center, axis=1), .25, atol=1e-10)
            for p in msg.poses:
                q = p.orientation
                assert abs(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w - 1) < 1e-10
        poses[key(msg)] = len(msg.poses)
    def image(msg):
        assert msg.encoding == 'bgr8' and (msg.width, msg.height) == (1440, 1080)
        assert msg.header.frame_id == 'camera_optical_frame'
        images[key(msg)] = True
    def marker(msg):
        assert len(msg.markers) == 5
        markers[key(msg.markers[0])] = [m.action for m in msg.markers]
    subscriptions = [node.create_subscription(PoseArray, '/armor/redo/poses', pose, 10),
                     node.create_subscription(Image, '/armor/redo/image_raw', image, 1),
                     node.create_subscription(MarkerArray, '/armor/redo/markers', marker, 10)]
    start = time.monotonic()
    try:
        while time.monotonic() - start < 40 and len(poses.keys() & images.keys() & markers.keys()) < 20:
            rclpy.spin_once(node, timeout_sec=.2)
        common = poses.keys() & images.keys() & markers.keys()
        assert len(common) >= 20, f'Only received {len(common)} synchronized message sets'
        for stamp in common:
            expected = Marker.ADD if poses[stamp] else Marker.DELETE
            assert markers[stamp] == [expected] * 5
        cleared = sum(not poses[k] for k in common)
        if args.require_clear:
            assert cleared > 0, 'No empty pose / DELETE marker frame received'
        report = dict(status='PASS', synchronized_message_sets=len(common),
                      valid_four_pose_sets=sum(bool(poses[k]) for k in common),
                      empty_pose_and_delete_sets=cleared,
                      image_resolution=[1440, 1080], matched_image_pose_marker_stamps=True)
        output = Path(__file__).parent / 'output' / 'ros_validation.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
