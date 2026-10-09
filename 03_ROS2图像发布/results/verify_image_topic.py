"""Receive real ROS images and validate payload, timestamps, and frame IDs."""
import time

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

rclpy.init()
node = Node('assessment_image_verifier')
bridge = CvBridge()
stamps = []


def receive(msg):
    assert msg.encoding == 'bgr8', msg.encoding
    assert (msg.width, msg.height) == (1440, 1080), (msg.width, msg.height)
    assert msg.header.frame_id == 'camera_optical_frame', msg.header.frame_id
    assert msg.step == msg.width * 3
    assert len(msg.data) == msg.step * msg.height
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    assert frame.shape == (1080, 1440, 3)
    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    assert not stamps or stamp > stamps[-1], 'timestamp did not increase'
    stamps.append(stamp)


sub = node.create_subscription(Image, '/armor/image_raw', receive, 10)
try:
    deadline = time.monotonic() + 20
    while len(stamps) < 30 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
    assert len(stamps) >= 30, f'Only received {len(stamps)} images'
    fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    assert 20 <= fps <= 40, f'Unexpected publish rate: {fps}'
    print(f'PASS: {len(stamps)} images, {fps:.2f} Hz, bgr8 1440x1080, valid timestamps and frame ID')
finally:
    node.destroy_node()
    rclpy.shutdown()
