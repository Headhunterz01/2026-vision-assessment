#!/usr/bin/env python3
"""Publish the independent NN + four-board reprojection pipeline in ROS 2."""
import argparse
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray, TransformStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from reproject import Detector, Estimator, Geometry, OBJECT, ROOT, draw, ry


class ReprojectionNode(Node):
    def __init__(self, args):
        super().__init__('armor_reprojection_redo')
        self.args = args
        self.geometry = Geometry(0. if args.profile == 'video' else 1.34)
        self.estimator = Estimator(self.geometry)
        self.detector = Detector(args.model, threads=args.threads)
        self.capture = cv2.VideoCapture(str(args.video))
        if not self.capture.isOpened():
            raise RuntimeError(f'Cannot open video: {args.video}')
        if (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (1440, 1080):
            raise ValueError('Video must match the 1440x1080 calibration')
        self.source_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if not self.source_fps > 0:
            raise ValueError('Invalid source FPS')
        self.bridge = CvBridge()
        self.input_pub = self.create_publisher(Image, '/armor/redo/source', 1)
        self.image_pub = self.create_publisher(Image, '/armor/redo/image_raw', 1)
        self.pose_pub = self.create_publisher(PoseArray, '/armor/redo/poses', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/armor/redo/markers', 10)
        self.tf = StaticTransformBroadcaster(self)
        self.publish_tf()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        self.frame = args.start_frame - 1
        self.processed = self.valid = 0
        self.finished = False
        self.timer = self.create_timer(1. / args.fps, self.tick)
        self.get_logger().info(f'NN model: {args.model}; profile={args.profile}; output=/armor/redo/image_raw')

    def publish_tf(self):
        camera_link_from_optical = np.array([[0, 0, 1.], [-1., 0, 0], [0, -1., 0]])
        edges = [('odom', 'gimbal_link', ry(self.geometry.gimbal_pitch), [0, 0, 0]),
                 ('gimbal_link', 'camera_link', ry(.091), [.0461892, 0, -.0603874]),
                 ('camera_link', 'camera_optical_frame', camera_link_from_optical, [0, 0, 0])]
        transforms = []
        for parent, child, rotation, xyz in edges:
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id, tf.child_frame_id = parent, child
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, xyz)
            q = Rotation.from_matrix(rotation).as_quat()
            tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
            transforms.append(tf)
        self.tf.sendTransform(transforms)

    def tick(self):
        while True:
            if not self.capture.grab():
                if self.args.loop:
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.estimator = Estimator(self.geometry)
                    self.frame = -1
                    return
                self.finished = True
                return
            self.frame += 1
            if self.frame % self.args.stride == 0:
                break
        ok, image = self.capture.retrieve()
        if not ok:
            raise RuntimeError('Video decode failed')
        detections = self.detector(image)
        fit, status = self.estimator.estimate(detections, self.frame / self.source_fps)
        stamp = self.get_clock().now().to_msg()
        for publisher, pixels in [(self.input_pub, image),
                                  (self.image_pub, draw(image, detections, fit, status, self.args.debug))]:
            message = self.bridge.cv2_to_imgmsg(pixels, encoding='bgr8')
            message.header.stamp, message.header.frame_id = stamp, 'camera_optical_frame'
            publisher.publish(message)
        poses = PoseArray()
        poses.header.stamp, poses.header.frame_id = stamp, 'odom'
        markers = MarkerArray()
        for i in range(5):
            marker = Marker()
            marker.header = poses.header
            marker.ns, marker.id = 'four_armors_redo', i
            marker.action = Marker.ADD if fit is not None else Marker.DELETE
            marker.pose.orientation.w = 1.
            marker.color.a = 1.
            if fit is not None and i < 4:
                plate = fit['plates'][i]
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = map(float, plate['position'])
                q = Rotation.from_matrix(plate['rotation']).as_quat()
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, q)
                poses.poses.append(pose)
                marker.type = Marker.LINE_STRIP
                marker.scale.x = .003
                marker.color.g = marker.color.b = 1.
                corners = OBJECT @ plate['rotation'].T + plate['position']
                for p in np.vstack([corners, corners[0]]):
                    marker.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            elif fit is not None:
                marker.type = Marker.SPHERE
                marker.scale.x = marker.scale.y = marker.scale.z = .025
                marker.color.r = 1.
                marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, fit['state'][:3])
            markers.markers.append(marker)
        self.pose_pub.publish(poses)
        self.marker_pub.publish(markers)
        self.processed += 1
        self.valid += fit is not None
        if self.args.max_frames and self.processed >= self.args.max_frames:
            self.finished = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(ROOT / '01_神经网络/yolo_pose/final_model/best.onnx'))
    parser.add_argument('--video', default=str(ROOT / '00_题目原件/videos/test.mp4'))
    parser.add_argument('--profile', choices=['video', 'pdf'], default='video')
    parser.add_argument('--fps', type=float, default=30.)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0 or args.stride < 1 or args.threads < 1 or args.max_frames < 0 or args.start_frame < 0:
        parser.error('Invalid fps, stride, threads or max-frames')
    rclpy.init()
    node = None
    try:
        node = ReprojectionNode(args)
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.get_logger().info(f'processed={node.processed}, valid_models={node.valid}')
            node.capture.release()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
