from __future__ import annotations

from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


class ArmorVideoPublisher(Node):
    def __init__(self) -> None:
        super().__init__("armor_video_publisher")

        self.declare_parameter("video_path", "")
        self.declare_parameter("topic_name", "/armor/image_raw")
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("loop", True)
        self.declare_parameter("fps", 0.0)

        video_path = str(self.get_parameter("video_path").value)
        if not video_path:
            raise ValueError("video_path must be provided")
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"video not found: {video_path}")

        self.topic_name = str(self.get_parameter("topic_name").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.loop_video = bool(self.get_parameter("loop").value)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")

        source_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        requested_fps = float(self.get_parameter("fps").value)
        self.fps = requested_fps if requested_fps > 0 else source_fps
        if self.fps <= 0:
            self.fps = 30.0

        self.publisher = self.create_publisher(Image, self.topic_name, 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(1.0 / self.fps, self.publish_frame)
        self.get_logger().info(
            f"publishing {video_path} to {self.topic_name} at {self.fps:.2f} FPS"
        )

    def publish_frame(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            if not self.loop_video:
                self.get_logger().info("video finished")
                self.destroy_timer(self.timer)
                return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()

        if not ok:
            self.get_logger().warning("failed to read video frame")
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)

    def destroy_node(self) -> bool:
        self.cap.release()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ArmorVideoPublisher()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"[armor_video_publisher] {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
