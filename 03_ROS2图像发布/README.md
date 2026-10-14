# 考核三：ROS 2 图像发布与 RViz2

图像发布器的唯一源码位于共享工作区 [`ros2_ws/src/armor_video_publisher/`](../ros2_ws/src/armor_video_publisher/)。工作区配置、构建及运行命令见 [`ros2_ws/README.md`](../ros2_ws/README.md)。

- 话题验证工具：[`results/verify_image_topic.py`](results/verify_image_topic.py)
- RViz2 画面：[`results/rviz_verified.png`](results/rviz_verified.png)
- 发布器验证截图与报告：[`results/ros_annotated.png`](results/ros_annotated.png)、[`results/report.json`](results/report.json)
- 两个原视频的话题接收日志：[`results/verify_armors.log`](results/verify_armors.log)、[`results/verify_test.log`](results/verify_test.log)。将 `test.mp4` 或 `armors_yolo_test.mp4` 放在 `00_题目原件/videos/`；视频由你另行提供。

节点将 OpenCV 视频帧发布到 `/armor/image_raw`，消息编码为 `bgr8`、frame_id 为 `camera_optical_frame`。验证检查图像尺寸、数据长度、时间戳和约 30 Hz 频率。
