# 考核四：装甲板 Yaw 扩展卡尔曼滤波

本项核心代码与后续两项共用 ROS 包，唯一源码在 [`ros2_ws/src/armor_ekf/`](../ros2_ws/src/armor_ekf/)；包级构建和复现说明见 [`armor_ekf/README.md`](../ros2_ws/src/armor_ekf/README.md)。

- 检测、PnP 和关联：[pipeline.cpp](../ros2_ws/src/armor_ekf/src/pipeline.cpp)、[motion_tracker.hpp](../ros2_ws/src/armor_ekf/include/armor_ekf/motion_tracker.hpp)
- Yaw EKF：[yaw_ekf.hpp](../ros2_ws/src/armor_ekf/include/armor_ekf/yaw_ekf.hpp)
- 最终标注回放：[ekf_tracking_final.mp4](https://github.com/Headhunterz01/2026-vision-assessment/releases/tag/Video)；结果图：[Yaw 曲线](results/ekf_yaw_plot.png)、[代表帧](results/ekf_tracking_sample.png)。
- 指标、逐帧数据和对照分析：[metrics.json](results/metrics.json)、[yaw.csv](results/yaw.csv)、[tracking_comparison.json](results/tracking_comparison.json)、[对照说明](results/TRACKING_COMPARISON.md)。
- 详细算法验证：[VERIFICATION.md](../ros2_ws/src/armor_ekf/VERIFICATION.md)。

在同一检测缓存上，图像目标观测覆盖率由旧关联的 51.08% 提升至运动关联的 72.38%；2,280 个有效姿态中 2,263 个进入滤波更新。视频没有 Yaw 真值或永久装甲板身份标签，因此覆盖率和角度平滑程度不能直接解释为绝对角度精度。输入视频：[test.mp4](https://github.com/Headhunterz01/2026-vision-assessment/releases/download/2026-season-submission/test.mp4)。
