# 考核五：相机坐标系到 odom 坐标系变换

坐标变换与 EKF 共用 [`armor_ekf` ROS 包](../ros2_ws/src/armor_ekf/)。关键实现为 [`coordinate_transform.hpp`](../ros2_ws/src/armor_ekf/include/armor_ekf/coordinate_transform.hpp)，世界系参数、launch 和数学测试分别位于共享包的 `config/world.yaml`、`launch/world.launch.py`、`test/world_test.cpp`。

- 最终变换回放：[world_yaw.mp4](https://github.com/Headhunterz01/2026-vision-assessment/releases/download/2026-season-submission/world_yaw.mp4)；本机结果图：[Yaw 曲线](results/world_yaw_plot.png)、[代表帧](results/world_yaw_sample.png)、[RViz2 姿态](results/rviz_world.png)。
- 数值结果：[指标](results/metrics.json)、[逐帧 CSV](results/yaw.csv)、[滤波诊断](results/filter_diagnostics.json)、[回放验证](results/replay_validation.json)。
- 推导、运行命令、合成测试及实拍限制见[坐标变换验证报告](../ros2_ws/src/armor_ekf/WORLD_VERIFICATION.md)。

完整回放 3,150 个采样时刻，2,280 个有有效选中姿态；2,116 次观测进入滤波更新。变换实现由合成姿态及独立矩阵基准交叉验证。
