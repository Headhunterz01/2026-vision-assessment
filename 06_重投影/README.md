# 考核六：四装甲板模型与图像重投影

重投影与前两项共用 [`armor_ekf` ROS 包](../ros2_ws/src/armor_ekf/)。关键实现为 [`reprojector.hpp`](../ros2_ws/src/armor_ekf/include/armor_ekf/reprojector.hpp) 和 [`reprojector.cpp`](../ros2_ws/src/armor_ekf/src/reprojector.cpp)；参数、launch 和测试在共享包的 `config/`、`launch/`、`test/` 中。

- 最终回放：[four_plate_reprojection.mp4](https://github.com/Headhunterz01/2026-vision-assessment/releases/tag/Video)。
- 保留的结果图：[代表帧 978](results/frame_0978.png)、[代表帧 906](results/frame_0906.png)、[RViz2 画面](results/rviz_reprojection.png)。
- 数值验证：[validation.json](results/validation.json)、[重投影 CSV](results/reprojection.csv)、[姿态候选诊断](results/pose_diagnosis.json)、[初版对照验证](results/initial_comparison_validation.json)。
- 实现原理与限制：[实现验证报告](../ros2_ws/src/armor_ekf/REPROJECTION_VERIFICATION.md)；[跳框和姿态选解修复](../ros2_ws/src/armor_ekf/POSE_SELECTION_VERIFICATION.md)。

最终回放 3,150 个采样时刻，其中 2,280 帧生成四板模型、870 帧因有效测量缺失而清除模型。独立坐标变换与投影数值检查通过；实拍轮廓仍受平面 PnP 歧义、相机外参和遮挡模型影响，视频中没有物理板位姿真值。输入视频及最终视频均通过 Release 附件清单获取。
