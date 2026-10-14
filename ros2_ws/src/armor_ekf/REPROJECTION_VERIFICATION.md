# 四装甲板重投影：实现与验证

**更新：** 跳框问题已分析并修复，现行结果请先看 [姿态选解与跳框修复](POSE_SELECTION_VERIFICATION.md)。下文保留初版实现和对照数据。

日期：2026-10-09。已实现考核要求的“由一块板推算四块板，再投影到图像”，完成构建、4 项 CTest、完整缓存回放、500 帧直接推理及 ROS/RViz 验证。数学与消息验证通过；PDF 给定外参下的实拍轮廓吻合度较差，需要实际相机/云台外参才能进一步确认物理准确度。

## 实现与坐标约定

技术栈沿用 C++17、Eigen、OpenCV、ROS 2 Lyrical、ONNX Runtime；不需要重新训练或导出 YOLO 模型。`reprojector.hpp/.cpp` 与原检测、PnP、世界系 EKF 共用配置。

PnP 装甲板局部系为 X 向外法向、Y 向左、Z 向上，四角顺序 TL、BL、BR、TR。PDF 位置公式 `p_i = center - r [cos(yaw_pdf_i),sin(yaw_pdf_i),0]` 的 Yaw 指向中心，因此：

```text
yaw_pdf_0 = wrap(yaw_pnp_0 + π)
center = p_0 + r [cos(yaw_pdf_0), sin(yaw_pdf_0), 0]
yaw_pdf_i = wrap(yaw_pdf_0 + iπ/2)
p_i = center - r [cos(yaw_pdf_i), sin(yaw_pdf_i), 0]
R_out_i = Rz(yaw_pdf_i) Ry(pitch_pdf) Rx(roll_pdf) Rz(π)
```

最后一个 Rz(π) 转换整个装甲板坐标架，不能只改 Yaw。PDF 的 Pitch=+15°、Roll=0 在现有外法向系下对应 Pitch=−15°。半径为 0.25 m、四块板共享中心高度。P0 保留测量的完整姿态，P1–P3 使用默认模型姿态；中心计算使用原始测量 Yaw，以保持 Pose 与几何一致。

每块板的局部四角先变换到 odom，再经 `T_odom_camera_optical_frame⁻¹` 转回相机，用相同 K、D 经 `cv::projectPoints` 投影。检测丢失时不沿用旧模型；ROS 发布空 PoseArray 和 DELETE Marker。正深度检查、有限值检查和边缘裁剪避免非法图形。

橙色 P0 是测量板；青色 P1–P3 是模型板；红色十字/球是机器人中心。板编号相对于当前选中板，不是永久物理 ID。图上的 `in image` 只表示投影轮廓与图像相交；未建模遮挡。四个板只要有正深度且投影落入图像就都会画出，背面板标记 `model back`；四块世界系 Pose 也都会发布。

## 运行

在资料根目录执行。`reprojection.launch.py` 由单个 `armor_ekf_node` 同时发布输入图像并执行识别、PnP、滤波和重投影：

```bash
bash ros2_ws/build_ekf.sh
source /home/seb/armor_ekf_runtime/install/setup.bash
ros2 launch armor_ekf reprojection.launch.py \
  model_path:="$PWD/01_神经网络/yolo_pose/final_model/best.onnx" \
  video_path:="$PWD/00_题目原件/videos/test.mp4" fps:=30 loop:=true
```

另一个终端加载相同环境后：

```bash
rviz2 -d ros2_ws/rviz/armor_reprojection.rviz
```

ROS 输出：

- `/armor/reproject/image_raw`：标注图，原图时间戳与 camera_optical_frame。
- `/armor/reproject/poses`：odom 下的 4 个 Pose，顺序 P0–P3；漏检为空。
- `/armor/reproject/markers`：4 块板及中心，共 5 个 Marker；漏检全部 DELETE。
- `/armor/world_yaw` 与 `/armor/world_markers`：原世界系观测、滤波及测量板输出。

参数：`enable_reprojection`、`robot_radius`、`model_pitch`、`model_roll`；角度均为弧度。外参 xyz/rpy 参数同时作用于计算和 TF。单独的 `world.launch.py` 默认不开启重投影。

完整缓存回放：如果 `ros2_ws/output/tracker_cache/detections.csv` 不存在，先按 [EKF README](README.md) 的命令从 Release 输入视频重新生成检测缓存。之后运行：

```bash
ros2 run armor_ekf armor_ekf_replay \
  --model "$PWD/01_神经网络/yolo_pose/final_model/best.onnx" \
  --detections "$PWD/ros2_ws/output/tracker_cache/detections.csv" \
  --video "$PWD/00_题目原件/videos/test.mp4" \
  --output-dir "$PWD/ros2_ws/output/reprojection" --reproject
```

`armor_ekf_offline` 支持同样的 `--reproject`，从视频直接推理；`--robot-radius`、`--model-pitch-rad`、`--model-roll-rad`、`--gimbal-pitch-rad` 可覆盖几何。CSV 每个有效帧输出 4 行，含中心、PDF Yaw、世界/相机 Pose、像素四角、正深度、图像内和朝向标志；漏检输出 plate=−1 的无效行。

## 验证结果

- 4/4 CTest 通过。新增测试覆盖已知中心、不同锚板反推、90°间隔、完整坐标架转换、独立径向/切向畸变公式、PnP 回投影、漏检、负深度、非有限值、无效半径和图像裁剪。
- 完整回放 3150 帧：2280 帧生成模型，870 帧不生成旧模型。默认配置的 `yaw.csv` 与此前世界系版本逐字段一致。
- 世界/相机 Pose 的独立变换误差最大约 2.1×10⁻¹⁵；独立角点投影最大差约 3.1×10⁻¹¹ px；半径误差约 8.3×10⁻¹⁷ m。这些是数值一致性指标，不是物理测量精度。
- P0 相对检测四角的 RMS 误差中位数 0.348 px、95 分位 0.957 px，与既有 PnP 误差一致。
- 500 帧直接 ONNX 推理的世界姿态/滤波及完整投影 CSV 与同帧缓存回放一致。
- 实际 ROS 检查 300 条输出：290 条有效、10 条漏检，300 条源时间戳匹配；290 次四板位置/姿态检查、10 次清除检查通过；TF、世界 Pose/PnP、Marker、标注图均通过。软件渲染 RViz 同时运行，视频输入 60 Hz，观察到算法输出约 13.1 Hz；这不是原视频 100 Hz 实时性能证明。
- RViz 实际截图已检查，图像、TF、四板及中心正常显示。

验证脚本：`tools/verify_reprojection_replay.py`、`tools/verify_reprojection_ros.py`。可再生成的运行输出写入被忽略的 `ros2_ws/output/`；最终验证数据和代表帧整理在 [`06_重投影/results/`](../../../06_重投影/results/)，完整视频 `four_plate_reprojection.mp4` 由你另行提供，不纳入 Git。

## 实拍偏差与外参对照

PDF 明确给出 `odom→gimbal` Pitch=1.34 **rad**，默认配置完整保留该值。它使当前视频测量板的世界系 Pitch 中位数约 −63°，与模型板的 −15°明显不同。

对同帧其它检测板做一对一最大 IoU 分配，IoU 相同时按中心距离打破平局，共得到 959 对：

| 配置 | 其它板中心偏差中位数 | 轮廓 IoU 中位数 |
| --- | ---: | ---: |
| PDF 默认，云台 Pitch=1.34 rad | 139.6 px | 0.000 |
| 实拍对照，云台 Pitch=0 rad | 17.7 px | 0.488 |

对照版保持相机内参、畸变、0.25 m 半径及 PDF Pitch/Roll，只改云台俯仰。使用同一模型、同一缓存，偏差显著降低；这支持外参与实拍姿态不匹配的判断。另做每十帧采样对照：反转中心方向或只反转模型 Pitch 没有解决默认外参下的偏差，详见 `geometry_diagnostics.json`，生成脚本为 `tools/diagnose_reprojection_geometry.py`。

0 rad 是诊断配置，**不是标定结论**。部分帧仍有明显错误；单板平面 PnP 的姿态歧义、关键点误差、半径/高度模型及遮挡都可能影响结果。其它检测板没有永久物理 ID、中心或姿态真值，因此 IoU 对照不能证明绝对定位精度，也不能凭该实验完全排除模型误差。

对照配置为 `config/reprojection_video_diagnostic.yaml`：

```bash
ros2 launch armor_ekf reprojection.launch.py \
  model_path:="$PWD/01_神经网络/yolo_pose/final_model/best.onnx" \
  video_path:="$PWD/00_题目原件/videos/test.mp4" \
  reprojection_config:="$PWD/ros2_ws/src/armor_ekf/config/reprojection_video_diagnostic.yaml"
```

离线/回放在上述默认命令追加 `--gimbal-pitch-rad 0`，并将输出目录改为 `ros2_ws/output/reprojection/video_diagnostic`。改变外参也会改变世界 Yaw 和滤波结果，所以对照版不要求与默认世界系 CSV 一致。

## 查看成果

- `test.mp4` 由你另行提供，放在 `00_题目原件/videos/`。
- [默认验证数据](../../../06_重投影/results/validation.json)
- [对照验证数据](../../../06_重投影/results/initial_comparison_validation.json)
- [ROS 验证数据](../../../06_重投影/results/pose_selection_ros_report.json)
- [RViz 截图](../../../06_重投影/results/rviz_reprojection.png)

工程功能已完成。姿态歧义的多板约束见上方修复报告；实际相机到世界/云台的姿态仍需标定，不能把当前诊断参数当成真实外参直接用于机器人。
