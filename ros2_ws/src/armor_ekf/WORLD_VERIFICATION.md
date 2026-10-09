# 坐标变换考核：实现与验证

2026-10-09 完成。实现满足考核的固定外参场景：PnP完整姿态从 camera_optical_frame 转至 odom，提取世界系观测 Yaw，使用现有 EKF 滤波，通过 ROS 2 自定义消息同时发布观测和滤波角度。

## 实现

- `include/armor_ekf/coordinate_transform.hpp` 集中定义三段考核外参，采用 `p_parent = R * p_child + t`、`R = Rz(yaw) Ry(pitch) Rx(roll)`。位置同时经过旋转和平移，姿态经过完整旋转矩阵左乘。
- 世界Yaw由变换后的装甲板向外法向计算 `atan2(n_y,n_x)`，并在 EKF 更新和 IPPE 候选连续性判断之前完成转换。角度不是直接相加。
- 世界法向水平投影小于 `min_world_horizontal_normal`（默认0.001）的观测被拒绝，记录 `world_yaw_rejects`。PnP可见性、深度和像素重投影仍在相机系检查。
- ROS节点通过 `tf2_ros::StaticTransformBroadcaster` 发布 `odom → gimbal_link → camera_link → camera_optical_frame`。计算与TF广播使用同一个已解析的 `FrameConfig`，外参默认值只有一处。
- `world.launch.py` 开启世界模式；原 `ekf.launch.py` 保持相机模式。`YawEstimate.msg` 明确了 `header.frame_id` 的语义。世界消息 pose、观测Yaw、滤波Yaw均为odom系，时间戳为源图像时间戳；corners仍为图像像素。
- 发布 `/armor/world_yaw`、`/armor/world/image_raw`、`/armor/world_markers`。图像header仍为camera_optical_frame；RViz固定帧为odom，橙色薄板表示观测完整姿态，绿色水平箭头表示滤波Yaw。
- 离线与缓存回放使用同一C++转换/PnP/EKF代码。CSV中的 `x,y,z,qx,qy,qz,qw` 属于 `frame_id`；`camera_*` 字段保留相机姿态用于交叉验证。无有效检测时应忽略pose字段。

## 验证结果

构建成功，3项CTest全部通过：原EKF/PnP、图像跟踪、新增世界坐标变换。

新增测试包括 TF2与Eigen的完整链一致性、光学轴映射、平移顺序、位置往返、PnP竖直法向的Yaw拒绝、7组已知世界姿态的合成投影/PnP恢复，以及100帧世界系EKF跨±π、短时丢失预测、超时失效和时间戳回退重置。

- 合成PnP最大Yaw误差 `6.31459e-7 rad`，最大位置误差 `8.55375e-7 m`。这是合成输入的数值一致性结果，不代表真实视频精度。
- 完整缓存回放3150帧（源视频约63秒，stride=2），2280帧有选中世界姿态，覆盖率72.38%。2116帧进入最终滤波更新/初始化，占有效观测92.81%。
- 全部2280个有效世界姿态对独立Python矩阵基准验证：位置分量最大误差 `9.85e-12 m`，旋转分量误差 `3.48e-12`，Yaw环绕误差 `9.47e-12 rad`（受CSV写出精度影响）。
- 相机模式3150帧对原已完成EKF结果回归通过；选中检测、跟踪ID、滤波状态和数值一致。世界模式与相机模式的选中检测及该视频的PnP候选解也一致。
- 世界模式直接模型推理前750帧与同输入缓存回放结果一致，覆盖旋转开始阶段。
- 实际ROS订阅300条世界Yaw消息：234条有效观测，227条测量进入滤波，300张标注图像、300次TF检查、300组Marker、234次Marker姿态检查；300条输出时间戳均匹配原图像。世界姿态逆变换与源角点的两个PnP候选之一匹配，最大分量差 `4.76e-7`。
- 该ROS测试包含高频视频输入、可靠源图像审计和软件渲染RViz，观测输出约11.42Hz；这是本次测试条件下的速率，不是100Hz实时处理保证。节点保留最新图像以避免推理积压。
- RViz在Xvfb中实际显示标注图像、TF链、世界姿态与滤波朝向，截图确认Global Status、Image、TF为OK。

## 视频滤波现象与局限

本次世界模式保留已有EKF参数与异常观测门限。世界Yaw偶发大跳跃增多：同一视觉ID相邻有效观测中，角度差超过0.4rad的次数从相机模式21次变为世界模式86次，但中位绝对角度差从0.07894rad降为0.03441rad。两种Yaw定义对同一个三维姿态的变化有不同敏感性，不能用单一平滑指标比较绝对精度。

世界模式184次观测先被EKF门限拒绝，其中20次在持续拒绝后触发已有重初始化逻辑，因此最终未使用的观测是164次；总重置237次 = 首次初始化1次 + 视觉目标切换216次 + 持续异常重初始化20次。原相机模式拒绝17次、重置217次。没有修改检测或门限来消除这些现象。

世界法向水平投影最小约0.164，16条有效观测低于0.2，本视频没有触发默认0.001的竖直法向拒绝。准确竖直法向的角度拒绝通过数学单测；合成PnP在0.01阈值下验证拒绝路径，容纳float32角点带来的近正视姿态数值扰动。当前数据缺少真实世界Yaw与物理装甲板身份标签，无法断言大跳跃分别来自PnP噪声、视觉身份关联或真实运动，也无法确认考核外参与实际拍摄装置的物理标定完全一致。

严格采用考核给定固定外参。该模式不订阅实时云台姿态；真实云台运动时，需要动态图及按图像时间戳查询TF。这属于未来硬件接入。后续“重投影”考核未在此次实现中开展。

## 运行和复现

从 GitHub Release 下载 `test.mp4`，放回 `00_题目原件/videos/`；从资料根目录运行：

```bash
bash ros2_ws/build_ekf.sh
source /home/seb/armor_ekf_runtime/install/setup.bash
ros2 launch armor_ekf world.launch.py \
  model_path:="$PWD/01_神经网络/yolo_pose/final_model/best.onnx" \
  video_path:="$PWD/00_题目原件/videos/test.mp4" fps:=30 loop:=false
```

另一个已加载环境的终端：

```bash
ros2 topic echo /armor/world_yaw --once
rviz2 -d ros2_ws/rviz/armor_world.rviz
python3 ros2_ws/src/armor_ekf/tools/verify_world_ros.py \
  --output-dir ros2_ws/output/coordinate_transform/ros --count 300
```

源图像时间戳审计需在视频发布前启动验证器；完整的300条验证使用`fps:=0`播放源视频。30FPS播放下源视频被放慢，验证器50秒截止前可能不能收满300条；可使用`--count 100`。

```bash
ctest --test-dir /home/seb/armor_ekf_runtime/build/armor_ekf --output-on-failure
ros2 run armor_ekf armor_ekf_replay \
  --model "$PWD/01_神经网络/yolo_pose/final_model/best.onnx" \
  --detections "$PWD/ros2_ws/output/tracker_cache/detections.csv" \
  --yaw-frame odom --video "$PWD/00_题目原件/videos/test.mp4" \
  --output-dir "$PWD/ros2_ws/output/coordinate_transform/world"
python_env/.venv/bin/python ros2_ws/src/armor_ekf/tools/plot_yaw.py \
  ros2_ws/output/coordinate_transform/world
python_env/.venv/bin/python ros2_ws/src/armor_ekf/tools/verify_world_replay.py \
  --world-dir ros2_ws/output/coordinate_transform/world \
  --camera-dir ros2_ws/output/coordinate_transform/camera_regression \
  --old-camera-dir ros2_ws/output/ekf_tracking_final \
  --direct-world-dir ros2_ws/output/coordinate_transform/direct_world
```

若从干净 checkout 复现，先按 [EKF 报告](README.md) 的“检测缓存”命令生成 `ros2_ws/output/tracker_cache/detections.csv`。相机回归和直接推理目录按本段命令生成；它们均为可重建中间结果，不纳入发布仓库。

世界外参可在 `config/world.yaml` 中覆盖 `odom_gimbal_xyz/rpy`、`gimbal_camera_xyz/rpy`、`camera_optical_xyz/rpy`。默认值集中在coordinate_transform.hpp；ROS计算和TF会一起变化。离线CLI的`--yaw-frame odom`使用考核默认外参，不读取ROS YAML；验证器基准也固定为考核外参。

## 输出

`05_坐标变换/results/` 保存提交版的曲线、代表帧、CSV、指标、验证数据和 ROS 报告。完整 `world_yaw.mp4` 是 Release 附件。本机运行命令把可再生成的缓存写入被忽略的 `ros2_ws/output/`。

- `world_yaw.mp4`、`world_yaw_plot.png`、`world_yaw_sample.png`、`yaw.csv`、`metrics.json`、`replay_validation.json`、`filter_diagnostics.json`。
- `camera_regression/yaw.csv`、`direct_world/yaw.csv`及直接推理检测缓存。
- `ros/report.json`、`ros_annotated.png`、`rviz_world.png`、运行日志。
- `build.log`、`tests.log`。

构建日志与运行过程中资源路径曾使用旧的嵌套final_model目录；资产目录整理后已更新当前文档和集成测试入口为 `01_神经网络/yolo_pose/final_model/best.onnx`。
