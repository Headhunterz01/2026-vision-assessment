# 装甲板相机 Yaw EKF 考核

本阶段接入已训练的 YOLO26n Pose ONNX 模型，对 `test.mp4` 中的装甲板进行四角检测、PnP 姿态解算和 EKF 滤波，通过 ROS 2 自定义消息同时发布观测角与滤波角。

## 技术栈与文件

- C++17、ROS 2 Lyrical、ament_cmake。
- ONNX Runtime 1.30.0 CPU C++ SDK：推理已导出的模型。
- OpenCV 4.10：letterbox、平面 PnP、重投影验证、图像绘制及视频读写。
- Eigen 3.4：EKF 的矩阵计算。
- `armor_interfaces/msg/YawEstimate.msg`：自定义消息。
- `src/pipeline.cpp`：检测、姿态候选及目标关联。
- `include/armor_ekf/yaw_ekf.hpp`：角度滤波器。
- `include/armor_ekf/motion_tracker.hpp`：图像中心/速度 Kalman 预测与多目标一对一分配。
- `src/replay.cpp`：同一检测缓存的新旧关联回放。
- `src/node.cpp`：集成视频帧读取与原始图像发布、YOLO/PnP、EKF、自定义消息和标注图像发布；也支持订阅外部相机图像。
- `src/offline.cpp`：与 ROS 节点共用同一套核心代码的离线视频运行器。
- `config/ekf.yaml`：相机、尺寸、噪声、置信度及角点顺序配置。

## 构建

在考核资料根目录运行，使用系统 Python 和已安装的 ROS 环境：

```bash
bash ros2_ws/build_ekf.sh
source /opt/ros/lyrical/setup.bash
source "$HOME/armor_ekf_runtime/install/setup.bash"
```

构建脚本会下载官方 ONNX Runtime C++ SDK 到 `ros2_ws/deps/`。已安装的 ROS desktop 提供其余系统依赖。独立 venv 用于 ONNX 检查和结果绘图，运行 ROS 节点时先退出 venv。

ROS 消息生成器在此机器上读取含中文路径的中间文件时会失败，因此脚本创建 `~/armor_ekf_ws` 源码路径别名，并使用实际英文目录 `~/armor_ekf_runtime/{build,install,log}`。源码仍保存在原资料目录。`ARMOR_EKF_RUNTIME` 可以覆盖运行目录，覆盖时应使用英文实际路径。

## 一体化 ROS 2 节点运行

从资料根目录执行：

```bash
ros2 launch armor_ekf ekf.launch.py \
  model_path:="$PWD/01_神经网络/yolo_pose/final_model/best.onnx" \
  video_path:="$PWD/00_题目原件/videos/test.mp4" \
  fps:=20.0 loop:=true
```

同一个 `armor_ekf_node` 读取视频并发布 `/armor/image_raw`，然后对该帧直接执行 YOLO、PnP 和 EKF，发布自定义消息及标注图像，不需要启动第二个视频发布节点。`fps` 控制回放速度；`fps:=0.0` 使用视频原始帧率，`loop:=true` 可循环播放。ROS 中的角速度对应当前回放时间，降低回放帧率会降低其数值；离线程序使用原始视频时间。

只运行算法节点、接入已有图像发布器：

```bash
ros2 run armor_ekf armor_ekf_node --ros-args \
  --params-file ros2_ws/src/armor_ekf/config/ekf.yaml \
  -p model_path:="$PWD/01_神经网络/yolo_pose/final_model/best.onnx"
```

输出：

```bash
ros2 interface show armor_interfaces/msg/YawEstimate
ros2 topic echo /armor/yaw --once
rviz2 -d ros2_ws/rviz/armor_ekf.rviz
```

- `/armor/yaw`：观测角、滤波角、角速度、方差、状态、轨迹编号、姿态及角点。
- `/armor/ekf/image_raw`：角点、PnP 坐标轴和观测/滤波数值标注图像。有人订阅时才生成，节省计算量。

## 模型、尺寸和坐标约定

使用 `01_神经网络/yolo_pose/final_model/best.onnx`，SHA-256 为 `b6c20fa44427ef5f0dae115cf0f0ca13a26589a7983e12cab1785041711b5444`。训练目录中的另一个 ONNX 文件与它相同。

输入为 float32 RGB `[1,3,640,640]`，按比例缩放并用 114 居中填充，除以 255。输出 `[1,300,14]`，每行是 `x1,y1,x2,y2,score,class_id` 和四组 `x,y`。导出时已含 NMS；额外消除不同类别对同一装甲板的重复框。此实现校验该输出格式，不直接接受未包含 NMS 的另一种导出格式。

多帧推理可视化确认当前模型角点次序是左上、左下、右下、右上。配置 `corner_order` 可以调整模型索引到该次序的映射。相机内参与畸变系数来自考核 PDF；图像必须为校准对应的 1440×1080。

当前测试目标统一采用题目图示的 135×55 mm（0.135×0.055 m）。模型类别名 `armor_0..armor_12` 没提供类别语义和大小映射，所以不从类别推断尺寸。换用不同尺寸装甲板时需改配置；当前版本不处理同一视频中混合大小装甲板。

装甲板物体坐标系以中心为原点：X 是向外法向，Y 是机器人自身向左方向，Z 向上。四个物点为：

```text
左上  (0, -w/2, +h/2)
左下  (0, -w/2, -h/2)
右下  (0, +w/2, -h/2)
右上  (0, +w/2, +h/2)
```

PnP 返回此装甲板坐标系到 `camera_optical_frame` 的旋转和平移。相机光学系为 X 向右、Y 向下、Z 向前。设装甲板向外单位法向为 `n = R[:,0]`，本阶段相机水平朝向定义为：

```text
yaw_camera = atan2(n.x, -n.z)
```

正对相机时为 0，法向朝图像右侧偏转为正。这是相机视角下装甲板法向的水平朝向，不是光学系绕 Z 的面内旋转角，也不是 `odom` 中的地面 Yaw。后续坐标变换题应先变换整个姿态，再按世界系法向计算 `atan2(n_world.y,n_world.x)`。

平面 IPPE 返回多个姿态解；拒绝角点非凸、正深度不满足、背对相机、重投影误差超过阈值的结果。对于同一条视觉轨迹，重投影误差与最优解相差不超过 0.5 px 时优先选择与预测 Yaw 连续的候选；Yaw 不参与图像轨迹的身份分配。近正面平面姿态存在固有歧义，低重投影误差不能证明真实 Yaw 完全正确。

## EKF 和目标关联

状态为 `[yaw, yaw_rate]`，匀速预测的状态转移矩阵为 `[[1,dt],[0,1]]`。过程噪声为 `sigma_acc² * g*gᵀ`，其中 `g=[dt²/2,dt]`。

观测采用角度的非线性单位圆表示：`z=[cos(yaw_observed),sin(yaw_observed)]`，`h(x)=[cos(yaw),sin(yaw)]`，其 Jacobian 为 `[[-sin(yaw),0],[cos(yaw),0]]`。这是带非线性观测的 EKF。单位圆观测和归一化角度差处理 ±π 环绕，协方差用 Joseph 形式更新。

默认 `association_mode=motion`。为每块同时可见的装甲板建立图像轨迹，状态是中心像素坐标及速度 `[cx,cy,vx,vy]`，使用时间戳推进 Kalman 预测，再以中心预测距离、框尺寸和弱类别提示构造代价，通过最小总代价的一对一分配关联检测。类别不能唯一标识物理板，因此不按类别强制匹配。`association_mode=legacy` 保留此前的单目标最近中心实现，供对照。

首次选择置信度最高的有效目标。当前板短暂消失时保留预测，超过 `target_switch_timeout=0.08` 秒且已有另一条至少两次观测的可见轨迹时切换，并重新初始化 Yaw EKF；图像轨迹最多保留 `reset_timeout=0.3` 秒。不会把不同视觉轨迹的 Yaw 连续混入同一滤波段。连续五次异常 Yaw 仍可触发同一视觉轨迹的滤波重置，消息和 CSV 可以区分两种事件。

`visual_track_id` 是临时图像轨迹身份，不能证明板在长时间遮挡后仍保持永久身份；`track_id` 是滤波段编号，每次重新初始化递增。`target_switched` 标记换视觉轨迹，`target_reset` 标记任何滤波初始化/重置。标注视频为所有候选显示 V 编号、类别与置信度；无观测时橙色十字表示当前目标的图像中心预测。

运动关联门限以 `max(40 px, 框宽 × association_gate_scale)` 为基准，中心预测误差通过此门限筛选，短时遮挡期间额外增加有限容差；默认系数 0.8。`target_switch_timeout` 必须为正且不超过 `reset_timeout`。离线 CSV 继续记录每帧候选、角点筛选、PnP 解及物理/重投影拒绝数、关联距离和 EKF 拒绝标志。

无选中目标的有效 PnP 观测时 `detection_valid=false`，`observed_yaw=NaN`；未超时的滤波器仍可给出预测。未初始化或超时失效时 `filter_initialized=false`，滤波数值为 NaN。`measurement_used` 表示观测是否进入更新；`target_reset` 表示滤波重新初始化，避免把不同板面的角度连续混合。

## 离线结果与验证

```bash
ros2 run armor_ekf armor_ekf_offline \
  --model "$PWD/01_神经网络/yolo_pose/final_model/best.onnx" \
  --video "$PWD/00_题目原件/videos/test.mp4" \
  --output-dir "$PWD/ros2_ws/output/ekf_final" --stride 2
python_env/.venv/bin/python ros2_ws/src/armor_ekf/tools/plot_yaw.py ros2_ws/output/ekf_final
```

`--stride 2` 覆盖完整视频并每两帧处理一次，输出视频保持原时间长度。`--stride 1` 处理所有帧；`--no-video` 只输出 CSV 和统计；`--max-frames N` 可用于短段运行。CSV 和视频使用同一帧的检测、PnP 和滤波结果。

离线推理会同时生成 `detections.csv`，保存所有去重后的候选框、角点、类别、置信度与原始时间。回放通过同一 C++ PnP/关联/EKF 代码处理这些检测，不再执行网络推理，检测前端统计字段 `-1` 表示本次未执行该阶段。首次回放前，先取得你另行提供的 `test.mp4` 并放入 `00_题目原件/videos/`。以下命令会生成可再生成的检测缓存（输出目录受 Git 忽略）：

```bash
ros2 run armor_ekf armor_ekf_offline \
  --model "$PWD/01_神经网络/yolo_pose/final_model/best.onnx" \
  --video "$PWD/00_题目原件/videos/test.mp4" \
  --output-dir "$PWD/ros2_ws/output/tracker_cache" --stride 2 --no-video
```

可用完全相同输入比较新旧方法：

```bash
ros2 run armor_ekf armor_ekf_replay \
  --model "$PWD/01_神经网络/yolo_pose/final_model/best.onnx" \
  --detections "$PWD/ros2_ws/output/tracker_cache/detections.csv" \
  --output-dir "$PWD/ros2_ws/output/tracker_comparison/legacy" --association-mode legacy
ros2 run armor_ekf armor_ekf_replay \
  --model "$PWD/01_神经网络/yolo_pose/final_model/best.onnx" \
  --detections "$PWD/ros2_ws/output/tracker_cache/detections.csv" \
  --output-dir "$PWD/ros2_ws/output/ekf_tracking_final" --association-mode motion \
  --video "$PWD/00_题目原件/videos/test.mp4"
python_env/.venv/bin/python ros2_ws/src/armor_ekf/tools/compare_tracking.py \
  ros2_ws/output/tracker_comparison/legacy/yaw.csv \
  ros2_ws/output/ekf_tracking_final/yaw.csv ros2_ws/output/ekf_tracking_final
```

传入 `--video` 会从原始视频绘制标注回放视频；完整缓存包含 3150 个时刻，每两源帧取一帧。`--max-frames` 可用于短段对照，`--target-switch-timeout` 可调整短期目标等待时间。

数学检查和 ROS 消息检查：

```bash
source /opt/ros/lyrical/setup.bash
source "$HOME/armor_ekf_runtime/install/setup.bash"
colcon --log-base "$HOME/armor_ekf_runtime/log" test \
  --build-base "$HOME/armor_ekf_runtime/build" \
  --install-base "$HOME/armor_ekf_runtime/install" \
  --packages-select armor_ekf --event-handlers console_direct+
colcon test-result --test-result-base "$HOME/armor_ekf_runtime/build" --verbose
# 联合节点运行时，另开终端从资料根目录执行：
python3 ros2_ws/src/armor_ekf/tools/verify_ros.py --output-dir ros2_ws/output/ros_verification
```

验证覆盖 EKF 噪声抑制、±π 跨越、协方差、异常观测门限、漏检预测及重置，以及同类别交叉、短期遮挡、运动重关联、过期目标切换与一对一分配；用合成角点检查 PnP 的位置、角度符号和重投影；ROS 验证接收真实自定义消息并检查姿态四元数与 Yaw 一致、时间戳、方差及标注图像。

完整同输入对照的选中目标有效观测覆盖率从 51.08% 提升到 72.38%，2263 / 2280 条观测进入 EKF；216 次视觉轨迹切换均明确初始化滤波。结果与限制见 [`VERIFICATION.md`](VERIFICATION.md) 和 [`04_EKF/results/`](../../../04_EKF/results/)。完整标注视频 `ekf_tracking_final.mp4` 由你另行提供，不纳入 Git。实拍视频没有真实角度或永久物理身份标签，因此覆盖率、重投影误差和曲线平滑程度不能当作绝对 Yaw 或永久身份精度。

## 来源

- 考核 PDF：目标尺寸、相机标定和任务要求。
- 用户提供的模型、训练 notebook：YOLO26n Pose 模型与导出配置。
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0)：C++ 推理 SDK，MIT。
- [OpenCV PnP 文档](https://docs.opencv.org/4.10.0/d5/d1f/calib3d_solvePnP.html)：IPPE、候选姿态和重投影接口。
- [Ultralytics](https://github.com/ultralytics/ultralytics)：训练工具，模型元数据记录版本 8.4.174、AGPL-3.0。

## 世界坐标变换

`world.launch.py` 在同一 `armor_ekf_node` 内读取并发布视频帧，PnP姿态经考核给定的三段静态外参转到odom，提取世界Yaw后进入EKF，同时发布 `/armor/world_yaw`、完整世界姿态、TF链与RViz Marker。`ekf.launch.py` 保持相机系模式。

完整运行命令、验证结果和限制见 [WORLD_VERIFICATION.md](WORLD_VERIFICATION.md)。RViz配置为 `ros2_ws/rviz/armor_world.rviz`；最终数据与代表帧位于 [`05_坐标变换/results/`](../../../05_坐标变换/results/)；可再生成的回放缓存位于被忽略的 `ros2_ws/output/`。

## 四装甲板重投影

`reprojection.launch.py` 也在同一节点内完成视频发布、识别、滤波与四板重投影，输出 PoseArray、MarkerArray 和标注图。运行、坐标约定与实拍外参限制见 [REPROJECTION_VERIFICATION.md](REPROJECTION_VERIFICATION.md)，跳框修复及最新验证见 [POSE_SELECTION_VERIFICATION.md](POSE_SELECTION_VERIFICATION.md)。
