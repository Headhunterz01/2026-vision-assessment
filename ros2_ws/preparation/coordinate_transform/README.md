# 坐标变换考核：准备记录

准备日期：2026-10-09。此文记录准备阶段的方案；实现和验证现已完成，见 [坐标变换验证报告](../../src/armor_ekf/WORLD_VERIFICATION.md)。

## 目标与范围

依据[题目原件](../../../00_题目原件/暑假考核.pdf)第4、5页：将 solvePnP 的装甲板完整姿态从 camera_optical_frame 转到 odom，提取 odom Yaw，进入现有 EKF，并通过 ROS 2 自定义消息同时发布世界系观测和滤波 Yaw。后面的“重投影”考核另行开展。

## 已确认环境与输入

- 已安装 ROS 2 Lyrical、RViz2、tf2、tf2_ros、tf2_geometry_msgs；OpenCV、Eigen、ONNX Runtime 为现有工程依赖。
- 现有 python_env/.venv 可导入 NumPy 2.5.3、Matplotlib 3.11.2，无需新增环境或安装库。
- 模型：`01_神经网络/yolo_pose/final_model/best.onnx`。
- 视频：`00_题目原件/videos/test.mp4`。
- 检测缓存：ros2_ws/output/tracker_cache/detections.csv，可绕开重复推理用于同输入对比。
- 相机内参、畸变、小装甲板尺寸 0.135 × 0.055 m 已在 src/armor_ekf/config/ekf.yaml 中配置。
- 工程目前没有 TF 或实时云台角度接入；UART 是独立串口考核回传协议，未定义云台姿态。
- 准备开始前 ros2_ws 工作树干净，基线提交 a80dc48。

## 坐标和矩阵约定

使用列向量，T_parent_child 表示把 child 坐标中的点转换到 parent：

```text
p_parent = R_parent_child * p_child + t_parent_child
R(r,p,y) = Rz(y) * Ry(p) * Rx(r)
T_odom_optical = T_odom_gimbal * T_gimbal_camera * T_camera_optical
R_odom_armor = R_odom_optical * R_optical_armor
t_odom_armor = R_odom_optical * t_optical_armor + t_odom_optical
```

旋转后平移；平移向量以父坐标系表示。外参不是简单的向量相加。

| 父坐标系 | 子坐标系 | xyz / m | rpy / rad |
| --- | --- | --- | --- |
| odom | gimbal_link | (0,0,0) | (0,1.34,0) |
| gimbal_link | camera_link | (0.0461892,0,-0.0603874) | (0,0.091,0) |
| camera_link | camera_optical_frame | (0,0,0) | (-π/2,0,-π/2) |

按考核固定参数建立首版链路。odom/gimbal/camera_link 为 X前、Y左、Z上；optical 为 X右、Y下、Z前。Pitch 向下为正，Yaw 向左为正。tf2 本机头文件的 setRPY 使用对应固定轴旋转约定。

现有装甲板物体坐标系 X 为向外法向、Y 为左、Z 为上，因此：

```text
n_odom = R_odom_armor.col(0)
yaw_odom = atan2(n_odom.y, n_odom.x)
```

当法向水平投影接近零时，Yaw 不可可靠定义，需要拒绝该角度观测。相机系正对时 Yaw=0 是原有的相机视角定义；世界系朝向可能接近±π，不能把这种偏移当作错误。

## 实现接入方案

1. 使用 C++17 + Eigen 实现可供 ROS 节点和离线回放共用的姿态转换模块；OpenCV Rodrigues 负责 rvec 与旋转矩阵之间的转换。
2. ROS节点使用 tf2_ros 静态发布器发布三段考核外参；数值集中定义，计算与 TF 发布使用同一个已解析的配置，避免两份外参漂移。
3. PnP 的可见性、深度和重投影检查仍在相机系完成；新增世界系姿态、Yaw及水平法向有效性判断。保持相机 rvec/tvec 供图像轴线绘制使用。
4. 在 EKF 更新之前转换每个候选姿态。当前 IPPE 候选解的连续性比较使用 p.yaw 与 filter_.yaw；两者必须同时采用 odom 定义，不能等 EKF 后才换算。
5. 继续使用图像运动关联、选中目标切换及 EKF reset 逻辑。滤波状态、观测、门限角度差均使用世界系 Yaw。
6. 保留原相机系 /armor/yaw 路径作为已完成考核的复现入口；通过独立世界模式/launch 发布 /armor/world_yaw。复用自定义 YawEstimate 类型时，角度与 pose 统一属于 header.frame_id；世界消息 frame_id=odom，时间戳沿用源图像。corners 始终为像素坐标。消息注释需要更新明确两种约定。
7. 为 CSV/标注/绘图新增世界系结果和明确的坐标系标签，避免相机 pose 与世界 yaw 同列却无说明。
8. RViz 固定帧设为 odom，显示 TF 与世界系装甲板姿态（可用 Marker）；同时保留图像可视化。

动态云台为后续扩展：若以后接入实时角度，odom→gimbal 改为动态图，必须按图像时间戳查询 TF，缺失时禁止用旧变换冒充当前观测。固定考核参数下不需要实时云台数据。

## 验证顺序与验收条件

1. 准备数学基准：光学轴映射、旋转正交与行列式、链式组合、位置逆变换、已知世界姿态在不同相机旋转下恢复同一 Yaw。check_geometry.py 生成 geometry_baseline.json；它只检查数学基准，尚未验证产品代码。
2. 实现后用已知世界装甲板姿态反变换至相机并投影为角点，通过真实 PnP→转换→EKF 链路恢复世界位置与 Yaw。覆盖正负角、±π、云台旋转和装甲板15°俯仰。
3. 对比 Eigen 与 ROS TF2 变换输出；检查 TF 三段父子关系、外参数值、最终位置与姿态。
4. 同一视频/检测缓存比较相机和世界角度结果；检查选中目标、切换、重投影误差未被错误坐标语义改变，必要的候选解变化需解释。
5. ROS 实际订阅验证自定义消息：frame_id=odom、源时间戳、四元数归一化、Yaw与pose法向一致、滤波有效标志、NaN/丢失语义、协方差和切换reset。角度误差使用环绕差。
6. RViz 与世界Yaw曲线、标注视频确认通过后再报告考核完成。

没有真实视频世界Yaw真值时，可证明转换计算和端到端一致性，不能仅凭曲线更平滑声称真实绝对精度提升。完整姿态中的15°固定装甲板俯仰属于后续重投影已知条件，当前视频PnP不应强行固定这个角度。

## 准备阶段结果

准备阶段已运行数学基准检查，结果见 geometry_baseline.json。当时尚未修改生产代码、消息定义、launch 或运行结果；后续已完成实现和端到端验证，详见上方报告。
