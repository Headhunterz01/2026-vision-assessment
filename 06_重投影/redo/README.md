# 神经网络识别与四装甲板重投影：稳定修复版

2026-10-13 更新：修复静止机器人上预测框的抖动，以及末段单板朝向偏差造成的侧板偏离。独立实现，不调用原 `armor_ekf` 的检测、跟踪或重投影代码。

- [修复后的完整视频](output_fixed/reprojection.mp4)
- [开头静止帧](output_fixed/frame_0450.png)、[末段第 6000 帧](output_fixed/frame_6000.png)
- [稳定性及末段对齐比较](output_fixed/stability_comparison.json)
- [独立数值验证](output_fixed/validation.json)、[运行统计](output_fixed/summary.json)、[ROS 实测](output_fixed/ros_validation.json)

青色框与角点是推算出的四块板；黄绿色框和交叉线是 NN 定位后，经灯条细化与时间平滑的观测。原始 NN 输出保存在逐帧日志的 `raw_detections`，处理后观测在 `detections`。红色菱形为机器人中心。模型包括背面及被遮挡的板，编号相对于当前锚板，不是永久物理 ID。

## 原因与修复

初版每帧独立解算姿态，仅按像素拟合误差和弱连续性选择候选。静止段约 1 px 的 NN 角点波动会引起平面朝向候选切换，推算出的板单帧跳动约 21 px。末段一块板拟合 RMS 仅 0.39 px，但朝向仍有偏差，因此不能用该板误差证明另外三块板正确。

现在保留 NN 作为主识别来源，加入以下约束：

1. 在 NN 给出的机身区域内提取蓝色灯条，用其轴线和端点细化角点。长度、方向、端点偏移均须与 NN 观测一致；过长的光晕、地面反射和残缺灯条会被排除。
2. 对只看到一块 NN 装甲板的帧，使用左右相邻板可见灯条的两端辅助解算；两个侧面必须各有一个独立灯条，才能校正单板朝向。未知的左右边分别尝试，一对一匹配。灯条可以从近切向及背面看到，因此辅助边不按整块板是否朝前筛除。
3. 在实际图像上计算光流，决定角点和完整模型的平滑强度。静止图像使用较强平滑；图像开始运动时迅速降低平滑强度。平滑前按 90° 的相位对齐锚板编号，避免换板时错误混合姿态。

半径、尺寸和俯仰均未为了截图调整。四块板仍从一个共享中心和 Yaw 推算。当前无有效观测时立即隐藏模型；超过 0.3 s 中断或时间回退会清除姿态先验。

## 运行

在资料根目录执行，或使用脚本绝对路径。默认模型与视频路径从脚本位置解析，不依赖当前终端目录。

```bash
bash 06_重投影/redo/run.sh
```

默认输入为提供的 `test.mp4`，模型为 `01_神经网络/yolo_pose/final_model/best.onnx`。本机没有名为 `yolo_pose.onnx` 的模型，可用 `--model` 指定另一个同格式模型。首次运行创建 `.venv` 并安装 `requirements.txt`；需要 Python 3、venv 和可用网络。

已验证环境：Python 3.14.4、NumPy 2.3.5、OpenCV 4.10.0、ONNX Runtime 1.31.0、SciPy 1.18.1、imageio-ffmpeg 0.6.0。

默认输出目录为 `output_fixed/`。输入为 1440×1080、约 100.016 fps；默认 `--stride 2` 隔帧识别，输出 H.264 MP4 约 50.008 fps，保持原播放时长。需要每个原始帧时使用 `--stride 1`，调试时用 `--debug`。

```bash
bash 06_重投影/redo/run.sh --stride 1 --output 06_重投影/redo/output_full
bash 06_重投影/redo/run.sh --max-frames 100 --debug --output 06_重投影/redo/preview
```

输出包括视频、代表帧、逐帧 `frames.jsonl`、`poses.csv` 和 `summary.json`。无效帧 CSV 为 `valid=0,plate=-1`。MP4、JSONL 和 CSV 是可再生成的大文件，不提交 Git；PNG 与验证汇总保存在仓库。

此次完整修复视频复用了初版记录的原始 NN 输出，重新读取原图执行灯条处理、光流、姿态和投影，以便公平比较。80 帧直接 ONNX 推理与该回放的最终角点完全一致。默认运行始终直接执行 NN；只有显式提供缓存参数才复用检测。

```bash
bash 06_重投影/redo/run.sh \
  --detections-cache 06_重投影/redo/output/frames.jsonl \
  --output 06_重投影/redo/output_fixed
```

缓存须对应同一模型、视频、置信度及所需采样帧。初版基线保留在 `output/`，稳定修复版在 `output_fixed/`。

## 几何与坐标

YOLO26 输入使用居中 letterbox、RGB、NCHW 和归一化，输出为 `[1,N,14]`：xyxy、置信度、类别和四个 xy 角点。顺序为左上、左下、右下、右上。

装甲板尺寸 0.135×0.055 m，相机 K/D 使用题目值。`solvePnPGeneric(..., SOLVEPNP_IPPE)` 求出两组平面姿态初值，剔除背向与负深度候选。姿态经光学系 → 相机系 → 云台系 → odom 变换后，优化中心 XYZ 与锚板 Yaw。

装甲板局部 X 为向外法向，Y 向左，Z 向上。题目位置公式中的 Yaw 指向中心，故 `yaw_pdf = wrap(yaw_out + π)`。固定半径 0.25 m、Pitch=+15°、Roll=0；向外系旋转 `Rz(yaw_out) Ry(-15°)` 等价于题目向内系 `Rz(yaw_pdf) Ry(+15°) Rz(π)`。

```text
theta_i = theta_0 + i*pi/2
p_i = center + radius * [cos(theta_i), sin(theta_i), 0]
```

四块板共用高度与俯仰，包括锚板。局部四角经世界 → 相机变换及畸变投影得到像素。单 NN 板拟合门限 3 px RMS，多 NN 板每板门限 8 px；有左右灯条支持的单板校正允许主板最多 4 px、辅助灯条最多 8 px，辅助灯条拟合权重为 0.5。时间平滑后的误差从最终投影重新计算并记录。这些是接受门限，不是绝对精度承诺。

## 外参与 ROS 2

默认 `video` 仍采用云台 Pitch=0 rad 的视频对照假设，未进行物理标定。其它变换保持题目值：相机 Pitch=0.091 rad、平移 `(0.0461892,0,-0.0603874)` m，以及给定光学系旋转。需要题目的云台 Pitch=1.34 rad 时使用 `--profile pdf`，也可通过 `--gimbal-pitch-rad` 指定实测值。

```bash
bash 06_重投影/redo/run.sh --profile pdf --output 06_重投影/redo/preview/pdf
bash 06_重投影/redo/run.sh ros --loop
```

ROS 入口加载 `/opt/ros/lyrical/setup.bash`，发布 `/armor/redo/source`、`/armor/redo/image_raw`、`/armor/redo/poses`、`/armor/redo/markers`，并广播三段静态 TF。一个输出组时间戳一致；无效观测发送空 PoseArray 和五个 DELETE Marker。另开终端运行：

```bash
source /opt/ros/lyrical/setup.bash
rviz2 -d 06_重投影/redo/reprojection.rviz
```

`--fps 30` 为回调请求频率，并不代表达到 30 fps。本机此次 80 帧直接推理示例约 6.2 fps，不能视为原视频 100 fps 的实时处理能力。不要同时运行使用相同 TF 坐标名、不同外参的旧节点。

## 验证

```bash
06_重投影/redo/.venv/bin/python -m unittest discover -s 06_重投影/redo -p 'test_*.py' -v
06_重投影/redo/.venv/bin/python 06_重投影/redo/validate.py 06_重投影/redo/output_fixed
06_重投影/redo/.venv/bin/python 06_重投影/redo/compare_stability.py \
  --output 06_重投影/redo/output_fixed/stability_comparison.json
```

19 项测试通过，覆盖真实末段灯条、长光晕与残缺灯条、静止噪声、400°/s 旋转及锚板切换、运动启停、断帧和不同目标重获；原几何测试仍通过。完整视频 3,150 个采样帧中 3,140 帧生成模型、10 帧拒绝，独立变换、半径、Brown-Conrady 投影检查通过，整段 MP4 解码通过。

原图光流显示机器人约 8.9 s 开始转动，因此静止窗口取 0–8 s 和 60–63 s。四板经循环重编号对齐后，相邻有效帧角点 RMS 跳动如下：

| 指标 | 初版 | 修复后 |
| --- | ---: | ---: |
| 开头静止段角点跳动 P95 | 20.925 px | 0.040 px |
| 末段静止段角点跳动 P95 | 1.562 px | 0.013 px |
| 第 6000 帧同一左侧灯条端点 RMS | 49.159 px | 1.770 px |
| 第 6000 帧同一右侧灯条端点 RMS | 48.606 px | 4.947 px |

末段比较在新旧模型上使用相同的原图灯条端点。它们参与新姿态拟合，属于图像一致性指标，不是独立物理位姿真值。

ROS 从第 1238 帧实际处理 60 帧，接收验证了 20 组同步输出：19 组四板位姿、1 组空位姿与 DELETE 标记。复测时一个终端运行 `run.sh ros --start-frame 1238 --max-frames 100`，另一个加载 ROS 环境后运行 `.venv/bin/python verify_ros.py --require-clear`。

来源：用户提供的 YOLO26 Pose 模型和测试视频；依赖 OpenCV、ONNX Runtime、NumPy、SciPy、ROS 2 和 imageio-ffmpeg。没有写死视频帧号对应的预测坐标，也没有手工补画装甲板。
