# 重做：神经网络识别与四装甲板重投影

这是独立实现，不调用原 `armor_ekf` 的检测、跟踪或重投影代码。输入为提供的 `test.mp4`，模型使用 `01_神经网络/yolo_pose/final_model/best.onnx`。本机该目录没有名为 `yolo_pose.onnx` 的文件，运行时也可以用 `--model` 指定另一模型。

已生成完整时长的 [结果视频](output/reprojection.mp4)、[第 906 帧](output/frame_0906.png)、[第 978 帧](output/frame_0978.png)、[运行统计](output/summary.json)、[独立数值验证](output/validation.json) 和 [ROS 实测验证](output/ros_validation.json)。青色框与角点是推算出的四块板，黄绿色框与交叉线是 NN 检测，红色菱形是推算的机器人中心。背面和被遮挡的模型板也会显示，四板编号相对于当前锚板，不是永久物理 ID。

## 直接运行

在资料根目录执行，或使用 `run.sh` 的绝对路径；模型和视频的默认路径从脚本位置解析，不依赖终端当前目录。

```bash
bash 06_重投影/redo/run.sh
```

首次运行自动建立 `.venv` 并安装 `requirements.txt`。需要 Python 3、venv 和可用网络。当前已验证环境为 Ubuntu、Python 3.14.4、NumPy 2.3.5、OpenCV 4.10.0、ONNX Runtime 1.31.0、SciPy 1.18.1、imageio-ffmpeg 0.6.0；这些版本来自本机实际运行环境。

默认 `--profile video --stride 2`：输入为 1440×1080、约 100.016 fps，输出 H.264 MP4 为约 50.008 fps；隔一帧执行一次识别，保留原视频播放时长。需要逐原始帧处理时指定 `--stride 1`，排查时添加 `--debug` 显示板编号与状态。

```bash
bash 06_重投影/redo/run.sh --stride 1 --output 06_重投影/redo/output_full
bash 06_重投影/redo/run.sh --debug --max-frames 100 --output 06_重投影/redo/preview
```

输出包含 `reprojection.mp4`、代表帧 PNG、逐帧 `frames.jsonl`、四板位姿与角点 `poses.csv` 和 `summary.json`。无效帧在 CSV 中记录 `valid=0,plate=-1`，没有有效当前姿态时不绘制旧模型。MP4、JSONL 和 CSV 是可重新生成的大文件，不提交到 Git；代表帧及汇总验证保存到仓库。

## 坐标与求解

1. ONNX Runtime 执行 YOLO26 Pose 推理。输入使用 RGB、NCHW、归一化和居中 letterbox；输出为 `[1,N,14]`，包含框、置信度、类别和四个 xy 角点。角点顺序为左上、左下、右下、右上。
2. 装甲板尺寸使用 0.135×0.055 m，相机内参与畸变使用题目值。`solvePnPGeneric(..., SOLVEPNP_IPPE)` 求出两个平面姿态候选，剔除背向相机及负深度候选。
3. 姿态经相机光学系 → 相机系 → 云台系 → odom 变换。局部装甲板 X 为向外法向，Y 向左、Z 向上；题目位置公式的 X/Yaw 指向机器人中心，因此 `yaw_pdf = wrap(yaw_out + π)`。
4. 固定半径 0.25 m、题目 Pitch=+15°、Roll=0，优化机器人中心 XYZ 和锚板 Yaw 共四个量，使模型角点贴合识别角点。向外法向系下模型旋转为 `Rz(yaw_out) Ry(-15°)`，等价于题目向内系的 `Rz(yaw_pdf) Ry(+15°) Rz(π)`。两组 PnP 初值分别优化并检查，避免直接采用平面镜像解。
5. 单板即可推算四板。若识别到另一块板，对正面模型板作角点与中心匹配，再一对一联合优化；类别标签不作为永久板 ID。单板允许最多 3 px 角点 RMS，联合拟合允许每板最多 8 px RMS，后者容纳实拍与固定同高、固定俯仰模型之间的偏差。它们是接受阈值，不是精度承诺。
6. 每块板的向外 Yaw 为 `θ_i = θ_0 + iπ/2`，位置为 `p_i = center + r[cos θ_i,sin θ_i,0]`。四角经世界 → 相机变换和畸变投影得到像素位置。所有板共享高度与给定俯仰，包括锚板。

前一可靠中心和相邻板 90° 的相位关系仅用于候选选解，不使用旧姿态绘制漏检帧。超过 0.3 s 没有可靠姿态会清除先验。两组单板解中心差异明显、像素误差接近且无可靠先验时，当前帧不输出模型。

## 外参配置

默认 `video` 配置使用 `odom→gimbal` Pitch=0 rad，其余变换沿用题目：`gimbal→camera` Pitch=0.091 rad、平移 `(0.0461892,0,-0.0603874)` m，光学系采用题目给定旋转。0 rad 是针对这段视频的对照假设，**不是实测标定值**；该视频的拟合结果不能证明世界坐标下绝对位置准确。

`pdf` 配置严格使用题目给定的云台 Pitch=1.34 rad，可单独对照运行：

```bash
bash 06_重投影/redo/run.sh --profile pdf --output 06_重投影/redo/preview/pdf
```

也可通过 `--gimbal-pitch-rad` 指定实际测量值。视频要求与标定分辨率一致；更换相机或视频尺寸时，需修改 K/D 或重新标定。

本次实际对照了 `pdf` 配置的前 20 个采样帧：20 帧均因固定四板模型与角点不一致而拒绝输出。这支持当前视频外参与题目给定值不匹配的判断，但不构成对实际外参的标定。

## ROS 2 与 RViz2

该实现提供 Python ROS 2 入口，不需要编译原 C++ 包。运行脚本会加载本机 `/opt/ros/lyrical/setup.bash`：

```bash
bash 06_重投影/redo/run.sh ros --loop
```

另开一个终端：

```bash
source /opt/ros/lyrical/setup.bash
rviz2 -d 06_重投影/redo/reprojection.rviz
```

话题为 `/armor/redo/source`（原图）、`/armor/redo/image_raw`（标注图）、`/armor/redo/poses`（odom 下四板 PoseArray）和 `/armor/redo/markers`（四板轮廓与中心）。一组输出的时间戳保持一致，同时广播题目规定的三段静态 TF。有效测量缺失时发布空 PoseArray 和五个 DELETE Marker。不要同时启动使用相同 TF 坐标名、不同外参的旧节点。

`--fps 30` 是 ROS 处理回调的请求频率；实际推理约 9 fps，不能视为原视频 100 fps 实时处理能力。视频采样帧号用于求解的时间连续性，ROS 消息使用当前 ROS 时钟。

## 验证

```bash
06_重投影/redo/.venv/bin/python -m unittest discover -s 06_重投影/redo -p 'test_*.py' -v
06_重投影/redo/.venv/bin/python 06_重投影/redo/validate.py 06_重投影/redo/output
```

`test_geometry.py` 使用已知中心的合成场景、双板角点噪声、非零外参、真实第 978 帧、矛盾检测、漏检和负深度。`validate.py` 独立写出 Brown-Conrady 畸变公式，复算整个输出的世界/相机变换、四板半径和角点；另检查 MP4 帧数、帧率及首/中/尾帧解码。

完整视频共处理 3,150 个采样帧，3,138 帧生成四板模型，12 帧拒绝几何不一致的测量；其中 947 帧有另一块检测板参与约束。联合拟合误差来自同一 NN 的角点，属于拟合残差，不是独立定位精度。详细数值见 `output/summary.json` 和 `output/validation.json`。8 项几何测试通过。ROS 从第 970 帧起实际处理 60 帧，并接收检查了 20 组时间戳匹配的图像、PoseArray 和 MarkerArray，其中 19 组包含四板位姿、1 组为空位姿与 DELETE 标记，验证了拒绝测量后的清除行为。

如需重新验证 ROS，在一个终端运行 `run.sh ros --max-frames 100`，另一个已加载 ROS 环境的终端运行 `.venv/bin/python verify_ros.py`。

验证清除行为时，将发布命令改为 `run.sh ros --start-frame 970 --max-frames 100`，检查脚本加 `--require-clear`。

使用来源：用户提供的 YOLO26 Pose ONNX 模型与测试视频；计算和推理依赖 OpenCV、ONNX Runtime、NumPy、SciPy、ROS 2、imageio-ffmpeg。无重新训练模型、手工补画预测板或写死逐帧坐标。
