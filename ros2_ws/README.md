# ROS 2 工作区

ROS 2 图像发布、EKF、坐标变换、重投影共用此工作区。源码真源位于 `src/`：图像发布器在 `src/armor_video_publisher/`；EKF、坐标变换和重投影集成在 `src/armor_ekf/`，消息定义在 `src/armor_interfaces/`。

## 环境配置与构建

目标环境为 **Ubuntu 26.04 + ROS 2 Lyrical desktop（含 RViz2）**。若 ROS 2 已安装，只需安装项目的系统构建依赖。以下命令从资料根目录执行：

```bash
sudo apt update
sudo apt install -y build-essential cmake curl tar \
  python3-colcon-common-extensions python3-opencv python3-setuptools \
  libopencv-dev libeigen3-dev ros-lyrical-cv-bridge
source /opt/ros/lyrical/setup.bash
bash ros2_ws/build_ekf.sh
source "$HOME/armor_ekf_runtime/install/setup.bash"
```

构建脚本在缺少时从 ONNX Runtime 官方发布页下载 1.30.0 C++ SDK 到被 Git 忽略的 `ros2_ws/deps/`。构建产物放在 `$HOME/armor_ekf_runtime/{build,install,log}`，中文资料路径通过 ASCII 别名 `$HOME/armor_ekf_ws` 供 colcon/ROS 消息生成器使用。源码目录本身不保存安装产物。

离线绘图工具另用资料根目录的 Python venv，安装步骤见 [`python_env/README.md`](../python_env/README.md)。ROS 节点和 ROS 验证工具应使用系统 ROS Python 环境，不要从该 venv 启动。

## 发布测试视频图像

先从 GitHub Release 下载 `armors_yolo_test.mp4` 或 `test.mp4`，放回 `00_题目原件/videos/`。从资料根目录运行：

```bash
ros2 run armor_video_publisher video_publisher --ros-args \
  -p video_path:="$PWD/00_题目原件/videos/armors_yolo_test.mp4" \
  -p topic_name:=/armor/image_raw \
  -p frame_id:=camera_optical_frame \
  -p loop:=true
```

在另一个已 source ROS 环境和工作区的终端查看：

```bash
rviz2 -d ros2_ws/rviz/armor_video.rviz
ros2 topic hz /armor/image_raw
```

验证器命令：

```bash
python3 '03_ROS2图像发布/results/verify_image_topic.py'
```

验证器检查 30 帧图像的尺寸、`bgr8` 编码、数据长度、frame_id、递增时间戳和约 30 Hz 频率。验证截图、报告和两个视频的接收日志在 [`03_ROS2图像发布/results/`](../03_ROS2图像发布/results/)。

## ROS 视觉考核入口

- [EKF 运行与实现](src/armor_ekf/README.md)，[验证报告](src/armor_ekf/VERIFICATION.md)。
- [坐标变换运行与验证](src/armor_ekf/WORLD_VERIFICATION.md)。
- [四板重投影实现与限制](src/armor_ekf/REPROJECTION_VERIFICATION.md)，[姿态选解和跳框修复](src/armor_ekf/POSE_SELECTION_VERIFICATION.md)。
- 每项考核的结果目录分别是 [`04_EKF/results/`](../04_EKF/results/)、[`05_坐标变换/results/`](../05_坐标变换/results/) 和 [`06_重投影/results/`](../06_重投影/results/)。最终视频作为 GitHub Release 附件发布，名称见 [`RELEASE_ASSETS.md`](../RELEASE_ASSETS.md)。

## 图像发布验证结论

在 Ubuntu 26.04 / ROS 2 Lyrical 上构建通过。两个输入视频均收到合法图像，RViz2 的 Image 显示订阅 `/armor/image_raw` 并显示正常；验证日志和截图保存在 `03_ROS2图像发布/results/`。发布器收到 Ctrl+C 后正常退出，输入路径无效时返回非零退出码。RViz2 在尚未发布 TF 时可能报告 Fixed Frame 不存在，但不影响 Image 显示。
