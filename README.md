# 2026 赛季视觉组考核

六项考核资料与实现记录。题目原件、源码入口和每项结果分别按考核编号归档；细节、运行方式与技术分析见各项 README。

## 环境配置

ROS 题目使用 Ubuntu 26.04、ROS 2 Lyrical desktop（含 RViz2）。Python 辅助工具单独使用虚拟环境：

```bash
python3 -m venv python_env/.venv
python_env/.venv/bin/python -m pip install -r python_env/requirements.txt
```

ROS 构建依赖和工作区构建命令见 [`ros2_ws/README.md`](ros2_ws/README.md)。虚拟环境、ROS 构建目录和 ONNX Runtime SDK 缓存均由命令生成，不属于源码；完整 Python 配置见 [`python_env/README.md`](python_env/README.md)。

## 核心交付入口

- ROS 2 集成包：[`armor_ekf`](ros2_ws/src/armor_ekf/)，入口节点 `armor_ekf_node` 在单节点内完成视频图像发布、YOLO 识别、PnP 姿态解算、EKF 滤波、标注图像和结果发布；自定义 ROS 消息定义在 [`armor_interfaces`](ros2_ws/src/armor_interfaces/)。构建及一条命令启动方式见 [`ROS 2 工作区说明`](ros2_ws/README.md)。
- UART 源码：[`uart_echo.py`](02_UART/uart/uart_echo.py)，协议与校验测试见 [`UART README`](02_UART/uart/README.md)。

## 考核目录

| 编号 | 内容 |
| --- | --- |
| [00 题目原件](00_题目原件/README.md) | PDF、题目视频名称和本地放置说明 |
| [01 神经网络](01_神经网络/README.md) | YOLO Pose notebook、最终模型、训练曲线和验证结果 |
| [02 UART](02_UART/README.md) | 协议源码、测试器和测试记录 |
| [03 ROS 2 图像发布](03_ROS2图像发布/README.md) | 图像发布节点、RViz2 画面和话题验证 |
| [04 EKF](04_EKF/README.md) | 姿态解算、目标关联和 Yaw EKF |
| [05 坐标变换](05_坐标变换/README.md) | 相机坐标系到 odom 的姿态变换 |
| [06 重投影](06_重投影/README.md) | 四装甲板模型与图像重投影 |

原始视频和结果视频不纳入 Git。复现所需的视频由你另行提供，按各考核 README 中的文件名放入指定目录。
