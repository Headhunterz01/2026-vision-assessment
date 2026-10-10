# 2026 赛季视觉组考核

六项考核资料与实现记录。题目原件、源码入口和每项结果分别按考核编号归档；细节、运行方式与技术分析见各项 README。

## 环境配置

ROS 题目使用 Ubuntu 26.04、ROS 2 Lyrical desktop（含 RViz2）。Python 辅助工具单独使用虚拟环境：

```bash
python3 -m venv python_env/.venv
python_env/.venv/bin/python -m pip install -r python_env/requirements.txt
```

ROS 构建依赖和工作区构建命令见 [`ros2_ws/README.md`](ros2_ws/README.md)。虚拟环境、ROS 构建目录和 ONNX Runtime SDK 缓存均由命令生成，不属于源码；完整 Python 配置见 [`python_env/README.md`](python_env/README.md)。

## 考核目录

| 编号 | 内容 |
| --- | --- |
| [00 题目原件](00_题目原件/README.md) | PDF、原始测试视频和视频 Release 资产说明 |
| [01 神经网络](01_神经网络/README.md) | YOLO Pose notebook、最终模型、训练曲线和验证结果 |
| [02 UART](02_UART/README.md) | 协议源码、测试器和测试记录 |
| [03 ROS 2 图像发布](03_ROS2图像发布/README.md) | 图像发布节点、RViz2 画面和话题验证 |
| [04 EKF](04_EKF/README.md) | 姿态解算、目标关联和 Yaw EKF |
| [05 坐标变换](05_坐标变换/README.md) | 相机坐标系到 odom 的姿态变换 |
| [06 重投影](06_重投影/README.md) | 四装甲板模型与图像重投影 |

原视频和最终结果视频位于release，但不纳入 Git；上传文件清单、SHA-256 和获取说明见 [`RELEASE_ASSETS.md`](RELEASE_ASSETS.md)。
