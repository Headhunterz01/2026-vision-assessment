# Python 辅助工具环境

此目录只保存依赖清单和环境验证脚本；虚拟环境本身不纳入仓库。资料根目录运行：

```bash
python3 -m venv python_env/.venv
python_env/.venv/bin/python -m pip install --upgrade pip
python_env/.venv/bin/python -m pip install -r python_env/requirements.txt
```

检查依赖和原始视频：

```bash
python_env/.venv/bin/python python_env/verify_environment.py
python_env/.venv/bin/python -m pip check
```

退出环境使用 `deactivate`。Python 清单只用于离线分析、画图及串口环境检查；ROS 2、`cv_bridge`、OpenCV C++、Eigen 和 colcon 通过 Ubuntu/ROS apt 包安装，见 [`ros2_ws/README.md`](../ros2_ws/README.md)。ROS 节点使用系统 ROS Python 环境，不要从此 venv 启动。
