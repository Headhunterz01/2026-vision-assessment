# 考核一：YOLO Pose 神经网络

训练 notebook、训练记录和最终模型归档在本目录。原始测试视频由 GitHub Release 提供，附件名称见 [`RELEASE_ASSETS.md`](../RELEASE_ASSETS.md)。

- 训练与导出流程：[`yolo_pose/yolo26_pose.ipynb`](yolo_pose/yolo26_pose.ipynb)
- 训练输出、指标和可视化：[`yolo_pose/yolo26n-full/`](yolo_pose/yolo26n-full/)，包括 Box/Pose 曲线、混淆矩阵及验证预测图。
- 最终权重：[`best.pt`](yolo_pose/final_model/best.pt)、[`best.onnx`](yolo_pose/final_model/best.onnx)。ONNX 模型也供后续 ROS 视觉考核使用。
- 题目指定的 OpenVINO C++ 推理/可视化代码及本地运行说明：[`YOLO_Pose_OpenVINO_C_Inference`](yolo_pose/YOLO_Pose_OpenVINO_C_Inference/README.md)。
- 指定代码使用最终 ONNX 权重处理完整测试视频后的静态抽帧结果和运行统计：[`results/`](results/README.md)。按约定不保存 NN 结果视频。

训练 notebook 在 Colab 中安装 Ultralytics 并挂载训练数据；本地 Python/OpenVINO 环境配置见 [`python_env/README.md`](../python_env/README.md)。
