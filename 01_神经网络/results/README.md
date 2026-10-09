# YOLO OpenVINO 推理验证

使用 PDF 指定的 C++ OpenVINO 推理库和 `best.onnx` 对 `armors_yolo_test.mp4` 逐帧推理至解码结束。程序以 CPU 运行，原视频为 1440×1080、30 FPS；解码器实际返回 2055 帧，1998 帧至少检测到一个目标，总计 5127 个检测，OpenVINO 模型调用平均耗时 29.9155 ms/帧。OpenCV 元数据显示 2061 帧，但 C++/Python 两种 OpenCV 解码循环都实际读取 2055 帧，因此统计按实际处理帧数记录。耗时是模型推理阶段计时，不包含视频解码、绘制和图片保存开销；检测数量不是准确率指标。

模型 SHA-256：`b6c20fa44427ef5f0dae115cf0f0ca13a26589a7983e12cab1785041711b5444`。

静态可视化抽帧：

- [第 1 帧](openvino_frames/frame_000001.png)
- [第 400 帧](openvino_frames/frame_000400.png)
- [第 800 帧](openvino_frames/frame_000800.png)
- [第 1200 帧](openvino_frames/frame_001200.png)
- [第 1600 帧](openvino_frames/frame_001600.png)
- [第 2000 帧](openvino_frames/frame_002000.png)

这些图片由程序的检测框、类别、置信度、四个姿态关键点及推理时间绘制接口生成。验证运行只保存 PNG，没有创建视频文件。完整构建与运行命令见 [`OpenVINO 推理代码说明`](../yolo_pose/YOLO_Pose_OpenVINO_C_Inference/README.md)。
