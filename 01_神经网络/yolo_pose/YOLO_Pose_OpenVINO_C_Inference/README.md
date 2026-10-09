# qd::YoloPoseOpenVINO — YOLO Pose 推理库

本目录基于考核 PDF 指定的 [YOLO_Pose_OpenVINO_C_Inference](https://gitee.com/slime0rimiru0/YOLO_Pose_OpenVINO_C_Inference) 示例仓库。为适配本项目整理了标准目录名，并给视频示例增加无窗口运行和静态图片抽样保存选项；仍使用该库的 OpenVINO 推理、后处理与 `drawDetections` 可视化实现。

上游下载内容未附 `LICENSE` 文件；本考核使用依据为 PDF 中允许参考或使用考核示例代码的说明。

基于 **C++17 + OpenVINO Runtime + OpenCV** 的 YOLO Pose 推理库，namespace 为 `qd`。支持加载 ONNX 格式的 YOLO Pose 模型，执行姿态/关键点检测，并在画面中绘制检测框、关键点和连线。

> 本库不包含训练权重或测试视频。请自行准备 ONNX 模型。

## 快速集成

将本仓库 clone 到你的项目目录中，然后在 CMakeLists.txt 中添加 `add_subdirectory` 即可：

```bash
# 在你的项目根目录下
git clone https://gitee.com/slime0rimiru0/YOLO_Pose_OpenVINO_C_Inference.git inference
```

**你的 CMakeLists.txt：**

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_app LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(inference)  # clone 到的目录名

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE qd_yolo_pose)
```

链接 `qd_yolo_pose` 后，OpenVINO 和 OpenCV 的依赖会自动传递，无需额外配置。

## 调用示例

**最简用法（单图推理）：**

```cpp
#include <qd/yolo_pose.hpp>
#include <opencv2/opencv.hpp>

int main() {
    // 1. 配置推理器（仅 model_path 必填，其余使用默认值）
    qd::InferConfig config;
    config.model_path = "/path/to/model.onnx";

    // 2. 创建推理器
    qd::YoloPoseOpenVINO detector(config);

    // 3. 推理
    cv::Mat image = cv::imread("/path/to/image.jpg");
    auto results = detector.infer(image);

    // 4. 读取结果
    for (const auto& det : results) {
        std::cout << "confidence=" << det.confidence
                  << " box=(" << det.box.x << "," << det.box.y
                  << "," << det.box.width << "," << det.box.height << ")"
                  << " keypoints=" << det.keypoints.size() << std::endl;
    }

    // 5. 绘制（使用 RoboMaster 默认配置）
    qd::YoloPoseOpenVINO::drawDetections(image, results, 0.5F);
    cv::imshow("Result", image);
    cv::waitKey(0);

    return 0;
}
```

**自定义推理配置：**

```cpp
qd::InferConfig config;
config.model_path = "/path/to/model.onnx";
config.device_name = "GPU";         // 使用 GPU 推理
config.input_size = 640;            // 模型输入尺寸
config.conf_threshold = 0.3F;       // 降低置信度阈值
config.iou_threshold = 0.4F;        // NMS IoU 阈值
config.keypoint_threshold = 0.4F;   // 关键点显示阈值
config.debug = true;                // 打印耗时信息
```

**查看推理耗时：**

```cpp
const auto& latency = detector.last_latency();
std::cout << "前处理: " << latency.preprocess_ms << " ms\n"
          << "推理:   " << latency.inference_ms << " ms\n"
          << "后处理: " << latency.postprocess_ms << " ms\n"
          << "总耗时: " << latency.total_ms << " ms\n";
```

## 公共 API

### `qd::InferConfig`

推理器配置结构体，所有字段均有默认值（`model_path` 除外）。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model_path` | `std::string` | 无（必填） | ONNX 模型路径 |
| `device_name` | `std::string` | `"CPU"` | OpenVINO 设备名（`CPU`、`GPU`、`AUTO`） |
| `input_size` | `int` | `640` | 模型输入尺寸 |
| `conf_threshold` | `float` | `0.25` | 检测置信度阈值 |
| `iou_threshold` | `float` | `0.45` | NMS IoU 阈值 |
| `keypoint_threshold` | `float` | `0.5` | 关键点显示阈值 |
| `debug` | `bool` | `false` | 是否输出调试信息 |

### `qd::KeyPoint`

单个关键点信息。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `x` | `float` | 关键点在原图中的 x 坐标 |
| `y` | `float` | 关键点在原图中的 y 坐标 |
| `confidence` | `float` | 关键点置信度 |

### `qd::PoseDetection`

单个姿态检测结果。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `box` | `cv::Rect` | 检测框，坐标已还原到原图 |
| `confidence` | `float` | 目标置信度 |
| `class_id` | `int` | 类别 ID |
| `keypoints` | `std::vector<KeyPoint>` | 姿态关键点 |

### `qd::LatencyInfo`

分阶段耗时统计，单位为毫秒。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `preprocess_ms` | `double` | 前处理总耗时 |
| `preprocess_resize_ms` | `double` | 前处理 resize 耗时 |
| `preprocess_fill_ms` | `double` | 前处理 padding 填充耗时 |
| `preprocess_write_ms` | `double` | 前处理写入 Tensor 耗时 |
| `inference_ms` | `double` | OpenVINO 推理耗时 |
| `postprocess_ms` | `double` | 后处理耗时 |
| `total_ms` | `double` | 总耗时 |

### `qd::DrawConfig`

绘制配置，控制 `drawDetections` 的行为。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `class_names` | `std::vector<std::string>` | 类别名列表 |
| `class_color_fn` | `std::function<cv::Scalar(const std::string&)>` | 类别名 → BGR 颜色映射 |
| `keypoint_color` | `cv::Scalar` | 关键点绘制颜色 |
| `edge_pairs` | `std::vector<std::pair<size_t, size_t>>` | 关键点连线配置 |
| `robomasterDefaults()` | `static DrawConfig` | RoboMaster 装甲板默认配置 |

### `qd::YoloPoseOpenVINO`

推理器类，PIMPL 设计，公共头文件不依赖 OpenVINO。

| 方法 | 说明 |
| --- | --- |
| `YoloPoseOpenVINO(const InferConfig&)` | 构造推理器并加载模型 |
| `infer(const cv::Mat& frame)` | 对一帧 BGR 图像执行姿态估计，返回检测结果列表 |
| `last_latency()` | 获取最近一帧的耗时统计 |
| `drawDetections(image, detections, threshold, draw_config)` | 在图像上绘制检测框和关键点 |

## 绘制配置自定义

`drawDetections` 默认使用 `DrawConfig::robomasterDefaults()`（RoboMaster 装甲板红蓝配色和闭合四边形连线）。你可以传入自定义 `DrawConfig` 适配其他模型：

```cpp
// 自定义类别名和颜色
qd::DrawConfig draw_config;
draw_config.class_names = {"person", "car", "dog"};
draw_config.class_color_fn = [](const std::string& name) -> cv::Scalar {
    if (name == "person") return cv::Scalar(0, 255, 0);   // 绿色
    if (name == "car") return cv::Scalar(0, 0, 255);      // 红色
    return cv::Scalar(255, 255, 255);                      // 白色兜底
};
draw_config.keypoint_color = cv::Scalar(255, 128, 0);     // 橙色关键点
draw_config.edge_pairs = {{0, 1}, {1, 2}, {2, 3}, {3, 0}}; // 四边形连线

qd::YoloPoseOpenVINO::drawDetections(image, results, 0.5F, draw_config);
```

## 示例程序

本仓库附带两个示例程序，仅在作为顶层项目构建时编译（通过 `add_subdirectory` 引入时不会构建示例）。

### 构建示例

```bash
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

### 视频推理示例

```bash
./build/yolo_pose_demo \
  --model path/to/model.onnx \
  --video path/to/video.mp4 \
  --device CPU \
  --input-size 640 \
  --conf 0.25 \
  --iou 0.45 \
  --kpt-conf 0.5 \
  --debug
```

窗口中按 `q` 或 `Esc` 退出。

### 本考核模型验证（无窗口，不生成结果视频）

在资料根目录安装 `python_env/requirements.txt` 后运行：

```bash
OV_CMAKE_DIR="$(python_env/.venv/bin/python -c 'from openvino.utils import get_cmake_path; print(get_cmake_path())')"
cmake -S 01_神经网络/yolo_pose/YOLO_Pose_OpenVINO_C_Inference \
  -B 01_神经网络/yolo_pose/YOLO_Pose_OpenVINO_C_Inference/build \
  -DOpenVINO_DIR="$OV_CMAKE_DIR"
cmake --build 01_神经网络/yolo_pose/YOLO_Pose_OpenVINO_C_Inference/build -j"$(nproc)"
01_神经网络/yolo_pose/YOLO_Pose_OpenVINO_C_Inference/build/yolo_pose_demo \
  --model 01_神经网络/yolo_pose/final_model/best.onnx \
  --video 00_题目原件/videos/armors_yolo_test.mp4 \
  --device CPU --headless \
  --save-dir 01_神经网络/results/openvino_frames --save-every 300
```

程序逐帧读取题目测试视频并调用指定示例库完成 OpenVINO 推理。`--save-dir` 只保存首帧和每 300 帧的一张 PNG 可视化图，不创建视频文件。绘图类别名使用训练数据中的 `armor_0` 至 `armor_12`，不映射成示例库中不匹配的 RoboMaster 类别名。本项目导出的 ONNX 含 NMS，输出为端到端 `[1,300,14]`；该库的后处理支持此输出形式，因此不需要按其通用建议重新导出 `nms=False` 模型。

### 单图推理示例

```bash
./build/yolo_pose_minimal_example
```

需要在可执行文件同目录下准备 `model.onnx` 和 `test.jpg`。

### 命令行参数（视频示例）

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--model` | 是 | 无 | ONNX 模型路径 |
| `--video` | 是 | 无 | 输入视频路径 |
| `--device` | 否 | `CPU` | OpenVINO 推理设备 |
| `--input-size` | 否 | `640` | 模型输入尺��� |
| `--conf` | 否 | `0.25` | 检测置信度阈值 |
| `--iou` | 否 | `0.45` | NMS IoU 阈值 |
| `--kpt-conf` | 否 | `0.5` | 关键点显示阈值 |
| `--debug` | 否 | 关闭 | 输出耗时信息 |
| `--help` | 否 | 无 | 显示帮助 |

## 模型准备

请准备一个 YOLO Pose 的 ONNX 模型，例如通过 Ultralytics 导出：

```bash
yolo export model=path/to/model.pt format=onnx imgsz=640 opset=12 simplify=True dynamic=False nms=False
```

推荐导出参数：

- `format=onnx`：导出 ONNX，便于 OpenVINO 读取
- `imgsz=640`：固定输入尺寸，运行时 `input_size` 需与之保持一致
- `opset=12`：通常具有较好的 OpenVINO 兼容性
- `simplify=True`：简化 ONNX 图
- `dynamic=False`：使用固定输入尺寸
- `nms=False`：保留 YOLO Pose 原始输出，在 C++ 端执行后处理和 NMS

## 环境依赖

- CMake 3.16+
- 支持 C++17 的编译器
- OpenCV（编译时依赖，公共 API 使用 `cv::Mat`、`cv::Rect` 等）
- OpenVINO Runtime（链接时依赖，通过 PIMPL 隐藏，公共头文件不包含 OpenVINO）

Ubuntu 安装：

```bash
# OpenCV
sudo apt update
sudo apt install -y libopencv-dev

# OpenVINO（使用 Intel APT 源）
# https://docs.openvino.ai/2025/get-started/install-openvino.html
```

## 输出格式兼容性

本库支持 Ultralytics YOLO Pose 常见的两种输出排列：

```text
[1, attributes, anchors]
[1, anchors, attributes]
```

同时兼容部分端到端导出模型的输出形式，例如：

```text
[1, 300, 14]
```

对于 `kpt_shape: [4, 2]` 的模型，14 个属性通常表示：

```text
x1, y1, x2, y2, score, class_id, 4 * (keypoint_x, keypoint_y)
```

普通 YOLO Pose 原始输出的 attributes 通常是：

```text
4 个框参数 + 1 个目标置信度 + N * 3 个关键点参数
```

建议按 `nms=False` 导出 ONNX 模型，在 C++ 端完成后处理。

## 项目结构

```text
.
├── CMakeLists.txt
├── README.md
├── example/
│   ├── example.cpp              # 视频推理示例
│   └── minimal_example.cpp      # 单图推理示例
├── include/
│   └── qd/
│       └── yolo_pose.hpp        # 公共头文件（PIMPL，namespace qd）
└── src/
    └── yolo_pose_openvino.cpp   # 推理实现（PIMPL Impl）
```

## 常见问题

### CMake 找不到 OpenCV

Ubuntu 安装开发包：

```bash
sudo apt install -y libopencv-dev
```

如果从源码安装 OpenCV，需要设置 `OpenCV_DIR`：

```bash
cmake -S . -B build -DOpenCV_DIR=path/to/opencv/cmake
```

### 没有检测结果

可以尝试降低阈值：

```cpp
config.conf_threshold = 0.1F;
config.keypoint_threshold = 0.2F;
```

同时确认：

- `input_size` 与导出 ONNX 时的 `imgsz` 一致
- 模型类别和关键点格式与后处理逻辑兼容
- 输入中确实存在模型可识别的目标
