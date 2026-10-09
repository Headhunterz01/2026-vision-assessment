#include "qd/yolo_pose.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. 配置推理器（仅需 model_path，其余使用默认值）
    qd::InferConfig config;
    config.model_path = "model.onnx";

    // 2. 创建推理器
    qd::YoloPoseOpenVINO detector(config);

    // 3. 读取图像并推理
    cv::Mat image = cv::imread("test.jpg");
    if (image.empty()) {
        std::cerr << "无法读取图像: test.jpg" << std::endl;
        return 1;
    }

    auto results = detector.infer(image);

    // 4. 输出检测结果
    for (const auto& det : results) {
        std::cout << "confidence=" << det.confidence
                  << " box=(" << det.box.x << "," << det.box.y
                  << "," << det.box.width << "," << det.box.height << ")"
                  << " keypoints=" << det.keypoints.size() << std::endl;
    }

    // 5. 绘制结果（使用 RoboMaster 默认配置）
    qd::YoloPoseOpenVINO::drawDetections(image, results, 0.5F);

    cv::imshow("Result", image);
    cv::waitKey(0);

    // 6. 查看耗时
    const auto& latency = detector.last_latency();
    std::cout << "Total: " << latency.total_ms << " ms" << std::endl;

    return 0;
}
