#pragma once

#include <opencv2/opencv.hpp>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief qd 命名空间：YOLO Pose OpenVINO 推理库。
 */
namespace qd {

/**
 * @brief 单个关键点的信息。
 */
struct KeyPoint {
    float x = 0.0F;          ///< 关键点在原图中的 x 坐标
    float y = 0.0F;          ///< 关键点在原图中的 y 坐标
    float confidence = 0.0F; ///< 关键点置信度
};

/**
 * @brief 单个姿态检测结果。
 */
struct PoseDetection {
    cv::Rect box;                    ///< 检测框，坐标已经还原到原图
    float confidence = 0.0F;         ///< 目标置信度
    int class_id = 0;                ///< 类别 ID，单类别模型通常为 0
    std::vector<KeyPoint> keypoints; ///< 姿态关键点
};

/**
 * @brief 分阶段耗时统计，单位为毫秒。
 */
struct LatencyInfo {
    double preprocess_ms = 0.0;        ///< 前处理总耗时
    double preprocess_resize_ms = 0.0; ///< 前处理 resize 耗时
    double preprocess_fill_ms = 0.0;   ///< 前处理 padding 填充耗时
    double preprocess_write_ms = 0.0;  ///< 前处理写入 Tensor 耗时
    double inference_ms = 0.0;         ///< OpenVINO 推理耗时
    double postprocess_ms = 0.0;       ///< 后处理耗时
    double total_ms = 0.0;             ///< 总耗时
};

/**
 * @brief 推理器配置，所有字段均有默认值（model_path 除外）。
 */
struct InferConfig {
    std::string model_path;           ///< ONNX 模型路径（必填）
    std::string device_name = "CPU";  ///< OpenVINO 设备名，例如 CPU、GPU、AUTO
    int input_size = 640;             ///< 模型输入尺寸
    float conf_threshold = 0.25F;     ///< 目标置信度阈值
    float iou_threshold = 0.45F;      ///< NMS 的 IoU 阈值
    float keypoint_threshold = 0.5F;  ///< 关键点显示阈值
    bool debug = false;               ///< 是否输出调试信息
};

/**
 * @brief 绘制配置，控制 drawDetections 的行为。
 */
struct DrawConfig {
    std::vector<std::string> class_names;                                    ///< 类别名列表
    std::function<cv::Scalar(const std::string&)> class_color_fn;            ///< 类别名 → BGR 颜色
    cv::Scalar keypoint_color = cv::Scalar(0, 255, 0);                      ///< 关键点绘制颜色（BGR 绿色）
    std::vector<std::pair<size_t, size_t>> edge_pairs;                       ///< 关键点连线配置

    /**
     * @brief 返回 RoboMaster 装甲板检测的默认绘制配置。
     * @return 包含 B1-B5/BO/BS 和 R1-R5/RO/RS 类别、红蓝配色和四边形连线的配置。
     */
    static DrawConfig robomasterDefaults();
};

/**
 * @brief YOLO pose OpenVINO 推理类。
 *
 * 该类封装模型加载、letterbox 前处理、OpenVINO 同步推理、YOLO pose 后处理、NMS 和绘制工具。
 * 使用 PIMPL 隐藏 OpenVINO 类型，公共头文件不依赖 OpenVINO。
 */
class YoloPoseOpenVINO {
public:
    /**
     * @brief 构造推理器并加载模型。
     * @param config 推理配置。
     */
    explicit YoloPoseOpenVINO(const InferConfig& config);

    /**
     * @brief 析构推理器。
     */
    ~YoloPoseOpenVINO();

    // Move-only：OpenVINO 句柄不可拷贝。
    YoloPoseOpenVINO(YoloPoseOpenVINO&&) noexcept;
    YoloPoseOpenVINO& operator=(YoloPoseOpenVINO&&) noexcept;
    YoloPoseOpenVINO(const YoloPoseOpenVINO&) = delete;
    YoloPoseOpenVINO& operator=(const YoloPoseOpenVINO&) = delete;

    /**
     * @brief 对一帧图像执行姿态估计。
     * @param frame 输入 BGR 图像。
     * @return 姿态检测结果列表。
     */
    std::vector<PoseDetection> infer(const cv::Mat& frame);

    /**
     * @brief 获取最近一帧的耗时统计。
     * @return 最近一次 infer 的耗时信息。
     */
    const LatencyInfo& last_latency() const;

    /**
     * @brief 在图像上绘制检测框和关键点。
     * @param image 需要绘制的 BGR 图像。
     * @param detections 检测结果。
     * @param keypoint_threshold 关键点显示阈值。
     * @param draw_config 绘制配置，默认使用 RoboMaster 装甲板配置。
     */
    static void drawDetections(cv::Mat& image,
                               const std::vector<PoseDetection>& detections,
                               float keypoint_threshold,
                               const DrawConfig& draw_config = DrawConfig::robomasterDefaults());

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qd
