#include "qd/yolo_pose.hpp"

#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace qd {

namespace {

/**
 * @brief 计算时间间隔，单位毫秒。
 */
double elapsedMs(const std::chrono::steady_clock::time_point& begin,
                 const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

/**
 * @brief 将数值限制在指定范围内。
 */
float clampFloat(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

/**
 * @brief 计算两个矩形框的 IoU。
 */
float iou(const cv::Rect& a, const cv::Rect& b) {
    const int inter_area = (a & b).area();
    const int union_area = a.area() + b.area() - inter_area;
    if (union_area <= 0) {
        return 0.0F;
    }
    return static_cast<float>(inter_area) / static_cast<float>(union_area);
}

}  // namespace

/**
 * @brief PIMPL 实现结构，存放所有 OpenVINO 类型及私有方法。
 */
struct YoloPoseOpenVINO::Impl {
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    std::string input_name;
    std::string output_name;
    int input_size = 640;
    float conf_threshold = 0.25F;
    float iou_threshold = 0.45F;
    float keypoint_threshold = 0.5F;
    bool debug = false;
    LatencyInfo last_latency;

    /**
     * @brief 融合后的前处理结果。
     */
    struct PreprocessResult {
        ov::Tensor tensor;       ///< NHWC U8 BGR 输入张量
        cv::Mat letterbox_image; ///< 持有 tensor 引用的图像内存
        float scale = 1.0F;     ///< 原图到输入图的缩放比例
        int pad_x = 0;           ///< 左侧填充像素
        int pad_y = 0;           ///< 上侧填充像素
        double resize_ms = 0.0;  ///< resize 耗时
        double fill_ms = 0.0;    ///< letterbox 背景填充耗时
        double write_ms = 0.0;   ///< 写入耗时
    };

    PreprocessResult preprocess(const cv::Mat& frame) const;
    std::vector<PoseDetection> postprocess(const ov::Tensor& output_tensor,
                                           const cv::Size& original_size,
                                           float scale,
                                           int pad_x,
                                           int pad_y) const;
};

// ============================================================================
// YoloPoseOpenVINO 公共方法
// ============================================================================

YoloPoseOpenVINO::YoloPoseOpenVINO(const InferConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->input_size = config.input_size;
    impl_->conf_threshold = config.conf_threshold;
    impl_->iou_threshold = config.iou_threshold;
    impl_->keypoint_threshold = config.keypoint_threshold;
    impl_->debug = config.debug;

    if (impl_->input_size <= 0) {
        throw std::invalid_argument("input_size 必须大于 0");
    }

    if (impl_->debug) {
        std::cout << "[DEBUG] 读取模型: " << config.model_path << std::endl;
        std::cout << "[DEBUG] OpenVINO 设备: " << config.device_name << std::endl;
    }

    auto model = impl_->core.read_model(config.model_path);

    // 固定模型本体输入尺寸，和导出 ONNX 时的 imgsz 保持一致。
    const ov::Shape model_input_shape{1, 3,
                                      static_cast<size_t>(impl_->input_size),
                                      static_cast<size_t>(impl_->input_size)};
    model->reshape({{model->input().get_any_name(), model_input_shape}});

    // 使用 OpenVINO PrePostProcessor 接收 NHWC/U8/BGR 输入。
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_shape({1, impl_->input_size, impl_->input_size, 3})
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    ppp.input().model().set_layout("NCHW");
    ppp.input().preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale(255.0F)
        .convert_layout("NCHW");
    model = ppp.build();

    impl_->compiled_model = impl_->core.compile_model(model, config.device_name);
    impl_->infer_request = impl_->compiled_model.create_infer_request();
    impl_->input_name = impl_->compiled_model.input().get_any_name();
    impl_->output_name = impl_->compiled_model.output().get_any_name();

    if (impl_->debug) {
        std::cout << "[DEBUG] 输入节点: " << impl_->input_name << std::endl;
        std::cout << "[DEBUG] 输出节点: " << impl_->output_name << std::endl;
        std::cout << "[DEBUG] 输入尺寸: " << impl_->input_size << "x" << impl_->input_size << std::endl;
    }
}

YoloPoseOpenVINO::~YoloPoseOpenVINO() = default;

YoloPoseOpenVINO::YoloPoseOpenVINO(YoloPoseOpenVINO&&) noexcept = default;
YoloPoseOpenVINO& YoloPoseOpenVINO::operator=(YoloPoseOpenVINO&&) noexcept = default;

std::vector<PoseDetection> YoloPoseOpenVINO::infer(const cv::Mat& frame) {
    if (frame.empty()) {
        throw std::invalid_argument("输入帧为空");
    }

    const auto total_begin = std::chrono::steady_clock::now();

    const auto preprocess_begin = std::chrono::steady_clock::now();
    Impl::PreprocessResult pp = impl_->preprocess(frame);
    const auto preprocess_end = std::chrono::steady_clock::now();

    const auto inference_begin = std::chrono::steady_clock::now();
    impl_->infer_request.set_tensor(impl_->input_name, pp.tensor);
    impl_->infer_request.infer();
    ov::Tensor output_tensor = impl_->infer_request.get_tensor(impl_->output_name);
    const auto inference_end = std::chrono::steady_clock::now();

    const auto postprocess_begin = std::chrono::steady_clock::now();
    std::vector<PoseDetection> detections =
        impl_->postprocess(output_tensor, frame.size(), pp.scale, pp.pad_x, pp.pad_y);
    const auto postprocess_end = std::chrono::steady_clock::now();

    const auto total_end = std::chrono::steady_clock::now();
    impl_->last_latency.preprocess_ms = elapsedMs(preprocess_begin, preprocess_end);
    impl_->last_latency.preprocess_resize_ms = pp.resize_ms;
    impl_->last_latency.preprocess_fill_ms = pp.fill_ms;
    impl_->last_latency.preprocess_write_ms = pp.write_ms;
    impl_->last_latency.inference_ms = elapsedMs(inference_begin, inference_end);
    impl_->last_latency.postprocess_ms = elapsedMs(postprocess_begin, postprocess_end);
    impl_->last_latency.total_ms = elapsedMs(total_begin, total_end);

    if (impl_->debug) {
        std::cout << "[DEBUG] detections=" << detections.size()
                  << " preprocess(前处理)=" << impl_->last_latency.preprocess_ms << " ms"
                  << " inference(推理)=" << impl_->last_latency.inference_ms << " ms"
                  << " postprocess(后处理)=" << impl_->last_latency.postprocess_ms << " ms"
                  << " total(总耗时)=" << impl_->last_latency.total_ms << " ms" << std::endl;
    }

    return detections;
}

const LatencyInfo& YoloPoseOpenVINO::last_latency() const {
    return impl_->last_latency;
}

void YoloPoseOpenVINO::drawDetections(cv::Mat& image,
                                       const std::vector<PoseDetection>& detections,
                                       float keypoint_threshold,
                                       const DrawConfig& draw_config) {
    auto className = [&](int class_id) -> std::string {
        if (class_id >= 0 && static_cast<size_t>(class_id) < draw_config.class_names.size()) {
            return draw_config.class_names[static_cast<size_t>(class_id)];
        }
        return "ID" + std::to_string(class_id);
    };

    auto classColor = [&](const std::string& name) -> cv::Scalar {
        if (draw_config.class_color_fn) {
            return draw_config.class_color_fn(name);
        }
        return cv::Scalar(0, 255, 0);  // 无颜色函数时使用绿色兜底
    };

    for (const PoseDetection& det : detections) {
        const std::string name = className(det.class_id);
        const cv::Scalar color = classColor(name);
        cv::rectangle(image, det.box, color, 2);
        const std::string label = name + " " + std::to_string(det.confidence).substr(0, 4);
        cv::putText(image, label, cv::Point(det.box.x, std::max(0, det.box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

        // 绘制关键点。
        for (size_t i = 0; i < det.keypoints.size(); ++i) {
            const KeyPoint& kp = det.keypoints[i];
            if (kp.confidence < keypoint_threshold) {
                continue;
            }
            cv::circle(image,
                       cv::Point(static_cast<int>(std::round(kp.x)),
                                 static_cast<int>(std::round(kp.y))),
                       4, draw_config.keypoint_color, -1);
            cv::putText(image, std::to_string(i),
                        cv::Point(static_cast<int>(std::round(kp.x)) + 4,
                                  static_cast<int>(std::round(kp.y)) - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, draw_config.keypoint_color, 1);
        }

        // 绘制关键点连线。
        for (const auto& edge : draw_config.edge_pairs) {
            if (edge.first >= det.keypoints.size() || edge.second >= det.keypoints.size()) {
                continue;
            }
            const KeyPoint& a = det.keypoints[edge.first];
            const KeyPoint& b = det.keypoints[edge.second];
            if (a.confidence < keypoint_threshold || b.confidence < keypoint_threshold) {
                continue;
            }
            cv::line(image,
                     cv::Point(static_cast<int>(std::round(a.x)),
                               static_cast<int>(std::round(a.y))),
                     cv::Point(static_cast<int>(std::round(b.x)),
                               static_cast<int>(std::round(b.y))),
                     draw_config.keypoint_color,
                     1);
        }
    }
}

// ============================================================================
// DrawConfig 工厂方法
// ============================================================================

DrawConfig DrawConfig::robomasterDefaults() {
    DrawConfig config;
    config.class_names = {
        "B1", "B2", "B3", "B4", "B5", "BO", "BS",
        "R1", "R2", "R3", "R4", "R5", "RO", "RS",
    };
    config.class_color_fn = [](const std::string& name) -> cv::Scalar {
        if (!name.empty() && name[0] == 'B') {
            return cv::Scalar(255, 0, 0);  // 蓝方：BGR (255, 0, 0)
        }
        if (!name.empty() && name[0] == 'R') {
            return cv::Scalar(0, 0, 255);  // 红方：BGR (0, 0, 255)
        }
        return cv::Scalar(0, 255, 0);      // 未知类别：绿色兜底
    };
    config.keypoint_color = cv::Scalar(0, 255, 0);  // 绿色
    config.edge_pairs = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};  // 闭合四边形
    return config;
}

// ============================================================================
// Impl 私有方法
// ============================================================================

YoloPoseOpenVINO::Impl::PreprocessResult
YoloPoseOpenVINO::Impl::preprocess(const cv::Mat& frame) const {
    const int width = frame.cols;
    const int height = frame.rows;
    const float scale = std::min(static_cast<float>(input_size) / static_cast<float>(width),
                                 static_cast<float>(input_size) / static_cast<float>(height));
    const int new_width = static_cast<int>(std::round(static_cast<float>(width) * scale));
    const int new_height = static_cast<int>(std::round(static_cast<float>(height) * scale));
    const int pad_x = (input_size - new_width) / 2;
    const int pad_y = (input_size - new_height) / 2;

    const auto resize_begin = std::chrono::steady_clock::now();
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_width, new_height));
    const auto resize_end = std::chrono::steady_clock::now();

    const auto fill_begin = std::chrono::steady_clock::now();
    cv::Mat letterbox_image(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
    const auto fill_end = std::chrono::steady_clock::now();

    const auto write_begin = std::chrono::steady_clock::now();
    resized.copyTo(letterbox_image(cv::Rect(pad_x, pad_y, new_width, new_height)));
    ov::Tensor tensor(ov::element::u8,
                      {1, static_cast<size_t>(input_size), static_cast<size_t>(input_size), 3},
                      letterbox_image.data);
    const auto write_end = std::chrono::steady_clock::now();

    return PreprocessResult{
        tensor,
        letterbox_image,
        scale,
        pad_x,
        pad_y,
        elapsedMs(resize_begin, resize_end),
        elapsedMs(fill_begin, fill_end),
        elapsedMs(write_begin, write_end),
    };
}

std::vector<PoseDetection>
YoloPoseOpenVINO::Impl::postprocess(const ov::Tensor& output_tensor,
                                    const cv::Size& original_size,
                                    float scale,
                                    int pad_x,
                                    int pad_y) const {
    const ov::Shape shape = output_tensor.get_shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("不支持的 YOLO pose 输出维度，期望 [1, C, N] 或 [1, N, C]");
    }

    const float* output = output_tensor.data<const float>();

    const size_t dim1 = shape[1];
    const size_t dim2 = shape[2];
    const bool transposed = dim1 < dim2;
    const size_t attributes = transposed ? dim1 : dim2;
    const size_t anchors = transposed ? dim2 : dim1;

    if (attributes < 6) {
        throw std::runtime_error("输出 attributes 太少，无法解析 bbox/conf/keypoints");
    }

    const size_t keypoint_values_from_pose = attributes >= 6 ? attributes - 6 : 0;
    const bool end2end_xyxy = keypoint_values_from_pose > 0 && (keypoint_values_from_pose % 2 == 0) && anchors <= 1000;
    const size_t keypoint_values = end2end_xyxy ? keypoint_values_from_pose : attributes - 5;
    const size_t keypoint_stride = end2end_xyxy ? 2 : 3;
    const size_t num_keypoints = keypoint_values / keypoint_stride;

    if (debug) {
        std::cout << "[DEBUG] output_shape=[" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]"
                  << " attributes=" << attributes
                  << " anchors=" << anchors
                  << " keypoints=" << num_keypoints
                  << " format=" << (end2end_xyxy ? "xyxy_score_class_kpts" : "xywh_score_kpts")
                  << std::endl;
    }

    auto at = [&](size_t anchor, size_t attr) -> float {
        if (transposed) {
            return output[attr * anchors + anchor];
        }
        return output[anchor * attributes + attr];
    };

    std::vector<PoseDetection> candidates;
    candidates.reserve(128);

    for (size_t i = 0; i < anchors; ++i) {
        const float score = at(i, 4);

        if (score < conf_threshold) {
            continue;
        }

        float x1 = 0.0F;
        float y1 = 0.0F;
        float x2 = 0.0F;
        float y2 = 0.0F;
        if (end2end_xyxy) {
            x1 = (at(i, 0) - static_cast<float>(pad_x)) / scale;
            y1 = (at(i, 1) - static_cast<float>(pad_y)) / scale;
            x2 = (at(i, 2) - static_cast<float>(pad_x)) / scale;
            y2 = (at(i, 3) - static_cast<float>(pad_y)) / scale;
        } else {
            const float cx = at(i, 0);
            const float cy = at(i, 1);
            const float w = at(i, 2);
            const float h = at(i, 3);
            x1 = (cx - w * 0.5F - static_cast<float>(pad_x)) / scale;
            y1 = (cy - h * 0.5F - static_cast<float>(pad_y)) / scale;
            x2 = (cx + w * 0.5F - static_cast<float>(pad_x)) / scale;
            y2 = (cy + h * 0.5F - static_cast<float>(pad_y)) / scale;
        }

        x1 = clampFloat(x1, 0.0F, static_cast<float>(original_size.width - 1));
        y1 = clampFloat(y1, 0.0F, static_cast<float>(original_size.height - 1));
        x2 = clampFloat(x2, 0.0F, static_cast<float>(original_size.width - 1));
        y2 = clampFloat(y2, 0.0F, static_cast<float>(original_size.height - 1));

        PoseDetection det;
        det.box = cv::Rect(cv::Point(static_cast<int>(std::round(x1)),
                                     static_cast<int>(std::round(y1))),
                           cv::Point(static_cast<int>(std::round(x2)),
                                     static_cast<int>(std::round(y2))));
        det.confidence = score;
        det.class_id = end2end_xyxy ? static_cast<int>(std::round(at(i, 5))) : 0;
        det.keypoints.reserve(num_keypoints);

        for (size_t k = 0; k < num_keypoints; ++k) {
            const size_t base = end2end_xyxy ? 6 + k * keypoint_stride : 5 + k * keypoint_stride;
            const float kx = (at(i, base) - static_cast<float>(pad_x)) / scale;
            const float ky = (at(i, base + 1) - static_cast<float>(pad_y)) / scale;
            const float kc = end2end_xyxy ? score : at(i, base + 2);
            det.keypoints.push_back(KeyPoint{
                clampFloat(kx, 0.0F, static_cast<float>(original_size.width - 1)),
                clampFloat(ky, 0.0F, static_cast<float>(original_size.height - 1)),
                kc,
            });
        }

        if (det.box.width > 1 && det.box.height > 1) {
            candidates.push_back(std::move(det));
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const PoseDetection& a, const PoseDetection& b) {
        return a.confidence > b.confidence;
    });

    std::vector<PoseDetection> results;
    std::vector<bool> removed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (removed[i]) {
            continue;
        }
        results.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (!removed[j] && iou(candidates[i].box, candidates[j].box) > iou_threshold) {
                removed[j] = true;
            }
        }
    }

    return results;
}

}  // namespace qd