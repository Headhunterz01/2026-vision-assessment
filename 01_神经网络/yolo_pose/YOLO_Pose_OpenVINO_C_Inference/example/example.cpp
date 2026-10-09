#include "qd/yolo_pose.hpp"

#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

/**
 * @brief 命令行参数集合。
 */
struct AppConfig {
    std::string model_path;
    std::string video_path;
    std::string save_dir;
    std::string device = "CPU";
    int input_size = 640;
    int save_every = 0;
    float conf_threshold = 0.25F;
    float iou_threshold = 0.45F;
    float keypoint_threshold = 0.5F;
    bool debug = false;
    bool headless = false;
};

/**
 * @brief 打印使用说明。
 */
void printUsage(const char* program) {
    std::cout << "用法:\n"
              << "  " << program << " --model <best.onnx> --video <video.mp4> [选项]\n\n"
              << "选项:\n"
              << "  --device <CPU|GPU|AUTO>       OpenVINO 推理设备，默认 CPU\n"
              << "  --input-size <int>            模型输入尺寸，默认 640\n"
              << "  --conf <float>                检测置信度阈值，默认 0.25\n"
              << "  --iou <float>                 NMS IoU 阈值，默认 0.45\n"
              << "  --kpt-conf <float>            关键点显示阈值，默认 0.5\n"
              << "  --save-dir <dir>              每隔指定帧数保存静态可视化图片（不写视频）\n"
              << "  --save-every <int>            保存间隔帧数，默认 300\n"
              << "  --headless                    不创建显示窗口\n"
              << "  --debug                       输出前处理/推理/后处理耗时\n"
              << "  --help                        显示帮助\n";
}

/**
 * @brief 解析命令行参数。
 */
AppConfig parseArgs(int argc, char** argv) {
    AppConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("参数缺少取值: " + name);
            }
            return argv[++i];
        };

        if (arg == "--model") {
            config.model_path = requireValue(arg);
        } else if (arg == "--video") {
            config.video_path = requireValue(arg);
        } else if (arg == "--device") {
            config.device = requireValue(arg);
        } else if (arg == "--input-size") {
            config.input_size = std::stoi(requireValue(arg));
        } else if (arg == "--conf") {
            config.conf_threshold = std::stof(requireValue(arg));
        } else if (arg == "--iou") {
            config.iou_threshold = std::stof(requireValue(arg));
        } else if (arg == "--kpt-conf") {
            config.keypoint_threshold = std::stof(requireValue(arg));
        } else if (arg == "--save-dir") {
            config.save_dir = requireValue(arg);
        } else if (arg == "--save-every") {
            config.save_every = std::stoi(requireValue(arg));
        } else if (arg == "--headless") {
            config.headless = true;
        } else if (arg == "--debug") {
            config.debug = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("未知参数: " + arg);
        }
    }

    if (config.model_path.empty() || config.video_path.empty()) {
        printUsage(argv[0]);
        throw std::runtime_error("必须提供 --model 和 --video");
    }
    if (config.save_every < 0) {
        throw std::runtime_error("--save-every 必须大于或等于 0");
    }
    if (!config.save_dir.empty() && config.save_every == 0) {
        config.save_every = 300;
    }
    return config;
}

}  // namespace

/**
 * @brief 程序入口：打开视频，逐帧执行 YOLO pose OpenVINO 推理并显示结果。
 */
int main(int argc, char** argv) {
    try {
        const AppConfig config = parseArgs(argc, argv);

        std::cout << "[INFO] model=" << config.model_path << std::endl;
        std::cout << "[INFO] video=" << config.video_path << std::endl;
        std::cout << "[INFO] device=" << config.device << std::endl;

        qd::InferConfig infer_config;
        infer_config.model_path = config.model_path;
        infer_config.device_name = config.device;
        infer_config.input_size = config.input_size;
        infer_config.conf_threshold = config.conf_threshold;
        infer_config.iou_threshold = config.iou_threshold;
        infer_config.keypoint_threshold = config.keypoint_threshold;
        infer_config.debug = config.debug;

        qd::YoloPoseOpenVINO detector(infer_config);
        qd::DrawConfig draw_config;
        for (int i = 0; i < 13; ++i) {
            draw_config.class_names.push_back("armor_" + std::to_string(i));
        }
        draw_config.class_color_fn = [](const std::string&) { return cv::Scalar(0, 220, 255); };
        draw_config.keypoint_color = cv::Scalar(255, 80, 0);
        draw_config.edge_pairs = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
        if (!config.save_dir.empty()) {
            std::filesystem::create_directories(config.save_dir);
        }

        cv::VideoCapture cap(config.video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("无法打开视频: " + config.video_path);
        }

        const double fps = cap.get(cv::CAP_PROP_FPS);
        const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        std::cout << "[INFO] video_info=" << width << "x" << height << " fps=" << fps << std::endl;

        cv::Mat frame;
        int frame_index = 0;
        int frames_with_detections = 0;
        size_t detection_count = 0;
        double total_inference_ms = 0.0;
        while (cap.read(frame)) {
            ++frame_index;

            std::vector<qd::PoseDetection> detections = detector.infer(frame);
            qd::YoloPoseOpenVINO::drawDetections(frame, detections, config.keypoint_threshold, draw_config);
            detection_count += detections.size();
            if (!detections.empty()) {
                ++frames_with_detections;
            }

            const qd::LatencyInfo& latency = detector.last_latency();
            total_inference_ms += latency.inference_ms;
            const std::string latency_text = "pre " + std::to_string(latency.preprocess_ms).substr(0, 5) +
                                             " ms | infer " + std::to_string(latency.inference_ms).substr(0, 5) +
                                             " ms | post " + std::to_string(latency.postprocess_ms).substr(0, 5) + " ms";
            cv::putText(frame, latency_text, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

            if (config.debug) {
                std::cout << "[DEBUG] frame=" << frame_index
                          << " det=" << detections.size()
                          << " preprocess(前处理)=" << latency.preprocess_ms << " ms"
                          << " inference(推理)=" << latency.inference_ms << " ms"
                          << " postprocess(后处理)=" << latency.postprocess_ms << " ms"
                          << " total(总耗时)=" << latency.total_ms << " ms" << std::endl;
            }

            if (!config.save_dir.empty() && (frame_index == 1 || frame_index % config.save_every == 0)) {
                std::ostringstream name;
                name << "frame_" << std::setw(6) << std::setfill('0') << frame_index << ".png";
                const auto path = std::filesystem::path(config.save_dir) / name.str();
                if (!cv::imwrite(path.string(), frame)) {
                    throw std::runtime_error("无法保存可视化图片: " + path.string());
                }
            }

            if (!config.headless) {
                cv::imshow("YOLO Pose OpenVINO Demo", frame);
                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q') {
                    break;
                }
            }
        }

        const double mean_inference_ms = frame_index > 0 ? total_inference_ms / frame_index : 0.0;
        std::cout << "[INFO] 处理完成，总帧数: " << frame_index
                  << "，有检测帧数: " << frames_with_detections
                  << "，检测总数: " << detection_count
                  << "，平均推理耗时: " << mean_inference_ms << " ms" << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
