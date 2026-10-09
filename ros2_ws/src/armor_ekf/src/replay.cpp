#include "armor_ekf/pipeline.hpp"
#include "armor_ekf/result_csv.hpp"
#include "armor_ekf/reprojection_csv.hpp"
#include <opencv2/videoio.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>

int main(int argc, char ** argv) {
  try {
    armor_ekf::Config cfg;
    std::string cache, output, video;
    int max_frames = 0;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--reproject") { cfg.reprojection.enabled = true; cfg.yaw_frame = "odom"; continue; }
      if (i + 1 >= argc) throw std::invalid_argument("Missing argument value");
      const std::string arg = argv[i], value = argv[++i];
      if (arg == "--model") cfg.model_path = value;
      else if (arg == "--gimbal-pitch-rad") cfg.frames.odom_gimbal_rpy[1] = std::stod(value);
      else if (arg == "--robot-radius") cfg.reprojection.radius = std::stod(value);
      else if (arg == "--model-pitch-rad") cfg.reprojection.pitch = std::stod(value);
      else if (arg == "--model-roll-rad") cfg.reprojection.roll = std::stod(value);
      else if (arg == "--yaw-frame") cfg.yaw_frame = value;
      else if (arg == "--detections") cache = value;
      else if (arg == "--output-dir") output = value;
      else if (arg == "--video") video = value;
      else if (arg == "--max-frames") max_frames = std::stoi(value);
      else if (arg == "--association-mode") cfg.association_mode = value;
      else if (arg == "--target-switch-timeout") cfg.target_switch_timeout = std::stod(value);
      else throw std::invalid_argument("Unknown option: " + arg);
    }
    if (cache.empty() || output.empty()) throw std::invalid_argument("--model, --detections and --output-dir are required");
    std::ifstream in(cache);
    if (!in) throw std::runtime_error("Cannot read detection cache");
    std::filesystem::create_directories(output);
    std::ofstream out(output + "/yaw.csv");
    if (!out) throw std::runtime_error("Cannot write replay results");
    armor_ekf::writeResultHeader(out);
    std::ofstream projection_csv;
    if (cfg.reprojection.enabled) {
      projection_csv.open(output + "/reprojection.csv");
      if (!projection_csv) throw std::runtime_error("Cannot write reprojection CSV");
      armor_ekf::writeProjectionHeader(projection_csv);
    }
    armor_ekf::Pipeline pipeline(cfg);
    cv::VideoCapture capture;
    cv::VideoWriter writer;
    if (!video.empty()) {
      capture.open(video);
      if (!capture.isOpened()) throw std::runtime_error("Cannot read source video");
    }
    int group_frame = -1, processed = 0, observed = 0, used = 0, resets = 0, source_frame = -1;
    double group_stamp = 0, previous_stamp = -1;
    std::vector<armor_ekf::Detection> detections;
    auto flush = [&]() {
      if (group_frame < 0 || (max_frames > 0 && processed >= max_frames)) return;
      const auto r = pipeline.processDetections(detections, group_stamp);
      armor_ekf::writeResult(out, group_frame, group_stamp, r);
      if (projection_csv.is_open()) armor_ekf::writeProjection(projection_csv, group_frame, group_stamp, r.projection);
      ++processed; observed += r.detection_valid; used += r.measurement_used; resets += r.target_reset;
      if (capture.isOpened()) {
        cv::Mat image;
        while (source_frame < group_frame) {
          if (!capture.read(image)) throw std::runtime_error("Source video ended before cache");
          ++source_frame;
        }
        if (previous_stamp >= 0 && !writer.isOpened()) {
          writer.open(output + "/annotated.mp4", cv::VideoWriter::fourcc('m','p','4','v'),
                      1.0 / (group_stamp - previous_stamp), image.size());
          if (!writer.isOpened()) throw std::runtime_error("Cannot write replay video");
        }
        // The first image is held until the second timestamp establishes cadence.
        static cv::Mat first_image;
        if (previous_stamp < 0) first_image = pipeline.annotate(image, r);
        else {
          if (!first_image.empty()) { writer.write(first_image); first_image.release(); }
          writer.write(pipeline.annotate(image, r));
        }
      }
      previous_stamp = group_stamp;
    };
    std::string line;
    std::getline(in, line);
    while (std::getline(in, line)) {
      std::stringstream ss(line); std::string field; std::vector<double> v;
      while (std::getline(ss, field, ',')) v.push_back(std::stod(field));
      if (v.size() != 17) throw std::runtime_error("Expected 17 columns in detection cache");
      const int frame = static_cast<int>(v[0]);
      if (group_frame != frame) {
        flush();
        if (max_frames > 0 && processed >= max_frames) break;
        group_frame = frame; group_stamp = v[1]; detections.clear();
      }
      if (v[2] < 0) continue;
      armor_ekf::Detection d;
      d.class_id = static_cast<int>(v[3]); d.confidence = v[4];
      d.box = cv::Rect2f(v[5],v[6],v[7],v[8]);
      for (int k = 0; k < 4; ++k) d.corners[k] = cv::Point2f(v[9+2*k],v[10+2*k]);
      detections.push_back(d);
    }
    flush();
    if (processed == 0) throw std::runtime_error("Empty detection cache");
    std::cout << "frames=" << processed << " observed=" << observed << " used=" << used << " resets=" << resets << '\n';
    return 0;
  } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
