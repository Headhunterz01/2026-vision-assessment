#include "armor_ekf/pipeline.hpp"
#include "armor_ekf/result_csv.hpp"
#include "armor_ekf/reprojection_csv.hpp"
#include <opencv2/videoio.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>

int main(int argc, char ** argv) {
  try {
    armor_ekf::Config cfg;
    std::string video, output;
    int stride = 1, max_frames = 0;
    bool save_video = true;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--reproject") { cfg.reprojection.enabled = true; cfg.yaw_frame = "odom"; continue; }
      if (arg == "--no-video") { save_video = false; continue; }
      if (i + 1 >= argc) throw std::invalid_argument("Missing argument value");
      const std::string value = argv[++i];
      if (arg == "--model") cfg.model_path = value;
      else if (arg == "--gimbal-pitch-rad") cfg.frames.odom_gimbal_rpy[1] = std::stod(value);
      else if (arg == "--robot-radius") cfg.reprojection.radius = std::stod(value);
      else if (arg == "--model-pitch-rad") cfg.reprojection.pitch = std::stod(value);
      else if (arg == "--model-roll-rad") cfg.reprojection.roll = std::stod(value);
      else if (arg == "--yaw-frame") cfg.yaw_frame = value;
      else if (arg == "--video") video = value;
      else if (arg == "--output-dir") output = value;
      else if (arg == "--stride") stride = std::stoi(value);
      else if (arg == "--max-frames") max_frames = std::stoi(value);
      else if (arg == "--confidence") cfg.confidence = std::stod(value);
      else if (arg == "--measurement-std") cfg.measurement_std = std::stod(value);
      else if (arg == "--acceleration-std") cfg.acceleration_std = std::stod(value);
      else if (arg == "--association-gate-scale") cfg.association_gate_scale = std::stod(value);
      else if (arg == "--association-mode") cfg.association_mode = value;
      else if (arg == "--target-switch-timeout") cfg.target_switch_timeout = std::stod(value);
      else throw std::invalid_argument("Unknown option: " + arg);
    }
    if (video.empty() || output.empty() || stride < 1 || max_frames < 0)
      throw std::invalid_argument("Usage: armor_ekf_offline --model MODEL --video VIDEO --output-dir DIR [--stride N] [--max-frames N] [--no-video]");
    cv::VideoCapture capture(video);
    if (!capture.isOpened()) throw std::runtime_error("Cannot open video: " + video);
    const double fps = capture.get(cv::CAP_PROP_FPS);
    if (!(fps > 0)) throw std::runtime_error("Invalid video frame rate");
    armor_ekf::Pipeline pipeline(cfg);
    std::filesystem::create_directories(output);
    std::ofstream csv(output + "/yaw.csv");
    if (!csv) throw std::runtime_error("Cannot write CSV");
    armor_ekf::writeResultHeader(csv);
    std::ofstream detections_csv(output + "/detections.csv");
    if (!detections_csv) throw std::runtime_error("Cannot write detection cache");
    detections_csv << "frame,time,detection_index,class_id,confidence,x,y,width,height,k0x,k0y,k1x,k1y,k2x,k2y,k3x,k3y\n" << std::setprecision(17);
    std::ofstream projection_csv;
    if (cfg.reprojection.enabled) {
      projection_csv.open(output + "/reprojection.csv");
      if (!projection_csv) throw std::runtime_error("Cannot write reprojection CSV");
      armor_ekf::writeProjectionHeader(projection_csv);
    }
    cv::VideoWriter writer;
    if (save_video) {
      writer.open(output + "/annotated.mp4", cv::VideoWriter::fourcc('m','p','4','v'), fps / stride,
        cv::Size(int(capture.get(cv::CAP_PROP_FRAME_WIDTH)), int(capture.get(cv::CAP_PROP_FRAME_HEIGHT))));
      if (!writer.isOpened()) throw std::runtime_error("Cannot write annotated video");
    }
    const auto start = std::chrono::steady_clock::now();
    int frame_index = 0, processed = 0, detected = 0, used = 0, resets = 0;
    long long confidence_candidates = 0, valid_corner_sets = 0, pnp_attempts = 0;
    long long valid_poses = 0, association_rejects = 0, ekf_rejects = 0;
    cv::Mat image;
    while (!max_frames || processed < max_frames) {
      if (!capture.grab()) break;
      const int index = frame_index++;
      if (index % stride != 0) continue;
      if (!capture.retrieve(image)) throw std::runtime_error("Video frame decode failed");
      const auto r = pipeline.process(image, index / fps);
      armor_ekf::writeResult(csv, index, index / fps, r);
      if (projection_csv.is_open()) armor_ekf::writeProjection(projection_csv, index, index / fps, r.projection);
      if (r.detections.empty()) detections_csv << index << ',' << index / fps << ",-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0\n";
      for (size_t di = 0; di < r.detections.size(); ++di) {
        const auto & d = r.detections[di];
        detections_csv << index << ',' << index / fps << ',' << di << ',' << d.class_id << ',' << d.confidence
                       << ',' << d.box.x << ',' << d.box.y << ',' << d.box.width << ',' << d.box.height;
        for (const auto & corner : d.corners) detections_csv << ',' << corner.x << ',' << corner.y;
        detections_csv << '\n';
      }
      ++processed; detected += r.detection_valid; used += r.measurement_used; resets += r.target_reset;
      confidence_candidates += r.confidence_class_pass;
      valid_corner_sets += r.valid_corner_sets;
      pnp_attempts += r.pnp_attempts;
      valid_poses += r.valid_poses;
      association_rejects += r.association_rejected;
      ekf_rejects += r.ekf_rejected;
      if (save_video) writer.write(pipeline.annotate(image, r));
      if (processed % 500 == 0) std::cout << "Processed " << processed << " frames\n" << std::flush;
    }
    writer.release(); capture.release(); csv.close();
    if (processed == 0) throw std::runtime_error("No frames processed");
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream summary(output + "/summary.txt");
    summary << "yaw_frame=" << cfg.yaw_frame << '\n';
    summary << "processed_frames=" << processed << "\nselected_pose_frames=" << detected
            << "\nmeasurements_used=" << used << "\nconfidence_class_candidates=" << confidence_candidates
            << "\nvalid_corner_sets=" << valid_corner_sets << "\npnp_attempts=" << pnp_attempts
            << "\nvalid_pnp_poses=" << valid_poses << "\nassociation_reject_frames=" << association_rejects
            << "\nekf_reject_frames=" << ekf_rejects << "\ntrack_resets=" << resets
            << "\nsource_fps=" << fps << "\nstride=" << stride
            << "\nprocessing_seconds=" << seconds << "\nprocessing_fps=" << processed / seconds << '\n';
    std::cout << "Frames=" << processed << " selected_pose_frames=" << detected << " used=" << used << " resets=" << resets
              << " speed=" << processed/seconds << " FPS\n";
    return 0;
  } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
