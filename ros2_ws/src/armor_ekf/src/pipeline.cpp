#include "armor_ekf/pipeline.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace armor_ekf {
bool validCorners(const std::array<cv::Point2f, 4> & corners) {
  for (const auto & p : corners) if (!std::isfinite(p.x) || !std::isfinite(p.y)) return false;
  const std::vector<cv::Point2f> polygon(corners.begin(), corners.end());
  if (!cv::isContourConvex(polygon) || std::abs(cv::contourArea(polygon)) < 30) return false;
  for (size_t i = 0; i < 4; ++i)
    if (cv::norm(corners[i] - corners[(i + 1) % 4]) < 4) return false;
  return true;
}

std::vector<Pose> solvePoses(const Detection & d, const Config & cfg, PoseDiagnostics * diagnostics) {
  if (!validCorners(d.corners)) return {};
  if (cfg.yaw_frame != "odom" && cfg.yaw_frame != "camera_optical_frame")
    throw std::invalid_argument("yaw_frame must be odom or camera_optical_frame");
  const TransformChain transforms(cfg.frames);
  const double w = cfg.armor_width / 2, h = cfg.armor_height / 2;
  // Armor X is the outward normal, Y points left from the robot's perspective,
  // Z points up. With the armor facing the camera, its Y points image-right.
  const std::vector<cv::Point3d> object{{0,-w,h}, {0,-w,-h}, {0,w,-h}, {0,w,h}};
  const std::vector<cv::Point2f> image(d.corners.begin(), d.corners.end());
  const cv::Mat K(3, 3, CV_64F, const_cast<double *>(cfg.camera_matrix.data()));
  const cv::Mat D(1, 5, CV_64F, const_cast<double *>(cfg.distortion.data()));
  std::vector<cv::Mat> rotations, translations;
  cv::solvePnPGeneric(object, image, K, D, rotations, translations, false, cv::SOLVEPNP_IPPE);
  if (diagnostics) diagnostics->solutions += static_cast<int>(rotations.size());
  std::vector<Pose> poses;
  for (size_t i = 0; i < rotations.size(); ++i) {
    cv::Vec3d r(rotations[i]), t(translations[i]);
    cv::Matx33d R; cv::Rodrigues(r, R);
    const cv::Vec3d normal(R(0,0), R(1,0), R(2,0));
    if (normal.dot(t) >= 0 || (cfg.yaw_frame == "camera_optical_frame" && std::hypot(normal[0], normal[2]) < 0.2)) {
      if (diagnostics) ++diagnostics->physical_pose_rejects;
      continue;
    }
    bool positive_depth = true;
    for (const auto & p : object)
      if ((R * cv::Vec3d(p.x,p.y,p.z) + t)[2] <= 0) positive_depth = false;
    if (!positive_depth) {
      if (diagnostics) ++diagnostics->physical_pose_rejects;
      continue;
    }
    std::vector<cv::Point2d> projected;
    cv::projectPoints(object, r, t, K, D, projected);
    double squared_error = 0;
    for (size_t k = 0; k < 4; ++k) {
      const auto delta = projected[k] - cv::Point2d(image[k]);
      squared_error += delta.dot(delta);
    }
    const double error = std::sqrt(squared_error / 4);
    const double camera_yaw = std::atan2(normal[0], -normal[2]);
    double yaw = camera_yaw;
    cv::Vec3d output_rvec = r, output_tvec = t;
    if (cfg.yaw_frame == "odom") {
      const auto & world = transforms.odomFromOptical();
      const Eigen::Matrix3d world_rotation = world.linear() * rotationFromRvec(r);
      const auto heading = worldYaw(world_rotation, cfg.min_world_horizontal_normal);
      if (!heading) { if (diagnostics) ++diagnostics->world_yaw_rejects; continue; }
      yaw = *heading;
      output_rvec = rvecFromRotation(world_rotation);
      const Eigen::Vector3d position = world * Eigen::Vector3d(t[0], t[1], t[2]);
      output_tvec = cv::Vec3d(position.x(), position.y(), position.z());
    }
    if (std::isfinite(error) && std::isfinite(yaw) && error <= cfg.max_reprojection_error) {
      poses.push_back({r, t, yaw, error, output_rvec, output_tvec, camera_yaw});
      if (diagnostics) ++diagnostics->valid_poses;
    } else if (diagnostics) {
      ++diagnostics->reprojection_rejects;
    }
  }
  std::sort(poses.begin(), poses.end(), [](const Pose & a, const Pose & b) { return a.error < b.error; });
  return poses;
}

ReprojectionPoseChoice chooseReprojectionPose(const std::vector<Pose> & poses,
    const std::vector<Detection> & detections, size_t selected_index,
    const Config & cfg, const std::optional<Eigen::Vector3d> & trusted_center,
    const std::optional<double> & trusted_yaw) {
  if (poses.empty() || selected_index >= detections.size())
    throw std::invalid_argument("Pose selection requires a valid anchor");
  struct Evidence {
    RobotProjection model;
    double iou = 0;
    double center_distance = std::numeric_limits<double>::infinity();
    bool supported = false;
  };
  std::vector<Evidence> evidence(poses.size());
  const auto camera = TransformChain(cfg.frames).odomFromOptical();
  bool other_detection = false;
  for (size_t i=0; i<poses.size(); ++i) {
    Eigen::Isometry3d anchor = Eigen::Isometry3d::Identity();
    anchor.linear() = rotationFromRvec(poses[i].output_rvec);
    anchor.translation() = Eigen::Vector3d(poses[i].output_tvec[0],poses[i].output_tvec[1],poses[i].output_tvec[2]);
    auto & e = evidence[i];
    e.model = reprojectArmors(anchor,camera,cfg.reprojection,cfg.armor_width,cfg.armor_height,
                              cfg.camera_matrix,cfg.distortion);
    if (!e.model.valid) continue;
    for (size_t j=0; j<detections.size(); ++j) {
      if (j == selected_index || !validCorners(detections[j].corners)) continue;
      other_detection = true;
      const auto & d = detections[j];
      const std::vector<cv::Point2f> observed(d.corners.begin(),d.corners.end());
      const double observed_area = std::abs(cv::contourArea(observed));
      for (size_t k=1; k<4; ++k) {
        const auto & p = e.model.plates[k];
        if (!p.depth_valid || !p.in_image || !p.front_facing) continue;
        cv::Point2d center(0,0);
        std::vector<cv::Point2f> predicted;
        for (const auto & corner : p.pixels) {
          center += corner * 0.25;
          predicted.emplace_back(corner);
        }
        const double distance = cv::norm(center-cv::Point2d(d.center()));
        if (distance > std::max(50.0,0.8*d.box.width)) continue;
        std::vector<cv::Point2f> intersection;
        const double overlap = cv::intersectConvexConvex(predicted,observed,intersection,true);
        const double union_area = observed_area + std::abs(cv::contourArea(predicted)) - overlap;
        const double iou = union_area > 0 ? overlap/union_area : 0;
        if (iou > e.iou || (iou == e.iou && distance < e.center_distance)) {
          e.iou = iou; e.center_distance = distance;
        }
      }
    }
    e.supported = e.iou >= 0.12;
  }
  size_t chosen = 0;
  bool any_support = false;
  for (size_t i=0; i<evidence.size(); ++i) {
    if (!evidence[i].supported) continue;
    if (!any_support || evidence[i].iou > evidence[chosen].iou ||
        (evidence[i].iou == evidence[chosen].iou && poses[i].error < poses[chosen].error)) chosen = i;
    any_support = true;
  }
  bool trusted_match = false;
  if (!any_support && trusted_center) {
    double nearest = std::numeric_limits<double>::infinity(), second = nearest;
    size_t nearest_index = 0;
    for (size_t i=0;i<evidence.size();++i) {
      if (!evidence[i].model.valid) continue;
      const double d = (evidence[i].model.center-*trusted_center).norm();
      if (d < nearest) { second=nearest; nearest=d; nearest_index=i; }
      else if (d < second) second=d;
    }
    if (nearest < 0.18 && second-nearest > 0.08) {
      chosen=nearest_index; trusted_match=true;
    }
  }
  bool yaw_match = false;
  if (!any_support && !trusted_match && trusted_yaw && poses.size()>1) {
    // Hysteresis: keep the current planar branch unless the competing branch
    // improves reprojection error by a clear margin. A tiny per-frame corner
    // change must not flip the armor normal by several degrees.
    constexpr double switch_margin_px = 0.12;
    if (std::abs(wrap(poses[chosen].yaw-*trusted_yaw)) > 0.08) {
      size_t nearest_index = chosen;
      double nearest = std::abs(wrap(poses[chosen].yaw-*trusted_yaw));
      for (size_t i=0;i<poses.size();++i) {
        const double delta=std::abs(wrap(poses[i].yaw-*trusted_yaw));
        if (delta < nearest) { nearest=delta; nearest_index=i; }
      }
      if (nearest_index != chosen && poses[chosen].error <= poses[nearest_index].error+switch_margin_px) {
        chosen=nearest_index; yaw_match=true;
      }
    }
  }
  bool pitch_match = false;
  if (!any_support && !trusted_match && !yaw_match && poses.size()>1) {
    double nearest = std::numeric_limits<double>::infinity(), second = nearest;
    size_t nearest_index = 0;
    for (size_t i=0;i<poses.size();++i) {
      const Eigen::Matrix3d R = rotationFromRvec(poses[i].output_rvec);
      // PDF inward-X pitch maps to the opposite pitch in our outward-X frame.
      const double deviation = std::abs(std::asin(std::clamp(R(2,0),-1.0,1.0))-cfg.reprojection.pitch);
      if (deviation < nearest) { second=nearest; nearest=deviation; nearest_index=i; }
      else if (deviation < second) second=deviation;
    }
    if (nearest < 0.20 && second-nearest > 0.15 &&
        poses[nearest_index].error <= poses.front().error+0.10) {
      chosen=nearest_index; pitch_match=true;
    }
  }
  // A second observed plate contradicting every model candidate is stronger
  // evidence than a small single-plate PnP residual. Avoid drawing ghosts.
  bool ambiguous = other_detection && !any_support && !pitch_match && !yaw_match && !trusted_match;
  if (!any_support && !trusted_match && !pitch_match && !yaw_match && poses.size()>1 && evidence[0].model.valid && evidence[1].model.valid) {
    const double center_separation = (evidence[0].model.center-evidence[1].model.center).norm();
    const double error_gap = poses[1].error-poses[0].error;
    ambiguous = ambiguous || (center_separation > 0.12 && error_gap < 0.5);
  }
  return {chosen,evidence[chosen].model,any_support,ambiguous,evidence[chosen].iou};
}

Pipeline::Pipeline(const Config & cfg)
: config_(cfg), filter_(cfg.measurement_std, cfg.acceleration_std),
  motion_tracker_(cfg.reset_timeout, cfg.association_gate_scale, cfg.target_switch_timeout) {
  if (!std::filesystem::is_regular_file(cfg.model_path)) throw std::invalid_argument("ONNX model not found: " + cfg.model_path);
  if (!(cfg.confidence > 0 && cfg.confidence < 1 && cfg.armor_width > 0 && cfg.armor_height > 0 &&
      cfg.max_reprojection_error > 0 && cfg.reset_timeout > 0 && cfg.association_gate_scale > 0 && cfg.threads > 0))
    throw std::invalid_argument("Invalid detector, pose, or timing parameters");
  if (cfg.association_mode != "legacy" && cfg.association_mode != "motion")
    throw std::invalid_argument("association_mode must be legacy or motion");
  if (cfg.yaw_frame != "odom" && cfg.yaw_frame != "camera_optical_frame")
    throw std::invalid_argument("yaw_frame must be odom or camera_optical_frame");
  if (!(std::isfinite(cfg.min_world_horizontal_normal) && cfg.min_world_horizontal_normal > 0 && cfg.min_world_horizontal_normal <= 1))
    throw std::invalid_argument("min_world_horizontal_normal must be in (0,1]");
  (void)TransformChain(cfg.frames);
  if (cfg.reprojection.enabled && cfg.yaw_frame != "odom")
    throw std::invalid_argument("Reprojection requires yaw_frame=odom");
  if (!(std::isfinite(cfg.reprojection.radius) && cfg.reprojection.radius > 0 &&
        std::isfinite(cfg.reprojection.pitch) && std::isfinite(cfg.reprojection.roll)))
    throw std::invalid_argument("Invalid robot reprojection parameters");
  auto order = cfg.corner_order; std::sort(order.begin(), order.end());
  if (order != std::array<int,4>{0,1,2,3}) throw std::invalid_argument("corner_order must be a permutation of 0,1,2,3");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(cfg.threads);
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_ = std::make_unique<Ort::Session>(env_, cfg.model_path.c_str(), options);
  Ort::AllocatorWithDefaultOptions allocator;
  if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1)
    throw std::invalid_argument("Expected one input and one NMS output");
  input_name_ = session_->GetInputNameAllocated(0, allocator).get();
  output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
  const auto input_type = session_->GetInputTypeInfo(0);
  const auto input_info = input_type.GetTensorTypeAndShapeInfo();
  const auto shape = input_info.GetShape();
  if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] <= 0 || shape[3] <= 0 ||
      input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::invalid_argument("Expected fixed float32 input [1,3,H,W]");
  input_height_ = static_cast<int>(shape[2]); input_width_ = static_cast<int>(shape[3]);
  const auto output_type = session_->GetOutputTypeInfo(0);
  const auto output_info = output_type.GetTensorTypeAndShapeInfo();
  const auto output_shape = output_info.GetShape();
  if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[2] != 14 ||
      output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::invalid_argument("Expected float32 NMS output [1,N,14]: box, score, class, 4 xy points");
}

std::vector<Detection> Pipeline::detect(const cv::Mat & bgr, DetectionDiagnostics & diagnostics) {
  const double scale = std::min(double(input_width_) / bgr.cols, double(input_height_) / bgr.rows);
  const int width = std::lround(bgr.cols * scale), height = std::lround(bgr.rows * scale);
  const int left = (input_width_ - width) / 2, top = (input_height_ - height) / 2;
  cv::Mat padded(input_height_, input_width_, CV_8UC3, cv::Scalar(114,114,114));
  cv::resize(bgr, padded(cv::Rect(left,top,width,height)), cv::Size(width,height));
  const size_t plane = size_t(input_width_) * input_height_;
  std::vector<float> input(plane * 3);
  for (int y = 0; y < input_height_; ++y) {
    const auto * row = padded.ptr<cv::Vec3b>(y);
    for (int x = 0; x < input_width_; ++x)
      for (int c = 0; c < 3; ++c) input[c * plane + y * input_width_ + x] = row[x][2-c] / 255.0f;
  }
  const std::array<int64_t,4> shape{1,3,input_height_,input_width_};
  const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
  const char * in[] = {input_name_.c_str()}, * out[] = {output_name_.c_str()};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, in, &tensor, 1, out, 1);
  const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  if (out_shape.size() != 3 || out_shape[0] != 1 || out_shape[2] != 14)
    throw std::runtime_error("Unexpected inference output shape");
  const float * values = outputs[0].GetTensorData<float>();
  std::vector<Detection> detections;
  diagnostics.model_rows = static_cast<int>(out_shape[1]);
  for (int64_t i = 0; i < out_shape[1]; ++i) {
    const float * row = values + i * 14;
    if (!std::all_of(row, row + 14, [](float v) { return std::isfinite(v); }) ||
        row[4] < config_.confidence || row[4] > 1 || row[5] < 0 || row[5] > 12) continue;
    ++diagnostics.confidence_class_pass;
    Detection d;
    d.confidence = row[4]; d.class_id = std::lround(row[5]);
    d.box = cv::Rect2f((row[0]-left)/scale, (row[1]-top)/scale,
      (row[2]-row[0])/scale, (row[3]-row[1])/scale);
    if (d.box.width <= 0 || d.box.height <= 0) continue;
    ++diagnostics.valid_boxes;
    bool in_bounds = true;
    for (size_t k = 0; k < 4; ++k) {
      const int index = config_.corner_order[k];
      d.corners[k] = cv::Point2f((row[6+2*index]-left)/scale, (row[7+2*index]-top)/scale);
      if (d.corners[k].x < 0 || d.corners[k].x >= bgr.cols ||
          d.corners[k].y < 0 || d.corners[k].y >= bgr.rows) in_bounds = false;
    }
    if (in_bounds) {
      ++diagnostics.in_bounds_corners;
      if (validCorners(d.corners)) {
        ++diagnostics.valid_corner_sets;
        detections.push_back(d);
      }
    }
  }
  std::sort(detections.begin(), detections.end(), [](const Detection & a, const Detection & b) { return a.confidence > b.confidence; });
  // Exported NMS is class-aware. Remove duplicate boxes with conflicting labels.
  std::vector<Detection> unique;
  for (const auto & d : detections) {
    bool duplicate = false;
    for (const auto & a : unique) {
      const double overlap = (d.box & a.box).area();
      if (overlap / (d.box.area() + a.box.area() - overlap) > 0.7) duplicate = true;
    }
    if (!duplicate) unique.push_back(d);
  }
  diagnostics.deduplicated_detections = static_cast<int>(unique.size());
  return unique;
}

Result Pipeline::process(const cv::Mat & bgr, double stamp) {
  if (bgr.empty() || bgr.type() != CV_8UC3 || bgr.cols != 1440 || bgr.rows != 1080)
    throw std::invalid_argument("Expected calibrated 1440x1080 BGR8 image");
  DetectionDiagnostics diagnostics;
  const auto detections = detect(bgr, diagnostics);
  auto result = processDetections(detections, stamp);
  result.model_rows = diagnostics.model_rows;
  result.confidence_class_pass = diagnostics.confidence_class_pass;
  result.valid_boxes = diagnostics.valid_boxes;
  result.in_bounds_corners = diagnostics.in_bounds_corners;
  result.valid_corner_sets = diagnostics.valid_corner_sets;
  result.deduplicated_detections = diagnostics.deduplicated_detections;
  return result;
}

Result Pipeline::processDetections(const std::vector<Detection> & detections, double stamp) {
  if (!std::isfinite(stamp)) throw std::invalid_argument("Nonfinite image timestamp");
  const bool time_reset = std::isfinite(last_stamp_) && (stamp <= last_stamp_ || stamp-last_stamp_ > config_.reset_timeout);
  if (time_reset || (std::isfinite(last_detection_stamp_) && stamp-last_detection_stamp_ > config_.reset_timeout)) {
    filter_.clear(); last_detection_stamp_ = nan; rejected_measurements_ = 0; last_class_id_ = -1;
    trusted_robot_center_.reset();
  }
  if (std::isfinite(last_stamp_) && !time_reset) filter_.predict(stamp-last_stamp_);
  last_stamp_ = stamp;
  Result result;
  result.frame_id = config_.yaw_frame;
  result.detections = detections;
  // Detection replay starts after detector validation/deduplication. -1 means
  // the corresponding inference stage was not executed, rather than a miss.
  result.model_rows = result.confidence_class_pass = result.valid_boxes = -1;
  result.in_bounds_corners = result.valid_corner_sets = -1;
  result.deduplicated_detections = int(detections.size());
  struct Candidate { Detection detection; std::vector<Pose> poses; size_t source_index; };
  std::vector<Candidate> candidates;
  for (size_t di = 0; di < result.detections.size(); ++di) {
    const auto & d = result.detections[di];
    ++result.pnp_attempts;
    if (!validCorners(d.corners)) continue;
    PoseDiagnostics pose_diagnostics;
    auto poses = solvePoses(d, config_, &pose_diagnostics);
    result.pnp_solutions += pose_diagnostics.solutions;
    result.physical_pose_rejects += pose_diagnostics.physical_pose_rejects;
    result.reprojection_rejects += pose_diagnostics.reprojection_rejects;
    result.valid_poses += pose_diagnostics.valid_poses;
    result.world_yaw_rejects += pose_diagnostics.world_yaw_rejects;
    if (!poses.empty()) candidates.push_back({d, std::move(poses), di});
  }
  ImageAssociation motion_association;
  if (config_.association_mode == "motion") {
    std::vector<ImageObservation> observations;
    for (const auto & c : candidates)
      observations.push_back({c.detection.center(), c.detection.box.size(), c.detection.class_id, c.detection.confidence});
    motion_association = motion_tracker_.update(observations, stamp);
    result.visual_track_id = motion_association.target_id;
    result.target_switched = motion_association.target_switched;
    result.predicted_center = motion_association.predicted_center;
    result.detection_track_ids.assign(result.detections.size(), 0);
    for (size_t i = 0; i < candidates.size(); ++i)
      result.detection_track_ids[candidates[i].source_index] = motion_association.detection_ids[i];
    if (motion_association.target_changed) {
      filter_.clear(); rejected_measurements_ = 0; last_class_id_ = -1;
      trusted_robot_center_.reset();
    }
    result.association_rejected = !candidates.empty() && motion_association.selected < 0;
    result.association_gate_px = motion_association.gate;
    if (!observations.empty()) {
      result.association_min_distance_px = std::numeric_limits<double>::infinity();
      for (const auto & o : observations)
        result.association_min_distance_px = std::min(result.association_min_distance_px,
          cv::norm(o.center-motion_association.predicted_center));
    }
  }
  if (!candidates.empty()) {
    size_t selected = config_.association_mode == "motion" ?
      (motion_association.selected < 0 ? candidates.size() : size_t(motion_association.selected)) : 0;
    bool new_target = !filter_.initialized();
    if (!new_target && config_.association_mode == "legacy") {
      double closest = std::numeric_limits<double>::infinity();
      size_t nearest = 0;
      for (size_t i = 0; i < candidates.size(); ++i) {
        const double distance = cv::norm(candidates[i].detection.center() - last_center_);
        if (distance < closest) { closest = distance; nearest = i; }
      }
      result.nearest_candidate_class_id = candidates[nearest].detection.class_id;
      result.nearest_candidate_confidence = candidates[nearest].detection.confidence;
      result.tracked_class_id = last_class_id_;
      result.nearest_candidate_matches_track = last_class_id_ >= 0 &&
        candidates[nearest].detection.class_id == last_class_id_;
      // Do not jump immediately to a neighboring plate when this plate flickers.
      const double association_gate = std::max(40.0, last_width_ * config_.association_gate_scale);
      result.association_min_distance_px = closest;
      result.association_gate_px = association_gate;
      if (closest > association_gate) {
        selected = candidates.size();
        result.association_rejected = true;
      } else {
        selected = nearest;
      }
    }
    if (selected < candidates.size()) {
    const auto & candidate = candidates[selected];
    result.selected_detection_index = int(candidate.source_index);
    auto pose = candidate.poses.front();
    RobotProjection selected_projection;
    if (config_.reprojection.enabled) {
      const auto choice = chooseReprojectionPose(candidate.poses,result.detections,candidate.source_index,
        config_,trusted_robot_center_,previous_pose_yaw_);
      pose = candidate.poses[choice.index];
      previous_pose_yaw_ = pose.yaw;
      selected_projection = choice.projection;
      if (choice.ambiguous) {
        result.projection_status = "AMBIGUOUS POSE";
        result.track_id = track_id_;
        result.filter_initialized = filter_.initialized();
        if (result.filter_initialized) {
          result.filtered_yaw=filter_.yaw(); result.yaw_rate=filter_.rate();
          result.yaw_variance=filter_.variance();
        }
        return result;
      }
      result.projection_status = choice.supported ? "MULTI-PLATE" : "SINGLE-PLATE";
      if (choice.supported) trusted_robot_center_ = selected_projection.center;
    } else if (!new_target) {
      double cost = std::numeric_limits<double>::infinity();
      for (const auto & p : candidate.poses) {
        // Prefer the reprojection optimum; use continuity only for close planar alternatives.
        if (p.error > candidate.poses.front().error + 0.5) continue;
        const double delta = std::abs(wrap(p.yaw-filter_.yaw()));
        // For a matched visual identity, resolve only the close IPPE
        // alternatives by temporal continuity; do not use yaw to change plates.
        const double score = config_.association_mode == "motion" ?
          delta + 0.02*p.error : p.error + 0.2*delta;
        if (score < cost) { cost = score; pose = p; }
      }
    }
    if (new_target) { filter_.reset(pose.yaw); ++track_id_; result.target_reset = true; result.measurement_used = true; rejected_measurements_ = 0; }
    else {
      result.measurement_used = filter_.update(pose.yaw);
      result.ekf_rejected = !result.measurement_used;
      if (result.measurement_used) rejected_measurements_ = 0;
      else if (++rejected_measurements_ >= 5) {
        // Persistent pose discontinuity can indicate a different armor entering
        // the same image region; give it a fresh track rather than a stale EKF.
        filter_.reset(pose.yaw); ++track_id_; result.target_reset = true;
        result.measurement_used = true; rejected_measurements_ = 0;
      }
    }
    const auto & d = candidate.detection;
    result.detection_valid = true; result.class_id = d.class_id; result.confidence = d.confidence;
    result.observed_yaw = pose.yaw; result.reprojection_error = pose.error;
    result.rvec = pose.rvec; result.tvec = pose.tvec; result.corners = d.corners;
    result.output_rvec = pose.output_rvec; result.output_tvec = pose.output_tvec; result.camera_yaw = pose.camera_yaw;
    if (config_.reprojection.enabled) result.projection = selected_projection;
    last_center_ = d.center(); last_width_ = d.box.width; last_detection_stamp_ = stamp;
    last_class_id_ = d.class_id;
    }
  }
  result.track_id = track_id_; result.filter_initialized = filter_.initialized();
  if (filter_.initialized()) {
    result.filtered_yaw = filter_.yaw(); result.yaw_rate = filter_.rate(); result.yaw_variance = filter_.variance();
  }
  return result;
}

cv::Mat Pipeline::annotate(const cv::Mat & bgr, const Result & r) const {
  cv::Mat image = bgr.clone();
  for (size_t di = 0; di < r.detections.size(); ++di) {
    const auto & d = r.detections[di];
    std::vector<cv::Point> points;
    for (const auto & p : d.corners) points.emplace_back(p);
    cv::polylines(image, points, true, cv::Scalar(128,128,128), 1);
    std::ostringstream label;
    if (di < r.detection_track_ids.size()) label << "V" << r.detection_track_ids[di] << " ";
    label << "c" << d.class_id << " " << std::fixed << std::setprecision(2) << d.confidence;
    cv::putText(image, label.str(), d.corners[0]+cv::Point2f(0,-8),
      cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(180,180,180), 1);
  }
  if (r.visual_track_id && !r.detection_valid)
    cv::drawMarker(image,r.predicted_center,cv::Scalar(0,165,255),cv::MARKER_CROSS,16,2);
  if (r.detection_valid) {
    std::vector<cv::Point> polygon;
    for (size_t k = 0; k < 4; ++k) {
      polygon.emplace_back(r.corners[k]);
      cv::circle(image, r.corners[k], 4, cv::Scalar(0,255,0), -1);
      cv::putText(image, std::to_string(k), r.corners[k] + cv::Point2f(4,-4), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255), 1);
    }
    cv::polylines(image, polygon, true, cv::Scalar(0,255,0), 2);
    const cv::Mat K(3,3,CV_64F,const_cast<double *>(config_.camera_matrix.data()));
    const cv::Mat D(1,5,CV_64F,const_cast<double *>(config_.distortion.data()));
    cv::drawFrameAxes(image, K, D, r.rvec, r.tvec, 0.07f, 2);
  }
  if (config_.reprojection.enabled) drawReprojection(image,r.projection);
  std::ostringstream text;
  text << std::fixed << std::setprecision(3) << "track " << r.track_id << " class " << r.class_id
    << "  " << r.frame_id << " observed " << r.observed_yaw << "  EKF " << r.filtered_yaw << " rad";
  if (r.visual_track_id) text << " V" << r.visual_track_id;
  cv::rectangle(image, cv::Rect(0,0,image.cols,config_.reprojection.enabled ? 105 : 78), cv::Scalar(0,0,0), -1);
  cv::putText(image, text.str(), cv::Point(15,30), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0,255,0), 2);
  text.str(""); text.clear();
  text << std::fixed << std::setprecision(3) << "yaw rate " << r.yaw_rate << " rad/s  reprojection " << r.reprojection_error
    << " px  " << (r.measurement_used ? "MEASUREMENT" : r.filter_initialized ? "PREDICTION ONLY" : "NO TARGET");
  cv::putText(image, text.str(), cv::Point(15,62), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 1);
  if (config_.reprojection.enabled) {
    const int count = std::count_if(r.projection.plates.begin(),r.projection.plates.end(),
      [](const auto & p) { return p.in_image; });
    text.str(""); text.clear();
    text << (r.projection.valid ? "4-armor model" :
             r.projection_status.empty() ? "NO CURRENT POSE" : r.projection_status) << "  in image " << count
         << "/4  radius " << config_.reprojection.radius << "m  P0 orange / P1-3 cyan / center red";
    cv::putText(image,text.str(),cv::Point(15,94),cv::FONT_HERSHEY_SIMPLEX,0.55,cv::Scalar(255,220,0),1);
  }
  return image;
}
} // namespace armor_ekf
