#pragma once
#include "armor_ekf/yaw_ekf.hpp"
#include "armor_ekf/motion_tracker.hpp"
#include "armor_ekf/coordinate_transform.hpp"
#include "armor_ekf/reprojector.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace armor_ekf {
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
struct Config {
  std::string yaw_frame = "camera_optical_frame";
  FrameConfig frames;
  ReprojectionConfig reprojection;
  double min_world_horizontal_normal = 0.001;
  std::string model_path;
  double confidence = 0.5;
  double armor_width = 0.135, armor_height = 0.055;
  double max_reprojection_error = 3.0;
  double measurement_std = 0.08, acceleration_std = 8.0;
  double reset_timeout = 0.3;
  double association_gate_scale = 0.8;
  std::string association_mode = "motion";
  double target_switch_timeout = 0.08;
  int threads = 2;
  std::array<int, 4> corner_order{0, 1, 2, 3};
  std::array<double, 9> camera_matrix{3571.1890927935297, 0, 683.5999418028833,
    0, 3569.0234714126823, 565.5838917436471, 0, 0, 1};
  std::array<double, 5> distortion{-0.029757126887136703, 0.371503828585697,
    0.0011898367108863406, -0.0025516396408119418, 0.0};
};
struct Detection {
  std::array<cv::Point2f, 4> corners;
  cv::Rect2f box;
  int class_id;
  double confidence;
  cv::Point2f center() const { return (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25f; }
};
struct Pose {
  cv::Vec3d rvec, tvec;
  double yaw, error;
  cv::Vec3d output_rvec, output_tvec;
  double camera_yaw;
};
struct Result {
  RobotProjection projection;
  std::string projection_status;
  std::string frame_id = "camera_optical_frame";
  bool detection_valid = false, measurement_used = false, filter_initialized = false;
  bool target_reset = false;
  uint32_t track_id = 0;
  int class_id = -1;
  double confidence = 0, observed_yaw = nan, filtered_yaw = nan, yaw_rate = nan;
  double yaw_variance = nan, reprojection_error = nan;
  cv::Vec3d rvec{0, 0, 0}, tvec{0, 0, 0};
  cv::Vec3d output_rvec{0, 0, 0}, output_tvec{0, 0, 0};
  double camera_yaw = nan;
  std::array<cv::Point2f, 4> corners{};
  std::vector<Detection> detections;
  // Per-frame counters expose where detections are discarded in the pipeline.
  int model_rows = 0, confidence_class_pass = 0, valid_boxes = 0;
  int in_bounds_corners = 0, valid_corner_sets = 0, deduplicated_detections = 0;
  int pnp_attempts = 0, pnp_solutions = 0, physical_pose_rejects = 0;
  int reprojection_rejects = 0, valid_poses = 0;
  int world_yaw_rejects = 0;
  bool association_rejected = false, ekf_rejected = false;
  double association_min_distance_px = nan, association_gate_px = nan;
  int nearest_candidate_class_id = -1, tracked_class_id = -1;
  double nearest_candidate_confidence = nan;
  bool nearest_candidate_matches_track = false;
  uint32_t visual_track_id = 0;
  bool target_switched = false;
  cv::Point2f predicted_center{};
  std::vector<uint32_t> detection_track_ids;
  int selected_detection_index = -1;
};
struct DetectionDiagnostics {
  int model_rows = 0, confidence_class_pass = 0, valid_boxes = 0;
  int in_bounds_corners = 0, valid_corner_sets = 0, deduplicated_detections = 0;
};
struct PoseDiagnostics {
  int solutions = 0, physical_pose_rejects = 0, reprojection_rejects = 0, valid_poses = 0;
  int world_yaw_rejects = 0;
};
bool validCorners(const std::array<cv::Point2f, 4> & corners);
std::vector<Pose> solvePoses(const Detection & detection, const Config & config,
                            PoseDiagnostics * diagnostics = nullptr);
struct ReprojectionPoseChoice {
  size_t index = 0;
  RobotProjection projection;
  bool supported = false;
  bool ambiguous = false;
  double supporting_iou = 0;
};
ReprojectionPoseChoice chooseReprojectionPose(const std::vector<Pose> & poses,
    const std::vector<Detection> & detections, size_t selected_index,
    const Config & config, const std::optional<Eigen::Vector3d> & trusted_center,
    const std::optional<double> & trusted_yaw = std::nullopt);
class Pipeline {
public:
  explicit Pipeline(const Config & config);
  Result process(const cv::Mat & bgr, double stamp);
  Result processDetections(const std::vector<Detection> & detections, double stamp);
  cv::Mat annotate(const cv::Mat & bgr, const Result & result) const;
private:
  std::vector<Detection> detect(const cv::Mat & bgr, DetectionDiagnostics & diagnostics);
  Config config_;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "armor_ekf"};
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_, output_name_;
  int input_width_ = 640, input_height_ = 640;
  YawEkf filter_;
  MotionTracker motion_tracker_;
  double last_stamp_ = nan, last_detection_stamp_ = nan;
  cv::Point2f last_center_{};
  double last_width_ = 0;
  int last_class_id_ = -1;
  uint32_t track_id_ = 0;
  int rejected_measurements_ = 0;
  std::optional<Eigen::Vector3d> trusted_robot_center_;
  std::optional<double> previous_pose_yaw_;
};
} // namespace armor_ekf
