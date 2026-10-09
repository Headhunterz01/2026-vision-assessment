#pragma once
#include "armor_ekf/coordinate_transform.hpp"
#include <opencv2/core.hpp>
#include <array>
#include <vector>

namespace armor_ekf {
struct ReprojectionConfig {
  bool enabled = false;
  double radius = 0.25;
  // PDF frame X points toward the center; PnP armor X points outward.
  double pitch = 0.2617993877991494, roll = 0.0;
  double min_depth = 0.01;
};
struct ProjectedArmor {
  Eigen::Isometry3d world_pose = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d camera_pose = Eigen::Isometry3d::Identity();
  double pdf_yaw = 0;
  std::array<cv::Point2d,4> pixels{};
  bool depth_valid = false, in_image = false, front_facing = false;
};
struct RobotProjection {
  bool valid = false;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  cv::Point2d center_pixel;
  bool center_in_image = false;
  std::array<ProjectedArmor,4> plates;
};
RobotProjection reprojectArmors(const Eigen::Isometry3d & anchor,
    const Eigen::Isometry3d & world_from_camera, const ReprojectionConfig & parameters,
    double width, double height, const std::array<double,9> & intrinsics,
    const std::array<double,5> & distortion, cv::Size image_size = {1440,1080});
void drawReprojection(cv::Mat & image, const RobotProjection & projection);
} // namespace armor_ekf
