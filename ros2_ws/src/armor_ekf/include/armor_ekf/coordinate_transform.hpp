#pragma once
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace armor_ekf {
// T_parent_child maps child coordinates to parent: R * p_child + t.
struct FrameConfig {
  std::array<double, 3> odom_gimbal_xyz{0, 0, 0};
  std::array<double, 3> odom_gimbal_rpy{0, 1.34, 0};
  std::array<double, 3> gimbal_camera_xyz{0.0461892, 0, -0.0603874};
  std::array<double, 3> gimbal_camera_rpy{0, 0.091, 0};
  std::array<double, 3> camera_optical_xyz{0, 0, 0};
  std::array<double, 3> camera_optical_rpy{-1.5707963267948966, 0, -1.5707963267948966};
};
inline Eigen::Isometry3d rigidTransform(const std::array<double, 3> & xyz,
                                       const std::array<double, 3> & rpy) {
  for (double v : xyz) if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite frame translation");
  for (double v : rpy) if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite frame rotation");
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.linear() = (Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX())).toRotationMatrix();
  t.translation() = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
  return t;
}
class TransformChain {
public:
  explicit TransformChain(const FrameConfig & c) : edges_{
      rigidTransform(c.odom_gimbal_xyz, c.odom_gimbal_rpy),
      rigidTransform(c.gimbal_camera_xyz, c.gimbal_camera_rpy),
      rigidTransform(c.camera_optical_xyz, c.camera_optical_rpy)},
      odom_optical_(edges_[0] * edges_[1] * edges_[2]) {}
  const std::array<Eigen::Isometry3d, 3> & edges() const { return edges_; }
  const Eigen::Isometry3d & odomFromOptical() const { return odom_optical_; }
private:
  std::array<Eigen::Isometry3d, 3> edges_;
  Eigen::Isometry3d odom_optical_;
};
inline Eigen::Matrix3d rotationFromRvec(const cv::Vec3d & rvec) {
  cv::Matx33d matrix;
  cv::Rodrigues(rvec, matrix);
  Eigen::Matrix3d r;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) r(i, j) = matrix(i, j);
  return r;
}
inline cv::Vec3d rvecFromRotation(const Eigen::Matrix3d & r) {
  cv::Matx33d matrix;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) matrix(i, j) = r(i, j);
  cv::Vec3d rvec;
  cv::Rodrigues(matrix, rvec);
  return rvec;
}
inline std::optional<double> worldYaw(const Eigen::Matrix3d & r, double min_horizontal) {
  if (!r.allFinite() || std::hypot(r(0, 0), r(1, 0)) < min_horizontal) return std::nullopt;
  return std::atan2(r(1, 0), r(0, 0));
}
} // namespace armor_ekf
