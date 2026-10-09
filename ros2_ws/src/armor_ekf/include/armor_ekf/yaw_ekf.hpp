#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace armor_ekf {
inline double wrap(double value) { return std::atan2(std::sin(value), std::cos(value)); }

// State [yaw, yaw_rate]. Nonlinear observation h(x) = [cos(yaw), sin(yaw)].
// Expressing the measurement on the unit circle avoids a discontinuity at +/-pi.
class YawEkf {
public:
  YawEkf(double measurement_std = 0.08, double acceleration_std = 8.0)
  : r_(measurement_std * measurement_std), q_(acceleration_std * acceleration_std) {
    if (!(measurement_std > 0 && acceleration_std > 0))
      throw std::invalid_argument("EKF noise standard deviations must be positive");
  }
  void reset(double yaw) {
    x_ << wrap(yaw), 0.0;
    p_ = Eigen::Matrix2d::Zero();
    p_(0, 0) = r_;
    p_(1, 1) = 4.0;
    initialized_ = true;
  }
  void clear() { initialized_ = false; }
  void predict(double dt) {
    if (!initialized_) return;
    if (!std::isfinite(dt) || dt < 0) throw std::invalid_argument("Invalid EKF dt");
    Eigen::Matrix2d f; f << 1.0, dt, 0.0, 1.0;
    Eigen::Vector2d g(dt * dt / 2.0, dt);
    x_ = f * x_; x_(0) = wrap(x_(0));
    p_ = f * p_ * f.transpose() + q_ * g * g.transpose();
  }
  bool update(double yaw) {
    if (!std::isfinite(yaw)) return false;
    if (!initialized_) { reset(yaw); return true; }
    const double angle_innovation = wrap(yaw - x_(0));
    if (std::abs(angle_innovation) > std::max(0.4, 4.0 * std::sqrt(p_(0, 0) + r_)))
      return false;
    const Eigen::Vector2d z(std::cos(yaw), std::sin(yaw));
    const Eigen::Vector2d h(std::cos(x_(0)), std::sin(x_(0)));
    Eigen::Matrix2d H; H << -h(1), 0.0, h(0), 0.0;
    const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * r_;
    const Eigen::Matrix2d S = H * p_ * H.transpose() + R;
    const Eigen::Matrix2d K = p_ * H.transpose() * S.ldlt().solve(Eigen::Matrix2d::Identity());
    x_ += K * (z - h); x_(0) = wrap(x_(0));
    const Eigen::Matrix2d A = Eigen::Matrix2d::Identity() - K * H;
    p_ = A * p_ * A.transpose() + K * R * K.transpose(); // Joseph form.
    p_ = 0.5 * (p_ + p_.transpose()).eval();
    return true;
  }
  bool initialized() const { return initialized_; }
  double yaw() const { return x_(0); }
  double rate() const { return x_(1); }
  double variance() const { return p_(0, 0); }
  const Eigen::Matrix2d & covariance() const { return p_; }
private:
  Eigen::Vector2d x_ = Eigen::Vector2d::Zero();
  Eigen::Matrix2d p_ = Eigen::Matrix2d::Identity();
  double r_, q_;
  bool initialized_ = false;
};
} // namespace armor_ekf
