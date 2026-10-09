#include "armor_ekf/pipeline.hpp"
#include <opencv2/calib3d.hpp>
#include <Eigen/Eigenvalues>
#include <random>
#include <iostream>

void require(bool condition, const char * message) {
  if (!condition) throw std::runtime_error(message);
}
int main() {
  try {
    using namespace armor_ekf;
    YawEkf filter(0.08, 1.0);
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.08);
    double raw_sq = 0, filtered_sq = 0;
    for (int i = 0; i < 1000; ++i) {
      const double truth = wrap(3.10 + 0.5 * i * 0.01);
      const double measured = wrap(truth + noise(rng));
      filter.predict(0.01);
      require(filter.update(measured), "Normal measurement rejected");
      require(std::isfinite(filter.yaw()) && filter.variance() >= 0, "Nonfinite filter state");
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(filter.covariance());
      require(solver.eigenvalues().minCoeff() >= -1e-10, "Covariance is not positive semidefinite");
      if (i > 100) { raw_sq += std::pow(wrap(measured-truth),2); filtered_sq += std::pow(wrap(filter.yaw()-truth),2); }
    }
    require(filtered_sq < raw_sq * 0.4, "EKF does not improve noisy yaw");
    require(std::abs(filter.rate()-0.5) < 0.2, "Angular velocity failed to converge");
    const auto old = filter.yaw();
    require(!filter.update(wrap(old+2.0)), "Outlier was not rejected");
    require(std::abs(wrap(filter.yaw()-old)) < 1e-10, "Rejected outlier changed state");
    filter.predict(0.1);
    require(std::isfinite(filter.yaw()), "Prediction without detection failed");
    filter.clear(); require(!filter.initialized(), "Clear failed");
    require(!filter.update(std::numeric_limits<double>::quiet_NaN()), "NaN accepted");
    filter.reset(0.5); require(filter.update(0.51), "Reset failed");

    Config cfg;
    const double theta = 0.45, w = cfg.armor_width/2, h = cfg.armor_height/2;
    const cv::Matx33d base(0,1,0, 0,0,-1, -1,0,0);
    const cv::Matx33d rotation_y(std::cos(theta),0,-std::sin(theta), 0,1,0,
                                std::sin(theta),0,std::cos(theta));
    const auto R = rotation_y * base;
    cv::Vec3d r; cv::Rodrigues(R,r);
    const cv::Vec3d t(0.04,0.08,3.5);
    const std::vector<cv::Point3d> object{{0,-w,h},{0,-w,-h},{0,w,-h},{0,w,h}};
    const cv::Mat K(3,3,CV_64F,cfg.camera_matrix.data()), D(1,5,CV_64F,cfg.distortion.data());
    std::vector<cv::Point2d> projected;
    cv::projectPoints(object,r,t,K,D,projected);
    Detection d;
    for (size_t i = 0; i < 4; ++i) d.corners[i] = cv::Point2f(projected[i]);
    require(validCorners(d.corners), "Valid polygon rejected");
    const auto poses = solvePoses(d,cfg);
    require(!poses.empty(), "PnP failed on synthetic calibrated corners");
    require(cv::norm(poses[0].tvec-t) < 0.001, "PnP translation incorrect");
    require(std::abs(wrap(poses[0].yaw-theta)) < 0.001, "PnP yaw sign or axes incorrect");
    require(poses[0].error < 0.001, "PnP reprojection error too large");
    std::swap(d.corners[1],d.corners[2]);
    require(!validCorners(d.corners), "Self-intersecting corners accepted");
    std::cout << "PASS: EKF noise reduction, angle wrap, covariance, outlier gate, dropout/reset, and synthetic PnP pose\n";
    return 0;
  } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
