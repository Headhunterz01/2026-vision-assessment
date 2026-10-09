#include "armor_ekf/pipeline.hpp"
#include <tf2/LinearMath/Transform.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

void require(bool ok, const char * text) { if (!ok) throw std::runtime_error(text); }

armor_ekf::Detection synthetic(const armor_ekf::Config & c, double yaw, double pitch,
                              const cv::Vec3d & camera_position = cv::Vec3d(0.04,0.08,3.5)) {
  using namespace armor_ekf;
  const auto world = TransformChain(c.frames).odomFromOptical();
  const Eigen::Matrix3d armor_world = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                                      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())).toRotationMatrix();
  const cv::Vec3d r = rvecFromRotation(world.linear().transpose() * armor_world);
  const double w = c.armor_width / 2, h = c.armor_height / 2;
  const std::vector<cv::Point3d> object{{0,-w,h}, {0,-w,-h}, {0,w,-h}, {0,w,h}};
  const cv::Mat K(3,3,CV_64F,const_cast<double *>(c.camera_matrix.data()));
  const cv::Mat D(1,5,CV_64F,const_cast<double *>(c.distortion.data()));
  std::vector<cv::Point2d> projected;
  cv::projectPoints(object, r, camera_position, K, D, projected);
  Detection d; d.class_id = 1; d.confidence = 0.9;
  std::vector<cv::Point2f> polygon;
  for (int i = 0; i < 4; ++i) { d.corners[i] = cv::Point2f(projected[i]); polygon.push_back(d.corners[i]); }
  d.box = cv::boundingRect(polygon);
  return d;
}

int main(int argc, char ** argv) {
  try {
    using namespace armor_ekf;
    Config cfg; cfg.yaw_frame = "odom";
    const auto chain = TransformChain(cfg.frames);
    tf2::Transform tf = tf2::Transform::getIdentity();
    const std::array<std::array<double,3>,3> rpys{cfg.frames.odom_gimbal_rpy, cfg.frames.gimbal_camera_rpy, cfg.frames.camera_optical_rpy};
    const std::array<std::array<double,3>,3> xyzs{cfg.frames.odom_gimbal_xyz, cfg.frames.gimbal_camera_xyz, cfg.frames.camera_optical_xyz};
    for (int i = 0; i < 3; ++i) {
      tf2::Quaternion q; q.setRPY(rpys[i][0], rpys[i][1], rpys[i][2]);
      tf = tf * tf2::Transform(q, tf2::Vector3(xyzs[i][0],xyzs[i][1],xyzs[i][2]));
    }
    const auto & world = chain.odomFromOptical();
    for (int i = 0; i < 3; ++i) {
      require(std::abs(tf.getOrigin()[i]-world.translation()[i]) < 1e-12, "TF2 translation mismatch");
      for (int j = 0; j < 3; ++j)
        require(std::abs(tf.getBasis()[i][j]-world.linear()(i,j)) < 1e-12, "TF2 rotation mismatch");
    }
    Eigen::Matrix3d optical;
    optical << 0,0,1, -1,0,0, 0,-1,0;
    require((chain.edges()[2].linear()-optical).norm() < 1e-12, "Optical axes incorrect");
    require(std::abs(world.translation().x()+0.0482202912) < 1e-9, "Translation order incorrect");
    const Eigen::Vector3d p(0.04,0.08,3.5);
    require((world.inverse()*(world*p)-p).norm() < 1e-12, "Round trip failed");
    require(!worldYaw(Eigen::AngleAxisd(-1.5707963267948966,Eigen::Vector3d::UnitY()).toRotationMatrix(),0.001), "Vertical normal must have no yaw");
    PoseDiagnostics vertical_diagnostics;
    // Align the camera with the vertical normal so both planar solutions are
    // degenerate. An oblique view can instead have a nonvertical mirror solution.
    Config vertical_cfg = cfg;
    vertical_cfg.frames.odom_gimbal_rpy[1] = 1.5707963267948966 - cfg.frames.gimbal_camera_rpy[1];
    // Float32 image corners perturb an exactly frontal plane by about 1e-3.
    // Exercise the configurable near-vertical rejection with a 0.01 threshold.
    vertical_cfg.min_world_horizontal_normal = 0.01;
    const auto vertical_poses = solvePoses(synthetic(vertical_cfg,0,-1.5707963267948966,cv::Vec3d(0,0,3.5)),vertical_cfg,&vertical_diagnostics);
    require(vertical_poses.empty() && vertical_diagnostics.world_yaw_rejects > 0,
            "PnP must reject an undefined world yaw before filtering");
    double max_yaw_error = 0, max_position_error = 0;
    // Task extrinsics, plus a second camera orientation with positive/negative world yaw.
    for (int view = 0; view < 2; ++view) {
      Config c = cfg;
      if (view) c.frames.odom_gimbal_rpy = {0, 0.2, 3.141592653589793};
      const std::vector<double> headings = view ? std::vector<double>{-0.4,0,0.4} :
          std::vector<double>{2.7,3.14,-3.14,-2.7};
      for (double yaw : headings) {
        const double pitch = view ? 0.2617993877991494 : -0.2617993877991494;
        const auto poses = solvePoses(synthetic(c,yaw,pitch),c);
        require(!poses.empty(), "Synthetic world PnP failed");
        const auto & pose = poses.front();
        const double error = std::abs(wrap(pose.yaw-yaw));
        const Eigen::Vector3d expected = TransformChain(c.frames).odomFromOptical()*p;
        const double position_error = (Eigen::Vector3d(pose.output_tvec[0],pose.output_tvec[1],pose.output_tvec[2])-expected).norm();
        max_yaw_error = std::max(max_yaw_error,error); max_position_error = std::max(max_position_error,position_error);
        require(error < 0.001, "Known world yaw not recovered");
        require(position_error < 0.001, "Known world position not recovered");
      }
    }
    require(argc == 2, "Model path required for pipeline integration test");
    cfg.model_path = argv[1];
    Pipeline pipeline(cfg);
    for (int i = 0; i < 100; ++i) {
      const double yaw = wrap(3.10+0.001*i);
      const auto result = pipeline.processDetections({synthetic(cfg,yaw,-0.2617993877991494)}, i*0.02);
      require(result.frame_id == "odom" && result.detection_valid && result.measurement_used, "World pipeline observation invalid");
      require(std::abs(wrap(result.observed_yaw-yaw)) < 0.001, "Pipeline filtered camera heading instead of world yaw");
      require(std::abs(wrap(result.filtered_yaw-yaw)) < 0.01, "World EKF failed across pi wrap");
      require(i == 0 || !result.target_reset, "Stable synthetic identity unexpectedly reset");
    }
    auto lost = pipeline.processDetections({},2.0);
    require(!lost.detection_valid && std::isnan(lost.observed_yaw) && lost.filter_initialized, "World dropout prediction incorrect");
    lost = pipeline.processDetections({},2.4);
    require(!lost.filter_initialized && std::isnan(lost.filtered_yaw), "World dropout expiry incorrect");
    auto rewind = pipeline.processDetections({synthetic(cfg,3.1,-0.2617993877991494)},0.0);
    require(rewind.target_reset && rewind.measurement_used, "World timestamp rewind did not reinitialize");
    std::cout << "PASS: TF2/Eigen chain, optical axes, translation, vertical-normal rejection, synthetic PnP world pose, world EKF wrap/dropout/rewind\n"
              << "max_synthetic_yaw_error_rad=" << max_yaw_error << " max_position_error_m=" << max_position_error << '\n';
    return 0;
  } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
