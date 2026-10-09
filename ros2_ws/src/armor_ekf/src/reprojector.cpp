#include "armor_ekf/reprojector.hpp"
#include <opencv2/imgproc.hpp>
#include <limits>
#include <algorithm>

namespace armor_ekf {
namespace {
constexpr double pi = 3.14159265358979323846;
bool safePixel(const cv::Point2d & p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::abs(p.x) < 1e7 && std::abs(p.y) < 1e7;
}
bool clippedLine(cv::Size size, cv::Point2d a, cv::Point2d b, cv::Point & out_a, cv::Point & out_b) {
  if (!safePixel(a) || !safePixel(b)) return false;
  cv::Point2l first(std::llround(a.x),std::llround(a.y)), second(std::llround(b.x),std::llround(b.y));
  if (!cv::clipLine(cv::Size2l(size.width,size.height),first,second)) return false;
  out_a = cv::Point(first); out_b = cv::Point(second); return true;
}
}

RobotProjection reprojectArmors(const Eigen::Isometry3d & anchor,
    const Eigen::Isometry3d & world_from_camera, const ReprojectionConfig & cfg,
    double width, double height, const std::array<double,9> & intrinsics,
    const std::array<double,5> & distortion, cv::Size size) {
  if (!(std::isfinite(cfg.radius) && cfg.radius > 0 && std::isfinite(cfg.pitch) &&
        std::isfinite(cfg.roll) && std::isfinite(cfg.min_depth) && cfg.min_depth > 0 &&
        std::isfinite(width) && std::isfinite(height) && width > 0 && height > 0 && size.width > 0 && size.height > 0))
    throw std::invalid_argument("Invalid reprojection geometry parameters");
  RobotProjection result;
  if (!anchor.matrix().allFinite() || !world_from_camera.matrix().allFinite()) return result;
  const auto heading = worldYaw(anchor.linear(),0.001);
  if (!heading) return result;
  // PDF position: p_i = center - radius * [cos(pdf_yaw_i),sin(pdf_yaw_i),0].
  // Its X points inward. Right-multiply Rz(pi) to express the same plate in
  // the outward-X PnP frame; this converts yaw AND pitch/roll, not just yaw.
  const double yaw0 = std::atan2(std::sin(*heading + pi),std::cos(*heading + pi));
  result.center = anchor.translation() + cfg.radius * Eigen::Vector3d(std::cos(yaw0),std::sin(yaw0),0);
  const Eigen::Isometry3d camera_from_world = world_from_camera.inverse();
  const Eigen::Matrix3d outward_from_inward = Eigen::AngleAxisd(pi,Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const double w = width / 2, h = height / 2;
  const std::vector<cv::Point3d> object{{0,-w,h},{0,-w,-h},{0,w,-h},{0,w,h}};
  const cv::Mat K(3,3,CV_64F,const_cast<double *>(intrinsics.data()));
  const cv::Mat D(1,5,CV_64F,const_cast<double *>(distortion.data()));
  const std::vector<cv::Point2f> screen{{0,0},{float(size.width-1),0},
    {float(size.width-1),float(size.height-1)},{0,float(size.height-1)}};
  for (size_t i = 0; i < 4; ++i) {
    auto & plate = result.plates[i];
    const double yaw = yaw0 + pi * i / 2;
    plate.pdf_yaw = std::atan2(std::sin(yaw),std::cos(yaw));
    plate.world_pose.translation() = result.center - cfg.radius * Eigen::Vector3d(std::cos(yaw),std::sin(yaw),0);
    plate.world_pose.linear() = rigidTransform({0,0,0},{cfg.roll,cfg.pitch,yaw}).linear() * outward_from_inward;
    if (i == 0) plate.world_pose = anchor;
    plate.camera_pose = camera_from_world * plate.world_pose;
    plate.front_facing = plate.camera_pose.linear().col(0).dot(plate.camera_pose.translation()) < 0;
    plate.depth_valid = true;
    for (const auto & p : object)
      if ((plate.camera_pose * Eigen::Vector3d(p.x,p.y,p.z)).z() <= cfg.min_depth) plate.depth_valid = false;
    const double missing = std::numeric_limits<double>::quiet_NaN();
    plate.pixels.fill(cv::Point2d(missing,missing));
    if (!plate.depth_valid) continue;
    const auto & t = plate.camera_pose.translation();
    std::vector<cv::Point2d> pixels;
    cv::projectPoints(object,rvecFromRotation(plate.camera_pose.linear()),cv::Vec3d(t.x(),t.y(),t.z()),K,D,pixels);
    std::vector<cv::Point2f> polygon;
    bool finite = true;
    for (size_t k = 0; k < 4; ++k) {
      plate.pixels[k] = pixels[k];
      finite = finite && safePixel(pixels[k]);
      if (safePixel(pixels[k])) polygon.emplace_back(pixels[k]);
    }
    if (finite && cv::isContourConvex(polygon)) {
      std::vector<cv::Point2f> intersection;
      plate.in_image = cv::intersectConvexConvex(polygon,screen,intersection,true) > 0;
    }
  }
  const Eigen::Vector3d center_camera = camera_from_world * result.center;
  if (center_camera.z() > cfg.min_depth) {
    std::vector<cv::Point2d> pixels;
    cv::projectPoints(std::vector<cv::Point3d>{{center_camera.x(),center_camera.y(),center_camera.z()}},
                     cv::Vec3d(0,0,0),cv::Vec3d(0,0,0),K,D,pixels);
    result.center_pixel = pixels[0];
    result.center_in_image = safePixel(pixels[0]) && pixels[0].x >= 0 && pixels[0].x < size.width && pixels[0].y >= 0 && pixels[0].y < size.height;
  }
  result.valid = true;
  return result;
}

void drawReprojection(cv::Mat & image, const RobotProjection & projection) {
  if (!projection.valid) return;
  for (size_t i = 0; i < 4; ++i) {
    const auto & p = projection.plates[i];
    if (!p.depth_valid || !p.in_image) continue;
    const cv::Scalar color = i == 0 ? cv::Scalar(0,180,255) : cv::Scalar(255,220,0);
    cv::Point label;
    bool have_label = false;
    for (size_t k = 0; k < 4; ++k) {
      cv::Point a,b;
      if (clippedLine(image.size(),p.pixels[k],p.pixels[(k+1)%4],a,b)) {
        cv::line(image,a,b,color,2,cv::LINE_AA);
        if (!have_label) { label = a; have_label = true; }
      }
    }
    if (i != 0) for (size_t k = 0; k < 2; ++k) {
      cv::Point a,b;
      if (clippedLine(image.size(),p.pixels[k],p.pixels[k+2],a,b)) cv::line(image,a,b,color,1,cv::LINE_AA);
    }
    if (have_label) {
      label.x = std::clamp(label.x,0,std::max(0,image.cols-150));
      label.y = std::clamp(label.y-6,std::min(105,std::max(0,image.rows-5)),std::max(0,image.rows-5));
      cv::putText(image,"P"+std::to_string(i)+(i==0 ? " anchor" : p.front_facing ? " model" : " model back"),label,
                  cv::FONT_HERSHEY_SIMPLEX,0.5,color,1,cv::LINE_AA);
    }
  }
  if (projection.center_in_image)
    cv::drawMarker(image,projection.center_pixel,cv::Scalar(0,0,255),cv::MARKER_CROSS,15,2);
}
} // namespace armor_ekf
