#include "armor_ekf/pipeline.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <limits>
void require(bool value,const char * reason) { if (!value) throw std::runtime_error(reason); }
int main(int argc,char **argv) {
  try {
    using namespace armor_ekf;
    Config c; c.yaw_frame="odom"; c.reprojection.enabled=true;
    const auto camera=TransformChain(c.frames).odomFromOptical();
    Eigen::Isometry3d anchor=rigidTransform({0,0,0},{0,-c.reprojection.pitch,2.8});
    anchor.translation()=camera*Eigen::Vector3d(0.04,0.08,3.5);
    auto p=reprojectArmors(anchor,camera,c.reprojection,c.armor_width,c.armor_height,c.camera_matrix,c.distortion);
    require(p.valid && p.plates[0].depth_valid,"Known scene invalid");
    require((p.plates[0].world_pose.matrix()-anchor.matrix()).norm()<1e-12,"Anchor must preserve full pose");
    const Eigen::Vector3d expected_center=anchor.translation()-0.25*Eigen::Vector3d(std::cos(2.8),std::sin(2.8),0);
    require((p.center-expected_center).norm()<1e-12,"Center direction incorrect");
    for (int i=0;i<4;++i) {
      const auto & a=p.plates[i];
      const Eigen::Vector3d radial=a.world_pose.translation()-p.center;
      require(std::abs(radial.norm()-0.25)<1e-12 && std::abs(radial.z())<1e-12,"Radius or height incorrect");
      require(a.world_pose.linear().col(0).head<2>().dot(radial.head<2>())>0,"Normal must point outward");
      require(std::abs(a.world_pose.linear()(2,0)-std::sin(c.reprojection.pitch))<1e-12,"Full frame bridge pitch incorrect");
      auto recovered=reprojectArmors(a.world_pose,camera,c.reprojection,c.armor_width,c.armor_height,c.camera_matrix,c.distortion);
      require((recovered.center-expected_center).norm()<1e-12,"Alternate anchor yields different center");
      const auto round_trip=camera*a.camera_pose;
      require((round_trip.matrix()-a.world_pose.matrix()).norm()<1e-12,"Camera/world transform incorrect");
      if (a.depth_valid) {
        // Independent per-corner world -> camera -> distorted projection.
        const double w=c.armor_width/2,h=c.armor_height/2;
        const std::array<Eigen::Vector3d,4> corners{{{0,-w,h},{0,-w,-h},{0,w,-h},{0,w,h}}};
        for (int k=0;k<4;++k) {
          const auto t=camera.inverse()*(a.world_pose*corners[k]);
          double x=t.x()/t.z(),y=t.y()/t.z(),r2=x*x+y*y;
          const auto & d=c.distortion;
          const double radial_factor=1+d[0]*r2+d[1]*r2*r2+d[4]*r2*r2*r2;
          const double xd=x*radial_factor+2*d[2]*x*y+d[3]*(r2+2*x*x);
          const double yd=y*radial_factor+d[2]*(r2+2*y*y)+2*d[3]*x*y;
          require(cv::norm(a.pixels[k]-cv::Point2d(c.camera_matrix[0]*xd+c.camera_matrix[2],c.camera_matrix[4]*yd+c.camera_matrix[5]))<1e-8,"Independent projection mismatch");
        }
      }
    }
    Detection d; d.class_id=1; d.confidence=0.9;
    std::vector<cv::Point2f> polygon;
    for (int k=0;k<4;++k) { d.corners[k]=p.plates[0].pixels[k]; polygon.push_back(d.corners[k]); }
    d.box=cv::boundingRect(polygon);
    require(argc==2,"Missing model path"); c.model_path=argv[1];
    Pipeline pipeline(c);
    auto r=pipeline.processDetections({d},0);
    require(r.detection_valid && r.projection.valid,"PnP integration failed");
    for (int k=0;k<4;++k) require(cv::norm(r.projection.plates[0].pixels[k]-cv::Point2d(d.corners[k]))<0.001,"Anchor projection disagrees with synthetic detection");
    r=pipeline.processDetections({},0.02);
    require(!r.projection.valid,"Missing detection retains stale model");
    anchor.translation()=camera*Eigen::Vector3d(0,0,-3);
    p=reprojectArmors(anchor,camera,c.reprojection,c.armor_width,c.armor_height,c.camera_matrix,c.distortion);
    for (const auto & a:p.plates) require(!a.depth_valid && !a.in_image,"Behind-camera plate visible");
    anchor.translation().x()=std::numeric_limits<double>::quiet_NaN();
    require(!reprojectArmors(anchor,camera,c.reprojection,c.armor_width,c.armor_height,c.camera_matrix,c.distortion).valid,"Nonfinite anchor accepted");
    ReprojectionConfig bad=c.reprojection; bad.radius=0;
    bool rejected=false; try { reprojectArmors(Eigen::Isometry3d::Identity(),camera,bad,c.armor_width,c.armor_height,c.camera_matrix,c.distortion); } catch (const std::invalid_argument &) { rejected=true; }
    require(rejected,"Invalid radius accepted");
    // Recorded two-plate frame 978: temporal Yaw previously kept the mirror
    // IPPE pose, placing the robot centre 247 pixels to the wrong side.
    Config diagnostic=c; diagnostic.frames.odom_gimbal_rpy[1]=0;
    Detection left, right;
    right.class_id=left.class_id=2; right.confidence=.8377; left.confidence=.7666;
    right.corners={cv::Point2f(802.39990f,606.10162f),{811.46484f,650.86755f},
                   {891.95685f,635.80115f},{884.47504f,593.86316f}};
    left.corners={cv::Point2f(553.52179f,604.45142f),{548.88013f,642.62396f},
                  {617.40240f,651.78387f},{621.86859f,612.00391f}};
    right.box=cv::Rect2f(798.90533f,576.03015f,98.82875f,95.19962f);
    left.box=cv::Rect2f(545.67438f,584.28296f,80.60878f,88.53992f);
    const auto alternatives=solvePoses(right,diagnostic);
    require(alternatives.size()==2,"Recorded frame must have two valid IPPE solutions");
    const auto choice=chooseReprojectionPose(alternatives,{right,left},0,diagnostic,std::nullopt);
    require(choice.supported && !choice.ambiguous && choice.projection.valid,
            "Second plate must resolve pose ambiguity");
    require(std::abs(wrap(alternatives[choice.index].yaw-(-2.47962624)))<.001,
            "Multi-plate evidence selected the mirror pose");
    left.class_id=1;
    const auto cross_class=chooseReprojectionPose(alternatives,{right,left},0,diagnostic,std::nullopt);
    require(cross_class.supported && cross_class.index==choice.index,
            "Armor detections with different class labels may still belong to one robot");
    const auto alone=chooseReprojectionPose(alternatives,{right},0,diagnostic,std::nullopt);
    require(!alone.ambiguous && alone.index==choice.index,
            "Known armor pitch must resolve a single plate when candidates separate");
    Config uncertain=diagnostic; uncertain.reprojection.pitch=-0.7;
    const auto unresolved=chooseReprojectionPose(alternatives,{right},0,uncertain,std::nullopt);
    require(unresolved.ambiguous,"Unsupported single plate must suppress projection");
    const auto continued=chooseReprojectionPose(alternatives,{right},0,diagnostic,choice.projection.center);
    require(!continued.ambiguous && continued.index==choice.index,
            "Trusted centre must preserve the resolved pose during one-plate frames");
    const auto pdf_candidates=solvePoses(right,c);
    const auto conflicting=chooseReprojectionPose(pdf_candidates,{right,left},0,c,std::nullopt);
    require(conflicting.ambiguous,"Contradictory second plate must suppress a ghost projection");
    RobotProjection clipped; clipped.valid=true; clipped.plates[0].depth_valid=true;
    clipped.plates[0].in_image=true; clipped.plates[0].front_facing=true;
    clipped.plates[0].pixels={cv::Point2d(-20,5),{20,5},{20,25},{-20,25}};
    cv::Mat small(30,30,CV_8UC3,cv::Scalar(0)); drawReprojection(small,clipped);
    require(cv::countNonZero(small.reshape(1))>0,"Clipped outline not drawn");
    std::cout << "Reprojection geometry, full frame bridge, independent distortion, PnP, missing pose and clipping passed\n";
    return 0;
  } catch (const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
}
